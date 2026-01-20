import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
from typing import Optional
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from ml_core.data.pcam import PCAMDataset
from ml_core.models.mlp import MLP


def load_cfg(cfg_path: str) -> dict:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def build_model(cfg: dict) -> torch.nn.Module:
    # Baseline model uit jullie config: MLP
    input_shape = cfg["data"]["input_shape"]  # [3, 96, 96]
    in_features = int(np.prod(input_shape))
    hidden_units = cfg["model"]["hidden_units"]
    dropout = cfg["model"]["dropout_rate"]
    num_classes = cfg["model"]["num_classes"]
    return MLP(in_features=in_features, hidden_units=hidden_units, dropout_rate=dropout, num_classes=num_classes)


def save_grid(images_chw: np.ndarray, title: str, out_path: Path, n: int = 5) -> None:
    """
    images_chw: (N, C, H, W), values 0..1
    """
    n = min(n, images_chw.shape[0])
    fig = plt.figure(figsize=(12, 3))
    for i in range(n):
        ax = fig.add_subplot(1, n, i + 1)
        img = np.transpose(images_chw[i], (1, 2, 0))  # CHW -> HWC
        ax.imshow(img)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: str, max_batches: Optional[int] = None):
    model.eval()
    xs, ys, ps = [], [], []
    for b_idx, (x, y) in enumerate(loader):
        if max_batches is not None and b_idx >= max_batches:
            break
        x = x.to(device)
        logits = model(x)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        xs.append(x.cpu().numpy())
        ys.append(y.numpy())
        ps.append(pred)
    X = np.concatenate(xs, axis=0)
    Y = np.concatenate(ys, axis=0)
    P = np.concatenate(ps, axis=0)
    return X, Y, P


def main():
    cfg_path = os.environ.get("CFG", "experiments/configs/train_config.yaml")
    ckpt_path = os.environ.get("CKPT", "")  # optional
    outdir = Path(os.environ.get("OUTDIR", "q6_outputs"))
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = load_cfg(cfg_path)

    # --- Use VALID set as "test" for Q6 (because test_y.h5 is not available) ---
    data_dir = Path(cfg["data"]["data_path"]).resolve()

    valid_x = data_dir / "camelyonpatch_level_2_split_valid_x.h5"
    valid_y = data_dir / "camelyonpatch_level_2_split_valid_y.h5"

    if not valid_x.exists() or not valid_y.exists():
        raise FileNotFoundError(
            f"Could not find valid set at:\n{valid_x}\n{valid_y}\n"
            f"Check cfg['data']['data_path'] = {data_dir}"
        )

    ds = PCAMDataset(str(valid_x), str(valid_y), filter_data=False)
    loader = DataLoader(
        ds,
        batch_size=int(cfg["data"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["data"]["num_workers"]),
    )

    device = "cpu"
    model = build_model(cfg).to(device)

    # Load checkpoint if provided
    if ckpt_path:
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        elif isinstance(state, dict) and all(isinstance(k, str) for k in state.keys()):
            model.load_state_dict(state)
        else:
            raise ValueError("Unknown checkpoint format")

    # Predict
    X, Y, P = predict(model, loader, device=device, max_batches=200)

    cm = confusion_matrix(Y, P, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fp_idx = np.where((P == 1) & (Y == 0))[0]
    fn_idx = np.where((P == 0) & (Y == 1))[0]

    print("GLOBAL confusion matrix [[TN, FP],[FN, TP]]:")
    print(cm)
    print(f"TN={tn} FP={fp} FN={fn} TP={tp}")
    print("FP examples idx:", fp_idx[:10])
    print("FN examples idx:", fn_idx[:10])

    # Save 5 FP and 5 FN
    if len(fp_idx) > 0:
        save_grid(X[fp_idx[:5]], "False Positives (pred=1, y=0)", outdir / "false_positives.png", n=5)
    if len(fn_idx) > 0:
        save_grid(X[fn_idx[:5]], "False Negatives (pred=0, y=1)", outdir / "false_negatives.png", n=5)

    # --- Slice example: darkest 10% by mean intensity ---
    brightness = X.mean(axis=(1, 2, 3))  # mean over C,H,W
    thr = np.quantile(brightness, 0.10)
    slice_mask = brightness <= thr

    Ys = Y[slice_mask]
    Ps = P[slice_mask]
    cm_s = confusion_matrix(Ys, Ps, labels=[0, 1])

    print("\nSLICE: darkest 10% by mean intensity")
    print("Slice confusion matrix [[TN, FP],[FN, TP]]:")
    print(cm_s)

    # Save some slice examples
    slice_idx = np.where(slice_mask)[0][:5]
    if len(slice_idx) > 0:
        save_grid(X[slice_idx], "Slice: darkest 10% (sample)", outdir / "slice_dark_examples.png", n=5)

    # Write summary text
    summary = outdir / "summary.txt"
    with open(summary, "w") as f:
        f.write("GLOBAL CM [[TN, FP],[FN, TP]]:\n")
        f.write(str(cm) + "\n")
        f.write(f"TN={tn} FP={fp} FN={fn} TP={tp}\n\n")
        f.write("FP indices (first 10): " + str(fp_idx[:10]) + "\n")
        f.write("FN indices (first 10): " + str(fn_idx[:10]) + "\n\n")
        f.write("SLICE (darkest 10%) CM:\n")
        f.write(str(cm_s) + "\n")

    print(f"\nSaved outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()


