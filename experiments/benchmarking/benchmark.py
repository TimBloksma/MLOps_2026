import argparse
import time
from pathlib import Path

import torch
import yaml

from ml_core.data import get_dataloaders
from ml_core.models import MLP


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def measure_throughput(model, dataloader, device: str, num_steps: int, warmup_steps: int) -> float:
    model.eval()

    it = iter(dataloader)

    # warmup important for GPU kernels and caching
    for _ in range(warmup_steps):
        batch = next(it)
        images = batch[0].to(device, non_blocking=True)
        _ = model(images)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    # timed section
    images_seen = 0
    start = time.perf_counter()
    for _ in range(num_steps):
        batch = next(it)
        images = batch[0].to(device, non_blocking=True)
        _ = model(images)
        images_seen += images.size(0)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    end = time.perf_counter()

    seconds = end - start
    return images_seen / seconds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config (same style as training)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--warmup_steps", type=int, default=10)
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    # override config for benchmarking
    cfg.setdefault("data", {})
    cfg["data"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but CUDA is not available in this environment.")

    # build loaders + model
    train_loader, _ = get_dataloaders(cfg)

    model_cfg = cfg.get("model", {})
    input_shape = model_cfg.get("input_shape", [3, 96, 96])
    hidden_units = model_cfg.get("hidden_units", [64, 32])
    num_classes = model_cfg.get("num_classes", 2)

    model = MLP(input_shape=input_shape, hidden_units=hidden_units, num_classes=num_classes).to(device)

    # print device info for evidence
    print("=== BENCHMARK INFO ===")
    print("config:", Path(args.config).as_posix())
    print("device:", device)
    if device == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))
    print("batch_size:", args.batch_size)
    print("num_workers:", cfg["data"].get("num_workers", None))
    print("======================")

    img_s = measure_throughput(
        model=model,
        dataloader=train_loader,
        device=device,
        num_steps=args.num_steps,
        warmup_steps=args.warmup_steps,
    )

    print(f"THROUGHPUT_IMG_S={img_s:.3f}")


if __name__ == "__main__":
    main()

