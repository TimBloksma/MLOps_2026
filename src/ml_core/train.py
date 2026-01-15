# src/ml_core/train.py

from __future__ import annotations

import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from ml_core.models.mlp import MLP


def main() -> None:
    # ===== Config (pas aan indien nodig) =====
    input_shape = [3, 96, 96]
    hidden_units = [64, 32]
    num_classes = 2

    epochs = 3
    batch_size = 64
    lr = 1e-3
    seed = 42

    # ===== Reproducibility =====
    torch.manual_seed(seed)

    # ===== Device =====
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ===== Model / Optim / Loss =====
    model = MLP(
        input_shape=input_shape,
        hidden_units=hidden_units,
        num_classes=num_classes,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # ===== Track losses =====
    train_losses: list[float] = []
    val_losses: list[float] = []

    # ===== Training Loop =====
    for epoch in range(1, epochs + 1):
        # ---- TRAIN ----
        model.train()

        x = torch.randn(batch_size, *input_shape, device=device)
        y = torch.randint(0, num_classes, (batch_size,), device=device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        train_loss = criterion(logits, y)
        train_loss.backward()
        optimizer.step()

        train_losses.append(float(train_loss.item()))

        # ---- VALIDATION ----
        model.eval()
        with torch.no_grad():
            x_val = torch.randn(batch_size, *input_shape, device=device)
            y_val = torch.randint(0, num_classes, (batch_size,), device=device)

            val_logits = model(x_val)
            val_loss = criterion(val_logits, y_val)

        val_losses.append(float(val_loss.item()))

        print(
            f"Epoch {epoch}/{epochs} "
            f"- train loss: {train_losses[-1]:.4f} "
            f"- val loss: {val_losses[-1]:.4f}"
        )

    # ===== Save CSV =====
    out_dir = Path("experiments")
    out_dir.mkdir(exist_ok=True)

    csv_path = out_dir / "losses.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for i, (t, v) in enumerate(zip(train_losses, val_losses), start=1):
            writer.writerow([i, t, v])

    print(f"Saved losses to {csv_path}")

    # ===== Plot (optional but useful) =====
    try:
        import matplotlib.pyplot as plt

        epochs_list = list(range(1, epochs + 1))
        plt.figure()
        plt.plot(epochs_list, train_losses, label="train loss")
        plt.plot(epochs_list, val_losses, label="val loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("Learning curves")
        plt.legend()
        plot_path = out_dir / "loss_curve.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved loss plot to {plot_path}")
    except Exception as e:
        print(f"Plotting skipped (matplotlib issue): {e}")


if __name__ == "__main__":
    main()
