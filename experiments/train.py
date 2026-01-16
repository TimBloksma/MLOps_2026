import random
import wandb
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import hydra
from omegaconf import DictConfig, OmegaConf

from ml_core.data import get_dataloaders
from ml_core.models import MLP

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    run = wandb.init(
        project="MLOPS",
        config=cfg_dict,
    )

    wandb.config.update(
        {"hydra_config_yaml": OmegaConf.to_yaml(cfg)},
        allow_val_change=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    train_loader, val_loader = get_dataloaders(cfg.data)
    model = MLP(**cfg.model).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    wandb.watch(model, criterion, log="all")

    train_losses, val_losses = [], []
    train_loss_step, val_loss_step = [], []

    for epoch in range(cfg.epochs):
        model.train()
        epoch_train_loss = 0.0

        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            train_loss_step.append(loss.item())

            if step % 100 == 0:
                wandb.log({"train/loss_step": loss.item(), "epoch": epoch + 1})
                print(f"Epoch {epoch+1}, Step {step}, Train Loss: {loss.item():.4f}")

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item()
                val_loss_step.append(loss.item())

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        wandb.log({
            "train/loss_epoch": avg_train_loss,
            "val/loss_epoch": avg_val_loss,
            "epoch": epoch + 1,
        })

        print(f"--- Epoch {epoch+1} Summary | Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} ---")

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, cfg.epochs + 1), train_losses, label="Train Loss", marker="o")
    plt.plot(range(1, cfg.epochs + 1), val_losses, label="Val Loss", marker="o")
    plt.title("PCAM Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("CrossEntropy Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("resultaten.png")
    wandb.log({"loss_curve": wandb.Image("resultaten.png")})

    plt.figure(figsize=(12, 6))
    plt.plot(train_loss_step, label="Train Loss (step)", alpha=0.7)
    plt.plot(val_loss_step, label="Val Loss (step)", alpha=0.7)
    plt.title("PCAM Training and Validation Loss per Step")
    plt.xlabel("Step")
    plt.ylabel("CrossEntropy Loss")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_per_step.png")
    wandb.log({"loss_per_step": wandb.Image("loss_per_step.png")})

    print("Training complete. Plot saved as resultaten.png and loss_per_step.png")
    wandb.finish()


if __name__ == "__main__":
    main()
