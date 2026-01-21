import os
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import hydra
import time
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, auc, fbeta_score

from ml_core.data import get_dataloaders
from ml_core.models import MLP

# --------------------------
# Utilities
# --------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_grad_flow(model, step, epoch):
    grad = {}
    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            grad[f"grads/{n}_mean"] = p.grad.abs().mean().item()
            grad[f"grads/{n}_max"] = p.grad.abs().max().item()
    if grad:
        grad["epoch"] = epoch + 1
        wandb.log(grad, step=step)

def compute_metrics(y_true, y_pred, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    fbeta_val = fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_val,
        "pr_auc": pr_auc,
        "precision_curve": precision,
        "recall_curve": recall,
    }

# --------------------------
# Training/Validation Step
# --------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, grad_log, epoch, train_loss_step):
    model.train()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for step, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient logging
        if grad_log > 0 and step % grad_log == 0:
            global_step = epoch * len(loader) + step
            plot_grad_flow(model, global_step, epoch)

        optimizer.step()

        running_loss += loss.item()
        train_loss_step.append(loss.item())  # log per-step train loss

        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = torch.argmax(outputs, dim=1)
        all_probs.extend(probs.detach().cpu().numpy())
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

        if step % 100 == 0:
            wandb.log({"train/loss_step": loss.item(), "epoch": epoch + 1})
            print(f"Epoch {epoch+1}, Step {step}, Train Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(loader)
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    return avg_loss, metrics

def validate(model, loader, criterion, device, val_loss_step):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            val_loss_step.append(loss.item())  # log per-step val loss

            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)
            all_probs.extend(probs.detach().cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = running_loss / len(loader)
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    return avg_loss, metrics

# --------------------------
# Main
# --------------------------
@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.get("seed", 42))
    
    run_name = f"{cfg.model.hidden_units}_{cfg.data.batch_size}_lr{cfg.training.learning_rate}_{int(time.time())}"


    # WandB
    run = wandb.init(
        project="MLOPS",
        config=OmegaConf.to_container(cfg, resolve=True),
        name=run_name
    )
    wandb.config.update(
    {
        "hidden_units_str": "-".join(map(str, cfg.model.hidden_units)),
        "hidden_units_depth": len(cfg.model.hidden_units),
        "hidden_units_params": sum(cfg.model.hidden_units),
    },
    allow_val_change=True,
)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    # Data
    train_loader, val_loader = get_dataloaders(cfg.data)
    model = MLP(**cfg.model).to(device)

    images, labels = next(iter(train_loader))
    outputs = model(images.to(device))
    print("Outputs (first 5):", outputs[:5])
    print("Labels (first 5):", labels[:5])

    # Optimizer, criterion, scheduler
    optimizer = optim.SGD(model.parameters(),momentum=0.9, lr=cfg.training.learning_rate)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.training.scheduler.step_size,
        gamma=cfg.training.scheduler.gamma
    )

    wandb.watch(model, criterion, log="all")
    grad_log = cfg.training.gradlogger
    os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)

    best_metric = -float("inf") if cfg.training.mode == "max" else float("inf")
    best_epoch = -1
    champion_path = None
    train_losses, val_losses = [], []
    train_loss_step, val_loss_step = [], []

    # Training loop
    for epoch in range(cfg.training.epochs):
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, grad_log, epoch, train_loss_step
        )
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, val_loss_step
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        monitor_value = val_metrics[cfg.training.monitor_metric]
        is_best = (monitor_value > best_metric) if cfg.training.mode == "max" else (monitor_value < best_metric)
        if is_best:
            best_metric = monitor_value
            best_epoch = epoch + 1
            champion_path = os.path.join(
                cfg.training.checkpoint_dir,
                f"champion_lr{cfg.training.learning_rate}_bs{cfg.data.batch_size}_hu{'-'.join(map(str,cfg.model.hidden_units))}.pt"
            )
            torch.save({
                "epoch": best_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_metric": best_metric,
                "cfg": OmegaConf.to_container(cfg, resolve=True),
            }, champion_path)

        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        # WandB logging
        wandb.log({
            "epoch": epoch + 1,
            "learning_rate": lr,
            "train/loss_epoch": train_loss,
            "val/loss_epoch": val_loss,
            "val/accuracy": val_metrics["accuracy"],
            "val/precision": val_metrics["precision"],
            "val/recall": val_metrics["recall"],
            "val/f1": val_metrics["f1"],
            "val/f2": val_metrics["f2"],
            "val/pr_auc": val_metrics["pr_auc"],
            "champion_metric": best_metric,
            "train/loss_step_all": train_loss_step,
            "val/loss_step_all": val_loss_step,
        })

        # PR Curve
        plt.figure(figsize=(6,6))
        plt.plot(val_metrics["recall_curve"], val_metrics["precision_curve"], label=f'PR AUC={val_metrics["pr_auc"]:.3f}')
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve Epoch {epoch+1}")
        plt.legend()
        plt.grid(True)
        pr_curve_path = os.path.join(os.getcwd(), f"pr_curve_epoch_{epoch+1}.png")
        plt.savefig(pr_curve_path)
        wandb.log({f"pr_curve_epoch_{epoch+1}": wandb.Image(pr_curve_path)})
        plt.close()

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | PR AUC: {val_metrics['pr_auc']:.4f}")

    # Final Loss plots
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, cfg.training.epochs + 1), train_losses, label="Train Loss", marker="o")
    plt.plot(range(1, cfg.training.epochs + 1), val_losses, label="Val Loss", marker="o")
    plt.title("PCAM Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("CrossEntropy Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(os.getcwd(), "resultaten.png"))
    wandb.log({"loss_curve": wandb.Image(os.path.join(os.getcwd(), "resultaten.png"))})
    plt.close()

    # Step-level loss plot
    plt.figure(figsize=(12, 6))
    plt.plot(train_loss_step, label="Train Loss (step)", alpha=0.7)
    plt.plot(val_loss_step, label="Val Loss (step)", alpha=0.7)
    plt.title("PCAM Training and Validation Loss per Step")
    plt.xlabel("Step")
    plt.ylabel("CrossEntropy Loss")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(os.getcwd(), "loss_per_step.png"))
    wandb.log({"loss_per_step": wandb.Image(os.path.join(os.getcwd(), "loss_per_step.png"))})
    plt.close()

    # Save artifact
    if champion_path:
        artifact = wandb.Artifact(
            name="champion-model",
            type="model",
            metadata={"best_epoch": best_epoch, "best_metric": best_metric, "monitor_metric": cfg.training.monitor_metric},
        )
        artifact.add_file(champion_path)
        wandb.log_artifact(artifact)

    print(f"Training complete. Plots saved to {os.getcwd()}")
    wandb.finish()

if __name__ == "__main__":
    main()
