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

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, auc, fbeta_score
import os

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

    fbeta = fbeta_score(
        y_true,
        y_pred,
        beta=2.0,              
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f2": fbeta,                
        "pr_auc": pr_auc,
        "precision_curve": precision,
        "recall_curve": recall,
    }

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Unique WandB run name for multirun
    run = wandb.init(
        project="MLOPS",
        config=cfg_dict,
        name=" TEST 2"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    train_loader, val_loader = get_dataloaders(cfg.data)
    model = MLP(**cfg.model).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    criterion = nn.CrossEntropyLoss()

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.training.scheduler.step_size,
        gamma=cfg.training.scheduler.gamma
    )

    wandb.watch(model, criterion, log="all")
    wandb.log({"learning_rate": optimizer.param_groups[0]["lr"], "epoch": 0})

    train_losses, val_losses = [], []
    train_loss_step, val_loss_step = [], []

    grad_log = cfg.training.gradlogger
    os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)

    best_metric = -float("inf") if cfg.training.mode == "max" else float("inf")
    best_epoch = -1
    champion_path = None

    run_dir = os.getcwd()  # Hydra sets a unique dir per multirun

    for epoch in range(cfg.training.epochs):
        model.train()
        epoch_train_loss = 0.0

        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            if grad_log > 0 and step % grad_log == 0:
                global_step = epoch * len(train_loader) + step
                plot_grad_flow(model, global_step, epoch)

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
        # Reset lists per epoch
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item()
                val_loss_step.append(loss.item())

                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds = torch.argmax(outputs, dim=1)

                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        


        print(f"Epoch {epoch+1} val probs min/max: {min(all_probs):.4f}/{max(all_probs):.4f}, "f"unique values: {len(set(all_probs))}")

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        metrics = compute_metrics(all_labels, all_preds, all_probs)
        monitor_value = metrics[cfg.training.monitor_metric]

        is_best = (
            monitor_value > best_metric
            if cfg.training.mode == "max"
            else monitor_value < best_metric
        )

        if is_best:
            best_metric = monitor_value
            best_epoch = epoch + 1
            champion_path = os.path.join(
                cfg.training.checkpoint_dir,
                f"champion_lr{cfg.training.learning_rate}_bs{cfg.data.batch_size}_hu{'-'.join(map(str,cfg.model.hidden_units))}.pt"
            )

            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_metric": best_metric,
                    "cfg": OmegaConf.to_container(cfg, resolve=True),
                },
                champion_path,
            )

        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        wandb.log(
            {
                "epoch": epoch + 1,
                "learning_rate": lr,
                "train/loss_epoch": avg_train_loss,
                "val/loss_epoch": avg_val_loss,
                "val/accuracy": metrics["accuracy"],
                "val/precision": metrics["precision"],
                "val/recall": metrics["recall"],
                "val/f1": metrics["f1"],
                "val/f2": metrics["f2"],
                "val/pr_auc": metrics["pr_auc"],
                "champion_metric": best_metric,
            }
        )

        # Plot PR curve per epoch
        pr_curve_path = os.path.join(run_dir, f"pr_curve_epoch_{epoch+1}.png")
        plt.figure(figsize=(6,6))
        plt.plot(metrics["recall_curve"], metrics["precision_curve"], label=f'PR AUC={metrics["pr_auc"]:.3f}')
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve Epoch {epoch+1}")
        plt.legend()
        plt.grid(True)
        plt.savefig(pr_curve_path)
        wandb.log({f"pr_curve_epoch_{epoch+1}": wandb.Image(pr_curve_path)})
        plt.close()

        print(f"--- Epoch {epoch+1} Summary | Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} | PR AUC: {metrics['pr_auc']:.4f} ---")

    # Final loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, cfg.training.epochs + 1), train_losses, label="Train Loss", marker="o")
    plt.plot(range(1, cfg.training.epochs + 1), val_losses, label="Val Loss", marker="o")
    plt.title("PCAM Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("CrossEntropy Loss")
    plt.legend()
    plt.grid(True)
    loss_curve_path = os.path.join(run_dir, "resultaten.png")
    plt.savefig(loss_curve_path)
    wandb.log({"loss_curve": wandb.Image(loss_curve_path)})
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(train_loss_step, label="Train Loss (step)", alpha=0.7)
    plt.plot(val_loss_step, label="Val Loss (step)", alpha=0.7)
    plt.title("PCAM Training and Validation Loss per Step")
    plt.xlabel("Step")
    plt.ylabel("CrossEntropy Loss")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    loss_step_path = os.path.join(run_dir, "loss_per_step.png")
    plt.savefig(loss_step_path)
    wandb.log({"loss_per_step": wandb.Image(loss_step_path)})
    plt.close()

    if champion_path is not None:
        artifact = wandb.Artifact(
            name="champion-model",
            type="model",
            metadata={
                "best_epoch": best_epoch,
                "best_metric": best_metric,
                "monitor_metric": cfg.training.monitor_metric,
            },
        )
        artifact.add_file(champion_path)
        wandb.log_artifact(artifact)

    print(f"Training complete. Plots saved to {run_dir}")
    wandb.finish()

if __name__ == "__main__":
    main()
