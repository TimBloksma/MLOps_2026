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



def plot_grad_flow(model, step, epoch):
    '''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().item())
            max_grads.append(p.grad.abs().max().item())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])


    bron: https://gist.github.com/Flova/8bed128b41a74142a661883af9e51490
    '''
    grad = {}
    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            grad[f"grads/{n}_mean"] = p.grad.abs().mean().item()
            grad[f"grads/{n}_max"] = p.grad.abs().max().item()
    if grad:
        grad["epoch"] = epoch + 1
        wandb.log(grad, step=step)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    print(OmegaConf.to_yaml(cfg))
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

    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    criterion = nn.CrossEntropyLoss()

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.training.scheduler.step_size,
        gamma = cfg.training.scheduler.gamma
    )

    wandb.watch(model, criterion, log="all")
    wandb.log({"learning_rate": optimizer.param_groups[0]["lr"], "epoch": 0})

    train_losses, val_losses = [], []
    train_loss_step, val_loss_step = [], []

    grad_log = cfg.training.gradlogger

    for epoch in range(cfg.training.epochs):
        model.train()
        epoch_train_loss = 0.0

        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)


            loss.backward()

            if grad_log> 0 and step % grad_log == 0:
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
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item()
                val_loss_step.append(loss.item())

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        #kan hieronder nog samen in 1 snip voor toekomst
        wandb.log({
            "Learning_Rate": lr,
            "epoch": epoch +1,
        })

        wandb.log({
            "train/loss_epoch": avg_train_loss,
            "val/loss_epoch": avg_val_loss,
            "epoch": epoch + 1,
        })
        

        print(f"--- Epoch {epoch+1} Summary | Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} ---")

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, cfg.training.epochs + 1), train_losses, label="Train Loss", marker="o")
    plt.plot(range(1, cfg.training.epochs + 1), val_losses, label="Val Loss", marker="o")
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
