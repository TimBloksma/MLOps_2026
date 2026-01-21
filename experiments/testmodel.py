# experiments/testmodel.py
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from ml_core.models.mlp import MLP
from ml_core.data.loader_test import get_test_loader
import hydra
from omegaconf import DictConfig

def infer_hidden_units_from_checkpoint(checkpoint):
    """
    Infers hidden layer sizes from checkpoint state_dict.
    Excludes the final output layer.
    """
    state_dict = checkpoint["model_state_dict"]
    layers = []
    i = 0
    # Sequential MLP layers: Linear -> ReLU -> Dropout
    while f"network.{i}.weight" in state_dict:
        weight = state_dict[f"network.{i}.weight"]
        layers.append(weight.size(0))  # output size of the layer
        i += 3  # skip ReLU and Dropout
    # Last layer maps to num_classes, remove it from hidden_units
    hidden_units = layers[:-1]
    return hidden_units

def evaluate_checkpoint(checkpoint_path: Path, test_loader, device="cpu", save_dir: Path = Path(".")):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Infer hidden layers automatically
    hidden_units = infer_hidden_units_from_checkpoint(checkpoint)
    
    # Create model with the exact hidden units and input shape
    model = MLP(input_shape=[3, 96, 96], hidden_units=hidden_units, num_classes=2, dropout_rate=0.2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix for {checkpoint_path.name}")

    # Save plot
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_path = save_dir / f"confusion_matrix_{checkpoint_path.stem}.png"
    plt.savefig(plot_path)
    print(f"Confusion matrix saved to: {plot_path}")
    plt.show()

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Path to checkpoint from Hydra
    checkpoint_path = Path(cfg.get("checkpoint_file"))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load test dataset using your dedicated loader
    test_loader = get_test_loader(cfg.data)

    # Use Hydra run dir or current directory as save location
    save_dir = Path(hydra.utils.get_original_cwd()) / "results"
    evaluate_checkpoint(checkpoint_path, test_loader, device=device, save_dir=save_dir)

if __name__ == "__main__":
    main()
