import torch
from ml_core.models.mlp import MLP
from torchvision import transforms
import numpy as np

checkpoint_path = "checkpoints/champion_lr0.01_bs64_hu64-32_1769010302.pt"
model_cfg = {
    "input_shape": [3, 96, 96],
    "hidden_units": [64, 32],
    "num_classes": 2,
    "dropout_rate": 0.2
}
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MLP(**model_cfg).to(device)
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


sample_image = np.random.rand(3,96,96).astype(np.float32)
sample_image = torch.tensor(sample_image).unsqueeze(0).to(device) 

with torch.no_grad():
    output = model(sample_image)
    pred = torch.argmax(output, dim=1)
print("Predicted class:", int(pred.item()))
