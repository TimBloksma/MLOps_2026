from pathlib import Path
from typing import Dict
from torch.utils.data import DataLoader
from torchvision import transforms
from .pcam import PCAMDataset 

def get_test_loader(cfg: Dict) -> DataLoader:
    base_path = Path(cfg["data_path"])
    batch_size = cfg["batch_size"]

    x_test = str(base_path / "camelyonpatch_level_2_split_test_x.h5")
    y_test = str(base_path / "camelyonpatch_level_2_split_test_y.h5")

    test_ds = PCAMDataset(x_test, y_test, transform=transforms.ToTensor())  # no meta_test
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.get("num_workers", 0)
    )
    return test_loader