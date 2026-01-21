from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from .pcam import PCAMDataset


def get_dataloaders(data_cfg: Dict) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create Train and Validation DataLoaders using a WeightedRandomSampler.
    Expects cfg.data from Hydra.
    """

    base_path = Path(data_cfg["data_path"])
    batch_size = data_cfg["batch_size"]
    num_workers = data_cfg.get("num_workers", 0)

    transform = transforms.ToTensor()

    # --------------------
    # TRAIN DATASET
    # --------------------
    train_dataset = PCAMDataset(
        x_path=str(base_path / "camelyonpatch_level_2_split_train_x.h5"),
        y_path=str(base_path / "camelyonpatch_level_2_split_train_y.h5"),
        transform=transform,
        filter_data=False,
    )

    labels = torch.tensor(
        [train_dataset[i][1].item() for i in range(len(train_dataset))]
    )

    class_counts = torch.bincount(labels)
    class_counts = torch.clamp(class_counts, min=1)

    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    # --------------------
    # VALIDATION DATASET
    # --------------------
    val_loader = None
    val_x = base_path / "camelyonpatch_level_2_split_valid_x.h5"
    val_y = base_path / "camelyonpatch_level_2_split_valid_y.h5"

    if val_x.exists() and val_y.exists():
        val_dataset = PCAMDataset(
            x_path=str(val_x),
            y_path=str(val_y),
            transform=transform,
            filter_data=False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    print("[DEBUG] Train class counts:", class_counts.tolist())

    return train_loader, val_loader
