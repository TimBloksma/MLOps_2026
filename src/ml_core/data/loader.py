from __future__ import annotations

from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .pcam import PCAMDataset

dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "configs")
)

@hydra.main(
    version_base=None,
    config_path=dir,
    config_name="config"
)
def _read_labels(y_path: Path) -> np.ndarray:
    with h5py.File(y_path, "r") as f:
        yds = f["y"] if "y" in f else list(f.values())[0]
        return np.array(yds[:]).reshape(-1).astype(int)


def get_dataloaders(data_cfg: DictConfig) -> Tuple[DataLoader, DataLoader]:
    data_dir = Path(data_cfg.data.data_path)
    batch_size = int(data_cfg.data.batch_size)
    num_workers = int(data_cfg.data.num_workers)

    train_x = data_dir / "camelyonpatch_level_2_split_train_x.h5"
    train_y = data_dir / "camelyonpatch_level_2_split_train_y.h5"
    valid_x = data_dir / "camelyonpatch_level_2_split_valid_x.h5"
    valid_y = data_dir / "camelyonpatch_level_2_split_valid_y.h5"

    train_ds = PCAMDataset(str(train_x), str(train_y), filter_data=False)
    valid_ds = PCAMDataset(str(valid_x), str(valid_y), filter_data=False)

    y = _read_labels(train_y)
    class_counts = np.bincount(y, minlength=2)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[y]

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, valid_loader
