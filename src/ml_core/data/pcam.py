from typing import Callable, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class PCAMDataset(Dataset):
    def __init__(
        self,
        x_path: str,
        y_path: str,
        transform: Optional[Callable] = None,
        filter_data: bool = False,
    ):
        """
        PCAM Dataset for Camelyon16 patch classification.
        
        Args:
            x_path (str): Path to H5 file containing images (dataset 'x').
            y_path (str): Path to H5 file containing labels (dataset 'y').
            transform (Callable, optional): Transform to apply to images. Defaults to ToTensor().
            filter_data (bool): If True, removes all-black or all-white patches.
        """
        self.x_data = h5py.File(x_path, "r")["x"]
        self.y_data = h5py.File(y_path, "r")["y"]

        # Default transform to ToTensor if none provided
        self.transform = transform or transforms.ToTensor()

        # Initialize indices
        self.indices = np.arange(len(self.x_data))

        # Optional filtering
        if filter_data:
            valid_indices = []
            for i in range(len(self.x_data)):
                mean_val = np.mean(self.x_data[i])
                if 0 < mean_val < 255:  # drop blackouts (0) and washouts (255)
                    valid_indices.append(i)
            self.indices = np.array(valid_indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        real_idx = self.indices[idx]
        img = self.x_data[real_idx]
        label = self.y_data[real_idx].item()

        # Handle NaNs and clip
        img = np.nan_to_num(img, nan=0.0)
        img = np.clip(img, 0, 255).astype(np.uint8)

        # Apply transform if provided
        if self.transform:
            img = self.transform(img)
        else:
            # Convert to tensor in C,H,W format
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


        return img, torch.tensor(label, dtype=torch.long)
