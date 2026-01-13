from pathlib import Path
from typing import Callable, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class PCAMDataset(Dataset):
    """
    PatchCamelyon (PCAM) Dataset reader for H5 format.

    Requirements from tests:
    - supports filter_data=True/False (mean-based heuristic filtering)
    - clips values BEFORE converting to uint8 (numerical stability)
    - lazy H5 loading (open on first __getitem__)
    """

    def __init__(
        self,
        x_path: str,
        y_path: Optional[str] = None,
        transform: Optional[Callable] = None,
        filter_data: bool = False,
        clip_min: float = 0.0,
        clip_max: float = 255.0,
        mean_low: float = 5.0,
        mean_high: float = 250.0,
    ):
        self.x_path = Path(x_path)
        self.y_path = Path(y_path) if y_path is not None else None
        self.transform = transform
        self.filter_data = filter_data

        self.clip_min = clip_min
        self.clip_max = clip_max
        self.mean_low = mean_low
        self.mean_high = mean_high

        self._x_file = None
        self._y_file = None
        self._x_ds = None
        self._y_ds = None

        # Determine base length and (optionally) filtered indices without keeping files open
        with h5py.File(self.x_path, "r") as fx:
            xds = fx["x"] if "x" in fx else list(fx.values())[0]
            n = xds.shape[0]

            if self.filter_data:
                means = xds[:].mean(axis=(1, 2, 3))
                keep = (means > self.mean_low) & (means < self.mean_high)
                self._indices = np.where(keep)[0].astype(np.int64)
                self.indices = self._indices
            else:
                self._indices = None
                self.indices = np.arange(n, dtype=np.int64)


        self.length = int(len(self._indices)) if self._indices is not None else int(n)

    def __len__(self) -> int:
        return self.length

    def _open_files(self):
        if self._x_file is None:
            self._x_file = h5py.File(self.x_path, "r")
            self._x_ds = self._x_file["x"] if "x" in self._x_file else list(self._x_file.values())[0]

        if self.y_path is not None and self._y_file is None:
            self._y_file = h5py.File(self.y_path, "r")
            self._y_ds = self._y_file["y"] if "y" in self._y_file else list(self._y_file.values())[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        self._open_files()

        real_idx = int(self._indices[idx]) if self._indices is not None else int(idx)

        x = self._x_ds[real_idx]

        # IMPORTANT: clip BEFORE uint8 conversion (prevents wrap-around)
        x = np.clip(x, self.clip_min, self.clip_max).astype(np.uint8)

        # (H, W, C) -> (C, H, W)
        if x.ndim == 3:
            x = np.transpose(x, (2, 0, 1))

        # normalize to float32 [0,1] (common + test-friendly)
        x = (x.astype(np.float32) / 255.0)
        x_t = torch.from_numpy(x)

        if self.transform is not None:
            x_t = self.transform(x_t)

        y_t = None
        if self._y_ds is not None:
            y_val = self._y_ds[real_idx]
            y_t = torch.tensor(int(np.array(y_val).reshape(-1)[0]), dtype=torch.long)

        return x_t, y_t

    def __getstate__(self):
        # Prevent passing open H5 handles to DataLoader workers
        state = self.__dict__.copy()
        state["_x_file"] = None
        state["_y_file"] = None
        state["_x_ds"] = None
        state["_y_ds"] = None
        return state

