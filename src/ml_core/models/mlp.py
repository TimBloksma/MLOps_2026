from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Minimal MLP that:
    - flattens (B, C, H, W) -> (B, C*H*W)
    - outputs logits of shape (B, num_classes)
    """

    def __init__(
        self,
        input_shape: Sequence[int],
        hidden_units: List[int],
        num_classes: int,
    ):
        super().__init__()
        in_features = int(input_shape[0] * input_shape[1] * input_shape[2])

        layers: List[nn.Module] = []
        prev = in_features
        for h in hidden_units:
            layers.append(nn.Linear(prev, int(h)))
            layers.append(nn.ReLU())
            prev = int(h)
        layers.append(nn.Linear(prev, int(num_classes)))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.net(x)
