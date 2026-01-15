# src/ml_core/models/mlp.py

from __future__ import annotations
from typing import Sequence, List, Union

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    MLP die input flattened en logits teruggeeft.

    Verwachte config (volgens tests):
      - input_shape: [C, H, W] of [features]
      - hidden_units: lijst met hidden layer groottes
      - num_classes: aantal output classes (logits)
    """

    def __init__(
        self,
        input_shape: Sequence[int],
        hidden_units: Sequence[int],
        num_classes: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if len(input_shape) < 1:
            raise ValueError(f"input_shape moet minstens 1 dimensie hebben, kreeg {input_shape}")
        if num_classes <= 0:
            raise ValueError(f"num_classes moet > 0 zijn, kreeg {num_classes}")
        if not isinstance(hidden_units, (list, tuple)) or len(hidden_units) == 0:
            raise ValueError(f"hidden_units moet een niet-lege lijst/tuple zijn, kreeg {hidden_units}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout moet in [0, 1) liggen, kreeg {dropout}")

        # Bereken in_features uit input_shape, bv [3,96,96] -> 27648
        in_features = 1
        for d in input_shape:
            in_features *= int(d)

        layers: List[nn.Module] = []
        prev = in_features

        for h in hidden_units:
            if h <= 0:
                raise ValueError(f"hidden_units waarden moeten > 0 zijn, kreeg {hidden_units}")
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            prev = h

        layers.append(nn.Linear(prev, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten naar (B, features)
        x = x.view(x.size(0), -1)
        return self.net(x)

