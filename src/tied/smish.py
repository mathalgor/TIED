"""Smish activation function.

smish(x) = x * tanh(ln(1 + sigmoid(x)))

From Wang, Ren, Wang. "Smish: A Novel Activation Function for Deep
Learning Methods." Electronics 11.4 (2022): 540.
"""

from __future__ import annotations

import torch
from torch import nn


@torch.jit.script
def smish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.tanh(torch.log(1 + torch.sigmoid(x)))


class Smish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return smish(x)
