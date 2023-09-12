import torch
from torch import nn, einsum, Tensor
from torch.nn import Module
import torch.nn.functional as F

from beartype import beartype

from einops import rearrange

# helper functions

def exists(val):
    return val is not None

# main class

class BSRoformer(Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x):
        return x
