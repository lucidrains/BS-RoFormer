import torch
from torch import nn, einsum, Tensor
from torch.nn import Module
import torch.nn.functional as F

from beartype import beartype

from einops import rearrange

# helper functions

def exists(val):
    return val is not None

# attention

class Attention(Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        rotary_emb = None
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head **-0.5
        dim_inner = heads * dim_head

        self.rotary_emb = rotary_emb

        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)
        self.to_out = nn.Linear(dim_inner, dim, bias = False)

    def forward(self, x):
        q, k, v = rearrange(self.to_qkv(x), 'b n (qkv h d) -> qkv b n h d', qkv = 3, h = self.heads)

        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        sim = einsum('b h i d, b h j d -> b h i j') * self.scale

        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# main class

class BSRoformer(Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x):
        return x
