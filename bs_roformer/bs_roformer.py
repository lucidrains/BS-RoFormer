import torch
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from bs_roformer.attend import Attend

from beartype import beartype

from rotary_embedding_torch import RotaryEmbedding

from einops import rearrange, pack, unpack

# helper functions

def exists(val):
    return val is not None

# norm

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma

# attention

class FeedForward(Module):
    def __init__(
        self,
        dim,
        mult = 4,
        dropout = 0.
    ):
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        rotary_embed = None,
        flash = True
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head **-0.5
        dim_inner = heads * dim_head

        self.rotary_embed = rotary_embed

        self.attend = Attend(flash = flash, dropout = dropout)

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)

        q, k, v = rearrange(self.to_qkv(x), 'b n (qkv h d) -> qkv b h n d', qkv = 3, h = self.heads)

        if exists(self.rotary_embed):
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        ff_mult = 4,
        norm_output = True,
        rotary_embed = None,
        flash_attn = True
    ):
        super().__init__()
        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_embed = rotary_embed, flash = flash_attn),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
            ]))

        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x):

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

# main class

class BSRoformer(Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        time_transformer_depth = 2,
        freq_transformer_depth = 2,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        flash_attn = True
    ):
        super().__init__()

        self.layers = ModuleList([])

        transformer_kwargs = dict(
            dim = dim,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            flash_attn = flash_attn
        )

        time_rotary_embed = RotaryEmbedding(dim = dim_head)
        freq_rotary_embed = RotaryEmbedding(dim = dim_head)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Transformer(depth = time_transformer_depth, rotary_embed = time_rotary_embed, **transformer_kwargs),
                Transformer(depth = freq_transformer_depth, rotary_embed = freq_rotary_embed, **transformer_kwargs)
            ]))

    def forward(self, x):

        for time_transformer, freq_transformer in self.layers:

            x = rearrange(x, 'b t f d -> b f t d')
            x, ps = pack([x], 'b * d')

            x = time_transformer(x)

            x, = unpack(x, ps, 'b * d')
            x = rearrange(x, 'b f t d -> b t f d')
            x, ps = pack([x], 'b * d')

            x = freq_transformer(x)

            x, = unpack(x, ps, 'b * d')

        return x
