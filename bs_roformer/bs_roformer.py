import torch
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from bs_roformer.attend import Attend

from beartype.typing import Tuple, Optional, List
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

# bandsplit module

class BandSplit(Module):
    @beartype
    def __init__(
        self,
        dim,
        dim_inputs: Tuple[int, ...]
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = ModuleList([])

        for dim_in in dim_inputs:
            net = nn.Sequential(
                RMSNorm(dim_in),
                nn.Linear(dim_in, dim)
            )

            self.to_features.append(net)

    def forward(self, x):
        x = x.split(self.dim_inputs, dim = -1)

        outs = []
        for split_input, to_feature in zip(x, self.to_features):
            split_output = to_feature(split_input)
            outs.append(split_output)

        return torch.stack(outs, dim = -2)

class LinearGLUWithTanH(Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim = -1)
        return x.tanh() * gate.sigmoid()

class MaskEstimator(Module):
    @beartype
    def __init__(
        self,
        dim,
        dim_inputs: Tuple[int, ...],
        depth
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_freqs = ModuleList([])

        for dim_in in dim_inputs:
            net = []

            for ind in range(depth):
                is_last = ind == (depth - 1)
                dim_out = dim if not is_last else dim_in
                net.append(LinearGLUWithTanH(dim, dim_out))

            self.to_freqs.append(nn.Sequential(*net))

    def forward(self, x):
        x = x.unbind(dim = -2)

        outs = []

        for band_features, to_freq in zip(x, self.to_freqs):
            freq_out = to_freq(band_features)
            outs.append(freq_out)

        return torch.cat(outs, dim = -1)

# main class

class BSRoformer(Module):

    @beartype
    def __init__(
        self,
        dim,
        *,
        depth,
        time_transformer_depth = 2,
        freq_transformer_depth = 2,
        freqs_per_bands: Tuple[int, ...] = (512, 513),  # in the paper, they divide into ~60 bands, test with 1 for starters
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        flash_attn = True,
        dim_freqs_in = 1025,
        stft_n_fft = 2048,
        stft_hop_length = 512, # 10ms at 44100Hz, from sections 4.1, 4.4 in the paper - @faroit recommends // 2 or // 4 for better reconstruction
        stft_win_length = 2048,
        stft_normalized = False,
        mask_estimator_depth = 1,
        multi_stft_resolution_loss_weight = 1.,
        multi_stft_resolutions_window_sizes: Tuple[int, ...] = (4096, 2048, 1024, 512, 256),
        multi_stft_hop_size = 147,
        multi_stft_normalized = False
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

        self.stft_kwargs = dict(
            n_fft = stft_n_fft,
            hop_length = stft_hop_length,
            win_length = stft_win_length,
            normalized = stft_normalized
        )

        freqs = torch.stft(torch.randn(1, 1024), **self.stft_kwargs, return_complex = True).shape[1]

        assert len(freqs_per_bands) > 1
        assert sum(freqs_per_bands) == freqs, f'the number of freqs in the bands must equal {freqs} based on the STFT settings'

        freqs_per_bands_with_complex = tuple(2 * f for f in freqs_per_bands)

        self.band_split = BandSplit(
            dim = dim,
            dim_inputs = freqs_per_bands_with_complex
        )

        self.mask_estimator = MaskEstimator(
            dim = dim,
            dim_inputs = freqs_per_bands_with_complex,
            depth = mask_estimator_depth
        )

        # for the multi-resolution stft loss

        self.multi_stft_resolution_loss_weight = multi_stft_resolution_loss_weight
        self.multi_stft_resolutions_window_sizes = multi_stft_resolutions_window_sizes
        self.multi_stft_n_fft = stft_n_fft

        self.multi_stft_kwargs = dict(
            hop_length = multi_stft_hop_size,
            normalized = multi_stft_normalized
        )

    def forward(
        self,
        raw_audio,
        target = None,
        return_loss_breakdown = False
    ):
        """
        einops

        b - batch
        f - freq
        t - time
        c - complex (2)
        d - feature dimension
        """

        # to stft

        stft_repr = torch.stft(raw_audio, **self.stft_kwargs, return_complex = True)
        stft_repr = torch.view_as_real(stft_repr)

        x = rearrange(stft_repr, 'b f t c -> b t (f c)')

        x = self.band_split(x)

        # axial / hierarchical attention

        for time_transformer, freq_transformer in self.layers:

            x = rearrange(x, 'b t f d -> b f t d')
            x, ps = pack([x], 'b * d')

            x = time_transformer(x)

            x, = unpack(x, ps, 'b * d')
            x = rearrange(x, 'b f t d -> b t f d')
            x, ps = pack([x], 'b * d')

            x = freq_transformer(x)

            x, = unpack(x, ps, 'b * d')

        mask = self.mask_estimator(x)
        mask = rearrange(mask, 'b t (f c) -> b f t c', c = 2)

        # modulate frequency representation

        stft_repr = stft_repr * mask

        # istft

        stft_repr = torch.view_as_complex(stft_repr)

        recon_audio = torch.istft(stft_repr, **self.stft_kwargs, return_complex = False)

        # if a target is passed in, calculate loss for learning

        if not exists(target):
            return recon_audio

        target = target[..., :recon_audio.shape[-1]] # protect against lost length on istft

        loss = F.l1_loss(recon_audio, target)

        multi_stft_resolution_loss = 0.

        for window_size in self.multi_stft_resolutions_window_sizes:

            res_stft_kwargs = dict(
                n_fft = max(window_size, self.multi_stft_n_fft),  # not sure what n_fft is across multi resolution stft
                win_length = window_size,
                return_complex = True,
                **self.multi_stft_kwargs,
            )

            recon_Y = torch.stft(recon_audio, **res_stft_kwargs)
            target_Y = torch.stft(target, **res_stft_kwargs)

            multi_stft_resolution_loss = multi_stft_resolution_loss + F.l1_loss(recon_Y, target_Y)

        weighted_multi_resolution_loss = multi_stft_resolution_loss * self.multi_stft_resolution_loss_weight

        total_loss =  loss + weighted_multi_resolution_loss

        if not return_loss_breakdown:
            return total_loss

        return total_loss, (loss, multi_stft_resolution_loss)
