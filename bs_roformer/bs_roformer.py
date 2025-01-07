from __future__ import annotations
from functools import partial

import torch
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from bs_roformer.attend import Attend

from beartype.typing import Callable
from beartype import beartype

from rotary_embedding_torch import RotaryEmbedding

from einops import rearrange, pack, unpack

from hyper_connections import get_init_and_expand_reduce_stream_functions

# helper functions

def exists(val):
    return val is not None

def default(v, d):
    return v if exists(v) else d

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

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
        flash = True,
        learned_value_residual_mix = False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head **-0.5
        dim_inner = heads * dim_head

        self.rotary_embed = rotary_embed

        self.attend = Attend(flash = flash, dropout = dropout)

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)

        self.to_value_residual_mix = nn.Linear(dim, heads) if learned_value_residual_mix else None

        self.to_gates = nn.Linear(dim, heads)

        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(self, x, value_residual = None):
        x = self.norm(x)

        q, k, v = rearrange(self.to_qkv(x), 'b n (qkv h d) -> qkv b h n d', qkv = 3, h = self.heads)

        orig_v = v

        if exists(self.to_value_residual_mix):
            mix = self.to_value_residual_mix(x)
            mix = rearrange(mix, 'b n h -> b h n 1').sigmoid()

            assert exists(value_residual)
            v = v.lerp(value_residual, mix)

        if exists(self.rotary_embed):
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)

        out = self.attend(q, k, v)

        gates = self.to_gates(x)
        out = out * rearrange(gates, 'b n h -> b h n 1').sigmoid()

        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out), orig_v

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
        flash_attn = True,
        add_value_residual = False,
        num_residual_streams = 1
    ):
        super().__init__()
        self.layers = ModuleList([])

        init_hyper_conn, *_ = get_init_and_expand_reduce_stream_functions(num_residual_streams, disable = num_residual_streams == 1)

        for _ in range(depth):
            self.layers.append(ModuleList([
                init_hyper_conn(dim = dim, branch = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_embed = rotary_embed, flash = flash_attn, learned_value_residual_mix = add_value_residual)),
                init_hyper_conn(dim = dim, branch = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout))
            ]))

        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x, value_residual = None):

        first_values = None

        for attn, ff in self.layers:
            x, next_values = attn(x, value_residual = value_residual)

            first_values = default(first_values, next_values)

            x = ff(x)

        return self.norm(x), first_values

# bandsplit module

class BandSplit(Module):
    @beartype
    def __init__(
        self,
        dim,
        dim_inputs: tuple[int, ...]
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

def MLP(
    dim_in,
    dim_out,
    dim_hidden = None,
    depth = 1,
    activation = nn.Tanh
):
    dim_hidden = default(dim_hidden, dim_in)

    net = []
    dims = (dim_in, *((dim_hidden,) * (depth - 1)), dim_out)

    for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
        is_last = ind == (len(dims) - 2)

        net.append(nn.Linear(layer_dim_in, layer_dim_out))

        if is_last:
            continue

        net.append(activation())

    return nn.Sequential(*net)

class MaskEstimator(Module):
    @beartype
    def __init__(
        self,
        dim,
        dim_inputs: tuple[int, ...],
        depth,
        mlp_expansion_factor = 4
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_freqs = ModuleList([])
        dim_hidden = dim * mlp_expansion_factor

        for dim_in in dim_inputs:
            net = []

            mlp = nn.Sequential(
                MLP(dim, dim_in * 2, dim_hidden = dim_hidden, depth = depth),
                nn.GLU(dim = -1)
            )

            self.to_freqs.append(mlp)

    def forward(self, x):
        x = x.unbind(dim = -2)

        outs = []

        for band_features, mlp in zip(x, self.to_freqs):
            freq_out = mlp(band_features)
            outs.append(freq_out)

        return torch.cat(outs, dim = -1)

# main class

DEFAULT_FREQS_PER_BANDS = (
  2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
  2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
  2, 2, 2, 2,
  4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
  12, 12, 12, 12, 12, 12, 12, 12,
  24, 24, 24, 24, 24, 24, 24, 24,
  48, 48, 48, 48, 48, 48, 48, 48,
  128, 129,
)

class BSRoformer(Module):

    @beartype
    def __init__(
        self,
        dim,
        *,
        depth,
        stereo = False,
        num_stems = 1,
        time_transformer_depth = 2,
        freq_transformer_depth = 2,
        freqs_per_bands: tuple[int, ...] = DEFAULT_FREQS_PER_BANDS,  # in the paper, they divide into ~60 bands, test with 1 for starters
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        flash_attn = True,
        num_residual_streams = 4, # set to 1. to disable hyper connections
        dim_freqs_in = 1025,
        stft_n_fft = 2048,
        stft_hop_length = 512, # 10ms at 44100Hz, from sections 4.1, 4.4 in the paper - @faroit recommends // 2 or // 4 for better reconstruction
        stft_win_length = 2048,
        stft_normalized = False,
        stft_window_fn: Callable | None = None,
        mask_estimator_depth = 2,
        multi_stft_resolution_loss_weight = 1.,
        multi_stft_resolutions_window_sizes: tuple[int, ...] = (4096, 2048, 1024, 512, 256),
        multi_stft_hop_size = 147,
        multi_stft_normalized = False,
        multi_stft_window_fn: Callable = torch.hann_window
    ):
        super().__init__()

        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems

        _, self.expand_stream, self.reduce_stream = get_init_and_expand_reduce_stream_functions(num_residual_streams, disable = num_residual_streams == 1)

        self.layers = ModuleList([])

        transformer_kwargs = dict(
            dim = dim,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            flash_attn = flash_attn,
            num_residual_streams = num_residual_streams,
            norm_output = False,
        )

        time_rotary_embed = RotaryEmbedding(dim = dim_head)
        freq_rotary_embed = RotaryEmbedding(dim = dim_head)

        for layer_index in range(depth):
            is_first = layer_index == 0

            self.layers.append(nn.ModuleList([
                Transformer(depth = time_transformer_depth, rotary_embed = time_rotary_embed, add_value_residual = not is_first, **transformer_kwargs),
                Transformer(depth = freq_transformer_depth, rotary_embed = freq_rotary_embed, add_value_residual = not is_first, **transformer_kwargs)
            ]))

        self.final_norm = RMSNorm(dim)

        self.stft_kwargs = dict(
            n_fft = stft_n_fft,
            hop_length = stft_hop_length,
            win_length = stft_win_length,
            normalized = stft_normalized
        )

        self.stft_window_fn = partial(default(stft_window_fn, torch.hann_window), stft_win_length)

        freqs = torch.stft(torch.randn(1, 4096), **self.stft_kwargs, return_complex = True).shape[1]

        assert len(freqs_per_bands) > 1
        assert sum(freqs_per_bands) == freqs, f'the number of freqs in the bands must equal {freqs} based on the STFT settings, but got {sum(freqs_per_bands)}'

        freqs_per_bands_with_complex = tuple(2 * f * self.audio_channels for f in freqs_per_bands)

        self.band_split = BandSplit(
            dim = dim,
            dim_inputs = freqs_per_bands_with_complex
        )

        self.mask_estimators = nn.ModuleList([])

        for _ in range(num_stems):
            mask_estimator = MaskEstimator(
                dim = dim,
                dim_inputs = freqs_per_bands_with_complex,
                depth = mask_estimator_depth
            )

            self.mask_estimators.append(mask_estimator)

        # for the multi-resolution stft loss

        self.multi_stft_resolution_loss_weight = multi_stft_resolution_loss_weight
        self.multi_stft_resolutions_window_sizes = multi_stft_resolutions_window_sizes
        self.multi_stft_n_fft = stft_n_fft
        self.multi_stft_window_fn = multi_stft_window_fn

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
        s - audio channel (1 for mono, 2 for stereo)
        n - number of 'stems'
        c - complex (2)
        d - feature dimension
        """

        device = raw_audio.device

        if raw_audio.ndim == 2:
            raw_audio = rearrange(raw_audio, 'b t -> b 1 t')

        channels = raw_audio.shape[1]
        assert (not self.stereo and channels == 1) or (self.stereo and channels == 2), 'stereo needs to be set to True if passing in audio signal that is stereo (channel dimension of 2). also need to be False if mono (channel dimension of 1)'

        # to stft

        raw_audio, batch_audio_channel_packed_shape = pack_one(raw_audio, '* t')

        stft_window = self.stft_window_fn(device = device)

        stft_repr = torch.stft(raw_audio, **self.stft_kwargs, window = stft_window, return_complex = True)
        stft_repr = torch.view_as_real(stft_repr)

        stft_repr = unpack_one(stft_repr, batch_audio_channel_packed_shape, '* f t c')
        stft_repr = rearrange(stft_repr, 'b s f t c -> b (f s) t c') # merge stereo / mono into the frequency, with frequency leading dimension, for band splitting

        x = rearrange(stft_repr, 'b f t c -> b t (f c)')

        x = self.band_split(x)

        # value residuals

        time_v_residual = None
        freq_v_residual = None

        # maybe expand residual streams

        x = self.expand_stream(x)

        # axial / hierarchical attention

        for time_transformer, freq_transformer in self.layers:

            x = rearrange(x, 'b t f d -> b f t d')
            x, ps = pack([x], '* t d')

            x, next_time_v_residual = time_transformer(x, value_residual = time_v_residual)

            time_v_residual = default(time_v_residual, next_time_v_residual)

            x, = unpack(x, ps, '* t d')
            x = rearrange(x, 'b f t d -> b t f d')
            x, ps = pack([x], '* f d')

            x, next_freq_v_residual = freq_transformer(x, value_residual = freq_v_residual)

            freq_v_residual = default(freq_v_residual, next_freq_v_residual)

            x, = unpack(x, ps, '* f d')

        # maybe reduce residual streams

        x = self.reduce_stream(x)

        x = self.final_norm(x)

        num_stems = len(self.mask_estimators)

        mask = torch.stack([fn(x) for fn in self.mask_estimators], dim = 1)
        mask = rearrange(mask, 'b n t (f c) -> b n f t c', c = 2)

        # modulate frequency representation

        stft_repr = rearrange(stft_repr, 'b f t c -> b 1 f t c')

        # complex number multiplication

        stft_repr = torch.view_as_complex(stft_repr)
        mask = torch.view_as_complex(mask)

        stft_repr = stft_repr * mask

        # istft

        stft_repr = rearrange(stft_repr, 'b n (f s) t -> (b n s) f t', s = self.audio_channels)

        recon_audio = torch.istft(stft_repr, **self.stft_kwargs, window = stft_window, return_complex = False)

        recon_audio = rearrange(recon_audio, '(b n s) t -> b n s t', s = self.audio_channels, n = num_stems)

        if num_stems == 1:
            recon_audio = rearrange(recon_audio, 'b 1 s t -> b s t')

        # if a target is passed in, calculate loss for learning

        if not exists(target):
            return recon_audio

        if self.num_stems > 1:
            assert target.ndim == 4 and target.shape[1] == self.num_stems
        
        if target.ndim == 2:
            target = rearrange(target, '... t -> ... 1 t')

        target = target[..., :recon_audio.shape[-1]] # protect against lost length on istft

        loss = F.l1_loss(recon_audio, target)

        multi_stft_resolution_loss = 0.

        for window_size in self.multi_stft_resolutions_window_sizes:

            res_stft_kwargs = dict(
                n_fft = max(window_size, self.multi_stft_n_fft),  # not sure what n_fft is across multi resolution stft
                win_length = window_size,
                return_complex = True,
                window = self.multi_stft_window_fn(window_size, device = device),
                **self.multi_stft_kwargs,
            )

            recon_Y = torch.stft(rearrange(recon_audio, '... s t -> (... s) t'), **res_stft_kwargs)
            target_Y = torch.stft(rearrange(target, '... s t -> (... s) t'), **res_stft_kwargs)

            multi_stft_resolution_loss = multi_stft_resolution_loss + F.l1_loss(recon_Y, target_Y)

        weighted_multi_resolution_loss = multi_stft_resolution_loss * self.multi_stft_resolution_loss_weight

        total_loss =  loss + weighted_multi_resolution_loss

        if not return_loss_breakdown:
            return total_loss

        return total_loss, (loss, multi_stft_resolution_loss)
