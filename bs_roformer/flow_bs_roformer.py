from __future__ import annotations
import math
from functools import partial
from collections import namedtuple

import torch
from torch import nn, einsum, tensor, Tensor, cat
from torch.nn import Module, ModuleList, Linear
import torch.nn.functional as F

from bs_roformer.attend import Attend

from beartype.typing import Callable
from beartype import beartype

from rotary_embedding_torch import RotaryEmbedding
from PoPE_pytorch import PoPE, flash_attn_with_pope

import einx
from einops import rearrange, repeat, reduce

from hyper_connections import mc_get_init_and_expand_reduce_stream_functions
from torch_einops_utils import pad_right_ndim_to, pack_with_inverse, tree_flatten_with_inverse

# helper functions

Losses = namedtuple('Losses', ['loss', 'multi_stft_resolution_loss'])

def exists(val):
    return val is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

# norm

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale

class AdaLN(Module):
    def __init__(
        self,
        dim,
        time_cond_dim,
        branch,
        weight_init_std = 1e-3
    ):
        super().__init__()
        self.branch = branch
        self.norm = RMSNorm(dim)

        self.time_cond_dim = time_cond_dim

        if not exists(time_cond_dim):
            return

        self.to_gamma_beta_gate = Linear(time_cond_dim, dim * 3)

        # init

        nn.init.zeros_(self.to_gamma_beta_gate.weight)
        nn.init.normal_(self.to_gamma_beta_gate.weight[dim * 2:], std = weight_init_std) # https://openreview.net/forum?id=vE66hG5t1W
        nn.init.zeros_(self.to_gamma_beta_gate.bias)

    def forward(
        self,
        x,
        time_cond = None,
        **kwargs
    ):
        if not exists(self.time_cond_dim):
            return self.branch(self.norm(x), **kwargs)

        assert exists(time_cond)

        b, n = time_cond.shape[0], x.shape[0] // time_cond.shape[0]

        x = self.norm(x)

        gamma, beta, gate = self.to_gamma_beta_gate(time_cond).chunk(3, dim = -1)
        gamma, beta, gate = (repeat(t, 'b d -> (b n) 1 d', n = n) for t in (gamma, beta, gate))

        x = x * (gamma + 1) + beta

        out = self.branch(x, **kwargs)

        flat_out, pack_inverse = tree_flatten_with_inverse(out)
        out, *rest = flat_out

        out = out * gate.sigmoid()

        return pack_inverse((out, *rest))

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
            Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            Linear(dim_inner, dim),
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
        pope_embed = None,
        flash = True,
        learned_value_residual_mix = False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head **-0.5
        dim_inner = heads * dim_head

        self.rotary_embed = rotary_embed
        self.pope_embed = pope_embed

        assert not (exists(rotary_embed) and exists(pope_embed)), 'cannot have both rotary and pope embeddings'

        self.attend = Attend(flash = flash, dropout = dropout)

        self.to_qkv = Linear(dim, dim_inner * 3, bias = False)

        self.to_value_residual_mix = Linear(dim, heads) if learned_value_residual_mix else None

        self.to_gates = Linear(dim, heads)

        self.to_out = nn.Sequential(
            Linear(dim_inner, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(self, x, value_residual = None):
        q, k, v = rearrange(self.to_qkv(x), 'b n (qkv h d) -> qkv b h n d', qkv = 3, h = self.heads)

        orig_v = v

        if exists(self.to_value_residual_mix):
            mix = self.to_value_residual_mix(x)
            mix = rearrange(mix, 'b n h -> b h n 1').sigmoid()

            assert exists(value_residual)
            v = v.lerp(value_residual, mix)

        if exists(self.pope_embed):
            out = flash_attn_with_pope(q, k, v, pos_emb = self.pope_embed(q.shape[-2]), softmax_scale = self.scale)
        elif exists(self.rotary_embed):
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)
            out = self.attend(q, k, v)
        else:
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
        pope_embed = None,
        flash_attn = True,
        add_value_residual = False,
        num_residual_streams = 1,
        num_residual_fracs = 1,
        mc_hyper_conn_sinkhorn_iters = 2,
        time_cond_dim = None
    ):
        super().__init__()
        self.layers = ModuleList([])

        init_hyper_conn, *_ = mc_get_init_and_expand_reduce_stream_functions(num_residual_streams, num_fracs = num_residual_fracs, sinkhorn_iters = mc_hyper_conn_sinkhorn_iters)

        for _ in range(depth):
            self.layers.append(ModuleList([
                init_hyper_conn(dim = dim, branch = AdaLN(dim, time_cond_dim, Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_embed = rotary_embed, pope_embed = pope_embed, flash = flash_attn, learned_value_residual_mix = add_value_residual))),
                init_hyper_conn(dim = dim, branch = AdaLN(dim, time_cond_dim, FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)))
            ]))

        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x, value_residual = None, time_cond = None):

        first_values = None

        for attn, ff in self.layers:
            x, next_values = attn(x, value_residual = value_residual, time_cond = time_cond)

            first_values = default(first_values, next_values)

            x = ff(x, time_cond = time_cond)

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
                Linear(dim_in, dim)
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

        net.append(Linear(layer_dim_in, layer_dim_out))

        if is_last:
            continue

        net.append(activation())

    return nn.Sequential(*net)

class STFTEstimator(Module):
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

        return cat(outs, dim = -1)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        freq_seq = torch.arange(half_dim)
        inv_freq = (freq_seq * -(math.log(theta) / (half_dim - 1))).exp()
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        emb = einx.multiply('b, d -> b d', x, self.inv_freq)
        return cat((emb.sin(), emb.cos()), dim = -1)

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

class FlowBSRoformer(Module):

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
        freq_range: tuple[int, int] | None = None, # specifying a frequency range, with (<min freq>, <max freq). `-1` implies 0 and inf
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        flash_attn = True,
        num_residual_streams = 4, # set to 1. to disable hyper connections
        num_residual_fracs = 1,   # can be used as an alternative to residual streams for memory efficiency while retaining benefits of hyper connections
        mc_hyper_conn_sinkhorn_iters = 2,
        stft_n_fft = 2048,
        stft_hop_length = 512, # 10ms at 44100Hz, from sections 4.1, 4.4 in the paper - @faroit recommends // 2 or // 4 for better reconstruction
        stft_win_length = 2048,
        stft_normalized = False,
        zero_dc = False, # @firebirdblue23 in https://github.com/lucidrains/BS-RoFormer/issues/47
        stft_window_fn: Callable | None = None,
        stft_estimator_depth = 2,
        multi_stft_resolution_loss_weight = 1.,
        multi_stft_resolutions_window_sizes: tuple[int, ...] = (4096, 2048, 1024, 512, 256),
        multi_stft_hop_size = 147,
        multi_stft_normalized = False,
        multi_stft_window_fn: Callable = torch.hann_window,
        use_pope = False,
        noise_std = 0.1,
        noise_mean = 0.0,
        time_to_loss_weight_fn: Callable | None = None
    ):
        super().__init__()

        self.noise_std = noise_std
        self.noise_mean = noise_mean

        self.time_to_loss_weight_fn = time_to_loss_weight_fn

        self.time_cond_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            Linear(dim, self.time_cond_dim),
            nn.GELU(),
            Linear(self.time_cond_dim, self.time_cond_dim)
        )

        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems
        self.stft_channels = self.audio_channels * (1 + self.num_stems)

        _, self.expand_stream, self.reduce_stream = mc_get_init_and_expand_reduce_stream_functions(num_residual_streams, disable = num_residual_streams == 1)

        self.layers = ModuleList([])

        transformer_kwargs = dict(
            dim = dim,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            flash_attn = flash_attn,
            num_residual_streams = num_residual_streams,
            num_residual_fracs = num_residual_fracs,
            mc_hyper_conn_sinkhorn_iters = mc_hyper_conn_sinkhorn_iters,
            time_cond_dim = self.time_cond_dim,
            norm_output = False
        )

        if use_pope:
            time_pope_embed = PoPE(dim = dim_head, heads = heads)
            freq_pope_embed = PoPE(dim = dim_head, heads = heads)
            time_rotary_embed = freq_rotary_embed = None
        else:
            time_rotary_embed = RotaryEmbedding(dim = dim_head)
            freq_rotary_embed = RotaryEmbedding(dim = dim_head)
            time_pope_embed = freq_pope_embed = None

        for layer_index in range(depth):
            is_first = layer_index == 0

            self.layers.append(nn.ModuleList([
                Transformer(depth = time_transformer_depth, rotary_embed = time_rotary_embed, pope_embed = time_pope_embed, add_value_residual = not is_first, **transformer_kwargs),
                Transformer(depth = freq_transformer_depth, rotary_embed = freq_rotary_embed, pope_embed = freq_pope_embed, add_value_residual = not is_first, **transformer_kwargs)
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

        # enforcing a frequency range

        freq_range = default(freq_range, (-1, -1))
        min_freq, max_freq = freq_range

        min_freq = 0 if min_freq == -1 else min_freq
        max_freq = freqs if max_freq == -1 else max_freq

        assert min_freq >= 0 and max_freq <= freqs and min_freq < max_freq

        self.min_freq = min_freq
        self.max_freq = max_freq

        self.freq_slice = slice(min_freq, max_freq)  # for slicing out the frequency for training
        self.freq_pad = (min_freq, freqs - max_freq) # for reconstruction during istft

        freqs = max_freq - min_freq

        # some validation on freqs

        assert len(freqs_per_bands) > 1
        assert sum(freqs_per_bands) == freqs, f'the number of freqs in the bands must equal {freqs} based on the STFT settings, but got {sum(freqs_per_bands)}'

        freqs_per_bands_with_complex = tuple(2 * f * self.audio_channels for f in freqs_per_bands)
        freqs_per_bands_with_complex_input = tuple(2 * f * self.stft_channels for f in freqs_per_bands)

        self.band_split = BandSplit(
            dim = dim,
            dim_inputs = freqs_per_bands_with_complex_input
        )

        self.stft_estimators = nn.ModuleList([])

        for _ in range(num_stems):
            stft_estimator = STFTEstimator(
            dim = dim,
                dim_inputs = freqs_per_bands_with_complex,
                depth = stft_estimator_depth
            )

            self.stft_estimators.append(stft_estimator)

        # whether to zero out dc

        self.zero_dc = min_freq == 0 and zero_dc

        # for the multi-resolution stft loss

        self.multi_stft_resolution_loss_weight = multi_stft_resolution_loss_weight
        self.multi_stft_resolutions_window_sizes = multi_stft_resolutions_window_sizes
        self.multi_stft_n_fft = stft_n_fft
        self.multi_stft_window_fn = multi_stft_window_fn

        self.multi_stft_kwargs = dict(
            hop_length = multi_stft_hop_size,
            normalized = multi_stft_normalized
        )

        self.register_buffer('zero', tensor(0.), persistent = False)

    @torch.no_grad()
    def sample(
        self,
        raw_audio,
        steps = 10
    ):
        self.eval()
        device, batch = raw_audio.device, raw_audio.shape[0]

        num_stems = len(self.stft_estimators)
        channels = self.audio_channels
        time_len = raw_audio.shape[-1]

        # initial noise

        stems_dim = (num_stems,) if num_stems > 1 else tuple()

        noise = torch.randn((batch, *stems_dim, channels, time_len), device = device)
        noise = noise * self.noise_std + self.noise_mean

        audio = noise

        # times

        times = torch.linspace(0., 1., steps + 1, device = device)
        delta = 1. / steps

        # denoising loop

        for time in times[:-1].unbind():
            time_batch = torch.full((batch,), time.item(), device = device)

            # predict clean target

            pred_target = self.forward(
                raw_audio,
                noised_target = audio,
                times = time_batch
            )

            time = pad_right_ndim_to(time, audio.ndim)

            # flow matching

            pred_flow = (pred_target - audio) / (1. - time)

            # euler step

            audio = audio + pred_flow * delta

        return audio

    def forward(
        self,
        raw_audio,
        target = None,
        noised_target = None,
        times = None,
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

        if exists(target) and target.ndim == 2:
            target = rearrange(target, 'b t -> b 1 t')

        batch = raw_audio.shape[0]

        # flow matching training logic

        if exists(target) and not exists(noised_target):
            if not exists(times):
                times = torch.rand((batch,), device = device)

            noise = torch.randn_like(target) * self.noise_std + self.noise_mean

            t = pad_right_ndim_to(times, target.ndim)

            noised_target = noise.lerp(target, t)

        assert exists(noised_target), 'noised_target must be passed in directly or generated from target'

        # shape formatting

        if noised_target.ndim == 2:
            noised_target = rearrange(noised_target, 'b t -> b 1 t')
        elif noised_target.ndim == 4:
            noised_target = rearrange(noised_target, 'b n s t -> b (n s) t')

        audio_input = cat((raw_audio, noised_target), dim = 1)

        channels = raw_audio.shape[1]
        assert (not self.stereo and channels == 1) or (self.stereo and channels == 2), 'stereo needs to be set to True if passing in audio signal that is stereo (channel dimension of 2). also need to be False if mono (channel dimension of 1)'

        # to stft

        audio_input, unpack_audio = pack_with_inverse(audio_input, '* t')

        stft_window = self.stft_window_fn(device = device)

        stft_repr = torch.stft(audio_input, **self.stft_kwargs, window = stft_window, return_complex = True)
        stft_repr = torch.view_as_real(stft_repr)

        stft_repr = unpack_audio(stft_repr, '* f t c')

        stft_repr = stft_repr[:, :, self.freq_slice] # slice out frequency range

        stft_repr = rearrange(stft_repr, 'b s f t c -> b (f s) t c') # merge stereo / mono into the frequency, with frequency leading dimension, for band splitting

        x = rearrange(stft_repr, 'b f t c -> b t (f c)')

        x = self.band_split(x)

        # time condition

        time_cond = None
        if exists(times):
            time_cond = self.time_mlp(times)

        # value residuals

        time_v_residual = None
        freq_v_residual = None

        # maybe expand residual streams

        x = self.expand_stream(x)

        # axial / hierarchical attention

        for time_transformer, freq_transformer in self.layers:

            x = rearrange(x, 'b t f d -> b f t d')
            x, unpack_time = pack_with_inverse(x, '* t d')

            x, next_time_v_residual = time_transformer(x, value_residual = time_v_residual, time_cond = time_cond)

            time_v_residual = default(time_v_residual, next_time_v_residual)

            x = unpack_time(x, '* t d')
            x = rearrange(x, 'b f t d -> b t f d')
            x, unpack_freq = pack_with_inverse(x, '* f d')

            x, next_freq_v_residual = freq_transformer(x, value_residual = freq_v_residual, time_cond = time_cond)

            freq_v_residual = default(freq_v_residual, next_freq_v_residual)

            x = unpack_freq(x, '* f d')

        # maybe reduce residual streams

        x = self.reduce_stream(x)

        x = self.final_norm(x)

        num_stems = len(self.stft_estimators)

        pred_stft = torch.stack([fn(x) for fn in self.stft_estimators], dim = 1)
        pred_stft = rearrange(pred_stft, 'b n t (f c) -> b n f t c', c = 2)

        # complex number

        pred_stft = torch.view_as_complex(pred_stft)

        # istft

        pred_stft = rearrange(pred_stft, 'b n (f s) t -> (b n s) f t', s = self.audio_channels)

        pred_stft = F.pad(pred_stft, (0, 0, *self.freq_pad))

        if self.zero_dc:
            # whether to dc filter
            pred_stft = pred_stft.index_fill(1, tensor(0, device = device), 0.)

        audio_length = raw_audio.shape[-1]
        recon_audio = torch.istft(pred_stft, **self.stft_kwargs, window = stft_window, return_complex = False, length = audio_length)

        recon_audio = rearrange(recon_audio, '(b n s) t -> b n s t', s = self.audio_channels, n = num_stems)

        if num_stems == 1:
            recon_audio = rearrange(recon_audio, 'b 1 s t -> b s t')

        # if a target is passed in, calculate loss for learning

        if not exists(target):
            return recon_audio

        if self.num_stems > 1:
            assert target.ndim == 4 and target.shape[1] == self.num_stems

        target = target[..., :recon_audio.shape[-1]] # protect against lost length on istft

        # per example losses

        loss = F.l1_loss(recon_audio, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        multi_stft_resolution_loss = self.zero

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

            res_loss = F.l1_loss(recon_Y, target_Y, reduction = 'none')
            res_loss = reduce(res_loss, '(b r) ... -> b', 'mean', b = recon_audio.shape[0])

            multi_stft_resolution_loss = multi_stft_resolution_loss + res_loss

        weighted_multi_resolution_loss = multi_stft_resolution_loss * self.multi_stft_resolution_loss_weight

        total_loss = loss + weighted_multi_resolution_loss

        if exists(self.time_to_loss_weight_fn):
            loss_weight = self.time_to_loss_weight_fn(times)
            total_loss = total_loss * loss_weight

        if not return_loss_breakdown:
            return total_loss.mean()

        return total_loss.mean(), Losses(loss.mean(), multi_stft_resolution_loss.mean())
