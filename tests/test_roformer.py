import torch
import pytest
from bs_roformer import BSRoformer, MelBandRoformer
from PoPE_pytorch import PoPE

@pytest.mark.parametrize('use_pope', [True, False])
def test_bs_roformer(use_pope):
    model = BSRoformer(
        dim = 512,
        depth = 1,
        time_transformer_depth = 1,
        freq_transformer_depth = 1,
        use_pope = use_pope
    )

    inp = torch.randn(1, 1, 35280)
    out = model(inp)

@pytest.mark.parametrize('use_pope', [True, False])
def test_mel_band_roformer(use_pope):
    model = MelBandRoformer(
        dim = 512,
        depth = 1,
        time_transformer_depth = 1,
        freq_transformer_depth = 1,
        use_pope = use_pope
    )

    inp = torch.randn(1, 1, 35280)
    out = model(inp)
