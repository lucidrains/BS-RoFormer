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

@pytest.mark.parametrize('use_pope', [True, False])
@pytest.mark.parametrize('time_to_loss_weight_fn', [None, lambda t: t])
def test_flow_bs_roformer(use_pope, time_to_loss_weight_fn):
    from bs_roformer.flow_bs_roformer import FlowBSRoformer

    model = FlowBSRoformer(
        dim = 32,
        depth = 1,
        time_transformer_depth = 1,
        freq_transformer_depth = 1,
        use_pope = use_pope,
        time_to_loss_weight_fn = time_to_loss_weight_fn
    )

    inp = torch.randn(1, 35280)
    target = torch.randn(1, 35280)
    loss = model(inp, target=target)
    loss.backward()

    out = model.sample(inp)
    assert out.shape == (1, 1, 35280)
