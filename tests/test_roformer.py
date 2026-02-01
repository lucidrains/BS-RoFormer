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

    dummy_audio = torch.randn(1, 1, 44100)
    out = model(dummy_audio)
    
    assert out.shape[0] == dummy_audio.shape[0]
    assert abs(out.shape[-1] - dummy_audio.shape[-1]) < 1024

    # verify pope presence
    has_pope = any(isinstance(m, PoPE) for m in model.modules())
    assert has_pope == use_pope

@pytest.mark.parametrize('use_pope', [True, False])
def test_mel_band_roformer(use_pope):
    model = MelBandRoformer(
        dim = 512,
        depth = 1,
        time_transformer_depth = 1,
        freq_transformer_depth = 1,
        use_pope = use_pope
    )

    dummy_audio = torch.randn(1, 1, 44100)
    out = model(dummy_audio)
    
    assert out.shape[0] == dummy_audio.shape[0]
    assert abs(out.shape[-1] - dummy_audio.shape[-1]) < 1024

    # verify pope presence
    has_pope = any(isinstance(m, PoPE) for m in model.modules())
    assert has_pope == use_pope
