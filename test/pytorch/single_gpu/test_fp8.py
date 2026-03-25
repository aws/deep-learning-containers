"""Validate FP8 forward pass via Transformer Engine on a single GPU."""

import pytest
import torch


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="FP8 requires Hopper (sm90) or newer",
)
def test_te_fp8_forward():
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe

    fp8_recipe = recipe.DelayedScaling()
    m = te.Linear(64, 64).cuda()
    x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        out = m(x)
    assert out.shape == (4, 64)
    assert torch.isfinite(out).all(), "FP8 output contains NaN/Inf"
