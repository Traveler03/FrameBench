import os
import torch
import pytest
import torch.nn.functional as F
from torch.utils.cpp_extension import load

ops_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../ops"))
inc_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../third_party/sgl-kernel/csrc/mamba"))

ext = load(
    name="causal_conv1d_ext",
    sources=[
        os.path.join(ops_dir, "causal_conv1d.cu"),
        os.path.join(ops_dir, "causal_conv1d_bindings.cu"),
    ],
    extra_include_paths=[inc_dir],
    verbose=False,
)


def torch_ref_fwd(x, weight, bias=None, silu=False):
    # x: [B, C, L], weight: [C, W] depthwise causal conv per-channel
    B, C, L = x.shape
    W = weight.size(1)
    w = torch.zeros(C, 1, W, device=x.device, dtype=x.dtype)
    w[:,0,:] = weight
    y = F.conv1d(x, w, bias=bias, stride=1, padding=W-1, groups=C)
    y = y[:, :, :L]
    if silu:
        y = y * torch.sigmoid(y)
    return y


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("B,C,L,W", [(2, 4, 32, 3), (1, 2, 17, 4)])
def test_causal_conv1d_fwd(dtype, B, C, L, W):
    device = "cuda"
    torch.manual_seed(0)
    x = torch.randn(B, C, L, device=device, dtype=dtype)
    weight = torch.randn(C, W, device=device, dtype=dtype)
    bias = torch.randn(C, device=device, dtype=dtype)

    # use out=in-place per kernel design
    x_out = x.clone()
    ext.causal_conv1d_fwd(x_out, weight, bias, None, None, None, None, False, -1)

    y_ref = torch_ref_fwd(x, weight, bias, silu=False)

    assert x_out.shape == x.shape
    assert torch.allclose(x_out.float().cpu(), y_ref.float().cpu(), atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("B,C,L,W,S", [(2, 4, 8, 3, 6), (1, 3, 5, 2, 4)])
def test_causal_conv1d_update(dtype, B, C, L, W, S):
    device = "cuda"
    torch.manual_seed(0)
    x = torch.randn(B, C, L, device=device, dtype=dtype)
    weight = torch.randn(C, W, device=device, dtype=dtype)
    bias = torch.randn(C, device=device, dtype=dtype)

    # conv_state len >= W-1
    conv_state = torch.zeros(B, C, S, device=device, dtype=dtype)

    x_out = x.clone()
    ext.causal_conv1d_update(x_out, conv_state, weight, bias, False, None, None, -1)

    y_ref = torch_ref_fwd(x, weight, bias, silu=False)

    assert x_out.shape == x.shape
    assert torch.allclose(x_out.float().cpu(), y_ref.float().cpu(), atol=3e-2, rtol=3e-2)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"]) 