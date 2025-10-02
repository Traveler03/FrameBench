import os
import torch
import pytest
from torch.utils.cpp_extension import load

ops_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../ops"))
inc_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../third_party/sgl-kernel/include"))

ext = load(
    name="merge_attn_states_ext",
    sources=[
        os.path.join(ops_dir, "merge_attn_states.cu"),
        os.path.join(ops_dir, "merge_attn_states_bindings.cu"),
    ],
    extra_include_paths=[inc_dir],
    verbose=False,
)


def torch_ref_merge(v_a, s_a, v_b, s_b):
    # shapes: [n,h,d], [n,h] x2
    n,h,d = v_a.shape
    v_out = torch.empty_like(v_a)
    s_out = torch.empty_like(s_a, dtype=torch.float32)
    for i in range(n):
        for j in range(h):
            p_lse = s_a[i,j].float()
            s_lse = s_b[i,j].float()
            max_lse = torch.maximum(p_lse, s_lse)
            p_se = torch.exp(p_lse - max_lse)
            s_se = torch.exp(s_lse - max_lse)
            out_se = p_se + s_se
            p_scale = p_se / out_se
            s_scale = s_se / out_se
            v_out[i,j,:] = (v_a[i,j,:].float() * p_scale + v_b[i,j,:].float() * s_scale).to(v_a.dtype)
            s_out[i,j] = torch.log(out_se) + max_lse
    return v_out, s_out


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("n,h,d", [(4, 3, 16)])
def test_merge_attn_states(dtype, n, h, d):
    device = "cuda"
    torch.manual_seed(0)
    v_a = torch.randn(n,h,d, device=device, dtype=dtype)
    s_a = torch.randn(n,h, device=device, dtype=torch.float32)
    v_b = torch.randn(n,h,d, device=device, dtype=dtype)
    s_b = torch.randn(n,h, device=device, dtype=torch.float32)

    v_out = torch.empty_like(v_a)
    s_out = torch.empty_like(s_a)

    ext.merge_state_v2(v_a, s_a, v_b, s_b, v_out, s_out)

    v_ref, s_ref = torch_ref_merge(v_a, s_a, v_b, s_b)

    assert torch.allclose(v_out.float().cpu(), v_ref.float().cpu(), atol=3e-3, rtol=3e-3)
    assert torch.allclose(s_out.float().cpu(), s_ref.float().cpu(), atol=3e-4, rtol=3e-4)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"]) 