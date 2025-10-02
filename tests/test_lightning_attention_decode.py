import os
import torch
import pytest
from torch.utils.cpp_extension import load

ops_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../ops"))

ext = load(
    name="lightning_attention_decode_ext",
    sources=[
        os.path.join(ops_dir, "lightning_attention_decode_kernel.cu"),
        os.path.join(ops_dir, "lightning_attention_decode_bindings.cu"),
    ],
    verbose=False,
)


def torch_ref(q, k, v, past_kv, slope):
    # q: [b,h,1,d], k: [b,h,1,d], v: [b,h,1,e], past_kv: [b,h,d,e], slope: [h,1,1]
    b, h, _, d = q.shape
    e = v.size(-1)
    out = torch.empty(b, h, 1, e, device=q.device, dtype=q.dtype)
    new_kv = past_kv.clone().float()
    for bi in range(b):
        for hi in range(h):
            ratio = torch.exp(-slope[hi,0,0])
            kv = ratio * past_kv[bi,hi,:,:].float() + torch.ger(k[bi,hi,0,:].float(), v[bi,hi,0,:].float())
            new_kv[bi,hi,:,:] = kv
            out[bi,hi,0,:] = (q[bi,hi,0,:].float().unsqueeze(0) @ kv).squeeze(0).to(q.dtype)
    return out, new_kv


def get_out_tol(dtype):
    if dtype == torch.bfloat16:
        return 3e-2, 3e-2
    if dtype == torch.float16:
        return 2e-3, 2e-3
    return 1e-4, 1e-4


def get_newkv_tol(dtype):
    if dtype == torch.bfloat16:
        return 5e-3, 5e-3
    return 2e-3, 2e-3


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("b,h,d,e", [(2, 3, 16, 8)])
def test_lightning_attention_decode_correctness(dtype, b, h, d, e):
    device = "cuda"
    torch.manual_seed(0)
    q = torch.randn(b,h,1,d, device=device, dtype=dtype)
    k = torch.randn(b,h,1,d, device=device, dtype=dtype)
    v = torch.randn(b,h,1,e, device=device, dtype=dtype)
    past_kv = torch.randn(b,h,d,e, device=device, dtype=torch.float32)
    slope = torch.randn(h,1,1, device=device, dtype=torch.float32)

    out = torch.empty(b,h,1,e, device=device, dtype=dtype)
    new_kv = torch.empty(b,h,d,e, device=device, dtype=torch.float32)

    out_ref, new_kv_ref = torch_ref(q,k,v,past_kv,slope)

    ext.lightning_attention_decode(q,k,v,past_kv,slope,out,new_kv)

    assert out.shape == (b,h,1,e)
    assert new_kv.shape == (b,h,d,e)

    atol_out, rtol_out = get_out_tol(dtype)
    assert torch.allclose(out.float().cpu(), out_ref.float().cpu(), atol=atol_out, rtol=rtol_out)

    atol_kv, rtol_kv = get_newkv_tol(dtype)
    assert torch.allclose(new_kv.float().cpu(), new_kv_ref.float().cpu(), atol=atol_kv, rtol=rtol_kv)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"]) 