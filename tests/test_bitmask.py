import torch
import pytest
from torch.utils.cpp_extension import load

ext = load(name="bitmask_ext", sources=["ops/bitmask_kernel.cu"], verbose=False)


def torch_ref_apply_bitmask_(logits: torch.Tensor, bitmask: torch.Tensor, indices: torch.Tensor | None = None):
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
        bitmask = bitmask.unsqueeze(0)
    B, V = logits.shape
    blocks = bitmask.shape[1]
    assert blocks * 32 >= V

    rows = range(B) if indices is None else indices.tolist()
    for bi, b in enumerate(rows):
        for blk in range(blocks):
            # Kernel uses bitmask_val = (~bitmask_word >> ...) & ...
            # So after inversion, bit==1 => mask to -inf
            mask_word = (~bitmask[bi, blk].item()) & 0xFFFFFFFF
            for i in range(32):
                v = blk * 32 + i
                if v >= V:
                    break
                if (mask_word >> i) & 1:
                    logits[b, v] = float("-inf")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_bitmask_basic(dtype):
    device = "cuda"
    B, V = 3, 100
    blocks = (V + 31) // 32
    torch.manual_seed(0)

    logits = torch.randn(B, V, device=device, dtype=dtype)
    bitmask = torch.randint(0, 2, (B, blocks), device=device, dtype=torch.int32)

    logits_ref = logits.clone()
    torch_ref_apply_bitmask_(logits_ref, bitmask)

    logits_out = logits.clone()
    ext.apply_token_bitmask_inplace_cuda(logits_out, bitmask)

    ref_cpu = logits_ref.float().cpu()
    out_cpu = logits_out.float().cpu()
    assert torch.allclose(out_cpu, ref_cpu, atol=0, rtol=0) 