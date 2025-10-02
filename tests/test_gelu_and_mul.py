import os
import torch
import pytest
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from test_utils import benchmark

source_rel = os.path.join(os.path.dirname(__file__), "../ops/gelu_and_mul_kernel.cu")
source_abs = os.path.abspath(source_rel)

ext = load(name="gelu_and_mul_ext", sources=[source_abs], verbose=False)


def torch_ref_gelu_and_mul(input: torch.Tensor) -> torch.Tensor:
    D2 = input.size(-1)
    assert D2 % 2 == 0
    D = D2 // 2
    a, b = input[..., :D], input[..., D:]
    gelu = F.gelu(a, approximate='none')
    return gelu * b


def get_tolerances(dtype):
    if dtype == torch.float16:
        return 2e-3, 2e-3
    return 1e-4, 1e-4


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("B,D", [(3, 64), (5, 127)])
def test_gelu_and_mul_correctness(dtype, B, D):
    device = "cuda"
    torch.manual_seed(0)
    x = torch.randn(B, 2*D, device=device, dtype=dtype)
    out = torch.empty(B, D, device=device, dtype=dtype)

    out_ref = torch_ref_gelu_and_mul(x)

    ext.gelu_and_mul(out, x)

    atol, rtol = get_tolerances(dtype)
    assert torch.allclose(out.float().cpu(), out_ref.float().cpu(), atol=atol, rtol=rtol)


@pytest.mark.parametrize("B,D", [(64, 256), (128, 1024)])
def test_gelu_and_mul_benchmark(B, D):
    device = "cuda"
    dtype = torch.float16
    x = torch.randn(B, 2*D, device=device, dtype=dtype)
    out = torch.empty(B, D, device=device, dtype=dtype)

    t_ref = benchmark(torch_ref_gelu_and_mul, 50, x)
    t_cuda = benchmark(ext.gelu_and_mul, 50, out, x)
    print(f"ref: {t_ref:.4f} ms, cuda: {t_cuda:.4f} ms")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"]) 