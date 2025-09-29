import torch
import pytest
from torch.utils.cpp_extension import load
from test_utils import benchmark

ext = load(name="copy_ext", sources=["../ops/copy_kernel.cu"], verbose=False)

@pytest.mark.parametrize("N", [64, 72])
def test_copy_no_ce(N):
    cpu_in = torch.arange(N, dtype=torch.int32)
    gpu_out = torch.empty(N, dtype=torch.int32, device="cuda")

    ext.copy_to_gpu_no_ce(cpu_in, gpu_out)

    assert torch.equal(gpu_out.cpu(), cpu_in)
    
    t = benchmark(ext.copy_to_gpu_no_ce, 50, cpu_in, gpu_out)
    print(f"N={N}: {t:.4f} ms")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"]) 