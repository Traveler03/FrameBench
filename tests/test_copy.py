import torch
import pytest
from torch.utils.cpp_extension import load

ext = load(name="copy_ext", sources=["ops/copy_kernel.cu"], verbose=False)

@pytest.mark.parametrize("N", [64, 72])
def test_copy_no_ce(N):
    cpu_in = torch.arange(N, dtype=torch.int32)
    gpu_out = torch.empty(N, dtype=torch.int32, device="cuda")

    ext.copy_to_gpu_no_ce(cpu_in, gpu_out)

    assert torch.equal(gpu_out.cpu(), cpu_in)

if __name__ == "__main__":
    pytest.main([__file__, "-q"]) 