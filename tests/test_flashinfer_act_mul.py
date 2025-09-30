import torch
import pytest
from torch.utils.cpp_extension import load
from test_utils import benchmark
import os

# 加载CUDA扩展
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_THIS_DIR, "..", "ops", "flashinfer_act_mul.cu"))
ext = load(name="flashinfer_act_mul_ext", sources=[_SRC], verbose=False)

# 参考实现 (CPU)
def ref_act_and_mul(x):
    batch_size, total_dim = x.shape
    dim = total_dim // 2
    
    # 分割输入
    x_part = x[:, :dim]
    y_part = x[:, dim:]
    
    # 计算 relu(x) * y
    relu_x = torch.relu(x_part)
    return relu_x * y_part

@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_size,dim", [(1, 1024), (8, 256), (16, 4096)])
def test_act_and_mul_correctness_and_perf(dtype, batch_size, dim):
    if not torch.cuda.is_available():
        pytest.skip("需要CUDA GPU")
    
    # 生成随机输入数据
    torch.manual_seed(0)
    device = torch.device("cuda")
    x = torch.randn(batch_size, 2 * dim, device=device).to(dtype)
    
    # 调用CUDA实现
    out_cuda = ext.act_and_mul_cuda(x)
    
    # 调用参考实现
    out_ref = ref_act_and_mul(x.cpu()).to(device).to(dtype)
    
    # 检查结果正确性
    assert torch.allclose(out_cuda.float(), out_ref.float(), rtol=1e-2, atol=1e-2)
    
    # 性能测试
    t_ref = benchmark(lambda inp: ref_act_and_mul(inp.cpu()).to(device), 20, x)
    t_cuda = benchmark(ext.act_and_mul_cuda, 100, x)
    
    print(f"dtype={dtype}, batch_size={batch_size}, dim={dim} -> "
          f"ref: {t_ref:.4f} ms, cuda: {t_cuda:.4f} ms, speedup={t_ref/t_cuda:.2f}x")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"]) 