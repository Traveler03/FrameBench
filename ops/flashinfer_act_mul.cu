#undef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_HALF2_OPERATORS__

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

// 定义ReLU激活函数
__forceinline__ __device__ float relu_activation(const float& x) {
  return x > 0.0f ? x : 0.0f;
}

// 定义act_and_mul_kernel
template <typename T>
__global__ void act_and_mul_kernel(T* __restrict__ out, const T* __restrict__ input, const int d) {
  const int64_t token_idx = blockIdx.x;
  const int64_t thread_idx = threadIdx.x;
  const int64_t stride = blockDim.x;
  const int64_t offset = token_idx * 2 * d;

  // 处理向量化部分
  for (int64_t idx = thread_idx; idx < d; idx += stride) {
    float x = static_cast<float>(input[offset + idx]);
    float y = static_cast<float>(input[offset + d + idx]);
    out[token_idx * d + idx] = static_cast<T>(relu_activation(x) * y);
  }
}

// 封装成PyTorch操作
// 输入格式: [batch_size, 2*dim]，前dim是x，后dim是y
// 输出格式: [batch_size, dim]，结果是relu(x) * y
at::Tensor act_and_mul_cuda(const at::Tensor& input) {
  TORCH_CHECK(input.is_cuda() && input.is_contiguous(), "input must be contiguous CUDA tensor");
  TORCH_CHECK(input.dim() == 2, "input must be [batch_size, 2*dim]");
  
  const int64_t batch_size = input.size(0);
  const int64_t total_dim = input.size(1);
  TORCH_CHECK(total_dim % 2 == 0, "input dim must be divisible by 2");
  
  const int64_t dim = total_dim / 2;
  at::Tensor output = at::empty({batch_size, dim}, input.options());
  
  const int threads = 256;
  const dim3 blocks(batch_size);
  
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  
  // 根据数据类型调度
  if (input.scalar_type() == at::kFloat) {
    act_and_mul_kernel<float><<<blocks, threads, 0, stream>>>(
      output.data_ptr<float>(),
      input.data_ptr<float>(),
      dim
    );
  } else if (input.scalar_type() == at::kHalf) {
    act_and_mul_kernel<at::Half><<<blocks, threads, 0, stream>>>(
      output.data_ptr<at::Half>(),
      input.data_ptr<at::Half>(),
      dim
    );
  } else if (input.scalar_type() == at::kBFloat16) {
    act_and_mul_kernel<at::BFloat16><<<blocks, threads, 0, stream>>>(
      output.data_ptr<at::BFloat16>(),
      input.data_ptr<at::BFloat16>(),
      dim
    );
  } else {
    TORCH_CHECK(false, "Unsupported data type");
  }
  
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("act_and_mul_cuda", &act_and_mul_cuda, "ReLU activation and multiply (CUDA)", 
        py::arg("input"));
} 