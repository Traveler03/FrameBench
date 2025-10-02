#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_functions.h>

// Pure CUDA implementation: out = gelu(x[:,:D]) * x[:,D:]
// GELU uses erf-based definition: x * 0.5 * (1 + erf(x / sqrt(2)))
// input: [num_tokens, 2*D], output: [num_tokens, D]

namespace {

template <typename T>
__device__ __forceinline__ float to_float(T x);

template <>
__device__ __forceinline__ float to_float<float>(float x) { return x; }

template <>
__device__ __forceinline__ float to_float<__half>(__half x) { return __half2float(x); }


template <typename T>
__device__ __forceinline__ T from_float(float x);

template <>
__device__ __forceinline__ float from_float<float>(float x) { return x; }

template <>
__device__ __forceinline__ __half from_float<__half>(float x) { return __float2half(x); }


__device__ __forceinline__ float gelu_erf(float x) {
  // 0.5 * x * (1 + erf(x / sqrt(2)))
  const float inv_sqrt2 = 0.70710678118654752440f;
  return 0.5f * x * (1.0f + erff(x * inv_sqrt2));
}

template <typename T>
__global__ void gelu_and_mul_kernel(
    const T* __restrict__ input, // [tokens, 2*D]
    T* __restrict__ output,      // [tokens, D]
    int D) {
  const int token_idx = blockIdx.x;
  const int lane = threadIdx.x;

  const int input_stride = 2 * D;
  const T* in_ptr = input + token_idx * input_stride;
  T* out_ptr = output + token_idx * D;

  for (int i = lane; i < D; i += blockDim.x) {
    float a = to_float<T>(in_ptr[i]);        // left half
    float b = to_float<T>(in_ptr[D + i]);    // right half
    float gelu = gelu_erf(a);
    float c = gelu * b;
    out_ptr[i] = from_float<T>(c);
  }
}


// Host launcher
void gelu_and_mul(at::Tensor out, at::Tensor input) {
  TORCH_CHECK(out.is_cuda() && input.is_cuda(), "tensors must be CUDA");
  TORCH_CHECK(out.is_contiguous() && input.is_contiguous(), "tensors must be contiguous");
  TORCH_CHECK(input.dim() >= 2, "input must be at least 2-D: [tokens, 2*D]");
  TORCH_CHECK(out.dim() == input.dim(), "out dim must equal input dim");
  for (int i = 0; i < input.dim() - 1; ++i) {
    TORCH_CHECK(out.size(i) == input.size(i), "out shape mismatch before last dim");
  }
  TORCH_CHECK(input.size(-1) % 2 == 0, "last dim of input must be even (2*D)");
  TORCH_CHECK(out.size(-1) * 2 == input.size(-1), "out last dim must be input last dim / 2");

  const int64_t tokens = input.numel() / input.size(-1);
  const int D = static_cast<int>(out.size(-1));

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(tokens);
  dim3 block(static_cast<unsigned>(std::min<int64_t>(D, 256)));

  switch (input.scalar_type()) {
    case torch::kFloat32: {
      const float* in_ptr = input.data_ptr<float>();
      float* out_ptr = out.data_ptr<float>();
      gelu_and_mul_kernel<float><<<grid, block, 0, stream>>>(in_ptr, out_ptr, D);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    }
    case torch::kFloat16: {
      const __half* in_ptr = reinterpret_cast<const __half*>(input.data_ptr<at::Half>());
      __half* out_ptr = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
      gelu_and_mul_kernel<__half><<<grid, block, 0, stream>>>(in_ptr, out_ptr, D);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    }
    default:
      TORCH_CHECK(false, "gelu_and_mul only supports float32 and float16");
  }
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gelu_and_mul", &gelu_and_mul, "gelu_and_mul(out, input): out = gelu(input[:,:D]) * input[:,D:]");
} 