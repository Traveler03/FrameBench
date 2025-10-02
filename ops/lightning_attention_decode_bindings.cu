#include <torch/extension.h>

void lightning_attention_decode(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& past_kv,
    const torch::Tensor& slope,
    torch::Tensor output,
    torch::Tensor new_kv);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("lightning_attention_decode", &lightning_attention_decode);
} 