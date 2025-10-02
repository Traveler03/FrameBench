#include <torch/extension.h>

void merge_state_v2(
    at::Tensor v_a, at::Tensor s_a, at::Tensor v_b, at::Tensor s_b, at::Tensor v_merged, at::Tensor s_merged);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("merge_state_v2", &merge_state_v2);
} 