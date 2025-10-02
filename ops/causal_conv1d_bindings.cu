#include <torch/extension.h>

void causal_conv1d_fwd(const at::Tensor &x, const at::Tensor &weight,
                  const std::optional<at::Tensor> &bias_,
                  const std::optional<at::Tensor> &conv_states,
                  const std::optional<at::Tensor> &query_start_loc,
                  const std::optional<at::Tensor> &cache_indices,
                  const std::optional<at::Tensor> &has_initial_state,
                  bool silu_activation,
                  int64_t pad_slot_id);

void causal_conv1d_update(const at::Tensor &x,
                     const at::Tensor &conv_state,
                     const at::Tensor &weight,
                     const std::optional<at::Tensor> &bias_,
                     bool silu_activation,
                     const std::optional<at::Tensor> &cache_seqlens_,
                     const std::optional<at::Tensor> &conv_state_indices_,
                     int64_t pad_slot_id);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("causal_conv1d_fwd", &causal_conv1d_fwd);
  m.def("causal_conv1d_update", &causal_conv1d_update);
} 