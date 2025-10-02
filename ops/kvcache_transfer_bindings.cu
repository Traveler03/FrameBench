#include <torch/extension.h>

void transfer_kv_per_layer(const at::Tensor src_k, at::Tensor dst_k, const at::Tensor src_v, at::Tensor dst_v, const at::Tensor src_indices, const at::Tensor dst_indices, int64_t item_size, int64_t block_quota, int64_t num_warps_per_block);
void transfer_kv_per_layer_pf_lf(const at::Tensor src_k, at::Tensor dst_k, const at::Tensor src_v, at::Tensor dst_v, const at::Tensor src_indices, const at::Tensor dst_indices, int64_t layer_id, int64_t item_size, int64_t src_layout_dim, int64_t block_quota, int64_t num_warps_per_block);
void transfer_kv_all_layer(const at::Tensor src_k_layers, const at::Tensor dst_k_layers, const at::Tensor src_v_layers, const at::Tensor dst_v_layers, const at::Tensor src_indices, const at::Tensor dst_indices, int64_t item_size, int64_t num_layers, int64_t block_quota, int64_t num_warps_per_block);
void transfer_kv_all_layer_lf_pf(const at::Tensor src_k_layers, at::Tensor dst_k, const at::Tensor src_v_layers, at::Tensor dst_v, const at::Tensor src_indices, const at::Tensor dst_indices, int64_t item_size, int64_t dst_layout_dim, int64_t num_layers, int64_t block_quota, int64_t num_warps_per_block);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("transfer_kv_per_layer", &transfer_kv_per_layer);
  m.def("transfer_kv_per_layer_pf_lf", &transfer_kv_per_layer_pf_lf);
  m.def("transfer_kv_all_layer", &transfer_kv_all_layer);
  m.def("transfer_kv_all_layer_lf_pf", &transfer_kv_all_layer_lf_pf);
} 