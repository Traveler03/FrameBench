#include <torch/extension.h>

void reconstruct_indices_from_tree_mask(
    at::Tensor tree_mask,
    at::Tensor verified_seq_len,
    at::Tensor positions,
    at::Tensor retrive_index,
    at::Tensor retrive_next_token,
    at::Tensor retrive_next_sibling,
    int64_t batch_size,
    int64_t draft_token_num);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("reconstruct_indices_from_tree_mask", &reconstruct_indices_from_tree_mask);
} 