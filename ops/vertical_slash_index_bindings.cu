#include <torch/extension.h>

// Declarations of functions implemented in vertical_slash_index.cu
void convert_vertical_slash_indexes(
    torch::Tensor& block_count,
    torch::Tensor& block_offset,
    torch::Tensor& column_count,
    torch::Tensor& column_index,
    torch::Tensor q_seqlens,
    torch::Tensor kv_seqlens,
    torch::Tensor vertical_indexes,
    torch::Tensor slash_indexes,
    int64_t context_size,
    int64_t block_size_M,
    int64_t block_size_N,
    bool causal);

void convert_vertical_slash_indexes_mergehead(
    torch::Tensor& block_count,
    torch::Tensor& block_offset,
    torch::Tensor& column_count,
    torch::Tensor& column_index,
    torch::Tensor q_seqlens,
    torch::Tensor kv_seqlens,
    torch::Tensor vertical_indexes,
    torch::Tensor slash_indexes,
    torch::Tensor vertical_indices_count,
    torch::Tensor slash_indices_count,
    int64_t context_size,
    int64_t block_size_M,
    int64_t block_size_N,
    bool causal);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("convert_vertical_slash_indexes", &convert_vertical_slash_indexes);
  m.def("convert_vertical_slash_indexes_mergehead", &convert_vertical_slash_indexes_mergehead);
} 