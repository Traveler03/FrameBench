import os
import torch
import pytest
from torch.utils.cpp_extension import load

ops_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../ops"))
inc_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../third_party/sgl-kernel/include"))

ext = load(
    name="kvcache_transfer_ext",
    sources=[
        os.path.join(ops_dir, "kvcache_transfer.cu"),
        os.path.join(ops_dir, "kvcache_transfer_bindings.cu"),
    ],
    extra_include_paths=[inc_dir],
    verbose=False,
)


@pytest.mark.parametrize("items,item_bytes", [(64, 128)])
def test_transfer_kv_per_layer(items, item_bytes):
    device = "cuda"
    torch.manual_seed(0)
    src_k = torch.empty(items * item_bytes, dtype=torch.uint8, device=device)
    src_v = torch.empty(items * item_bytes, dtype=torch.uint8, device=device)
    dst_k = torch.empty_like(src_k)
    dst_v = torch.empty_like(src_v)
    src_indices = torch.arange(items, dtype=torch.int64, device=device)
    dst_indices = torch.arange(items, dtype=torch.int64, device=device)

    ext.transfer_kv_per_layer(src_k, dst_k, src_v, dst_v, src_indices, dst_indices, item_bytes, 8, 4)

    assert dst_k.numel() == src_k.numel()
    assert dst_v.numel() == src_v.numel()


@pytest.mark.parametrize("layers,items,item_bytes", [(3, 64, 64)])
def test_transfer_kv_all_layer(layers, items, item_bytes):
    device = "cuda"
    torch.manual_seed(0)
    src_k_layers = torch.randint(0, 2**16, (layers,), dtype=torch.uint64, device=device)
    dst_k_layers = torch.randint(0, 2**16, (layers,), dtype=torch.uint64, device=device)
    src_v_layers = torch.randint(0, 2**16, (layers,), dtype=torch.uint64, device=device)
    dst_v_layers = torch.randint(0, 2**16, (layers,), dtype=torch.uint64, device=device)
    src_indices = torch.arange(items, dtype=torch.int64, device=device)
    dst_indices = torch.arange(items, dtype=torch.int64, device=device)

    ext.transfer_kv_all_layer(src_k_layers, dst_k_layers, src_v_layers, dst_v_layers, src_indices, dst_indices, item_bytes, layers, 8, 2)


@pytest.mark.parametrize("layers,items,item_bytes,dst_layout", [(2, 32, 64, 512)])
def test_transfer_kv_all_layer_lf_pf(layers, items, item_bytes, dst_layout):
    device = "cuda"
    torch.manual_seed(0)
    src_k_layers = torch.randint(0, 2**16, (layers,), dtype=torch.uint64, device=device)
    dst_k = torch.empty(1, device=device, dtype=torch.uint8)  # placeholder
    src_v_layers = torch.randint(0, 2**16, (layers,), dtype=torch.uint64, device=device)
    dst_v = torch.empty(1, device=device, dtype=torch.uint8)  # placeholder
    src_indices = torch.arange(items, dtype=torch.int64, device=device)
    dst_indices = torch.arange(items, dtype=torch.int64, device=device)

    ext.transfer_kv_all_layer_lf_pf(src_k_layers, dst_k, src_v_layers, dst_v, src_indices, dst_indices, item_bytes, dst_layout, layers, 8, 2)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"]) 