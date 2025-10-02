import os
import torch
import pytest
from torch.utils.cpp_extension import load

ops_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../ops"))

ext = load(
    name="vertical_slash_index_ext",
    sources=[
        os.path.join(ops_dir, "vertical_slash_index.cu"),
        os.path.join(ops_dir, "vertical_slash_index_bindings.cu"),
    ],
    verbose=False,
)


def build_simple_case(B=1, H=2, ctx=16, block_m=4, block_n=4):
    device = "cuda"
    q_seqlens = torch.tensor([ctx], dtype=torch.int32, device=device)
    kv_seqlens = torch.tensor([ctx], dtype=torch.int32, device=device)

    # simple pattern: vertical at columns [0, 4, 8, 12], slash at position 8
    NNZ_V = 4
    NNZ_S = 1
    vertical = torch.tensor([[[0, 4, 8, 12], [0, 4, 8, 12]]], dtype=torch.int32, device=device)
    slash = torch.tensor([[[8], [8]]], dtype=torch.int32, device=device)

    num_rows = (ctx + block_m - 1)//block_m
    block_count = torch.zeros(B, H, num_rows, dtype=torch.int32, device=device)
    block_offset = torch.zeros(B, H, num_rows, NNZ_S, dtype=torch.int32, device=device)
    column_count = torch.zeros(B, H, num_rows, dtype=torch.int32, device=device)
    column_index = torch.zeros(B, H, num_rows, NNZ_V, dtype=torch.int32, device=device)
    return (block_count, block_offset, column_count, column_index, q_seqlens, kv_seqlens, vertical, slash, ctx, block_m, block_n)


@pytest.mark.parametrize("ctx,block_m,block_n", [(16, 4, 4), (32, 8, 8)])
def test_convert_vertical_slash_indexes_smoke(ctx, block_m, block_n):
    (bc, bo, cc, ci, ql, kl, vi, si, ctx, bm, bn) = build_simple_case(ctx=ctx, block_m=block_m, block_n=block_n)
    ext.convert_vertical_slash_indexes(bc, bo, cc, ci, ql, kl, vi, si, ctx, bm, bn, True)
    # Just basic invariants
    assert (bc >= 0).all()
    assert (cc >= 0).all()


def test_convert_vertical_slash_indexes_mergehead_smoke():
    (bc, bo, cc, ci, ql, kl, vi, si, ctx, bm, bn) = build_simple_case()
    H = vi.size(1)
    per_head_v = torch.tensor([4]*H, dtype=torch.int32, device=vi.device)
    per_head_s = torch.tensor([1]*H, dtype=torch.int32, device=vi.device)
    ext.convert_vertical_slash_indexes_mergehead(bc, bo, cc, ci, ql, kl, vi, si, per_head_v, per_head_s, ctx, bm, bn, True)
    assert (bc >= 0).all()
    assert (cc >= 0).all()

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"]) 