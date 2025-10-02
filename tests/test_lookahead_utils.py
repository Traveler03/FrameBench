import os
import torch
import pytest
from torch.utils.cpp_extension import load

ops_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../ops"))
inc_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../third_party/sgl-kernel/include"))

ext = load(
    name="lookahead_utils_ext",
    sources=[
        os.path.join(ops_dir, "lookahead_utils.cu"),
        os.path.join(ops_dir, "lookahead_utils_bindings.cu"),
    ],
    extra_include_paths=[inc_dir],
    verbose=False,
)


def build_tree_mask(bs=2, draft=4):
    device = "cuda"
    tree_mask = torch.zeros(bs * draft * draft, dtype=torch.bool, device=device)
    verified_seq_len = torch.randint(0, 5, (bs,), dtype=torch.int64, device=device)
    positions = torch.empty(bs * draft, dtype=torch.int64, device=device)
    retrive_index = torch.empty(bs, draft, dtype=torch.int64, device=device)
    retrive_next_token = torch.full((bs, draft), -1, dtype=torch.int64, device=device)
    retrive_next_sibling = torch.full((bs, draft), -1, dtype=torch.int64, device=device)

    # 简单造一个链状树：每个样本的column 0->1->2->3
    for b in range(bs):
        base = b * draft * draft
        for i in range(draft):
            tree_mask[base + i * draft + i] = True
            if i + 1 < draft:
                tree_mask[base + (i+1) * draft + i] = True
    return tree_mask, verified_seq_len, positions, retrive_index, retrive_next_token, retrive_next_sibling


@pytest.mark.parametrize("bs,draft", [(2, 4), (1, 5)])
def test_reconstruct_indices_from_tree_mask(bs, draft):
    tree_mask, verified_seq_len, positions, retrive_index, retrive_next_token, retrive_next_sibling = build_tree_mask(bs, draft)
    ext.reconstruct_indices_from_tree_mask(tree_mask, verified_seq_len, positions, retrive_index, retrive_next_token, retrive_next_sibling, bs, draft)

    assert positions.shape[0] == bs * draft
    assert (retrive_index.view(-1) >= 0).all()
    assert (retrive_next_token.view(-1) >= -1).all()
    assert (retrive_next_sibling.view(-1) >= -1).all()

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"]) 