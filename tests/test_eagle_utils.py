import os
import torch
import pytest
from torch.utils.cpp_extension import load

ops_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../ops"))
inc_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../third_party/sgl-kernel/include"))

ext = load(
    name="eagle_utils_ext",
    sources=[
        os.path.join(ops_dir, "eagle_utils.cu"),
        os.path.join(ops_dir, "eagle_utils_bindings.cu"),
    ],
    extra_include_paths=[inc_dir],
    verbose=False,
)


@pytest.mark.parametrize("bs,topk,depth,draft", [(2, 2, 3, 4)])
def test_build_and_verify_tree(bs, topk, depth, draft):
    device = "cuda"
    torch.manual_seed(0)
    # parent_list shape: [bs, topk*(depth-1)+1]
    parent_list = torch.randint(0, topk*(depth-1)+1, (bs, topk*(depth-1)+1), dtype=torch.int64, device=device)
    selected_index = torch.randint(0, topk*(depth-1)+1, (bs, draft-1), dtype=torch.int64, device=device)
    verified_seq_len = torch.randint(0, 5, (bs,), dtype=torch.int64, device=device)

    # masks/outputs
    tree_mask = torch.zeros(bs, draft, draft, dtype=torch.bool, device=device)
    positions = torch.empty(bs, draft, dtype=torch.int64, device=device)
    retrive_index = torch.empty(bs, draft, dtype=torch.int64, device=device)
    retrive_next_token = torch.full((bs, draft), -1, dtype=torch.int64, device=device)
    retrive_next_sibling = torch.full((bs, draft), -1, dtype=torch.int64, device=device)

    # build
    ext.build_tree_kernel_efficient(parent_list, selected_index, verified_seq_len, tree_mask, positions, retrive_index, retrive_next_token, retrive_next_sibling, topk, depth, draft, 0)

    # verify
    predicts = torch.empty(bs*draft, dtype=torch.int32, device=device)
    accept_index = torch.empty(bs, draft, dtype=torch.int32, device=device)
    accept_token_num = torch.empty(bs, dtype=torch.int32, device=device)
    candidates = torch.randint(0, draft, (bs, draft), dtype=torch.int64, device=device)
    target_predict = torch.randint(0, draft, (bs, draft), dtype=torch.int64, device=device)

    ext.verify_tree_greedy(predicts, accept_index, accept_token_num, candidates, retrive_index, retrive_next_token, retrive_next_sibling, target_predict, 0)

    assert (accept_index >= 0).all()
    assert (accept_token_num >= 0).all()

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"]) 