"""TP-aware sharding rules for the shared 125M mini/MCore GPT contract.

The source of truth is an unsharded mini-layout tensor.  Both frameworks get a
deterministic shard of that source, so a TP benchmark never compares unrelated
random initializations or different QKV partition conventions.
"""
try:
    from .fair_tp1_contract import is_qkv_pair, qkv_mini_to_mcore
except ImportError:  # Direct script execution from experiments/.
    from fair_tp1_contract import is_qkv_pair, qkv_mini_to_mcore
from parallel.tensor_parallel import strided_partition_rows


def is_column_parallel(mini_name):
    return (mini_name.endswith("token_embedding.weight")
            or mini_name.endswith("attn.qkv.weight")
            or mini_name.endswith("mlp.fc1.weight")
            or mini_name == "lm_head.weight")


def is_row_parallel(mini_name):
    return mini_name.endswith("attn.o.weight") or mini_name.endswith("mlp.fc2.weight")


def mini_tp_shard(full_tensor, mini_name, tp_size, tp_rank):
    """Return the mini shard of an unsharded mini-layout parameter."""
    if mini_name.endswith("attn.qkv.weight"):
        return strided_partition_rows(full_tensor, tp_size, tp_rank, stride=3)
    if is_column_parallel(mini_name):
        return strided_partition_rows(full_tensor, tp_size, tp_rank)
    if is_row_parallel(mini_name):
        width = full_tensor.size(1) // tp_size
        return full_tensor.narrow(1, tp_rank * width, width).contiguous()
    return full_tensor


def mcore_tp_shard(full_mini_tensor, mini_name, mcore_name, model_config, tp_size, tp_rank):
    """Return MCore's TP shard, including its distinct QKV row layout."""
    if is_qkv_pair(mini_name, mcore_name):
        mcore_full = qkv_mini_to_mcore(full_mini_tensor, model_config)
        # MCore first interleaves Q/K/V within each head, then its standard
        # ColumnParallelLinear (stride=1) gives each TP rank a *contiguous*
        # block of complete heads.  Applying a second stride=3 here would mix
        # head blocks and was the source of the rejected r02 parity gate.
        rows = mcore_full.size(0) // tp_size
        return mcore_full.narrow(0, tp_rank * rows, rows).contiguous()
    return mini_tp_shard(full_mini_tensor, mini_name, tp_size, tp_rank)
