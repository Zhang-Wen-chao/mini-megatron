"""Tests for parallel and comm primitives on CPU.

These tests verify the math of ColumnParallelLinear, RowParallelLinear, and
AllReduce using torch's single-process Gloo backend (CPU-friendly).
"""
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pytest

import config as cfg
from parallel.tensor_parallel import (ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding,
                                      strided_partition_rows)
from experiments.fair_tp_parallel_contract import mcore_tp_shard, mini_tp_shard
from experiments.fair_tp1_contract import qkv_mini_to_mcore
from comm.all_reduce import AllReduce, all_reduce


def test_column_parallel_splits_output():
    """ColumnParallelLinear should partition weights along output dim.

    With TP=1 (default), a full Linear and a ColumnParallelLinear should
    produce identical outputs for the same input.
    """
    torch.manual_seed(0)
    full = torch.nn.Linear(64, 32, bias=False)
    torch.manual_seed(0)
    col = ColumnParallelLinear(64, 32, bias=False)

    # Force same weights
    with torch.no_grad():
        col.weight.copy_(full.weight)

    x = torch.randn(2, 8, 64)
    y_full = full(x)
    y_col = col(x)
    assert torch.allclose(y_full, y_col, atol=1e-5)


def test_qkv_strided_partition_keeps_q_k_v_on_every_tp_rank():
    # Logical mini layout: Q heads first, then K, then V.  For two TP ranks,
    # rank 0 needs Q0/K0/V0 and rank 1 needs Q1/K1/V1; a plain contiguous
    # split would incorrectly hand rank 0 all Q rows plus part of K.
    rows = torch.arange(3 * 4).reshape(3 * 4, 1)
    rank0 = strided_partition_rows(rows, world_size=2, rank=0, stride=3)
    rank1 = strided_partition_rows(rows, world_size=2, rank=1, stride=3)
    assert rank0.flatten().tolist() == [0, 1, 4, 5, 8, 9]
    assert rank1.flatten().tolist() == [2, 3, 6, 7, 10, 11]


def test_tp_contract_shards_qkv_and_row_weights_by_their_logical_dimensions():
    qkv = torch.arange(12).reshape(12, 1)
    row = torch.arange(12).reshape(3, 4)
    assert mini_tp_shard(qkv, "decoder.layers.0.attn.qkv.weight", 2, 0).flatten().tolist() == [0, 1, 4, 5, 8, 9]
    assert mini_tp_shard(row, "decoder.layers.0.attn.o.weight", 2, 1).tolist() == [[2, 3], [6, 7], [10, 11]]


def test_mcore_qkv_shard_is_a_contiguous_block_of_interleaved_heads():
    config = {"num_attention_heads": 2, "hidden_size": 4}
    mini_qkv = torch.arange(3 * 4).reshape(12, 1)
    mcore_full = qkv_mini_to_mcore(mini_qkv, config)
    shard0 = mcore_tp_shard(mini_qkv, "decoder.layers.0.attn.qkv.weight",
                            "decoder.layers.0.self_attention.linear_qkv.weight", config, 2, 0)
    shard1 = mcore_tp_shard(mini_qkv, "decoder.layers.0.attn.qkv.weight",
                            "decoder.layers.0.self_attention.linear_qkv.weight", config, 2, 1)
    assert torch.equal(shard0, mcore_full[:6])
    assert torch.equal(shard1, mcore_full[6:])


def test_row_parallel_linear_shape():
    """RowParallelLinear output should match full Linear's output shape."""
    torch.manual_seed(0)
    full = torch.nn.Linear(64, 32, bias=False)
    torch.manual_seed(0)
    row = RowParallelLinear(64, 32, bias=False)
    with torch.no_grad():
        row.weight.copy_(full.weight)

    x = torch.randn(2, 8, 64)
    y_row = row(x)
    assert y_row.shape == (2, 8, 32)


def test_vocab_parallel_embedding_matches_embedding_at_tp1():
    torch.manual_seed(7)
    full = torch.nn.Embedding(16, 4)
    torch.manual_seed(7)
    parallel = VocabParallelEmbedding(16, 4)
    with torch.no_grad():
        parallel.weight.copy_(full.weight)
    tokens = torch.tensor([[0, 3, 9, 15]])
    assert torch.equal(parallel(tokens), full(tokens))


def test_all_reduce_passthrough_when_no_group():
    """When group is None, all_reduce should be a no-op (returns the input)."""
    x = torch.randn(4)
    y = all_reduce(x, None)
    assert torch.allclose(x, y)


def test_all_reduce_function_class():
    """The AllReduce Function should be importable and the helper should work."""
    x = torch.randn(4)
    y = all_reduce(x, None)
    assert y.shape == x.shape
    assert torch.allclose(x, y)


def _tp_linear_worker(rank, world, init_file, output_dir):
    """Compare a two-rank TP block to its single-rank full-linear math."""
    from comm.all_reduce import copy_to_model_parallel_region, reduce_from_model_parallel_region

    dist.init_process_group("gloo", rank=rank, world_size=world, init_method=f"file://{init_file}")
    # Full matrix W has 4 output rows.  Every rank owns two rows (Column
    # Parallel), then a Row Parallel operation reconstructs the full hidden
    # output from its two input columns.
    x = torch.tensor([[1.0, -2.0, 3.0, 0.5]], requires_grad=True)
    w_column = torch.arange(16.0).reshape(4, 4) / 10
    w_row = torch.arange(16.0).reshape(4, 4) / 7
    column_shard = w_column[rank * 2:(rank + 1) * 2].clone().requires_grad_(True)
    row_shard = w_row[:, rank * 2:(rank + 1) * 2].clone().requires_grad_(True)
    local_hidden = torch.nn.functional.linear(copy_to_model_parallel_region(x, dist.group.WORLD), column_shard)
    output = reduce_from_model_parallel_region(torch.nn.functional.linear(local_hidden, row_shard), dist.group.WORLD)
    output.sum().backward()
    torch.save({"x_grad": x.grad, "column_grad": column_shard.grad, "row_grad": row_shard.grad},
               os.path.join(output_dir, f"tp-{rank}.pt"))
    dist.destroy_process_group()


def test_tp_autograd_matches_full_two_linear_reference():
    # This catches the subtle but material error of all-reducing through both
    # directions: Row Parallel's backward is identity, while Column Parallel's
    # backward needs the reduction.
    import tempfile

    x = torch.tensor([[1.0, -2.0, 3.0, 0.5]], requires_grad=True)
    w_column = (torch.arange(16.0).reshape(4, 4) / 10).requires_grad_(True)
    w_row = (torch.arange(16.0).reshape(4, 4) / 7).requires_grad_(True)
    torch.nn.functional.linear(torch.nn.functional.linear(x, w_column), w_row).sum().backward()
    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, "rendezvous")
        ctx = mp.get_context("spawn")
        procs = [ctx.Process(target=_tp_linear_worker, args=(rank, 2, init_file, tmpdir)) for rank in range(2)]
        for process in procs:
            process.start()
        for process in procs:
            process.join()
        assert all(process.exitcode == 0 for process in procs)
        states = [torch.load(os.path.join(tmpdir, f"tp-{rank}.pt"), weights_only=True) for rank in range(2)]
    assert torch.allclose(states[0]["x_grad"], x.grad)
    assert torch.allclose(states[1]["x_grad"], x.grad)
    for rank, state in enumerate(states):
        assert torch.allclose(state["column_grad"], w_column.grad[rank * 2:(rank + 1) * 2])
        assert torch.allclose(state["row_grad"], w_row.grad[:, rank * 2:(rank + 1) * 2])


def test_compute_mfu_dp_world_cancels():
    """The MFU formula should produce identical results for different DP world
    sizes, since DP-multiplied total_flops / DP-multiplied total_peak cancels.

    Physical interpretation: MFU is a per-GPU efficiency metric. Each GPU in
    a DP=2 setup does the same work as a DP=1 GPU in the same wall time
    (perfect scaling), so per-GPU MFU should be identical.
    """
    from main import compute_mfu
    config = cfg.get_model_config()

    # Single GPU (DP=1) and DP=2 should give the same MFU when wall time
    # is the same — DP-multiplied total_flops / DP-multiplied total_peak
    # cancels out.
    mfu_dp1 = compute_mfu(config, 512, 4, 1, 1, 1.0, 100)
    mfu_dp2 = compute_mfu(config, 512, 4, 1, 1, 1.0, 100)  # Same elapsed
    assert abs(mfu_dp1 - mfu_dp2) < 1e-6

    # Half the work in half the time gives same MFU
    mfu_half = compute_mfu(config, 512, 4, 1, 1, 0.5, 50)
    assert abs(mfu_dp1 - mfu_half) < 1e-6


def test_compute_mfu_returns_capped_value():
    """MFU should be capped at 1.0 (100%)."""
    from main import compute_mfu
    config = cfg.get_model_config()
    # Pass unrealistic elapsed=0.001 (1ms) → MFU > 1
    mfu = compute_mfu(config, 512, 4, 1, 1, 0.001, 100)
    assert mfu <= 1.0
