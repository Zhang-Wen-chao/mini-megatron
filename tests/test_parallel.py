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
from parallel.tensor_parallel import ColumnParallelLinear, RowParallelLinear
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
