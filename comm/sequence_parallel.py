"""Sequence Parallelism: split along sequence dimension, gather before attention."""

import torch
import torch.distributed as dist
from parallel.process_groups import get_model_parallel


def sp_all_gather(tensor, dim=-2):
    """All-gather across sequence dimension."""
    mpu = get_model_parallel()
    group = mpu["tp_group"] if mpu else None
    if group is None or dist.get_world_size(group) <= 1:
        return tensor
    world = dist.get_world_size(group)
    gathered = [torch.empty_like(tensor) for _ in range(world)]
    dist.all_gather(gathered, tensor, group=group)
    return torch.cat(gathered, dim=dim)


def sp_reduce_scatter(tensor, dim=-2):
    """Reduce-scatter across sequence dimension."""
    mpu = get_model_parallel()
    group = mpu["tp_group"] if mpu else None
    if group is None or dist.get_world_size(group) <= 1:
        return tensor
    world = dist.get_world_size(group)
    chunks = tensor.chunk(world, dim=dim)
    output = torch.empty_like(chunks[0])
    dist.reduce_scatter(output, list(chunks), group=group)
    return output


class _ReduceScatterBwdAllGather(torch.autograd.Function):
    """Forward: reduce-scatter, Backward: all-gather."""

    @staticmethod
    def forward(ctx, tensor, group, seq_dim):
        ctx.group = group
        ctx.seq_dim = seq_dim
        return sp_reduce_scatter(tensor, dim=seq_dim)

    @staticmethod
    def backward(ctx, grad_output):
        return sp_all_gather(grad_output, dim=ctx.seq_dim), None, None


def sp_all_reduce(tensor, group, seq_dim=-2):
    """SP reduce-scatter (f) + all-gather (b) instead of all-reduce."""
    mpu = get_model_parallel()
    group = mpu["tp_group"] if mpu else None
    if group is None or dist.get_world_size(group) <= 1:
        return tensor
    return _ReduceScatterBwdAllGather.apply(tensor, group, seq_dim)


def sp_all_gather_qkv(qkv, dim=-2):
    """All-gather K and V for SP attention."""
    q, k, v = qkv
    k = sp_all_gather(k, dim=dim)
    v = sp_all_gather(v, dim=dim)
    return q, k, v
