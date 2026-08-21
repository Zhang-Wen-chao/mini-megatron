import torch
import torch.distributed as dist


class ReduceFromModelParallelRegion(torch.autograd.Function):
    """Sum TP partial outputs; pass their gradient through unchanged.

    Row-parallel linear computes one partial output per rank.  Its forward
    result must be summed, but the derivative of that sum with respect to each
    partial output is simply the upstream gradient—not another all-reduce.
    """
    @staticmethod
    def forward(ctx, tensor, group):
        ctx.group = group
        output = tensor.clone()
        if group is not None and dist.get_world_size(group=group) > 1:
            dist.all_reduce(output, group=group)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class CopyToModelParallelRegion(torch.autograd.Function):
    """Replicate a TP input in forward and sum its input gradient.

    Column-parallel linears consume the same activation on all tensor-parallel
    ranks.  Forward therefore has no communication, while backward combines
    the input-gradient contributions from every output shard.
    """

    @staticmethod
    def forward(ctx, tensor, group):
        ctx.group = group
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group
        if group is not None and dist.get_world_size(group=group) > 1:
            dist.all_reduce(grad_output, group=group)
        return grad_output, None


def reduce_from_model_parallel_region(tensor, group):
    return ReduceFromModelParallelRegion.apply(tensor, group)


def copy_to_model_parallel_region(tensor, group):
    return CopyToModelParallelRegion.apply(tensor, group)


class GatherFromModelParallelRegion(torch.autograd.Function):
    """Concatenate equal TP shards and route backward to the local slice."""

    @staticmethod
    def forward(ctx, tensor, group, dim):
        ctx.group = group
        ctx.dim = dim
        ctx.local_size = tensor.size(dim)
        if group is None or dist.get_world_size(group=group) == 1:
            return tensor
        pieces = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group=group))]
        dist.all_gather(pieces, tensor.contiguous(), group=group)
        return torch.cat(pieces, dim=dim)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.group is None or dist.get_world_size(group=ctx.group) == 1:
            return grad_output, None, None
        rank = dist.get_rank(group=ctx.group)
        return grad_output.narrow(ctx.dim, rank * ctx.local_size, ctx.local_size).contiguous(), None, None


def gather_from_model_parallel_region(tensor, group, dim=-1):
    return GatherFromModelParallelRegion.apply(tensor, group, dim)


# Compatibility alias for reference modules.  New TP code should spell the
# direction explicitly; it prevents accidental use of the wrong backward rule.
AllReduce = ReduceFromModelParallelRegion


def all_reduce(tensor, group):
    return reduce_from_model_parallel_region(tensor, group)
