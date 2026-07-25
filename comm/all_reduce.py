import torch
import torch.distributed as dist


class AllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, group):
        ctx.group = group
        output = tensor.clone()
        if group is not None and dist.get_world_size(group=group) > 1:
            dist.all_reduce(output, group=group)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group
        if group is not None and dist.get_world_size(group=group) > 1:
            dist.all_reduce(grad_output, group=group)
        return grad_output, None


def all_reduce(tensor, group):
    return AllReduce.apply(tensor, group)
