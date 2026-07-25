import torch
import torch.distributed as dist


def allreduce_grads(model_or_params, dp_group):
    if dp_group is None or dist.get_world_size(dp_group) <= 1:
        return
    if isinstance(model_or_params, list):
        params = model_or_params
    else:
        params = list(model_or_params.parameters())
    for param in params:
        if param.grad is not None:
            dist.all_reduce(param.grad, group=dp_group)
            param.grad.div_(dist.get_world_size(dp_group))
