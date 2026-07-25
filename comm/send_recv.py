import torch
import torch.distributed as dist


def send(tensor, dst, tag=0, group=None):
    dist.send(tensor, dst=dst, tag=tag, group=group)


def recv(tensor, src, tag=0, group=None):
    dist.recv(tensor, src=src, tag=tag, group=group)


def send_recv(tensor, dst, src, group=None, tag=0):
    if dst == src:
        return tensor
    if dist.get_rank() == dst:
        result = torch.empty_like(tensor)
        dist.recv(result, src=src, tag=tag)
        return result
    dist.send(tensor, dst=dst, tag=tag, group=group)
    return tensor.clone()
