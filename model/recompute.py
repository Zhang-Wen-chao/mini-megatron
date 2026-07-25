import torch


def checkpoint(layer, x):
    """Activation recomputation: save input, recompute activations on backward."""
    return torch.utils.checkpoint.checkpoint(layer._forward, x, use_reentrant=False)
