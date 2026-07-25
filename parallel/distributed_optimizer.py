"""ZeRO-1 Distributed Optimizer: split optimizer states across DP ranks."""

import torch
import torch.distributed as dist


class DistributedOptimizer:
    """Each DP rank maintains optimizer states for its partition of params."""

    def __init__(self, param_groups, dp_group, optimizer_cls=torch.optim.AdamW, **optim_kwargs):
        self.dp_group = dp_group
        self.dp_size = dist.get_world_size(dp_group) if dp_group else 1
        self.dp_rank = dist.get_rank(dp_group) if dp_group else 0

        all_params = []
        for group in param_groups:
            all_params.extend(group["params"])
        self.all_params = all_params

        self.owned_params = []
        for i, p in enumerate(all_params):
            if i % self.dp_size == self.dp_rank:
                self.owned_params.append(p)

        # Per-rank optimizer (only sees owned params)
        self.optimizer = optimizer_cls(
            [{"params": self.owned_params}],
            **optim_kwargs
        )

    def zero_grad(self):
        for p in self.all_params:
            if p.grad is not None:
                p.grad = None

    def step(self):
        self.optimizer.step()

    def sync_params(self):
        """Broadcast updated owned params to all ranks."""
        if self.dp_size <= 1:
            return
        for i, p in enumerate(self.all_params):
            dist.broadcast(p.data, src=i % self.dp_size, group=self.dp_group)

    @property
    def param_groups(self):
        return self.optimizer.param_groups
