import torch
import torch.nn as nn
import torch.nn.functional as F
from comm.all_reduce import all_reduce
from parallel.process_groups import get_model_parallel


def _normal_init(tensor, std=0.02):
    nn.init.normal_(tensor, std=std)


class ColumnParallelLinear(nn.Module):
    """Linear layer with column-wise tensor parallelism (split output dim)."""

    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        mpu = get_model_parallel()
        tp_size = mpu["tp_size"] if mpu else 1
        self.out_features_per_partition = out_features // tp_size
        self.weight = nn.Parameter(torch.empty(self.out_features_per_partition, in_features))
        _normal_init(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_partition))
            nn.init.zeros_(self.bias)
        else:
            self.bias = None

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """Linear layer with row-wise tensor parallelism (split input dim)."""

    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        mpu = get_model_parallel()
        tp_size = mpu["tp_size"] if mpu else 1
        self.in_features_per_partition = in_features // tp_size
        self.weight = nn.Parameter(torch.empty(out_features, self.in_features_per_partition))
        _normal_init(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.bias = None

    def forward(self, x):
        mpu = get_model_parallel()
        output = F.linear(x, self.weight, self.bias)
        return all_reduce(output, mpu["tp_group"] if mpu else None)
