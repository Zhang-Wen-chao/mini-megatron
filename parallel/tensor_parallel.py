import torch
import torch.nn as nn
import torch.nn.functional as F
from comm.all_reduce import (copy_to_model_parallel_region, gather_from_model_parallel_region,
                             reduce_from_model_parallel_region)
from parallel.process_groups import get_model_parallel


def _normal_init(tensor, std=0.02):
    nn.init.normal_(tensor, std=std)


def strided_partition_rows(tensor, world_size, rank, stride=1):
    """Return one TP row shard using Megatron's stride convention.

    ``stride=1`` is ordinary contiguous output-row sharding.  QKV is special:
    mini stores its logical rows as ``[all Q | all K | all V]``.  Splitting it
    with ``stride=3`` gives every TP rank an equal set of Q, K and V heads,
    rather than giving one rank mostly Q rows and another mostly V rows.  This
    is the same logical partitioning used by Megatron-Core's strided QKV
    ColumnParallelLinear.
    """
    if tensor.dim() < 1:
        raise ValueError("tensor must have a row dimension")
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid TP rank/world size")
    if stride < 1 or tensor.size(0) % (world_size * stride):
        raise ValueError("rows must divide evenly by world_size * stride")
    rows_per_chunk = tensor.size(0) // (world_size * stride)
    chunks = tensor.split(rows_per_chunk, dim=0)
    return torch.cat(chunks[rank::world_size], dim=0)


class ColumnParallelLinear(nn.Module):
    """Linear layer with column-wise tensor parallelism (split output dim)."""

    def __init__(self, in_features, out_features, bias=False, stride=1, gather_output=False):
        super().__init__()
        mpu = get_model_parallel()
        tp_size = mpu["tp_size"] if mpu else 1
        if out_features % (tp_size * stride):
            raise ValueError("out_features must divide evenly by TP size * stride")
        self.out_features_per_partition = out_features // tp_size
        self.stride = stride
        self.gather_output = gather_output
        self.weight = nn.Parameter(torch.empty(self.out_features_per_partition, in_features))
        _normal_init(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_partition))
            nn.init.zeros_(self.bias)
        else:
            self.bias = None

    def forward(self, x):
        mpu = get_model_parallel()
        group = mpu["tp_group"] if mpu else None
        output = F.linear(copy_to_model_parallel_region(x, group), self.weight, self.bias)
        return gather_from_model_parallel_region(output, group) if self.gather_output else output


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
        return reduce_from_model_parallel_region(output, mpu["tp_group"] if mpu else None)


class VocabParallelEmbedding(nn.Module):
    """Vocabulary-row sharded embedding with a replicated hidden output."""

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        mpu = get_model_parallel()
        self.tp_size = mpu["tp_size"] if mpu else 1
        self.tp_rank = mpu["tp_rank"] if mpu else 0
        if num_embeddings % self.tp_size:
            raise ValueError("vocabulary size must divide evenly by TP size")
        self.num_embeddings_per_partition = num_embeddings // self.tp_size
        self.vocab_start_index = self.tp_rank * self.num_embeddings_per_partition
        self.vocab_end_index = self.vocab_start_index + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        _normal_init(self.weight)

    def forward(self, input_ids):
        mpu = get_model_parallel()
        group = mpu["tp_group"] if mpu else None
        outside = (input_ids < self.vocab_start_index) | (input_ids >= self.vocab_end_index)
        local_ids = (input_ids - self.vocab_start_index).masked_fill(outside, 0)
        output = F.embedding(local_ids, self.weight).masked_fill(outside.unsqueeze(-1), 0)
        return reduce_from_model_parallel_region(output, group)
