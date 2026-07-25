import torch
import torch.distributed as dist
from comm.all_reduce import all_reduce


class AsyncAllReduce:
    """Overlap all-reduce with compute using separate CUDA stream."""

    def __init__(self, group):
        self.group = group
        self.stream = torch.cuda.Stream()
        self.pending = []

    def all_reduce_async(self, tensor):
        """Launch all-reduce on separate stream, return immediately."""
        with torch.cuda.stream(self.stream):
            all_reduce(tensor, self.group)
        event = torch.cuda.Event()
        event.record(stream=self.stream)
        self.pending.append((tensor, event))

    def synchronize(self):
        """Wait for all pending all-reduces to complete."""
        for _, event in self.pending:
            event.synchronize()
        self.pending.clear()
