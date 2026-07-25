import torch
import torch.distributed as dist


class AsyncP2P:
    """Non-blocking P2P communication with CUDA stream overlap."""

    def __init__(self, pp_group):
        self.group = pp_group
        self.stream = torch.cuda.Stream()
        self.requests = []

    def send_async(self, tensor, dst):
        with torch.cuda.stream(self.stream):
            req = dist.isend(tensor, dst=dst, group=self.group)
            self.requests.append(req)

    def recv_async(self, tensor, src):
        with torch.cuda.stream(self.stream):
            req = dist.irecv(tensor, src=src, group=self.group)
            self.requests.append(req)
        return tensor

    def synchronize(self):
        for req in self.requests:
            req.wait()
        self.requests.clear()
