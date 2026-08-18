"""Print the TP=1 Megatron-Core parameter surface for fair-comparison design.

Run with torchrun in the L20 experiment container.  This is deliberately an
audit tool: it does not measure throughput and does not mutate model weights.
"""
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

# torchrun executes this file with experiments/ as sys.path[0].
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.run_megatron_baseline import build_model, init_distributed


def main():
    init_distributed(tp=1, pp=1)
    model, config = build_model(tp=1, pp=1, use_bf16=False, no_scaled_init=True)
    if dist.get_rank() == 0:
        rows = [
            {"name": name, "shape": list(parameter.shape), "dtype": str(parameter.dtype)}
            for name, parameter in model.named_parameters()
        ]
        print(json.dumps({"parameter_count": sum(p.numel() for p in model.parameters()),
                          "parameters": rows, "config": config.__dict__}, indent=2, default=str))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
