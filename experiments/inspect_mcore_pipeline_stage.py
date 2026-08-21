"""Print the exact Megatron-Core parameter contract held by each PP stage.

This is a diagnostic-only inspection command.  It creates no experiment asset
and is used before building a fixed-weight PP artifact: the canonical source
must map to these stage-local names without guessing their layer ownership.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.run_megatron_baseline import build_model, init_distributed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--pp", type=int, required=True)
    args = parser.parse_args()
    init_distributed(args.tp, args.pp)
    model, _ = build_model(args.tp, args.pp, no_scaled_init=True, fair_config=True)
    report = {
        "rank": dist.get_rank(),
        "local_rank": int(os.environ["LOCAL_RANK"]),
        "parameter_names": [name for name, _ in model.named_parameters()],
        "parameter_shapes": {name: list(value.shape) for name, value in model.named_parameters()},
    }
    print(json.dumps(report, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
