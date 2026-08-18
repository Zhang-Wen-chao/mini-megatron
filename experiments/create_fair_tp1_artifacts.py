"""Create immutable shared weights and next-token batches for a TP=1 study."""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from fair_tp1_contract import CONTRACT, build_mini, copy_mini_to_mcore, sha256_file
from eval.run_megatron_baseline import build_model, init_distributed
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from parallel.process_groups import set_model_parallel


def cpu_state_dict(model):
    # MCore records a small number of non-tensor extra-state entries (currently
    # None). Preserve them verbatim so strict load_state_dict remains valid.
    return {name: value.detach().cpu().contiguous() if torch.is_tensor(value) else value
            for name, value in model.state_dict().items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--batches", type=int, default=230)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.batches < 1 or args.batch_size < 1:
        parser.error("--batches and --batch-size must be positive")
    init_distributed(tp=1, pp=1)
    rank = dist.get_rank()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.manual_seed(args.seed)
    model_parallel_cuda_manual_seed(args.seed)
    set_model_parallel(None)
    mini, model_config = build_mini(device)
    mcore, _ = build_model(1, 1, use_bf16=False, no_scaled_init=True, fair_config=True)
    mcore = mcore.to(device)
    pairs = copy_mini_to_mcore(mini, mcore, model_config)
    if rank == 0:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=False)
        generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
        input_ids = torch.randint(0, model_config["vocab_size"],
                                  (args.batches, args.batch_size, model_config["max_seq_len"]),
                                  generator=generator, dtype=torch.long)
        labels = torch.roll(input_ids, shifts=-1, dims=-1)
        labels[:, :, -1] = -100
        paths = {
            "mini_checkpoint": output_dir / "mini_tp1.pt",
            "mcore_checkpoint": output_dir / "mcore_tp1.pt",
            "batches": output_dir / "next_token_batches.pt",
        }
        torch.save({"schema_version": 1, "contract": CONTRACT, "seed": args.seed,
                    "state_dict": cpu_state_dict(mini)}, paths["mini_checkpoint"])
        torch.save({"schema_version": 1, "contract": CONTRACT, "seed": args.seed,
                    "state_dict": cpu_state_dict(mcore)}, paths["mcore_checkpoint"])
        torch.save({"schema_version": 1, "contract": CONTRACT, "seed": args.seed + 1,
                    "input_ids": input_ids, "labels": labels}, paths["batches"])
        report = {"schema_version": 1, "contract": CONTRACT, "seed": args.seed,
                  "mapped_parameters": len(pairs), "batch_shape": list(input_ids.shape),
                  "files": {key: {"path": value.name, "sha256": sha256_file(value), "bytes": value.stat().st_size}
                            for key, value in paths.items()}}
        (output_dir / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
