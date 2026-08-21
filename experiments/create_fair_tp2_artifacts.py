"""Freeze shared full weights, TP=2 shards, and fixed batches for fair runs.

Run with two ranks.  The output directory is created exactly once and is never
overwritten; its manifest fingerprints the canonical full source, every per-rank
framework shard and every training batch.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.run_megatron_baseline import build_model, init_distributed
from fair_tp1_contract import CONTRACT, build_mini, parameter_mappings
from fair_tp_parallel_contract import mcore_tp_shard, mini_tp_shard
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from parallel.process_groups import init_model_parallel, set_model_parallel


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cpu_state_dict(model):
    return {name: value.detach().cpu().contiguous() if torch.is_tensor(value) else value
            for name, value in model.state_dict().items()}


def artifact_info(path):
    return {"path": Path(path).name, "bytes": Path(path).stat().st_size, "sha256": sha256(path)}


def write_checksums(output_dir):
    records = []
    for path in sorted(Path(output_dir).iterdir()):
        if path.is_file() and path.name != "checksums.sha256":
            records.append(sha256(path) + "  " + path.name)
    (Path(output_dir) / "checksums.sha256").write_text("\n".join(records) + "\n")


def build_source(device, seed):
    set_model_parallel(None)
    torch.manual_seed(seed)
    model, config = build_mini(device)
    return {name: value.detach().cpu().contiguous() for name, value in model.named_parameters()}, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--batches", type=int, default=1840,
                        help="Must cover (warmup + measured) * microbatches/update.")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.batches < 1 or args.batch_size < 1:
        parser.error("--batches and --batch-size must be positive")

    init_distributed(2, 1)
    rank = dist.get_rank()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    model_parallel_cuda_manual_seed(args.seed)
    mini_mpu = init_model_parallel(2, 1)
    set_model_parallel(mini_mpu)
    mini, model_config = build_mini(device)
    mcore, _ = build_model(2, 1, no_scaled_init=True, fair_config=True)
    mcore = mcore.to(device)
    source, _ = build_source(device, args.seed)
    set_model_parallel(mini_mpu)
    mini_params, mcore_params = dict(mini.named_parameters()), dict(mcore.named_parameters())
    pairs = parameter_mappings(model_config)
    with torch.no_grad():
        for mini_name, mcore_name in pairs:
            mini_params[mini_name].copy_(mini_tp_shard(source[mini_name], mini_name, 2, mini_mpu["tp_rank"]))
            mcore_params[mcore_name].copy_(mcore_tp_shard(source[mini_name], mini_name, mcore_name, model_config, 2, mini_mpu["tp_rank"]))

    output_dir = args.output_dir.resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    mini_path = output_dir / f"mini_tp2_rank{rank}.pt"
    mcore_path = output_dir / f"mcore_tp2_rank{rank}.pt"
    torch.save({"schema_version": 1, "contract": CONTRACT, "topology": {"tp": 2, "pp": 1, "dp": 1},
                "seed": args.seed, "rank": rank, "state_dict": cpu_state_dict(mini)}, mini_path)
    torch.save({"schema_version": 1, "contract": CONTRACT, "topology": {"tp": 2, "pp": 1, "dp": 1},
                "seed": args.seed, "rank": rank, "state_dict": cpu_state_dict(mcore)}, mcore_path)
    local_files = {"rank": rank, "mini": artifact_info(mini_path), "mcore": artifact_info(mcore_path)}
    gathered = [None, None]
    dist.all_gather_object(gathered, local_files)
    if rank == 0:
        source_path = output_dir / "canonical_full_mini_source.pt"
        batches_path = output_dir / "next_token_batches.pt"
        torch.save({"schema_version": 1, "contract": CONTRACT, "seed": args.seed, "state_dict": source}, source_path)
        generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
        input_ids = torch.randint(0, model_config["vocab_size"],
                                  (args.batches, args.batch_size, model_config["max_seq_len"]),
                                  generator=generator, dtype=torch.long)
        labels = torch.roll(input_ids, shifts=-1, dims=-1)
        labels[:, :, -1] = -100
        torch.save({"schema_version": 1, "contract": CONTRACT, "seed": args.seed + 1,
                    "input_ids": input_ids, "labels": labels}, batches_path)
        manifest = {
            "schema_version": 1, "contract": CONTRACT, "topology": {"tp": 2, "pp": 1, "dp": 1},
            "seed": args.seed, "mapped_parameters": len(pairs), "batch_shape": list(input_ids.shape),
            "files": {"canonical_full_mini_source": artifact_info(source_path),
                      "next_token_batches": artifact_info(batches_path),
                      "rank_shards": gathered},
        }
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        write_checksums(output_dir)
        print(json.dumps(manifest, indent=2, sort_keys=True))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
