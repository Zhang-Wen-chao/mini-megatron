"""Freeze auditable fixed artifacts for a TP/PP fair-comparison topology.

The canonical source is an unsharded mini-megatron state dict.  Each rank gets
the exact local pipeline stage required by its framework, including TP shards.
This avoids the common PP mistake of assigning random, differently partitioned
weights to the two implementations.  The destination is immutable: a previous
artifact directory is never overwritten.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config as cfg
from eval.run_megatron_baseline import build_model, init_distributed
from fair_tp1_contract import CONTRACT, build_mini, parameter_mappings
from fair_tp_parallel_contract import mcore_tp_shard, mini_tp_shard
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from parallel.process_groups import init_model_parallel, set_model_parallel
from parallel.tensor_parallel import ColumnParallelLinear
from model.embedding import Embedding
from model.transformer import DecoderLayer


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_info(path):
    path = Path(path)
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}


def cpu_state_dict(model):
    return {name: value.detach().cpu().contiguous() if torch.is_tensor(value) else value
            for name, value in model.state_dict().items()}


def write_checksums(directory):
    directory = Path(directory)
    lines = [sha256(path) + "  " + path.name
             for path in sorted(directory.iterdir())
             if path.is_file() and path.name != "checksums.sha256"]
    (directory / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


class MiniPipelineStage(nn.Module):
    """Mini-megatron's local PP stage with deliberately stable state names."""

    def __init__(self, config, pp_rank, pp_size):
        super().__init__()
        layers_per_stage = config["num_layers"] // pp_size
        self.first = pp_rank == 0
        self.last = pp_rank == pp_size - 1
        self.embedding = (Embedding(config["vocab_size"], config["hidden_size"],
                                    config["max_seq_len"]) if self.first else None)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(config["hidden_size"], config["num_attention_heads"],
                         config["ffn_hidden_size"])
            for _ in range(layers_per_stage)
        ])
        self.ln_f = nn.LayerNorm(config["hidden_size"]) if self.last else None
        self.lm_head = (ColumnParallelLinear(config["hidden_size"], config["vocab_size"],
                                             bias=False, gather_output=True) if self.last else None)


def full_source(device, seed):
    """Construct the shared full source independently of TP RNG ordering."""
    set_model_parallel(None)
    torch.manual_seed(seed)
    model, config = build_mini(device)
    return {name: value.detach().cpu().contiguous() for name, value in model.named_parameters()}, config


def stage_parameter_plan(config, pp_rank, pp_size):
    """Return (source-mini-name, local-mini-name, local-mcore-name) triples."""
    layers_per_stage = config["num_layers"] // pp_size
    source_to_mcore = dict(parameter_mappings(config))
    triples = []
    if pp_rank == 0:
        for mini_name in ("embedding.token_embedding.weight", "embedding.position_embedding.weight"):
            triples.append((mini_name, mini_name, source_to_mcore[mini_name]))
    for local_layer in range(layers_per_stage):
        global_layer = pp_rank * layers_per_stage + local_layer
        source_prefix = f"decoder.layers.{global_layer}."
        local_mini_prefix = f"decoder_layers.{local_layer}."
        local_mcore_prefix = f"decoder.layers.{local_layer}."
        for source_name, global_mcore_name in source_to_mcore.items():
            if source_name.startswith(source_prefix):
                triples.append((
                    source_name,
                    local_mini_prefix + source_name.removeprefix(source_prefix),
                    local_mcore_prefix + global_mcore_name.removeprefix(source_prefix),
                ))
    if pp_rank == pp_size - 1:
        for mini_name in ("ln_f.weight", "ln_f.bias", "lm_head.weight"):
            triples.append((mini_name, mini_name, source_to_mcore[mini_name]))
    return triples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, choices=(1, 2), required=True)
    parser.add_argument("--pp", type=int, choices=(2,), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--batches", type=int, default=1840)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.batches < 1 or args.batch_size < 1:
        parser.error("--batches and --batch-size must be positive")

    init_distributed(args.tp, args.pp)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != args.tp * args.pp:
        raise ValueError(f"world size {world_size} does not equal TP*PP={args.tp * args.pp}")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    model_parallel_cuda_manual_seed(args.seed)
    mini_mpu = init_model_parallel(args.tp, args.pp)
    set_model_parallel(mini_mpu)
    config = cfg.get_model_config()
    if config["num_layers"] % args.pp:
        raise ValueError("num_layers must divide PP size")
    mini = MiniPipelineStage(config, mini_mpu["pp_rank"], args.pp).to(device)
    mcore, _ = build_model(args.tp, args.pp, no_scaled_init=True, fair_config=True)
    mcore = mcore.to(device)
    source, _ = full_source(device, args.seed)
    set_model_parallel(mini_mpu)
    triples = stage_parameter_plan(config, mini_mpu["pp_rank"], args.pp)
    mini_params, mcore_params = dict(mini.named_parameters()), dict(mcore.named_parameters())
    expected_mini = {local_mini for _, local_mini, _ in triples}
    expected_mcore = {local_mcore for _, _, local_mcore in triples}
    if set(mini_params) != expected_mini:
        raise RuntimeError("unexpected mini stage parameters: " + repr(sorted(set(mini_params) ^ expected_mini)))
    if set(mcore_params) != expected_mcore:
        raise RuntimeError("unexpected MCore stage parameters: " + repr(sorted(set(mcore_params) ^ expected_mcore)))
    with torch.no_grad():
        for source_name, mini_name, mcore_name in triples:
            mini_value = mini_tp_shard(source[source_name], source_name, args.tp, mini_mpu["tp_rank"])
            mcore_value = mcore_tp_shard(source[source_name], source_name, mcore_name, config,
                                         args.tp, mini_mpu["tp_rank"])
            if mini_value.shape != mini_params[mini_name].shape:
                raise RuntimeError(f"mini shard shape mismatch for {source_name}: {tuple(mini_value.shape)} != {tuple(mini_params[mini_name].shape)}")
            if mcore_value.shape != mcore_params[mcore_name].shape:
                raise RuntimeError(f"MCore shard shape mismatch for {source_name}: {tuple(mcore_value.shape)} != {tuple(mcore_params[mcore_name].shape)}")
            mini_params[mini_name].copy_(mini_value)
            mcore_params[mcore_name].copy_(mcore_value)

    output_dir = args.output_dir.resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    topology = {"tp": args.tp, "pp": args.pp, "dp": 1}
    mini_path = output_dir / f"mini_tp{args.tp}_pp{args.pp}_rank{rank}.pt"
    mcore_path = output_dir / f"mcore_tp{args.tp}_pp{args.pp}_rank{rank}.pt"
    for path, model in ((mini_path, mini), (mcore_path, mcore)):
        torch.save({"schema_version": 1, "contract": CONTRACT, "topology": topology,
                    "seed": args.seed, "rank": rank, "state_dict": cpu_state_dict(model)}, path)
    local_files = {"rank": rank, "mini": artifact_info(mini_path), "mcore": artifact_info(mcore_path)}
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_files)
    if rank == 0:
        source_path = output_dir / "canonical_full_mini_source.pt"
        batches_path = output_dir / "next_token_batches.pt"
        torch.save({"schema_version": 1, "contract": CONTRACT, "seed": args.seed,
                    "state_dict": source}, source_path)
        generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
        input_ids = torch.randint(0, config["vocab_size"],
                                  (args.batches, args.batch_size, config["max_seq_len"]),
                                  generator=generator, dtype=torch.long)
        labels = torch.roll(input_ids, shifts=-1, dims=-1)
        labels[:, :, -1] = -100
        torch.save({"schema_version": 1, "contract": CONTRACT, "seed": args.seed + 1,
                    "input_ids": input_ids, "labels": labels}, batches_path)
        manifest = {
            "schema_version": 1, "contract": CONTRACT, "topology": topology,
            "seed": args.seed, "mapped_parameters_per_rank": len(triples),
            "batch_shape": list(input_ids.shape),
            "stage_mapping": {"layers_per_pipeline_stage": config["num_layers"] // args.pp,
                              "rank_layout": "rank = pp_rank * tp + tp_rank"},
            "files": {"canonical_full_mini_source": artifact_info(source_path),
                      "next_token_batches": artifact_info(batches_path),
                      "rank_stages": gathered},
        }
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_checksums(output_dir)
        print(json.dumps(manifest, indent=2, sort_keys=True))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
