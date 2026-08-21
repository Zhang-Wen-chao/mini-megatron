"""Numerical TP=2 gate for the shared 125M mini/Megatron-Core contract.

It creates one unsharded mini source model, gives both implementations their
documented TP shards, and rejects the topology unless initial weights, logits,
gradients and a one-step FP32 AdamW update meet the declared tolerances.
"""
import argparse
import contextlib
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
from fair_tp1_contract import build_mini, mcore_tensor_in_mini_layout, parameter_mappings
from fair_tp_parallel_contract import mcore_tp_shard, mini_tp_shard
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from parallel.process_groups import init_model_parallel, set_model_parallel


def relative_l2(left, right):
    return float(((left.float() - right.float()).norm() / left.float().norm().clamp_min(1e-12)).item())


def max_abs(left, right):
    return float((left.float() - right.float()).abs().max().item())


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def worst_pair(mini_params, mcore_params, pairs, model_config, field, tp_size):
    values = []
    local_config = dict(model_config)
    # QKV conversion is applied to one TP shard.  Preserve the global head
    # dimension (H / global_heads) by shrinking H and heads together.
    local_config["hidden_size"] //= tp_size
    local_config["num_attention_heads"] //= tp_size
    for mini_name, mcore_name in pairs:
        left = mini_params[mini_name].grad if field == "gradient" else mini_params[mini_name]
        right = mcore_params[mcore_name].grad if field == "gradient" else mcore_params[mcore_name]
        right = mcore_tensor_in_mini_layout(right, mini_name, mcore_name, local_config)
        values.append({"mini_parameter": mini_name, "mcore_parameter": mcore_name,
                       "max_abs": max_abs(left, right), "relative_l2": relative_l2(left, right)})
    return max(values, key=lambda value: value["max_abs"])


def build_source(device, seed):
    """Build the unsharded source independent of model-parallel RNG order."""
    set_model_parallel(None)
    torch.manual_seed(seed)
    source, config = build_mini(device)
    return {name: value.detach().clone() for name, value in source.named_parameters()}, config


def load_checkpoint(model, path):
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    expected_topology = {"tp": 2, "pp": 1, "dp": 1}
    if artifact.get("topology") != expected_topology:
        raise ValueError("checkpoint topology mismatch: " + str(path))
    model.load_state_dict(artifact["state_dict"], strict=True)


def load_fixed_batch(artifact_dir, rank, batch_size, model_config, batch_index):
    data = torch.load(Path(artifact_dir) / "next_token_batches.pt", map_location="cpu", weights_only=True)
    if data["input_ids"].size(1) != batch_size:
        raise ValueError("artifact batch size does not match --batch-size")
    if not 0 <= batch_index < data["input_ids"].size(0):
        raise ValueError("artifact batch index out of range")
    ids_cpu = data["input_ids"][batch_index].contiguous()
    labels_cpu = data["labels"][batch_index].contiguous()
    if ids_cpu.size(1) != model_config["max_seq_len"]:
        raise ValueError("artifact sequence length mismatch")
    return ids_cpu, labels_cpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, default=2, choices=(2,))
    parser.add_argument("--pp", type=int, default=1, choices=(1,))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--max-logits-relative-l2", type=float, default=5e-4)
    parser.add_argument("--max-gradient-relative-l2", type=float, default=5e-4)
    parser.add_argument("--max-parameter-relative-l2", type=float, default=1e-4)
    parser.add_argument("--artifact-dir", type=Path,
                        help="Optional immutable TP=2 artifact directory; loads its rank shards and first fixed batch.")
    parser.add_argument("--batch-index", type=int, default=0,
                        help="Which fixed micro-batch to use with --artifact-dir.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    init_distributed(args.tp, args.pp)
    rank = dist.get_rank()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    model_parallel_cuda_manual_seed(args.seed)
    mini_mpu = init_model_parallel(args.tp, args.pp)
    set_model_parallel(mini_mpu)
    mini, model_config = build_mini(device)
    mcore, _ = build_model(args.tp, args.pp, no_scaled_init=True, fair_config=True)
    mcore = mcore.to(device)
    mini_params, mcore_params = dict(mini.named_parameters()), dict(mcore.named_parameters())
    pairs = parameter_mappings(model_config)
    if set(mini_params) != {left for left, _ in pairs} or set(mcore_params) != {right for _, right in pairs}:
        raise RuntimeError("incomplete TP parameter mapping")
    if args.artifact_dir:
        load_checkpoint(mini, args.artifact_dir / f"mini_tp2_rank{rank}.pt")
        load_checkpoint(mcore, args.artifact_dir / f"mcore_tp2_rank{rank}.pt")
        ids_cpu, labels_cpu = load_fixed_batch(args.artifact_dir, rank, args.batch_size, model_config, args.batch_index)
    else:
        source, _ = build_source(device, args.seed)
        set_model_parallel(mini_mpu)
        with torch.no_grad():
            for mini_name, mcore_name in pairs:
                mini_value = mini_tp_shard(source[mini_name], mini_name, args.tp, mini_mpu["tp_rank"])
                mcore_value = mcore_tp_shard(source[mini_name], mini_name, mcore_name, model_config, args.tp, mini_mpu["tp_rank"])
                if mini_value.shape != mini_params[mini_name].shape or mcore_value.shape != mcore_params[mcore_name].shape:
                    raise RuntimeError(f"shard shape mismatch for {mini_name}: {tuple(mini_value.shape)} / {tuple(mcore_value.shape)}")
                mini_params[mini_name].copy_(mini_value)
                mcore_params[mcore_name].copy_(mcore_value)
        generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
        ids_cpu = torch.randint(0, model_config["vocab_size"], (args.batch_size, model_config["max_seq_len"]),
                                generator=generator, dtype=torch.long)
        labels_cpu = torch.roll(ids_cpu, shifts=-1, dims=1)
        labels_cpu[:, -1] = -100
    input_ids = ids_cpu.to(device)
    labels = labels_cpu.to(device)
    mask = (labels != -100).float()
    positions = torch.arange(model_config["max_seq_len"], device=device).unsqueeze(0).expand(args.batch_size, -1)
    attention = torch.triu(torch.ones(model_config["max_seq_len"], model_config["max_seq_len"], dtype=torch.bool, device=device), diagonal=1).unsqueeze(0).unsqueeze(0)
    mini_optim = torch.optim.AdamW(mini.parameters(), lr=6e-4, weight_decay=0.1, betas=(0.9, 0.999), fused=False)
    mcore_optim = torch.optim.AdamW(mcore.parameters(), lr=6e-4, weight_decay=0.1, betas=(0.9, 0.999), fused=False)
    initial = worst_pair(mini_params, mcore_params, pairs, model_config, "parameter", args.tp)
    mini_optim.zero_grad(set_to_none=True)
    mcore_optim.zero_grad(set_to_none=True)
    with contextlib.nullcontext():
        mini_logits = mini(input_ids)
        mcore_logits = mcore(input_ids, position_ids=positions, attention_mask=attention)
        mini_loss = torch.nn.functional.cross_entropy(mini_logits[:, :-1].reshape(-1, model_config["vocab_size"]), labels[:, :-1].reshape(-1))
        mcore_loss = torch.nn.functional.cross_entropy(mcore_logits[:, :-1].reshape(-1, model_config["vocab_size"]), labels[:, :-1].reshape(-1))
    mini_loss.backward()
    mcore_loss.backward()
    gradient = worst_pair(mini_params, mcore_params, pairs, model_config, "gradient", args.tp)
    mini_optim.step()
    mcore_optim.step()
    updated = worst_pair(mini_params, mcore_params, pairs, model_config, "parameter", args.tp)
    logits_rel = relative_l2(mini_logits[:, :-1], mcore_logits[:, :-1])
    report = {
        "schema_version": 1, "topology": {"tp": args.tp, "pp": args.pp, "dp": 1},
        "precision": "fp32", "seed": args.seed, "batch_shape": list(input_ids.shape),
        "artifact_dir": str(args.artifact_dir.resolve()) if args.artifact_dir else None,
        "batch_index": args.batch_index if args.artifact_dir else None,
        "input_sha256": digest(ids_cpu), "mapped_parameters": len(pairs),
        "initial_weight_max_abs": initial["max_abs"], "worst_initial_parameter": initial,
        "logits_relative_l2": logits_rel, "logits_max_abs": max_abs(mini_logits[:, :-1], mcore_logits[:, :-1]),
        "loss_abs": abs(float(mini_loss.item()) - float(mcore_loss.item())),
        "worst_gradient": gradient, "worst_parameter_after_one_adamw_step": updated,
        "thresholds": {"max_logits_relative_l2": args.max_logits_relative_l2,
                       "max_gradient_relative_l2": args.max_gradient_relative_l2,
                       "max_parameter_relative_l2": args.max_parameter_relative_l2},
    }
    report["passed"] = (initial["max_abs"] == 0.0 and logits_rel <= args.max_logits_relative_l2
                        and gradient["relative_l2"] <= args.max_gradient_relative_l2
                        and updated["relative_l2"] <= args.max_parameter_relative_l2)
    if rank == 0:
        rendered = json.dumps(report, indent=2, sort_keys=True)
        print(rendered)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n")
    dist.barrier()
    dist.destroy_process_group()
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
