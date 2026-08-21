"""Compare fixed mini/MCore PP stage checkpoints before or after a matched run.

This validator is deliberately CPU-only.  It proves that each pipeline stage
has the same initial source weights and quantifies stage-local parameter drift
after a fixed number of matching custom-loop updates.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config as cfg
from create_fair_pp_artifacts import stage_parameter_plan
from fair_tp1_contract import mcore_tensor_in_mini_layout


def relative_l2(left, right):
    return float(((left.float() - right.float()).norm() / left.float().norm().clamp_min(1e-12)).item())


def max_abs(left, right):
    return float((left.float() - right.float()).abs().max().item())


def load_mapping(path, key):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if key not in data:
        raise ValueError(f"missing {key}: " + str(path))
    return data[key]


def checkpoint_path(root, implementation, tp, pp, rank, post_run):
    if post_run:
        return root / f"{implementation}_rank{rank}.pt"
    return root / f"{implementation}_tp{tp}_pp{pp}_rank{rank}.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--mini-state-dir", type=Path,
                        help="Post-run mini state directory written by run_fair_pp_benchmark.py.")
    parser.add_argument("--mcore-state-dir", type=Path,
                        help="Post-run MCore state directory written by run_fair_pp_benchmark.py.")
    parser.add_argument("--tp", type=int, choices=(1, 2), required=True)
    parser.add_argument("--pp", type=int, choices=(2,), required=True)
    parser.add_argument("--max-relative-l2", type=float, default=1e-4)
    parser.add_argument("--field", choices=("state_dict", "gradients"), default="state_dict")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if bool(args.mini_state_dir) != bool(args.mcore_state_dir):
        parser.error("provide both --mini-state-dir and --mcore-state-dir, or neither")
    post_run = args.mini_state_dir is not None
    world_size = args.tp * args.pp
    all_values = []
    squared_delta = 0.0
    squared_reference = 0.0
    max_abs_overall = 0.0
    config = cfg.get_model_config()
    for rank in range(world_size):
        pp_rank = rank // args.tp
        triples = stage_parameter_plan(config, pp_rank, args.pp)
        mini_root = args.mini_state_dir if post_run else args.artifact_dir
        mcore_root = args.mcore_state_dir if post_run else args.artifact_dir
        mini = load_mapping(checkpoint_path(mini_root, "mini", args.tp, args.pp, rank, post_run), args.field)
        mcore = load_mapping(checkpoint_path(mcore_root, "mcore", args.tp, args.pp, rank, post_run), args.field)
        for source_name, mini_name, mcore_name in triples:
            if mini_name not in mini or mcore_name not in mcore:
                raise KeyError(f"missing mapped parameter on rank {rank}: {mini_name} / {mcore_name}")
            # A TP shard owns H/TP hidden channels and heads.  QKV conversion
            # must use that local layout, not the global model dimensions.
            local_config = dict(config)
            local_config["hidden_size"] //= args.tp
            local_config["num_attention_heads"] //= args.tp
            right = mcore_tensor_in_mini_layout(mcore[mcore_name], source_name, mcore_name, local_config)
            left = mini[mini_name]
            if left.shape != right.shape:
                raise ValueError(f"shape mismatch on rank {rank}: {mini_name} {tuple(left.shape)} vs {mcore_name} {tuple(right.shape)}")
            all_values.append({
                "rank": rank, "source_parameter": source_name,
                "mini_parameter": mini_name, "mcore_parameter": mcore_name,
                "max_abs": max_abs(left, right), "relative_l2": relative_l2(left, right),
            })
            delta = (left.float() - right.float())
            squared_delta += float(delta.square().sum().item())
            squared_reference += float(left.float().square().sum().item())
            max_abs_overall = max(max_abs_overall, float(delta.abs().max().item()))
    worst_abs = max(all_values, key=lambda value: value["max_abs"])
    worst_rel = max(all_values, key=lambda value: value["relative_l2"])
    global_relative_l2 = (squared_delta ** 0.5) / max(squared_reference ** 0.5, 1e-12)
    report = {
        "schema_version": 1,
        "mode": ("post_update" if post_run else "initial_artifact"),
        "field": args.field,
        "topology": {"tp": args.tp, "pp": args.pp, "dp": 1},
        "mapped_parameters": len(all_values),
        "max_relative_l2_threshold": args.max_relative_l2,
        "worst_max_abs": worst_abs,
        "worst_relative_l2": worst_rel,
        "global_relative_l2": global_relative_l2,
        "global_max_abs": max_abs_overall,
        "passed": worst_abs["max_abs"] == 0.0 if not post_run else worst_rel["relative_l2"] <= args.max_relative_l2,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
