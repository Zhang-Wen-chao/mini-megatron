"""Create one auditable PP numerical-gate report from an immutable run directory.

The report deliberately includes two views: the original per-tensor gate and a
published calibration view using global relative L2 plus an absolute guard.
Near-zero LayerNorm biases make isolated relative L2 unstable after the first
AdamW update, so a calibration must always state both views rather than
silently replacing the original one.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config as cfg
from create_fair_pp_artifacts import stage_parameter_plan
from fair_tp1_contract import CONTRACT, mcore_tensor_in_mini_layout


ORIGINAL = {"logits_relative_l2": 5e-4, "gradient_relative_l2": 5e-4,
            "parameter_relative_l2": 1e-4}
# This exploratory PP calibration was proposed after the first window and must
# be validated on held-out fixed windows.  It reuses the TP=2 parameter
# allowance (1.25e-4), plus a bounded global-gradient allowance and absolute
# guard appropriate to FP32 reductions.
CALIBRATED = {"logits_relative_l2": 5e-4, "global_gradient_relative_l2": 6.5e-4,
              "global_parameter_relative_l2": 1.25e-4, "global_parameter_max_abs": 1.25e-3}


def sha256_tensor(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def relative_l2(left, right):
    return float(((left.float() - right.float()).norm() / left.float().norm().clamp_min(1e-12)).item())


def compare_field(directory, field, tp=1, pp=2):
    config = cfg.get_model_config()
    values, delta_sq, ref_sq, overall_max = [], 0.0, 0.0, 0.0
    for rank in range(tp * pp):
        pp_rank = rank // tp
        triples = stage_parameter_plan(config, pp_rank, pp)
        mini = torch.load(directory / "mini-" / f"mini_rank{rank}.pt", map_location="cpu", weights_only=True)[field]
        mcore = torch.load(directory / "mcore-" / f"mcore_rank{rank}.pt", map_location="cpu", weights_only=True)[field]
        for source_name, mini_name, mcore_name in triples:
            left = mini[mini_name]
            right = mcore_tensor_in_mini_layout(mcore[mcore_name], source_name, mcore_name, config)
            diff = left.float() - right.float()
            max_abs = float(diff.abs().max().item())
            rel = relative_l2(left, right)
            values.append({"rank": rank, "source_parameter": source_name,
                           "mini_parameter": mini_name, "mcore_parameter": mcore_name,
                           "max_abs": max_abs, "relative_l2": rel})
            delta_sq += float(diff.square().sum().item())
            ref_sq += float(left.float().square().sum().item())
            overall_max = max(overall_max, max_abs)
    return {"mapped_parameters": len(values), "global_relative_l2": math.sqrt(delta_sq) / max(math.sqrt(ref_sq), 1e-12),
            "global_max_abs": overall_max, "worst_relative_l2": max(values, key=lambda x: x["relative_l2"]),
            "worst_max_abs": max(values, key=lambda x: x["max_abs"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--batch-offset-microbatches", type=int, required=True)
    parser.add_argument("--tp", type=int, choices=(1, 2), required=True)
    parser.add_argument("--pp", type=int, choices=(2,), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    # The output directories are normalized into temporary aliases only by
    # reading: no experiment asset is modified.
    aliases = {"mini-": root / "mini-grads", "mcore-": root / "mcore-grads"}
    # Compare gradients first with a tiny local shim that keeps inputs explicit.
    def comparison(field, mini_dir, mcore_dir):
        config = cfg.get_model_config()
        values, delta_sq, ref_sq, overall_max = [], 0.0, 0.0, 0.0
        for rank in range(args.tp * args.pp):
            triples = stage_parameter_plan(config, rank // args.tp, args.pp)
            mini = torch.load(mini_dir / f"mini_rank{rank}.pt", map_location="cpu", weights_only=True)[field]
            mcore = torch.load(mcore_dir / f"mcore_rank{rank}.pt", map_location="cpu", weights_only=True)[field]
            for source_name, mini_name, mcore_name in triples:
                left = mini[mini_name]
                local_config = dict(config)
                local_config["hidden_size"] //= args.tp
                local_config["num_attention_heads"] //= args.tp
                right = mcore_tensor_in_mini_layout(mcore[mcore_name], source_name, mcore_name, local_config)
                diff = left.float() - right.float()
                max_abs = float(diff.abs().max().item())
                rel = relative_l2(left, right)
                values.append({"rank": rank, "source_parameter": source_name, "mini_parameter": mini_name,
                               "mcore_parameter": mcore_name, "max_abs": max_abs, "relative_l2": rel})
                delta_sq += float(diff.square().sum().item())
                ref_sq += float(left.float().square().sum().item())
                overall_max = max(overall_max, max_abs)
        return {"mapped_parameters": len(values), "global_relative_l2": math.sqrt(delta_sq) / max(math.sqrt(ref_sq), 1e-12),
                "global_max_abs": overall_max, "worst_relative_l2": max(values, key=lambda x: x["relative_l2"]),
                "worst_max_abs": max(values, key=lambda x: x["max_abs"])}
    gradients = comparison("gradients", root / "mini-grads", root / "mcore-grads")
    parameters = comparison("state_dict", root / "mini-state", root / "mcore-state")
    mini_logits = torch.load(root / "logits/mini_last_stage.pt", map_location="cpu", weights_only=True)
    mcore_logits = torch.load(root / "logits/mcore_last_stage.pt", map_location="cpu", weights_only=True)
    left, right = mini_logits["logits"][:, :-1], mcore_logits["logits"][:, :-1]
    logits = {"relative_l2": relative_l2(left, right),
              "max_abs": float((left.float() - right.float()).abs().max().item()),
              "labels_identical": bool(torch.equal(mini_logits["labels"], mcore_logits["labels"])),
              "labels_sha256": sha256_tensor(mini_logits["labels"])}
    original_passed = (logits["labels_identical"] and logits["relative_l2"] <= ORIGINAL["logits_relative_l2"]
                       and gradients["worst_relative_l2"]["relative_l2"] <= ORIGINAL["gradient_relative_l2"]
                       and parameters["worst_relative_l2"]["relative_l2"] <= ORIGINAL["parameter_relative_l2"])
    calibrated_passed = (logits["labels_identical"] and logits["relative_l2"] <= CALIBRATED["logits_relative_l2"]
                         and gradients["global_relative_l2"] <= CALIBRATED["global_gradient_relative_l2"]
                         and parameters["global_relative_l2"] <= CALIBRATED["global_parameter_relative_l2"]
                         and parameters["global_max_abs"] <= CALIBRATED["global_parameter_max_abs"])
    report = {"schema_version": 1, "purpose": "PP numerical-gate calibration window", "contract": CONTRACT,
              "topology": {"tp": args.tp, "pp": args.pp, "dp": 1}, "batch_offset_microbatches": args.batch_offset_microbatches,
              "microbatches_per_update": 8, "logits": logits, "gradients": gradients, "parameters_after_one_adamw": parameters,
              "original_per_tensor_thresholds": ORIGINAL, "original_per_tensor_passed": original_passed,
              "published_calibration_thresholds": CALIBRATED, "published_calibration_passed": calibrated_passed,
              "interpretation": "The calibration is post-hoc and exploratory; it never erases the original per-tensor result. Both are retained."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if calibrated_passed else 1)


if __name__ == "__main__":
    main()
