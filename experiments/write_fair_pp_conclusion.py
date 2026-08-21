"""Freeze one conditional PP throughput conclusion from immutable evidence.

The PP mappings in this study reproduce a post-hoc exploratory calibration,
not the campaign's original per-tensor gate.  This writer makes that limitation
machine-readable in the conclusion instead of relying on prose elsewhere.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evidence(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite immutable conclusion: " + str(args.output))
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    selection = summary.get("selection", {})
    if selection.get("topology") != args.topology:
        parser.error("summary topology selection does not match --topology")
    if summary.get("left_variant") != "mini" or summary.get("right_variant") != "mcore":
        parser.error("summary must compare mini against mcore")
    if len(summary.get("pairs", [])) != 5:
        parser.error("exactly five paired samples are required")
    expected_topology = {
        "tp1-pp2-dp1": {"tp": 1, "pp": 2, "dp": 1},
        "tp2-pp2-dp1": {"tp": 2, "pp": 2, "dp": 1},
    }.get(args.topology)
    if expected_topology is None or calibration.get("topology") != expected_topology:
        parser.error("calibration topology mismatch")
    original_gate = calibration.get("original_per_tensor_gate_passed_all_windows")
    calibrated_gate = calibration.get("post_hoc_exploratory_calibration_passed_all_windows")
    if original_gate or not calibrated_gate:
        parser.error("this writer is only for the explicit exploratory-calibration PP path")
    ratio = summary["paired_ratio_left_over_right"]
    report = {
        "schema_version": 1,
        "topology": args.topology,
        "status": "conditional_exploratory_throughput_complete",
        "claim_status": "conditional_exploratory",
        "scope": ("125M FP32 fixed-artifact matching custom-loop PP comparison; "
                  "this is not a full Megatron-Core training-stack comparison."),
        "numerical_gate": {
            "original_per_tensor_gate_passed_all_windows": original_gate,
            "post_hoc_exploratory_calibration_passed_all_windows": calibrated_gate,
            "caveat": calibration.get("calibration_caveat"),
        },
        "throughput": {
            "mini_tok_s": summary["left_throughput_tok_s"],
            "mcore_tok_s": summary["right_throughput_tok_s"],
            "mini_over_mcore": ratio,
            "five_pairs": [{
                "pair": pair["pair"],
                "mini_tok_s": pair["left"]["throughput_tok_s"],
                "mcore_tok_s": pair["right"]["throughput_tok_s"],
                "ratio": pair["ratio"],
            } for pair in summary["pairs"]],
        },
        "supported_claim": ("Only a conditional exploratory throughput observation for the exact recorded "
                            "configuration. It must not be described as passing the original campaign gate, "
                            "a general mini-megatron advantage, or a full MCore ranking."),
        "not_supported": [
            "original per-tensor numerical-gate pass",
            "general framework performance ranking",
            "default or full Megatron-Core production training-stack performance",
            "BF16, larger models, other hardware, multi-node, or training-quality claims",
        ],
        "source_evidence": {
            "paired_summary": evidence(args.summary),
            "three_window_calibration": evidence(args.calibration),
            "artifact_manifest": evidence(args.artifact_manifest),
        },
        "profile_requirement": "A separately captured multi-rank Nsight profile remains required for topology campaign closure.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
