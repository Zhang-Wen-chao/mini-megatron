"""Run five alternating, checksummed PP benchmark pairs from fixed artifacts.

This runner is intentionally limited to the post-hoc exploratory PP numerical
calibration recorded in the campaign archive.  Every bundle says so in its
tags.  It never upgrades that calibration into a claim that the original
per-tensor campaign gate passed.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "evidence/fair-125m-parallel-20260821"
RESULTS = CAMPAIGN / "benchmarks"


def selected_gpus_are_idle(gpus):
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                            text=True, capture_output=True, cwd=ROOT, check=False)
    if result.returncode:
        raise RuntimeError("nvidia-smi failed: " + result.stderr.strip())
    lines = [line.strip() for line in result.stdout.splitlines()
             if line.strip() and "No running" not in line]
    if lines:
        raise RuntimeError("GPU preflight found compute processes: " + "; ".join(lines))


def artifact_files(artifact):
    names = ["manifest.json", "next_token_batches.pt"]
    names.extend(sorted(path.name for path in artifact.glob("mini_*.pt")))
    names.extend(sorted(path.name for path in artifact.glob("mcore_*.pt")))
    return [artifact / name for name in names]


def bundle_for_name(name):
    matches = []
    for manifest_path in RESULTS.glob("*/manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("run_id", "").endswith("-" + name):
            matches.append(manifest_path.parent)
    if len(matches) != 1:
        raise RuntimeError(f"expected one bundle for {name}, found {len(matches)}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, choices=(1, 2), required=True)
    parser.add_argument("--pp", type=int, choices=(2,), required=True)
    parser.add_argument("--gpus", required=True, help="Physical CUDA ids, e.g. 0,1 or 0,1,2,3.")
    parser.add_argument("--start-port", type=int, required=True)
    args = parser.parse_args()
    gpus = tuple(int(value) for value in args.gpus.split(","))
    if len(gpus) != args.tp * args.pp or len(set(gpus)) != len(gpus):
        parser.error("--gpus must list exactly TP*PP unique GPU ids")
    topology_id = f"tp{args.tp}-pp{args.pp}-dp1"
    artifact = CAMPAIGN / "artifacts" / f"tp{args.tp}-pp{args.pp}-fp32-v1"
    calibration = CAMPAIGN / "parity" / f"tp{args.tp}-pp{args.pp}-fp32-v1-three-window-calibration.json"
    if not artifact.is_dir() or not calibration.is_file():
        raise RuntimeError("missing fixed artifact or three-window calibration")
    condition = f"125m-fp32-b4-s512-m8-u200-w30-{topology_id}-calibrated-custom-loop"
    schedule = []
    for pair in range(1, 6):
        order = ("mini", "mcore") if pair % 2 else ("mcore", "mini")
        schedule.extend((pair, ordinal, variant) for ordinal, variant in enumerate(order, start=1))
    outputs = []
    for index, (pair, ordinal, variant) in enumerate(schedule):
        selected_gpus_are_idle(gpus)
        name = f"calibrated-{topology_id}-p{pair:02d}-{variant}"
        env = dict(os.environ)
        env.update({"CUDA_VISIBLE_DEVICES": args.gpus, "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                    "NCCL_SHM_DISABLE": "1"})
        command = [sys.executable, "experiments/run_experiment.py", "--allow-dirty",
                   "--results-dir", str(RESULTS.relative_to(ROOT)), "--name", name,
                   "--tag", f"variant={variant}", "--tag", f"pair={pair:02d}",
                   "--tag", f"pair_ordinal={ordinal}", "--tag", f"topology={topology_id}",
                   "--tag", "evidence=exploratory_calibrated",
                   "--tag", "original_per_tensor_gate=not_passed",
                   "--tag", "calibration=post_hoc_three_window_reproduced",
                   "--tag", "comparison=matching_custom_loop",
                   "--tag", f"condition={condition}", "--tag", f"physical_gpus={args.gpus}",
                   "--artifact", str(calibration.relative_to(ROOT))]
        for path in artifact_files(artifact):
            command.extend(("--artifact", str(path.relative_to(ROOT))))
        command.extend(("--", "torchrun", "--master_port", str(args.start_port + index),
                        "--nproc_per_node", str(args.tp * args.pp),
                        "experiments/run_fair_pp_benchmark.py", "--implementation", variant,
                        "--tp", str(args.tp), "--pp", str(args.pp), "--artifact-dir",
                        str(artifact.relative_to(ROOT)), "--warmup-updates", "30",
                        "--num-updates", "200", "--microbatches-per-update", "8",
                        "--report-losses", "--progress-interval", "50"))
        print(json.dumps({"event": "start", "name": name, "variant": variant,
                          "pair": pair, "command": command}), flush=True)
        result = subprocess.run(command, cwd=ROOT, env=env)
        if result.returncode:
            raise RuntimeError(f"run failed ({result.returncode}): {name}")
        bundle = bundle_for_name(name)
        validation = subprocess.run([sys.executable, "experiments/validate_run_bundle.py", str(bundle)], cwd=ROOT)
        if validation.returncode:
            raise RuntimeError("bundle validation failed: " + str(bundle))
        metrics = json.loads((bundle / "metrics.json").read_text(encoding="utf-8"))
        outputs.append({"name": name, "pair": pair, "variant": variant,
                        "bundle": str(bundle), "metrics": metrics})
        print(json.dumps({"event": "complete", **outputs[-1]}, sort_keys=True), flush=True)
    report = CAMPAIGN / "reports" / f"{topology_id}-calibrated-five-pair-queue.json"
    if report.exists():
        raise RuntimeError("refusing to overwrite queue report: " + str(report))
    report.write_text(json.dumps({"schema_version": 1, "topology": topology_id,
                                  "status": "completed", "scope": "exploratory_calibrated matching custom-loop throughput",
                                  "samples": outputs}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "queue_complete", "report": str(report), "samples": len(outputs)}))


if __name__ == "__main__":
    main()
