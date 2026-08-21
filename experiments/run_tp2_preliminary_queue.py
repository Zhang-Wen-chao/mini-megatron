"""Run the remaining fixed-artifact TP=2 preliminary benchmark samples.

This deliberately runs one sample at a time.  Before each sample it requires
the selected GPUs to be idle; afterwards it validates the immutable run bundle.
Any failed preflight, failed command, or invalid checksum stops the queue rather
than silently skipping a sample.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evidence/fair-125m-parallel-20260821/benchmarks"
ARTIFACT = ROOT / "evidence/fair-125m-parallel-20260821/artifacts/tp2-pp1-fp32-v1"


def selected_gpus_are_idle(gpu_pair):
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True, capture_output=True, check=False, cwd=ROOT,
    )
    if result.returncode:
        raise RuntimeError("nvidia-smi failed: " + result.stderr.strip())
    gpu_info = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        text=True, capture_output=True, check=True, cwd=ROOT,
    )
    uuid_by_index = {int(line.split(",", 1)[0].strip()): line.split(",", 1)[1].strip()
                     for line in gpu_info.stdout.splitlines() if line.strip()}
    selected_uuids = {uuid_by_index[index] for index in gpu_pair}
    occupied = [line for line in result.stdout.splitlines()
                if line.split(",", 1)[0].strip() in selected_uuids]
    if occupied:
        raise RuntimeError("selected GPU(s) are occupied: " + "; ".join(occupied))


def bundle_for_name(name):
    matches = []
    for manifest in RESULTS.glob("*/manifest.json"):
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if data.get("run_id", "").endswith("-" + name):
            matches.append(manifest.parent)
    if len(matches) != 1:
        raise RuntimeError(f"expected one bundle for {name}, found {len(matches)}")
    return matches[0]


def run_sample(name, variant, pair, ordinal, port, gpu_pair):
    selected_gpus_are_idle(gpu_pair)
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": ",".join(str(value) for value in gpu_pair),
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_SHM_DISABLE": "1",
    })
    common = [
        sys.executable, "experiments/run_experiment.py", "--allow-dirty",
        "--allow-active-gpus", "--results-dir", str(RESULTS.relative_to(ROOT)),
        "--name", name,
        "--tag", f"variant={variant}", "--tag", f"pair={pair:02d}",
        "--tag", f"pair_ordinal={ordinal}", "--tag", "topology=tp2-pp1-dp1",
        "--tag", "evidence=preliminary", "--tag", "protocol=tp2-fp32-v11-device-bound",
        "--tag", "condition=125m-fp32-b4-s512-m8-u200-w30",
        "--tag", "gpu_pair=physical-0-1", "--tag", "topology_link=NODE",
        "--artifact", str((ARTIFACT / "manifest.json").relative_to(ROOT)),
        "--artifact", str((ARTIFACT / "next_token_batches.pt").relative_to(ROOT)),
        "--", "torchrun", "--master_port", str(port), "--nproc_per_node=2",
        "experiments/run_fair_tp2_benchmark.py", "--implementation", variant,
        "--artifact-dir", str(ARTIFACT.relative_to(ROOT)), "--num-updates", "200",
        "--warmup-updates", "30", "--microbatches-per-update", "8", "--report-losses",
    ]
    print(json.dumps({"event": "start", "name": name, "variant": variant, "pair": pair}), flush=True)
    result = subprocess.run(common, cwd=ROOT, env=env)
    if result.returncode:
        raise RuntimeError(f"run failed ({result.returncode}): {name}")
    bundle = bundle_for_name(name)
    validation = subprocess.run([sys.executable, "experiments/validate_run_bundle.py", str(bundle)], cwd=ROOT)
    if validation.returncode:
        raise RuntimeError(f"bundle validation failed: {bundle}")
    metrics = json.loads((bundle / "metrics.json").read_text(encoding="utf-8"))
    print(json.dumps({"event": "complete", "name": name, "bundle": str(bundle), "metrics": metrics}), flush=True)
    return {"name": name, "variant": variant, "pair": pair, "bundle": str(bundle), "metrics": metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-pair", type=int, default=3)
    parser.add_argument("--end-pair", type=int, default=5)
    parser.add_argument("--start-port", type=int, default=29661)
    args = parser.parse_args()
    if not 1 <= args.start_pair <= args.end_pair <= 5:
        parser.error("pairs must be within 1..5")
    schedule = []
    for pair in range(args.start_pair, args.end_pair + 1):
        order = ("mini", "mcore") if pair % 2 else ("mcore", "mini")
        schedule.extend((pair, ordinal, variant) for ordinal, variant in enumerate(order, start=1))
    results = []
    for index, (pair, ordinal, variant) in enumerate(schedule):
        name = f"preliminary-tp2-pp1-gpu01-p{pair:02d}-{variant}"
        results.append(run_sample(name, variant, pair, ordinal, args.start_port + index, (0, 1)))
    report = ROOT / "evidence/fair-125m-parallel-20260821/reports/tp2-pp1-gpu01-preliminary-queue.json"
    report.write_text(json.dumps({"status": "completed", "samples": results}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "queue_complete", "report": str(report), "samples": len(results)}), flush=True)


if __name__ == "__main__":
    main()
