"""Create an immutable multi-rank Nsight bundle for the PP custom-loop study.

This script deliberately does not emit a throughput number.  It profiles a
short, fixed diagnostic run and stores one `.nsys-rep` per CUDA rank, portable
SQLite/CSV exports, command, environment, input SHA-256 and recursive checksums.
It must be run only while the selected GPUs are idle.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "evidence/fair-125m-parallel-20260821"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command):
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    return {"command": command, "return_code": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def nvidia_compute_processes():
    result = run(["nvidia-smi", "--query-compute-apps=pid,process_name,gpu_uuid", "--format=csv,noheader"])
    ignored = {"No running processes found", "No running compute processes found"}
    result["active_processes"] = [line.strip() for line in result["stdout"].splitlines()
                                  if line.strip() and line.strip() not in ignored]
    return result


def artifacts(artifact_dir):
    paths = sorted(path for path in artifact_dir.iterdir() if path.is_file())
    return [{"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in paths]


def checksums(bundle):
    lines = []
    for path in sorted(bundle.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            lines.append(sha256(path) + "  " + str(path.relative_to(bundle)))
    (bundle / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=("mini", "mcore"), required=True)
    parser.add_argument("--tp", type=int, choices=(1, 2), required=True)
    parser.add_argument("--pp", type=int, choices=(2,), required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--master-port", type=int, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--warmup-updates", type=int, default=2)
    parser.add_argument("--num-updates", type=int, default=12)
    parser.add_argument("--microbatches-per-update", type=int, default=8)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    physical = tuple(value.strip() for value in args.gpus.split(",") if value.strip())
    if len(physical) != args.tp * args.pp or len(set(physical)) != len(physical):
        parser.error("--gpus must list TP*PP unique CUDA device ids")
    if not args.name or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for char in args.name):
        parser.error("--name may contain only letters, digits, dot, underscore and hyphen")
    if min(args.warmup_updates, args.num_updates, args.microbatches_per_update) < 1:
        parser.error("warmup, measured updates and microbatches must be positive")
    if shutil.which("nsys") is None:
        parser.error("nsys is unavailable")
    dirty = run(["git", "status", "--short"])["stdout"]
    if dirty and not args.allow_dirty:
        parser.error("working tree is dirty; use --allow-dirty to preserve this fact in the bundle")
    preflight = nvidia_compute_processes()
    if preflight["return_code"] != 0:
        raise RuntimeError("nvidia-smi preflight failed: " + preflight["stderr"].strip())
    if preflight["active_processes"]:
        raise RuntimeError("refusing to profile active GPUs: " + "; ".join(preflight["active_processes"]))
    topology = f"tp{args.tp}-pp{args.pp}-dp1"
    artifact_dir = CAMPAIGN / "artifacts" / f"tp{args.tp}-pp{args.pp}-fp32-v1"
    if not artifact_dir.is_dir():
        raise RuntimeError("missing fixed artifact directory: " + str(artifact_dir))
    bundle = CAMPAIGN / "profiles" / args.name
    if bundle.exists():
        raise RuntimeError("refusing to overwrite profile bundle: " + str(bundle))
    bundle.mkdir(parents=True)
    profile_dir = bundle / "profile"
    environment = {
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": run(["git", "rev-parse", "HEAD"])["stdout"].strip(),
        "git_status": dirty,
        "git_diff": run(["git", "diff", "--binary"])["stdout"],
        "nvidia_smi": run(["nvidia-smi"])["stdout"],
        "nvidia_topology": run(["nvidia-smi", "topo", "-m"])["stdout"],
        "nsys_version": run(["nsys", "--version"])["stdout"],
        "gpu_preflight": preflight,
        "environment_variables": {key: value for key, value in os.environ.items()
                                  if key.startswith(("CUDA", "NCCL", "TORCH", "PYTORCH"))},
    }
    (bundle / "environment.json").write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    command = ["torchrun", "--master_port", str(args.master_port), "--nproc_per_node", str(args.tp * args.pp),
               "experiments/profile_rank_with_nsys.py", "--profile-dir", str(profile_dir.relative_to(ROOT)), "--",
               sys.executable, "-u", "experiments/run_fair_pp_benchmark.py", "--implementation", args.implementation,
               "--tp", str(args.tp), "--pp", str(args.pp), "--artifact-dir", str(artifact_dir.relative_to(ROOT)),
               "--warmup-updates", str(args.warmup_updates), "--num-updates", str(args.num_updates),
               "--microbatches-per-update", str(args.microbatches_per_update)]
    (bundle / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": args.gpus, "CUDA_DEVICE_MAX_CONNECTIONS": "1", "NCCL_SHM_DISABLE": "1"})
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, env=env, check=False)
    (bundle / "stdout.log").write_text(result.stdout, encoding="utf-8")
    (bundle / "stderr.log").write_text(result.stderr, encoding="utf-8")
    exports = {}
    if result.returncode == 0:
        for rank in range(args.tp * args.pp):
            report = profile_dir / f"rank{rank}.nsys-rep"
            if not report.is_file():
                raise RuntimeError("missing rank-local Nsight report: " + str(report))
            exports[f"rank{rank}"] = {
                "sqlite": run(["nsys", "export", "--type", "sqlite", "--force-overwrite", "true",
                               "--output", str(profile_dir / f"rank{rank}"), str(report)]),
                "cuda_gpu_kern_sum": run(["nsys", "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv",
                                            "--output", str(profile_dir / f"rank{rank}-cuda_gpu_kern_sum"), str(report)]),
                "cuda_api_sum": run(["nsys", "stats", "--report", "cuda_api_sum", "--format", "csv",
                                     "--output", str(profile_dir / f"rank{rank}-cuda_api_sum"), str(report)]),
            }
    (profile_dir / "exports.json").write_text(json.dumps(exports, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1, "kind": "multi_rank_nsys_profile",
        "claim_scope": "Profiler evidence only; not a throughput sample.",
        "topology": topology, "implementation": args.implementation,
        "physical_gpus": args.gpus, "profiled_cuda_ranks": args.tp * args.pp,
        "source_tree_clean": not bool(dirty), "input_artifacts": artifacts(artifact_dir),
        "target_command": command, "return_code": result.returncode,
        "profile_reports": [f"profile/rank{rank}.nsys-rep" for rank in range(args.tp * args.pp)],
        "environment_file": "environment.json", "profile_exports": "profile/exports.json",
        "warmup_updates": args.warmup_updates, "measured_updates": args.num_updates,
        "microbatches_per_update": args.microbatches_per_update,
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksums(bundle)
    print(json.dumps({"bundle": str(bundle), "return_code": result.returncode,
                      "reports": manifest["profile_reports"]}, indent=2))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
