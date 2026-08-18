"""Create one reproducible benchmark or Nsight Systems run bundle.

Run exactly one command per invocation. Profiling changes timing, so profile
runs are evidence for bottleneck analysis and not throughput samples.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "runs"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def output(command):
    try:
        return subprocess.check_output(command, cwd=ROOT, text=True,
                                       stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return "UNAVAILABLE: " + str(error) + "\n" + getattr(error, "output", "")


def write(path, value):
    Path(path).write_text(value, encoding="utf-8")


def parse_metrics(stdout):
    patterns = {
        "throughput_tok_s": r"Throughput:\s+([\d,]+) tok/s",
        "mfu_percent": r"MFU:\s+([\d.]+)%",
        "peak_memory_gb": r"Peak memory:\s+([\d.]+) GB",
        "final_loss": r"Final loss:\s+([\d.]+)",
    }
    metrics = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, stdout)
        if match:
            metrics[key] = float(match.group(1).replace(",", ""))
    return metrics


def parse_compute_processes(stdout):
    """Normalize nvidia-smi's compute-process listing into non-empty rows."""
    ignored = {"No running processes found", "No running compute processes found"}
    return [line.strip() for line in stdout.splitlines()
            if line.strip() and line.strip() not in ignored]


def gpu_preflight():
    command = ["nvidia-smi", "--query-compute-apps=pid,process_name,gpu_uuid",
               "--format=csv,noheader"]
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    except OSError as error:
        return {"command": command, "return_code": None, "stdout": "",
                "stderr": "UNAVAILABLE: " + str(error), "active_processes": []}
    return {"command": command, "return_code": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr,
            "active_processes": parse_compute_processes(result.stdout) if result.returncode == 0 else []}


def capture_environment(bundle, preflight):
    environment = {
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "git_commit": output(["git", "rev-parse", "HEAD"]),
        "git_status": output(["git", "status", "--short"]),
        "git_diff": output(["git", "diff", "--binary"]),
        "pip_freeze": output([sys.executable, "-m", "pip", "freeze"]),
        "nvidia_smi": output(["nvidia-smi"]),
        "nvidia_topology": output(["nvidia-smi", "topo", "-m"]),
        "gpu_preflight": preflight,
        "nsys_version": output(["nsys", "--version"]),
        "environment_variables": {
            key: value for key, value in os.environ.items()
            if key.startswith(("CUDA", "NCCL", "TORCH", "PYTORCH"))
        },
    }
    write(bundle / "environment.json", json.dumps(environment, indent=2, sort_keys=True) + "\n")
    return environment


def export_profile(nsys, report, profile_dir):
    """Write portable SQLite and CSV exports, preserving command failures."""
    commands = {
        "sqlite": [nsys, "export", "--type", "sqlite", "--force-overwrite", "true",
                   "--output", str(profile_dir / "rank0"), str(report)],
        "cuda_gpu_kern_sum": [nsys, "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv",
                              "--output", str(profile_dir / "cuda_gpu_kern_sum"), str(report)],
        "cuda_api_sum": [nsys, "stats", "--report", "cuda_api_sum", "--format", "csv",
                         "--output", str(profile_dir / "cuda_api_sum"), str(report)],
        "cuda_gpu_trace": [nsys, "stats", "--report", "cuda_gpu_trace", "--format", "csv",
                            "--output", str(profile_dir / "cuda_gpu_trace"), str(report)],
    }
    exports = {}
    for name, command in commands.items():
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        exports[name] = {"command": command, "return_code": result.returncode,
                         "stdout": result.stdout, "stderr": result.stderr}
    write(profile_dir / "exports.json", json.dumps(exports, indent=2) + "\n")
    return exports


def write_checksums(bundle):
    records = []
    for path in sorted(bundle.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            records.append(sha256(path) + "  " + str(path.relative_to(bundle)))
    write(bundle / "checksums.sha256", "\n".join(records) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--nsys-bin", default="nsys")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="Run with uncommitted source changes; recorded in manifest but not reproducible by commit alone.")
    parser.add_argument("--allow-active-gpus", action="store_true",
                        help="Override the idle-GPU gate; this is unsuitable for a controlled benchmark.")
    parser.add_argument("--tag", action="append", default=[], metavar="KEY=VALUE",
                        help="Manifest metadata; use variant=mini or variant=megatron for paired summaries.")
    parser.add_argument("--artifact", action="append", default=[], metavar="PATH",
                        help="Input checkpoint or data file to fingerprint in the manifest; repeat as needed.")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.name):
        parser.error("name may contain only letters, digits, dot, underscore, and hyphen")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("provide a command after --")
    if args.profile and shutil.which(args.nsys_bin) is None:
        parser.error("Nsight Systems executable not found: " + args.nsys_bin)
    dirty_status = output(["git", "status", "--short"])
    if dirty_status and not dirty_status.startswith("UNAVAILABLE:") and not args.allow_dirty:
        parser.error("working tree is dirty; commit first or explicitly pass --allow-dirty")
    preflight = gpu_preflight()
    if preflight["active_processes"] and not args.allow_active_gpus:
        parser.error("GPU preflight found active compute processes; wait for an idle host or explicitly pass --allow-active-gpus: " +
                     "; ".join(preflight["active_processes"]))
    tags = {}
    for item in args.tag:
        key, separator, value = item.partition("=")
        if not separator or not key or not value:
            parser.error("--tag must use KEY=VALUE")
        if key in tags:
            parser.error("duplicate --tag key: " + key)
        tags[key] = value
    artifacts = []
    for item in args.artifact:
        artifact = Path(item).expanduser().resolve()
        if not artifact.is_file():
            parser.error("--artifact is not a readable file: " + str(artifact))
        artifacts.append({"path": str(artifact), "size_bytes": artifact.stat().st_size,
                          "sha256": sha256(artifact)})

    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + args.name
    bundle = Path(args.results_dir).expanduser().resolve() / run_id
    if bundle.exists():
        parser.error("refusing to overwrite bundle: " + str(bundle))
    bundle.mkdir(parents=True)
    environment = capture_environment(bundle, preflight)
    report = None
    executed = command
    if args.profile:
        profile_dir = bundle / "profile"
        profile_dir.mkdir()
        prefix = profile_dir / "rank0"
        report = Path(str(prefix) + ".nsys-rep")
        executed = [args.nsys_bin, "profile", "--force-overwrite", "true",
                    "--trace", "cuda,nvtx,osrt", "--sample", "none",
                    "--output", str(prefix)] + command
    write(bundle / "command.txt", " ".join(executed) + "\n")

    started = time.time()
    result = subprocess.run(executed, cwd=ROOT, text=True, capture_output=True)
    finished = time.time()
    write(bundle / "stdout.log", result.stdout)
    write(bundle / "stderr.log", result.stderr)
    profile_exports = None
    if report and result.returncode == 0 and report.exists():
        profile_exports = export_profile(args.nsys_bin, report, report.parent)
    metrics = parse_metrics(result.stdout)
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "kind": "nsys_profile" if args.profile else "benchmark",
        "claim_scope": "Profiler evidence only; not a throughput sample." if args.profile else
                       "One benchmark replicate; aggregate paired replicates before a claim.",
        "tags": tags,
        "input_artifacts": artifacts,
        "source_tree_clean": not bool(dirty_status),
        "gpu_preflight": preflight,
        "target_command": command,
        "executed_command": executed,
        "started_at_utc": dt.datetime.fromtimestamp(started, dt.timezone.utc).isoformat(),
        "finished_at_utc": dt.datetime.fromtimestamp(finished, dt.timezone.utc).isoformat(),
        "elapsed_wall_seconds": round(finished - started, 6),
        "return_code": result.returncode,
        "metrics": metrics,
        "profile_report": str(report.relative_to(bundle)) if report and report.exists() else None,
        "profile_exports": profile_exports,
        "environment_file": "environment.json",
        "environment_capture_time": environment["captured_at_utc"],
    }
    write(bundle / "metrics.json", json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    write(bundle / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_checksums(bundle)
    print(json.dumps({"bundle": str(bundle), "return_code": result.returncode,
                      "metrics": metrics}, indent=2))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
