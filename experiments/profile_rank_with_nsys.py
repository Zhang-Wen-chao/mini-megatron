"""Launch one distributed CUDA rank under its own Nsight Systems capture.

``torchrun`` creates the actual CUDA processes.  Wrapping only its launcher can
miss those children, so the PP profiling runner invokes this wrapper once per
rank.  The output prefix includes ``LOCAL_RANK`` and can never collide.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--nsys-bin", default="nsys")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("provide the rank command after --")
    rank = os.environ.get("LOCAL_RANK")
    if rank is None:
        parser.error("LOCAL_RANK is required; run this wrapper through torchrun")
    args.profile_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.profile_dir / f"rank{rank}"
    executed = [args.nsys_bin, "profile", "--force-overwrite", "true",
                "--trace", "cuda,nvtx,osrt", "--sample", "none",
                "--output", str(prefix)] + command
    return subprocess.run(executed, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
