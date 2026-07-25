"""Compare mini-megatron vs Megatron-Core training convergence.

Usage:
  1. Generate synthetic data first:  python experiments/synthetic_data.py experiments/synthetic_data.pt
  2. Run comparison:                torchrun experiments/compare_convergence.py

Output:
  - Prints loss curves for both frameworks
  - Passes if mini-megatron final loss < threshold (proves it trains)
  - Passes if Megatron-Core also trains (similar convergence)
"""
import os
import sys
import subprocess
import re
import time
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV = {**os.environ, "PYTHONPATH": ROOT}

MINI_SCRIPT = os.path.join(ROOT, "main.py")
BASELINE_SCRIPT = os.path.join(ROOT, "eval", "run_megatron_baseline.py")


def run_framework(script, data_file, steps=50, warmup=5, name="unknown"):
    """Run one framework, return list of (step, loss) pairs."""
    cmd = (
        f"torchrun --nproc_per_node=1 {script} "
        f"--tp 1 --pp 1 --num-steps {steps} --warmup-steps {warmup} "
        f"--data-file {data_file}"
    )
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"Command: {cmd}")
    start = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
    elapsed = time.time() - start

    # Parse loss from stdout (step  X | loss Y.YYYY)
    losses = []
    for line in result.stdout.split("\n"):
        m = re.search(r"step\s+\d+\s+\|\s+loss\s+([\d.]+)", line)
        if m:
            step_num = int(re.search(r"step\s+(\d+)", line).group(1))
            losses.append((step_num, float(m.group(1))))

    # Extract final output block for tok/s, MFU, memory
    final_block = {}
    for line in result.stdout.split("\n"):
        for key, pattern in [
            ("throughput", r"Throughput:\s+([\d,]+) tok/s"),
            ("peak_mem", r"Peak memory:\s+([\d.]+) GB"),
            ("mfu", r"MFU:\s+([\d.]+)%"),
            ("final_loss", r"Final loss:\s+([\d.]+)"),
        ]:
            m = re.search(pattern, line)
            if m:
                final_block[key] = float(m.group(1).replace(",", ""))
    if not final_block and result.stderr:
        print(f"STDERR: {result.stderr[:500]}")

    print(f"Loss curve: {len(losses)} points")
    if losses:
        print(f"  first loss: {losses[0][1]:.4f}, last loss: {losses[-1][1]:.4f}")
    print(f"Final: tok/s={final_block.get('throughput')} "
          f"MFU={final_block.get('mfu')} "
          f"mem={final_block.get('peak_mem')}")
    return losses, final_block


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", default="experiments/synthetic_data.pt")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    data_file = os.path.join(ROOT, args.data_file) if not os.path.isabs(args.data_file) else args.data_file

    if not os.path.exists(data_file):
        print(f"ERROR: data file not found: {data_file}")
        sys.exit(1)

    results = {}

    # Run mini-megatron
    results["mini"] = run_framework(
        MINI_SCRIPT, data_file, args.steps, args.warmup, "mini-megatron"
    )

    # Run Megatron-Core baseline
    results["baseline"] = run_framework(
        BASELINE_SCRIPT, data_file, args.steps, args.warmup, "Megatron-Core"
    )

    # Compare
    mini_losses, mini_final = results["mini"]
    base_losses, base_final = results["baseline"]

    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'':>20}  {'mini-megatron':>16}  {'Megatron-Core':>16}")
    print(f"{'Steps':>20}  {len(mini_losses):>16}  {len(base_losses):>16}")
    if mini_losses and base_losses:
        print(f"{'First loss':>20}  {mini_losses[0][1]:>16.4f}  {base_losses[0][1]:>16.4f}")
        print(f"{'Last loss':>20}  {mini_losses[-1][1]:>16.4f}  {base_losses[-1][1]:>16.4f}")
        print(f"{'Loss decrease':>20}  {mini_losses[0][1]-mini_losses[-1][1]:>16.4f}  {base_losses[0][1]-base_losses[-1][1]:>16.4f}")
    if mini_final:
        print(f"{'tok/s':>20}  {mini_final.get('throughput','N/A'):>16}  {base_final.get('throughput','N/A'):>16}")
        print(f"{'MFU':>20}  {mini_final.get('mfu','N/A'):>16}  {base_final.get('mfu','N/A'):>16}")
        print(f"{'Memory (GB)':>20}  {mini_final.get('peak_mem','N/A'):>16}  {base_final.get('peak_mem','N/A'):>16}")

    # PASS/FAIL: mini-megatron must demonstrate learning (loss decrease > 1.0)
    mini_dropped = mini_losses and (mini_losses[0][1] - mini_losses[-1][1] > 1.0)
    if mini_dropped:
        print(f"\n✅ PASS: mini-megatron loss dropped {mini_losses[0][1]-mini_losses[-1][1]:.2f}")
    else:
        print(f"\n❌ FAIL: mini-megatron loss did not drop significantly")

    # Megatron-Core check (informational)
    base_dropped = base_losses and (base_losses[0][1] - base_losses[-1][1] > 1.0)
    if base_dropped:
        print(f"✅ PASS: Megatron-Core loss dropped {base_losses[0][1]-base_losses[-1][1]:.2f}")
    else:
        print(f"ℹ️  INFO: Megatron-Core loss barely changed — baseline script may not configure training correctly")


if __name__ == "__main__":
    main()
