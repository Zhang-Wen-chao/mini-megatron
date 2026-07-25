"""Smoke-test: run mini-megatron and Megatron-Core side by side and print losses.

Note: losses will not match exactly because each framework generates its own
random data independently. This script is a sanity check that both run
end-to-end; for true loss equivalence, both would need to consume the
same pre-generated data batch.
"""
import os
import subprocess
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV = {**os.environ, "PYTHONPATH": ROOT}


def run_mini(tp=1, pp=1, steps=20, warmup=5, bs=4):
    main_script = os.path.join(ROOT, "main.py")
    cmd = f"torchrun --nproc_per_node={tp*pp} {main_script} --num-steps {steps} --warmup-steps {warmup} --micro-batch-size {bs} --tp {tp} --pp {pp}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300, env=ENV)
    losses = re.findall(r"step\s+\d+\s+\|\s+loss\s+([\d.]+)", result.stdout)
    return [float(l) for l in losses]


def run_baseline(tp=1, pp=1, steps=20, warmup=5, bs=4):
    nproc = tp * pp
    baseline_script = os.path.join(ROOT, "eval", "run_megatron_baseline.py")
    cmd = f"torchrun --nproc_per_node={nproc} {baseline_script} --tp {tp} --pp {pp} --num-steps {steps} --warmup-steps {warmup} --micro-batch-size {bs}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300, env=ENV)
    losses = re.findall(r"step\s+\d+\s+\|\s+loss\s+([\d.]+)", result.stdout)
    return [float(l) for l in losses]


def compare():
    for tp, pp, name in [(1, 1, "TP=1"), (2, 1, "TP=2")]:
        print(f"\n=== {name} ===")
        mini = run_mini(tp, pp)
        base = run_baseline(tp, pp)
        if not mini or not base:
            print(f"  SKIP: one run failed")
            continue
        for i, (m, b) in enumerate(zip(mini, base)):
            print(f"  step {i+1:4d}: mini={m:.4f}  base={b:.4f}  diff={abs(m-b):.4f}")
        print(f"  (Note: diff expected — random data is generated independently)")


if __name__ == "__main__":
    compare()
