"""Save original run outputs for the fused/compile benchmark story.

Reproduces every number quoted in README / docs/nsight-adamw-optimizer-bottleneck.md
and archives the RAW stdout of each run under results/logs/, plus a machine-readable
summary under results/benchmarks_fused_compile.json.

Usage (run at repo root, inside the L20 megatron container):

    NCCL_SHM_DISABLE=1 python3 experiments/save_benchmark_results.py

Outputs:
    results/benchmarks_fused_compile.json   summary (parsed tok/s, MFU, loss)
    results/logs/<config>_<round>.log       raw stdout of every run
"""
import os
import re
import sys
import json
import time
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, "results", "logs")
MINI = os.path.join(ROOT, "main.py")
MEGATRON = os.path.join(ROOT, "eval", "run_megatron_baseline.py")

ENV = {**os.environ, "PYTHONPATH": ROOT, "NCCL_SHM_DISABLE": "1"}


def run(script, extra, name, steps=50, warmup=10, nproc=1):
    cmd = [
        "torchrun", f"--nproc_per_node={nproc}",
        script, "--tp", "1", "--pp", "1",
        "--num-steps", str(steps), "--warmup-steps", str(warmup),
        "--micro-batch-size", "4", "--amp",
    ] + (extra or [])
    if nproc > 1:
        # multi-GPU configs use TP=2 (and PP=2 below)
        cmd[cmd.index("--tp") + 1] = "2"
        if nproc == 4:
            cmd[cmd.index("--pp") + 1] = "2"
    print(f"[{time.strftime('%H:%M:%S')}] {name}: {' '.join(cmd)}")
    start = time.time()
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=1200, env=ENV)
    except subprocess.TimeoutExpired as e:
        print(f"  TIMEOUT after 1200s")
        return {"error": "timeout", "elapsed": 1200}
    elapsed = time.time() - start
    out = res.stdout
    log_path = os.path.join(LOG_DIR, f"{name}.log")
    with open(log_path, "w") as f:
        f.write("CMD: " + " ".join(cmd) + "\n\n" + out)
        if res.stderr:
            f.write("\n\nSTDERR:\n" + res.stderr)
    tok = re.search(r"Throughput:\s+([\d,]+) tok/s", out)
    mfu = re.search(r"MFU:\s+([\d.]+)%", out)
    mem = re.search(r"Peak memory:\s+([\d.]+) GB", out)
    loss = re.search(r"Final loss:\s+([\d.]+)", out)
    entry = {
        "command": " ".join(cmd),
        "elapsed_s": round(elapsed, 2),
        "log": log_path.replace(ROOT + "/", ""),
    }
    if tok:
        entry["tok_s"] = int(tok.group(1).replace(",", ""))
    if mfu:
        entry["mfu"] = float(mfu.group(1))
    if mem:
        entry["peak_mem_gb"] = float(mem.group(1))
    if loss:
        entry["final_loss"] = float(loss.group(1))
    print(f"  -> {entry.get('tok_s', '?')} tok/s, MFU {entry.get('mfu', '?')}%, "
          f"mem {entry.get('peak_mem_gb', '?')} GB, {entry['elapsed_s']}s, log saved")
    return entry


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    results = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment": {
            "gpu": "4x NVIDIA L20 48GB",
            "container": "NGC PyTorch 26.01",
            "cuda": "13.1",
            "torch": "2.10.0a0+nv26.1",
        },
        "model": "125M GPT (12L/768H/12A, B=4, S=512)",
        "precision": "BF16 autocast, TF32 enabled",
        "optimizer": "AdamW lr=6e-4 wd=0.1, cosine, warmup 10",
        "method": "alternating A/B rounds, same-day, identical MFU formula",
        "runs": {},
    }

    # ---- Round 1: mini-megatron, 4 configs ----
    r1 = {}
    r1["mini_amp"] = run(MINI, [], "r1_mini_amp")
    r1["mini_fused"] = run(MINI, ["--fused"], "r1_mini_fused")
    r1["mini_compile"] = run(MINI, ["--compile"], "r1_mini_compile")
    r1["mini_fused_compile"] = run(MINI, ["--fused", "--compile"], "r1_mini_fused_compile")
    # ---- Round 1: Megatron-Core, 2 configs (no compile: deadlocks) ----
    r1["megatron_amp"] = run(MEGATRON, [], "r1_megatron_amp")
    r1["megatron_fused"] = run(MEGATRON, ["--fused"], "r1_megatron_fused")
    results["runs"]["round1"] = r1

    # ---- Round 2: reversed order ----
    r2 = {}
    r2["megatron_fused"] = run(MEGATRON, ["--fused"], "r2_megatron_fused")
    r2["megatron_amp"] = run(MEGATRON, [], "r2_megatron_amp")
    r2["mini_fused_compile"] = run(MINI, ["--fused", "--compile"], "r2_mini_fused_compile")
    r2["mini_compile"] = run(MINI, ["--compile"], "r2_mini_compile")
    r2["mini_fused"] = run(MINI, ["--fused"], "r2_mini_fused")
    r2["mini_amp"] = run(MINI, [], "r2_mini_amp")
    results["runs"]["round2"] = r2

    # ---- Multi-GPU: TP=2 PP=1 and TP=2 PP=2, mini fused vs unfused ----
    mg = {}
    mg["tp2_amp"] = run(MINI, [], "tp2_mini_amp", nproc=2)
    mg["tp2_fused"] = run(MINI, ["--fused"], "tp2_mini_fused", nproc=2)
    mg["tp2pp2_amp"] = run(MINI, [], "tp2pp2_mini_amp", nproc=4)
    mg["tp2pp2_fused"] = run(MINI, ["--fused"], "tp2pp2_mini_fused", nproc=4)
    results["runs"]["multigpu"] = mg

    # ---- Fixed-data loss equivalence: mini amp vs fused vs fused+compile ----
    data_file = os.path.join(ROOT, "experiments", "identity_data.pt")
    if os.path.exists(data_file):
        eq = {}
        for name, extra in [("eq_amp", []), ("eq_fused", ["--fused"]),
                            ("eq_fused_compile", ["--fused", "--compile"])]:
            cmd = [
                "torchrun", "--nproc_per_node=1", MINI,
                "--tp", "1", "--pp", "1",
                "--num-steps", "50", "--warmup-steps", "0",
                "--micro-batch-size", "4", "--amp",
                "--data-file", data_file,
            ] + extra
            log_path = os.path.join(LOG_DIR, f"{name}.log")
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=1200, env=ENV)
            with open(log_path, "w") as f:
                f.write("CMD: " + " ".join(cmd) + "\n\n" + res.stdout)
            loss = re.search(r"Final loss:\s+([\d.]+)", res.stdout)
            entry = {"command": " ".join(cmd), "log": log_path.replace(ROOT + "/", ""),
                     "elapsed_s": round(time.time(), 0)}
            if loss:
                entry["final_loss"] = float(loss.group(1))
            eq[name] = entry
            print(f"[{time.strftime('%H:%M:%S')}] {name}: final_loss={entry.get('final_loss', '?')}")
        results["runs"]["fixed_data_loss"] = eq
    else:
        print("SKIP fixed-data loss (identity_data.pt missing; run synthetic_data.py + make_identity.py)")

    # ---- Summary ratios ----
    def val(d, k):
        return d.get(k) or {}

    r = results["runs"]
    s = results["summary"] = {}
    m1, m2 = r["round1"], r["round2"]
    pairs = ["mini_amp", "mini_fused", "mini_compile", "mini_fused_compile",
             "megatron_amp", "megatron_fused"]
    for p in pairs:
        t1, t2 = m1[p].get("tok_s"), m2[p].get("tok_s")
        if t1 and t2:
            s[p] = {"round1_tok_s": t1, "round2_tok_s": t2, "mean_tok_s": (t1 + t2) // 2}
    if s.get("mini_fused_compile", {}).get("mean_tok_s") and s.get("megatron_amp", {}).get("mean_tok_s"):
        s["mini_best_vs_megatron_default"] = round(
            s["mini_fused_compile"]["mean_tok_s"] / s["megatron_amp"]["mean_tok_s"], 2)
    if s.get("mini_amp", {}).get("mean_tok_s") and s.get("megatron_amp", {}).get("mean_tok_s"):
        s["mini_amp_vs_megatron_amp"] = round(
            s["mini_amp"]["mean_tok_s"] / s["megatron_amp"]["mean_tok_s"], 2)
    if s.get("mini_fused", {}).get("mean_tok_s") and s.get("megatron_fused", {}).get("mean_tok_s"):
        s["mini_fused_vs_megatron_fused"] = round(
            s["mini_fused"]["mean_tok_s"] / s["megatron_fused"]["mean_tok_s"], 2)

    out_path = os.path.join(ROOT, "results", "benchmarks_fused_compile.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")
    print(json.dumps(results.get("summary", {}), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
