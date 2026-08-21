"""Run a fixed-artifact TP=2 FP32 training benchmark.

Each optimizer update accumulates an explicit number of micro-batches.  This
keeps global tokens/update stable for the later PP configurations and makes the
timed mini/MCore loops identical outside their model implementations.
"""
import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config as cfg
from eval.run_megatron_baseline import build_model, init_distributed
from fair_tp1_contract import CONTRACT, build_mini
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from parallel.process_groups import init_model_parallel, set_model_parallel


def load_checkpoint(model, path):
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if artifact.get("contract") != CONTRACT or artifact.get("topology") != {"tp": 2, "pp": 1, "dp": 1}:
        raise ValueError("checkpoint contract/topology mismatch: " + str(path))
    model.load_state_dict(artifact["state_dict"], strict=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=("mini", "mcore"), required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--num-updates", type=int, default=200)
    parser.add_argument("--warmup-updates", type=int, default=30)
    parser.add_argument("--microbatches-per-update", type=int, default=8)
    parser.add_argument("--report-losses", action="store_true")
    parser.add_argument("--phase-timing-output", type=Path,
                        help="Optional JSON output for low-perturbation CUDA Event phase totals.")
    parser.add_argument("--progress-interval", type=int, default=0,
                        help="Rank-0 progress cadence in optimizer updates; 0 disables progress logs.")
    args = parser.parse_args()
    if args.num_updates < 1 or args.warmup_updates < 0 or args.microbatches_per_update < 1:
        parser.error("invalid update or microbatch count")
    init_distributed(2, 1)
    rank = dist.get_rank()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    model_parallel_cuda_manual_seed(20260821)
    mini_mpu = init_model_parallel(2, 1)
    set_model_parallel(mini_mpu)
    if args.implementation == "mini":
        model, _ = build_mini(device)
    else:
        model, _ = build_model(2, 1, no_scaled_init=True, fair_config=True)
        model = model.to(device)
    load_checkpoint(model, args.artifact_dir / f"{args.implementation}_tp2_rank{rank}.pt")
    data = torch.load(args.artifact_dir / "next_token_batches.pt", map_location="cpu", weights_only=True)
    if data.get("contract") != CONTRACT:
        raise ValueError("data contract mismatch")
    required = (args.warmup_updates + args.num_updates) * args.microbatches_per_update
    if data["input_ids"].size(0) < required:
        raise ValueError("fixed artifact has fewer batches than requested")
    input_ids, labels = data["input_ids"][:required], data["labels"][:required]
    batch_size, sequence_length = input_ids.shape[1:]
    vocab_size = cfg.get_model_config()["vocab_size"]
    positions = torch.arange(sequence_length, device=device).unsqueeze(0).expand(batch_size, -1)
    attention = torch.triu(torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=device), diagonal=1).unsqueeze(0).unsqueeze(0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.1, betas=(0.9, 0.999), fused=False)
    losses = []
    phase_names = ("forward", "loss_vocab", "backward", "optimizer")
    # Do not time warmup.  CUDA Event records are deliberately enabled only
    # for the measured interval, so the reported event count has the same
    # denominator as ``num_updates``.
    phase_events = None

    def mark_phase(name, callback):
        if phase_events is None:
            return callback()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        result = callback()
        end_event.record()
        phase_events[name].append((start_event, end_event))
        return result

    def update(update_index):
        optimizer.zero_grad(set_to_none=True)
        last_loss = None
        base = update_index * args.microbatches_per_update
        for microbatch in range(args.microbatches_per_update):
            tokens = input_ids[base + microbatch].to(device, non_blocking=True)
            target = labels[base + microbatch].to(device, non_blocking=True)
            if args.implementation == "mini":
                logits = mark_phase("forward", lambda: model(tokens))
            else:
                logits = mark_phase("forward", lambda: model(tokens, position_ids=positions, attention_mask=attention))
            last_loss = mark_phase(
                "loss_vocab",
                lambda: torch.nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, vocab_size), target[:, :-1].reshape(-1),
                ),
            )
            mark_phase("backward", lambda: (last_loss / args.microbatches_per_update).backward())
        mark_phase("optimizer", optimizer.step)
        losses.append(float(last_loss.item()))
        return last_loss

    model.train()
    for update_index in range(args.warmup_updates):
        update(update_index)
    torch.cuda.synchronize()
    if args.phase_timing_output:
        phase_events = {name: [] for name in phase_names}
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    loss = None
    for update_index in range(args.warmup_updates, args.warmup_updates + args.num_updates):
        loss = update(update_index)
        completed = update_index - args.warmup_updates + 1
        if rank == 0 and args.progress_interval and (
                completed % args.progress_interval == 0 or completed == args.num_updates):
            elapsed_so_far = time.perf_counter() - start
            print(json.dumps({
                "event": "progress",
                "implementation": args.implementation,
                "completed_updates": completed,
                "total_updates": args.num_updates,
                "elapsed_seconds": elapsed_so_far,
                "latest_loss": float(loss.item()),
            }, sort_keys=True), flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    phase_report = None
    if phase_events is not None:
        phase_report = {
            name: {
                "milliseconds_total": sum(begin.elapsed_time(end) for begin, end in records),
                "event_pairs": len(records),
            }
            for name, records in phase_events.items()
        }
        phase_report["rank"] = rank
        phase_report["num_updates"] = args.num_updates
        phase_report["microbatches_per_update"] = args.microbatches_per_update
        gathered_phase_reports = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered_phase_reports, phase_report)
        if rank == 0:
            output = args.phase_timing_output.resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({
                "schema_version": 1,
                "purpose": "CUDA Event phase timing; profiler-free diagnostic, not a throughput substitute",
                "implementation": args.implementation,
                "topology": {"tp": 2, "pp": 1, "dp": 1},
                "contract": CONTRACT,
                "ranks": gathered_phase_reports,
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if rank == 0:
        report = {"implementation": args.implementation, "contract": CONTRACT,
                  "topology": {"tp": 2, "pp": 1, "dp": 1}, "precision": "fp32",
                  "artifact_dir": str(args.artifact_dir.resolve()), "num_updates": args.num_updates,
                  "warmup_updates": args.warmup_updates, "microbatches_per_update": args.microbatches_per_update,
                  "micro_batch_size": batch_size, "sequence_length": sequence_length,
                  "tokens_per_update": batch_size * sequence_length * args.microbatches_per_update,
                  "elapsed_seconds": elapsed,
                  "throughput_tok_s": batch_size * sequence_length * args.microbatches_per_update * args.num_updates / elapsed,
                  "peak_memory_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
                  "final_loss": float(loss.item())}
        if args.report_losses:
            report["losses"] = losses
        print(json.dumps(report, sort_keys=True))
        print(f"Throughput:      {report['throughput_tok_s']:,.0f} tok/s")
        print(f"Peak memory:     {report['peak_memory_gb']:.2f} GB/GPU")
        print(f"Final loss:      {report['final_loss']:.6f}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
