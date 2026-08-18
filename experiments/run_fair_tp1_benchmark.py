"""Run one compute-only TP=1 benchmark from shared fair-comparison artifacts."""
import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config as cfg
from fair_tp1_contract import CONTRACT, build_mini, sha256_file
from eval.run_megatron_baseline import build_model, init_distributed
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from parallel.process_groups import set_model_parallel


def load_checkpoint(model, path):
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if artifact.get("contract") != CONTRACT:
        raise ValueError("checkpoint contract mismatch: " + str(path))
    model.load_state_dict(artifact["state_dict"], strict=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=("mini", "mcore"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--fused", action="store_true")
    parser.add_argument("--report-losses", action="store_true", help="Include every warmup and measured loss in JSON for a numerical trajectory audit.")
    args = parser.parse_args()
    if args.num_steps < 1 or args.warmup_steps < 0:
        parser.error("invalid step count")
    init_distributed(tp=1, pp=1)
    rank = dist.get_rank()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.manual_seed(20260818)
    model_parallel_cuda_manual_seed(20260818)
    set_model_parallel(None)
    if args.implementation == "mini":
        model, model_config = build_mini(device)
    else:
        model, model_config = build_model(1, 1, use_bf16=args.amp, no_scaled_init=True, fair_config=True)
        model = model.to(device)
    vocab_size = cfg.get_model_config()["vocab_size"]
    load_checkpoint(model, args.checkpoint)
    batches = torch.load(args.data_file, map_location="cpu", weights_only=True)
    if batches.get("contract") != CONTRACT:
        raise ValueError("data contract mismatch: " + str(args.data_file))
    required = args.warmup_steps + args.num_steps
    if batches["input_ids"].size(0) < required:
        raise ValueError("data artifact has fewer batches than requested")
    input_ids = batches["input_ids"][:required].to(device)
    labels = batches["labels"][:required].to(device)
    batch_size, sequence_length = input_ids.shape[1:]
    position_ids = torch.arange(sequence_length, device=device).unsqueeze(0).expand(batch_size, -1)
    attention_mask = torch.triu(torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=device), diagonal=1).unsqueeze(0).unsqueeze(0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.1, betas=(0.9, 0.999), fused=args.fused)
    amp_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if args.amp else contextlib.nullcontext()

    loss_history = []

    def step(index):
        optimizer.zero_grad(set_to_none=True)
        with amp_context:
            if args.implementation == "mini":
                logits = model(input_ids[index])
            else:
                logits = model(input_ids[index], position_ids=position_ids, attention_mask=attention_mask)
            loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, vocab_size),
                                                     labels[index, :, :-1].reshape(-1))
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))
        return loss

    model.train()
    for index in range(args.warmup_steps):
        step(index)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    loss = None
    for index in range(args.warmup_steps, required):
        loss = step(index)
    end.record()
    end.synchronize()
    elapsed_seconds = start.elapsed_time(end) / 1000.0
    if rank == 0:
        report = {"implementation": args.implementation, "contract": CONTRACT,
                  "checkpoint_sha256": sha256_file(args.checkpoint), "data_sha256": sha256_file(args.data_file),
                  "num_steps": args.num_steps, "warmup_steps": args.warmup_steps,
                  "micro_batch_size": batch_size, "sequence_length": sequence_length,
                  "amp": args.amp, "fused": args.fused, "elapsed_seconds": elapsed_seconds,
                  "throughput_tok_s": batch_size * sequence_length * args.num_steps / elapsed_seconds,
                  "peak_memory_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
                  "final_loss": float(loss.item())}
        if args.report_losses:
            report["losses"] = loss_history
        print(json.dumps(report, sort_keys=True))
        print(f"Throughput:      {report['throughput_tok_s']:,.0f} tok/s")
        print(f"Peak memory:     {report['peak_memory_gb']:.2f} GB/GPU")
        print(f"Final loss:      {report['final_loss']:.6f}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
