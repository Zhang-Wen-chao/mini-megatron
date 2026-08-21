"""Run a fixed-artifact fair PP training loop for mini-megatron or MCore.

Both implementations use this explicit non-interleaved 1F1B P2P schedule:
one optimizer update is exactly ``microbatches_per_update`` forward/backward
microbatches, whose losses are divided by that count before one AdamW step on
every pipeline stage.  It is deliberately a *matching custom-loop* comparison,
not a claim about MCore's full training stack.
"""
import argparse
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
from create_fair_pp_artifacts import MiniPipelineStage
from eval.run_megatron_baseline import build_model, init_distributed
from fair_tp1_contract import CONTRACT
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from parallel.pipeline_parallel import build_1f1b_schedule
from parallel.process_groups import init_model_parallel, set_model_parallel


def load_checkpoint(model, path, topology):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("contract") != CONTRACT or checkpoint.get("topology") != topology:
        raise ValueError("checkpoint contract/topology mismatch: " + str(path))
    model.load_state_dict(checkpoint["state_dict"], strict=True)


def cpu_state_dict(model):
    return {name: value.detach().cpu().contiguous() if torch.is_tensor(value) else value
            for name, value in model.state_dict().items()}


def cpu_grad_dict(model):
    """Persist gradients separately so a failed PP gate is diagnosable."""
    return {name: value.grad.detach().cpu().contiguous()
            for name, value in model.named_parameters() if value.grad is not None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=("mini", "mcore"), required=True)
    parser.add_argument("--tp", type=int, choices=(1, 2), required=True)
    parser.add_argument("--pp", type=int, choices=(2,), required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--num-updates", type=int, default=200)
    parser.add_argument("--warmup-updates", type=int, default=30)
    parser.add_argument("--microbatches-per-update", type=int, default=8)
    parser.add_argument("--batch-offset-microbatches", type=int, default=0,
                        help="Fixed-batch offset used only for independent numerical-gate windows.")
    parser.add_argument("--progress-interval", type=int, default=0)
    parser.add_argument("--report-losses", action="store_true")
    parser.add_argument("--state-output-dir", type=Path,
                        help="Optional rank-local post-run state checkpoint directory for numerical gates.")
    parser.add_argument("--gradient-output-dir", type=Path,
                        help="Optional rank-local gradient checkpoint directory before the final AdamW step.")
    parser.add_argument("--logits-output-dir", type=Path,
                        help="Optional last-stage logits/labels for an explicit PP numerical gate.")
    args = parser.parse_args()
    if (args.num_updates < 1 or args.warmup_updates < 0
            or args.microbatches_per_update < 1 or args.batch_offset_microbatches < 0):
        parser.error("invalid update or microbatch count")

    init_distributed(args.tp, args.pp)
    rank = dist.get_rank()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    model_parallel_cuda_manual_seed(20260821)
    mpu = init_model_parallel(args.tp, args.pp)
    set_model_parallel(mpu)
    topology = {"tp": args.tp, "pp": args.pp, "dp": 1}
    config = cfg.get_model_config()
    if args.implementation == "mini":
        model = MiniPipelineStage(config, mpu["pp_rank"], args.pp).to(device)
    else:
        model, _ = build_model(args.tp, args.pp, no_scaled_init=True, fair_config=True)
        model = model.to(device)
    load_checkpoint(model, args.artifact_dir /
                    f"{args.implementation}_tp{args.tp}_pp{args.pp}_rank{rank}.pt", topology)
    data = torch.load(args.artifact_dir / "next_token_batches.pt", map_location="cpu", weights_only=True)
    if data.get("contract") != CONTRACT:
        raise ValueError("data contract mismatch")
    required = args.batch_offset_microbatches + (args.warmup_updates + args.num_updates) * args.microbatches_per_update
    if data["input_ids"].size(0) < required:
        raise ValueError("fixed artifact has fewer batches than requested")
    input_ids = data["input_ids"][args.batch_offset_microbatches:required]
    labels = data["labels"][args.batch_offset_microbatches:required]
    batch_size, sequence_length = input_ids.shape[1:]
    positions = torch.arange(sequence_length, device=device).unsqueeze(0).expand(batch_size, -1)
    attention = torch.triu(torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=device), diagonal=1).unsqueeze(0).unsqueeze(0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.1, betas=(0.9, 0.999), fused=False)
    is_first = mpu["pp_rank"] == 0
    is_last = mpu["pp_rank"] == args.pp - 1
    previous_rank, next_rank = rank - args.tp, rank + args.tp
    losses = []

    def forward(tokens, activation):
        if args.implementation == "mini":
            if is_first:
                value = model.embedding(tokens)
            else:
                value = activation
            for layer in model.decoder_layers:
                value = layer(value)
            if is_last:
                value = model.lm_head(model.ln_f(value))
            return value
        if not is_first:
            model.set_input_tensor(activation)
        return model(tokens, position_ids=positions, attention_mask=attention)

    def run_update(update_index):
        optimizer.zero_grad(set_to_none=True)
        saved_inputs, saved_outputs, received_grads = {}, {}, {}
        outstanding = []
        last_loss = None
        last_logits = None
        last_labels = None
        for operation, microbatch in build_1f1b_schedule(mpu["pp_rank"], args.pp, args.microbatches_per_update):
            index = update_index * args.microbatches_per_update + microbatch
            if operation == "F":
                tokens = input_ids[index].to(device, non_blocking=True)
                if is_first:
                    activation = None
                else:
                    # MCore's P2P tensor is [S,B,H]; mini's is [B,S,H].
                    shape = (sequence_length, batch_size, config["hidden_size"]) if args.implementation == "mcore" else (batch_size, sequence_length, config["hidden_size"])
                    activation = torch.empty(shape, device=device).requires_grad_(True)
                    dist.recv(activation, src=previous_rank)
                output = forward(tokens, activation)
                saved_inputs[microbatch] = activation if not is_first else output
                if is_last:
                    target = labels[index].to(device, non_blocking=True)
                    last_loss = torch.nn.functional.cross_entropy(
                        output[:, :-1].reshape(-1, config["vocab_size"]), target[:, :-1].reshape(-1)
                    )
                    last_logits = output
                    last_labels = target
                    saved_outputs[microbatch] = last_loss
                else:
                    saved_outputs[microbatch] = output
                    outstanding.append((dist.isend(output.contiguous(), dst=next_rank), output))
            else:
                stage_input = saved_inputs[microbatch]
                if is_last:
                    (saved_outputs[microbatch] / args.microbatches_per_update).backward()
                else:
                    stage_output = saved_outputs[microbatch]
                    grad = torch.empty_like(stage_output)
                    dist.recv(grad, src=next_rank)
                    stage_output.backward(grad)
                if not is_first:
                    # The first-stage saved input is its output activation; other stages retain input grad.
                    grad_input = stage_input.grad
                    outstanding.append((dist.isend(grad_input.contiguous(), dst=previous_rank), grad_input))
                saved_inputs.pop(microbatch)
                saved_outputs.pop(microbatch)
        for request, tensor in outstanding:
            request.wait()
        # Retain the last measured update's gradients for the numerical gate.
        if args.gradient_output_dir and update_index == args.warmup_updates + args.num_updates - 1:
            args.gradient_output_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"schema_version": 1, "contract": CONTRACT, "topology": topology,
                        "implementation": args.implementation, "rank": rank,
                        "gradients": cpu_grad_dict(model)},
                       args.gradient_output_dir / f"{args.implementation}_rank{rank}.pt")
        optimizer.step()
        return last_loss, last_logits, last_labels

    model.train()
    for update in range(args.warmup_updates):
        run_update(update)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    loss = logits = target = None
    for update in range(args.warmup_updates, args.warmup_updates + args.num_updates):
        loss, logits, target = run_update(update)
        if is_last and mpu["tp_rank"] == 0:
            losses.append(float(loss.item()))
            completed = update - args.warmup_updates + 1
            if args.progress_interval and (completed % args.progress_interval == 0 or completed == args.num_updates):
                print(json.dumps({"event": "progress", "implementation": args.implementation,
                                  "completed_updates": completed, "total_updates": args.num_updates,
                                  "elapsed_seconds": time.perf_counter() - started,
                                  "latest_loss": float(loss.item())}, sort_keys=True), flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if args.logits_output_dir and is_last and mpu["tp_rank"] == 0:
        args.logits_output_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"schema_version": 1, "contract": CONTRACT, "topology": topology,
                    "implementation": args.implementation, "rank": rank,
                    "logits": logits.detach().cpu().contiguous(),
                    "labels": target.detach().cpu().contiguous()},
                   args.logits_output_dir / f"{args.implementation}_last_stage.pt")
    if args.state_output_dir:
        args.state_output_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"schema_version": 1, "contract": CONTRACT, "topology": topology,
                    "implementation": args.implementation, "rank": rank,
                    "state_dict": cpu_state_dict(model)},
                   args.state_output_dir / f"{args.implementation}_rank{rank}.pt")
    if is_last and mpu["tp_rank"] == 0:
        report = {"implementation": args.implementation, "contract": CONTRACT, "topology": topology,
                  "precision": "fp32", "artifact_dir": str(args.artifact_dir.resolve()),
                  "num_updates": args.num_updates, "warmup_updates": args.warmup_updates,
                  "batch_offset_microbatches": args.batch_offset_microbatches,
                  "microbatches_per_update": args.microbatches_per_update,
                  "micro_batch_size": batch_size, "sequence_length": sequence_length,
                  "tokens_per_update": batch_size * sequence_length * args.microbatches_per_update,
                  "elapsed_seconds": elapsed,
                  "throughput_tok_s": batch_size * sequence_length * args.microbatches_per_update * args.num_updates / elapsed,
                  "peak_memory_gb": torch.cuda.max_memory_allocated(device) / 1024 ** 3,
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
