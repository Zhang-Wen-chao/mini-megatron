import math
import os
import time
import argparse
import contextlib
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import AdamW

import config as cfg
from model.embedding import Embedding
from model.transformer import DecoderLayer, Decoder, GPT
from model.loss import CrossEntropyLoss
from parallel.process_groups import init_model_parallel, set_model_parallel
from parallel.pipeline_parallel import train_pipeline
from parallel.data_parallel import allreduce_grads


def make_data_iterator(batch_size, seq_len, vocab_size, preloaded_data=None):
    """Yield (tokens, labels) pairs.
    
    Without data: tokens = random, labels = tokens (identity task).
    With data: tokens = preloaded input_ids, labels = preloaded labels.
    """
    if preloaded_data is not None:
        tokens_data, labels_data = preloaded_data
        for i in range(tokens_data.size(0)):
            yield tokens_data[i].contiguous(), labels_data[i].contiguous()
    while True:
        tok = torch.randint(0, vocab_size, (batch_size, seq_len))
        yield tok, tok


def make_lr_lambda(warmup_steps, num_steps):
    """Cosine schedule with linear warmup. num_steps is the absolute total."""
    def get_lr(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(num_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(progress * math.pi))
    return get_lr


def compute_mfu(model, seq_len, micro_batch_size, tp, pp, elapsed, num_steps):
    L = model["num_layers"]
    h = model["hidden_size"]
    B = micro_batch_size
    s = seq_len
    V = model["vocab_size"]
    tokens = B * s
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    dp_w = max(1, ws // (tp * pp))
    gpu_world = tp * pp * dp_w

    attn_proj = 24 * h * h
    mlp = 48 * h * h
    logits = 6 * h * V
    token_linear = (attn_proj + mlp) * L + logits
    core_attn = 6 * h * L * B

    flops_per_step = (token_linear * tokens + core_attn * s * s)
    total_flops = flops_per_step * num_steps * dp_w
    mfu = total_flops / (elapsed * 110e12 * gpu_world)
    return min(mfu, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-steps", type=int, default=cfg.MAX_TRAIN_STEPS)
    parser.add_argument("--warmup-steps", type=int, default=cfg.WARMUP_STEPS)
    parser.add_argument("--micro-batch-size", type=int, default=cfg.MICRO_BATCH_SIZE)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--amp", action="store_true", help="Enable BF16 mixed precision (autocast)")
    parser.add_argument("--fused", action="store_true", help="Use fused AdamW (single kernel per step)")
    parser.add_argument("--compile", action="store_true", help="Wrap model in torch.compile (fusion + fewer launches)")
    parser.add_argument("--data-file", type=str, default=None, help="Path to pre-generated .pt data (overrides random data)")
    args = parser.parse_args()

    cfg.enable_tf32()
    tp, pp = args.tp, args.pp

    # Load pre-generated data if specified
    preload_data = None
    if args.data_file:
        loaded = torch.load(args.data_file, map_location="cpu", weights_only=True)
        preload_data = (loaded["input_ids"], loaded["labels"])
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if args.amp else contextlib.nullcontext()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    mpu = init_model_parallel(tp, pp)
    set_model_parallel(mpu)

    torch.manual_seed(42)

    config = cfg.get_model_config()
    B = args.micro_batch_size
    S = config["max_seq_len"]
    V = config["vocab_size"]
    HS = config["hidden_size"]
    NH = config["num_attention_heads"]
    FFN = config["ffn_hidden_size"]
    NL = config["num_layers"]

    # Build PP model
    if pp > 1:
        pp_rank = mpu["pp_rank"]
        pp_size = mpu["pp_size"]
        layers_per_stage = NL // pp
        start = pp_rank * layers_per_stage
        end = start + layers_per_stage if pp_rank < pp_size - 1 else NL

        embedding = Embedding(V, HS, S).to(device) if pp_rank == 0 else None

        decoder_layers = nn.ModuleList([
            DecoderLayer(HS, NH, FFN).to(device)
            for _ in range(start, end)
        ])

        ln_f = nn.LayerNorm(HS).to(device) if pp_rank == pp_size - 1 else None
        lm_head = nn.Linear(HS, V, bias=False).to(device) if pp_rank == pp_size - 1 else None
        loss_fn = CrossEntropyLoss().to(device) if pp_rank == pp_size - 1 else None

        all_params = []
        if embedding is not None:
            all_params += list(embedding.parameters())
        all_params += list(decoder_layers.parameters())
        if ln_f is not None:
            all_params += list(ln_f.parameters())
        if lm_head is not None:
            all_params += list(lm_head.parameters())
        optimizer = AdamW(all_params, lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY, fused=args.fused)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, make_lr_lambda(args.warmup_steps, args.num_steps)
        )

        dp_rank = mpu["dp_rank"] if mpu else 0
        torch.manual_seed(42 + dp_rank * 10000)
        data_iter = make_data_iterator(B, S, V, preload_data)

        dp_group = mpu.get("dp_group")
        elapsed = train_pipeline(
            embedding, decoder_layers, ln_f, lm_head, loss_fn,
            optimizer, scheduler, data_iter, B, S, V, device,
            args.num_steps,
            pp_rank, pp, rank, HS, tp, args.log_interval, dp_group, amp_ctx
        )

        if pp_rank == 0:
            mfu = compute_mfu(config, S, B, tp, pp, elapsed, args.num_steps)
            print(f"MFU:             {mfu:.2%}")

    else:
        # Single GPU / TP only
        embedding = Embedding(V, HS, S).to(device)
        decoder = Decoder(HS, NH, FFN, NL).to(device)
        loss_fn = CrossEntropyLoss()
        ln_f = nn.LayerNorm(HS).to(device)
        lm_head = nn.Linear(HS, V, bias=False).to(device)
        model = GPT(embedding, decoder, ln_f, lm_head, loss_fn).to(device)
        if args.compile:
            model = torch.compile(model)

        optimizer = AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY, fused=args.fused)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, make_lr_lambda(args.warmup_steps, args.num_steps)
        )

        dp_rank = mpu["dp_rank"] if mpu else 0
        torch.manual_seed(42 + dp_rank * 10000)
        data_iter = make_data_iterator(B, S, V, preload_data)

        for step in range(args.warmup_steps):
            tokens, labels = next(data_iter)
            tokens, labels = tokens.to(device), labels.to(device)
            optimizer.zero_grad()
            with amp_ctx:
                _, loss = model(tokens, labels, torch.ones(B, S, dtype=torch.float32, device=device))
            loss.backward()
            allreduce_grads(model, mpu.get("dp_group"))
            optimizer.step()
            scheduler.step()
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        losses = []
        for step in range(args.num_steps):
            tokens, labels = next(data_iter)
            tokens, labels = tokens.to(device), labels.to(device)
            optimizer.zero_grad()
            with amp_ctx:
                _, loss = model(tokens, labels, torch.ones(B, S, dtype=torch.float32, device=device))
            loss.backward()
            allreduce_grads(model, mpu.get("dp_group"))
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            if rank == 0 and (step + 1) % args.log_interval == 0:
                print(f"step {step+1:4d} | loss {loss.item():.4f}")

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        if rank == 0:
            peak_mem = torch.cuda.max_memory_allocated() / 1024**3
            throughput = B * S * args.num_steps / elapsed
            mfu = compute_mfu(config, S, B, tp, pp, elapsed, args.num_steps)
            print(f"\n{'='*60}")
            print(f"Mini-Megatron Results (TP={tp}, PP={pp}, AMP={'BF16' if args.amp else 'FP32'})")
            print(f"{'='*60}")
            print(f"Model:           {HS}hid {NL}lay {NH}head")
            print(f"Micro batch:     {B}  |  Seq len: {S}  |  Steps: {args.num_steps}")
            print(f"{'-'*60}")
            print(f"Throughput:      {throughput:,.0f} tok/s")
            print(f"Peak memory:     {peak_mem:.2f} GB/GPU")
            print(f"MFU:             {mfu:.2%}")
            print(f"Final loss:      {losses[-1]:.4f}")
            print(f"{'='*60}\n")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
