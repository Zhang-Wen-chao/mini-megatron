"""Megatron-Core baseline with manual training loop (identical to mini-megatron).

All differences between the two frameworks are isolated to the MODEL itself.
Same optimizer, same loss (F.cross_entropy), same data iterator, same loop structure.
"""
import math, os, sys, time, argparse, contextlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg

import torch, torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

MODEL_CONFIG = cfg.get_model_config()


def init_distributed(tp, pp):
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp,
    )


def build_model(tp, pp, use_bf16=False, no_scaled_init=False, fair_config=False):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # If no_scaled_init, use the same init as the input layer (std=0.02)
    # to match mini-megatron's initialization scheme.
    output_layer_init_method = None  # default = scaled_init
    if no_scaled_init:
        def output_layer_init_method(weight):
            torch.nn.init.normal_(weight, mean=0.0, std=0.02)

    config = TransformerConfig(
        num_layers=MODEL_CONFIG["num_layers"],
        hidden_size=MODEL_CONFIG["hidden_size"],
        num_attention_heads=MODEL_CONFIG["num_attention_heads"],
        ffn_hidden_size=MODEL_CONFIG["ffn_hidden_size"],
        use_cpu_initialization=True,
        pipeline_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        bf16=use_bf16, fp16=False,
        sequence_parallel=False,
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        pipeline_model_parallel_comm_backend="nccl",
        tp_comm_overlap=True,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        # mini-megatron's projection, QKV and MLP layers are bias-free.  This
        # flag creates the same parameter contract for semantic-parity audits.
        add_bias_linear=not fair_config,
        output_layer_init_method=output_layer_init_method,
    )
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=MODEL_CONFIG["vocab_size"],
        max_sequence_length=MODEL_CONFIG["max_seq_len"],
        pre_process=(pp_rank == 0),
        post_process=(pp_rank == pp_size - 1),
        position_embedding_type="learned_absolute",
    )
    return model, config


def compute_mfu(seq_len, micro_batch_size, tp, pp, elapsed, num_steps):
    L = MODEL_CONFIG["num_layers"]
    h = MODEL_CONFIG["hidden_size"]
    B = micro_batch_size
    s = seq_len
    V = MODEL_CONFIG["vocab_size"]
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


def make_data_iterator(seq_len, micro_batch_size, vocab_size, preload_input_ids=None, preload_labels=None):
    step_idx = 0 if preload_input_ids is not None else -1
    while True:
        if preload_input_ids is not None and step_idx < preload_input_ids.size(0):
            tokens = preload_input_ids[step_idx].to("cuda")
            labels = preload_labels[step_idx].to("cuda")
            step_idx += 1
        else:
            tokens = torch.randint(0, vocab_size, (micro_batch_size, seq_len), device="cuda")
            labels = tokens.clone()
        yield tokens, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--fused", action="store_true", help="Use fused AdamW (single kernel per step)")
    parser.add_argument("--compile", action="store_true", help="Wrap model in torch.compile")
    parser.add_argument("--data-file", type=str, default=None)
    parser.add_argument("--no-scaled-init", action="store_true", help="Use std=0.02 for output layer (matches input layer init)")
    parser.add_argument("--fair-config", action="store_true",
                        help="Use mini-megatron's bias-free GPT parameter contract; required for parity experiments.")
    args = parser.parse_args()

    tp, pp = args.tp, args.pp
    B, total_steps, warmup = args.micro_batch_size, args.num_steps, args.warmup_steps
    S, V = MODEL_CONFIG["max_seq_len"], MODEL_CONFIG["vocab_size"]
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if args.amp else contextlib.nullcontext()

    preload_input_ids = preload_labels = None
    if args.data_file:
        loaded = torch.load(args.data_file, map_location="cpu", weights_only=True)
        preload_input_ids = loaded["input_ids"]
        preload_labels = loaded["labels"]

    init_distributed(tp, pp)
    model_parallel_cuda_manual_seed(42)

    model, config = build_model(tp, pp, use_bf16=args.amp, no_scaled_init=args.no_scaled_init,
                                fair_config=args.fair_config)
    model.cuda()
    if args.compile:
        model = torch.compile(model)

    rank = dist.get_rank()
    torch.manual_seed(42)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.1, betas=(0.9, 0.999), fused=args.fused)

    loss_fn = lambda step: 0.5 * (1.0 + math.cos(math.pi * max(0, step - warmup) / max(1, total_steps - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, loss_fn)

    data_iter = make_data_iterator(S, B, V, preload_input_ids, preload_labels)

    # Warmup
    for s in range(warmup):
        tokens, labels = next(data_iter)
        tokens, labels = tokens.to(torch.cuda.current_device()), labels.to(torch.cuda.current_device())
        optimizer.zero_grad()
        with amp_ctx:
            logits = model(tokens, position_ids=torch.arange(S, device=tokens.device).unsqueeze(0).expand(B, -1),
                           attention_mask=torch.triu(torch.ones(S, S, dtype=torch.bool, device=tokens.device), diagonal=1).unsqueeze(0).unsqueeze(0))
            loss = torch.nn.functional.cross_entropy(logits.view(-1, V), labels.view(-1))
        loss.backward()
        optimizer.step()
        scheduler.step()
    torch.cuda.synchronize()

    # Benchmark
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    losses = []
    for s in range(total_steps):
        tokens, labels = next(data_iter)
        tokens, labels = tokens.to(torch.cuda.current_device()), labels.to(torch.cuda.current_device())
        optimizer.zero_grad()
        with amp_ctx:
            logits = model(tokens, position_ids=torch.arange(S, device=tokens.device).unsqueeze(0).expand(B, -1),
                           attention_mask=torch.triu(torch.ones(S, S, dtype=torch.bool, device=tokens.device), diagonal=1).unsqueeze(0).unsqueeze(0))
            loss = torch.nn.functional.cross_entropy(logits.view(-1, V), labels.view(-1))
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        if rank == 0 and (s + 1) % args.log_interval == 0:
            print(f"step {s+1:4d} | loss {loss.item():.4f}")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    if rank == 0:
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        throughput = B * S * total_steps / elapsed
        mfu = compute_mfu(S, B, tp, pp, elapsed, total_steps)
        print(f"\n{'='*60}")
        print(f"Megatron-Core Baseline Results (AMP={'BF16' if args.amp else 'FP32'})")
        print(f"{'='*60}")
        print(f"Config:          TP={tp} PP={pp}")
        print(f"Model:           {MODEL_CONFIG['hidden_size']}hid {MODEL_CONFIG['num_layers']}lay {MODEL_CONFIG['num_attention_heads']}head")
        print(f"Micro batch:     {B}  |  Seq len: {S}  |  Steps: {total_steps}")
        print(f"{'-'*60}")
        print(f"Throughput:      {throughput:,.0f} tok/s")
        print(f"Peak memory:     {peak_mem:.2f} GB/GPU")
        print(f"MFU:             {mfu:.2%}")
        if losses:
            print(f"Final loss:      {losses[-1]:.4f}")
        print(f"{'='*60}\n")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
