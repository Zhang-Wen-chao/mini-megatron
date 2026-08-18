"""Validate TP=1 numerical parity between mini-megatron and Megatron-Core.

This is the gate before any cross-framework throughput benchmark. It builds a
shared bias-free GPT contract, copies every parameter from mini to Megatron,
uses one fixed next-token batch, and compares logits, loss, gradients, and the
parameters after one AdamW update.  It prints a JSON report and returns nonzero
when a declared tolerance is exceeded. Native attention implementations are
allowed a small, documented FP32 reduction difference; initial copied weights
must nevertheless match exactly.
"""
import argparse
import contextlib
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config as cfg
from eval.run_megatron_baseline import build_model, init_distributed
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from model.embedding import Embedding
from model.loss import CrossEntropyLoss
from model.transformer import Decoder, GPT
from parallel.process_groups import set_model_parallel


def _max_abs(left, right):
    return float((left.float() - right.float()).abs().max().item())


def _relative_l2(left, right):
    delta = (left.float() - right.float()).norm()
    reference = left.float().norm().clamp_min(1e-12)
    return float((delta / reference).item())


def _worst_pair(mini_params, mcore_params, pairs, model_config, field):
    values = []
    for left, right in pairs:
        mini_value = mini_params[left].grad if field == "gradient" else mini_params[left]
        mcore_value = mcore_params[right].grad if field == "gradient" else mcore_params[right]
        mcore_value = mcore_in_mini_layout(mcore_value, left, right, model_config)
        values.append({"mini_parameter": left, "mcore_parameter": right,
                       "max_abs": _max_abs(mini_value, mcore_value),
                       "relative_l2": _relative_l2(mini_value, mcore_value)})
    return max(values, key=lambda item: item["max_abs"])


def _digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def build_mini(device):
    model_config = cfg.get_model_config()
    embedding = Embedding(model_config["vocab_size"], model_config["hidden_size"], model_config["max_seq_len"])
    decoder = Decoder(model_config["hidden_size"], model_config["num_attention_heads"],
                      model_config["ffn_hidden_size"], model_config["num_layers"])
    model = GPT(embedding, decoder, torch.nn.LayerNorm(model_config["hidden_size"]),
                torch.nn.Linear(model_config["hidden_size"], model_config["vocab_size"], bias=False),
                CrossEntropyLoss()).to(device)
    return model, model_config


def mappings(model_config):
    pairs = [
        ("embedding.token_embedding.weight", "embedding.word_embeddings.weight"),
        ("embedding.position_embedding.weight", "embedding.position_embeddings.weight"),
        ("ln_f.weight", "decoder.final_layernorm.weight"),
        ("ln_f.bias", "decoder.final_layernorm.bias"),
        ("lm_head.weight", "output_layer.weight"),
    ]
    for index in range(model_config["num_layers"]):
        mini = f"decoder.layers.{index}"
        mcore = f"decoder.layers.{index}"
        pairs.extend([
            (f"{mini}.ln1.weight", f"{mcore}.input_layernorm.weight"),
            (f"{mini}.ln1.bias", f"{mcore}.input_layernorm.bias"),
            (f"{mini}.attn.qkv.weight", f"{mcore}.self_attention.linear_qkv.weight"),
            (f"{mini}.attn.o.weight", f"{mcore}.self_attention.linear_proj.weight"),
            (f"{mini}.ln2.weight", f"{mcore}.pre_mlp_layernorm.weight"),
            (f"{mini}.ln2.bias", f"{mcore}.pre_mlp_layernorm.bias"),
            (f"{mini}.mlp.fc1.weight", f"{mcore}.mlp.linear_fc1.weight"),
            (f"{mini}.mlp.fc2.weight", f"{mcore}.mlp.linear_fc2.weight"),
        ])
    return pairs


def is_qkv_pair(mini_name, mcore_name):
    return mini_name.endswith(".attn.qkv.weight") and mcore_name.endswith(".self_attention.linear_qkv.weight")


def qkv_mini_to_mcore(tensor, model_config):
    """Convert [all Q | all K | all V] rows to MCore's per-group rows.

    For standard MHA at TP=1, MCore stores Q_i, K_i, V_i for each attention
    group; mini stores all Q heads, then all K heads, then all V heads.
    """
    heads = model_config["num_attention_heads"]
    head_dim = model_config["hidden_size"] // heads
    if tensor.dim() != 2:
        raise ValueError("QKV conversion expects a 2-D weight matrix")
    return tensor.view(3, heads, head_dim, tensor.shape[1]).permute(1, 0, 2, 3).reshape_as(tensor)


def qkv_mcore_to_mini(tensor, model_config):
    heads = model_config["num_attention_heads"]
    head_dim = model_config["hidden_size"] // heads
    if tensor.dim() != 2:
        raise ValueError("QKV conversion expects a 2-D weight matrix")
    return tensor.view(heads, 3, head_dim, tensor.shape[1]).permute(1, 0, 2, 3).reshape_as(tensor)


def mcore_in_mini_layout(tensor, mini_name, mcore_name, model_config):
    return qkv_mcore_to_mini(tensor, model_config) if is_qkv_pair(mini_name, mcore_name) else tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--atol", type=float, default=3e-4)
    parser.add_argument("--rtol", type=float, default=3e-4)
    parser.add_argument("--max-logits-relative-l2", type=float, default=5e-4)
    parser.add_argument("--max-gradient-relative-l2", type=float, default=5e-4)
    parser.add_argument("--max-parameter-relative-l2", type=float, default=1e-4)
    parser.add_argument("--amp", action="store_true", help="Validate the BF16 autocast execution path separately from FP32.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    init_distributed(tp=1, pp=1)
    rank = dist.get_rank()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.manual_seed(args.seed)
    model_parallel_cuda_manual_seed(args.seed)
    set_model_parallel(None)
    mini, model_config = build_mini(device)
    mcore, _ = build_model(1, 1, use_bf16=args.amp, no_scaled_init=True, fair_config=True)
    mcore = mcore.to(device)
    mini_params, mcore_params = dict(mini.named_parameters()), dict(mcore.named_parameters())
    pairs = mappings(model_config)
    unmapped_mini = sorted(set(mini_params) - {left for left, _ in pairs})
    unmapped_mcore = sorted(set(mcore_params) - {right for _, right in pairs})
    if unmapped_mini or unmapped_mcore:
        raise RuntimeError("parameter mapping is incomplete: mini=" + repr(unmapped_mini) + " mcore=" + repr(unmapped_mcore))
    with torch.no_grad():
        for mini_name, mcore_name in pairs:
            if mini_params[mini_name].shape != mcore_params[mcore_name].shape:
                raise RuntimeError("shape mismatch: " + mini_name + " vs " + mcore_name)
            source = qkv_mini_to_mcore(mini_params[mini_name], model_config) if is_qkv_pair(mini_name, mcore_name) else mini_params[mini_name]
            mcore_params[mcore_name].copy_(source)

    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    # Generate on CPU from an explicit generator, then transfer once.  This is
    # deterministic across CUDA RNG state and mirrors the future fixed artifact.
    input_ids = torch.randint(0, model_config["vocab_size"],
                              (args.batch_size, model_config["max_seq_len"]), generator=generator,
                              dtype=torch.long).to(device)
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    labels[:, -1] = -100
    loss_mask = (labels != -100).float()
    position_ids = torch.arange(model_config["max_seq_len"], device=device).unsqueeze(0).expand(args.batch_size, -1)
    attention_mask = torch.triu(torch.ones(model_config["max_seq_len"], model_config["max_seq_len"],
                                            dtype=torch.bool, device=device), diagonal=1).unsqueeze(0).unsqueeze(0)

    mini_optimizer = torch.optim.AdamW(mini.parameters(), lr=6e-4, weight_decay=0.1, betas=(0.9, 0.999), fused=False)
    mcore_optimizer = torch.optim.AdamW(mcore.parameters(), lr=6e-4, weight_decay=0.1, betas=(0.9, 0.999), fused=False)
    mini_optimizer.zero_grad(set_to_none=True)
    mcore_optimizer.zero_grad(set_to_none=True)
    initial_parameter_worst = _worst_pair(mini_params, mcore_params, pairs, model_config, "parameter")
    amp_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if args.amp else contextlib.nullcontext()
    with amp_context:
        mini_logits, mini_loss = mini(input_ids, labels, loss_mask)
        mcore_logits = mcore(input_ids, position_ids=position_ids, attention_mask=attention_mask)
        mcore_loss = torch.nn.functional.cross_entropy(mcore_logits[:, :-1].reshape(-1, model_config["vocab_size"]),
                                                       labels[:, :-1].reshape(-1))
    mini_loss.backward()
    mcore_loss.backward()
    gradient_worst = _worst_pair(mini_params, mcore_params, pairs, model_config, "gradient")
    gradient_max_abs = gradient_worst["max_abs"]
    logits_max_abs = _max_abs(mini_logits[:, :-1], mcore_logits[:, :-1])
    loss_abs = abs(float(mini_loss.item()) - float(mcore_loss.item()))
    mini_optimizer.step()
    mcore_optimizer.step()
    parameter_worst = _worst_pair(mini_params, mcore_params, pairs, model_config, "parameter")
    parameter_max_abs = parameter_worst["max_abs"]
    logits_relative_l2 = _relative_l2(mini_logits[:, :-1], mcore_logits[:, :-1])
    initial_exact = initial_parameter_worst["max_abs"] == 0.0
    passed = (initial_exact and loss_abs <= args.atol
              and logits_relative_l2 <= args.max_logits_relative_l2
              and gradient_worst["relative_l2"] <= args.max_gradient_relative_l2
              and parameter_worst["relative_l2"] <= args.max_parameter_relative_l2)
    report = {
        "schema_version": 1,
        "contract": "12L-768H-12head-3072FFN, learned absolute position, pre-LN, GELU, no dropout, bias-free linears",
        "amp": args.amp,
        "seed": args.seed,
        "batch_shape": list(input_ids.shape),
        "input_sha256": _digest(input_ids),
        "mapped_parameters": len(pairs),
        "mini_parameter_count": sum(p.numel() for p in mini.parameters()),
        "mcore_parameter_count": sum(p.numel() for p in mcore.parameters()),
        "logits_max_abs": logits_max_abs,
        "logits_relative_l2": logits_relative_l2,
        "loss_abs": loss_abs,
        "gradient_max_abs": gradient_max_abs,
        "worst_gradient": gradient_worst,
        "worst_initial_parameter": initial_parameter_worst,
        "parameter_max_abs_after_one_adamw_step": parameter_max_abs,
        "worst_parameter_after_one_adamw_step": parameter_worst,
        "atol": args.atol,
        "rtol": args.rtol,
        "max_logits_relative_l2": args.max_logits_relative_l2,
        "max_gradient_relative_l2": args.max_gradient_relative_l2,
        "max_parameter_relative_l2": args.max_parameter_relative_l2,
        "passed": bool(passed),
    }
    if rank == 0:
        rendered = json.dumps(report, indent=2, sort_keys=True)
        print(rendered)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n")
    dist.destroy_process_group()
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
