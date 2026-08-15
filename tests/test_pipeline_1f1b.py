"""1F1B pipeline schedule tests: op ordering + training equivalence.

- build_1f1b_schedule must match Megatron-LM's warmup/steady/drain counts.
- End-to-end: multi-process PP training with the 1F1B schedule (Gloo, CPU)
  must produce the same parameters as a single-process reference run on
  the same data — same math, only the interleaving differs.
"""
import os
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pytest

from model.embedding import Embedding
from model.loss import CrossEntropyLoss
from model.transformer import Decoder, DecoderLayer
from parallel.pipeline_parallel import build_1f1b_schedule, train_pipeline_1f1b

H, HEADS, FFN, L, V, S, B, M = 32, 4, 64, 4, 64, 8, 2, 8


def test_schedule_counts_and_warmup_formula():
    """Each stage runs m forwards + m backwards; first op is a warmup forward."""
    for pp in (2, 3, 4):
        for rank in range(pp):
            ops = build_1f1b_schedule(rank, pp, M)
            fs = [mb for k, mb in ops if k == "F"]
            bs = [mb for k, mb in ops if k == "B"]
            assert fs == list(range(M))          # every microbatch forward once
            assert bs == list(range(M))          # every microbatch backward once
            assert ops[0][0] == "F"
            # Megatron: warmup = pp - rank - 1 (clamped to m)
            warmup = min(pp - rank - 1, M)
            assert [k for k, _ in ops[:warmup]] == ["F"] * warmup


def test_schedule_first_stage_warmup_then_alternating():
    # pp=4, m=8, rank 0: warmup=3 forwards, then 5 alternating pairs, then 3 drain
    ops = build_1f1b_schedule(0, 4, M)
    assert [k for k, _ in ops] == ["F"] * 4 + ["B", "F"] * 4 + ["B"] * 4


def test_schedule_last_stage_alternates():
    ops = build_1f1b_schedule(3, 4, M)
    assert [k for k, _ in ops] == ["F", "B"] * M


def test_schedule_middle_stage_known_sequence():
    # pp=4, m=8, rank 1: warmup=2, then 6 alternating pairs, then 2 drain
    ops = build_1f1b_schedule(1, 4, M)
    assert [k for k, _ in ops] == ["F"] * 3 + ["B", "F"] * 5 + ["B"] * 3


def _make_data(B, S, V, m):
    """Deterministic per-microbatch data: seq[j][i] = (i + j) % V (identity task)."""
    data = []
    for j in range(m):
        seq = torch.tensor([[(i + j) % V for i in range(S)]] * B)
        data.append((seq, seq))
    return data


def _fix_weights(mod):
    """Overwrite all weights with deterministic values (no RNG order dependence)."""
    with torch.no_grad():
        for p in mod.parameters():
            p.copy_((torch.arange(p.numel(), dtype=p.dtype) + 1).reshape(p.shape) / p.numel())


def _build_stage(rank, world):
    """Per-stage modules, mirroring main.py's PP construction."""
    layers_per_stage = L // world
    start = rank * layers_per_stage
    end = start + layers_per_stage if rank < world - 1 else L
    embedding = Embedding(V, H, S) if rank == 0 else None
    decoder_layers = torch.nn.ModuleList(
        [DecoderLayer(H, HEADS, FFN) for _ in range(start, end)]
    )
    ln_f = torch.nn.LayerNorm(H) if rank == world - 1 else None
    lm_head = torch.nn.Linear(H, V, bias=False) if rank == world - 1 else None
    loss_fn = CrossEntropyLoss() if rank == world - 1 else None
    for mod in (embedding, decoder_layers, ln_f, lm_head):
        if mod is not None:
            _fix_weights(mod)
    return embedding, decoder_layers, ln_f, lm_head, loss_fn, start


def _state_dict(rank, world, embedding, decoder_layers, ln_f, lm_head, start):
    state = {}
    if embedding is not None:
        for n, p in embedding.named_parameters():
            state[f"embedding.{n}"] = p.detach().clone()
    for i, layer in enumerate(decoder_layers):
        for n, p in layer.named_parameters():
            state[f"layer{start + i}.{n}"] = p.detach().clone()
    if ln_f is not None:
        for n, p in ln_f.named_parameters():
            state[f"ln_f.{n}"] = p.detach().clone()
    if lm_head is not None:
        for n, p in lm_head.named_parameters():
            state[f"lm_head.{n}"] = p.detach().clone()
    return state


def _worker(rank, world, init_file, out_dir):
    dist.init_process_group("gloo", rank=rank, world_size=world, init_method=f"file://{init_file}")
    embedding, decoder_layers, ln_f, lm_head, loss_fn, start = _build_stage(rank, world)

    params = []
    for mod in (embedding, decoder_layers, ln_f, lm_head):
        if mod is not None:
            params += list(mod.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)

    data = _make_data(B, S, V, M)
    data_iter = iter(data)
    train_pipeline_1f1b(
        embedding, decoder_layers, ln_f, lm_head, loss_fn,
        optimizer, scheduler, data_iter, B, S, V, "cpu",
        M, rank, world, rank, H, tp_size=1, log_interval=1000,
    )

    state = _state_dict(rank, world, embedding, decoder_layers, ln_f, lm_head, start)
    torch.save(state, os.path.join(out_dir, f"state_{rank}.pt"))
    dist.destroy_process_group()


def _run_1f1b_reference(pp):
    """Single-process ground truth: full model, all micro-batches, one optimizer step."""
    embed = Embedding(V, H, S)
    decoder = Decoder(H, HEADS, FFN, L)
    ln_f = torch.nn.LayerNorm(H)
    lm_head = torch.nn.Linear(H, V, bias=False)
    loss_fn = CrossEntropyLoss()
    for mod in (embed, decoder, ln_f, lm_head):
        _fix_weights(mod)

    params = list(embed.parameters()) + list(decoder.parameters()) \
        + list(ln_f.parameters()) + list(lm_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    optimizer.zero_grad()
    loss_mask = torch.ones(B, S)
    for tokens, labels in _make_data(B, S, V, M):
        x = embed(tokens)
        x = decoder(x)
        x = ln_f(x)
        logits = lm_head(x)
        loss = loss_fn(logits, labels, loss_mask)
        loss.backward()
    optimizer.step()
    scheduler.step()

    state = {}
    for n, p in embed.named_parameters():
        state[f"embedding.{n}"] = p.detach().clone()
    for i, layer in enumerate(decoder.layers):
        for n, p in layer.named_parameters():
            state[f"layer{i}.{n}"] = p.detach().clone()
    for n, p in ln_f.named_parameters():
        state[f"ln_f.{n}"] = p.detach().clone()
    for n, p in lm_head.named_parameters():
        state[f"lm_head.{n}"] = p.detach().clone()
    return state


@pytest.mark.parametrize("pp", [2, 4])
def test_1f1b_matches_single_gpu_reference(pp):
    """1F1B training (any pp) must produce the same params as the reference."""
    reference = _run_1f1b_reference(pp)
    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, "rendezvous")
        ctx = mp.get_context("spawn")
        procs = [
            ctx.Process(target=_worker, args=(rank, pp, init_file, tmpdir))
            for rank in range(pp)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        assert all(p.exitcode == 0 for p in procs), "a pipeline process crashed"
        for rank in range(pp):
            stage_state = torch.load(os.path.join(tmpdir, f"state_{rank}.pt"), weights_only=True)
            for name, tensor in stage_state.items():
                assert torch.allclose(tensor, reference[name], atol=1e-6), \
                    f"rank {rank} param {name} diverged from reference"
