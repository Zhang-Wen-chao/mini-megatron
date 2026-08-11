# mini-megatron

> A PyTorch implementation of Megatron-LM's TP + PP + DP + AMP, in **~800 lines** of readable code.
> Demonstrates that 1% of Megatron's code can match or exceed its performance on small models.

## Why?

Megatron-LM is ~300K lines of production code. The core ideas behind its parallelism
strategies are simple. This repo implements **Tensor Parallelism, Pipeline Parallelism,
Data Parallelism, and BF16 Mixed Precision** in ~800 lines of pure PyTorch — covering
the four most-used parallel strategies with 0.3% of the code.

This is **not a production framework**. It's a learning artifact that:
- Works end-to-end (TP/PP/DP/AMP all wired and tested)
- Achieves 1.6-2.4x Megatron-Core's throughput on 125M models
- Fits in one sitting so the entire training loop is readable

For more complete coverage (ZeRO-1, Sequence Parallelism, 1F1B interleaved, etc.),
see [Nano-Megatron](https://github.com/pyy233/Nano-Megatron) (~50K lines, production-grade).

## Quick Start

```bash
# Single GPU, FP32
torchrun --nproc_per_node=1 main.py --tp 1 --pp 1

# Single GPU, BF16 mixed precision (faster, less memory)
torchrun --nproc_per_node=1 main.py --tp 1 --pp 1 --amp

# Single GPU, BF16 + fused AdamW (single kernel optimizer step)
torchrun --nproc_per_node=1 main.py --tp 1 --pp 1 --amp --fused

# TP=2 (2 GPUs)
torchrun --nproc_per_node=2 main.py --tp 2 --pp 1

# TP=2 PP=2 (4 GPUs)
torchrun --nproc_per_node=4 main.py --tp 2 --pp 2

# All combined
torchrun --nproc_per_node=4 main.py --tp 2 --pp 2 --amp
```

### Configuration

Default (125M-parameter GPT model):

| Parameter | Value |
|-----------|-------|
| Layers | 12 |
| Hidden size | 768 |
| Attention heads | 12 |
| Sequence length | 512 |
| Learning rate | 6e-4 |
| Warmup steps | 10 |
| Max steps | 100 |

Override via CLI: `--num-steps`, `--micro-batch-size`, `--warmup-steps`, `--log-interval`, `--amp`, `--fused`.
Edit `config.py` for model architecture changes.

`--fused` uses PyTorch's fused AdamW kernel, which collapses the optimizer step
into a single kernel launch (vs ~9 per-step full-parameter scans by default). On
a 125M model this cuts optimizer GPU time by ~57% and raises MFU from 36.2% to
42.4% (TP=1, BF16, alternating A/B re-runs). Loss curves match the unfused path
(max diff ~1e-4 on fixed data, 200 steps).

## Requirements

- **PyTorch** >= 2.0 (with CUDA)
- **NCCL** (bundled with PyTorch)
- **1-4 NVIDIA GPUs** (tested on L20 48GB)

Install: `pip install -r requirements.txt`

For baseline comparison (`eval/run_megatron_baseline.py`), additionally:
- `megatron-core` (optional, for performance comparison only)

## Architecture

```
                    Data Parallel (DP)
        ┌─────────┬─────────┬─────────┬─────────┐
        │  GPU 0  │  GPU 1  │  GPU 2  │  GPU 3  │
        │ model_0 │ model_0 │ model_0 │ model_0 │  ← each has full model copy
        │ data_0  │ data_1  │ data_2  │ data_3  │  ← different data
        └─────────┴─────────┴─────────┴─────────┘

                Tensor Parallelism (TP)
        ┌──────────────┬──────────────┐
        │    GPU 0     │    GPU 1     │
        │  ColumnParallel  │  RowParallel  │  ← split weight matrices
        │  Linear (A/2)   │  Linear (B/2)  │     across GPUs
        └──────────────┴──────────────┘

                Pipeline Parallelism (PP)
        ┌─────────┐     ┌─────────┐     ┌─────────┐
        │  GPU 0  │ ──▶ │  GPU 1  │ ──▶ │  GPU 2  │
        │ layers  │     │ layers  │     │ layers  │  ← each GPU has
        │  0-3    │     │  4-7    │     │  8-11   │     different layers
        └─────────┘     └─────────┘     └─────────┘
```

## Project Structure

**Core training loop** (wired into `main.py`):

```
mini-megatron/
├── main.py                  # Entry point + TP/PP/DP training loop
├── config.py                # Model & training hyperparameters
├── model/
│   ├── transformer.py       # Attention, DecoderLayer, Decoder
│   ├── embedding.py         # Token + position embeddings
│   └── loss.py              # Cross-entropy loss
├── parallel/
│   ├── tensor_parallel.py   # ColumnParallelLinear, RowParallelLinear
│   ├── pipeline_parallel.py # Serial pipeline schedule + warmup
│   ├── data_parallel.py     # Gradient all-reduce
│   └── process_groups.py    # TP/PP/DP communicator setup
└── comm/
    └── all_reduce.py        # All-reduce primitive (autograd Function)
```

**Reference implementations** (standalone, not wired into training loop):

```
├── parallel/
│   └── distributed_optimizer.py  # ZeRO-1 optimizer state partition
├── comm/
│   ├── send_recv.py         # P2P send/recv for PP
│   ├── sequence_parallel.py # Sequence parallelism primitives
│   └── overlap_*.py         # Communication-computation overlap
└── checkpoint.py            # Save/load model weights

**Evaluation** (optional, requires megatron-core):

```
└── eval/
    ├── compare_loss.py      # Loss comparison with Megatron baseline
    └── run_megatron_baseline.py  # Run official Megatron for comparison
```

## Benchmarks

All tests: 125M model, 4× L20 48GB, 50 steps, B=4, S=512.

### mini-megatron: AMP vs FP32 (50 steps)

```
                FP32                          BF16 (--amp)
TP=1 PP=1 |  34,152 tok/s | 23.90% MFU |  38,873 tok/s | 27.20% MFU
TP=2 PP=1 |  31,048 tok/s | 10.86% MFU |  47,126 tok/s | 16.49% MFU
TP=2 PP=2 |  31,699 tok/s |  5.62% MFU |  32,191 tok/s |  5.70% MFU
```

> AMP gives **1.1-1.5x speedup** on compute-bound configs (TP=1, TP=2). PP=2 sees
> little benefit because communication overhead dominates over compute.

### Fused AdamW optimization (2026-08-11, 50 steps, same conditions)

```
                BF16 (--amp)                 BF16 + fused (--amp --fused)   gain
TP=1 PP=1 |  51,700 tok/s | 36.18% MFU |  60,617 tok/s | 42.42% MFU |  +17.2% tok/s
TP=2 PP=1 |  26,133 tok/s |  9.14% MFU |  28,126 tok/s |  9.84% MFU |   +7.6%
TP=2 PP=2 |  23,186 tok/s |  4.21% MFU |  23,261 tok/s |  4.22% MFU |   +0.3%
```

> `--fused` collapses the AdamW step into a single kernel. On a single GPU it
> cuts optimizer GPU time by ~57% (Nsight Systems: AdamW was 45.2% of kernel
> time before, 26.3% after) and raises MFU from 36.2% to 42.4%.
> The gain shrinks with TP/PP because each rank owns fewer parameters, so the
> optimizer's memory-bandwidth cost no longer dominates.
> Full story + complete test conditions: `docs/nsight-adamw-optimizer-bottleneck.md`.

### vs Megatron-Core (2026-08-11, strict A/B, 50 steps, same MFU formula)

Same day, alternating rounds, BF16, TP=1 PP=1, both frameworks with and without `--fused`:

| Config | mini-megatron | Megatron-Core | mini / Megatron |
|---|---|---|---|
| unfused | 51,841 tok/s (36.3% MFU) | 24,661 tok/s (17.3% MFU) | **2.10x** |
| fused | 60,754 tok/s (42.5% MFU) | 26,641 tok/s (18.6% MFU) | **2.28x** |

> Both scripts use the identical MFU formula and throughput definition
> (`B × S × steps / elapsed`), measured back-to-back in alternating rounds.
> mini-megatron stays ~2.1x faster even when Megatron-Core also enables fused
> AdamW (fused only helps it +8%, because its bottleneck is not the optimizer).

### vs Megatron-Core (2026-07-24, historical)

**FP32:**

| Metric | Megatron-Core TP=1 | mini-megatron TP=1 | mini / Megatron |
|---|---|---|---|
| tok/s | 16,479 | **33,967** | **2.06x** |
| MFU | 11.53% | **23.77%** | **2.06x** |
| Memory | 4.26 GB | 4.52 GB | 0.94x |

| Metric | Megatron-Core TP=2 | mini-megatron TP=2 | mini / Megatron |
|---|---|---|---|
| tok/s | 19,471 | **30,897** | **1.59x** |
| MFU | 6.81% | **10.81%** | **1.59x** |
| Memory | 2.32 GB | 3.63 GB | 0.64x |

**BF16 (`--amp`):**

| Metric | Megatron-Core TP=1 | mini-megatron TP=1 |
|---|---|---|
| tok/s | 16,391 | **38,873** |
| MFU | 11.47% | **27.20%** |
| Memory | 4.26 GB | 3.89 GB |

> Historical numbers (2026-07-24), 30-step comparison table. Absolute values
> differ from 2026-08-11 runs (different measurement conditions), so only
> compare within the same table. Note: both absolute throughputs are higher
> in the 2026-08-11 runs (e.g. Megatron-Core 16.4k -> 24.7k tok/s), likely
> from environment/version changes.

### Compare against Megatron-LM

```bash
# Run Megatron-Core baseline (requires megatron-core)
torchrun --nproc_per_node=1 eval/run_megatron_baseline.py --tp 1 --pp 1

# Compare loss curves
python eval/compare_loss.py
```

## Key Design Decisions

- **No HuggingFace dependency** — pure PyTorch. No transformers, no accelerate.
- **Random data** for benchmarking, no real dataset downloads needed.
- **Serial pipeline schedule** — each stage processes one micro-batch per iteration,
  with warmup to fill the pipeline. Not interleaved 1F1B.
- **BF16 mixed precision** via `torch.autocast` (no loss scaling needed; L20 supports BF16 natively).
- **Gradient accumulation** in PP mode (gradients sum across micro-batches, one
  optimizer step per sweep). No accumulation in non-PP mode.

## Limitations (What's Not Implemented)

To stay under 800 lines, this repo omits:

- **ZeRO-1 optimizer** (code in `parallel/distributed_optimizer.py`, not wired)
- **Sequence Parallelism** (code in `comm/sequence_parallel.py`, not wired)
- **Communication-computation overlap** (code in `comm/overlap_*.py`, not wired)
- **Activation checkpointing** (interface in `model/recompute.py`, not invoked)
- **Distributed data loading** (code in `data/dataset.py`, not used; main.py uses
  inline random data generation)
- **Interleaved 1F1B** (only serial pipeline)
- **CUDA graphs / Flash Attention / fused kernels** (all PyTorch native)
- **FSDP, MoE, CP, EMA, dynamic loss scaling**

These are all in the "reference implementations" directory for reading, but
the main training loop uses only the modules marked ✅ above.

## Comparison with Similar Projects

| Project | Stars | Code size | Coverage |
|---|---|---|---|
| **mini-megatron** (this) | — | ~800 lines | TP + PP + DP + AMP |
| [Tiny-Megatron](https://github.com/liangyuwang/Tiny-Megatron) | 32 | ~3K | TP + DP + 2D |
| [Nano-Megatron](https://github.com/pyy233/Nano-Megatron) | 3 | ~50K | TP+SP+PP+DP+ZeRO+CP (311M TinyStories verified) |

**Choose mini-megatron** if you want to read the entire training loop in one sitting.
**Choose Tiny-Megatron** if you want a clean 2D parallel API.
**Choose Nano-Megatron** if you want a production-grade training framework with full
Megatron coverage.

## Where to Start Reading

1. `main.py` — the entry point. Read `compute_mfu` and the two paths (`if pp > 1` and `else`).
2. `parallel/tensor_parallel.py` — ColumnParallelLinear, RowParallelLinear.
3. `parallel/pipeline_parallel.py` — the serial pipeline schedule.
4. `comm/all_reduce.py` — autograd Function for all-reduce.

Total core implementation: ~800 lines. Reference modules add ~300 more.

## Testing

```bash
pip install -r requirements.txt
pytest                # runs all tests in tests/
```

The test suite runs on CPU (no GPU required) and covers:
- Model component shapes and loss computation
- TP/AllReduce correctness (single-process)
- **End-to-end training**: verifies loss actually decreases on a synthetic task
  (proves forward + backward + optimizer step are all wired correctly)
- Gradient flow to all parameters
- Reference results against 2000-step training on identity task (`results/identity_2000steps.json`)

End-to-end training tests (test_training.py) run on CPU but use CUDA operations
when available — run them on a CUDA-enabled machine for realistic behavior.

## References

- [Megatron-LM (NVIDIA)](https://github.com/NVIDIA/Megatron-LM) — the real thing
- [Megatron-LM Paper (1909.08053)](https://arxiv.org/abs/1909.08053) — original TP paper
- [Efficient Large-Scale Language Model Training on GPU Clusters (2104.04473)](https://arxiv.org/abs/2104.04473) — PP + DP paper

## License

MIT — see [LICENSE](LICENSE).
