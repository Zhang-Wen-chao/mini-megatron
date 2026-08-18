# 2026-08-17/18 L20 experiment evidence ledger

This is an auditable record of the L20 experiment sessions. It does not replace
the [experiment protocol](experiment-protocol.md). The original 2026-08-17
samples remain useful historical corroboration but are provisional because their
manifests record `source_tree_clean=false`. The current conclusion below comes
from the 2026-08-18 clean-tree rerun and supersedes those samples.

## Environment and completed checks

- Host/container: L20 host, zhangwenchao-megatron (nerdctl), 4 x NVIDIA L20;
  NGC PyTorch 26.01, CUDA 13.1, PyTorch 2.10, Megatron-Core 0.18.
- Historical profiling used BF16. The current fair comparison uses a 125M GPT,
  TP=1/PP=1, FP32, micro-batch 4, sequence 512, 30 warm-up, and 200 measured
  steps.
- Clean rerun source commit: `ad82d7edf9a1a11f61672aae492697cf15434b85`
  (the L20-history equivalent of local source commit `f402fea`).
- Test suite: `python3 -m pytest -q` in that container completed with
  **38 passed in 11.28s** before the clean rerun. This is the full suite,
  rather than a Mac-only partial check.
- Every sample cited below passed the runner's idle-GPU preflight. An earlier
  active-GPU sample was retained in its own bundle and excluded.

## Paired throughput evidence

Five alternating pairs were recorded for each comparison. All figures are
tokens/s; sd is sample standard deviation. The complete aggregates are kept on
L20 as results/aggregates/.

| Comparison / condition | Left result | Right result | Paired ratio | What it supports |
|---|---:|---:|---:|---|
| mini fused vs Megatron-Core custom loop, 125m-s512-b4-bf16-fused-200x30 | mini mean 60,670, median 60,702, sd 75.58 | baseline mean 26,503, median 26,541, sd 130.18 | mini/baseline mean **2.289x**, range 2.272-2.305 | Diagnostic only: a paired measurement of two non-equivalent scripts, not a framework-performance result. |
| mini fused vs mini unfused, mini-125m-s512-b4-bf16-200x30 | fused mean 60,675.8, median 60,674, sd 34.07 | unfused mean 51,823.4, median 51,829, sd 67.07 | fused/unfused mean **1.1708x** (+17.08%), range 1.1692-1.1727 | A repeatable local throughput gain from the fused optimizer path. |

The first comparison is not strict same-weight/semantic-model equivalence and
is now explicitly withdrawn as a performance claim. It used separately
initialized models, separately generated random identity-token batches, and
different native forward graphs; it therefore measures the two scripts' total
step cost, not an apples-to-apples framework difference. It does **not**
establish that mini-megatron is faster than Megatron generally, nor does it
cover larger models, multi-node jobs, production features, or training quality.
The fused/unfused random-data losses were close at the last step, but that is
not a substitute for the fixed-input equivalence protocol.

## Fair TP=1 FP32 comparison (2026-08-18, clean tree)

This replaced the invalid cross-framework claim above with a deliberately
shared model contract:

- 12 layers, hidden 768, 12 heads, FFN 3072, learned absolute positions,
  pre-LayerNorm, GELU, no dropout, bias-free QKV/projection/MLP linears, and
  causal next-token cross entropy.
- mini and Megatron-Core each contain 162,633,216 parameters. All 101 parameter
  tensors are mapped; QKV is explicitly transposed between mini's all-Q then
  all-K then all-V layout and Megatron-Core's per-head Q_i then K_i then V_i
  layout.
- Immutable L20 artifact directory:
  /mnt/storage01/zhangwenchao02/repos/mini-megatron-test/artifacts/fair-tp1-20260818-v2/.
  It contains framework-specific checkpoints derived from the same weights and
  230 fixed next-token batches. SHA-256: mini checkpoint
  93d634e9e45699266633946a5f9436f08369f160360fc6c229a71c8fdad47619;
  MCore checkpoint
  2c7609572698f0cbb660b62a3ddb51fb49e3730b27b74a626ac532c61b250222;
  batch artifact 3e0f943f435db4bac634f57af2b6a6a21f0f6ecbe2c0f15d5c7efecaa49cabc5.
- The clean-tree FP32 equivalence gate passed: initial mapped weights have exact
  zero difference; logits relative L2 is 3.5498e-4, worst gradient relative L2
  is 3.5508e-4, and post-one-step parameter relative L2 is 6.3139e-5, within
  declared limits of 5e-4, 5e-4, and 1e-4.

Five idle-GPU, ABBA/BAAB-style pairs used those artifacts, standard unfused
AdamW, FP32, 30 warm-up steps, and 200 measured steps. Every bundle passed
checksum validation and records `source_tree_clean=true`; aggregate:

| Metric | mini | Megatron-Core | mini / MCore |
|---|---:|---:|---:|
| Mean throughput | 32,669.6 tok/s | 27,704.8 tok/s | **1.179204x** |
| Median throughput | 32,664 tok/s | 27,716 tok/s | **1.178825x** |
| Sample standard deviation | 17.34 | 24.39 | 0.001047 |
| Range | 32,654-32,690 | 27,672-27,731 | 1.178079-1.180554 |
| Peak allocated memory | 4.52 GB | 5.13 GB | — |

The immutable aggregate is
`results/aggregates/fair-tp1-fp32-unfused-clean-ad82d7e.json` (SHA-256
`132cda05cc47c0042759eb053380df31b08efbd7b704989e47e8881ef731a213`).
Its ten source bundles are in `results/runs-clean-ad82d7e/`, pairs 01–05. This
supports one deliberately narrow statement: **on this L20, for this shared
bias-free 125M GPT contract at TP=1/PP=1 in FP32 with standard AdamW, mini
achieved 1.179204x the throughput of the matching Megatron-Core path.** It does
not support claims about default MCore, BF16, fused optimizers, larger models,
TP/PP/multi-GPU scaling, production workloads, or training quality.

The prior dirty-tree aggregate is retained as a historical reproducibility
record, not combined with this result. The old 2.289x cross-framework BF16
measurement remains a non-equivalent diagnostic and is not a performance claim.

BF16 is explicitly excluded from the fair performance claim. Under the same
initial weights and one batch, its logits relative L2 was 5.3204e-3, the worst
gradient relative L2 was 7.0706e-2, and the declared FP32 parity gate failed.
The two implementations may still be compared as a same-initial-condition BF16
performance observation, but not as semantically equivalent training until that
numerical difference is resolved.

## Nsight Systems source evidence

The two clean-tree profile bundles are versioned in this repository under
`results/runs-clean-ad82d7e/`; `git pull` retrieves their raw `.nsys-rep`,
SQLite, CSV exports, manifests, commands, checksums, and conservative analyzer
output. An identical copy remains in the L20 experiment archive at
`/mnt/storage01/zhangwenchao02/repos/mini-megatron-test/results/`. Profile
timing is not included in throughput statistics.

The clean-tree profiles use the same checkpoints/batches and FP32 contract as
the throughput study, with 10 warm-up and 20 measured steps:

| Bundle | rank0.nsys-rep | rank0.sqlite | kernel time | Conservative kernel-time split | SHA-256 (.nsys-rep / .sqlite) |
|---|---:|---:|---:|---|---|
| `20260818T003611Z-fair-tp1-fp32-clean-profile-mini` | 3.8 MB | 12 MB | 1.8362 s | GEMM 61.54%, copy/cast 4.09%, unclassified 34.36% | `52ff06a0d9099f5b879a3c1fab72b96027fece3e2293e1c994f4b6c37296c902` / `cde5afc1de50c83d48dbddb49d6e951ad364ddd615fbb5b697d9bfea45279ec0` |
| `20260818T003641Z-fair-tp1-fp32-clean-profile-mcore` | 4.2 MB | 13 MB | 2.0363 s | GEMM 50.18%, copy/cast 10.51%, unclassified 39.31% | `48780b2bb1579d316598f834620951dfbe71af6d986d0e470249905961b59520` / `2f37c97a9748072965f55f5cf2d2d2a30d20a0d597da84c2c5c94afdb26a94fe` |

These are kernel-time descriptions, not a causal proof of the throughput gap:
the classifier intentionally leaves unmatched kernels unclassified and makes no
unfused-AdamW attribution from generic elementwise names.

Historical BF16 profiles remain preserved separately:

| Bundle | rank0.nsys-rep | rank0.sqlite | SHA-256 (.nsys-rep / .sqlite) |
|---|---:|---:|---|
| 20260817T153432Z-mini-125m-tp1-bf16-unfused-profile-p01 | 5.3 MB | 15 MB | 6353186078149f68ba871d50b15f8b5ca0c1e427a26083ea222ff4fa4eae2fc1 / 535ac642f093e66c440f621200bbd6a983c8db142f592d04f6f8d26a95e89b25 |
| 20260817T153506Z-mini-125m-tp1-bf16-fused-profile-p01 | 2.9 MB | 8.1 MB | c131f1fda66b3b8308f41b9437ea02154631b576746962f9f5140457d5feead5 / c32591aa2a4fe82f107fe548e64f3c043c65ffe5b7f997e7b533788bc222092b |

The conservative CSV analyzer reports 1.9083 s total GPU-kernel time for the
unfused capture (GEMM 42.53%, copy/cast 10.28%, unclassified 47.19%). It does
not attribute generic elementwise kernels to AdamW. In the fused capture it
identifies the explicit fused AdamW kernel: 432.31 ms, 450 invocations, 26.32%
of 1.6428 s total kernel time (GEMM 49.52%, copy/cast 12.00%, unclassified
12.16%). These are kernel-time shares, not wall-clock shares.

## What remains before broader claims

1. Extend shared-weight conversion and numerical parity gates to TP/PP before
   making any scaling claim.
2. Resolve the BF16 parity failure before presenting a BF16 cross-framework
   result as semantically equivalent.
3. Repeat the protocol for larger models and production-relevant configurations
   if those scopes need conclusions.
4. Preserve raw reports in the immutable archive or Git LFS and keep their
   checksums in this ledger.
