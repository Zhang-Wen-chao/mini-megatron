# 2026-08-17 L20 experiment evidence ledger

This is an auditable record of the current L20 experiment session. It does not
replace the [experiment protocol](experiment-protocol.md), and its benchmark
numbers are **provisional**: the experiment tooling had not yet been committed
when the runs were made, so every manifest records source_tree_clean=false.
They are useful corroborating evidence, but are not release-quality claims.

## Environment and completed checks

- Host/container: L20 host, zhangwenchao-megatron (nerdctl), 4 x NVIDIA L20;
  NGC PyTorch 26.01, CUDA 13.1, PyTorch 2.10, Megatron-Core 0.18.
- Historical profiling used BF16. The current fair comparison uses a 125M GPT,
  TP=1/PP=1, FP32, micro-batch 4, sequence 512, 30 warm-up, and 200 measured
  steps.
- Test suite: python3 -m pytest -q in that container completed with
  **38 passed in 11.51s** after the fair-contract QKV conversion test was added.
  This is the full suite, rather than a Mac-only
  partial check.
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

## Fair TP=1 FP32 comparison (2026-08-17, provisional)

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
- FP32 equivalence gate passed: initial mapped weights have exact zero
  difference; logits relative L2 is 3.5498e-4, gradients 3.5508e-4, and
  post-one-step parameters 6.3139e-5, within declared limits of 5e-4, 5e-4,
  and 1e-4. A 25-update fixed-sequence smoke test ended at loss 11.144128
  (mini) vs 11.146294 (MCore), a 0.019% difference.

Five idle-GPU, ABBA/BAAB-style pairs used those artifacts, standard unfused
AdamW, and FP32. Every bundle passed checksum validation; aggregate:

| Metric | mini | Megatron-Core | mini / MCore |
|---|---:|---:|---:|
| Mean throughput | 32,657 tok/s | 27,642.6 tok/s | **1.181404x** |
| Median throughput | 32,654 tok/s | 27,673 tok/s | **1.180517x** |
| Sample standard deviation | 29.93 | 60.55 | 0.001972 |
| Range | 32,622-32,699 | 27,577-27,708 | 1.179417-1.184017 |
| Peak allocated memory | 4.52 GB | 5.13 GB | — |

The aggregate is results/aggregates/fair-tp1-fp32-unfused-provisional.json; the
ten source bundles are 20260817T161633Z through 20260817T162209Z with the
fair-tp1-fp32-unfused name. It remains provisional because the source tree was
dirty (source_tree_clean=false), but it is a materially fairer result than the
old 2.289x diagnostic: it supports only this exact bias-free TP=1 FP32 contract,
not production, BF16, larger-model, multi-GPU, or training-quality claims.

BF16 is explicitly excluded from the fair performance claim. Under the same
initial weights and one batch, its logits relative L2 was 5.3204e-3, the worst
gradient relative L2 was 7.0706e-2, and the declared FP32 parity gate failed.
The two implementations may still be compared as a same-initial-condition BF16
performance observation, but not as semantically equivalent training until that
numerical difference is resolved.

## Nsight Systems source evidence

Raw reports are retained on the L20 project archive (not committed to Git):
/mnt/storage01/zhangwenchao02/repos/mini-megatron-test/results/runs/. Both
bundles passed experiments/validate_run_bundle.py; their manifests, command
logs, CSV exports, checksums, and analyzer output are alongside the reports.

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

## Promoting this session to a publishable result

1. Commit the runner, validators, analyzer, and documentation first.
2. The TP=1 FP32 canonical conversion and fixed batch now exist. Commit them,
   rerun their equivalence gate and the five pairs on a clean source tree, then
   extend the conversion and parity gate to TP/PP before making scaling claims.
3. Re-capture the two profiles, validate every bundle, and publish the new
   aggregate only when its manifests show source_tree_clean=true.
4. Preserve the raw reports in the immutable archive or Git LFS and record
   their checksums in the resulting ledger.
