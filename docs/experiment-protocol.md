# Credible experiment protocol

This document defines what mini-megatron experiments can claim and how their
evidence must be retained. It applies to every new number in README and in the
benchmark document. Existing tables are legacy evidence: useful context, but not
sufficient on their own for a new conclusion.

## Claim boundaries

Use this wording for a controlled comparison:

> Under the recorded hardware, software, model, precision, batch, sequence
> length, and parallel configuration, mini-megatron achieved the reported
> throughput relative to the repository's Megatron-Core custom-loop baseline.

Do not turn this into “mini-megatron is faster than Megatron”. A custom loop
isolates a narrow implementation path; it does not cover all production
features, large models, multi-node operation, or full training recipes. A
profile explains a bottleneck; it is never a throughput measurement.

## Evidence required before publishing a result

1. Semantic equivalence. With a canonical checkpoint and fixed tokens, compare
   logits, loss, gradients, and parameters after one optimizer update against an
   unpartitioned reference. Record tolerances and precision.
2. Configuration parity. Freeze architecture, initialization/checkpoint,
   inputs/labels/mask, global batch, update count, token budget, TF32/BF16,
   optimizer, scheduler, clipping, and software versions. A differing field
   makes the result non-equivalent and must be reported.
3. Paired repetition. Run at least five independent ABBA or BAAB pairs on an
   idle host. Publish every replicate, mean, median, sample standard deviation,
   min/max, and paired speed ratio. Do not publish only the fastest run.
4. Separate scaling from learning. Report TP/PP/DP scaling and memory separately
   from single-GPU throughput. Use a fixed real tokenized train/validation split
   and held-out PPL for learning claims; identity loss is only a wiring smoke test.
5. Profile separately. Retain raw Nsight reports, SQLite and CSV exports, exact
   commands, and analysis rules. Never quote profile elapsed time as throughput.

## Run bundles and raw Nsight evidence

The run_experiment script executes one command and creates one immutable bundle:

    results/runs/<timestamp>-<name>/
    ├── manifest.json       # scope, commands, timing, metrics, return code
    ├── environment.json    # git state, packages, GPU/topology, CUDA/NCCL env
    ├── command.txt
    ├── stdout.log / stderr.log
    ├── metrics.json
    ├── checksums.sha256
    └── profile/            # only with --profile
        ├── rank0.nsys-rep
        ├── rank0.sqlite
        ├── cuda_gpu_kern_sum.csv
        ├── cuda_api_sum.csv
        ├── cuda_gpu_trace.csv
        └── exports.json

The nsys-rep and sqlite files are source evidence. Put them in Git LFS or an
immutable experiment archive and record its URI, file sizes, and SHA-256 in the
aggregate report. Keep manifests, commands, environment, checksums, CSV, and
human-readable analysis in Git. Never replace a prior run bundle in place.

Pass each fixed input checkpoint, token file, or dataset shard with the artifact
option so its byte size and SHA-256 are recorded in the manifest. This is
stronger than recording only a random seed because framework construction can
consume RNG in different orders. The runner refuses a dirty source tree unless
the explicit allow-dirty option is used; such a run is marked as non-clean in its
manifest and should not be a primary published result.

The kernel-summary analyzer records its CSV SHA-256, regex classification rules,
and all unmatched kernel time in profile/analysis.json. Percentages are kernel
time, not wall-clock time. Never label generic elementwise kernels as unfused
AdamW without an independently auditable attribution rule.

Before trusting a bundle, run:

    python3 experiments/validate_run_bundle.py results/runs/<run-id>

## Execution on the shared L20 host

The host uses the nerdctl container runtime; Docker is not the experiment
entrypoint. The container has a stale configured workdir, so -w / is required:

    ssh l20 'nerdctl exec -w / zhangwenchao-megatron /bin/bash -lc "cd /mnt/storage01/zhangwenchao02/repos/mini-megatron-test && <command>"'

Use NCCL_SHM_DISABLE=1; add CUDA_DEVICE_MAX_CONNECTIONS=1 for multi-GPU runs.
See dev-guides/local-to-l20-handoff.md for the current host handoff.

Before every command, check that the requested GPUs are idle:

    nvidia-smi --query-compute-apps=pid --format=csv,noheader

The runner performs the same preflight and refuses to start when a compute
process is present, unless the active-GPU override is explicitly supplied. An
override run is contamination-prone and must be excluded from controlled results.

Run exactly one command at a time. For a framework comparison, alternate
frameworks (mini -> megatron -> megatron -> mini) instead of batching all runs
of one framework. Each command gets its own bundle.

A bundle with source_tree_clean=false (created via --allow-dirty) is
provisional evidence only. Do not use it as the primary result in README or a
release. Commit the runner and workload first, then repeat the paired run on a
clean tree.

Example benchmark replicate:

    python3 experiments/run_experiment.py --name mini-125m-tp1-fused-r01 \
      --tag variant=mini --tag pair=01 --tag condition=125m-s512-b4-bf16 \
      --artifact artifacts/canonical-125m.pt --artifact data/fixed_tokens.bin -- \
      torchrun --nproc_per_node=1 main.py --tp 1 --pp 1 \
      --num-steps 200 --warmup-steps 30 --micro-batch-size 4 --amp --fused

Example profile, which is not a throughput sample:

    python3 experiments/run_experiment.py --profile \
      --name mini-125m-tp1-fused-profile-r01 -- \
      torchrun --nproc_per_node=1 main.py --tp 1 --pp 1 \
      --num-steps 40 --warmup-steps 10 --micro-batch-size 4 --amp --fused

For multi-rank profiling, start one nsys profile wrapper per CUDA rank with a
unique output prefix. Do not profile only the torchrun launcher and assume its
children were captured.

After at least five complete pairs, create an aggregate without selecting a best
run:

    python3 experiments/summarize_paired_results.py --results-dir results/runs \
      --left mini --right megatron --output results/aggregates/125m-tp1.json
