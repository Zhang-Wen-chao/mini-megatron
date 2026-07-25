# Contributing to mini-megatron

Thanks for your interest in contributing! This is a small learning project,
so the bar is low — any improvement is welcome.

## Repository Layout

```
mini-megatron/
├── main.py                       # Training entry point (TP/PP/DP/AMP paths)
├── config.py                    # Model + training hyperparameters
├── model/                        # Embedding, Decoder, GPT, loss
├── parallel/                     # TP, PP, DP, process groups, ZeRO-1 (reference)
├── comm/                         # AllReduce, P2P, sequence-parallel, overlap (reference)
├── eval/                         # Megatron-Core baseline + loss comparison script
├── experiments/                  # Synthetic data generation + comparison runner
├── tests/                        # 26 pytest tests (model, parallel, training, results)
├── results/                      # Reference results (identity_2000steps.json)
├── checkpoint.py                 # Save/load (reference, not wired)
└── README.md
```

## Running Tests

```bash
pip install -r requirements.txt
pytest                            # all 26 tests
pytest tests/test_model.py -v     # just model tests
pytest tests/test_identity_results.py -v  # validate JSON reference
```

Most tests run on CPU. The end-to-end training tests (`test_training.py`)
use CUDA kernels when available, so run them on a GPU machine for realistic
behavior. The JSON-validation tests don't need GPU.

## Regenerating Reference Results

`results/identity_2000steps.json` contains the 2000-step loss curves for both
mini-megatron and Megatron-Core on the identity task. To regenerate:

```bash
# 1. Requires GPU + Megatron-Core installed
# 2. Generate the synthetic + identity dataset
python experiments/synthetic_data.py experiments/synthetic_data.pt
python experiments/make_identity.py    # creates identity_data.pt

# 3. Run comparison (takes ~10 min on L20)
python experiments/compare_convergence.py --data-file experiments/identity_data.pt \
       --steps 2000 --warmup 50

# 4. Manually save the loss curves to results/identity_2000steps.json
#    (compare_convergence.py prints them; the JSON format is documented
#    in tests/test_identity_results.py)
```

The JSON must keep the same schema (see `tests/test_identity_results.py`)
or the validation tests will fail.

## Adding a Feature

1. Open an issue first if the change is non-trivial — saves wasted work.
2. Keep the core codebase under ~800 lines. Move non-essential pieces to
   `comm/` or `parallel/` as reference implementations (clearly marked in
   README's "Reference implementations" section).
3. Add a test in `tests/` if you add functionality.
4. Don't add new dependencies — the project stays PyTorch-only.

## Style

- Follow existing style (4-space indent, no emojis, line length ~100).
- No comments unless the code is genuinely non-obvious.
- Commit messages: `<type>: <short summary>` then a blank line and a body.
  Types: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`.

## License

By contributing, you agree your contributions are licensed under the MIT
license (see `LICENSE`).