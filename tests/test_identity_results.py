"""Validate the identity-task training results from results/identity_2000steps.json.

This test verifies:
1. Results file exists
2. Both frameworks have loss curves
3. Both frameworks converged (final loss << initial)
4. mini-megatron converges significantly faster than Megatron-Core
"""
import json
import os
import pytest

RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "identity_2000steps.json"
)


def test_results_file_exists():
    """Results JSON must exist (committed to repo)."""
    assert os.path.exists(RESULTS_PATH), f"Missing {RESULTS_PATH}"


def test_results_format():
    """Results JSON has the expected structure."""
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    assert "results" in data
    assert "mini_megatron" in data["results"]
    assert "megatron_core" in data["results"]
    assert len(data["results"]["mini_megatron"]) > 10
    assert len(data["results"]["megatron_core"]) > 10


def test_mini_megatron_converges():
    """mini-megatron should drop loss from ~10 to near 0 in 2000 steps."""
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    curve = data["results"]["mini_megatron"]
    first_loss = curve[0]["loss"]
    last_loss = curve[-1]["loss"]
    # mini-megatron should converge to near-zero on the identity task
    assert first_loss > 5.0, f"First loss should be high (random), got {first_loss}"
    assert last_loss < 0.1, f"Final loss should be <0.1, got {last_loss}"
    # Should reach low loss well before step 2000
    early = next(d for d in curve if d["step"] >= 200)
    assert early["loss"] < 0.5, f"Loss at step 200 should be <0.5, got {early['loss']}"


def test_megatron_core_converges():
    """Megatron-Core should also converge to low loss in 2000 steps."""
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    curve = data["results"]["megatron_core"]
    first_loss = curve[0]["loss"]
    last_loss = curve[-1]["loss"]
    assert first_loss > 5.0, f"First loss should be high (random), got {first_loss}"
    # Megatron converges more slowly but should still reach low loss by step 2000
    assert last_loss < 0.5, f"Final loss should be <0.5, got {last_loss}"


def test_mini_faster_than_megatron():
    """At any given step, mini-megatron should have lower loss than Megatron-Core."""
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    mini = {d["step"]: d["loss"] for d in data["results"]["mini_megatron"]}
    base = {d["step"]: d["loss"] for d in data["results"]["megatron_core"]}
    # Compare at matching steps
    common_steps = sorted(set(mini.keys()) & set(base.keys()))
    assert len(common_steps) >= 10
    for step in common_steps:
        assert mini[step] < base[step], (
            f"mini-megatron loss {mini[step]:.4f} should be < "
            f"Megatron loss {base[step]:.4f} at step {step}"
        )


def test_mini_faster_by_significant_margin():
    """mini-megatron should converge ~5x faster (reaching loss 0.01 much earlier)."""
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    mini = data["results"]["mini_megatron"]
    base = data["results"]["megatron_core"]

    # Find the step where mini first reaches loss < 0.05
    mini_target = next((d["step"] for d in mini if d["loss"] < 0.05), None)
    assert mini_target is not None, "mini-megatron never reached loss < 0.05"

    # Find the step where Megatron first reaches loss < 0.05 (if ever)
    base_target = next((d["step"] for d in base if d["loss"] < 0.05), None)
    if base_target is not None:
        ratio = base_target / mini_target
        assert ratio >= 3, (
            f"mini-megatron reached loss 0.05 at step {mini_target}, "
            f"Megatron at step {base_target}, ratio={ratio:.1f}x (expected >= 3x)"
        )
