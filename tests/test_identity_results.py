"""Validate schema of the legacy identity-task observation.

Identity loss is a wiring smoke test, not a framework-quality comparison.
Performance claims require a reproducible run bundle under the experiment
protocol instead of a permanent unit-test oracle.
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


def test_loss_curves_have_numeric_monotonic_steps():
    """Legacy curves remain parseable observations without ranking frameworks."""
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    for name, curve in data["results"].items():
        assert len(curve) > 10, name
        steps = [point["step"] for point in curve]
        losses = [point["loss"] for point in curve]
        assert all(isinstance(step, int) and step > 0 for step in steps)
        assert steps == sorted(steps) and len(set(steps)) == len(steps)
        assert all(isinstance(loss, (int, float)) and loss >= 0 for loss in losses)
