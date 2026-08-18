"""Tests for the dependency-free experiment evidence format."""
import json
from pathlib import Path

from experiments.run_experiment import parse_compute_processes, parse_metrics, write_checksums
from experiments.analyze_nsys_kernel_summary import analyze
from experiments.summarize_paired_results import load_bundles, summarize
from experiments.validate_run_bundle import validate
from experiments.fair_tp1_contract import qkv_mcore_to_mini, qkv_mini_to_mcore


def _make_bundle(tmp_path):
    bundle = Path(tmp_path) / "run"
    bundle.mkdir()
    manifest = {
        "schema_version": 1,
        "run_id": "test-run",
        "kind": "benchmark",
        "claim_scope": "test",
        "target_command": ["python", "main.py"],
        "executed_command": ["python", "main.py"],
        "started_at_utc": "2026-01-01T00:00:00+00:00",
        "finished_at_utc": "2026-01-01T00:00:01+00:00",
        "elapsed_wall_seconds": 1.0,
        "return_code": 0,
        "metrics": {"throughput_tok_s": 100.0},
        "profile_report": None,
        "environment_file": "environment.json",
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    (bundle / "metrics.json").write_text(json.dumps(manifest["metrics"]))
    (bundle / "environment.json").write_text("{}")
    (bundle / "stdout.log").write_text("ok")
    (bundle / "stderr.log").write_text("")
    (bundle / "command.txt").write_text("python main.py\n")
    write_checksums(bundle)
    return bundle


def test_parse_metrics_reads_training_output():
    output = "Throughput:      60,718 tok/s\nPeak memory:     3.88 GB/GPU\nMFU:             42.49%\nFinal loss:      9.8482"
    assert parse_metrics(output) == {
        "throughput_tok_s": 60718.0,
        "peak_memory_gb": 3.88,
        "mfu_percent": 42.49,
        "final_loss": 9.8482,
    }


def test_parse_compute_processes_ignores_idle_messages():
    assert parse_compute_processes("No running processes found\n") == []
    assert parse_compute_processes("123, python, GPU-abc\n") == ["123, python, GPU-abc"]


def test_nsys_analysis_keeps_generic_kernels_unclassified():
    rows = [
        {"name": "multi_tensor_apply_kernel<FusedAdamMathFunctor>", "total_time_ns": 30, "instances": 2},
        {"name": "cutlass bf16 gemm", "total_time_ns": 50, "instances": 3},
        {"name": "vectorized_elementwise_kernel", "total_time_ns": 20, "instances": 4},
    ]
    result = analyze(rows)
    assert result["categories"]["fused_adamw"]["percent_kernel_time"] == 30.0
    assert result["categories"]["unclassified"]["percent_kernel_time"] == 20.0


def test_fair_qkv_conversion_round_trips_for_multi_head_attention():
    import torch

    config = {"num_attention_heads": 2, "hidden_size": 8}
    mini_qkv = torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(24, 8)
    mcore_qkv = qkv_mini_to_mcore(mini_qkv, config)
    assert not torch.equal(mini_qkv, mcore_qkv)
    assert torch.equal(qkv_mcore_to_mini(mcore_qkv, config), mini_qkv)


def test_valid_bundle_passes_checksum_validation(tmp_path):
    assert validate(_make_bundle(tmp_path)) == []


def test_changed_artifact_fails_checksum_validation(tmp_path):
    bundle = _make_bundle(tmp_path)
    (bundle / "stdout.log").write_text("tampered")
    assert any("checksum mismatch: stdout.log" in error for error in validate(bundle))


def test_paired_summary_reports_ratios_without_selecting_best_run():
    records = [
        {"throughput_tok_s": 120.0, "tags": {"variant": "mini", "pair": "01", "condition": "125m"}},
        {"throughput_tok_s": 100.0, "tags": {"variant": "megatron", "pair": "01", "condition": "125m"}},
        {"throughput_tok_s": 110.0, "tags": {"variant": "mini", "pair": "02", "condition": "125m"}},
        {"throughput_tok_s": 100.0, "tags": {"variant": "megatron", "pair": "02", "condition": "125m"}},
    ]
    report = summarize(records, "mini", "megatron", min_pairs=2)
    assert report["paired_ratio_left_over_right"]["count"] == 2
    assert report["paired_ratio_left_over_right"]["mean"] == 1.15


def test_paired_summary_rejects_mismatched_conditions():
    records = [
        {"throughput_tok_s": 120.0, "tags": {"variant": "mini", "pair": "01", "condition": "125m"}},
        {"throughput_tok_s": 100.0, "tags": {"variant": "megatron", "pair": "01", "condition": "1b"}},
    ]
    try:
        summarize(records, "mini", "megatron", min_pairs=1)
    except ValueError as error:
        assert "same non-empty condition" in str(error)
    else:
        raise AssertionError("mismatched conditions must fail")


def test_load_bundles_rejects_missing_or_non_idle_preflight(tmp_path):
    bundle = _make_bundle(tmp_path)
    assert load_bundles(tmp_path, allow_dirty=True) == []
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["source_tree_clean"] = True
    manifest["gpu_preflight"] = {"return_code": 0, "active_processes": ["123, python, GPU-abc"]}
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    write_checksums(bundle)
    assert load_bundles(tmp_path) == []
