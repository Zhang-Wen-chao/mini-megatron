"""Freeze a three-window PP numerical-gate conclusion without rewriting evidence."""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", action="append", required=True,
                        help="OFFSET:PATH_TO_GATE_SUMMARY.json; repeat exactly three times.")
    parser.add_argument("--tp", type=int, choices=(1, 2), required=True)
    parser.add_argument("--pp", type=int, choices=(2,), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.window) != 3:
        parser.error("exactly three independent gate windows are required")
    windows = []
    for item in args.window:
        raw_offset, separator, raw_path = item.partition(":")
        if not separator or not raw_offset.isdecimal():
            parser.error("--window must use OFFSET:PATH")
        path = Path(raw_path).resolve()
        data = json.loads(path.read_text(encoding="utf-8"))
        offset = int(raw_offset)
        if data.get("batch_offset_microbatches") != offset or data.get("topology") != {"tp": args.tp, "pp": args.pp, "dp": 1}:
            parser.error(f"offset mismatch for {path}")
        windows.append({"offset_microbatches": offset, "source_path": str(path),
                        "source_sha256": sha256(path), "summary": data})
    windows.sort(key=lambda item: item["offset_microbatches"])
    calibrated = all(item["summary"].get("published_calibration_passed") for item in windows)
    original = all(item["summary"].get("original_per_tensor_passed") for item in windows)
    report = {
        "schema_version": 1,
        "topology": {"tp": args.tp, "pp": args.pp, "dp": 1},
        "status": "exploratory_calibrated_gate_passed" if calibrated else "calibration_failed",
        "original_per_tensor_gate_passed_all_windows": original,
        "post_hoc_exploratory_calibration_passed_all_windows": calibrated,
        "calibration_caveat": (
            "The original per-tensor relative-L2 rule was not met. The alternative global metric and "
            "thresholds were chosen after the offset-0 observation and then reproduced on offsets 8 and 16; "
            "any subsequent throughput comparison is a calibrated matching-custom-loop result, not proof that "
            "the original campaign gate passed."
        ),
        "windows": windows,
        "next_requirement": "Five alternating throughput pairs and a separately captured profile remain required before any conditional PP performance conclusion.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise ValueError("refusing to overwrite conclusion: " + str(args.output))
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
