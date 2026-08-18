"""Create an auditable kernel-time summary from an Nsight Systems CSV export.

The classifier is intentionally conservative. A generic elementwise kernel is
not labelled as AdamW merely because it occurs around optimizer.step(); only the
explicit fused AdamW kernel receives the fused_adamw category. All unmatched time
remains visible as unclassified.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

try:
    from .run_experiment import write_checksums
except ImportError:
    from run_experiment import write_checksums


CLASSIFIERS = [
    ("fused_adamw", re.compile(r"FusedAdamMathFunctor|multi_tensor_apply_kernel", re.I)),
    ("gemm", re.compile(r"cutlass|gemm|cublas|matmul", re.I)),
    ("attention", re.compile(r"flash|scaled_dot_product|fmha", re.I)),
    ("copy_cast", re.compile(r"bfloat16_copy|copy_kernel|cast", re.I)),
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_int(value):
    return int(value.replace(",", ""))


def classify(name):
    for category, pattern in CLASSIFIERS:
        if pattern.search(name):
            return category
    return "unclassified"


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append({
                "name": row["Name"],
                "total_time_ns": parse_int(row["Total Time (ns)"]),
                "instances": parse_int(row["Instances"]),
            })
    return rows


def analyze(rows):
    total_ns = sum(row["total_time_ns"] for row in rows)
    categories = {}
    for row in rows:
        category = classify(row["name"])
        summary = categories.setdefault(category, {"total_time_ns": 0, "instances": 0, "kernels": []})
        summary["total_time_ns"] += row["total_time_ns"]
        summary["instances"] += row["instances"]
        summary["kernels"].append(row)
    for summary in categories.values():
        summary["percent_kernel_time"] = 100 * summary["total_time_ns"] / total_ns if total_ns else 0.0
        summary["kernels"].sort(key=lambda row: row["total_time_ns"], reverse=True)
    return {
        "total_kernel_time_ns": total_ns,
        "total_kernel_instances": sum(row["instances"] for row in rows),
        "categories": categories,
        "top_kernels": sorted(rows, key=lambda row: row["total_time_ns"], reverse=True)[:30],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", help="Run-bundle directory containing profile/*.csv")
    args = parser.parse_args()
    bundle = Path(args.bundle)
    candidates = sorted((bundle / "profile").glob("*cuda_gpu_kern_sum*.csv"))
    if len(candidates) != 1:
        parser.error("expected exactly one CUDA kernel summary CSV, found " + str(len(candidates)))
    source = candidates[0]
    result = analyze(read_rows(source))
    result.update({
        "schema_version": 1,
        "source_csv": str(source.relative_to(bundle)),
        "source_csv_sha256": sha256(source),
        "classification_rules": [
            {"category": category, "regex": pattern.pattern} for category, pattern in CLASSIFIERS
        ],
        "method_notes": [
            "All percentages use total CUDA kernel time, not wall-clock time.",
            "Unmatched kernel time is deliberately retained as unclassified.",
            "Only explicit FusedAdamMathFunctor/multi_tensor_apply_kernel names are labelled fused_adamw.",
            "Do not infer an unfused AdamW total from generic elementwise kernel names alone.",
        ],
    })
    destination = bundle / "profile" / "analysis.json"
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_checksums(bundle)
    print(json.dumps({
        "analysis": str(destination),
        "total_kernel_time_ns": result["total_kernel_time_ns"],
        "categories": {
            name: {"percent_kernel_time": value["percent_kernel_time"],
                   "instances": value["instances"]}
            for name, value in result["categories"].items()
        },
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
