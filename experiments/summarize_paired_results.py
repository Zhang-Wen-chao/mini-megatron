"""Summarize paired mini and Megatron benchmark run bundles.

Each benchmark bundle needs tags variant=<name> and pair=<id>. Pair IDs connect
same-condition neighbors in an ABBA or BAAB schedule. Only successful,
unprofiled bundles with throughput metrics enter the calculation.
"""
import argparse
import json
from pathlib import Path
import statistics

try:
    from .validate_run_bundle import validate
except ImportError:
    from validate_run_bundle import validate


def load_bundles(results_dir, allow_dirty=False, topology=None, condition=None):
    records = []
    for manifest_path in sorted(Path(results_dir).glob("*/manifest.json")):
        validation_errors = validate(manifest_path.parent)
        if validation_errors:
            raise ValueError("invalid run bundle " + str(manifest_path.parent) + ": " + "; ".join(validation_errors))
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            continue
        if manifest.get("kind") != "benchmark" or manifest.get("return_code") != 0:
            continue
        preflight = manifest.get("gpu_preflight", {})
        if preflight.get("return_code") != 0 or preflight.get("active_processes") != []:
            continue
        if not manifest.get("source_tree_clean") and not allow_dirty:
            continue
        throughput = manifest.get("metrics", {}).get("throughput_tok_s")
        tags = manifest.get("tags", {})
        if topology is not None and tags.get("topology") != topology:
            continue
        if condition is not None and tags.get("condition") != condition:
            continue
        if isinstance(throughput, (int, float)) and isinstance(tags, dict):
            records.append({"bundle": str(manifest_path.parent), "throughput_tok_s": throughput,
                            "tags": tags, "run_id": manifest.get("run_id"),
                            "source_tree_clean": manifest.get("source_tree_clean"),
                            "git_commit": json.loads((manifest_path.parent / "environment.json").read_text()).get("git_commit")})
    return records


def summarize(records, left, right, min_pairs=5):
    pairs = {}
    for record in records:
        variant, pair = record["tags"].get("variant"), record["tags"].get("pair")
        if variant in (left, right) and pair:
            pairs.setdefault(pair, {})[variant] = record
    completed = []
    for pair, values in sorted(pairs.items()):
        if left in values and right in values:
            left_condition = values[left]["tags"].get("condition")
            right_condition = values[right]["tags"].get("condition")
            if not left_condition or left_condition != right_condition:
                raise ValueError("pair " + pair + " must have the same non-empty condition tag")
            completed.append({"pair": pair, "left": values[left], "right": values[right],
                              "ratio": values[left]["throughput_tok_s"] / values[right]["throughput_tok_s"]})
    if not completed:
        raise ValueError("no completed pairs; tag runs with variant=<name> and pair=<id>")
    if len(completed) < min_pairs:
        raise ValueError("need at least " + str(min_pairs) + " completed pairs, found " + str(len(completed)))
    ratios = [item["ratio"] for item in completed]
    left_values = [item["left"]["throughput_tok_s"] for item in completed]
    right_values = [item["right"]["throughput_tok_s"] for item in completed]
    def stats(values):
        return {"count": len(values), "mean": statistics.mean(values), "median": statistics.median(values),
                "sample_stddev": statistics.stdev(values) if len(values) > 1 else None,
                "min": min(values), "max": max(values)}
    return {"left_variant": left, "right_variant": right, "pairs": completed,
            "left_throughput_tok_s": stats(left_values), "right_throughput_tok_s": stats(right_values),
            "paired_ratio_left_over_right": stats(ratios),
            "source_tree_clean": all(item["left"].get("source_tree_clean", False) and item["right"].get("source_tree_clean", False)
                                     for item in completed),
            "git_commits": sorted({item["left"].get("git_commit") for item in completed} |
                                  {item["right"].get("git_commit") for item in completed}),
            "interpretation": "Ratios above 1 favor the left variant only for the recorded configuration."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/runs")
    parser.add_argument("--left", default="mini")
    parser.add_argument("--right", default="megatron")
    parser.add_argument("--min-pairs", type=int, default=5)
    parser.add_argument("--allow-dirty", action="store_true",
                        help="Include dirty-tree samples; resulting report is provisional.")
    parser.add_argument("--topology",
                        help="Require an exact topology tag, preventing pair IDs from different topologies mixing.")
    parser.add_argument("--condition",
                        help="Require an exact condition tag in addition to the topology filter.")
    parser.add_argument("--output", help="Optional JSON file for the immutable aggregate report.")
    args = parser.parse_args()
    if args.min_pairs < 1:
        parser.error("--min-pairs must be positive")
    try:
        report = summarize(load_bundles(args.results_dir, args.allow_dirty, args.topology, args.condition),
                           args.left, args.right, args.min_pairs)
    except ValueError as error:
        parser.error(str(error))
    report["selection"] = {
        "results_dir": str(Path(args.results_dir)),
        "topology": args.topology,
        "condition": args.condition,
        "allow_dirty": args.allow_dirty,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        output = Path(args.output)
        if output.exists():
            parser.error("refusing to overwrite aggregate: " + str(output))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
