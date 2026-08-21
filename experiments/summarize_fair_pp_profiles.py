"""Index checksummed multi-rank PP Nsight bundles without timing interpretation.

The resulting JSON points at every raw report and SQLite export by SHA-256.
It deliberately contains no elapsed-time comparison: Nsight perturbation makes
it diagnostic evidence only, never a throughput measurement.
"""
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


def validate_checksums(bundle):
    errors = []
    for line in (bundle / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        expected, relative = line.split("  ", 1)
        target = bundle / relative
        if not target.is_file() or sha256(target) != expected:
            errors.append(relative)
    return errors


def file_record(path, bundle):
    return {"path": str(path.resolve()), "relative_path": str(path.relative_to(bundle)),
            "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mini-bundle", type=Path, required=True)
    parser.add_argument("--mcore-bundle", type=Path, required=True)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite immutable profile index: " + str(args.output))
    implementations = {}
    for name, bundle in (("mini", args.mini_bundle), ("mcore", args.mcore_bundle)):
        bundle = bundle.resolve()
        manifest_path = bundle / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("kind") != "multi_rank_nsys_profile":
            parser.error(f"{name} is not a multi-rank profile bundle")
        if manifest.get("implementation") != name or manifest.get("topology") != args.topology:
            parser.error(f"{name} implementation/topology mismatch")
        if manifest.get("return_code") != 0:
            parser.error(f"{name} profile returned non-zero")
        errors = validate_checksums(bundle)
        if errors:
            parser.error(f"{name} checksum mismatches: " + ", ".join(errors))
        reports = [bundle / value for value in manifest.get("profile_reports", [])]
        if len(reports) != manifest.get("profiled_cuda_ranks") or any(not report.is_file() for report in reports):
            parser.error(f"{name} has missing rank-local reports")
        sqlite = [report.with_suffix(".sqlite") for report in reports]
        if any(not path.is_file() for path in sqlite):
            parser.error(f"{name} has missing SQLite exports")
        implementations[name] = {
            "bundle_manifest": file_record(manifest_path, bundle),
            "checksums": file_record(bundle / "checksums.sha256", bundle),
            "profile_reports": [file_record(path, bundle) for path in reports],
            "sqlite_exports": [file_record(path, bundle) for path in sqlite],
        }
    result = {
        "schema_version": 1, "topology": args.topology,
        "status": "multi_rank_profile_complete",
        "claim_scope": "Profiler evidence only; never a throughput sample or absolute-time performance comparison.",
        "implementations": implementations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
