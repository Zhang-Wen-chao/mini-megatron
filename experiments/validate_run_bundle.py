"""Validate the manifest and checksums of a run bundle."""
import argparse
import hashlib
import json
from pathlib import Path


REQUIRED_FILES = {"manifest.json", "metrics.json", "environment.json", "stdout.log",
                  "stderr.log", "command.txt", "checksums.sha256"}
REQUIRED_FIELDS = {"schema_version", "run_id", "kind", "claim_scope", "target_command",
                   "executed_command", "started_at_utc", "finished_at_utc",
                   "elapsed_wall_seconds", "return_code", "metrics", "environment_file"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(bundle):
    bundle = Path(bundle)
    missing = sorted(name for name in REQUIRED_FILES if not (bundle / name).is_file())
    if missing:
        return ["missing required files: " + ", ".join(missing)]
    try:
        manifest = json.loads((bundle / "manifest.json").read_text())
    except json.JSONDecodeError as error:
        return ["invalid manifest.json: " + str(error)]
    errors = []
    missing_fields = sorted(REQUIRED_FIELDS - set(manifest))
    if missing_fields:
        errors.append("manifest missing fields: " + ", ".join(missing_fields))
    if manifest.get("schema_version") != 1:
        errors.append("unsupported schema_version")
    if not isinstance(manifest.get("target_command"), list) or not manifest.get("target_command"):
        errors.append("target_command must be a non-empty list")
    if not isinstance(manifest.get("metrics"), dict):
        errors.append("metrics must be an object")
    if manifest.get("kind") == "nsys_profile" and not manifest.get("profile_report"):
        errors.append("profile bundle has no profile_report")
    for line in (bundle / "checksums.sha256").read_text().splitlines():
        if not line:
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            errors.append("malformed checksum line: " + line)
            continue
        target = bundle / relative
        if not target.is_file():
            errors.append("checksummed file missing: " + relative)
        elif sha256(target) != expected:
            errors.append("checksum mismatch: " + relative)
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle")
    args = parser.parse_args()
    errors = validate(args.bundle)
    if errors:
        print("INVALID RUN BUNDLE")
        print("\n".join("- " + error for error in errors))
        return 1
    print("VALID RUN BUNDLE: " + str(Path(args.bundle).resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
