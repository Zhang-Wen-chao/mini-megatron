"""Create and audit the append-only evidence ledger for fair parallel studies.

The ledger deliberately lives beside the large L20 archive.  It records hashes
and absolute archive paths; benchmark code never has to guess which checkpoint,
batch file, or run bundle supports a published number.  Entries are append-only
JSON files so an old experiment cannot be silently rewritten after publication.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path


SCHEMA_VERSION = 1
CONTRACT = {
    "model": "bias-free GPT next-token training",
    "layers": 12,
    "hidden_size": 768,
    "attention_heads": 12,
    "ffn_hidden_size": 3072,
    "sequence_length": 512,
    "precision": "fp32",
    "optimizer": "torch.optim.AdamW(fused=False, lr=6e-4, betas=(0.9,0.999), weight_decay=0.1)",
    "dropout": 0.0,
    "position_embedding": "learned_absolute",
}

# The existing TP=1 result remains an anchor.  The three remaining entries are
# the complete 125M multi-GPU matrix requested for this campaign.
TOPOLOGIES = (
    {"id": "tp1-pp1-dp1", "tp": 1, "pp": 1, "dp": 1, "status": "completed_anchor"},
    {"id": "tp2-pp1-dp1", "tp": 2, "pp": 1, "dp": 1, "status": "planned"},
    {"id": "tp1-pp2-dp1", "tp": 1, "pp": 2, "dp": 1, "status": "planned"},
    {"id": "tp2-pp2-dp1", "tp": 2, "pp": 2, "dp": 1, "status": "planned"},
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, value):
    path = Path(path)
    if path.exists():
        raise ValueError("refusing to overwrite immutable evidence: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_campaign(campaign_dir):
    campaign_file = Path(campaign_dir) / "campaign.json"
    if not campaign_file.is_file():
        raise ValueError("campaign.json not found: " + str(campaign_file))
    campaign = json.loads(campaign_file.read_text(encoding="utf-8"))
    if campaign.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported campaign schema")
    return campaign


def execution_schedule():
    # Ten timed bundles per topology: five paired observations, in alternating
    # order.  This prevents one implementation from always getting a colder or
    # hotter host.  The order is intentionally fixed before any measurement.
    schedule = []
    for pair in range(1, 6):
        variants = ("mini", "mcore") if pair % 2 else ("mcore", "mini")
        for ordinal, variant in enumerate(variants, start=1):
            schedule.append({"pair": f"{pair:02d}", "variant": variant,
                             "pair_ordinal": ordinal, "execution_ordinal": len(schedule) + 1})
    return schedule


def init_campaign(args):
    campaign_dir = Path(args.campaign_dir).resolve()
    if campaign_dir.exists():
        raise ValueError("campaign directory already exists: " + str(campaign_dir))
    campaign = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": args.campaign_id,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "archive_root": str(campaign_dir),
        "purpose": "125M fair mini-megatron vs Megatron-Core parallel comparison",
        "contract": CONTRACT,
        "fairness_requirements": {
            "same_logical_model": True,
            "same_source_weights": True,
            "same_immutable_batches": True,
            "same_global_tokens_per_optimizer_update": True,
            "numerical_gate": {
                "initial_weight_max_abs": 0.0,
                "max_logits_relative_l2": 0.0005,
                "max_gradient_relative_l2": 0.0005,
                "max_parameter_relative_l2_after_one_adamw_step": 0.0001,
            },
            "throughput_pairs": 5,
            "profile_is_not_throughput_sample": True,
        },
        "measurement_plan": {
            "warmup_optimizer_steps": 30,
            "measured_optimizer_steps": 200,
            "micro_batch_size": 4,
            "micro_batches_per_optimizer_step": 8,
            "tokens_per_optimizer_step": 16384,
            "execution_schedule": execution_schedule(),
        },
        "topologies": list(TOPOLOGIES),
        "publication_rule": (
            "A topology is publishable only when its artifact, numerical_parity, "
            "benchmark_summary, and profile entries all validate."),
    }
    campaign_dir.mkdir(parents=True)
    write_new(campaign_dir / "campaign.json", campaign)
    for name in ("ledger", "artifacts", "parity", "benchmarks", "profiles", "reports"):
        (campaign_dir / name).mkdir()
    print(json.dumps(campaign, indent=2, sort_keys=True))


def next_ledger_path(campaign_dir, kind, topology):
    ledger = Path(campaign_dir) / "ledger"
    existing = sorted(ledger.glob("*.json"))
    return ledger / f"{len(existing) + 1:04d}-{kind}-{topology}.json"


def record(args):
    campaign = load_campaign(args.campaign_dir)
    topology_ids = {item["id"] for item in campaign["topologies"]}
    if args.topology not in topology_ids:
        raise ValueError("unknown topology: " + args.topology)
    source = Path(args.source).resolve()
    if not source.is_file():
        raise ValueError("source evidence is not a file: " + str(source))
    entry = {
        "schema_version": SCHEMA_VERSION,
        "kind": args.kind,
        "topology": args.topology,
        "recorded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_path": str(source),
        "source_sha256": sha256(source),
        "source_size_bytes": source.stat().st_size,
        "campaign_id": campaign["campaign_id"],
    }
    if args.note:
        entry["note"] = args.note
    destination = next_ledger_path(args.campaign_dir, args.kind, args.topology)
    write_new(destination, entry)
    print(json.dumps({"entry": str(destination), **entry}, indent=2, sort_keys=True))


def validate(args):
    campaign_dir = Path(args.campaign_dir).resolve()
    campaign = load_campaign(campaign_dir)
    errors = []
    records = []
    for path in sorted((campaign_dir / "ledger").glob("*.json")):
        try:
            record_data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            errors.append(f"invalid ledger entry {path.name}: {error}")
            continue
        records.append((path, record_data))
        source = Path(record_data.get("source_path", ""))
        if not source.is_file():
            errors.append(f"{path.name}: source missing: {source}")
        elif sha256(source) != record_data.get("source_sha256"):
            errors.append(f"{path.name}: source checksum mismatch")
    status = {}
    for topology in campaign["topologies"]:
        topology_id = topology["id"]
        kinds = {data.get("kind") for _, data in records if data.get("topology") == topology_id}
        required = {"artifact", "numerical_parity", "benchmark_summary", "profile"}
        status[topology_id] = {
            "recorded_kinds": sorted(kind for kind in kinds if kind),
            "missing_publishable_evidence": sorted(required - kinds),
            "publishable": not bool(required - kinds),
        }
    result = {"campaign_id": campaign["campaign_id"], "valid": not errors,
              "errors": errors, "topologies": status,
              "note": "publishable means the required evidence types are recorded; inspect their source reports before claiming a result."}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    init_parser = subcommands.add_parser("init", help="create an immutable campaign plan")
    init_parser.add_argument("--campaign-dir", required=True)
    init_parser.add_argument("--campaign-id", required=True)
    init_parser.set_defaults(func=init_campaign)
    record_parser = subcommands.add_parser("record", help="append a checksummed evidence reference")
    record_parser.add_argument("--campaign-dir", required=True)
    record_parser.add_argument("--kind", required=True, choices=("artifact", "numerical_parity", "benchmark_summary", "profile", "diagnostic"),
                               help="diagnostic records failed/excluded smoke tests and never satisfy publication evidence.")
    record_parser.add_argument("--topology", required=True)
    record_parser.add_argument("--source", required=True)
    record_parser.add_argument("--note")
    record_parser.set_defaults(func=record)
    validate_parser = subcommands.add_parser("validate", help="verify all archived references still hash-match")
    validate_parser.add_argument("--campaign-dir", required=True)
    validate_parser.set_defaults(func=validate)
    args = parser.parse_args()
    try:
        return args.func(args) or 0
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
