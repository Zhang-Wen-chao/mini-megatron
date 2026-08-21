"""Tests for the append-only multi-GPU fair-experiment ledger."""
import json
from argparse import Namespace

from experiments.fair_parallel_campaign import execution_schedule, init_campaign, record, validate


def test_campaign_schedule_is_fixed_and_balanced():
    schedule = execution_schedule()
    assert len(schedule) == 10
    assert [entry["variant"] for entry in schedule] == [
        "mini", "mcore", "mcore", "mini", "mini", "mcore", "mcore", "mini", "mini", "mcore",
    ]
    assert [entry["pair"] for entry in schedule] == ["01", "01", "02", "02", "03", "03", "04", "04", "05", "05"]


def test_campaign_records_hashes_and_detects_tampering(tmp_path, capsys):
    campaign_dir = tmp_path / "fair-125m"
    init_campaign(Namespace(campaign_dir=campaign_dir, campaign_id="fair-125m-test"))
    source = tmp_path / "artifact.json"
    source.write_text(json.dumps({"ok": True}))
    record(Namespace(campaign_dir=campaign_dir, kind="artifact", topology="tp2-pp1-dp1", source=source, note=None))
    assert validate(Namespace(campaign_dir=campaign_dir)) == 0
    source.write_text(json.dumps({"ok": False}))
    assert validate(Namespace(campaign_dir=campaign_dir)) == 1
    assert "source checksum mismatch" in capsys.readouterr().out
