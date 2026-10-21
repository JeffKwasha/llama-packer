"""Tests for the long-horizon VRAM baseline logger."""

import csv
import json

from llama_packer import vram_log


class _R:
    def __init__(self, returncode=0, stdout=""):
        self.returncode = returncode
        self.stdout = stdout


def test_visible_mem_amd_json(monkeypatch):
    # Validates: docs/plans/phase0-vram-baseline.md, 'Idle-floor campaign (same box, later session, 2026-09-26)': amd-smi's JSON is the primary source for (total, used, free, gtt).
    payload = {"gpu_data": [{"mem_usage": {
        "total_vram": {"value": 32624}, "used_vram": {"value": 4254},
        "free_vram": {"value": 28370}, "used_gtt": {"value": 336}}}]}
    monkeypatch.setattr(vram_log.subprocess, "run",
                        lambda *a, **k: _R(0, json.dumps(payload)))
    assert vram_log.visible_mem() == (32624, 4254, 28370, 336)


def test_visible_mem_falls_back_to_nvidia(monkeypatch):
    # Validates: docs/plans/phase0-vram-baseline.md, 'Idle-floor campaign (same box, later session, 2026-09-26)': nvidia-smi is the fallback, and it reports no GTT figure.
    def run(cmd, **k):
        if "amd-smi" in cmd[0]:
            return _R(1, "")
        return _R(0, "32624, 4254, 28370")
    monkeypatch.setattr(vram_log.subprocess, "run", run)
    # nvidia-smi has no GTT figure → reported as 0.
    assert vram_log.visible_mem() == (32624, 4254, 28370, 0)


def test_visible_mem_none_when_all_tools_fail(monkeypatch):
    # No GPU tools, no sysfs matches → None (sample records blank memory).
    # Validates: docs/plans/phase0-vram-baseline.md, 'Idle-floor campaign (same box, later session, 2026-09-26)': with no GPU tool and no sysfs match there is nothing to report, not a zero.
    monkeypatch.setattr(vram_log.subprocess, "run", lambda *a, **k: _R(1, ""))
    monkeypatch.setattr(vram_log.Path, "glob", lambda self, pat: [])
    assert vram_log.visible_mem() is None


def test_tenants_counts_each_class():
    # Validates: docs/plans/phase0-vram-baseline.md, 'Idle-floor campaign (same box, later session, 2026-09-26)': the process table is counted per tenant class (audio, models, immich, browsers, image).
    lines = [
        "user /app/audiocpp_server --config x",
        "user /usr/bin/llama-server -m model.gguf",
        "user python -m immich_ml",
        "user /usr/lib/librewolf/librewolf",
        "user /usr/bin/whisper-server",
    ]
    t = vram_log.tenants(lines)
    assert t["audio"] == 2          # audiocpp + whisper
    assert t["models"] == 1         # llama-server, not the swap manager
    assert t["immich"] == 1
    assert t["browsers"] == 1
    assert t["image"] == 0


def test_llama_swap_manager_is_not_a_model(monkeypatch):
    # Validates: docs/plans/phase0-vram-baseline.md, 'Idle-floor campaign (same box, later session, 2026-09-26)': the swap manager is not a model tenant.
    t = vram_log.tenants(["jk /mnt/pool/data/tools/llama/llama-swap -listen :8123"])
    assert t["models"] == 0


def _row(used, **kw):
    row = {"time": "2026-09-27T00:00:00-04:00", "total_mib": 32624,
           "used_mib": used, "free_mib": 0, "gtt_mib": 0, "count": 1,
           "models": 0, "audio": 0, "image": 0, "immich": 0, "browsers": 0}
    row.update(kw)
    return row


def _read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_append_writes_header_once(tmp_path):
    # Validates: docs/plans/phase0-vram-baseline.md, 'Idle-floor campaign (same box, later session, 2026-09-26)': the VRAM log is a CSV: one header, then a row per sample
    p = tmp_path / "v.csv"
    vram_log.append_row(p, _row(1700))
    vram_log.append_row(p, _row(1800))
    text = p.read_text()
    assert text.count("time,total_mib") == 1
    assert "1700" in text and "1800" in text


def test_report_floor_and_tenant_medians(tmp_path):
    # Validates: docs/plans/phase0-vram-baseline.md, 'Idle-floor campaign (same box, later session, 2026-09-26)': the report names the floor, the max, the spread, and each tenant's median.
    p = tmp_path / "v.csv"
    for used, audio, browser in ((1700, 0, 0), (1750, 0, 1),
                                 (4200, 1, 1), (4250, 1, 1)):
        vram_log.append_row(p, _row(used, audio=audio, browsers=browser))
    txt = vram_log.report(p)
    assert "floor=1700" in txt
    assert "max=4250" in txt
    assert "spread=2550" in txt
    # nearest-rank percentile: median of [4200, 4250] is 4200
    assert "with audio" in txt and "median=4200" in txt


def test_report_no_samples(tmp_path):
    # Validates: docs/plans/phase0-vram-baseline.md, 'The log format': with no usable sample the report says so rather than printing empty numbers
    p = tmp_path / "v.csv"
    p.write_text("time,total_mib,used_mib,free_mib,gtt_mib,count,models,audio,"
                 "image,immich,browsers\n")
    assert "no usable memory samples" in vram_log.report(p)


def test_report_weights_by_count(tmp_path):
    # Validates: docs/plans/phase0-vram-baseline.md, 'The log format': rows are weighted by their count, because the envelope is a slow sawtooth
    p = tmp_path / "v.csv"
    vram_log.append_row(p, _row(1700, count=10))
    vram_log.append_row(p, _row(4000, count=1))
    txt = vram_log.report(p)
    assert "samples: 11" in txt
    assert "rows: 2" in txt
    assert "floor=1700" in txt and "p50=1700" in txt   # plateau not outvoted


# ── the accumulator: run-length counts + deferred idle writes ──────────────


def test_accumulator_compresses_within_threshold(tmp_path):
    # Validates: docs/plans/phase0-vram-baseline.md, 'Idle-floor campaign (same box, later session, 2026-09-26)': samples within the threshold collapse into one row with a count.
    p = tmp_path / "v.csv"
    acc = vram_log.Accumulator(p, threshold=128, low_mib=0)
    for used in (4000, 4050, 4100):        # all within 128 of 4000
        acc.add(_row(used))
    acc.close()
    rows = _read(p)
    assert len(rows) == 1
    assert rows[0]["used_mib"] == "4000"
    assert rows[0]["count"] == "3"


def test_accumulator_opens_row_on_big_change(tmp_path):
    # Validates: docs/plans/phase0-vram-baseline.md, 'Idle-floor campaign (same box, later session, 2026-09-26)': a change past the threshold opens a new row.
    p = tmp_path / "v.csv"
    acc = vram_log.Accumulator(p, threshold=128, low_mib=0)
    acc.add(_row(2000))
    acc.add(_row(4300))                    # +2300 > 128 → new row
    acc.close()
    assert [r["used_mib"] for r in _read(p)] == ["2000", "4300"]


def test_accumulator_holds_rows_while_low(tmp_path):
    # Validates: docs/plans/phase0-vram-baseline.md, 'Idle-floor campaign (same box, later session, 2026-09-26)': while usage stays below the idle floor nothing is written; the rows are flushed on close.
    p = tmp_path / "v.csv"
    acc = vram_log.Accumulator(p, threshold=128, low_mib=2048)
    acc.add(_row(1700))                    # below low → buffered in RAM
    acc.add(_row(1750))                    # within threshold → count bump
    assert not p.exists(), "no disk writes while idle"
    acc.close()                            # quit → flush
    rows = _read(p)
    assert len(rows) == 1 and rows[0]["count"] == "2"


def test_accumulator_flushes_when_usage_rises(tmp_path):
    # Validates: docs/plans/phase0-vram-baseline.md, 'Idle-floor campaign (same box, later session, 2026-09-26)': a rise out of the idle band finalises and flushes the buffered row.
    p = tmp_path / "v.csv"
    acc = vram_log.Accumulator(p, threshold=128, low_mib=2048)
    acc.add(_row(1700))                    # low, buffered
    acc.add(_row(4300))                    # high → finalize+flush the low row
    assert p.exists()
    assert _read(p)[0]["used_mib"] == "1700"
    acc.close()
    assert [r["used_mib"] for r in _read(p)] == ["1700", "4300"]
