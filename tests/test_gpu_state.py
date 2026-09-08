# tests/test_gpu_state.py
"""Measurement-window guardrails: resident detection, process-group kill,
single-flight lock, journal — the 2026-09-07 contamination post-mortem."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest

from llama_packer import gpu_state


@pytest.fixture
def cache(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    root.mkdir()
    monkeypatch.setenv("LLAMA_PACKER_CACHE_DIR", str(root))
    return root


# ── resident pre-flight ───────────────────────────────────────────────────

def test_llama_residents_matches(monkeypatch):
    out = subprocess.CompletedProcess([], 0, stdout="\n".join([
        "  101 jk /opt/llama/bin/llama-server -m /models/x.gguf",
        "  102 jk /bin/sh -c llama-swap -f /etc/llama-swap.yaml",
        "  103 jk /opt/llama/bin/llama-fit-params --fit off",
        "  104 jk /tmp/fake-llama-server -m x",
        "  105 jk emacs llama-server-notes.txt",
        "  106 jk bash -c 'echo llama-server'",
    ]), stderr="")
    monkeypatch.setattr(gpu_state.subprocess, "run", lambda *a, **k: out)
    residents = gpu_state.llama_residents()
    assert len(residents) == 3
    assert residents[0].startswith("101 ")


# ── process-group kill ────────────────────────────────────────────────────

def test_kill_process_group_kills_descendants():
    proc = subprocess.Popen(
        [sys.executable, "-c",
         "import subprocess, time; subprocess.Popen(['sleep', '30']);"
         " time.sleep(30)"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True)
    time.sleep(0.5)  # let the grandchild spawn
    gpu_state.kill_process_group(proc)
    assert proc.poll() is not None
    with pytest.raises(ProcessLookupError):
        os.killpg(proc.pid, 0)  # the whole group — grandchild included


# ── single-flight lock ────────────────────────────────────────────────────

def test_measurement_lock_exclusive(cache):
    with gpu_state.measurement_lock("probe"):
        with pytest.raises(RuntimeError):
            with gpu_state.measurement_lock("probe"):
                pass


def test_measurement_lock_steals_dead_owner(cache):
    (cache / "measure.lock").write_text("999999999")
    with gpu_state.measurement_lock("measure"):
        assert (cache / "measure.lock").read_text() == str(os.getpid())


# ── journal ───────────────────────────────────────────────────────────────

def test_journal_appends_jsonl(cache):
    gpu_state.journal({"mode": "serve", "outcome": "ok"})
    gpu_state.journal({"mode": "fit", "outcome": "spill"})
    lines = (cache / "measure-journal.jsonl").read_text().splitlines()
    assert [json.loads(line)["mode"] for line in lines] == ["serve", "fit"]
    assert all(json.loads(line)["ts"] for line in lines)
