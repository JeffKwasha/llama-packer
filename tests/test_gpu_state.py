# tests/test_gpu_state.py
"""Measurement-window guardrails: resident detection, process-group kill,
single-flight lock, journal — the 2026-09-07 contamination post-mortem."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

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
    # Validates: docs/plans/auto-parallel.md, 'Findings: the affine VRAM law (2026-09-07, SEAL record)': only real llama-server processes count as residents; a shell that merely mentions the name does not.
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

def _proc_state(pid: int) -> str | None:
    """The state letter of *pid*, or None when the pid is gone.

    The executable name in ``/proc/<pid>/stat`` is parenthesised and may
    contain spaces, so the state is the character after the *last* ``)``.
    """
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return None
    return stat[stat.rindex(")") + 2]


def test_kill_process_group_kills_descendants(tmp_path):
    # Validates: docs/plans/auto-parallel.md, 'Findings: the affine VRAM law (2026-09-07, SEAL record)': the whole process group dies, so a server's children cannot outlive it.
    """The descendant is dead; a zombie waiting for init still counts.

    The grandchild is an orphan the moment the leader dies, and until init
    reaps it the zombie still belongs to the group — so the group outliving
    the kill proves nothing.
    """
    pidfile = tmp_path / "grandchild.pid"
    proc = subprocess.Popen(
        [sys.executable, "-c",
         "import subprocess, time;"
         " p = subprocess.Popen(['sleep', '30']);"
         f" open({str(pidfile)!r}, 'w').write(str(p.pid));"
         " time.sleep(30)"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not pidfile.exists():
        time.sleep(0.05)
    assert pidfile.exists(), "the grandchild never started"
    grandchild = int(pidfile.read_text())
    gpu_state.kill_process_group(proc)
    assert proc.poll() is not None
    state = _proc_state(grandchild)
    assert state in (None, "Z"), f"the grandchild is still {state}"


# ── single-flight lock ────────────────────────────────────────────────────

def test_measurement_lock_exclusive(cache):
    # Validates: docs/plans/auto-parallel.md, 'Findings: the affine VRAM law (2026-09-07, SEAL record)': one measurement campaign at a time.
    with gpu_state.measurement_lock("probe"):
        with pytest.raises(RuntimeError):
            with gpu_state.measurement_lock("probe"):
                pass


def test_measurement_lock_steals_dead_owner(cache):
    # Validates: docs/plans/auto-parallel.md, 'Findings: the affine VRAM law (2026-09-07, SEAL record)': a lock whose owner is gone is taken over, not waited on forever.
    (cache / "measure.lock").write_text("999999999")
    with gpu_state.measurement_lock("measure"):
        assert (cache / "measure.lock").read_text() == str(os.getpid())


# ── journal ───────────────────────────────────────────────────────────────

def test_journal_appends_jsonl(cache):
    # Validates: docs/plans/auto-parallel.md, 'Findings: the affine VRAM law (2026-09-07, SEAL record)': each measurement appends one timestamped JSON line to the journal.
    gpu_state.journal({"mode": "serve", "outcome": "ok"})
    gpu_state.journal({"mode": "fit", "outcome": "spill"})
    lines = (cache / "measure-journal.jsonl").read_text().splitlines()
    assert [json.loads(line)["mode"] for line in lines] == ["serve", "fit"]
    assert all(json.loads(line)["ts"] for line in lines)
