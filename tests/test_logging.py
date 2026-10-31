# tests/test_logging.py
"""The log level a pack run sets — SPEC.md, 'Log Levels'."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_SIDECAR = """---
name: dummy
context_length: 32768
usage: chat
engines: [llama-server]
---
"""


def _run(tmp_path: Path, *extra: str) -> subprocess.CompletedProcess:
    """Pack a one-model tree; return the finished run.

    A subprocess, because the level the CLI sets is the level the user sees.
    The model is a one-byte file — the pack sizes it from the sidecar.  The
    budget is given (``--vram``) so the run does not depend on the machine's
    GPU, and ``llama-server`` is a dummy path because the pack only has to
    resolve it.
    """
    models = tmp_path / "models" / "chat"
    models.mkdir(parents=True)
    (models / "dummy.gguf").write_bytes(b"x")
    (models / "dummy.md").write_text(_SIDECAR)
    server = tmp_path / "llama-server"
    server.write_text("#!/bin/sh\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "llama_packer",
         "--models-dir", str(tmp_path / "models"),
         "--output", str(tmp_path / "config.yaml"),
         "--llama-server", str(server),
         "--vram", "24576",
         *extra],
        cwd=str(ROOT), env=env, capture_output=True, text=True)


def _lines_at(stderr: str, level: str) -> list[str]:
    """The run's messages at *level* — the log format is "I | message"."""
    return [ln for ln in stderr.splitlines() if ln.startswith(f"{level} | ")]


def test_pack_run_is_quiet_by_default(tmp_path):
    # Validates: SPEC.md, 'Log Levels': the default level is WARNING — a run that goes well says nothing, and anything that went wrong is reported without a flag.
    proc = _run(tmp_path)
    # Silence is what a *finished* run sounds like, not what a run that died
    # on entry sounds like.
    assert proc.returncode == 0, proc.stderr
    assert _lines_at(proc.stderr, "I") == [], proc.stderr


def test_verbose_flag_lifts_the_level(tmp_path):
    # Validates: SPEC.md, 'Log Levels': -v lifts the level to INFO, -vv to DEBUG, -vvv to whatever the libraries log.
    proc = _run(tmp_path, "-v")
    assert proc.returncode == 0, proc.stderr
    assert _lines_at(proc.stderr, "I"), proc.stderr


def test_dry_run_preview_is_visible_without_a_flag(tmp_path):
    # Validates: SPEC.md, 'Log Levels': --dry-run lifts the floor to INFO whatever the verbosity, because its report — every write it withheld — is the output of the run.
    proc = _run(tmp_path, "--dry-run")
    assert proc.returncode == 0, proc.stderr
    assert [ln for ln in _lines_at(proc.stderr, "I") if "would " in ln], proc.stderr
