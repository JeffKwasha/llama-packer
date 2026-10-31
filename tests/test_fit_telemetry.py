# tests/test_fit_telemetry.py
"""Fit-params failure evidence: the artifact a later diagnosis reads.

A crash prints its cause to stderr and exits 1; a warning carrying only the
exit code throws that cause away. These pin the artifact, its journal row,
and the dedupe that keeps a repeated invocation from spamming both.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from llama_packer import gpu_state, vram


@pytest.fixture
def budget(make_model):
    return vram.VramBudget(make_model("embed-model", role="embeddings"))


@pytest.fixture
def artifact_dir():
    return gpu_state.cache_dir() / "fit-params-failures"


def _artifacts(artifact_dir):
    return sorted(artifact_dir.iterdir())


def _journal_rows():
    path = gpu_state.cache_dir() / "measure-journal.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def _run(budget, **kwargs):
    return budget.fit_params(fit_bin="/opt/llama/tools/llama-b1/llama-fit-params",
                             fit_ctx=4096, cache_type="q8_0", parallel=1,
                             llama_args="--flash-attn on", **kwargs)


def test_fit_crash_records_argv_and_stderr(budget, artifact_dir, monkeypatch,
                                           caplog):
    # Validates: requirement 94 — a crash keeps argv, model facts, exit code and the full log; the warning names the artifact.
    def fake_run(cmd, **kwargs):
        assert kwargs["capture_output"] is True
        return subprocess.CompletedProcess(
            cmd, 1, "stdout noise\n",
            "llama_model_load: error loading model\nfailed to load model from file\n")

    monkeypatch.setattr(vram.subprocess, "run", fake_run)
    assert _run(budget) is None

    [artifact] = _artifacts(artifact_dir)
    text = artifact.read_text()
    assert "outcome: crash" in text
    assert "exit: 1" in text
    assert "llama-fit-params" in text and "--cache-type-k q8_0" in text
    assert "-c 4096" in text and "--flash-attn on" in text
    assert "failed to load model from file" in text
    assert "stdout noise" in text

    row = _journal_rows()[-1]
    assert row["mode"] == "fit"
    assert row["outcome"] == "crash"
    assert row["exit"] == 1
    assert row["artifact"] == str(artifact)
    assert "failed to load model from file" in row["stderr_tail"]
    assert str(artifact) in caplog.text
    assert "failed to load model from file" in caplog.text


def test_fit_crash_recorded_once_per_invocation(budget, artifact_dir,
                                                monkeypatch, caplog):
    # Validates: requirement 94 — one artifact per distinct failing invocation; repeats warn once, not per call.
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, "", "error: boom\n")

    monkeypatch.setattr(vram.subprocess, "run", fake_run)
    _run(budget)
    _run(budget)

    assert len(_artifacts(artifact_dir)) == 1
    assert caplog.text.count("fit-params crashed") == 1


def test_fit_timeout_records_argv(budget, artifact_dir, monkeypatch):
    # Validates: requirement 94 — a timeout records the command it was running, even though there is no output to keep.
    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 60)

    monkeypatch.setattr(vram.subprocess, "run", fake_run)
    assert _run(budget) is None

    [artifact] = _artifacts(artifact_dir)
    text = artifact.read_text()
    assert "outcome: timeout" in text
    assert "argv: /opt/llama/tools/llama-b1/llama-fit-params" in text
    assert "-c 4096" in text


def test_fit_unparseable_records_output(budget, artifact_dir, monkeypatch):
    # Validates: requirement 94 — an exit-0 run whose log will not parse keeps the output too, since that is the whole evidence.
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, "not a fit-params log\n",
                                           "")

    monkeypatch.setattr(vram.subprocess, "run", fake_run)
    assert _run(budget) is None

    [artifact] = _artifacts(artifact_dir)
    text = artifact.read_text()
    assert "outcome: unparseable" in text
    assert "exit: 0" in text
    assert "not a fit-params log" in text
