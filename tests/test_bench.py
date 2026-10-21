# tests/test_bench.py
"""Tests for extras/bench.py — performance history for serving engines.

A benchmark tool has one failure mode worse than being wrong: it produces a
number that looks like a measurement and is not. Everything here is about
that — refusing to report half a measurement, refusing to compare a run
against a differently-configured one, and never losing or rewriting history
that already exists.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

EXTRAS = Path(__file__).resolve().parents[1] / "extras"
sys.path.insert(0, str(EXTRAS))

_spec = importlib.util.spec_from_file_location("bench", EXTRAS / "bench.py")
bench = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = bench
_spec.loader.exec_module(bench)


# --- history ---------------------------------------------------------------


def test_history_is_appended_and_read_back(tmp_path):
    # Validates: docs/bench.md, 'The history is append-only': one JSON line per measurement, appended to the bench history
    path = tmp_path / "bench.jsonl"
    for i in range(3):
        bench.append_history({"ts": i, "tok_per_s": 10 + i}, path)
    records = bench.load_history(path)
    assert [r["tok_per_s"] for r in records] == [10, 11, 12]


def test_an_existing_record_is_never_rewritten(tmp_path):
    # Validates: docs/bench.md, 'The history is append-only': an existing record is never rewritten
    """Append-only: a benchmark history is evidence."""
    path = tmp_path / "bench.jsonl"
    bench.append_history({"ts": 1, "tok_per_s": 100}, path)
    bench.append_history({"ts": 2, "tok_per_s": 90}, path)
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["tok_per_s"] == 100


def test_a_truncated_last_line_does_not_break_the_history(tmp_path):
    # Validates: docs/bench.md, 'The history is append-only': a truncated last line does not break the history
    """A crash mid-write must cost one record, not the whole file."""
    path = tmp_path / "bench.jsonl"
    bench.append_history({"ts": 1, "tok_per_s": 100}, path)
    with path.open("a") as handle:
        handle.write('{"ts": 2, "tok_pe')      # power cut
    records = bench.load_history(path)
    assert len(records) == 1
    assert records[0]["tok_per_s"] == 100


def test_missing_history_reads_as_empty_not_an_error(tmp_path):
    # Validates: docs/bench.md, 'The history is append-only': a missing history reads as empty, not as an error
    assert bench.load_history(tmp_path / "nope.jsonl") == []


# --- what counts as comparable ---------------------------------------------


def _rec(**kw):
    base = {"engine": "llama-server", "version": "b11717", "backend": "vulkan",
            "device": "Vulkan0: AMD Radeon AI Pro R9700", "model": "a.gguf",
            "batch": 512, "ubatch": 512, "metric": "prompt_tok_s",
            "ts": 0, "tok_per_s": 1000.0}
    base.update(kw)
    return base


def test_the_same_configuration_compares():
    # Validates: docs/bench.md, 'What counts as the same measurement': the identity key is the configuration: engine, version, backend, device, model, batch, ubatch and metric
    old = _rec()
    assert bench.previous_for(_rec(ts=99, tok_per_s=900), [old]) is old


@pytest.mark.parametrize("field,value", [
    ("version", "b11658"),      # a different build is a different measurement
    ("backend", "cpu"),         # ditto a different backend
    ("device", "Vulkan0: other card"),
    ("model", "b.gguf"),
    ("batch", 256),
    ("ubatch", 256),
    ("metric", "gen_tok_s"),    # prompt vs generation are not each other
])
def test_a_changed_configuration_is_never_compared(field, value):
    assert bench.previous_for(_rec(ts=99, **{field: value}), [_rec()]) is None


def test_timestamp_and_repetitions_are_not_part_of_the_identity():
    # Validates: docs/bench.md, 'What counts as the same measurement': the timestamp and the repetition count are not part of the identity
    """Otherwise nothing would ever be comparable to anything."""
    old = _rec(ts=1, samples=3)
    assert bench.previous_for(_rec(ts=9999, samples=10), [old]) is old


def test_delta_is_relative_to_the_previous_run():
    assert round(bench.delta_pct(110, 100), 1) == 10.0
    assert round(bench.delta_pct(90, 100), 1) == -10.0
    assert bench.delta_pct(5, 0) is None      # no false "infinite" regression


def test_the_most_recent_previous_run_is_the_one_compared():
    # Validates: docs/bench.md, 'What counts as the same measurement': the comparison is against the most recent earlier record with the same key
    first = _rec(ts=1, tok_per_s=1000)
    second = _rec(ts=2, tok_per_s=800)
    third = _rec(ts=3, tok_per_s=700)
    assert bench.previous_for(_rec(ts=4, tok_per_s=690),
                              [first, second, third]) is third


# --- engine identity -------------------------------------------------------


def test_device_lines_keep_the_card_name(tmp_path):
    # Validates: docs/bench.md, 'What counts as the same measurement': the device identity is the card name and its architecture
    """Same binary and backend on two cards are two measurements."""
    binary = tmp_path / "srv"
    binary.write_text("#!/bin/sh\ncat <<'EOF'\nAvailable devices:\n"
                      "  Vulkan0: AMD Radeon AI Pro R9700 (RADV GFX1201) "
                      "(32624 MiB, 29833 MiB free)\n"
                      "  CPU0: CPU\nEOF\n")
    binary.chmod(0o755)
    devices = bench.detect_devices(binary)
    assert devices[0].startswith("Vulkan0: AMD Radeon AI Pro R9700")
    assert bench.backend_of(devices[0]) == "vulkan"
    assert bench.backend_of(devices[1]) == "cpu"


def test_a_binary_that_cannot_run_still_yields_a_version(tmp_path):
    # Validates: docs/bench.md, 'Versions and builds': a binary that cannot run still yields a version
    """An unrunnable engine is exactly when a history entry matters most."""
    broken = tmp_path / "srv"
    broken.write_bytes(b"\x7fELF not really")
    broken.chmod(0o755)
    version = bench.engine_version(broken)
    assert "unrunnable" in version or "unknown" in version


def test_version_is_read_from_the_build_number(tmp_path):
    # Validates: docs/bench.md, 'Versions and builds': the version is read from the build number
    binary = tmp_path / "llama-server"
    binary.write_text('#!/bin/sh\necho "version: 0.7.0-dev (build 11717, '
                      'commit c0fd78626)"\n')
    binary.chmod(0o755)
    assert bench.engine_version(binary) == "b11717"


# --- picking the right row out of llama-bench ------------------------------


def test_a_prompt_row_is_never_reported_as_a_generation_row():
    # Validates: docs/bench.md, 'Reading a row': a row is selected by its fields, never by its position
    """Selecting by position rather than by fields is how this goes wrong."""
    rows = [
        {"n_prompt": 512, "n_gen": 0, "avg_ts": 6400.0},
        {"n_prompt": 0, "n_gen": 64, "avg_ts": 117.0},
    ]
    assert bench.pick_row(rows, 512, 0)["avg_ts"] == 6400.0
    assert bench.pick_row(rows, 0, 64)["avg_ts"] == 117.0


def test_a_missing_row_is_an_error_not_a_zero():
    # Validates: docs/bench.md, 'Reading a row': a missing row is an error, not a zero
    rows = [{"n_prompt": 512, "n_gen": 0, "avg_ts": 6400.0}]
    with pytest.raises(bench.BenchError, match="no row"):
        bench.pick_row(rows, 0, 64)


def test_stddev_travels_with_the_mean():
    """A run whose spread can invert a 5% change has not measured 5%."""
    row = {"avg_ts": 100.0, "stddev": 9.0, "n_samples": 3,
           "model_filename": "a.gguf"}
    got = bench.record_from_row(row)
    assert got["tok_per_s"] == 100.0
    assert got["stddev"] == 9.0
    assert got["samples"] == 3


def test_empty_bench_output_is_an_error():
    # Validates: docs/bench.md, 'Reading a row': empty bench output is an error
    with pytest.raises(bench.BenchError, match="no output"):
        bench.llama_bench_rows(Path("/bin/false"), Path("/nonexistent.gguf"),
                               512, 512, 512, 64, 1)


# --- finding something to measure ------------------------------------------


def test_old_builds_are_found_so_versions_can_be_compared(tmp_path):
    # Validates: docs/bench.md, 'Versions and builds': old builds are found so versions can be compared
    """After an update the previous binary is still on disk — that is the
    only way "was it faster before" is answerable at all."""
    for build in ("llama-b11658", "llama-b11717"):
        d = tmp_path / build
        d.mkdir()
        (d / "llama-server").write_bytes(b"bin")
    found = bench.llama_binaries(tmp_path)
    assert [p.parent.name for p in found] == ["llama-b11717", "llama-b11658"]


def test_the_stable_name_is_found_when_there_is_no_build_dir(tmp_path):
    # Validates: docs/bench.md, 'Versions and builds': the stable name is found when there is no build directory
    (tmp_path / "llama-server").write_bytes(b"bin")
    assert bench.llama_binaries(tmp_path) == [tmp_path / "llama-server"]


def test_the_default_is_the_model_the_fleet_actually_runs(tmp_path):
    # Validates: docs/bench.md, 'Picking what to measure': the default is the model the fleet actually runs
    """Largest, not smallest.

    The default used to take the first small model in the config, which on
    this box is a 0.6B embedder. Measurements there do not survive contact
    with the fleet: batch depth is worth 2x on a 4B model and nothing at all
    on the 27B that real work runs on, because decode there is bound by
    streaming weights. A default that answers a different question than the
    fleet is worse than no default.
    """
    config = tmp_path / "config.yaml"
    tiny = tmp_path / "tiny.gguf"
    tiny.write_bytes(b"x")
    huge = tmp_path / "huge.gguf"
    with huge.open("wb") as handle:      # sparse: tmpfs has 32G, not 32GB of RAM
        handle.truncate(19 * 1024 ** 3)
    config.write_text(
        "models:\n"
        f"  tiny:\n    cmd: /x/llama-server -m {tiny}\n"
        f"  huge:\n    cmd: /x/llama-server -m {huge}\n")
    assert bench.default_models(str(config)) == [huge]
    # the proxy tier exists for speed, but it has to be asked for by name
    assert bench.default_models(str(config), "smallest") == [tiny]


def test_a_model_over_the_cap_falls_back_to_the_largest_that_fits(tmp_path):
    # Validates: docs/bench.md, 'Picking what to measure': a model over the size cap falls back to the largest that fits
    config = tmp_path / "config.yaml"
    mid = tmp_path / "mid.gguf"
    mid.write_bytes(b"x")
    over = tmp_path / "over.gguf"
    with over.open("wb") as handle:
        handle.truncate(80 * 1024 ** 3)
    config.write_text(
        "models:\n"
        f"  mid:\n    cmd: /x/llama-server -m {mid}\n"
        f"  over:\n    cmd: /x/llama-server -m {over}\n")
    assert bench.default_models(str(config), cap_gib=24.0) == [mid]


def test_indirect_paths_in_the_config_are_expanded(tmp_path, monkeypatch):
    # Validates: docs/bench.md, 'Picking what to measure': indirect paths in the config are expanded
    """The config is the source of truth, indirection and all."""
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    config = tmp_path / "config.yaml"
    config.write_text(
        "models:\n"
        f"  m:\n    cmd: /x/llama-server -m ${{MODELS_DIR}}/m.gguf\n")
    assert bench.default_models(str(config)) == [model]


def test_audio_entries_come_from_the_config_ids(tmp_path):
    # Validates: docs/bench.md, 'Picking what to measure': the audio entries come from the config's own ids
    config = tmp_path / "config.yaml"
    config.write_text(
        "models:\n"
        "  kokoro-82m.s: {}\n"
        "  parakeet-tdt-0.6b-v3.a: {}\n"
        "  nuextract3: {}\n")
    assert bench.audio_models(str(config)) == ["kokoro-82m.s",
                                               "parakeet-tdt-0.6b-v3.a"]


def test_the_audio_backend_is_read_out_of_the_emitted_payload(tmp_path):
    # Validates: docs/bench.md, 'Picking what to measure': the audio backend is read out of the emitted payload
    config = tmp_path / "config.yaml"
    config.write_text(
        "models:\n"
        "  kokoro-82m.s:\n"
        "    cmd: \"cat > x <<JSON\\n"
        "{\\\"host\\\":\\\"127.0.0.1\\\",\\\"backend\\\":\\\"cpu\\\"}\\nJSON\\n\"\n")
    assert bench.audio_backend(str(config)) == "cpu"


# --- the report ------------------------------------------------------------


def test_the_report_says_so_when_there_is_no_history(tmp_path, capsys):
    # Validates: docs/bench.md, 'The report': with no history, the report says so rather than printing empty numbers
    bench.print_report([], )
    out = capsys.readouterr().out
    assert "no history yet" in out


def test_the_report_lists_records_newest_last(tmp_path, capsys):
    # Validates: docs/bench.md, 'The report': records are listed newest last
    bench.print_report([
        _rec(ts=0, model="a.gguf", metric="prompt_tok_s", tok_per_s=1000),
        _rec(ts=1, model="b.gguf", metric="gen_tok_s", tok_per_s=50),
    ])
    out = capsys.readouterr().out
    assert "a.gguf" in out and "b.gguf" in out
    assert "prompt_tok_s" in out and "gen_tok_s" in out


def test_rtf_records_print_as_rtf_not_as_tok_per_second(tmp_path, capsys):
    # Validates: docs/bench.md, 'The report': the unit follows the metric name: an RTF record prints as RTF
    bench.print_report([{"ts": 0, "engine": "audiocpp_server",
                         "version": "v0.9.2", "backend": "cpu",
                         "model": "kokoro-82m.s", "metric": "tts_rtf",
                         "rtf": 0.076}])
    assert "RTF" in capsys.readouterr().out


def test_the_first_run_of_a_configuration_says_so(capsys):
    # Validates: docs/bench.md, 'The report': the first run of a configuration says so
    bench.compare_and_print(_rec(ts=1, tok_per_s=1000), [])
    assert "first recorded run" in capsys.readouterr().out


def test_the_free_memory_counter_is_not_part_of_a_device_identity():
    # Validates: docs/bench.md, 'What counts as the same measurement': the free-memory counter changes every run and is not part of the device identity
    """The bug that made every run a first run.

    --list-devices ends its line with the card's free memory, which moves
    between two runs minutes apart. Keyed on the whole line, nothing ever
    compares against anything.
    """
    before = "Vulkan0: AMD Radeon AI Pro R9700 (RADV GFX1201) (32624 MiB, 29951 MiB free)"
    after = "Vulkan0: AMD Radeon AI Pro R9700 (RADV GFX1201) (32624 MiB, 21003 MiB free)"
    assert bench.device_identity(before) == bench.device_identity(after)
    assert "29951" not in bench.device_identity(before)
    assert "R9700" in bench.device_identity(before)
    assert bench.measurement_key(_rec(device=before)) == \
        bench.measurement_key(_rec(device=after))


def test_a_different_card_is_still_a_different_measurement():
    # Validates: docs/bench.md, 'What counts as the same measurement': a different card is a different measurement
    r9700 = "Vulkan0: AMD Radeon AI Pro R9700 (RADV GFX1201) (32624 MiB, 29951 MiB free)"
    w7900 = "Vulkan0: NVIDIA GeForce RTX 7900 (32624 MiB, 29951 MiB free)"
    assert bench.measurement_key(_rec(device=r9700)) != \
        bench.measurement_key(_rec(device=w7900))


def test_a_later_run_prints_the_change(capsys):
    # Validates: docs/bench.md, 'The report': the report prints the change since the previous run of the same configuration
    bench.compare_and_print(_rec(ts=2, tok_per_s=900),
                            [_rec(ts=1, tok_per_s=1000)])
    out = capsys.readouterr().out
    assert "-10.0%" in out

def test_the_unit_follows_the_metric_name():
    # Validates: docs/bench.md, 'The report': the unit follows the metric name
    """A prompt-throughput record must not print as an RTF."""
    assert bench.metric_unit("prompt_tok_s") == ("tok_per_s", "tok/s")
    assert bench.metric_unit("gen_tok_s") == ("tok_per_s", "tok/s")
    assert bench.metric_unit("tts_rtf") == ("rtf", "RTF")
    assert bench.metric_unit("asr_rtf") == ("rtf", "RTF")


def test_a_prompt_throughput_record_prints_its_number(capsys):
    # Validates: docs/bench.md, 'The report': a prompt-throughput record prints its own number
    bench.compare_and_print(_rec(ts=2, metric="prompt_tok_s", tok_per_s=6400),
                            [_rec(ts=1, metric="prompt_tok_s",
                                  tok_per_s=6400)])
    out = capsys.readouterr().out
    assert "6400 tok/s" in out
    assert "RTF" not in out
