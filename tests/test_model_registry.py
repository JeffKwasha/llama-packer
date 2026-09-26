# tests/test_model_registry.py
"""File-identity registry: from_file factory, Model[...] lookup, from_ref,
subclass choice (incl. zero-IO test backends), and the single measured branch."""

from __future__ import annotations

import logging
import struct
from pathlib import Path

import pytest

from llama_packer.model import (
    GGUFModel,
    Model,
    SafetensorsModel,
    WeightFinder,
)
from llama_packer.vram import FitParams


@pytest.fixture(autouse=True)
def clean_registry():
    Model.clear_registry()
    Model._file_classes.clear()
    yield
    Model.clear_registry()
    Model._file_classes.clear()


def _gguf(path: Path, arch="qwen3", ctx=131072) -> Path:
    kb = b""
    for k, v in (("general.architecture", arch),
                 (f"{arch}.context_length", ctx)):
        kb += struct.pack("<Q", len(k)) + k.encode()
        if isinstance(v, str):
            kb += struct.pack("<I", 8) + struct.pack("<Q", len(v)) + v.encode()
        else:
            kb += struct.pack("<I", 4) + struct.pack("<I", v)
    path.write_bytes(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
                     + struct.pack("<Q", 2) + kb)
    return path


def test_from_file_identity_and_subclass(tmp_path):
    g = _gguf(tmp_path / "a.gguf")
    s = tmp_path / "b.safetensors"
    s.write_bytes(struct.pack("<Q", 2) + b"{}")
    b = tmp_path / "c.bin"
    b.write_bytes(b"\x00" * 8)

    assert Model.from_file(g) is Model.from_file(g)
    assert isinstance(Model.from_file(g), GGUFModel)
    assert isinstance(Model.from_file(s), SafetensorsModel)
    assert type(Model.from_file(b)) is Model
    assert Model.from_file(tmp_path / "missing.gguf") is None


def test_zero_io_backend_via_register_file_class(tmp_path):
    g = _gguf(tmp_path / "z.gguf")

    class FakeGGUF(GGUFModel):
        @property
        def arch(self):
            return "fake-arch"

        @property
        def max_context(self):
            return 4242

        @property
        def kind(self):
            return "text"

    Model.register_file_class(".gguf", FakeGGUF)
    m = Model.from_file(g)
    assert isinstance(m, FakeGGUF)
    assert m.arch == "fake-arch"
    assert m.max_context == 4242
    assert m.kind == "text"


def test_getitem_exact_and_pattern(tmp_path):
    g1 = _gguf(tmp_path / "qwen3-8b-q4.gguf")
    g2 = _gguf(tmp_path / "qwen3-32b-q4.gguf")
    Model.from_file(g1)
    Model.from_file(g2)

    assert [m.stem for m in Model["qwen3-8b-q4.gguf"]] == ["qwen3-8b-q4"]  # type: ignore[misc]
    got = sorted(m.stem for m in Model["qwen3-*-q4.gguf"])  # type: ignore[misc]
    assert got == ["qwen3-32b-q4", "qwen3-8b-q4"]
    assert Model["nope-*.gguf"] == []  # type: ignore[misc]
    assert Model[""] == []  # type: ignore[misc]


def test_getitem_dedups_sidecar_and_canonical(tmp_path):
    g = _gguf(tmp_path / "dup.gguf")
    md = tmp_path / "dup.md"
    md.write_text("---\nmodel: dup.gguf\n---\n")
    Model(md, {"model": "dup.gguf"})
    assert Model.from_file(g) is not None
    # sidecar-bound claimant shadows the canonical file instance
    assert [m.stem for m in Model["dup.gguf"]] == ["dup"]  # type: ignore[misc]


def test_header_read_once_shared_across_claimants(tmp_path, caplog):
    from llama_packer import utils

    utils._GGUF_PROBE_CACHE.clear()
    _gguf(tmp_path / "shared.gguf")
    md1 = tmp_path / "one.md"
    md1.write_text("---\nmodel: shared.gguf\n---\n")
    md2 = tmp_path / "two.md"
    md2.write_text("---\nmodel: shared.gguf\n---\n")

    with caplog.at_level(logging.INFO, logger="llama_packer.utils"):
        m1 = Model(md1, {"model": "shared.gguf"})
        m2 = Model(md2, {"model": "shared.gguf"})
        assert m1.arch == "qwen3" == m2.arch
        assert m1.gguf_context_length == 131072 == m2.gguf_context_length
    reads = [r for r in caplog.records if "reading GGUF header" in r.message]
    # one probe walk + one context walk, shared by both claimants
    assert len(reads) == 2


def test_from_ref_with_stub_finder(tmp_path):
    d = tmp_path / "models"
    d.mkdir()
    g = _gguf(d / "m.gguf")

    class StubFinder(WeightFinder):
        def find_local(self, name, dirs):
            return g if name == "m.gguf" else None

        def snapshot_exact(self, repo, name, hf_home=None, *a, **k):
            return g if (repo, name) == ("org/r", "m.gguf") else None

        def match_snapshot(self, repo, pattern, hf_home=None, *a, **k):
            return [g] if (repo, pattern) == ("org/r", "*m*.gguf") else []

    f = StubFinder()

    def _resolve(**kw):
        hit = Model.from_ref(kw.pop("ref"), finder=f, **kw)
        assert hit is not None and hit.gguf_path == g
        return hit

    _resolve(ref="m.gguf", anchors=[d])
    _resolve(ref="m.gguf", anchors=[], hf_repo="org/r")
    _resolve(ref="*m*.gguf", anchors=[], hf_repo="org/r")
    _resolve(ref="hub:org/r:m.gguf")
    assert Model.from_ref("missing.gguf", anchors=[d], finder=f) is None


def test_sidecar_model_string_uses_from_ref(tmp_path):
    _gguf(tmp_path / "w.gguf")
    md = tmp_path / "w.md"
    md.write_text("---\nmodel: w.gguf\n---\n")
    m = Model(md, {"model": "w.gguf"})
    assert m.gguf_path is not None and m.gguf_path.name == "w.gguf"
    assert isinstance(m._file, GGUFModel)


def test_measured_file_sync_and_validity(tmp_path, caplog):
    g = _gguf(tmp_path / "v.gguf")
    md = tmp_path / "v.md"
    md.write_text("---\nname: v\n---\n")
    m = Model(md, {"name": "v", "model": "v.gguf"})

    assert not m.measured_file_valid()
    with caplog.at_level(logging.INFO, logger="llama_packer.utils"):
        m.sync_measured_file()
    assert m.measured_file_valid()
    content = md.read_text()
    assert "derived:" in content
    assert "arch: qwen3" in content

    # second instance over the same files: no header re-read, still valid
    caplog.clear()
    m2 = Model(md, {"name": "v", "model": "v.gguf"})
    with caplog.at_level(logging.INFO, logger="llama_packer.utils"):
        m2.sync_measured_file()
    assert m2.measured_file_valid()
    assert not [r for r in caplog.records if "reading GGUF header" in r.message]

    # touching the file invalidates the block
    g.write_bytes(g.read_bytes() + b"\x00")
    assert not m2.measured_file_valid()


def test_legacy_fit_params_block_still_read(make_model, fit_params_block):
    m = make_model("lp", **{"derived": fit_params_block})
    saved = m.vram.saved_for("q8_0")
    assert saved is not None and saved.model_mib == 10000


def test_snapshot_listing_cached_by_mtime(tmp_path):
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "a.gguf").write_bytes(b"x")

    class Snappy(WeightFinder):
        def snapshot_dir(self, repo, hf_home=None, mode=None):
            return snap

    s = Snappy()
    _, first = s.snapshot_files("org/r")
    _, second = s.snapshot_files("org/r")
    assert first == second == ["a.gguf"]
    (snap / "b.gguf").write_bytes(b"y")
    import os
    import time
    # force a visible mtime change (listing validity is mtime-keyed)
    ns = time.time_ns() + 10_000_000
    os.utime(snap, ns=(ns, ns))
    _, third = s.snapshot_files("org/r")
    assert third == ["a.gguf", "b.gguf"]


def test_fit_params_round_trip_through_measured(make_model):
    m = make_model("rt", context_length=32768)
    params = FitParams(1000, 0.5, 150.0, 100, "llama-server", "q8_0")
    m.vram._persist(params)
    saved = m.vram.saved_for("q8_0")
    assert saved is not None and saved.model_mib == 1000
    assert m.measured_block()["source"] == "llama-server"


def test_min_context_is_a_consumed_field(make_model):
    # documented sidecar key: consumed by the builder, never leaked to
    # clients as metadata
    m = make_model("mc", min_context=65536)
    assert m.frontmatter["min_context"] == 65536
    assert "min_context" not in m.pass_through_metadata()


def test_audio_keys_are_consumed_fields(make_model, caplog):
    # documented sidecar keys (models_AGENTS.md): vram_mb pins fixed-overhead
    # VRAM (vram.py/writer.py), audio_cpp drives AudioCppBackend.build_cmd.
    # Neither may warn nor leak to clients as metadata.
    import logging
    m = make_model("a", vram_mb=4096,
                   audio_cpp={"family": "qwen3_asr", "task": "asr"})
    with caplog.at_level(logging.WARNING):
        meta = m.pass_through_metadata()
    assert "vram_mb" not in meta
    assert "audio_cpp" not in meta
    assert not [r for r in caplog.records if "unhandled frontmatter" in r.message]


def test_removed_keys_warn_and_pass_through(make_model, caplog):
    # attention/kv_cache/tool_args/targets/sidecar-spare are accepted but
    # unread: using one warns (fail loud) and the value flows to metadata
    # rather than silently doing nothing.
    import logging
    m = make_model("r", targets=["x"], spare="1G")
    with caplog.at_level(logging.WARNING):
        meta = m.pass_through_metadata()
    assert meta["targets"] == ["x"]
    assert any("unhandled frontmatter" in r.message for r in caplog.records)


def test_mtp_draft_p_min_consumed(make_model, caplog):
    import logging
    m = make_model("p", mtp_draft_p_min=0.8)
    assert m.mtp_draft_p_min == 0.8
    with caplog.at_level(logging.WARNING):
        meta = m.pass_through_metadata()
    assert "mtp_draft_p_min" not in meta
    assert not [r for r in caplog.records if "unhandled frontmatter" in r.message]
