# tests/test_model_resolution.py
"""Contract tests for Model reference resolution, serialization and accessors.

Focus: observable resolution semantics (absolute/anchor/hub/glob), the
canonical-instance registry, sidecar round-tripping, and the header-derived
safetensors path.  Header parsing itself is stubbed so these exercise the
resolution contracts, not the byte layout.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llama_packer import utils
from llama_packer.model import Model, WeightFinder, _DEFAULT_CONTEXT_LENGTH


@pytest.fixture(autouse=True)
def _clean_model_registry():
    Model.clear_registry()
    yield
    Model.clear_registry()


class FakeFinder(WeightFinder):
    """WeightFinder stand-in: canned local/hub hits over real dispatch."""

    def __init__(self, local=None, matches=None, exact=None):
        super().__init__()
        self._local = local
        self._matches = list(matches or [])
        self._exact = exact

    def find_local(self, name, dirs):
        return self._local

    def match_snapshot(self, repo, pattern, hf_home=None, *a, **k):
        return list(self._matches)

    def snapshot_exact(self, repo, name, hf_home=None, *a, **k):
        return self._exact

    def snapshot_files(self, repo, hf_home=None, mode=None):
        return (None, [])


# ── from_ref resolution ───────────────────────────────────────────────────

def test_from_ref_absolute_path(tmp_path):
    f = tmp_path / "m.gguf"
    f.write_bytes(b"x")
    m = Model.from_ref(str(f))
    assert m is not None and m.gguf_path.samefile(f)


def test_from_ref_missing_absolute_path_is_none(tmp_path):
    assert Model.from_ref(str(tmp_path / "nope.gguf")) is None


def test_from_ref_searches_anchor_dirs(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    (d / "w.gguf").write_bytes(b"x")
    m = Model.from_ref("w.gguf", anchors=[d])
    assert m is not None and m.stem == "w"


def test_from_ref_hub_form_and_malformed(tmp_path):
    f = tmp_path / "w.gguf"
    f.write_bytes(b"x")
    assert Model.from_ref("hub:org/repo:w.gguf",
                          finder=FakeFinder(exact=f)) is not None
    assert Model.from_ref("hub:onlyrepo") is None       # missing ':file'


def test_from_ref_glob_requires_single_match(tmp_path):
    f = tmp_path / "w.gguf"
    f.write_bytes(b"x")
    assert Model.from_ref("*.gguf", hf_repo="org/repo",
                          finder=FakeFinder(matches=[f])) is not None
    assert Model.from_ref("*.gguf", hf_repo="org/repo",
                          finder=FakeFinder(matches=[f, f])) is None


def test_from_ref_unresolvable_is_none():
    assert Model.from_ref("nope.gguf", finder=FakeFinder()) is None


# ── canonical registry ────────────────────────────────────────────────────

def test_find_by_gguf_returns_canonical_instance(tmp_path):
    f = tmp_path / "m.gguf"
    f.write_bytes(b"x")
    m = Model.from_file(f)
    assert Model.find_by_gguf(f) is m
    assert Model.find_by_gguf(tmp_path / "other.gguf") is None


def test_find_by_gguf_resolves_symlink_to_canonical(tmp_path):
    real = tmp_path / "real.gguf"
    real.write_bytes(b"x")
    link = tmp_path / "link.gguf"
    link.symlink_to(real)
    m = Model.from_file(real)
    assert Model.find_by_gguf(link) is m


def test_get_or_create_companion_is_idempotent(tmp_path):
    f = tmp_path / "mmproj.gguf"
    f.write_bytes(b"x")
    assert Model._get_or_create_companion(f) is Model._get_or_create_companion(f)


# ── _resolve_ref / _resolve_hub_ref wrappers ──────────────────────────────

def test_resolve_ref_finds_file_beside_sidecar(make_model, tmp_path):
    m = make_model("m")
    (tmp_path / "aux.gguf").write_bytes(b"x")
    p = m._resolve_ref("aux.gguf")
    assert p is not None and p.samefile(tmp_path / "aux.gguf")
    assert m._resolve_ref("missing.gguf") is None


def test_resolve_ref_searches_parent_dir(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (tmp_path / "aux.gguf").write_bytes(b"x")
    m = object.__new__(Model)
    m.md_path = sub / "m.md"
    m.frontmatter = {}
    m._hf_home = None
    p = m._resolve_ref("aux.gguf")
    assert p is not None and p.samefile(tmp_path / "aux.gguf")


def test_resolve_hub_ref_falls_back_to_pattern_hint(make_model, monkeypatch):
    m = make_model("m", hf_repo="org/repo")
    seen: list[str] = []

    class Hit:
        gguf_path = Path("/hub/x.gguf")

    def fake(cls, ref, **kw):
        seen.append(ref)
        return None if ref == "missing.bin" else Hit()

    monkeypatch.setattr(Model, "from_ref", classmethod(fake))
    assert m._resolve_hub_ref("missing.bin", pattern_hint="*.gguf") == \
        Path("/hub/x.gguf")
    assert seen == ["missing.bin", "*.gguf"]


def test_resolve_hub_ref_without_hint_returns_none(make_model, monkeypatch):
    m = make_model("m", hf_repo="org/repo")
    monkeypatch.setattr(Model, "from_ref", classmethod(lambda cls, *a, **k: None))
    assert m._resolve_hub_ref("missing.bin") is None


def test_resolve_hub_ref_hint_requires_repo(make_model, monkeypatch):
    m = make_model("m")  # no hf_repo
    seen: list[str] = []

    class Hit:
        gguf_path = Path("/hub/x.gguf")

    def fake(cls, ref, **kw):
        seen.append(ref)
        return None if ref == "missing.bin" else Hit()

    monkeypatch.setattr(Model, "from_ref", classmethod(fake))
    assert m._resolve_hub_ref("missing.bin", pattern_hint="*.gguf") is None
    assert seen == ["missing.bin"]


# ── write_md ──────────────────────────────────────────────────────────────

def test_write_md_round_trips_all_frontmatter(make_model):
    m = make_model("m", strengths=["fast"], custom_meta={"a": 1})
    m.write_md()
    assert utils.parse_frontmatter(m.md_path) == m.frontmatter


def test_write_md_to_alternate_path(make_model, tmp_path):
    m = make_model("m", license="apache-2.0")
    out = tmp_path / "copy.md"
    m.write_md(output_path=out)
    assert out.exists()
    assert utils.parse_frontmatter(out) == m.frontmatter


# ── value accessors ───────────────────────────────────────────────────────

def test_context_length_override_and_default(make_model):
    assert make_model("a", context_length=8192).context_length == 8192
    bare = object.__new__(Model)
    bare.frontmatter = {}
    assert bare.context_length == _DEFAULT_CONTEXT_LENGTH


def test_cli_args_defaults_empty(make_model):
    assert make_model("m").cli_args == ""
    assert make_model("n", cli_args="--x 1").cli_args == "--x 1"


def test_freethought_coercion(make_model):
    assert make_model("a").freethought is None
    assert make_model("b", freethought=0.7).freethought == 0.7
    assert make_model("c", freethought="not-a-number").freethought is None


def test_size_mb_and_vram_mb_track_file(make_model):
    m = make_model("m")
    m.gguf_path.write_bytes(b"x" * (3 * 1024 * 1024))
    assert m.size_mb == 3
    assert m.vram_mb == m.size_mb == 3


def test_size_mb_zero_without_file():
    m = object.__new__(Model)
    m.gguf_path = None
    assert m.size_mb == 0


def test_mmproj_size_mb_zero_without_companion(make_model):
    assert make_model("m").mmproj_size_mb == 0


# ── SafetensorsModel path ─────────────────────────────────────────────────

def test_safetensors_kind_and_numeric_estimate(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "sniff_safetensors", lambda p: "text")
    calls: list[str] = []
    monkeypatch.setattr(utils, "estimate_safetensors",
                        lambda p, ct: (calls.append(ct) or (1000, 50.0)))
    f = tmp_path / "m.safetensors"
    f.write_bytes(b"{}")
    m = Model.from_file(f)
    assert m is not None
    assert m.kind == "text"
    # Estimate is computed once per cache_type (memoized on the header).
    assert m.safetensors_numbers("q8_0") == (1000, 50.0)
    assert m.safetensors_numbers("q8_0") == (1000, 50.0)
    assert calls == ["q8_0"]
    assert m.safetensors_numbers("f16") == (1000, 50.0)
    assert calls == ["q8_0", "f16"]


def test_safetensors_numbers_none_on_estimate_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "sniff_safetensors", lambda p: "unknown")

    def boom(p, ct):
        raise ValueError("bad header")

    monkeypatch.setattr(utils, "estimate_safetensors", boom)
    f = tmp_path / "m.safetensors"
    f.write_bytes(b"{}")
    m = Model.from_file(f)
    assert m.safetensors_numbers("q8_0") is None


def test_base_model_safetensors_numbers_is_none():
    # A non-safetensors canonical file has no numeric estimate.
    m = object.__new__(Model)
    m._file = None
    assert m.safetensors_numbers() is None
