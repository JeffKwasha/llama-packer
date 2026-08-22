# tests/test_discovery.py
"""Multi-directory discovery and symlink deduplication."""

from __future__ import annotations

from llama_packer.model import Model


def _sidecar(name: str) -> str:
    return f"---\nname: {name}\n---\n"


def test_from_dir_multiple_dirs(tmp_path):
    d1 = tmp_path / "m1"
    d2 = tmp_path / "m2"
    d1.mkdir(); d2.mkdir()
    (d1 / "foo.gguf").write_bytes(b"x")
    (d1 / "foo.md").write_text(_sidecar("Foo"))
    (d2 / "bar.gguf").write_bytes(b"x")
    (d2 / "bar.md").write_text(_sidecar("Bar"))

    models = Model.from_dir([d1, d2], generate_stubs=False)
    assert {m.stem for m in models} == {"foo", "bar"}


def test_from_dir_dedupes_symlinks_to_same_file(tmp_path):
    d1 = tmp_path / "m1"
    d2 = tmp_path / "m2"
    d1.mkdir(); d2.mkdir()
    (d1 / "foo.gguf").write_bytes(b"x")
    (d1 / "foo.md").write_text(_sidecar("Foo"))
    (d2 / "bar.gguf").symlink_to(d1 / "foo.gguf")
    (d2 / "bar.md").write_text(_sidecar("Bar"))

    models = Model.from_dir([d1, d2], generate_stubs=False)
    assert [m.stem for m in models] == ["foo"]
