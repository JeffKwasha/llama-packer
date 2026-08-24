# tests/test_discovery.py
"""Multi-directory discovery, role-mapped subdirs, and link deduplication."""

from __future__ import annotations

import logging

from llama_packer.model import Model
from llama_packer.utils import classify_models, hf_snapshot_file


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


def test_classify_role_dirs_and_whitelist(tmp_path):
    root = tmp_path / "models"
    for sub, files in {
        "": ["plain.gguf"],
        "chat": ["c1.gguf"],
        "vision": ["v1.gguf", "v1-mmproj.gguf"],
        "doc": ["d1.gguf"],
        "embed": ["e1.gguf"],
        "embed/jina-v5": ["e2.gguf"],  # nested keeps the top-level role
        "rerank": ["r1.gguf"],
        "img": ["sd-checkpoint.gguf"],   # not in the map: skipped
        "misc": ["junk.gguf"],           # not in the map: skipped
    }.items():
        d = root / sub
        d.mkdir(parents=True)
        for f in files:
            (d / f).write_bytes(b"x")

    classified = dict()
    for p, kind in classify_models([root]):
        classified[str(p.relative_to(root))] = kind

    assert classified["plain.gguf"] == "chat"          # root-level default
    assert classified["chat/c1.gguf"] == "chat"
    assert classified["vision/v1.gguf"] == "chat"
    assert classified["vision/v1-mmproj.gguf"] == "mmproj"
    assert classified["doc/d1.gguf"] == "chat"
    assert classified["embed/e1.gguf"] == "embeddings"
    assert classified["embed/jina-v5/e2.gguf"] == "embeddings"
    assert classified["rerank/r1.gguf"] == "rerank"
    assert "img/sd-checkpoint.gguf" not in classified
    assert "misc/junk.gguf" not in classified


def test_classify_dir_roles_override(tmp_path):
    root = tmp_path / "models"
    (root / "ocr").mkdir(parents=True)
    (root / "ocr" / "o1.gguf").write_bytes(b"x")
    (root / "img").mkdir()
    (root / "img" / "sd.gguf").write_bytes(b"x")

    classified = {str(p.relative_to(root)): k
                  for p, k in classify_models([root], dir_roles={"ocr": "chat", "img": "chat"})}
    assert classified["ocr/o1.gguf"] == "chat"
    assert classified["img/sd.gguf"] == "chat"


def test_embed_rerank_orphans_get_role_stubs(tmp_path):
    root = tmp_path / "models"
    (root / "embed").mkdir(parents=True)
    (root / "embed" / "e1.gguf").write_bytes(b"x")
    (root / "rerank").mkdir()
    (root / "rerank" / "r1.gguf").write_bytes(b"x")

    models = Model.from_dir([root])  # stubs on
    by_stem = {m.stem: m for m in models}
    assert by_stem["e1"].role == "embeddings"
    assert by_stem["r1"].role == "rerank"
    # Stub files were written with the role baked into frontmatter.
    assert "role: embeddings" in (root / "embed" / "e1.md").read_text()
    assert "role: rerank" in (root / "rerank" / "r1.md").read_text()


def _hf_tree(tmp_path, repo="org/repo", rev="abc123", files=("model.gguf",), with_ref=True):
    hub = tmp_path / "hf" / "hub"
    snap = hub / f"models--{repo.replace('/', '--')}" / "snapshots" / rev
    snap.mkdir(parents=True)
    for f in files:
        (snap / f).write_bytes(b"x")
    if with_ref:
        refs = hub / f"models--{repo.replace('/', '--')}" / "refs"
        refs.mkdir(parents=True)
        (refs / "main").write_text(rev)
    return tmp_path / "hf"


def test_hf_snapshot_file_refs_main(tmp_path):
    hf_home = _hf_tree(tmp_path)
    hit = hf_snapshot_file("org/repo", "model.gguf", hf_home)
    assert hit is not None and hit.name == "model.gguf"
    assert "abc123" in str(hit)
    assert hf_snapshot_file("org/repo", "missing.gguf", hf_home) is None
    assert hf_snapshot_file("org/other", "model.gguf", hf_home) is None


def test_hf_snapshot_file_no_ref_single_snapshot(tmp_path):
    hf_home = _hf_tree(tmp_path, with_ref=False)
    hit = hf_snapshot_file("org/repo", "model.gguf", hf_home)
    assert hit is not None and hit.name == "model.gguf"


def test_model_resolves_gguf_from_hf_cache(tmp_path):
    hf_home = _hf_tree(tmp_path, files=("Qwen-x.Q4_K_M.gguf", "Qwen-x-mmproj.gguf"))
    md_path = tmp_path / "qwen.md"
    fm = {"name": "qwen", "model": "Qwen-x.Q4_K_M.gguf", "hf_repo": "org/repo"}
    m = Model(md_path, fm, hf_home=hf_home)
    assert m.gguf_path is not None
    assert m.gguf_path.name == "Qwen-x.Q4_K_M.gguf"
    # Companion resolution searches the HF snapshot dir too.
    assert m.mmproj is not None and m.mmproj.stem == "Qwen-x-mmproj"


def test_model_explicit_model_missing_everywhere_raises(tmp_path):
    import pytest
    hf_home = _hf_tree(tmp_path)
    md_path = tmp_path / "qwen.md"
    fm = {"name": "qwen", "model": "nope.gguf", "hf_repo": "org/repo"}
    with pytest.raises(ValueError, match="not found"):
        Model(md_path, fm, hf_home=hf_home)


def test_hf_snapshot_file_glob_exact_wins(tmp_path):
    hf_home = _hf_tree(tmp_path, files=("model.gguf", "mmproj-F16.gguf"))
    assert hf_snapshot_file("org/repo", "model.gguf", hf_home).name == "model.gguf"
    assert hf_snapshot_file("org/repo", "mmproj*.gguf", hf_home).name == "mmproj-F16.gguf"


def test_hf_snapshot_file_ambiguous_glob_is_none(tmp_path, caplog):
    from llama_packer.utils import hf_snapshot_dir
    hf_home = _hf_tree(tmp_path, files=("mmproj-F16.gguf", "mmproj-BF16.gguf"))
    with caplog.at_level(logging.WARNING):
        assert hf_snapshot_file("org/repo", "mmproj*.gguf", hf_home) is None
    assert any("ambiguous" in r.message for r in caplog.records)
    assert hf_snapshot_dir("org/repo", hf_home).is_dir()


def test_model_companion_mmproj_from_hub_by_name_and_glob(tmp_path):
    # Sidecar references the snapshot filename (not a local symlink name).
    hf_home = _hf_tree(tmp_path, files=("Dirk-Q4_K_M.gguf", "mmproj-F16.gguf"))
    md_path = tmp_path / "dirk.md"
    fm = {"name": "dirk", "model": "Dirk-Q4_K_M.gguf",
          "mmproj": "mmproj-F16.gguf", "hf_repo": "org/repo"}
    m = Model(md_path, fm, hf_home=hf_home)
    assert m.mmproj is not None and m.mmproj.gguf_path.name == "mmproj-F16.gguf"

    # Same via glob — covers repos that name it mmproj-model-f16 etc.
    fm2 = {**fm, "name": "dirk2", "mmproj": "mmproj*.gguf"}
    m2 = Model(md_path.parent / "dirk2.md", fm2, hf_home=hf_home)
    assert m2.mmproj is not None


def test_model_companion_fuzzy_from_hub_when_local_absent(tmp_path):
    hf_home = _hf_tree(tmp_path, files=("Qwen-x.Q4_K_M.gguf", "Qwen-x-mmproj.gguf"))
    md_path = tmp_path / "qwen.md"
    fm = {"name": "qwen", "model": "Qwen-x.Q4_K_M.gguf", "hf_repo": "org/repo"}
    m = Model(md_path, fm, hf_home=hf_home)
    assert m.mmproj is not None and m.mmproj.gguf_path.name == "Qwen-x-mmproj.gguf"


def test_model_companion_cross_repo_hub_ref(tmp_path):
    hf_home = _hf_tree(tmp_path, files=("model.gguf",))
    _hf_tree(tmp_path / "x", repo="other/vision-proj", files=("mmproj-F16.gguf",))
    import shutil
    shutil.move(tmp_path / "x" / "hf" / "hub" / "models--other--vision-proj",
                tmp_path / "hf" / "hub" / "models--other--vision-proj")
    md_path = tmp_path / "m.md"
    fm = {"name": "m", "model": "model.gguf", "hf_repo": "org/repo",
          "mmproj": "hub:other/vision-proj:mmproj-F16.gguf"}
    m = Model(md_path, fm, hf_home=hf_home)
    assert m.mmproj is not None
    assert "models--other--vision-proj" in str(m.mmproj.gguf_path)


def test_model_speculative_from_hub(tmp_path):
    hf_home = _hf_tree(tmp_path,
                       files=("big-Q4_K_M.gguf", "big-mtp.gguf"))
    md_path = tmp_path / "big.md"
    fm = {"name": "big", "model": "big-Q4_K_M.gguf", "hf_repo": "org/repo",
          "speculative": "big-mtp.gguf"}
    m = Model(md_path, fm, hf_home=hf_home)
    assert m.mtp is not None


def test_modelignore_excludes_files(tmp_path):
    from llama_packer.utils import load_model_ignore
    root = tmp_path / "models"
    (root / "vision").mkdir(parents=True)
    (root / ".modelignore").write_text(
        "# comment\nR3-rerank\nadetailer*\n*.tmp.gguf\n\n")
    (root / "vision" / "keep.md").write_text(_sidecar("Keep"))
    (root / "vision" / "adetailer-x.md").write_text(_sidecar("Bad"))
    sub = root / "vision" / "R3-rerank"
    sub.mkdir()
    (sub / "model.md").write_text(_sidecar("Bad2"))

    assert load_model_ignore(root) == ["R3-rerank", "adetailer*", "*.tmp.gguf"]
    kinds = classify_models(root)
    names = {p.name for p, _ in kinds}
    assert "keep.md" in names
    assert "adetailer-x.md" not in names
    assert "model.md" not in names
