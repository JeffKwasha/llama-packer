# tests/test_discovery.py
"""Multi-directory discovery, role-mapped subdirs, and link deduplication."""

from __future__ import annotations

import logging

from llama_packer.model import Model
from llama_packer.utils import hf_snapshot_file


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


def test_role_dirs_whitelist_and_companions(tmp_path):
    root = tmp_path / "models"
    for sub, files in {
        "": ["plain.gguf"],
        "chat": ["c1.gguf"],
        "vision": ["v1.gguf", "v1-mmproj.gguf"],  # mmproj is a companion, not a main
        "doc": ["d1.gguf"],
        "embed": ["e1.gguf"],
        "embed/jina-v5": ["e2.gguf"],  # nested keeps the top-level role
        "rerank": ["r1.gguf"],
        "img": ["sd-checkpoint.gguf"],   # not in the map: skipped
        "misc": ["junk.gguf"],           # not in the map: skipped
    }.items():
        d = root / sub
        d.mkdir(parents=True, exist_ok=True)
        for f in files:
            (d / f).write_bytes(b"x")

    models = Model.from_dir(root, generate_stubs=False)
    by_stem = {m.stem: m for m in models}
    assert set(by_stem) == {"plain", "c1", "v1", "d1", "e1", "e2", "r1"}
    assert by_stem["plain"].role == "chat"      # root-level default
    assert by_stem["v1"].role == "chat"         # vision colocates with chat
    assert by_stem["e1"].role == "embeddings"
    assert by_stem["e2"].role == "embeddings"   # nested keeps top-level role
    assert by_stem["r1"].role == "rerank"


def test_dir_roles_override(tmp_path):
    from llama_packer.discover import discover
    root = tmp_path / "models"
    (root / "ocr").mkdir(parents=True)
    (root / "ocr" / "o1.gguf").write_bytes(b"x")
    (root / "img").mkdir()
    (root / "img" / "sd.gguf").write_bytes(b"x")

    models = discover([root], generate_stubs=False,
                      dir_roles={"ocr": "chat", "img": "chat"})
    assert sorted(m.stem for m in models) == ["o1", "sd"]
    assert all(m.role == "chat" for m in models)


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
    # Stub sidecars are EMPTY (no inferred data) — role is derived from the
    # directory at discovery time, not baked into the file.
    assert (root / "embed" / "e1.md").read_text() == "---\n---\n\n# e1\n"
    assert (root / "rerank" / "r1.md").read_text() == "---\n---\n\n# r1\n"


def test_sidecar_type_field_classifies_outside_role_dirs(tmp_path):
    root = tmp_path / "models"
    (root / "chat").mkdir(parents=True)
    (root / "chat" / "e.gguf").write_bytes(b"x")
    (root / "chat" / "e.md").write_text(
        "---\nname: E\ntype: embedding\n---\n")

    models = Model.from_dir([root], generate_stubs=False)
    assert [m.role for m in models] == ["embeddings"]


def test_unmapped_depth1_dirs_are_skipped(tmp_path, caplog):
    import logging
    from llama_packer.discover import discover
    root = tmp_path / "models"
    for sub, served in (("chat", True), ("img", False), ("misc", False)):
        d = root / sub
        d.mkdir(parents=True)
        (d / f"{sub}.gguf").write_bytes(b"x")
    (root / "chat" / "img").mkdir()          # nested unmapped name is fine:
    (root / "chat" / "img" / "deep.gguf").write_bytes(b"x")  # depth-1 decides

    with caplog.at_level(logging.INFO):
        models = discover(root, generate_stubs=False)
    stems = {m.stem for m in models}
    assert stems == {"chat", "deep"}
    assert any("img/, misc/" in r.message or "misc/, img/" in r.message
               for r in caplog.records)


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
    m.resolve_companions()
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
    m.resolve_companions()
    assert m.mmproj is not None and m.mmproj.gguf_path.name == "mmproj-F16.gguf"

    # Same via glob — covers repos that name it mmproj-model-f16 etc.
    fm2 = {**fm, "name": "dirk2", "mmproj": "mmproj*.gguf"}
    m2 = Model(md_path.parent / "dirk2.md", fm2, hf_home=hf_home)
    m2.resolve_companions()
    assert m2.mmproj is not None


def test_model_companion_fuzzy_from_hub_when_local_absent(tmp_path):
    hf_home = _hf_tree(tmp_path, files=("Qwen-x.Q4_K_M.gguf", "Qwen-x-mmproj.gguf"))
    md_path = tmp_path / "qwen.md"
    fm = {"name": "qwen", "model": "Qwen-x.Q4_K_M.gguf", "hf_repo": "org/repo"}
    m = Model(md_path, fm, hf_home=hf_home)
    m.resolve_companions()
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
    m.resolve_companions()
    assert m.mmproj is not None
    assert "models--other--vision-proj" in str(m.mmproj.gguf_path)


def test_model_speculative_from_hub(tmp_path):
    hf_home = _hf_tree(tmp_path,
                       files=("big-Q4_K_M.gguf", "big-mtp.gguf"))
    md_path = tmp_path / "big.md"
    fm = {"name": "big", "model": "big-Q4_K_M.gguf", "hf_repo": "org/repo",
          "speculative": "big-mtp.gguf"}
    m = Model(md_path, fm, hf_home=hf_home)
    m.resolve_companions()
    assert m.mtp is not None


def test_modelignore_excludes_files(tmp_path):
    from llama_packer.discover import discover
    from llama_packer.utils import load_model_ignore
    root = tmp_path / "models"
    (root / "vision").mkdir(parents=True)
    (root / ".modelignore").write_text(
        "# comment\nR3-rerank\nadetailer*\n*.tmp.gguf\n\n")
    (root / "vision" / "keep.gguf").write_bytes(b"x")
    (root / "vision" / "keep.md").write_text(_sidecar("Keep"))
    (root / "vision" / "adetailer-x.gguf").write_bytes(b"x")
    (root / "vision" / "adetailer-x.md").write_text(_sidecar("Bad"))
    sub = root / "vision" / "R3-rerank"
    sub.mkdir()
    (sub / "model.gguf").write_bytes(b"x")
    (sub / "model.md").write_text(_sidecar("Bad2"))

    assert load_model_ignore(root) == ["R3-rerank", "adetailer*", "*.tmp.gguf"]
    models = discover([root], generate_stubs=False)
    assert {m.stem for m in models} == {"keep"}


def _gguf_bytes(kv: dict) -> bytes:
    import struct
    out = (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
           + struct.pack("<Q", len(kv)))
    for k, v in kv.items():
        kb = k.encode()
        out += struct.pack("<Q", len(kb)) + kb
        if isinstance(v, str):
            vb = v.encode()
            out += struct.pack("<I", 8) + struct.pack("<Q", len(vb)) + vb
        else:
            out += struct.pack("<I", 4) + struct.pack("<I", int(v))
    return out


def test_diffusion_weights_excluded_from_served_role(tmp_path, caplog):
    # A GGUF whose header architecture is a diffusion family is excluded from
    # a served chat role with an error log; the run itself continues.
    root = tmp_path / "models"
    chat = root / "chat"
    chat.mkdir(parents=True)
    (chat / "fluxdev.gguf").write_bytes(
        _gguf_bytes({"general.architecture": "flux1"}))
    (chat / "ok.gguf").write_bytes(b"x")  # unparseable header → still served

    with caplog.at_level(logging.ERROR):
        models = Model.from_dir(root, generate_stubs=False)

    stems = {m.stem for m in models}
    assert "fluxdev" not in stems
    assert "ok" in stems
    assert any("fluxdev" in r.message and r.levelno == logging.ERROR
               for r in caplog.records)


def test_text_gguf_media_like_name_still_served(tmp_path):
    # Header classification only: an LLM whose filename mentions media stays.
    root = tmp_path / "models"
    chat = root / "chat"
    chat.mkdir(parents=True)
    (chat / "minimax-h3.gguf").write_bytes(
        _gguf_bytes({"general.architecture": "minimaxh3",
                     "minimaxh3.context_length": 1000000}))
    models = Model.from_dir(root, generate_stubs=False)
    assert [m.stem for m in models] == ["minimax-h3"]


# ── s2t (whisper) discovery ───────────────────────────────────────────────

def test_s2t_optin_serves_sidecar_bin_models(tmp_path):
    # Opt-in via dirs: {s2t: s2t}; each whisper .bin needs an authored
    # same-stem sidecar (no stubs are generated for .bin orphans).
    root = tmp_path / "models"
    (root / "s2t").mkdir(parents=True)
    (root / "s2t" / "ggml-large-v3.bin").write_bytes(b"x")
    (root / "s2t" / "ggml-large-v3.md").write_text(_sidecar("Whisper Large V3"))

    models = Model.from_dir(root, generate_stubs=False,
                            dir_roles={"s2t": "s2t"})
    assert len(models) == 1
    m = models[0]
    assert m.role == "s2t"
    assert m.gguf_path is not None and m.gguf_path.name == "ggml-large-v3.bin"


def test_s2t_orphan_bin_without_sidecar_skipped_no_stub(tmp_path, caplog):
    root = tmp_path / "models"
    (root / "s2t").mkdir(parents=True)
    (root / "s2t" / "ggml-base.bin").write_bytes(b"x")

    with caplog.at_level(logging.INFO):
        models = Model.from_dir(root, generate_stubs=True,
                                dir_roles={"s2t": "s2t"})
    assert models == []
    assert not (root / "s2t" / "ggml-base.md").exists()  # no stub written
    assert any("ggml-base.bin" in r.message and "sidecar" in r.message
               for r in caplog.records)


def test_s2t_not_opted_in_is_skipped(tmp_path):
    root = tmp_path / "models"
    (root / "s2t").mkdir(parents=True)
    (root / "s2t" / "ggml-base.bin").write_bytes(b"x")
    (root / "s2t" / "ggml-base.md").write_text(_sidecar("W"))

    models = Model.from_dir(root, generate_stubs=False)
    assert models == []


def test_bin_outside_s2t_never_served(tmp_path, caplog):
    # A .bin next to a sidecar in a non-s2t role resolves by stem (ordered
    # last), but no backend supports .bin outside whisper-server's s2t role —
    # finalize flags it (warning: expected in ordinary fleets, not an
    # operator error) and _filter_supported drops it from the config.
    root = tmp_path / "models"
    (root / "chat").mkdir(parents=True)
    (root / "chat" / "mysterious.bin").write_bytes(b"x")
    (root / "chat" / "mysterious.md").write_text(_sidecar("Mystery"))

    with caplog.at_level(logging.WARNING):
        models = Model.from_dir(root, generate_stubs=False)
    assert len(models) == 1
    assert getattr(models[0], "_override_error", None) is not None
    assert any("no available backend supports format '.bin'" in r.message
               for r in caplog.records)
    assert not any(r.levelno >= logging.ERROR
                   and "no available backend" in r.message
                   for r in caplog.records)


# ── t2s (kokoro) discovery ────────────────────────────────────────────────

def test_t2s_optin_hf_repo_only_sidecar(tmp_path, caplog):
    # Kokoro weights are baked into the container image: a t2s sidecar needs
    # no local model file at all — hf_repo alone identifies it.
    from llama_packer.backends import infer_backend
    root = tmp_path / "models"
    (root / "t2s").mkdir(parents=True)
    (root / "t2s" / "kokoro-v1.md").write_text(
        "---\nname: kokoro-v1\nhf_repo: hexgrad/Kokoro-82M\n---\n")

    with caplog.at_level(logging.ERROR):
        models = Model.from_dir(root, generate_stubs=False,
                                dir_roles={"t2s": "t2s"})
    assert len(models) == 1
    m = models[0]
    assert m.role == "t2s"
    # Backend inference needs the configured image (from_dir passes no avail,
    # so availability gating happens at pack time, not discovery time).
    assert infer_backend(m, {"kokoro_image": "img"}) == "kokoro-podman"


def test_t2s_onnx_sidecar_stem_resolves(tmp_path):
    # A locally downloaded .onnx copy resolves by same-stem convention.
    root = tmp_path / "models"
    (root / "t2s").mkdir(parents=True)
    (root / "t2s" / "kokoro-v1.onnx").write_bytes(b"x")
    (root / "t2s" / "kokoro-v1.md").write_text(_sidecar("Kokoro v1"))

    models = Model.from_dir(root, generate_stubs=False,
                            dir_roles={"t2s": "t2s"})
    assert len(models) == 1
    assert models[0].gguf_path is not None
    assert models[0].gguf_path.name == "kokoro-v1.onnx"
