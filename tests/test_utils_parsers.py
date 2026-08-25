# tests/test_utils_parsers.py
"""Memory string parsers."""

from __future__ import annotations

import pytest

from llama_packer.utils import parse_context_length, parse_mem_mb


def test_resolve_spare_suffixed_gigabytes():
    assert parse_mem_mb("2G", 32768) == 2048


def test_resolve_spare_suffixed_megabytes():
    assert parse_mem_mb("512m", 32768) == 512


def test_resolve_spare_bare_gb_hint():
    # bare number < 3 * VRAM(GB) -> treated as GB
    assert parse_mem_mb("2", 32768) == 2048


def test_resolve_spare_bare_mb_hint():
    # bare number >= 3 * VRAM(GB) -> treated as MB
    assert parse_mem_mb("512", 32768) == 512


def test_resolve_spare_invalid_returns_zero():
    assert parse_mem_mb("nonsense", 32768) == 0


def test_parse_context_length_k():
    assert parse_context_length("128k") == 131072


def test_parse_context_length_m():
    assert parse_context_length("1m") == 1048576


def test_parse_context_length_bare():
    assert parse_context_length("65536") == 65536


# ── HF cache grouping ─────────────────────────────────────────────────────


def test_compute_env_prefixes_hf_grouping(tmp_path, monkeypatch):
    from llama_packer.utils import compute_env_prefixes, hf_cache_root
    models = tmp_path / "models"
    hf = tmp_path / "hf"
    models.mkdir(); hf.mkdir()
    gguf = models / "a.gguf"
    gguf.write_bytes(b"x")
    ct = hf / "chat_template.jinja"
    ct.write_text("x")
    monkeypatch.setenv("HF_HOME", str(hf))
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)

    _p2v, v2v = compute_env_prefixes([str(gguf), str(ct)])
    assert v2v["HF_HOME"] == hf_cache_root()
    assert v2v["MODELS_DIR"] == str(models)
    # The chat-template path must NOT widen MODELS_DIR up to tmp_path.
    assert v2v["MODELS_DIR"] == str(models)


def test_compute_env_prefixes_hf_home_override(tmp_path, monkeypatch):
    from llama_packer.utils import compute_env_prefixes
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))  # no default cache dir
    models = tmp_path / "models"
    hf = tmp_path / "custom_hf"
    models.mkdir(); hf.mkdir()
    gguf = models / "a.gguf"; gguf.write_bytes(b"x")
    ct = hf / "ct.jinja"; ct.write_text("x")

    _p2v, v2v = compute_env_prefixes([str(gguf), str(ct)], hf_home=str(hf))
    assert v2v["HF_HOME"] == str(hf)
    assert v2v["MODELS_DIR"] == str(models)


# ── Model-kind classification (header-only) ───────────────────────────────


def _gguf_bytes(kv: dict) -> bytes:
    """Minimal GGUF: magic, v3 header, no tensors, string/u32 metadata only."""
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


def test_classify_gguf_text_architecture(tmp_path):
    from llama_packer.utils import classify_file, gguf_header_probe
    p = tmp_path / "m.gguf"
    p.write_bytes(_gguf_bytes({"general.architecture": "qwen3vl",
                               "qwen3vl.context_length": 262144}))
    assert gguf_header_probe(p) == ("qwen3vl", True)
    assert classify_file(p) == "text"


def test_classify_gguf_diffusion_architecture(tmp_path):
    from llama_packer.utils import classify_file, gguf_header_probe
    p = tmp_path / "flux.gguf"
    p.write_bytes(_gguf_bytes({"general.architecture": "flux1"}))
    assert gguf_header_probe(p) == ("flux1", False)
    assert classify_file(p) == "image"


def test_classify_ignores_filename_llm_named_like_media(tmp_path):
    # MiniMax H3-style case: a text model whose *filename* looks media-ish is
    # still classified by its header architecture, never the name.
    from llama_packer.utils import classify_file
    p = tmp_path / "MiniMax-H3-video-sounding-name.gguf"
    p.write_bytes(_gguf_bytes({"general.architecture": "minimaxh3",
                               "minimaxh3.context_length": 1000000}))
    assert classify_file(p) == "text"


def test_classify_gguf_unknown(tmp_path):
    from llama_packer.utils import classify_file
    p = tmp_path / "x.gguf"
    p.write_bytes(b"x")  # not GGUF at all
    assert classify_file(p) == "unknown"


def _safetensors_bytes(names: list[str]) -> bytes:
    import json
    import struct
    header = {n: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}
              for n in names}
    hb = json.dumps(header).encode()
    return struct.pack("<Q", len(hb)) + hb


def test_sniff_safetensors_diffusion_blocks(tmp_path):
    from llama_packer.utils import classify_file, sniff_safetensors
    p = tmp_path / "flux.safetensors"
    p.write_bytes(_safetensors_bytes([
        "double_blocks.0.img_attn.norm.key_norm.scale",
        "double_blocks.0.img_attn.proj.weight",
    ]))
    assert sniff_safetensors(p) == "image"
    assert classify_file(p) == "image"


def test_sniff_safetensors_text_transformer(tmp_path):
    from llama_packer.utils import sniff_safetensors
    p = tmp_path / "llm.safetensors"
    p.write_bytes(_safetensors_bytes([
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "lm_head.weight",
    ]))
    assert sniff_safetensors(p) == "text"


def test_sniff_safetensors_unknown(tmp_path):
    from llama_packer.utils import sniff_safetensors
    p = tmp_path / "odd.safetensors"
    p.write_bytes(_safetensors_bytes(["some.random.tensor.weight"]))
    assert sniff_safetensors(p) == "unknown"


def test_hf_readme_kind_from_cached_card(tmp_path, monkeypatch):
    # Offline signal: pipeline_tag in the snapshot README.md frontmatter.
    from llama_packer.utils import hf_readme_kind
    hub = tmp_path / "hub"
    snap = hub / "models--org--repo" / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (snap / "README.md").write_text(
        "---\npipeline_tag: text-to-image\ntags:\n- diffusers\n---\n\n# card\n")
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert hf_readme_kind("org/repo", str(tmp_path)) == "image"


def test_hf_readme_kind_text_tag_returns_none(tmp_path):
    from llama_packer.utils import hf_readme_kind
    hub = tmp_path / "hub"
    snap = hub / "models--org--text" / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (snap / "README.md").write_text(
        "---\npipeline_tag: text-generation\n---\n\n# card\n")
    assert hf_readme_kind("org/text", str(tmp_path)) is None
