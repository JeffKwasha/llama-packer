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


def test_make_fast_storage_predicate(tmp_path):
    from llama_packer.utils import make_fast_storage_predicate

    # a merged view (/ai) over two branches; mergerfs keeps directory
    # structure identical across branches, so a merged path classifies by
    # whether the same relative path exists under the fast branch
    ai = tmp_path / "ai"
    fast = tmp_path / "ssd_ai"
    slow = tmp_path / "r1_ai"
    for d in (ai, fast, slow):
        d.mkdir()
    # a file that exists under the fast branch at the same relative spot
    (fast / "hub" / "blobs").mkdir(parents=True)
    (fast / "hub" / "blobs" / "dirk.gguf").write_bytes(b"x")
    # a symlink in the merged view pointing at the fast-branch blob
    (ai / "hub").mkdir()
    (ai / "hub" / "dirk-link.gguf").symlink_to(fast / "hub" / "blobs" / "dirk.gguf")
    # a file only on the slow branch
    (slow / "hub" / "blobs").mkdir(parents=True)
    (slow / "hub" / "blobs" / "big.gguf").write_bytes(b"x")
    (ai / "hub" / "big-link.gguf").symlink_to(slow / "hub" / "blobs" / "big.gguf")

    def exists(path: str) -> bool:
        return (tmp_path / path.lstrip("/")).exists()

    def norm(p):
        return str(p).replace(str(tmp_path) + "/", "/")

    is_fast = make_fast_storage_predicate(str(fast), exists=exists)
    # merged view + symlink: classifies by the blob's physical location
    assert is_fast(str(ai / "hub" / "dirk-link.gguf"))
    assert not is_fast(str(ai / "hub" / "big-link.gguf"))
    # direct branch paths are trivially their own tier
    assert is_fast(str(fast / "hub" / "blobs" / "dirk.gguf"))
    assert not is_fast(str(slow / "hub" / "blobs" / "big.gguf"))
    assert not is_fast(str(tmp_path / "ai" / "nothing.gguf"))

    # None/empty roots: no tier knowledge
    assert not make_fast_storage_predicate(None, exists=exists)(
        norm(fast / "hub" / "blobs" / "dirk.gguf"))
    # multiple roots
    both = make_fast_storage_predicate([str(fast), str(slow)], exists=exists)
    assert both(str(ai / "hub" / "big-link.gguf"))


# ── safetensors header estimate ──────────────────────────────────────────


def _safetensors_file(tmp_path, dtype: str):
    """Minimal safetensors file: one k_proj + one v_proj tensor header."""
    import json
    import struct

    t = {"dtype": dtype, "shape": [512, 1024], "data_offsets": [0, 0]}
    header = {
        "model.layers.0.self_attn.k_proj.weight": t,
        "model.layers.0.self_attn.v_proj.weight": dict(t),
    }
    header_bytes = json.dumps(header).encode()
    p = tmp_path / "m.safetensors"
    p.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + b"\0" * 8)
    return p


def test_estimate_safetensors_fp8_dtype_names(tmp_path):
    # FP8 checkpoints carry F8_E4M3 in the header (the safetensors spec
    # spelling) — 1 B/elem, the same density as a Q8_0 GGUF.
    from llama_packer.utils import estimate_safetensors

    p = _safetensors_file(tmp_path, "F8_E4M3")
    model_mib, kv_per_token = estimate_safetensors(str(p), "q8_0")
    assert model_mib == 1  # 2 x 512x1024 @ 1 B = 1 MiB
    assert kv_per_token == 2 * 512 * 1.0625 / (1024 * 1024)

    p = _safetensors_file(tmp_path, "F8_E5M2")
    model_mib, _ = estimate_safetensors(str(p), "q8_0")
    assert model_mib == 1


def test_estimate_safetensors_bf16_control(tmp_path):
    from llama_packer.utils import estimate_safetensors

    p = _safetensors_file(tmp_path, "BF16")
    model_mib, _ = estimate_safetensors(str(p), "q8_0")
    assert model_mib == 2  # 2 x 512x1024 @ 2 B = 2 MiB
