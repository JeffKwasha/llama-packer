# tests/test_edge_branches.py
"""Branch-focused tests for decision paths the happy-path suite never hit.

These target error handling and fallbacks (where the two bugs found earlier
lived) plus the locally-added code paths: the role/backend guard, matrix
category reservation failures, and audio-cpp config fallbacks.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from llama_packer import writer
from llama_packer.backends import get_backend
from llama_packer.macros import Macro, Macros
from llama_packer.profiles import Profiles


def _stub_vram(m):
    m.vram.effective_static = lambda *a, **k: (1000.0, 0.0, 0.0, 50.0)
    m.vram.fit_params_static = lambda *a, **k: SimpleNamespace(
        model_mib=1000.0, kv_per_token_mib=0.0, slot_mib=0.0,
        compute_mib=50.0, source="fit-params")
    return m


# ── role/backend guard ────────────────────────────────────────────────────

@pytest.mark.parametrize("role", ["t2s", "s2t", "image"])
def test_filter_supported_rejects_audio_image_roles_on_llama_server(
        make_model, caplog, role):
    # Non-chat roles are rejected before they can reach llama-server (via the
    # engine's role set).  The audible contract is rejection + an error log.
    # Validates: SPEC.md, 'Backend Selection': a role the engine does not serve is dropped, with the reason logged.
    m = make_model("m", role=role, backend="llama-server")
    with caplog.at_level("ERROR"):
        out = writer._filter_supported([m], "q8_0")
    assert out == []
    assert "not supported" in caplog.text


def test_filter_supported_allows_role_on_matching_backend(make_model):
    # Validates: SPEC.md, 'Backend Selection': the same role passes on the backend that serves it.
    m = _stub_vram(make_model("m", role="t2s", backend="audio-cpp"))
    assert writer._filter_supported([m], "q8_0") != []


# ── shared matrix solve: degenerate inputs ────────────────────────────────

def test_solve_matrix_no_chat_participants_returns_none(make_model):
    # Validates: docs/plans/matrix-categories.md, 'Design sketch': a matrix with no chat participant has no shared solve.
    embed = _stub_vram(make_model("e", role="embeddings"))
    rerank = _stub_vram(make_model("r", role="rerank"))
    profiles = Profiles({"defaults": {}, "profiles": {"default": {}}})
    assert writer._solve_matrix_context(
        [], embed, rerank, "unused", 100000, None, profiles) is None


def _matrix_fixture(make_model, monkeypatch, chat_backend, chat_parallel):
    chat = _stub_vram(make_model("c", backend=chat_backend,
                                 parallel=chat_parallel, context_length=32768))
    embed = _stub_vram(make_model("e", role="embeddings", context_length=32768))
    rerank = _stub_vram(make_model("r", role="rerank", context_length=16384))
    captured: dict = {}

    def fake_solve(**kw):
        captured.update(kw)
        return 100000

    monkeypatch.setattr(writer, "solve_matrix_ctx", fake_solve)
    profiles = Profiles({"defaults": {"parallel": 4},
                         "profiles": {"default": {}}})
    result = writer._solve_matrix_context(
        [chat], embed, rerank, "unused", 100000, None, profiles,
        knobs=writer.MatrixKnobs(embed_context=32768, rerank_context=16384))
    return result, captured


def test_matrix_parallel_zero_vllm_is_treated_as_single_seq(make_model, monkeypatch):
    # Validates: docs/plans/matrix-categories.md, 'Design sketch': a vLLM entry with parallel 0 is one sequence, not zero.
    result, captured = _matrix_fixture(make_model, monkeypatch, "vllm", 0)
    assert result is not None
    # vLLM uncapped -> the shared solve evaluates one max-length sequence.
    assert captured["chat_models"][0][5] == 1


def test_matrix_parallel_zero_llama_cpp_falls_back_to_default(make_model, monkeypatch):
    # Validates: docs/plans/matrix-categories.md, 'Design sketch': a llama.cpp entry with parallel 0 falls back to the default count.
    result, captured = _matrix_fixture(make_model, monkeypatch, "llama-server", 0)
    assert result is not None
    assert captured["chat_models"][0][5] == 4


# ── declared matrix category reservation failures ─────────────────────────

def test_matrix_category_unsizable_is_skipped(make_model, caplog, monkeypatch):
    # Validates: SPEC.md, 'VRAM Allocation': a category that cannot be
    # sized is skipped, never reserved.
    chat = _stub_vram(make_model("c", context_length=32768))
    embed = _stub_vram(make_model("e", role="embeddings", context_length=32768))
    rerank = _stub_vram(make_model("r", role="rerank", context_length=16384))
    tts = make_model("t", role="t2s")
    tts.vram.effective_static = lambda *a, **k: None   # cannot be sized
    tts.vram.fit_params_static = lambda *a, **k: SimpleNamespace(
        model_mib=1.0, kv_per_token_mib=0.0, slot_mib=0.0, compute_mib=1.0,
        source="fit-params")
    monkeypatch.setattr(writer, "solve_matrix_ctx", lambda **kw: 100000)
    profiles = Profiles({"defaults": {}, "profiles": {"default": {}}})
    with caplog.at_level("WARNING"):
        result = writer._solve_matrix_context(
            [chat], embed, rerank, "unused", 100000, None, profiles,
            knobs=writer.MatrixKnobs(embed_context=32768, rerank_context=16384),
            fixed_categories=[("tts", tts)])
    assert result is not None
    assert result.coloads == ()
    assert "cannot size" in caplog.text


def test_matrix_category_that_does_not_fit_is_not_reserved(make_model, caplog,
                                                           monkeypatch):
    # Validates: SPEC.md, 'VRAM Allocation': a category that does not fit
    # is not charged against the chat context.
    chat = _stub_vram(make_model("c", context_length=32768))
    embed = _stub_vram(make_model("e", role="embeddings", context_length=32768))
    rerank = _stub_vram(make_model("r", role="rerank", context_length=16384))
    tts = _stub_vram(make_model("t", role="t2s", context_length=4096))

    def fake_solve(**kw):
        # Baseline solves comfortably; reserving any category collapses it.
        return 100 if kw.get("fixed_overhead_mb", 0) else 100000

    monkeypatch.setattr(writer, "solve_matrix_ctx", fake_solve)
    profiles = Profiles({"defaults": {}, "profiles": {"default": {}}})
    with caplog.at_level("WARNING"):
        result = writer._solve_matrix_context(
            [chat], embed, rerank, "unused", 100000, None, profiles,
            knobs=writer.MatrixKnobs(embed_context=32768, rerank_context=16384),
            fixed_categories=[("tts", tts)])
    assert result is not None
    assert result.chat_ctx == 100000          # category not charged
    assert "does not fit" in caplog.text


# ── audio-cpp config fallbacks ────────────────────────────────────────────

def test_audio_cpp_invalid_backend_falls_back_to_cpu(make_model, caplog):
    # Validates: docs/backends/audio-cpp.md, 'Emitted entry': an invalid audio-cpp backend falls back to cpu, and says so.
    m = make_model("a", role="t2s")
    with caplog.at_level("WARNING"):
        cmd, _ = get_backend("audio-cpp").build_cmd(
            m, 0, 1, "q8_0",
            {"audio_cpp_bin": "x", "audio_cpp_backend": "bogus"})
    assert '"backend":"cpu"' in cmd
    assert "bogus" in caplog.text


def test_audio_cpp_non_mapping_block_is_ignored(make_model, caplog):
    # Validates: docs/backends/audio-cpp.md, 'Emitted entry': a non-mapping audio_cpp block is ignored and the defaults still apply.
    m = make_model("a", role="t2s", audio_cpp="not-a-mapping")
    with caplog.at_level("WARNING"):
        cmd, _ = get_backend("audio-cpp").build_cmd(
            m, 0, 1, "q8_0", {"audio_cpp_bin": "x"})
    assert '"task":"tts"' in cmd          # defaults still applied
    assert caplog.text


# ── vLLM container args override ──────────────────────────────────────────

def test_vllm_explicit_container_args_replace_default_device_flags(make_model):
    # Validates: docs/backends/vllm.md, 'Container launch (docker/podman)': container_args replaces the device-flag set entirely
    m = make_model("v", hf_repo="org/model")
    cmd, _ = get_backend("vllm-docker").build_cmd(
        m, 4096, 1, "q8_0",
        {"vllm_image": "img", "vllm_bin": "vllm",
         "container_args": "--device /dev/custom"})
    assert "--device /dev/custom" in cmd
    assert "--shm-size" not in cmd        # default args fully replaced


# ── macros / utils small branches ─────────────────────────────────────────

def test_macros_root_models_yaml_named_after_root(tmp_path):
    # Validates: SPEC.md, 'Path Macros (`macros:` block and `config.env`)': a models.yaml at the models root is named after that root.
    Macro.clear()
    root = tmp_path / "models"
    root.mkdir()
    (root / "models.yaml").write_text("defaults:\n  cache_type: f16\n")
    Macros({}, None, [root])
    m = Macro.get("MODELS_MODELS")
    assert m is not None and m.flags["--cache-type-k"] == "f16"
    Macro.clear()


def test_macros_apply_whitespace_only_is_identity():
    # Validates: SPEC.md, 'Path Macros (`macros:` block and `config.env`)': a value with no macro in it comes back unchanged.
    Macro.clear()
    Macro("A", "s", {"--a": "1"})
    assert Macro.apply("   ") == "   "
    Macro.clear()


def test_parse_mem_mb_kilobyte_unit():
    # Validates: SPEC.md, 'Context calculation formula': the k suffix floors at 1 MiB.
    from llama_packer import utils
    assert utils.parse_mem_mb("2048k") == 2
    assert utils.parse_mem_mb("64k") == 1          # floor at 1 MiB
