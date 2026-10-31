# tests/test_engines.py
"""Sidecar ``engines:``: declared compatibility, resolved to a bound backend.

The list names engines (``vllm``), never transport pairs (``vllm-podman``);
the registry picks the transport. When nothing resolves, the refusal names
every declared engine and why it was rejected.
"""

from __future__ import annotations

from types import SimpleNamespace

from llama_packer import endpoints
from llama_packer.backends import ENGINE_NAMES, resolve_engines
from llama_packer.scope import ScopeStack


class _FakeBackend:
    """A bound backend that loads .gguf and serves chat, for order tests."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.engine = SimpleNamespace(name=name)
        self.formats = frozenset({".gguf"})
        self.endpoints = frozenset({endpoints.CHAT})

    def is_available(self, avail: dict) -> bool:
        return True


def test_engines_vocabulary_is_bare_engine_names():
    # Validates: docs/plans/requirement-trace.md #95 -- `engines:` names engines, not transport-bound pairs.
    assert ENGINE_NAMES == ("llama-server", "vllm", "sd-server",
                            "whisper-server", "audio-cpp")
    assert "vllm-podman" not in ENGINE_NAMES


def test_engines_resolution_honours_declared_order(make_model, monkeypatch):
    # Validates: docs/plans/requirement-trace.md #95 -- the first declared engine that can serve wins.
    import llama_packer.backends as backends

    monkeypatch.setattr(backends, "BACKENDS",
                        {"alpha": _FakeBackend("alpha"),
                         "beta": _FakeBackend("beta")})
    m = make_model("gemma4-12b")
    assert resolve_engines(m, ("alpha", "beta"))[0] == "alpha"
    assert resolve_engines(m, ("beta", "alpha"))[0] == "beta"


def test_engines_resolution_picks_the_transport(make_model):
    # Validates: docs/plans/requirement-trace.md #95 -- a declared engine resolves to the registry's transport preference.
    m = make_model("qwen3-32b", hf_repo="org/model")
    m.gguf_path = None
    assert resolve_engines(m, ("vllm",), {"vllm_bin": "vllm"})[0] == "vllm"
    assert resolve_engines(m, ("vllm",),
                           {"vllm_image": "img"})[0] == "vllm-podman"
    assert resolve_engines(m, ("vllm",),
                           {"vllm_image": "img",
                            "podman": False})[0] == "vllm-docker"


def test_engines_refusal_names_every_engine(make_model):
    # Validates: docs/plans/requirement-trace.md #95 -- a declared engine that cannot serve is reported by name and reason.
    m = make_model("flux2-dev", role="image")
    resolved, reasons = resolve_engines(m, ("llama-server",),
                                        {"llama_bin": "/opt/llama-server"})
    assert resolved is None
    assert "does not serve role 'image'" in reasons["llama-server"]


def test_engines_finalize_sets_the_backend(make_model):
    # Validates: docs/plans/requirement-trace.md #95 -- a resolvable declaration becomes the model's backend.
    m = make_model("flux2-dev", role="image", engines=["sd-server"])
    ScopeStack(avail={"sd_bin": "/opt/sd-server"}).finalize(m)
    assert m.backend == "sd-server"
    assert getattr(m, "_override_error", None) is None


def test_engines_finalize_refuses_by_name(make_model):
    # Validates: docs/plans/requirement-trace.md #95 -- nothing resolvable skips the model with the declared names in the reason.
    m = make_model("qwen3-8b", engines=["sd-server"])
    ScopeStack(avail={"sd_bin": "/opt/sd-server",
                      "llama_bin": "/opt/llama-server"}).finalize(m)
    err = getattr(m, "_override_error", None)
    assert err and "engines:" in err and "sd-server" in err


def test_engines_wins_over_the_alien_backend_key(make_model, caplog):
    # Validates: docs/plans/requirement-trace.md #96 -- `backend:` warns as alien and `engines:` decides.
    m = make_model("qwen3-8b", engines=["llama-server"], backend="sd-server")
    ScopeStack(avail={"llama_bin": "/opt/llama-server",
                      "sd_bin": "/opt/sd-server"}).finalize(m)
    assert m.backend == "llama-server"
    assert "not a sidecar key" in caplog.text


def test_engines_unknown_name_warns_with_a_hint(make_model, caplog):
    # Validates: docs/plans/requirement-trace.md #95 -- an unknown engine name warns at discovery with a near-miss hint.
    from llama_packer.discover import _warn_unsupported_values

    m = make_model("qwen3-8b", engines=["llama-servr"])
    _warn_unsupported_values(m, "qwen3-8b.md")
    assert "did you mean 'llama-server'" in caplog.text


def test_sidecar_without_engines_is_refused(tmp_path, make_model):
    # Validates: docs/plans/requirement-trace.md #96 -- an authored sidecar must declare engines:.
    m = make_model("qwen3-8b")
    (tmp_path / "qwen3-8b.md").write_text("---\nname: qwen3-8b\n---\n")
    m.stub = False
    ScopeStack(avail={"llama_bin": "/opt/llama-server"}).finalize(m)
    err = getattr(m, "_override_error", None)
    assert err and "engines: required" in err and "qwen3-8b.md" in err


def test_orphan_without_engines_still_infers(make_model):
    # Validates: docs/plans/requirement-trace.md #96 -- a model with no authored sidecar keeps inference.
    m = make_model("gemma4-12b")
    assert m.stub is True
    ScopeStack(avail={"llama_bin": "/opt/llama-server"}).finalize(m)
    assert m.backend == "llama-server"
    assert getattr(m, "_override_error", None) is None
