"""TTL resolution: sidecar > task > placement default > global."""

from __future__ import annotations

import logging
from types import SimpleNamespace

from llama_packer.writer import CPU_HELPER_TTL, resolve_entry_ttl


def _m(**kw):
    base = dict(label="m", on_cpu=False, role="chat",
                sidecar_keys=frozenset(), frontmatter={})
    base.update(kw)
    return SimpleNamespace(**base)


def _task(ttl=None, role="chat"):
    return SimpleNamespace(ttl=ttl, role=role)


def test_ttl_sidecar_wins_when_tighter():
    # Validates: SPEC.md, "TTL": declared lifetimes resolve sidecar > task >
    # global, and the smallest lifetime wins.
    assert resolve_entry_ttl(
        _m(sidecar_keys=frozenset({"ttl"}), frontmatter={"ttl": 1800}),
        _task(ttl=3600)) == 1800


def test_ttl_task_wins_when_tighter():
    # Validates: SPEC.md, "TTL": declared lifetimes resolve sidecar > task >
    # global, and the smallest lifetime wins.
    assert resolve_entry_ttl(
        _m(sidecar_keys=frozenset({"ttl"}), frontmatter={"ttl": 7200}),
        _task(ttl=3600)) == 3600


def test_ttl_zero_means_never_unload_and_wins():
    # Validates: SPEC.md, "TTL": ttl 0 disables automatic unloading.
    assert resolve_entry_ttl(
        _m(sidecar_keys=frozenset({"ttl"}), frontmatter={"ttl": 0}),
        _task(ttl=3600)) == 0


def test_ttl_cpu_helper_defaults_to_host_lifetime():
    # Validates: SPEC.md, "TTL": an undeclared CPU helper gets the host
    # default so its RAM is eventually yielded.
    assert resolve_entry_ttl(
        _m(on_cpu=True, role="embeddings"),
        _task(role="embeddings")) == CPU_HELPER_TTL


def test_ttl_cpu_serving_gets_cpu_default():
    # Validates: SPEC.md, "TTL": every CPU-resident serving gets the host
    # default; chat is not special.
    assert resolve_entry_ttl(_m(on_cpu=True, role="chat"),
                             _task(role="chat")) == CPU_HELPER_TTL


def test_ttl_device_entry_inherits_global():
    # Validates: SPEC.md, "TTL": an undeclared device serving inherits
    # globalTTL (no per-entry ttl is emitted).
    assert resolve_entry_ttl(_m(), _task()) is None


def test_ttl_role_default_reaches_a_helper_without_a_task_demand():
    # Validates: SPEC.md, "TTL": a helper planned from its role still gets
    # the role's declared task lifetime.
    assert resolve_entry_ttl(_m(on_cpu=True, role="embeddings"), None, 0) == 0


def test_ttl_invalid_sidecar_value_warns(caplog):
    # Validates: SPEC.md, "TTL": a malformed declared lifetime is ignored,
    # not silently applied.
    with caplog.at_level(logging.WARNING):
        got = resolve_entry_ttl(
            _m(sidecar_keys=frozenset({"ttl"}), frontmatter={"ttl": "soon"}),
            _task())
    assert got is None
    assert "not a non-negative integer" in caplog.text
