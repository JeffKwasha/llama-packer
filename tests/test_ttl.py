"""TTL resolution: sidecar > task > role > top-level ``ttl``."""

from __future__ import annotations

import logging
from types import SimpleNamespace

from llama_packer.writer import resolve_entry_ttl


def _m(**kw):
    base = dict(label="m", on_cpu=False, role="chat",
                sidecar_keys=frozenset(), frontmatter={})
    base.update(kw)
    return SimpleNamespace(**base)


def _task(ttl=None, role="chat"):
    return SimpleNamespace(ttl=ttl, role=role)


def test_sidecar_wins_over_task_and_role():
    # Validates: docs/allocation.md, 'TTL': sidecar > task > role.
    assert resolve_entry_ttl(
        _m(sidecar_keys=frozenset({"ttl"}), frontmatter={"ttl": 1800}),
        _task(ttl=3600), role_ttl=600) == 1800


def test_task_wins_over_role_even_when_larger():
    # Validates: docs/allocation.md, 'TTL': the most specific value wins,
    # not the smallest.
    assert resolve_entry_ttl(_m(), _task(ttl=7200), role_ttl=600) == 7200


def test_role_wins_when_nothing_more_specific():
    # Validates: docs/plans/requirement-trace.md, requirement #80
    assert resolve_entry_ttl(_m(), _task(), role_ttl=600) == 600


def test_zero_is_a_value_not_an_absence():
    # Validates: docs/plans/requirement-trace.md, requirement #80
    assert resolve_entry_ttl(
        _m(sidecar_keys=frozenset({"ttl"}), frontmatter={"ttl": 0}),
        _task(ttl=3600), role_ttl=600) == 0
    assert resolve_entry_ttl(_m(), _task(ttl=0), role_ttl=600) == 0


def test_device_entry_with_nothing_declared_inherits_the_top_level():
    # Validates: docs/plans/requirement-trace.md, requirement #80
    assert resolve_entry_ttl(_m(), _task()) is None


def test_role_ttl_reaches_a_helper_without_a_task_demand():
    # Validates: docs/plans/requirement-trace.md, requirement #80
    assert resolve_entry_ttl(_m(on_cpu=True, role="embeddings"), None,
                             role_ttl=0) == 0


def test_invalid_sidecar_value_warns_and_falls_through(caplog):
    with caplog.at_level(logging.WARNING):
        got = resolve_entry_ttl(
            _m(sidecar_keys=frozenset({"ttl"}), frontmatter={"ttl": "soon"}),
            _task(ttl=3600))
    assert got == 3600
    assert "not a non-negative integer" in caplog.text
