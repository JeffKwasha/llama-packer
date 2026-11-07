# tests/test_endpoint_registry.py
"""The endpoint registry: closed ids, one table, and docs that agree."""

from __future__ import annotations

from pathlib import Path

from llama_packer import endpoints, roles
from llama_packer.backends import BACKENDS

ROOT = Path(__file__).resolve().parent.parent


def test_registry_invariants_hold():
    # Validates: docs/plans/requirement-trace.md, requirement #97
    assert endpoints.check_registry() == []


def test_registry_task_mapping_is_total():
    # Validates: docs/plans/requirement-trace.md, requirement #97
    assert set(endpoints.TASK_ENDPOINTS) == set(roles.SERVED_ROLES)
    assert set(endpoints.TASK_ENDPOINTS.values()) <= endpoints.ENDPOINTS


def test_registry_engine_table_matches_backends():
    # Validates: docs/plans/requirement-trace.md, requirement #97
    for bound in BACKENDS.values():
        assert bound.endpoints == endpoints.ENGINE_ENDPOINTS[bound.engine.name]


def test_registry_docs_agree():
    # Validates: docs/plans/requirement-trace.md, requirement #97
    assert endpoints.docs_agree(ROOT) == []


def test_registry_unknown_endpoint_is_rejected():
    # Validates: docs/plans/requirement-trace.md, requirement #97
    assert endpoints.for_task("nope") is None
    assert endpoints.serves({endpoints.CHAT}, "nope") is False


def test_registry_engines_value_shape():
    # Validates: docs/plans/requirement-trace.md, requirement #97
    assert endpoints.validate_engines_value(None) == []
    assert endpoints.validate_engines_value("vllm") == []
    assert endpoints.validate_engines_value([]) == [
        "engines: must name at least one engine"]
    assert any("is not an engine" in p
               for p in endpoints.validate_engines_value(["bogus"]))
    assert any("did you mean" in p
               for p in endpoints.validate_engines_value(["llama-servr"]))
    assert endpoints.validate_engines_value({"a": 1}) == [
        "engines: must be a list of engine names"]
