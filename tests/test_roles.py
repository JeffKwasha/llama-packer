"""The role registry: one table, and every task reaches a served endpoint."""

from __future__ import annotations

from llama_packer import roles
from llama_packer.backends import BACKENDS


def test_role_registry_derives_its_sets():
    # Validates: docs/plans/cpu-residency-budget.md #30 -- served,
    # non-chat, device and context facts all come from one table.
    assert roles.SERVED_ROLES == ("chat", "embeddings", "rerank",
                                  "image", "s2t", "t2s")
    assert roles.NON_CHAT_ROLES == frozenset(
        {"embeddings", "rerank", "image", "s2t", "t2s"})
    assert roles.DEVICE_ROLES == frozenset(
        {"embeddings", "rerank", "s2t", "t2s"})
    assert roles.get("embeddings").ctx_knob == "embed_context"
    assert roles.get("rerank").ctx_knob == "rerank_context"
    assert roles.get("chat").ctx_knob is None
    assert roles.get("embedder") is None


def test_every_task_maps_to_a_served_endpoint():
    # Validates: docs/plans/requirement-trace.md #90 -- every task maps to a
    # closed endpoint that at least one engine serves, and no engine
    # advertises an endpoint no task uses.
    from llama_packer import endpoints

    assert set(endpoints.TASK_ENDPOINTS) == set(roles.SERVED_ROLES)
    assert set(endpoints.TASK_ENDPOINTS.values()) <= endpoints.ENDPOINTS
    for task, endpoint in endpoints.TASK_ENDPOINTS.items():
        assert any(endpoint in b.endpoints for b in BACKENDS.values()), task
    served = {e for b in BACKENDS.values() for e in b.endpoints}
    assert served <= set(endpoints.TASK_ENDPOINTS.values())
