"""The role registry: one table, and every operation has an engine."""

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


def test_every_role_operation_is_implemented():
    # Validates: docs/plans/cpu-residency-budget.md #30 -- no role lacks an
    # engine and no engine advertises an operation no role needs.
    covered = {op for b in BACKENDS.values() for op in b.operations}
    assert covered == roles.OPERATIONS
    for role in roles.ROLES:
        assert any(b.supports_role(role.name) for b in BACKENDS.values()), \
            role.name
