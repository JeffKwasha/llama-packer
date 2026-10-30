"""Roles: what a model does, and the operation it needs from an engine.

A role is a kind of work with exactly one name.  An engine serves *models*
and implements *operations* (generate, embed, transcribe, …); a role names
the operation it requires.  The indirection is deliberate: an STT engine
supports some models that happen to transcribe, it does not "support the s2t
role", and a second STT engine must not require editing a role.

Everything derived from a role — the served set, the chat-pass set, the
``device_roles`` set, the matrix context knob — comes from this one table.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Role:
    """One kind of work a model can do."""

    name: str
    operation: str
    #: Participates in the chat passes (mmproj keep/drop, shared solve, …).
    chat: bool
    #: ``pinned`` (always resident), ``optional`` (resident when declared),
    #: ``demand`` (never pinned).
    residency: str
    #: ``MatrixKnobs`` attribute holding this role's served context, if any.
    ctx_knob: str | None
    #: May be named in ``profiles.yaml device_roles:``.
    device_role: bool


ROLES: tuple[Role, ...] = (
    Role("chat", "generate", True, "demand", None, False),
    Role("embeddings", "embed", False, "pinned", "embed_context", True),
    Role("rerank", "rerank", False, "pinned", "rerank_context", True),
    Role("image", "generate_image", False, "demand", None, False),
    Role("s2t", "transcribe", False, "optional", None, True),
    Role("t2s", "synthesize", False, "optional", None, True),
)

BY_NAME: dict[str, Role] = {r.name: r for r in ROLES}

SERVED_ROLES: tuple[str, ...] = tuple(r.name for r in ROLES)
NON_CHAT_ROLES: frozenset[str] = frozenset(r.name for r in ROLES if not r.chat)
DEVICE_ROLES: frozenset[str] = frozenset(r.name for r in ROLES if r.device_role)
OPERATIONS: frozenset[str] = frozenset(r.operation for r in ROLES)


def get(name: object) -> Role | None:
    """The role called *name*, or None when it is not a served role."""
    return BY_NAME.get(name) if isinstance(name, str) else None


def roles_for_operations(operations: frozenset[str] | set[str]) -> tuple[str, ...]:
    """Role names whose operation is implemented — for diagnostics."""
    return tuple(r.name for r in ROLES if r.operation in operations)
