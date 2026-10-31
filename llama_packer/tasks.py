"""The task catalogue: what a model does, where it is served, how it is sized.

A task is a closed id. It names one kind of work, the endpoint that work is
served at, the scheduling role it implies, and — for chat-sized work — the
serving demand the planner must meet. Tasks sharing an endpoint are aliases
of one serving: one process covers them, and they evict each other by
request rather than loading twice.

The catalogue is the *defaults*. An operator overrides a task's demand in
``profiles.yaml tasks:``; an unknown id is not a task.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

from llama_packer.consts import KNOWN_CAPABILITIES
from llama_packer.utils import parse_ctx_size
from llama_packer.endpoints import (
    CHAT,
    EMBEDDINGS,
    IMAGES,
    RERANK,
    SPEECH,
    TRANSCRIPTIONS,
)


@dataclass(frozen=True)
class Task:
    """One kind of work: a closed id, its endpoint, and its demand."""

    id: str
    endpoint: str
    role: str
    intent: str = ""
    #: Minimum serving an entry must afford to satisfy this task.
    min_context: int = 0
    min_parallel: int = 1
    require: frozenset[str] = frozenset()
    keep: frozenset[str] = frozenset()
    sacrifice: tuple[str, ...] | None = ()

    def satisfied_by(self, ctx: int, parallel: int) -> bool:
        """True when an entry served at (ctx, parallel) meets this demand."""
        return ctx >= self.min_context and parallel >= self.min_parallel


K = 1024

TASKS: tuple[Task, ...] = (
    Task("chat", CHAT, "chat", "conversation", 64 * K, 2,
         sacrifice=("draft", "vision")),
    Task("coding", CHAT, "chat", "agentic coding", 128 * K, 1,
         keep=frozenset({"draft"}), sacrifice=("vision",)),
    Task("qa", CHAT, "chat", "question answering", 32 * K, 2),
    Task("aux", CHAT, "chat", "one-off work", 80 * K, 2,
         sacrifice=("draft",)),
    Task("embeddings", EMBEDDINGS, "embeddings", "vectors"),
    Task("rerank", RERANK, "rerank", "relevance scores"),
    Task("image", IMAGES, "image", "image generation"),
    Task("s2t", TRANSCRIPTIONS, "s2t", "transcription"),
    Task("t2s", SPEECH, "t2s", "speech synthesis"),
)

BY_ID: dict[str, Task] = {t.id: t for t in TASKS}
IDS: tuple[str, ...] = tuple(t.id for t in TASKS)


def get(task: object) -> Task | None:
    """The task called *task*, or None when it is not a catalog id."""
    return BY_ID.get(task) if isinstance(task, str) else None


def role_of(tasks: tuple[str, ...] | list[str]) -> str | None:
    """The scheduling role a task list implies, or None when it is mixed."""
    roles = {t.role for t in (get(name) for name in tasks) if t is not None}
    return roles.pop() if len(roles) == 1 else None


def for_role(role: str) -> tuple[Task, ...]:
    """The tasks a scheduling role serves."""
    return tuple(t for t in TASKS if t.role == role)


@dataclass(frozen=True)
class TaskCatalogue:
    """Catalog defaults merged with the operator's ``profiles.yaml tasks:``."""

    by_id: Mapping[str, Task] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.by_id)

    def get(self, name: str) -> Task | None:
        return self.by_id.get(name)

    @property
    def order(self) -> tuple[str, ...]:
        return tuple(self.by_id)

    @classmethod
    def from_cfg(cls, cfg: dict | None) -> "TaskCatalogue":
        """Merge ``tasks:`` overrides onto the catalog.

        An override names a catalog id and states only the demand keys it
        changes; an unknown id is warned about and dropped, never invented.
        """
        import logging
        log = logging.getLogger(__name__)
        merged = dict(BY_ID)
        for name, spec in (cfg or {}).items():
            if not isinstance(name, str) or name.strip() not in BY_ID:
                close = _close(name)
                log.warning("tasks: %r is not a task%s; ignoring it "
                            "(known: %s)", name, close, ", ".join(IDS))
                continue
            task = BY_ID[name.strip()]
            if isinstance(spec, int) and not isinstance(spec, bool):
                spec = {"min_context": spec}
            if not isinstance(spec, dict):
                log.warning("tasks: %r is not a mapping; ignoring it", name)
                continue
            merged[task.id] = _override(task, spec, log)
        return cls(by_id=merged)


def validate_task_ids(raw: object) -> list[str]:
    """Shape and vocabulary errors in a sidecar's ``tasks:`` value.

    Returns human messages (empty when sound).  Absence is not an error: a
    stub/orphan has no claim to make.
    """
    if raw is None or raw == "":
        return []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return ["tasks: must be a list of task names"]
    if not raw:
        return ["tasks: must name at least one task"]
    problems: list[str] = []
    for name in raw:
        if not isinstance(name, str) or not name.strip():
            problems.append(f"tasks: {name!r} is not a task name")
        elif name.strip() not in BY_ID:
            problems.append(f"tasks: {name!r} is not a task{_close(name)}")
    return problems


def resolve_tasks(tags: object, catalogue: TaskCatalogue,
                  log=None, stem: str | None = None) -> tuple[Task, ...]:
    """Resolve a sidecar's ``tasks:`` ids against the catalogue.

    An unknown id is warned about and ignored, so the effective tasks are
    always exactly the subset of the catalogue the operator can name.  A
    bare string is one task; no tags means no demands.
    """
    import logging
    log = log or logging.getLogger(__name__)
    who = f"{stem}: tasks:" if stem else "tasks:"
    if tags is None:
        return ()
    if isinstance(tags, str):
        tags = [tags]
    if not isinstance(tags, (list, tuple)):
        log.warning("%s %r is not a list of task names; ignoring it",
                    who, tags)
        return ()
    out: list[Task] = []
    for tag in tags:
        if not isinstance(tag, str):
            log.warning("%s %r is not a task name; ignoring it", who, tag)
            continue
        task = catalogue.get(tag.strip())
        if task is None:
            log.warning("%s %r is not a task%s; ignoring it (known: %s)",
                        who, tag.strip(), _close(tag), ", ".join(catalogue.order))
            continue
        if task not in out:
            out.append(task)
    return tuple(out)


def _close(name: object) -> str:
    import difflib
    close = difflib.get_close_matches(str(name), IDS, n=1, cutoff=0.6)
    return f" (did you mean {close[0]!r}?)" if close else ""


def _override(task: Task, spec: dict, log) -> Task:
    """One task's demand overridden by the keys present in *spec*."""
    changes: dict = {}
    for key, value in spec.items():
        if key == "min_context":
            parsed = parse_ctx_size(value)
            if parsed is None or parsed <= 0:
                log.warning("tasks: %s.min_context=%r is not a positive "
                            "size; keeping %d", task.id, value,
                            task.min_context)
                continue
            changes["min_context"] = parsed
        elif key == "min_parallel":
            if not isinstance(value, int) or isinstance(value, bool) \
                    or value < 1:
                log.warning("tasks: %s.min_parallel=%r is not a positive "
                            "integer; keeping %d", task.id, value,
                            task.min_parallel)
                continue
            changes["min_parallel"] = value
        elif key in ("require", "keep"):
            if not isinstance(value, (list, tuple)):
                log.warning("tasks: %s.%s=%r is not a list of capabilities; "
                            "ignoring it", task.id, key, value)
                continue
            unknown = [str(v) for v in value
                       if str(v) not in KNOWN_CAPABILITIES]
            if unknown:
                log.warning("tasks: %s.%s names unknown capabilities %s; "
                            "ignoring them", task.id, key,
                            ", ".join(unknown))
            changes[key] = frozenset(
                str(v) for v in value if str(v) in KNOWN_CAPABILITIES)
        elif key == "sacrifice":
            if not isinstance(value, (list, tuple)):
                log.warning("tasks: %s.sacrifice=%r is not a list; ignoring "
                            "it", task.id, value)
                continue
            changes[key] = tuple(str(v) for v in value)
        else:
            log.warning("tasks: %s.%s is not a task key; ignoring it",
                        task.id, key)
    return replace(task, **changes)
