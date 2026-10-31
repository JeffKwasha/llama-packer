"""The task catalogue: what a model does, where it is served, how it is sized.

A task is declared in ``profiles.yaml tasks:``. It names one kind of work,
the endpoint that work is served at, the scheduling role it implies, and —
for chat-sized work — the serving demand the planner must meet. Tasks
sharing an endpoint are aliases of one serving: one process covers them,
and they evict each other by request rather than loading twice.

The code hardcodes only the contract a declaration is checked against: the
endpoint ids (:mod:`llama_packer.endpoints`) and the scheduling roles
(:mod:`llama_packer.roles`). The vocabulary of tasks belongs to the
operator; an undeclared id is not a task.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, replace
from typing import Mapping

from llama_packer.consts import KNOWN_CAPABILITIES
from llama_packer.endpoints import ENDPOINTS
from llama_packer.roles import SERVED_ROLES
from llama_packer.utils import parse_ctx_size

log = logging.getLogger(__name__)

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


@dataclass(frozen=True)
class Task:
    """One kind of work: an id, its endpoint, and its demand."""

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


@dataclass(frozen=True)
class TaskCatalogue:
    """The tasks declared in ``profiles.yaml tasks:``."""

    by_id: Mapping[str, Task] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.by_id)

    def get(self, name: str) -> Task | None:
        return self.by_id.get(name)

    @property
    def order(self) -> tuple[str, ...]:
        return tuple(self.by_id)

    def role_of(self, ids: tuple[str, ...] | list[str]) -> str | None:
        """The scheduling role a task list implies, or None when it is mixed."""
        roles = {t.role for t in (self.get(name) for name in ids)
                 if t is not None}
        return roles.pop() if len(roles) == 1 else None

    @classmethod
    def from_cfg(cls, cfg: dict | None) -> "TaskCatalogue":
        """Parse ``tasks:`` from profiles.yaml.

        A malformed declaration is warned about and dropped rather than
        failing the pack: a bad task must not take the fleet offline.
        """
        by_id: dict[str, Task] = {}
        for name, spec in (cfg or {}).items():
            task = _parse(name, spec)
            if task is not None:
                by_id[task.id] = task
        return cls(by_id=by_id)


def _parse(name: object, spec: object) -> Task | None:
    if not isinstance(name, str) or not _ID_RE.match(name.strip()):
        log.warning("tasks: %r is not a task id (lowercase letters, digits, "
                    "'.', '-', '_'); ignoring it", name)
        return None
    task_id = name.strip()
    if isinstance(spec, int) and not isinstance(spec, bool):
        spec = {"min_context": spec}
    if not isinstance(spec, dict):
        log.warning("tasks: %s is not a mapping; ignoring it", task_id)
        return None
    endpoint = spec.get("endpoint")
    if endpoint not in ENDPOINTS:
        log.warning("tasks: %s.endpoint=%r is not an endpoint (one of %s); "
                    "ignoring the task", task_id, endpoint,
                    ", ".join(sorted(ENDPOINTS)))
        return None
    role = spec.get("role")
    if role not in SERVED_ROLES:
        log.warning("tasks: %s.role=%r is not a scheduling role (one of %s); "
                    "ignoring the task", task_id, role,
                    ", ".join(SERVED_ROLES))
        return None
    task = Task(task_id, str(endpoint), str(role))
    changes: dict = {}
    for key, value in spec.items():
        if key in ("endpoint", "role"):
            continue
        if key == "intent":
            changes["intent"] = str(value)
        elif key == "min_context":
            parsed = parse_ctx_size(value)
            if parsed is None or parsed <= 0:
                log.warning("tasks: %s.min_context=%r is not a positive size; "
                            "ignoring it", task_id, value)
                continue
            changes["min_context"] = parsed
        elif key == "min_parallel":
            if not isinstance(value, int) or isinstance(value, bool) \
                    or value < 1:
                log.warning("tasks: %s.min_parallel=%r is not a positive "
                            "integer; ignoring it", task_id, value)
                continue
            changes["min_parallel"] = value
        elif key in ("require", "keep"):
            if not isinstance(value, (list, tuple)):
                log.warning("tasks: %s.%s=%r is not a list of capabilities; "
                            "ignoring it", task_id, key, value)
                continue
            unknown = [str(v) for v in value
                       if str(v) not in KNOWN_CAPABILITIES]
            if unknown:
                log.warning("tasks: %s.%s names unknown capabilities %s; "
                            "ignoring them", task_id, key, ", ".join(unknown))
            changes[key] = frozenset(
                str(v) for v in value if str(v) in KNOWN_CAPABILITIES)
        elif key == "sacrifice":
            if not isinstance(value, (list, tuple)):
                log.warning("tasks: %s.sacrifice=%r is not a list; ignoring "
                            "it", task_id, value)
                continue
            changes[key] = tuple(str(v) for v in value)
        else:
            log.warning("tasks: %s.%s is not a task key; ignoring it",
                        task_id, key)
    return replace(task, **changes)


def validate_task_ids(raw: object, catalogue: TaskCatalogue) -> list[str]:
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
        elif catalogue.get(name.strip()) is None:
            problems.append(f"tasks: {name!r} is not a declared task")
    return problems


def resolve_tasks(tags: object, catalogue: TaskCatalogue,
                  log_=None, stem: str | None = None) -> tuple[Task, ...]:
    """Resolve a sidecar's ``tasks:`` ids against the catalogue.

    An undeclared id is warned about and ignored, so the effective tasks
    are always exactly the subset of ``profiles.yaml`` the operator wrote.
    A bare string is one task; no tags means no demands.
    """
    log_ = log_ or log
    who = f"{stem}: tasks:" if stem else "tasks:"
    if tags is None:
        return ()
    if isinstance(tags, str):
        tags = [tags]
    if not isinstance(tags, (list, tuple)):
        log_.warning("%s %r is not a list of task names; ignoring it",
                     who, tags)
        return ()
    out: list[Task] = []
    for tag in tags:
        if not isinstance(tag, str):
            log_.warning("%s %r is not a task name; ignoring it", who, tag)
            continue
        task = catalogue.get(tag.strip())
        if task is None:
            log_.warning("%s %r is not a declared task; ignoring it "
                         "(declared: %s)", who, tag.strip(),
                         ", ".join(catalogue.order) or "(none)")
            continue
        if task not in out:
            out.append(task)
    return tuple(out)
