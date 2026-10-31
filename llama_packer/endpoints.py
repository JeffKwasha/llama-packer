"""The serving surfaces an engine exposes.

An endpoint names *where* work is served: a standardized id a server and a
client both understand. It is not a task (what work is being done) and not
an engine (how the weights run). The set is closed — an id that is not here
is not an endpoint — so compatibility is a set intersection, not a guess:

    task -> endpoint   (TASK_ENDPOINTS)
    engine -> endpoints it serves
    compatible iff the task's endpoint is in the engine's set

The task mapping is today's six role ids. P4 moves the catalog into
``tasks.py`` and this table becomes a view of it; until then this is the one
place the two vocabularies meet, so no engine ever names a task.
"""

from __future__ import annotations

import re
from pathlib import Path

CHAT = "openai.chat.completions"
EMBEDDINGS = "openai.embeddings"
RERANK = "openai.rerank"
IMAGES = "openai.images.generations"
TRANSCRIPTIONS = "openai.audio.transcriptions"
SPEECH = "openai.audio.speech"

ENDPOINTS: frozenset[str] = frozenset({
    CHAT, EMBEDDINGS, RERANK, IMAGES, TRANSCRIPTIONS, SPEECH,
})

#: task id -> the one endpoint it is served at.
TASK_ENDPOINTS: dict[str, str] = {
    "chat": CHAT,
    "embeddings": EMBEDDINGS,
    "rerank": RERANK,
    "image": IMAGES,
    "s2t": TRANSCRIPTIONS,
    "t2s": SPEECH,
}


#: engine name -> the endpoints it serves.  The one home: engines read their
#: ``endpoints`` from here, so a backend cannot advertise a surface this table
#: does not know.
ENGINE_ENDPOINTS: dict[str, frozenset[str]] = {
    "llama-server": frozenset({CHAT, EMBEDDINGS, RERANK}),
    "vllm": frozenset({CHAT, EMBEDDINGS, RERANK}),
    "sd-server": frozenset({IMAGES}),
    "whisper-server": frozenset({TRANSCRIPTIONS}),
    "audio-cpp": frozenset({TRANSCRIPTIONS, SPEECH}),
}


def for_task(task: object) -> str | None:
    """The endpoint *task* is served at, or None when it is not a task."""
    return TASK_ENDPOINTS.get(task) if isinstance(task, str) else None


def serves(endpoints: frozenset[str] | set[str], task: object) -> bool:
    """Whether an engine serving *endpoints* can perform *task*."""
    endpoint = for_task(task)
    return endpoint is not None and endpoint in endpoints


def tasks_for(endpoints: frozenset[str] | set[str]) -> tuple[str, ...]:
    """Task ids an engine serving *endpoints* can perform — for diagnostics."""
    return tuple(task for task, endpoint in TASK_ENDPOINTS.items()
                 if endpoint in endpoints)


def validate_engines_value(raw: object) -> list[str]:
    """Shape and vocabulary errors in a sidecar's ``engines:`` value.

    Returns human messages (empty when the value is sound).  Absence is not an
    error here — the caller decides whether the key is required.
    """
    if raw is None or raw == "":
        return []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return ["engines: must be a list of engine names"]
    if not raw:
        return ["engines: must name at least one engine"]
    problems: list[str] = []
    for name in raw:
        if not isinstance(name, str) or not name.strip():
            problems.append(f"engines: {name!r} is not an engine name")
        elif name not in ENGINE_ENDPOINTS:
            import difflib
            close = difflib.get_close_matches(name, sorted(ENGINE_ENDPOINTS),
                                              n=1, cutoff=0.6)
            hint = f" (did you mean {close[0]!r}?)" if close else ""
            problems.append(f"engines: {name!r} is not an engine{hint}")
    return problems


def check_registry() -> list[str]:
    """Invariant violations in the registry (empty when sound).

    Run by the test suite; a violation means the table, the engines and the
    closed endpoint set disagree.
    """
    problems: list[str] = []
    for task, endpoint in TASK_ENDPOINTS.items():
        if endpoint not in ENDPOINTS:
            problems.append(f"task {task!r} maps to unknown endpoint {endpoint!r}")
    served = {endpoint for eps in ENGINE_ENDPOINTS.values() for endpoint in eps}
    for endpoint in sorted(ENDPOINTS - served):
        problems.append(f"endpoint {endpoint!r} is served by no engine")
    for engine, eps in ENGINE_ENDPOINTS.items():
        if not eps:
            problems.append(f"engine {engine!r} serves no endpoint")
        for endpoint in sorted(eps - ENDPOINTS):
            problems.append(f"engine {engine!r} serves unknown endpoint {endpoint!r}")
    from llama_packer.backends import BACKENDS, ENGINE_NAMES
    for engine in sorted(ENGINE_ENDPOINTS):
        if engine not in ENGINE_NAMES:
            problems.append(f"engine {engine!r} is not registered")
    for bound in BACKENDS.values():
        expected = ENGINE_ENDPOINTS.get(bound.engine.name)
        if expected is not None and bound.endpoints != expected:
            problems.append(
                f"{bound.name!r} advertises {sorted(bound.endpoints)}, "
                f"registry says {sorted(expected)}")
    return problems


def docs_agree(root: str | Path) -> list[str]:
    """Drift between the registry and the docs that state it.

    Parses ``SPEC.md``'s engine/endpoint matrix and each
    ``docs/backends/<engine>.md`` **Endpoints:** line.  Returns human
    messages; empty means the docs match the table.
    """
    root = Path(root)
    problems: list[str] = []
    spec = (root / "SPEC.md").read_text()
    seen: dict[str, set[str]] = {}
    for line in spec.splitlines():
        if not line.startswith("| `") or "openai." not in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        engines = [e.strip("` ") for e in cells[0].split("/")]
        endpoints = set(re.findall(r"`(openai\.[a-z.]+)`", cells[-1]))
        for engine in engines:
            if engine in ENGINE_ENDPOINTS:
                seen[engine] = endpoints
    for engine, expected in ENGINE_ENDPOINTS.items():
        if engine not in seen:
            problems.append(f"SPEC.md states no endpoints for {engine!r}")
        elif seen[engine] != set(expected):
            problems.append(
                f"SPEC.md {engine!r}: {sorted(seen[engine])} != "
                f"registry {sorted(expected)}")
    for engine, expected in ENGINE_ENDPOINTS.items():
        doc = root / "docs" / "backends" / f"{engine}.md"
        if not doc.exists():
            problems.append(f"missing engine doc {doc}")
            continue
        text = doc.read_text()
        match = re.search(r"^- \*\*Endpoints:\*\*(.+)$", text, re.MULTILINE)
        if match is None:
            problems.append(f"{doc}: no '**Endpoints:**' line")
            continue
        stated = set(re.findall(r"`(openai\.[a-z.]+)`", match.group(1)))
        if stated != set(expected):
            problems.append(
                f"{doc}: {sorted(stated)} != registry {sorted(expected)}")
    return problems
