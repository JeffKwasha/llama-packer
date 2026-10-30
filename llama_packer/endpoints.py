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
