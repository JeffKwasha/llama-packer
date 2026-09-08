"""Measurement-window GPU introspection: my children, my leaks.

A VRAM measurement is only as good as the exclusivity of the GPU it runs
on — and the 2026-09-07 incident (a measurement ``llama-server`` left
resident for ~8 h, silently shifting every later reading) happened
because nothing verified the probe's own children died.  Non-root, this
process can only ever see per-pid VRAM for *its own* processes — and
that is exactly sufficient:

- **my spawned servers** — fully visible to me (pid, ``/proc``); the
  structural kill (:func:`kill_process_group`, used by the serve
  measurement on every exit path) plus post-flight verification means a
  child cannot outlive its measurement.
- **foreign llama processes** — a zombie from any earlier run, or the
  live llama-swap — are visible by name via ``ps``;
  :func:`llama_residents` is the pre-flight refusal list.
- **everything else** (games, desktop) is invisible per-process and
  irrelevant: a measurement whose buffers cannot fit beside it spills
  into host-visible memory, which the llama.cpp log itself reports —
  the spill check in :mod:`llama_packer.vram` rejects such a point
  instead of quietly undercounting it.

Also home to the measurement campaign's shared machine-local artifacts:
the single-flight :func:`measurement_lock` and the append-only
:func:`journal` (pid, command, outcome — the forensic record that was
missing when the contamination had to be reconstructed by hand).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import signal
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)

#: A llama.cpp process resident on the box (any user, any working
#: directory) makes a VRAM measurement untrustworthy: it holds VRAM our
#: measurement server cannot account for.  The match is anchored on a
#: path component, so a test double named ``fake-llama-server`` (or a
#: document called ``llama-server.txt``) does not trigger it.
_RESIDENT_RE = re.compile(
    r"(?:^|/)llama-(?:server|fit-params)(?:\s|$)|llama-swap(?:\s|$)")

#: Seconds a graceful kill is given before escalation to SIGKILL.
_KILL_GRACE_S = 15.0


def cache_dir() -> Path:
    """Machine-local home for measurement artifacts (corrections, lock,
    journal).  ``LLAMA_PACKER_CACHE_DIR`` overrides; created on demand."""
    base = os.environ.get("LLAMA_PACKER_CACHE_DIR")
    root = Path(base) if base else Path.home() / ".cache" / "llama-packer"
    root.mkdir(parents=True, exist_ok=True)
    return root


def llama_residents(ps_bin: str = "ps") -> list[str]:
    """Descriptions of resident llama.cpp processes, ``[]`` when clear.

    The pre-flight for any measurement that allocates VRAM: a leftover
    measurement server from an earlier run squats the GPU and poisons
    every reading taken beside it.
    """
    try:
        out = subprocess.run([ps_bin, "-eo", "pid=,user=,args="],
                             capture_output=True, text=True, timeout=15)
        out.check_returncode()
    except (OSError, subprocess.SubprocessError) as e:
        # ps is coreutils-grade available; failing open is the honest
        # choice only because the spill check independently rejects
        # points that could not fit device memory.
        logger.warning("resident scan failed (%s); proceeding unguarded", e)
        return []
    return [line.strip() for line in out.stdout.splitlines()
            if _RESIDENT_RE.search(line)]


def kill_process_group(proc: subprocess.Popen) -> None:
    """Terminate, then hard-kill, the child's whole process group.

    The measurement server is our child — its death on every exit path
    is not optional (2026-09-07 incident).  SIGTERM first so the server
    can flush, SIGKILL after the grace window, then verify: our own
    processes are always visible to us, so a survivor is detectable.
    """
    if proc.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=_KILL_GRACE_S)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    try:
        os.kill(proc.pid, 0)
    except ProcessLookupError:
        return
    logger.error("kill_process_group: pid %s survived SIGKILL", proc.pid)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextlib.contextmanager
def measurement_lock(name: str = "measure"):
    """Single-flight lock for measurement campaigns.

    An ``O_EXCL`` lock file in :func:`cache_dir` holding the owner's
    pid; a lock whose pid is provably dead is stolen (crashed runs must
    not wedge the tool forever).  Raises ``RuntimeError`` when another
    live process holds it — two concurrent measurement campaigns would
    contend for the GPU (and, on platter storage, the drives).
    """
    path = cache_dir() / f"{name}.lock"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            other = int(path.read_text().strip() or "0")
        except (OSError, ValueError):
            other = 0
        if other and _pid_alive(other):
            raise RuntimeError(
                f"another measurement holds {path} (pid {other})") from None
        logger.warning("stealing stale measurement lock (dead pid %s)", other)
        with contextlib.suppress(OSError):
            path.unlink()
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, str(os.getpid()).encode())
    finally:
        os.close(fd)
    try:
        yield path
    finally:
        with contextlib.suppress(OSError):
            path.unlink()


def journal(entry: dict) -> None:
    """Append one JSON line to the measurement journal (best effort).

    The forensic record: every measurement's pid, command shape, outcome
    and duration — so contamination, if it ever happens again, is
    bounded by inspection instead of hours of guessing.  Never raises:
    a journaling failure must not break a measurement.
    """
    try:
        with open(cache_dir() / "measure-journal.jsonl", "a") as f:
            f.write(json.dumps(
                {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **entry},
                sort_keys=True) + "\n")
    except OSError:
        logger.warning("measurement journal write failed", exc_info=True)
