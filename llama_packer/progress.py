# llama_packer/progress.py
"""Optional progress bar for long runs. No-op when ``rich`` is unavailable.

``rich`` is an optional dependency (``pip install llama-packer[progress]``).
Every method here returns instantly when ``rich`` cannot be imported, when
output is not a TTY, or when running under pytest — so callers never need
their own guards and tests never see a progress bar.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

logger = logging.getLogger(__name__)

try:
    import rich.progress as _rp

    _RICH_AVAILABLE = True
except ImportError:  # rich is optional; every method below no-ops then
    _rp = None  # type: ignore[assignment]
    _RICH_AVAILABLE = False


class PackerProgress:
    """Monotonic models-completed/total bar with a current-action label.

    The bar is only shown once the denominator is known — call :meth:`start`
    after discovery with ``total=len(models)``. Progress is simply completed
    items over total; no ETA math.
    """

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled
        self._progress: Any = None
        self._task: Any = None
        self._total = 0
        self._done = 0

    def _active(self) -> bool:
        if not self._enabled or not _RICH_AVAILABLE or _rp is None:
            return False
        if os.environ.get("PYTEST_CURRENT_TEST"):
            return False
        try:
            if not sys.stderr.isatty():
                return False
        except Exception:
            return False
        return True

    def start(self, total: int, description: str = "budgeting") -> None:
        """Show the bar now that the denominator is known. No-op otherwise."""
        self._total = total
        self._done = 0
        if not self._active() or total <= 0:
            return
        try:
            assert _rp is not None
            # Fixed-width columns first (bar, counts, elapsed stay put);
            # the variable-width action label goes last so nothing jumps.
            self._progress = _rp.Progress(
                _rp.BarColumn(),
                _rp.TextColumn("{task.completed}/{task.total}"),
                _rp.TimeElapsedColumn(),
                _rp.TextColumn("[progress.description]{task.description}"),
            )
            self._task = self._progress.add_task(description, total=total)
            self._progress.start()
        except Exception as e:
            logger.debug("progress bar start failed (%s); continuing without it", e)
            self._progress = None
            self._task = None

    def advance(self, description: str = "") -> None:
        """Tick one completed item, optionally updating the action label."""
        self._done += 1
        if self._progress is None or self._task is None:
            return
        try:
            if description:
                self._progress.update(self._task, description=description)
            self._progress.advance(self._task)
        except Exception as e:
            logger.debug("progress bar advance failed (%s)", e)

    def stop(self) -> None:
        """Clear the status line. Normal log output above it persists."""
        if self._progress is None:
            return
        try:
            self._progress.stop()
        except Exception as e:
            logger.debug("progress bar stop failed (%s)", e)
        finally:
            self._progress = None
            self._task = None
