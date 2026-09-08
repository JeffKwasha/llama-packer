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

#: Per-level style used when log records are rendered through the rich
#: console while the bar is live (keeps the `X | message` log shape).
_LEVEL_STYLES: dict[int, str] = {
    logging.DEBUG: "dim",
    logging.WARNING: "yellow",
    logging.ERROR: "bold red",
    logging.CRITICAL: "bold red",
}

try:
    import rich.console as _rc
    import rich.progress as _rp

    _RICH_AVAILABLE = True
except ImportError:  # rich is optional; every method below no-ops then
    _rp = None  # type: ignore[assignment]
    _rc = None  # type: ignore[assignment]
    _RICH_AVAILABLE = False


class _ConsoleLogHandler(logging.Handler):
    """Render log records through the progress bar's rich console.

    While a rich Live bar is on screen, plain writes to another stream
    land *after* the bar and force it to redraw ("scrolling"). Records
    printed via the bar's own console are positioned above it by rich.
    """

    def __init__(self, progress: "PackerProgress") -> None:
        super().__init__()
        self._progress = progress

    def emit(self, record: logging.LogRecord) -> None:
        progress = self._progress._progress
        if progress is None:
            return
        text = f"{record.levelname[:1]} | {record.getMessage()}"
        try:
            progress.console.print(
                text, style=_LEVEL_STYLES.get(record.levelno),
                markup=False, highlight=False)
        except Exception:
            self.handleError(record)


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
        self._log_handler: _ConsoleLogHandler | None = None
        self._saved_handlers: list[logging.Handler] | None = None

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
            # Own console bound to stderr: log output (also stderr) shares
            # this stream, and records printed via it land above the bar
            # instead of fighting a separate stdout Live display.
            console = _rc.Console(stderr=True, highlight=False)  # type: ignore[union-attr]
            self._progress = _rp.Progress(
                _rp.BarColumn(),
                _rp.TextColumn("{task.completed}/{task.total}"),
                _rp.TimeElapsedColumn(),
                _rp.TextColumn("[progress.description]{task.description}"),
                console=console,
            )
            self._task = self._progress.add_task(description, total=total)
            self._progress.start()
        except Exception as e:
            logger.debug("progress bar start failed (%s); continuing without it", e)
            self._progress = None
            self._task = None
            return
        # Route logging through the bar's console so messages print above
        # it; the plain stderr handler is restored on stop().
        root = logging.getLogger()
        self._saved_handlers = root.handlers[:]
        for handler in self._saved_handlers:
            root.removeHandler(handler)
        self._log_handler = _ConsoleLogHandler(self)
        root.addHandler(self._log_handler)

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
        # Restore the handlers that were active before the bar started.
        root = logging.getLogger()
        if self._log_handler is not None:
            root.removeHandler(self._log_handler)
            self._log_handler = None
        if self._saved_handlers is not None:
            for handler in self._saved_handlers:
                if handler not in root.handlers:
                    root.addHandler(handler)
            self._saved_handlers = None
