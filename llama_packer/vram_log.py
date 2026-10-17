"""Long-horizon VRAM baseline logger.

Background VRAM usage varies on a timescale of **days**, not minutes, and it
is asymmetric: loading a model file takes minutes (platter read) and a
browser/desktop footprint drifts up over hours, but release is fast and the
result then hangs at a plateau for a long time.  A short sample therefore
cannot separate the always-resident *base* (driver + compositor + scanout)
from the *variable* envelope the operator must cover with ``--spare`` — and a
fast cadence would not help, because the peaks are long-lived.

This tool samples once per ``--interval`` and writes a machine-local CSV,
tagging each row with a lightweight tenant fingerprint.  Two economies keep
it cheap enough to leave running for days:

- **Run-length counts.**  A sample within ``--threshold`` MiB of the pending
  row does not open a new row; it bumps that row's ``count``.  Flat plateaus
  (the common case) cost one row no matter how long they last.
- **Deferred writes.**  While usage is at or below ``--low-mib`` the rows are
  held in RAM and not written at all — spinning up the disks to log an idle
  machine defeats the purpose.  They are flushed once usage rises above the
  mark, and always at shutdown.

Leave it running for a few days; the floor of ``used_mib`` is the base, the
upper percentiles are the variable, and the tenant columns say *what* drove
each excursion.

    python -m llama_packer.vram_log                 # log forever (5 min)
    python -m llama_packer.vram_log --once          # one sample, to stdout
    python -m llama_packer.vram_log --report        # analyze what's logged

The report's ``floor`` is the number to hand the planner as the fixed base;
the gap between the floor and the tenant-specific medians is the slack for
``--spare``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from llama_packer import gpu_state

logger = logging.getLogger("llama_packer.vram_log")

#: The envelope is a slow sawtooth: a model file loads over **minutes**
#: (platter read) and the desktop/browser footprint drifts **up over hours**,
#: while release is fast and the result then plateaus for a long time.  A
#: five-minute cadence samples that shape faithfully — the peaks are
#: long-lived, so they are never aliased away, and the run-length counter
#: collapses the plateau.  Faster sampling buys nothing the shape gives.
_DEFAULT_INTERVAL = 300.0

#: Default run-length threshold (MiB): a sample within this of the pending
#: row increments its ``count`` instead of opening a new row.
_DEFAULT_THRESHOLD = 128

#: Default idle ceiling (MiB): while ``used_mib`` is at or below this, rows
#: are held in RAM and not written, so an idle machine's disks are not woken.
_DEFAULT_LOW_MIB = 2048

_FIELDS = ("time", "total_mib", "used_mib", "free_mib", "gtt_mib", "count",
           "models", "audio", "image", "immich", "browsers")

# Tenant matchers, applied to `ps -eo user=,args=`.  Keep them cheap and
# over-inclusive: a false positive shifts a bucket, a false negative hides a
# load that actually moved VRAM.  ``models`` is a served *model* process
# (llama-server / vllm) — not the llama-swap manager, which is always up.
_TENANT_PATTERNS = {
    "models": ("llama-server", "llama-fit-params", "vllm"),
    "audio": ("audiocpp_server", "whisper-server"),
    "image": ("sd-server", "comfy"),
    "immich": ("immich_ml", "immich-machine-learning", "/immich/"),
    "browsers": ("librewolf", "firefox", "chrome", "chromium", "/opt/google/chrome"),
}


def default_log_path() -> Path:
    """``~/.cache/llama-packer/vram_baseline.csv`` (honors the cache env)."""
    return gpu_state.cache_dir() / "vram_baseline.csv"


def visible_mem() -> tuple[int, int, int, int] | None:
    """(total, used, free, gtt) MiB for the visible GPU, or ``None``.

    amd-smi is preferred (it reports GTT too); nvidia-smi is the fallback
    (no GTT — reported as 0); sysfs is the last resort.
    """
    try:
        out = subprocess.run(["amd-smi", "metric", "-m", "--json"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            m = json.loads(out.stdout)["gpu_data"][0]["mem_usage"]
            return (int(m["total_vram"]["value"]), int(m["used_vram"]["value"]),
                    int(m["free_vram"]["value"]), int(m["used_gtt"]["value"]))
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError):
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            total, used, free = (int(x) for x in
                                 out.stdout.strip().splitlines()[0].split(","))
            return (total, used, free, 0)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        pass
    try:
        for card in Path("/sys/class/drm").glob("card*/device"):
            total_f = card / "mem_info_vram_total"
            used_f = card / "mem_info_vram_used"
            if not (total_f.is_file() and used_f.is_file()):
                continue
            total = int(total_f.read_text()) // (1024 * 1024)
            used = int(used_f.read_text()) // (1024 * 1024)
            gtt_f = card / "mem_info_gtt_used"
            gtt = int(gtt_f.read_text()) // (1024 * 1024) if gtt_f.is_file() else 0
            return (total, used, total - used, gtt)
    except (OSError, ValueError):
        pass
    return None


def _process_lines() -> list[str]:
    try:
        out = subprocess.run(["ps", "-eo", "user=,args="],
                             capture_output=True, text=True, timeout=15)
        out.check_returncode()
        return [ln for ln in out.stdout.splitlines() if ln.strip()]
    except (OSError, subprocess.SubprocessError):
        return []


def tenants(lines: list[str] | None = None) -> dict[str, int]:
    """Counts of the known VRAM tenants in the process table."""
    lines = _process_lines() if lines is None else lines
    counts = {name: 0 for name in _TENANT_PATTERNS}
    for ln in lines:
        low = ln.lower()
        for name, pats in _TENANT_PATTERNS.items():
            if any(p in low for p in pats):
                counts[name] += 1
    return counts


def sample() -> dict:
    """One row: ISO-8601 timestamp + visible memory + tenant fingerprint."""
    mem = visible_mem()
    row: dict[str, object] = {
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    if mem is None:
        row.update(total_mib="", used_mib="", free_mib="", gtt_mib="")
    else:
        total, used, free, gtt = mem
        row.update(total_mib=total, used_mib=used, free_mib=free, gtt_mib=gtt)
    row.update(tenants())
    row["count"] = 1
    return row


def append_row(path: Path, row: dict) -> None:
    """Append *row*, writing the header once (create parent on demand)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(_FIELDS), extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)
        f.flush()


class Accumulator:
    """Run-length compress samples and defer disk writes while idle.

    A sample within *threshold* MiB of the pending row bumps that row's
    ``count``; only a bigger move (or a tenant-driven value change) opens a
    new row.  Rows are buffered in RAM and written only once usage exceeds
    *low_mib* — or at :meth:`close` — so an idle machine's disks are left
    alone.  The trade is that the still-open row and any buffered idle rows
    are lost if the process is killed outright, which is the point: silence
    over durability while there is nothing to record.
    """

    def __init__(self, path: Path, *, threshold: int = _DEFAULT_THRESHOLD,
                 low_mib: int = _DEFAULT_LOW_MIB) -> None:
        self.path = Path(path)
        self.threshold = threshold
        self.low_mib = low_mib
        self.pending: dict | None = None
        self.buffer: list[dict] = []

    @staticmethod
    def _used(row: dict) -> int | None:
        v = row.get("used_mib")
        if v in ("", None):
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    def add(self, row: dict) -> bool:
        """Fold *row* in; return True when rows were flushed to disk."""
        used = self._used(row)
        pend = self._used(self.pending) if self.pending is not None else None
        if (self.pending is not None and used is not None and pend is not None
                and abs(used - pend) <= self.threshold):
            self.pending["count"] = int(self.pending.get("count", 1)) + 1
            return False
        if self.pending is not None:
            self.buffer.append(self.pending)
        self.pending = {**row, "count": 1}
        if used is not None and used > self.low_mib:
            self.flush()
            return True
        return False

    def flush(self) -> None:
        """Write buffered (finalized) rows; the open row is untouched."""
        for r in self.buffer:
            append_row(self.path, r)
        self.buffer.clear()

    def close(self) -> None:
        """Finalize the open row and flush everything still in RAM."""
        if self.pending is not None:
            self.buffer.append(self.pending)
            self.pending = None
        self.flush()


def _percentile(sorted_vals: list[int], q: float) -> int:
    if not sorted_vals:
        return 0
    idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def _weight(row: dict) -> int:
    try:
        return max(1, int(row.get("count") or 1))
    except (TypeError, ValueError):
        return 1


def report(path: Path) -> str:
    """Human summary: floor/percentiles plus per-tenant used_mib medians.

    Each row stands for ``count`` samples (run-length), so the statistics are
    weighted by it — a long plateau is not outvoted by a shower of one-off
    samples.  Old files without a ``count`` column weigh 1.
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    used: list[int] = []
    buckets: dict[str, list[int]] = {}
    for r in rows:
        if not r.get("used_mib"):
            continue
        try:
            u = int(r["used_mib"])
        except ValueError:
            continue
        n = _weight(r)
        used.extend([u] * n)
        for name in ("models", "audio", "image", "immich", "browsers"):
            if int(r.get(name) or 0) > 0:
                buckets.setdefault(name, []).extend([u] * n)
    used.sort()
    out: list[str] = [
        f"samples: {len(used)}  rows: {len(rows)}  window: "
        f"{rows[0]['time'] if rows else '-'} .. "
        f"{rows[-1]['time'] if rows else '-'}"]
    if not used:
        return "\n".join(out) + "\n(no usable memory samples)"
    out.append(
        f"used_mib: floor={used[0]} p10={_percentile(used, .10)} "
        f"p50={_percentile(used, .50)} p90={_percentile(used, .90)} "
        f"p99={_percentile(used, .99)} max={used[-1]} "
        f"spread={used[-1] - used[0]}")
    for name, vals in sorted(buckets.items()):
        vals.sort()
        out.append(f"  with {name:8s}: n={len(vals):6d} median={_percentile(vals, .5)} "
                   f"min={vals[0]} max={vals[-1]}")
    out.append("\nfloor ≈ fixed base (driver+compositor); the tenant deltas "
               "above the floor are the slack for --spare")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m llama_packer.vram_log",
        description="Log visible VRAM + tenant fingerprint over days.")
    parser.add_argument("--out", type=Path, default=None,
                        help="CSV path (default: ~/.cache/llama-packer/vram_baseline.csv)")
    parser.add_argument("--interval", type=float, default=_DEFAULT_INTERVAL,
                        help="seconds between samples (default 300 = 5 min)")
    parser.add_argument("--threshold", type=int, default=_DEFAULT_THRESHOLD,
                        help="MiB within which a sample bumps the pending "
                             "row's count instead of opening a new row "
                             "(default 128)")
    parser.add_argument("--low-mib", type=int, default=_DEFAULT_LOW_MIB,
                        help="while used_mib is at or below this, hold rows "
                             "in RAM (no disk writes) until usage rises or "
                             "the process exits (default 2048)")
    parser.add_argument("--once", action="store_true",
                        help="print a single sample and exit")
    parser.add_argument("--report", action="store_true",
                        help="print a summary of --out and exit")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="stop after N samples (0 = forever)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    path = args.out or default_log_path()

    if args.report:
        if not path.exists():
            logger.error("no log at %s", path)
            return 1
        sys.stdout.write(report(path))
        return 0

    if args.once:
        row = sample()
        w = csv.DictWriter(sys.stdout, fieldnames=list(_FIELDS),
                           extrasaction="ignore")
        w.writeheader()
        w.writerow(row)
        return 0

    logger.info("logging VRAM to %s every %.0fs (pid %d) — Ctrl-C to stop",
                path, args.interval, os.getpid())
    logger.info("run-length ±%d MiB; no disk writes while used ≤ %d MiB",
                args.threshold, args.low_mib)
    acc = Accumulator(path, threshold=args.threshold, low_mib=args.low_mib)
    n = 0
    try:
        while args.max_samples <= 0 or n < args.max_samples:
            acc.add(sample())
            n += 1
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        acc.close()
    logger.info("stopped after %d samples", n)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
