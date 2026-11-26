#!/usr/bin/env python3
"""Local VRAM JSON API. See docs/memory-and-eviction.md §Live VRAM attribution."""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

VRAM_USED = "/sys/class/drm/card1/device/mem_info_vram_used"
VRAM_TOTAL = "/sys/class/drm/card1/device/mem_info_vram_total"

SAMPLE_INTERVAL = 15
HISTORY_SECONDS = 300


def read_sysfs(path):
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


DEBUGFS_CARD = "/sys/kernel/debug/dri/0000:2d:00.0"


def _fdinfo_kv(path):
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return {}
    out = {}
    for line in lines:
        key, sep, value = line.partition(":")
        if sep:
            out[key.strip()] = value.strip()
    return out


def _mib(value):
    """An fdinfo size like '711892 KiB' in MiB; None when unusable."""
    if not value:
        return None
    parts = value.split()
    try:
        number = float(parts[0])
    except ValueError:
        return None
    unit = parts[1].upper() if len(parts) > 1 else "KIB"
    if unit == "KIB":
        number /= 1024
    elif unit == "B":
        number /= 1024 * 1024
    elif unit not in ("MIB", "MB"):
        return None
    return round(number, 1)


def sample_gpu_processes_fdinfo(proc_root="/proc"):
    """Per-process VRAM from DRM fdinfo, no root and no debugfs.

    Every fd on a DRM or KFD node carries `drm-client-id` plus resident VRAM
    and GTT sizes. Several fds on one client are one allocation, so dedupe by
    (pid, client) and sum distinct clients in a process. Processes the caller
    cannot read are skipped; they show up in the debugfs fallback when root
    is available, or not at all.
    """
    root = Path(proc_root)
    seen: set[tuple[int, str]] = set()
    per_pid: dict[int, dict] = {}
    for pid_dir in root.iterdir():
        if not pid_dir.name.isdigit():
            continue
        pid = int(pid_dir.name)
        try:
            fds = list((pid_dir / "fd").iterdir())
        except OSError:
            continue
        for fd in fds:
            try:
                target = os.readlink(fd)
            except OSError:
                continue
            node = Path(target).name
            if not node.startswith(("renderD", "card", "kfd")):
                continue
            info = _fdinfo_kv(pid_dir / "fdinfo" / fd.name)
            client = info.get("drm-client-id") or info.get("pasid") or fd.name
            key = (pid, client)
            if key in seen:
                continue
            seen.add(key)
            vram = _mib(info.get("drm-resident-vram")
                        or info.get("drm-total-vram"))
            gtt = _mib(info.get("drm-resident-gtt")
                       or info.get("drm-total-gtt"))
            if vram is None and gtt is None:
                continue
            entry = per_pid.setdefault(pid, {
                "pid": pid,
                "name": (pid_dir / "comm").read_text().strip()
                if (pid_dir / "comm").is_file() else "?",
                "vram_mib": 0.0, "gtt_mib": 0.0,
            })
            entry["vram_mib"] += vram or 0.0
            entry["gtt_mib"] += gtt or 0.0
    procs = sorted(per_pid.values(), key=lambda p: p["vram_mib"], reverse=True)
    for proc in procs:
        proc["vram_mib"] = round(proc["vram_mib"], 1)
        proc["gtt_mib"] = round(proc["gtt_mib"], 1)
    return procs


def sample_gpu_processes_with_source():
    """``(processes, source)`` so callers can judge attribution coverage."""
    procs = sample_gpu_processes_fdinfo()
    if procs:
        return procs, "fdinfo"
    procs = _sample_gpu_processes_debugfs()
    if procs and any(p.get("vram_mib") is not None for p in procs):
        return procs, "debugfs"
    if procs:
        return procs, "process-names"
    return [], "none"


def _sample_gpu_processes_debugfs():
    """Root-only fallback; docs/memory-and-eviction.md §Live VRAM attribution."""
    card = Path(DEBUGFS_CARD)
    if not card.is_dir():
        # Fall back to /proc scanning (no VRAM, just process names)
        return _sample_gpu_processes_fallback()

    # Parse the clients file: command, tgid, dev, master, a, uid, magic, name, id
    clients_file = card / "clients"
    try:
        lines = clients_file.read_text().splitlines()
    except OSError:
        return _sample_gpu_processes_fallback()

    # Map client id -> (pid, name)
    client_map = {}
    for line in lines[1:]:  # skip header
        parts = line.split()
        if len(parts) < 9:
            continue
        name = parts[0]
        try:
            tgid = int(parts[1])
            client_id = int(parts[8])
        except (ValueError, IndexError):
            continue
        client_map[client_id] = (tgid, name)

    procs = []
    for client_id, (pid, name) in client_map.items():
        vram_file = card / f"client-{client_id}" / "amdgpu_vram_mm"
        try:
            content = vram_file.read_text()
        except OSError:
            continue
        usage = None
        for line in content.splitlines():
            if line.strip().startswith("usage:"):
                try:
                    usage = int(line.split(":")[1].strip()) // (1024 * 1024)
                except (ValueError, IndexError):
                    pass
                break
        if usage is not None:
            procs.append({"pid": pid, "name": name, "vram_mib": usage})

    # Sort by VRAM descending so the biggest user is first
    procs.sort(key=lambda p: p.get("vram_mib", 0), reverse=True)
    return procs


def _sample_gpu_processes_fallback():
    """Fallback: scan /proc for processes with GPU open (no VRAM)."""
    procs = []
    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit():
            continue
        pid = int(pid_dir.name)
        fd_dir = pid_dir / "fd"
        try:
            targets = [os.readlink(str(fd)) for fd in fd_dir.iterdir()
                       if os.path.islink(fd)]
        except OSError:
            continue
        if not any(t.startswith("/dev/dri") or t == "/dev/kfd"
                   for t in targets):
            continue
        try:
            name = (pid_dir / "comm").read_text().strip()
        except OSError:
            name = "?"
        procs.append({"pid": pid, "name": name, "vram_mib": None})
    return procs


def take_sample():
    total = read_sysfs(VRAM_USED)
    total_max = read_sysfs(VRAM_TOTAL)
    procs, source = sample_gpu_processes_with_source()
    complete = source in ("fdinfo", "debugfs")
    note = ""
    if not complete and total is not None and total > 512 * 1024**2:
        note = ("GPU is in use but no process has a VRAM figure; other "
                "users' processes (e.g. lemonade's llama-swap children) "
                "are not visible via fdinfo without their credentials")
    return {
        "ts": time.time(),
        "vram_used_mib": total // (1024 * 1024) if total is not None else None,
        "vram_total_mib": total_max // (1024 * 1024) if total_max is not None else None,
        "processes": procs,
        "attribution": {
            "source": source,
            "complete": complete,
            "note": note,
        },
    }


class Monitor:
    def __init__(self, interval, history_seconds):
        self.interval = interval
        self.history_seconds = history_seconds
        self.samples = []
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    def _loop(self):
        while not self._stop.is_set():
            s = take_sample()
            with self.lock:
                self.samples.append(s)
                # trim to history window
                cutoff = s["ts"] - self.history_seconds
                self.samples = [x for x in self.samples if x["ts"] >= cutoff]
            self._stop.wait(self.interval)

    def current(self):
        with self.lock:
            return self.samples[-1] if self.samples else None

    def history(self, from_ts=None, to_ts=None):
        with self.lock:
            out = list(self.samples)
        if from_ts is not None:
            out = [s for s in out if s["ts"] >= from_ts]
        if to_ts is not None:
            out = [s for s in out if s["ts"] <= to_ts]
        return out


class Handler(BaseHTTPRequestHandler):
    monitor = None

    def do_GET(self):
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._respond(200, {"status": "ok"})
        elif parsed.path == "/current":
            s = self.monitor.current()
            self._respond(200, s or {"status": "no samples yet"})
        elif parsed.path == "/history":
            qs = parse_qs(parsed.query)
            from_ts = float(qs["from"][0]) if "from" in qs else None
            to_ts = float(qs["to"][0]) if "to" in qs else None
            samples = self.monitor.history(from_ts, to_ts)
            self._respond(200, {"count": len(samples), "samples": samples})
        else:
            self._respond(404, {"error": "not found"})

    def _respond(self, code, body):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body, indent=2).encode())

    def log_message(self, *args):
        pass  # quiet


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=18099)
    ap.add_argument("--interval", type=int, default=15)
    ap.add_argument("--history", type=int, default=300)
    args = ap.parse_args()

    mon = Monitor(args.interval, args.history)
    Handler.monitor = mon
    mon.start()

    # take one immediate sample so /current works right away
    mon.samples.append(take_sample())

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"gpu-vram-monitor on http://127.0.0.1:{args.port}")
    print(f"  sampling every {args.interval}s, {args.history}s history")
    print(f"  endpoints: /current  /history?from=T&to=T  /health")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        mon.stop()
        print("\nstopped")


if __name__ == "__main__":
    main()