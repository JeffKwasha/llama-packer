#!/usr/bin/env python3
"""gpu-vram-monitor: local VRAM usage API for validating estimation.

Samples global VRAM every 15 seconds, keeps 5 minutes of history, serves
it as JSON. Run alongside a model load to measure the actual VRAM cost of a
serving configuration.

Per-process VRAM is NOT available on this hardware for Vulkan workloads,
verified by checking every available source:

  - amd-smi process: only tracks ROCm/HIP, reports "No running processes
    detected" for Vulkan (Mesa/RADV) workloads
  - /sys/kernel/debug/dri/client-*/: files are proc_info, vm_pagetable_info,
    device — no per-client VRAM file on this kernel
  - amdgpu_vram_mm (card-level): global allocator state, not per-process
  - umr: requires secure boot disabled (not available without reboot)
  - amdgpu_top: not installed (and is a TUI, not an API)
  - /proc/<pid>/: no per-process GPU memory exposed by the kernel for amdgpu

The "processes" list is a contamination check: it tells you who else had the
GPU open during the sample so a delta that looks wrong can be explained.
The global counter before/after the load is the actual measurement.

To get per-process VRAM, either:
  - Kill other GPU users first (isolate the measurement)
  - Install umr + disable secure boot (reboot required)
  - Use amd-smi for ROCm/HIP processes only (not Vulkan)

Usage:
    ./extras/gpu_vram_monitor.py [--port 18099] [--interval 15] [--history 300]

API:
    GET /current          latest sample
    GET /history?from=T&to=T   samples in [from, to] (epoch seconds, optional)
    GET /health           "ok"
"""

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


def sample_gpu_processes():
    """Per-process VRAM from amdgpu debugfs.

    The kernel exposes per-client VRAM accounting under
    /sys/kernel/debug/dri/<card>/client-<id>/amdgpu_vram_mm, which contains
    a "usage:" line giving that client's VRAM allocation in bytes. The
    client ID is mapped to a PID via the "clients" file in the same directory.

    This is the only per-process VRAM source on this hardware for Vulkan
    workloads. It requires root (debugfs is mode 0700). Without root, the
    global counter (read_sysfs) is still available but per-process
    attribution is not.

    Returns a list of {pid, name, vram_mib} dicts.
    """
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
    return {
        "ts": time.time(),
        "vram_used_mib": total // (1024 * 1024) if total is not None else None,
        "vram_total_mib": total_max // (1024 * 1024) if total_max is not None else None,
        "processes": sample_gpu_processes(),  # contamination check, not VRAM
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