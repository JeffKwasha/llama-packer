#!/usr/bin/env python3
"""bench — measure serving engines and keep the results as history.

llama.cpp ships prebuilt binaries and changes its defaults often enough that
"it used to be faster" is not a question anyone can answer after the fact.
This records a measurement instead: one line per run in a machine-local JSONL
file, keyed by engine version, backend, model and flags, so a later run can be
compared against the last run of the same thing and a regression is a number
rather than a feeling.

Two kinds of measurement, because they answer different questions:

* **Isolated** (``--engine``): ``llama-bench`` drives a binary directly over a
  grid of batch/ubatch values. Nothing else on the box matters, which makes it
  the honest A/B across versions and backends — and the way the batch/ubatch
  defaults on this fleet were chosen.
* **Live** (``--live``): timed requests through the running llama-swap, which
  is the only way to measure the audio engines at all. llama-bench knows
  nothing about them; audio.cpp reports a real-time factor in its own
  responses, and that is what gets recorded.

The history file is append-only and machine-local (``~/.cache/llama-packer/``).
Nothing here writes to a config, a profile, or a sidecar: a benchmark that can
change what it measures is not a benchmark.

Usage:
    ./extras/bench.py --engine llama-server            # grid, isolated
    ./extras/bench.py --engine llama-server --quick    # one point, no grid
    ./extras/bench.py --live --audio                   # TTS/ASR real-time factor
    ./extras/bench.py --report                         # the history so far
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import update as upd  # noqa: E402  (needs the path above)

HISTORY_NAME = "bench.jsonl"


class BenchError(RuntimeError):
    """A measurement that could not be taken."""


# --- where results live ----------------------------------------------------


def history_path() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(base) / "llama-packer" / HISTORY_NAME


def append_history(record: dict, path: Path | None = None) -> Path:
    """Append one record. Returns the file it went to.

    Append-only and line-delimited on purpose: a benchmark history is
    evidence, and the useful property of evidence is that appending to it
    cannot rewrite what came before. A crash mid-write costs one truncated
    last line, which is why each record is written in a single ``write`` of a
    line that ends in a newline.
    """
    path = path or history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return path


def load_history(path: Path | None = None) -> list[dict]:
    path = path or history_path()
    if not path.is_file():
        return []
    out: list[dict] = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue          # a truncated final line is not a reason to fail
    return out


# --- what to measure against ----------------------------------------------


def measurement_key(record: dict) -> tuple:
    """Identity of "the same measurement", for comparison purposes.

    Deliberately excludes wall-clock time, host load and repetition counts:
    two runs are comparable when they measured the same thing on the same
    configuration, and the numbers that differ between them are exactly what
    the history exists to show.
    """
    return (
        record.get("engine"), record.get("version"), record.get("backend"),
        device_identity(str(record.get("device", ""))), record.get("model"),
        record.get("batch"), record.get("ubatch"), record.get("metric"),
    )


def previous_for(record: dict, history: list[dict]) -> dict | None:
    """The most recent earlier record with the same key, or None."""
    key = measurement_key(record)
    earlier = [r for r in history if measurement_key(r) == key]
    return earlier[-1] if earlier else None


def delta_pct(new: float, old: float) -> float | None:
    if not old:
        return None
    return (new - old) / old * 100.0


# --- engines ---------------------------------------------------------------


def engine_version(binary: Path, timeout: int = 60) -> str:
    """``version: 0.7.0-dev (build 11717, commit c0fd78626)`` -> ``b11717``.

    Falls back to the file's mtime and size when the binary cannot be run —
    an unrunnable engine is exactly when a history entry is most worth having,
    so the record still gets made, with a version that says so.
    """
    try:
        result = subprocess.run([str(binary), "--version"], capture_output=True,
                                text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unrunnable ({exc.__class__.__name__})"
    out = f"{result.stdout}\n{result.stderr}"
    build = re.search(r"build (\d+)", out)
    if build:
        return f"b{build.group(1)}"
    version = re.search(r"version:\s*(\S+)", out)
    if version:
        return version.group(1)
    return f"unknown (mtime {int(binary.stat().st_mtime)})"


def detect_devices(binary: Path, timeout: int = 60) -> list[str]:
    """``Vulkan0: AMD Radeon AI Pro R9700 (RADV GFX1201)`` -> the full string.

    Device *name* is part of the measurement key on purpose: the same binary
    and backend on two different cards are two different measurements, and
    merging them produces a history that lies to whoever reads it.
    """
    try:
        result = subprocess.run([str(binary), "--list-devices"],
                                capture_output=True, text=True, timeout=timeout,
                                check=False)
    except (OSError, subprocess.TimeoutExpired):
        return []
    out = f"{result.stdout}\n{result.stderr}"
    found = []
    for line in out.splitlines():
        if re.match(r"^\s*(Vulkan|HIP|CUDA|CPU|ROCm|OpenCL)\s*\d+\s*:", line):
            found.append(" ".join(line.split()))
    return found


def device_identity(device_line: str) -> str:
    """The stable part of a device line — card name and architecture.

    ``--list-devices`` ends the line with the card's total and free memory,
    and that free counter moves by megabytes between two runs minutes apart.
    Keying history on the whole line makes every run a first run, which is a
    history that silently never compares anything. So the identity is the line
    with the trailing memory tuple removed, and the full line is kept beside
    it in the record as detail.
    """
    return re.sub(r"\s*\((?:\d+\s*(?:GiB|MiB)[^)]*)\)\s*$", "",
                  device_line).strip()


def backend_of(device_line: str) -> str:
    match = re.match(r"^\s*(Vulkan|HIP|CUDA|CPU|ROCm|OpenCL)\s*\d+", device_line)
    return (match.group(1).lower() if match else "unknown")


def vram_mib(device: str = "card1") -> int:
    """Current VRAM use, or -1 when there is no readable counter."""
    path = Path(f"/sys/class/drm/{device}/device/mem_info_vram_used")
    try:
        return int(path.read_text().strip()) // (1024 * 1024)
    except (OSError, ValueError):
        return -1


class VramSampler:
    """Samples VRAM on a thread while a benchmark runs.

    Peak VRAM is the number that decides how many servers fit, and it cannot
    be read after the fact — once the process exits, the counter is back to
    whatever else on the box is using the card. Sampling is the only way to
    have it in the record.
    """

    def __init__(self, device: str = "card1", interval: float = 0.05):
        self.device = device
        self.interval = interval
        self.peak = vram_mib(device)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "VramSampler":
        def run() -> None:
            while not self._stop.is_set():
                current = vram_mib(self.device)
                if current > self.peak:
                    self.peak = current
                time.sleep(self.interval)
        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)


# --- isolated: llama-bench -------------------------------------------------


def llama_bench_rows(bench: Path, model: Path, batch: int, ubatch: int,
                     n_prompt: int, n_gen: int, reps: int,
                     n_gpu_layers: int = 999, timeout: int = 300,
                     extra: list[str] | None = None) -> list[dict]:
    """One ``llama-bench`` invocation, parsed.

    Raises rather than returning partial numbers: a benchmark that reports
    half of what it was asked for is worse than one that says it could not
    run, because the half it does report looks like a result.
    """
    argv = [str(bench), "-m", str(model), "-b", str(batch), "-ub", str(ubatch),
            "-p", str(n_prompt), "-n", str(n_gen), "-r", str(reps),
            "-ngl", str(n_gpu_layers), "-o", "json", *(extra or [])]
    try:
        result = subprocess.run(argv, capture_output=True, text=True,
                                timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise BenchError(f"llama-bench timed out after {timeout}s "
                         f"({model.name} b{batch} ub{ubatch})") from exc
    if not result.stdout.strip():
        raise BenchError(f"llama-bench produced no output for {model.name} "
                         f"b{batch} ub{ubatch}: "
                         f"{(result.stderr or '').strip()[:200]}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise BenchError(f"llama-bench output was not JSON: {exc}") from exc


def pick_row(rows: list[dict], n_prompt: int, n_gen: int) -> dict:
    """The row for one (prompt, gen) pair, by decoded-token count.

    ``llama-bench`` emits a row per test case, and with ``-p 0``/``-n 0`` a
    case is skipped rather than absent — so selecting on the fields rather
    than on position is what keeps a prompt-processing number from being
    reported as a generation number.
    """
    for row in rows:
        n_p = row.get("n_prompt")
        n_g = row.get("n_gen")
        if n_p is None or n_g is None:
            continue
        if int(n_p) == n_prompt and int(n_g) == n_gen:
            return row
    raise BenchError(f"llama-bench returned no row for p={n_prompt} n={n_gen}")


def record_from_row(row: dict) -> dict:
    """``avg_ts`` -> tok/s, with the sample count kept alongside it.

    ``stddev`` travels with the mean on purpose: a run whose spread is large
    enough to invert a 5% change has not measured a 5% change.
    """
    return {
        "tok_per_s": round(float(row.get("avg_ts", 0.0)), 2),
        "stddev": round(float(row.get("stddev", 0.0)), 3),
        "samples": row.get("n_samples"),
        "model_filename": row.get("model_filename"),
        "model_type": row.get("model_type"),
        "model_size": row.get("model_size"),
        "n_gpu_layers": row.get("n_gpu_layers"),
    }


# --- live: the audio engines ----------------------------------------------


def _post(url: str, payload: dict, out: Path, timeout: int = 300,
          multipart: bool = False) -> dict:
    """One HTTP request, written with curl (stdlib-only, as this script is).

    Returns the parsed JSON body when there is one; ``timing`` in it is
    audio.cpp's own account of the work, which is the number worth keeping.
    """
    argv = ["curl", "-s", "--max-time", str(timeout), "-o", str(out),
            "-w", "%{http_code}"]
    if multipart:
        argv += ["-X", "POST", url]
    else:
        argv += ["-X", "POST", url, "-H", "Content-Type: application/json",
                 "-d", json.dumps(payload)]
    try:
        result = subprocess.run(argv, capture_output=True, text=True,
                                timeout=timeout + 15, check=False)
    except subprocess.TimeoutExpired as exc:
        raise BenchError(f"{url} timed out after {timeout}s") from exc
    code = result.stdout.strip()[-3:]
    if code != "200":
        body = out.read_text(errors="replace")[:200] if out.is_file() else ""
        raise BenchError(f"{url} returned HTTP {code}: {body}")
    body: dict = {}
    if out.is_file() and out.suffix == ".json":
        try:
            body = json.loads(out.read_text(errors="replace"))
        except json.JSONDecodeError:
            body = {}
    return {"http": code, "body": body,
            "bytes": out.stat().st_size if out.is_file() else 0}


def bench_audio_live(base_url: str, models: list[str], tmp: Path,
                     device: str = "card1") -> list[dict]:
    """Real-time factor for TTS and ASR through the running service.

    Uses the *requested model id*, which is the thing that has to work end to
    end: it is the string llama-swap routes on and the one audio.cpp
    validates, so this also fails loudly if those two ever disagree again.
    """
    records: list[dict] = []
    for model in models:
        if model.endswith(".a"):
            # ASR needs audio to transcribe; synthesize a short clip first if
            # one is not supplied, so a run needs no external fixture.
            clip = tmp / f"{model}-clip.wav"
            synth = {"model": models[0] if models[0].endswith(".s") else model,
                     "input": "one two three, four five six.",
                     "voice": "af_heart", "response_format": "wav"}
            out = tmp / f"{model}-synth.wav"
            _post(f"{base_url}/v1/audio/speech", synth, out)
            shutil_path = shutil_copy(clip, out)
            payload = None
            argv = ["curl", "-s", "--max-time", "300", "-o",
                    str(tmp / f"{model}-asr.json"), "-w", "%{http_code}",
                    "-X", "POST", f"{base_url}/v1/audio/transcriptions",
                    "-F", f"file=@{shutil_path}", "-F", f"model={model}"]
            result = subprocess.run(argv, capture_output=True, text=True,
                                    timeout=315, check=False)
            if result.stdout.strip()[-3:] != "200":
                raise BenchError(f"ASR {model} failed: HTTP "
                                 f"{result.stdout.strip()[-3:]}")
            body = json.loads((tmp / f"{model}-asr.json").read_text())
            timing = body.get("timing", {})
            records.append({
                "metric": "asr_rtf", "model": model,
                "rtf": timing.get("rtf"),
                "wall_ms": timing.get("wall_ms"),
                "audio_duration_ms": timing.get("audio_duration_ms"),
                "text": (body.get("text") or "")[:80],
            })
        else:
            out = tmp / f"{model}-tts.wav"
            payload = {"model": model, "input": "The quick brown fox jumps "
                                                 "over the lazy dog.",
                       "voice": "af_heart", "response_format": "wav"}
            result = _post(f"{base_url}/v1/audio/speech", payload, out)
            timing = (result["body"] or {}).get("timing", {})
            records.append({
                "metric": "tts_rtf", "model": model,
                "rtf": timing.get("rtf"),
                "wall_ms": timing.get("wall_ms"),
                "audio_duration_ms": timing.get("audio_duration_ms"),
                "wav_bytes": result["bytes"],
            })
        records[-1].setdefault("vram_peak_mib", vram_mib(device))
    return records


def shutil_copy(dest: Path, src: Path) -> Path:
    import shutil
    shutil.copyfile(src, dest)
    return dest


# --- report ----------------------------------------------------------------


def metric_unit(metric: str) -> tuple[str, str]:
    """``(value_field, unit)`` for a metric name.

    Matching on the exact name rather than the shape of it is how a
    ``prompt_tok_s`` record ends up printed as "None RTF" — the label has to
    follow the metric, not a hardcoded string that used to be right.
    """
    if metric.endswith("_rtf"):
        return "rtf", "RTF"
    if metric.endswith("_tok_s"):
        return "tok_per_s", "tok/s"
    return "tok_per_s", metric or "?"


def compare_and_print(record: dict, history: list[dict]) -> None:
    """Print the measurement and, when possible, the last one's delta."""
    metric = record.get("metric", "")
    field, label = metric_unit(metric)
    value = record.get(field)
    where = f"{record['engine']} {record.get('version')} " \
            f"{record.get('backend')} {record.get('model', '')}".strip()
    flags = ""
    if record.get("batch"):
        flags = f" [b{record['batch']}/ub{record.get('ubatch')}]"
    print(f"{where}{flags} {metric}: {value} {label}")
    old = previous_for(record, history)
    if old is None:
        print("  (first recorded run for this configuration)")
        return
    old_value = old.get(field)
    if old_value in (None, 0) or value is None:
        return
    change = delta_pct(float(value), float(old_value))
    print(f"  previous: {old_value} {label} on {old.get('version')} "
          f"({time.strftime('%Y-%m-%d %H:%M', time.localtime(old.get('ts', 0)))})"
          f" -> {change:+.1f}%")


def print_report(history: list[dict], limit: int = 40) -> None:
    if not history:
        print(f"no history yet at {history_path()}")
        return
    print(f"history: {history_path()}  ({len(history)} records)")
    print(f"{'when':<20} {'engine':<14} {'version':<10} {'backend':<8} "
          f"{'model':<22} {'metric':<10} value")
    for record in history[-limit:]:
        ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(record.get("ts", 0)))
        value = record.get("tok_per_s")
        if record.get("metric") in ("tts_rtf", "asr_rtf"):
            value = f"{record.get('rtf')} RTF"
        print(f"{ts:<20} {record.get('engine', '?'):<14} "
              f"{str(record.get('version', '?'))[:9]:<10} "
              f"{str(record.get('backend', '?'))[:7]:<8} "
              f"{str(record.get('model', '?'))[:21]:<22} "
              f"{str(record.get('metric', '?')):<10} {value}")


# --- CLI -------------------------------------------------------------------

DEFAULT_GRID = [(256, 256), (512, 512), (1024, 1024)]
QUICK_GRID = [(512, 512)]


def resolve_tools_dir(args: argparse.Namespace) -> Path:
    if args.dir:
        return Path(args.dir)
    root, per_engine = upd.load_engines_config()
    return upd.resolve_install_dirs(None, (root, per_engine))["cpp"][0]


def llama_binaries(tools: Path) -> list[Path]:
    """Every installed llama.cpp build, newest first.

    Old builds are kept deliberately (``update.py --keep-swap N``), so this is
    what makes "compare against the previous version" possible at all: after an
    update the old binary is still on disk and can still be measured.
    """
    found = [d / "llama-server" for d in tools.glob("llama-b[0-9]*")
             if (d / "llama-server").is_file()]
    found += [tools / "llama-server"] if (tools / "llama-server").is_file() else []
    return sorted(set(found),
                  key=lambda p: upd.cpp_build_number(
                      re.search(r"llama-b(\d+)", str(p)).group(1)
                      if re.search(r"llama-b(\d+)", str(p)) else "0") or 0,
                  reverse=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure serving engines and keep the results as history.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n\nUsage")[-1],
    )
    parser.add_argument("--engine", default="llama-server",
                        help="llama-server (isolated, via llama-bench) or "
                             "'none' to skip")
    parser.add_argument("--model", action="append", default=[],
                        help="GGUF to measure (repeatable); default: a small "
                             "chat model found in the served config")
    parser.add_argument("--versions", type=int, default=1,
                        help="how many installed llama.cpp builds to measure "
                             "(default 1 = newest; 2+ compares versions)")
    parser.add_argument("--quick", action="store_true",
                        help="one batch point instead of the grid, and the "
                             "smallest served model instead of the largest — "
                             "fast, but it measures a proxy, not the fleet")
    parser.add_argument("--smallest", action="store_true",
                        help="measure the smallest served model (a proxy for "
                             "build/backend A/B; ~10x faster than the default)")
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--gen-tokens", type=int, default=64)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--backend", default="",
                        help="only measure this backend (vulkan, cpu, hip…)")
    parser.add_argument("--live", action="store_true",
                        help="also measure the running llama-swap")
    parser.add_argument("--audio", action="store_true",
                        help="with --live: measure audio TTS/ASR real-time "
                             "factor")
    parser.add_argument("--base-url", default="http://127.0.0.1:8123")
    parser.add_argument("--config", default="/opt/llama/config.yaml")
    parser.add_argument("--dir", default="", help="tools directory")
    parser.add_argument("--device", default="card1")
    parser.add_argument("--report", action="store_true",
                        help="print the history and exit")
    args = parser.parse_args(argv)

    if args.report:
        print_report(load_history())
        return 0

    history = load_history()
    tools = resolve_tools_dir(args)
    written = 0
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / "llama-bench"
    tmp.mkdir(parents=True, exist_ok=True)

    if args.engine != "none":
        servers = llama_binaries(tools)[:args.versions]
        if not servers:
            print(f"Error: no llama.cpp build found in {tools}",
                  file=sys.stderr)
            return 1
        models = [Path(m) for m in args.model]
        if not models:
            mode = "smallest" if (args.quick or args.smallest) else "largest"
            models = default_models(args.config, mode)
            if models:
                print(f"Measuring {models[0].name} ({mode} served model, "
                      f"{models[0].stat().st_size / 1024 ** 3:.1f} GiB)")
                if mode == "smallest":
                    print("  (a proxy: small models exaggerate the value of "
                          "batch depth — the fleet's 27B shows none)")
        if not models:
            print("Error: no model to measure; pass --model <path>",
                  file=sys.stderr)
            return 1
        grid = QUICK_GRID if args.quick else DEFAULT_GRID
        for server in servers:
            version = engine_version(server)
            devices = detect_devices(server)
            if not devices:
                devices = ["unknown: none reported"]
            for device_line in devices:
                backend = backend_of(device_line)
                if args.backend and backend != args.backend:
                    continue
                bench = server.parent / "llama-bench"
                if not bench.is_file():
                    print(f"Error: no llama-bench beside {server}",
                          file=sys.stderr)
                    return 1
                for model in models:
                    for batch, ubatch in grid:
                        with VramSampler(args.device) as sampler:
                            rows = llama_bench_rows(
                                bench, model, batch, ubatch,
                                args.prompt_tokens, args.gen_tokens,
                                args.reps)
                        for metric, n_p, n_g in (
                                ("prompt_tok_s", args.prompt_tokens, 0),
                                ("gen_tok_s", 0, args.gen_tokens)):
                            # No "skip zero" guard here: llama-bench *is* how a
                            # skipped case is reported — a prompt-only test
                            # comes back with n_gen 0 and a generation-only
                            # test with n_prompt 0. Filtering on the zeros
                            # discarded both rows and reported nothing
                            # measured, which is worse than an error.
                            try:
                                row = pick_row(rows, n_p, n_g)
                            except BenchError as exc:
                                print(f"  skipping {metric}: {exc}")
                                continue
                            record = {
                                "ts": int(time.time()),
                                "engine": "llama-server",
                                "version": version,
                                "backend": backend,
                                "device": device_line,
                                "model": model.name,
                                "batch": batch, "ubatch": ubatch,
                                "metric": metric,
                                "vram_peak_mib": sampler.peak,
                                "host": platform.node(),
                                "kernel": platform.release(),
                            }
                            record.update(record_from_row(row))
                            record["metric"] = metric
                            compare_and_print(record, history)
                            append_history(record)
                            history.append(record)
                            written += 1

    if args.live and args.audio:
        models = audio_models(args.config)
        if not models:
            print("Error: no audio entries found in " + args.config,
                  file=sys.stderr)
            return 1
        audio_version = engine_version(tools / "audiocpp_server")
        for record in bench_audio_live(args.base_url, models, tmp, args.device):
            record.update({
                "ts": int(time.time()), "engine": "audiocpp_server",
                "version": audio_version,
                "backend": audio_backend(args.config),
                "device": "n/a", "batch": None, "ubatch": None,
                "host": platform.node(), "kernel": platform.release(),
            })
            compare_and_print(record, history)
            append_history(record)
            history.append(record)
            written += 1

    if not written:
        print("nothing measured (check --engine/--model/--backend)")
        return 1
    print(f"\n{written} record(s) appended to {history_path()}")
    return 0


def config_models(config: str) -> list[tuple[Path, float, str]]:
    """``(path, size_gib, entry_id)`` for every model the config serves.

    Paths come out of the served commands rather than a model directory
    scan: the config is what the fleet actually runs, including the
    ``${HF_HOME}`` / ``${MODELS_DIR}`` indirections that a scan would miss.
    """
    try:
        import yaml
        data = yaml.safe_load(Path(config).read_text())
    except Exception:
        return []
    subs = {"HF_HOME": os.environ.get("HF_HOME", ""),
            "MODELS_DIR": os.environ.get("MODELS_DIR", "/mnt/ai/models")}
    out: list[tuple[Path, float, str]] = []
    for entry_id, entry in (data.get("models") or {}).items():
        cmd = entry.get("cmd", "") or ""
        match = re.search(r"-m (\S+)", cmd)
        if not match:
            continue
        raw = match.group(1)
        for name, value in subs.items():
            raw = raw.replace(f"${{{name}}}", value)
        path = Path(raw)
        try:
            size = path.stat().st_size / 1024 ** 3
        except OSError:
            continue
        out.append((path, size, entry_id))
    return out


def default_models(config: str, mode: str = "largest",
                   cap_gib: float = 24.0) -> list[Path]:
    """What to measure when nobody named a model.

    ``largest`` (the default) is the biggest model the fleet serves that fits
    under *cap_gib* — on this box the 27B the operator actually runs for real
    work. The earlier default took the *first* small model in the config,
    which meant a 0.6B embedder, and measurements on it do not survive
    contact with the fleet: a 4B shows batch depth worth 2x, and the 27B
    shows batch depth worth nothing, because decode there is bound by
    streaming weights rather than by batch parallelism. A benchmark whose
    default answers a different question than the fleet is worse than no
    default, so it names the model it picked and why.
    """
    models = config_models(config)
    if not models:
        return []
    fits = [m for m in models if m[1] <= cap_gib]
    if not fits:
        fits = models
    if mode == "smallest":
        chosen = [min(fits, key=lambda m: m[1])]
    else:
        chosen = [max(fits, key=lambda m: m[1])]
    return [path for path, _size, _eid in chosen]


def audio_models(config: str) -> list[str]:
    try:
        import yaml
        data = yaml.safe_load(Path(config).read_text())
    except Exception:
        return []
    return [eid for eid in (data.get("models") or {})
            if eid.endswith((".s", ".a"))]


def audio_backend(config: str) -> str:
    try:
        import yaml
        data = yaml.safe_load(Path(config).read_text())
        for entry in (data.get("models") or {}).values():
            match = re.search(r'"backend":"(\w+)"', entry.get("cmd", "") or "")
            if match:
                return match.group(1)
    except Exception:
        pass
    return "unknown"


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (BenchError, upd.ApiError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)