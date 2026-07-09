#!/usr/bin/env python3
"""llama-server launcher for RDNA4 (gfx1201). Supports ROCm and Vulkan backends."""

import argparse
import fnmatch
import os
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# --- Colors ---
_BOLD = "\033[1m"
_BLUE = "\033[34m"
_GREEN = "\033[32m"
_RESET = "\033[0m"

def _label(text: str) -> str:
    return f"{_BOLD}{_GREEN}{text}{_RESET}"

def _banner(text: str) -> str:
    return f"{_BOLD}{_BLUE}{text}{_RESET}"


# --- Helpers ---
def get_available_versions() -> list[int]:
    return sorted(
        int(d.name.removeprefix("llama-b"))
        for d in SCRIPT_DIR.glob("llama-b[0-9]*")
        if d.is_dir()
    )


def find_bin_dir(version: str, /) -> str:
    bin_dir = os.environ.get("LLAMA_BIN_DIR")
    if bin_dir:
        return bin_dir

    if version == "latest":
        versions = get_available_versions()
        if not versions:
            print("Error: no llama-b#### directory found", file=sys.stderr)
            sys.exit(1)
        return f"llama-b{versions[-1]}"

    if (SCRIPT_DIR / f"llama-b{version}").is_dir():
        return f"llama-b{version}"

    print(f"Error: version {version} not found", file=sys.stderr)
    print(f"Available versions: {' '.join(str(v) for v in get_available_versions())}", file=sys.stderr)
    sys.exit(1)


def parse_quantifier(value) -> int:
    """Parse numeric values with quantifiers: k=1024, m=1024², g=1024³.
    Append 'i' for decimal (1000-based), 'b' for bytes (no-op)."""
    if isinstance(value, (int, float)):
        return int(value)

    value = str(value).strip().lower()
    if not value:
        return 0

    mult = 1024
    while value and value[-1] in "ib":
        if value[-1] == "i":
            mult = 1000
        value = value[:-1]

    suffixes = {"g": 3, "m": 2, "k": 1}
    if value and value[-1] in suffixes:
        try:
            return int(value[:-1]) * mult ** suffixes[value[-1]]
        except ValueError:
            raise

    if value.isdigit():
        return int(value)

    raise ValueError(f"Cannot parse quantifier: {value!r}")


def select_env_file(env_arg: str | None, /) -> str:
    if os.environ.get("LLAMA_ENV_FILE"):
        return os.environ["LLAMA_ENV_FILE"]

    if env_arg:
        pat = env_arg.replace("-", "X").replace("_", "X").replace("X", "[-_]")
        matches = [str(f) for f in SCRIPT_DIR.glob("*.env") if fnmatch.fnmatch(f.name, pat)]

        if not matches:
            print(f"Error: no .env file matching '{env_arg}' found", file=sys.stderr)
            sys.exit(1)
        if len(matches) > 1:
            print(f"Error: '{env_arg}' matches multiple .env files:", file=sys.stderr)
            for m in matches:
                print(f"  - {m}", file=sys.stderr)
            sys.exit(1)
        return matches[0]

    symlink = SCRIPT_DIR / "llama-server.env"
    if symlink.is_symlink():
        return str(symlink)

    return str(SCRIPT_DIR / "qwen3.6_27B.env")


def load_env_file(env_file: str, /) -> dict[str, str]:
    path = Path(env_file)
    if not path.is_absolute():
        path = SCRIPT_DIR / env_file

    if not path.is_file():
        return {}

    env = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            env[key] = value.strip('"\'')
    return env


def calculate_ctx_size(model_size_mb: int, vram_total_mb: int, /) -> int:
    available = vram_total_mb - 2048
    remaining = available - model_size_mb

    if remaining <= 0:
        print(f"Error: Model ({model_size_mb} MB) exceeds available VRAM ({available} MB)", file=sys.stderr)
        print(f"Need at least {model_size_mb + 2048} MB VRAM", file=sys.stderr)
        sys.exit(1)

    max_tokens = (remaining * 1024) // 20
    ctx = (max_tokens // 8192) * 8192
    return max(ctx, 4096)


def get_model_size_mb(model_path: str, /) -> int:
    path = Path(model_path)
    if not path.is_absolute():
        path = SCRIPT_DIR / model_path
    return path.stat().st_size // (1024 ** 2)


def get_vram_mb() -> int:
    try:
        out = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=10)
        m = re.search(r'Pool 1.*?Size:\s*(\d+)', out.stdout, re.DOTALL)
        if m:
            vram = int(m.group(1))
            if vram > 0:
                return vram
    except Exception:
        pass
    return 32768


def detect_mmproj(model_path: str, text_mode: bool, env: dict[str, str], /) -> str | None:
    if text_mode:
        return None

    mmproj = env.get("MMPROJ", os.environ.get("MMPROJ", ""))
    if mmproj.lower() in ("0", "false", "none", "off"):
        return None
    if mmproj.lower() in ("1", "true", "on", "auto", ""):
        pass  # fall through to auto-detect
    elif Path(mmproj).is_file():
        return str(mmproj)

    model_dir = Path(model_path).parent
    if not model_dir.is_absolute():
        model_dir = SCRIPT_DIR / model_dir

    model_family = re.sub(r'\W[iI]?Q\d[_KMSXL]*$', '', Path(model_path).stem)

    for f in model_dir.glob("*mmproj*.gguf"):
        if model_family in f.name:
            return str(f)

    return None


# --- Config display ---
def print_config(cfg: dict) -> None:
    lines = [
        ("Binary:", cfg["bin_dir"]),
        ("Model:", f'{cfg["model"]} ({cfg["model_size_mb"]}MB)'),
        ("VRAM:", f'{cfg["vram_total_mb"]}MB total, ~2GB reserved for KV'),
        ("Context:", f'{cfg["ctx_size"]} tokens'),
        ("Cache:", f'{cfg["cache_ram"]}MB RAM, {cfg["cache_timeout"]}s timeout'),
        ("Sampling:", f'temp={cfg["temp"]} top_p={cfg["top_p"]} top_k={cfg["top_k"]} min_p={cfg["min_p"]} pres_pen={cfg["pres_pen"]}'),
    ]
    if cfg.get("cache_type"):
        lines.append(("CacheTypeK:", cfg["cache_type"]))
    if cfg.get("mmproj"):
        lines.append(("MMProj:", cfg["mmproj"]))

    print(f"\n{_banner('=== llama-server configuration ===')}")
    for label, value in lines:
        print(f"{_label(label)} {value}")
    print(f"{_banner('===================================')}\n")


# --- Argument parsing ---
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start llama-server with automatic configuration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "ENV_FILE_PATTERN:\n"
            "  Optional pattern to match .env files. Supports wildcards (?).\n"
            "  If not provided, uses LLAMA_ENV_FILE env var, llama-server.env symlink,\n"
            "  or defaults to qwen3.6_27B.env\n"
            "\n"
            "Environment Variables:\n"
            "  LLAMA_BIN_DIR           Override binary directory\n"
            "  LLAMA_ENV_FILE          Override .env file\n"
            "  LLAMA_MODEL             Model file path\n"
            "  LLAMA_CACHE             RAM cache size in MB\n"
            "  LLAMA_CACHE_TIMEOUT     Cache timeout in seconds\n"
            "  LLAMA_CACHE_TYPE        KV cache type (f16, q8_0, q4_0)\n"
            "  LLAMA_TEMP              Temperature\n"
            "  LLAMA_TOP_P             Top-p sampling\n"
            "  LLAMA_TOP_K             Top-k sampling\n"
            "  LLAMA_MIN_P             Min-p sampling\n"
            "  LLAMA_PRES_PEN          Presence penalty\n"
            "  LLAMA_HOST              Bind host\n"
            "  LLAMA_PORT              Bind port\n"
            "  LLAMA_PARALLEL          Parallel requests\n"
            "  LLAMA_CONTEXT           Override context size\n"
            "  MMPROJ                  Multimodal projector file\n"
        ),
    )
    parser.add_argument("-v", "--version", default="latest", help="llama version (e.g., 8929) or 'latest'")
    parser.add_argument("-t", "--text", action="store_true", help="Text-only mode (no multimodal)")
    parser.add_argument("env_pattern", nargs="?", default=None, help=".env file pattern")

    return parser.parse_args(argv[1:] if argv else None)


# --- Main ---
def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    bin_dir = find_bin_dir(args.version)
    env = load_env_file(select_env_file(args.env_pattern))

    cfg = {
        "bin_dir": bin_dir,
        "model": env.get("LLAMA_MODEL", "models/Qwen3.6-27B-Q6_K.gguf"),
        "cache_ram": parse_quantifier(env.get("LLAMA_CACHE", 4096)),
        "cache_timeout": parse_quantifier(env.get("LLAMA_CACHE_TIMEOUT", 8 * 3600)),
        "cache_type": env.get("LLAMA_CACHE_TYPE"),
        "temp": float(env.get("LLAMA_TEMP", 0.6)),
        "top_p": float(env.get("LLAMA_TOP_P", 0.95)),
        "top_k": parse_quantifier(env.get("LLAMA_TOP_K", 20)),
        "min_p": float(env.get("LLAMA_MIN_P", 0.0)),
        "pres_pen": float(env.get("LLAMA_PRES_PEN", 0.01)),
        "host": env.get("LLAMA_HOST", "0.0.0.0"),
        "port": parse_quantifier(env.get("LLAMA_PORT", 8000)),
        "parallel": parse_quantifier(env.get("LLAMA_PARALLEL", 1)),
    }

    cfg["model_size_mb"] = get_model_size_mb(cfg["model"])
    cfg["vram_total_mb"] = get_vram_mb()
    cfg["ctx_size"] = parse_quantifier(env["LLAMA_CONTEXT"]) if env.get("LLAMA_CONTEXT") else calculate_ctx_size(cfg["model_size_mb"], cfg["vram_total_mb"])
    cfg["mmproj"] = detect_mmproj(cfg["model"], args.text, env)

    print_config(cfg)

    os.environ["GGML_HIP_GRAPHS"] = "0"

    cmd = [
        str(SCRIPT_DIR / cfg["bin_dir"] / "llama-server"),
        "--alias", "local",
        "--cache-ram", str(cfg["cache_ram"]),
        "--timeout", str(cfg["cache_timeout"]),
        "--parallel", str(cfg["parallel"]),
        "--ctx-size", str(cfg["ctx_size"]),
        "--flash-attn", "on",
        "--no-webui",
        "--host", cfg["host"],
        "--port", str(cfg["port"]),
        "--jinja",
        "--cont-batching",
        "--reasoning", "auto",
        "--temp", str(cfg["temp"]),
        "--top-p", str(cfg["top_p"]),
        "--top-k", str(cfg["top_k"]),
        "--min-p", str(cfg["min_p"]),
        "--presence-penalty", str(cfg["pres_pen"]),
        "--model", cfg["model"],
        "--no-mmap",
    ]

    if cfg["cache_type"]:
        cmd.extend(["--cache-type-k", cfg["cache_type"]])
        cmd.extend(["--cache-type-v", cfg["cache_type"]])
    if cfg["mmproj"]:
        cmd.extend(["--mmproj", cfg["mmproj"]])

    os.execv(cmd[0], cmd)


if __name__ == "__main__":
    main()
