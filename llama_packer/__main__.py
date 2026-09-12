# llama_packer/__main__.py
"""CLI entry point for llama-packer."""

from __future__ import annotations

import argparse
import importlib.resources
import logging
import os
import re
import shlex
import shutil
import sys
import textwrap
from typing import NoReturn
from pathlib import Path

import yaml

from llama_packer import Model, __version__, find_bin_dir
from llama_packer.hardware import GpuProfile, detect_gpu_vendor
from llama_packer.profiles import Profiles
from llama_packer.scope import ScopeStack
from llama_packer.discover import discover
from llama_packer.utils import (
    compute_env_prefixes, make_subst, make_fast_storage_predicate,
    _detect_drive_speed, validate_dir_roles, NON_CHAT_ROLES,
)
from llama_packer.consts import (
    _MIN_AGENTIC_CTX, _MEMORY_MARGIN, _RESERVE_SYSTEM, _RESERVE_VIDEO,
    VLLM_DEFAULT_IMAGE, VLLM_DEFAULT_BIN, VLLM_DEFAULT_DOCKER_ARGS,
    VLLM_DEFAULT_CONTAINER_PORT, VLLM_DEFAULT_GPU_MEM_UTIL,
)
from llama_packer.writer import build_config, write_yaml, EmittedConfig
from llama_packer.progress import PackerProgress
from llama_packer.backends import (SD_BACKENDS, VLLM_BACKENDS,
                                   validate_backend_names)
from llama_packer.backends.audio_cpp import (AUDIO_CPP_DEFAULT_BIN,
                                             AUDIO_CPP_SERVER_KNOBS)



logger = logging.getLogger(__name__)


def setup_logging(verbosity: int = 0) -> None:
    # Each -v drops LOGLEVEL by 10 from the WARNING default (-v: INFO,
    # -vv: DEBUG, -vvv: 0/NOTSET); never negative.
    level = max(0, logging.WARNING - 10 * max(0, verbosity))
    logging.basicConfig(
        level=level,
        format="%(levelname).1s | %(message)s",
        stream=sys.stderr,
        force=True,
    )


def fatal(msg: str, *args) -> "NoReturn":
    """Log a CRITICAL complaint, then abort — for unrecoverable config errors."""
    logger.critical(msg, *args)
    sys.exit(1)


def backend_args(cfg: dict | None, section: str) -> str:
    """Validate and return a backend section's free-form ``args`` string.

    Performance/tuning flags shared by every command that backend renders
    (e.g. ``llama_server: {args: "--flash-attn on -b 512"}``).  Parsed with
    shlex here so quoting errors fail fast at build time, not per command.
    """
    value = (cfg or {}).get("args")
    if value is None or (isinstance(value, str) and not value.strip()):
        return ""
    if not isinstance(value, str):
        fatal("profiles.yaml %s.args: must be a string of flags, got %s",
              section, type(value).__name__)
    raw = value.strip()
    if raw:
        try:
            shlex.split(raw)
        except ValueError as e:
            fatal("profiles.yaml %s.args: bad quoting: %s", section, e)
        if raw.startswith("[") or raw.endswith("]"):
            logger.warning(
                "profiles.yaml %s.args: looks like a stringified container "
                "(starts with '[' / ends with ']'); emitting tokens verbatim. "
                "Did you mean a quoted flag string?", section)
    return raw


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate llama-swap config.yaml from model metadata and profiles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              llama-packer                              # defaults
              llama-packer --dry-run                    # preview
              llama-packer --llama-version 8929         # specific llama-server version
              llama-packer --llama-server /opt/lsrv     # explicit binary path
              llama-packer --output /etc/ls/config.yaml
        """),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print config to stdout instead of writing")
    parser.add_argument("--remeasure", action="store_true",
                        help="Ignore saved VRAM measurements (sidecar `derived:` "
                             "blocks) and re-run the fast fit-params trio for "
                             "every model — never a server, seconds per model")
    parser.add_argument("--probe-memory", nargs="*", metavar="ARCH",
                        help="Measure serve-shaped llama-server VRAM over a "
                             "(pool ctx x parallel) grid for one model per "
                             "GGUF arch family (optionally restricted to the "
                             "named families), least-squares fit the affine "
                             "law and report the max residual. Starts a real "
                             "server (opt-in calibration, never a pack "
                             "dependency); discovery may persist `file:` "
                             "blocks to sidecars. Then exits")
    parser.add_argument("--output", default="config.yaml", help="Output path (default: config.yaml)")
    parser.add_argument("--llama-version", default="latest", dest="version",
                        help="llama-server version (default: latest)")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}",
                        help="Show llama-packer version and exit")
    parser.add_argument("--llama-server", help="Explicit path to llama-server binary")
    parser.add_argument("--models-dir", nargs="+", default=None,
                        help="Model directories (default: profiles.yaml models_dirs, "
                             "else ./models); pass multiple to scan several")
    parser.add_argument("--hf-home", help="HF_HOME root (the dir containing hub/) for hub snapshot "
                        "resolution and path grouping (overrides profiles.yaml hf_home / $HF_HOME)")
    parser.add_argument("--profiles", default="profiles.yaml", help="Profiles file (default: profiles.yaml)")
    parser.add_argument("--no-stubs", action="store_true", help="Skip generating stub .md files")
    parser.add_argument("--agents", action="store_true",
                        help="Write the AGENTS.md model guide to the models directory if missing "
                             "(never overwrites an existing file; bundled template)")
    parser.add_argument("--extra-dirs", nargs="*", default=["embed", "rerank"], help="Extra subdirectories of --models-dir to scan for orphan GGUFs (default: embed rerank)")
    parser.add_argument("--spare", help="Additional VRAM to reserve on TOP of the fixed 2048 MiB system+driver reserve "
                                         "(1024 MiB OS/driver + 1024 MiB video framebuffer): suffixed (2G, 512m, 64k) or bare number "
                                         "(auto: GB if < 3×VRAM, else MB)")
    parser.add_argument("--vram", help="Total GPU VRAM: suffixed (32G, 24576m) or bare MB (overrides auto-detection). "
                                         "The fixed 2048 MiB system+video reserve plus --spare are subtracted before budgeting context.")
    parser.add_argument("--baseline", help="Driver/compositor VRAM already in use, added to the fixed 2048 MiB reserve "
                                          "(default: 0; live auto-detection is disabled so the packer's own resident model "
                                          "servers are not counted against the budget). Set only when other non-model processes occupy VRAM.")
    parser.add_argument("--unified-system-mb", help="System memory reserved for the OS on unified-memory hosts "
                                          "(GB10/DGX Spark, Apple Silicon): suffixed (8G, 4096m) or bare MB. "
                                          "Default: 8192 (8 GiB). Only applies to auto-detected unified pools; "
                                          "explicit --vram/hardware.vram overrides the whole budget instead.")
    parser.add_argument("--gpu-family", help="GPU family for chip-specific calculation rules (default: auto-detect or profiles.yaml hardware.gpu_family)")
    parser.add_argument("--max-context", help="Hard cap on context length for all models (e.g. 128k, 65536)")
    parser.add_argument("--min-context", help="Minimum useful context for chat models; mmproj (vision) is skipped when needed to reach it (default: 131072 = 128k)")
    parser.add_argument("--no-env", action="store_true", help="Do not write the sibling config.env file")
    parser.add_argument("--health-check-timeout", type=int, default=None,
                        help="Health check timeout in seconds (default: auto-calculated from model sizes)")
    parser.add_argument("--drive-speed", type=int, default=None,
                        help="Slowest drive speed in MB/s for timeout calc (default: auto-detect, else 100)")
    parser.add_argument("--verbose", "-v", "-V", action="count", default=0,
                        help="Increase verbosity, repeatable (-v: INFO, -vv: DEBUG, -vvv: all)")
    parser.add_argument("--embed", help="Substring selector for the embedder; else smallest embed-type model")
    parser.add_argument("--rerank", help="Substring selector for the reranker; else smallest rerank-type model")
    parser.add_argument("--vllm-image", help="vLLM docker image for `vllm-docker` backend models "
                         "(overrides profiles.yaml vllm.image)")
    parser.add_argument("--vllm-server", help="vLLM binary for `vllm` backend models "
                         "(overrides profiles.yaml vllm.bin; default: vllm on PATH)")
    parser.add_argument("--sd-server", help="sd-server binary for `sd-server` backend "
                         "(overrides profiles.yaml sd.bin / $SD_BIN_DIR / sd-server on PATH)")
    parser.add_argument("--whisper-server", help="whisper-server binary for `whisper-server` backend "
                        "(overrides profiles.yaml whisper.bin / $WHISPER_BIN_DIR / whisper-server on PATH)")
    parser.add_argument("--audio-cpp-server", help="audiocpp_server binary for `audio-cpp` backend "
                        "(overrides profiles.yaml audio_cpp.bin / $AUDIOCPP_BIN_DIR / audiocpp_server on PATH)")
    parser.add_argument("--no-macros", action="store_true",
                         help="Disable flag macros (emit fully expanded cmds)")
    return parser.parse_args(argv[1:] if argv else None)


def write_agents_md(models_dir: Path) -> None:
    """Write the bundled AGENTS.md guide into ``models_dir`` if missing.

    The canonical guide ships inside the package
    (``llama_packer/templates/models_AGENTS.md``) so it travels with the tool.
    It is copied only when ``models/AGENTS.md`` does not already exist — user
    edits are never overwritten.  A failure to write is logged and ignored so
    config generation can continue.
    """
    dest = models_dir / "AGENTS.md"
    if dest.exists():
        logger.info("AGENTS.md exists, keeping: %s", dest)
        return
    try:
        src = importlib.resources.files("llama_packer").joinpath("templates", "models_AGENTS.md")
        if not src.is_file():
            logger.warning("bundled AGENTS.md template missing: %s", src)
            return
        shutil.copyfile(str(src), dest)
        try:
            os.chmod(dest, 0o644)
        except OSError:
            pass
        logger.info("wrote AGENTS.md guide: %s", dest)
    except Exception as e:
        logger.warning("could not write AGENTS.md (%s), continuing", e)


def _apply_env_subst(config: EmittedConfig, sub, raw_paths: list[str]) -> EmittedConfig:
    """Replace each raw emitted path in the generated cmds with its ${VAR} macro form."""
    subs = {raw: sub(raw) for raw in raw_paths}
    order = sorted(subs, key=len, reverse=True)
    for entry in config.get("models", {}).values():
        cmd = entry.get("cmd")
        if not cmd:
            continue
        for raw in order:
            if raw in cmd:
                cmd = cmd.replace(raw, subs[raw])
        entry["cmd"] = cmd
    # Also rewrite flag-macro definitions (they may contain absolute paths).
    for k, v in list(config.get("macros", {}).items()):
        if isinstance(v, str):
            nv = v
            for raw in order:
                if raw in nv:
                    nv = nv.replace(raw, subs[raw])
            config["macros"][k] = nv
    return config


def _select_model(models: list, type_name: str, selector: str | None, logger) -> "Model | None":
    """Pick a model of `type_name` (role).

    With no selector, returns the smallest by VRAM footprint. With a selector
    substring, returns the single model whose id/name/stem contains it; errors
    if the match is not exactly one.
    """
    cands = [m for m in models if m.role == type_name]
    if not cands:
        return None
    if selector:
        hits = [
            m for m in cands
            if (selector in m.stem
                or selector in str(m.frontmatter.get("name", ""))
                or selector in m.template_id)
        ]
        if len(hits) != 1:
            fatal("selector %r matched %d %s models (need exactly 1): %s",
                  selector, len(hits), type_name,
                  [h.stem for h in hits])
        return hits[0]
    return min(cands, key=lambda m: m.vram_mb)


def _detect_matrix(profiles_cfg: dict, models: list, args, logger) -> tuple:
    """Resolve the swap-matrix configuration before build_config.

    Returns ``(matrix_cfg, embed_model, rerank_model, categories, fixed)``.

    ``categories`` maps each declared category name → its selected model
    (defaults: ``emb``/``rnk`` bound to the embeddings/rerank models).  The
    RAG pair still drives the shared chat-context solve; every *other*
    category (e.g. ``tts``/``stt``) is returned in ``fixed`` as a
    fixed-overhead resident reserved alongside chat and RAG.

    Without a ``matrix:`` section the matrix is disabled — make that state
    visible when the fleet actually has RAG models, otherwise a silently
    missing co-loading setup looks exactly like a bug (it has, repeatedly).
    """
    matrix_cfg = profiles_cfg.get("matrix")
    if not matrix_cfg:
        emb = _select_model(models, "embeddings", args.embed, logger)
        rnk = _select_model(models, "rerank", args.rerank, logger)
        if emb is not None or rnk is not None:
            logger.warning(
                "matrix: disabled — profiles.yaml has no matrix: section; "
                "RAG co-loading off (emb: %s, rnk: %s)",
                emb.stem if emb else "none", rnk.stem if rnk else "none")
        return None, None, None, {}, ()
    embed_model = _select_model(models, "embeddings", args.embed, logger)
    rerank_model = _select_model(models, "rerank", args.rerank, logger)
    if embed_model is None:
        logger.warning("no embeddings model found; skipping matrix")
        return None, None, None, {}, ()
    if rerank_model is None:
        logger.warning("no rerank model found; skipping matrix")
        return None, None, None, {}, ()

    # Declared categories (defaults bind the RAG pair).  Category names are
    # the matrix var names; roles may repeat (e.g. tts/stt on t2s/s2t).
    cat_specs = matrix_cfg.get("categories") or {
        "emb": {"role": "embeddings"}, "rnk": {"role": "rerank"}}
    categories: dict[str, "Model"] = {}
    for name, spec in cat_specs.items():
        spec = spec if isinstance(spec, dict) else {}
        role = str(spec.get("role") or "")
        selector = spec.get("selector")
        if not role:
            logger.warning("matrix: category %r has no role; omitted", name)
            continue
        model = _select_model(models, role, selector, logger)
        if model is None:
            logger.warning("matrix: category %r: no %s model%s; omitted",
                           name, role,
                           f" matching {selector!r}" if selector else "")
            continue
        categories[str(name)] = model
    # Back-compat: the canonical RAG names always resolve.
    categories.setdefault("emb", embed_model)
    categories.setdefault("rnk", rerank_model)

    known = set(categories)
    for key in (matrix_cfg.get("evict_costs") or {}):
        if key not in known:
            logger.warning("matrix: evict_costs key %r is not a declared "
                           "category %s; llama-swap will ignore it",
                           key, sorted(known))

    fixed = [(n, m) for n, m in categories.items()
             if m is not embed_model and m is not rerank_model]
    logger.info("matrix embed: %s", embed_model.stem)
    logger.info("matrix rerank: %s", rerank_model.stem)
    if fixed:
        logger.info("matrix categories: %s",
                    ", ".join(f"{n}={m.stem}" for n, m in fixed))
    return matrix_cfg, embed_model, rerank_model, categories, fixed


# Var-name prefix per co-load role, used in set expressions
# (e.g. ``__CHAT_VARS__ & emb & rnk & __COLOAD_VARS__``).
_COLOAD_VAR_PREFIX = {"s2t": "s2t", "image": "img", "t2s": "t2s"}


def _build_matrix_vars(models: list, embed_model, rerank_model, categories,
                       coload_stems: list[str],
                       entry_ids_by_stem: dict[str, list[str]], logger) -> tuple[dict, list[str]]:
    """Auto-collect matrix vars: chat entries + selected embed/rerank + co-loads.

    Each chat model contributes a var per emitted entry id (the bare id and,
    when present, its text-only variant — see writer.TEXT_SUFFIX), so text
    variants join the same co-loading sets as their parent entry.  The
    stem → ids mapping comes from the emitter (EmittedConfig) so naming is
    owned in exactly one place.  Opportunistically included non-chat models
    (``coload_stems``, from the matrix solve) contribute one role-prefixed
    var each (``s2t``, ``img``; numbered on collision).

    Returns (vars, coload_var_names) — the latter feeds the
    ``__COLOAD_VARS__`` placeholder in set expressions.
    """
    vars_: dict[str, str] = {}
    chat_idx = 0
    for m in models:
        if m.role in NON_CHAT_ROLES:
            continue
        for eid in entry_ids_by_stem.get(m.stem, []):
            chat_idx += 1
            vars_[f"c{chat_idx}"] = eid
    # Declared categories contribute one var each (emb, rnk, tts, stt, …);
    # the category names are what `sets:` expressions reference.
    for name, model in (categories or {}).items():
        if name in vars_:
            logger.warning("matrix: category var %r collides with an existing "
                           "var; skipped", name)
            continue
        vars_[name] = model.template_id
    if embed_model is not None:
        vars_.setdefault("emb", embed_model.template_id)
    if rerank_model is not None:
        vars_.setdefault("rnk", rerank_model.template_id)
    by_stem = {m.stem: m for m in models}
    coload_vars: list[str] = []
    for stem in coload_stems:
        m = by_stem.get(stem)
        if m is None:
            continue
        prefix = _COLOAD_VAR_PREFIX.get(m.role, "coload")
        name, n = prefix, 1
        while name in vars_:
            n += 1
            name = f"{prefix}{n}"
        vars_[name] = m.template_id
        coload_vars.append(name)
    logger.info("matrix vars: %d chat + %d category + %d coload",
                chat_idx, len(categories or {}), len(coload_vars))
    return vars_, coload_vars


def _health_check_timeout(models, args) -> int:
    """Auto-calculated healthCheckTimeout when not set explicitly.

    max(120, 1.2 * largest_model_mb / drive_speed_mb).  vLLM models get their
    own floor at model_size_mb / 100 — a conservative 100 MB/s load-rate
    assumption (docker pull, HF hub resolution and weight load exceed the
    llama.cpp load path by minutes; a ~60 GB NVFP4 model loads in ~10 min on
    DGX Spark).  Repo-only vLLM models (no measurable local file) keep the
    300 s floor.
    """
    largest_mb = max(
        (m.gguf_path.stat().st_size // (1024 * 1024)
         for m in models if m.gguf_path and m.gguf_path.is_file()),
        default=0,
    )
    # Resolve drive speed: CLI > env > auto-detect > default 100 MB/s
    drive_speed = args.drive_speed
    if drive_speed is None:
        env_speed = os.environ.get("GEN_CONFIG_DRIVE_SPEED")
        if env_speed:
            try:
                drive_speed = int(env_speed)
            except ValueError:
                logger.warning("GEN_CONFIG_DRIVE_SPEED=%r is not numeric; ignoring", env_speed)
                drive_speed = None
    if drive_speed is None:
        model_paths = [m.gguf_path for m in models if m.gguf_path and m.gguf_path.is_file()]
        drive_speed = _detect_drive_speed(model_paths)
    hct = max(120, int(1.2 * largest_mb / drive_speed))
    # vLLM floor: assumed 100 MB/s load rate on the largest measurable model.
    vllm_size_mb = max(
        (m.gguf_path.stat().st_size // (1024 * 1024)
         for m in models
         if m.backend in VLLM_BACKENDS and m.gguf_path and m.gguf_path.is_file()),
        default=0,
    )
    if any(m.backend in VLLM_BACKENDS for m in models):
        hct = max(hct, 300, vllm_size_mb // 100)
    # sd-server models also need generous timeout (diffusion weights load minutes)
    if any(m.backend in SD_BACKENDS for m in models):
        hct = max(hct, 300)
    logger.info("healthCheckTimeout: %ds (largest=%dMB, drive=%dMB/s%s)",
                hct, largest_mb, drive_speed,
                f", vllm-largest={vllm_size_mb}MB" if vllm_size_mb else "")
    return hct


def _expand_matrix_sets(sets_cfg: dict, chat_expr: str,
                        coload_vars: list[str], logger) -> dict:
    """Expand ``__CHAT_VARS__``/``__COLOAD_VARS__`` placeholders in set exprs.

    Placeholders expand to parenthesized OR-lists of VAR NAMES (not model
    IDs — the llama-swap sets DSL references var names, which map to model
    IDs via ``vars``).  With no co-loads included, a ``__COLOAD_VARS__`` term
    is dropped from the expression entirely.
    """
    coload_expr = "(" + " | ".join(coload_vars) + ")" if coload_vars else None
    expanded: dict[str, str] = {}
    for sname, sexpr in sets_cfg.items():
        expr = sexpr.replace("__CHAT_VARS__", chat_expr)
        if coload_expr:
            expr = expr.replace("__COLOAD_VARS__", coload_expr)
        else:
            expr = (expr.replace("& __COLOAD_VARS__", "")
                        .replace("__COLOAD_VARS__ &", "")
                        .replace("__COLOAD_VARS__", ""))
            if "__COLOAD_VARS__" in sexpr:
                logger.warning(
                    "matrix: set %r references __COLOAD_VARS__ but no "
                    "co-loads were included; term dropped", sname)
        expanded[sname] = expr.strip()
    return expanded


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    if args.verbose:
        logger.info("llama-packer %s", __version__)

    profiles_path = Path(args.profiles).absolute()
    if not profiles_path.is_file():
        bundled = importlib.resources.files("llama_packer").joinpath("profiles.yaml")
        if bundled.is_file():
            logger.warning(
                "no profiles file at %s — proceeding with bundled defaults. "
                "Copy profiles.yaml.example (%s) to that location to configure "
                "models_dirs, hf_home, overrides and sampling profiles.",
                profiles_path, Path(str(bundled)).parent)
            profiles_path = Path(str(bundled))
        else:
            fatal("profiles file not found: %s (see profiles.yaml.example)",
                  profiles_path)

    with open(profiles_path) as f:
        profiles_cfg = yaml.safe_load(f) or {}

    if not Profiles(profiles_cfg).profile_list:
        fatal("no profiles defined in profiles.yaml")

    # Models dirs: CLI --models-dir > profiles.yaml models_dirs: > ./models.
    if args.models_dir:
        models_dirs = [Path(d).absolute() for d in args.models_dir]
    else:
        models_dirs = [Path(d).absolute() for d in (profiles_cfg.get("models_dirs") or ["models"])]
    missing = [d for d in models_dirs if not d.is_dir()]
    if missing:
        for d in missing:
            fatal("models directory not found: %s", d)

    if args.agents:
        for d in models_dirs:
            write_agents_md(d)

    # HF cache root: CLI > profiles.yaml > env (used for hub snapshot
    # resolution and ${HF_HOME} path grouping).
    hf_home = args.hf_home or profiles_cfg.get("hf_home")

    # Directory-name → role map extension from profiles.yaml `dirs:`.
    dir_roles = profiles_cfg.get("dirs") or {}
    if not isinstance(dir_roles, dict):
        fatal("profiles.yaml dirs: must be a mapping of directory name to role")
    err = validate_dir_roles(dir_roles)
    if err:
        fatal("profiles.yaml %s", err)

    # Backend enable/prefer list (ordered; absent = all registered).
    backends_cfg = profiles_cfg.get("backends") or []
    if not isinstance(backends_cfg, list):
        fatal("profiles.yaml backends: must be a list of backend names")
    err = validate_backend_names(backends_cfg)
    if err:
        fatal("profiles.yaml %s", err)

    if args.llama_server:
        llama_bin = str(Path(args.llama_server).resolve())
        bin_dir = str(Path(args.llama_server).parent)
    else:
        bin_dir = find_bin_dir(args.version, Path.cwd())
        llama_bin = str(Path.cwd() / bin_dir / "llama-server")

    fit_bin = str(Path.cwd() / bin_dir / "llama-fit-params")

    template_vars = {
        "llama_bin": llama_bin,
        "models_dir": str(models_dirs[0]),
        "models_dirs": [str(d) for d in models_dirs],
        # Fleet-wide flag strings, assigned in main() below:
        # llama_args  — profiles.yaml llama_server.args
        # vllm_args   — profiles.yaml vllm.args
        # sd_args     — profiles.yaml sd.args
        # whisper_args — profiles.yaml whisper.args
    }

    max_ctx = None
    if args.max_context:
        from llama_packer.utils import parse_context_length
        max_ctx = parse_context_length(args.max_context)

    min_ctx = None
    if args.min_context:
        from llama_packer.utils import parse_context_length
        min_ctx = parse_context_length(args.min_context)

    # vLLM resource configuration (CLI > profiles.yaml `vllm:` section >
    # built-in constants).  Resolved *before* discovery so backend inference
    # knows whether a vLLM backend can actually run.
    vllm_cfg = profiles_cfg.get("vllm") or {}
    vllm_image = str(args.vllm_image or vllm_cfg.get("image") or VLLM_DEFAULT_IMAGE)
    vllm_bin = str(args.vllm_server or vllm_cfg.get("bin") or VLLM_DEFAULT_BIN)

    # sd-server resource configuration (CLI > profiles.yaml `sd:` section >
    # $SD_BIN_DIR > sd-server on PATH).  Single host binary for now (docker follow-up).
    sd_cfg = profiles_cfg.get("sd") or {}
    sd_bin_raw = args.sd_server or sd_cfg.get("bin") or os.environ.get("SD_BIN_DIR") or shutil.which("sd-server")
    sd_bin = None
    if sd_bin_raw:
        raw = str(sd_bin_raw)
        # SD_BIN_DIR may be a directory (mirrors LLAMA_BIN_DIR); expand to binary inside.
        cand = Path(raw)
        if cand.is_dir():
            cand = cand / "sd-server"
        sd_bin = str(cand) if cand else None
        # When the explicit path doesn't exist, fall back to which() so a
        # stale profile bin doesn't disable inference entirely.
        if sd_bin and not Path(sd_bin).is_file() and not shutil.which(str(sd_bin)):
            # Keep the raw value — is_available will be False, inference won't pick it,
            # but an explicit `backend: sd-server` still reports a clear error.
            pass

    # whisper-server resource configuration (CLI > profiles.yaml `whisper:` section >
    # $WHISPER_BIN_DIR > whisper-server on PATH).  Single host binary.
    whisper_cfg = profiles_cfg.get("whisper") or {}
    whisper_bin_raw = (args.whisper_server or whisper_cfg.get("bin")
                       or os.environ.get("WHISPER_BIN_DIR") or shutil.which("whisper-server"))
    whisper_bin = None
    if whisper_bin_raw:
        cand = Path(str(whisper_bin_raw))
        if cand.is_dir():  # WHISPER_BIN_DIR may be a directory (mirrors LLAMA_BIN_DIR)
            cand = cand / "whisper-server"
        whisper_bin = str(cand)

    # audio-cpp (audio.cpp) resource configuration (CLI > profiles.yaml
    # `audio_cpp:` section > $AUDIOCPP_BIN_DIR > audiocpp_server on PATH).
    # `backend:` (auto|cuda|vulkan|cpu) selects the engine runtime; auto
    # follows the detected GPU vendor (NVIDIA→cuda, AMD→vulkan, else cpu).
    audio_cpp_cfg = profiles_cfg.get("audio_cpp") or {}
    audio_cpp_bin_raw = (args.audio_cpp_server or audio_cpp_cfg.get("bin")
                         or os.environ.get("AUDIOCPP_BIN_DIR")
                         or shutil.which("audiocpp_server"))
    audio_cpp_bin = None
    if audio_cpp_bin_raw:
        cand = Path(str(audio_cpp_bin_raw))
        if cand.is_dir():  # AUDIOCPP_BIN_DIR may be a directory
            cand = cand / "audiocpp_server"
        audio_cpp_bin = str(cand)
    audio_cpp_backend = str(audio_cpp_cfg.get("backend") or "auto").lower()
    if audio_cpp_backend in ("", "auto"):
        audio_cpp_backend = {"nvidia": "cuda", "amd": "vulkan"}.get(
            detect_gpu_vendor(), "cpu")

    # Discover models via a depth-first walk.  The scope stack carries the
    # global override rules (bottom scope); each directory's models.yaml is
    # pushed/popped around its level.  Defaults, rules, companion resolution
    # and backend finalization all happen inside discover().
    stack = ScopeStack(
        avail={
            "llama_bin": llama_bin,
            "vllm_image": vllm_image,
            "vllm_bin": vllm_bin,
            "sd_bin": sd_bin or "",
            "whisper_bin": whisper_bin or "",
            "audio_cpp_bin": audio_cpp_bin or "",
            # Container runtimes: probed once so container transports are only
            # inferred when their runtime is actually on PATH.
            "docker": bool(shutil.which("docker")),
            "podman": bool(shutil.which("podman")),
        },
        allowed=[str(b) for b in backends_cfg] or None,
    )
    stack.push({"overrides": profiles_cfg.get("overrides")},
               origin=str(profiles_path))
    models = discover(models_dirs, stack=stack,
                      generate_stubs=not args.no_stubs,
                      extra_dirs=args.extra_dirs, dir_roles=dir_roles,
                      hf_home=hf_home)
    if not models:
        fatal("no models found (create a .md sidecar file)")
    logger.info("models: %d found", len(models))

    if args.remeasure:
        for m in models:
            m.vram.remeasure = True
        logger.info("--remeasure: ignoring saved VRAM measurements "
                    "(sidecar derived: blocks) — re-running the fit-params "
                    "trio per model")

    if args.probe_memory is not None:
        from llama_packer.memory_probe import run_probe
        llama_args = backend_args(profiles_cfg.get("llama_server"),
                                  "llama_server")
        is_fast = make_fast_storage_predicate(
            (profiles_cfg.get("hardware") or {}).get("fast_storage"))
        print(run_probe(
            models, fit_bin, llama_bin,
            Profiles(profiles_cfg).default_cache_type,
            only_archs=tuple(args.probe_memory) or None,
            llama_args=llama_args,
            is_fast=is_fast,
            profiles=Profiles(profiles_cfg)))
        return

    # Auto-calculated healthCheckTimeout: max(120, 1.2 * largest_model_mb / drive_speed_mb)
    if args.health_check_timeout is None:
        hct = _health_check_timeout(models, args)
    else:
        hct = args.health_check_timeout

    # Get VRAM / GPU profile (CLI > profiles.yaml hardware section > auto-detect)
    yaml_hw = profiles_cfg.get("hardware") or {}
    gpu = GpuProfile.from_args(
        vram=args.vram,
        gpu_family=args.gpu_family,
        yaml_hw=yaml_hw,
        baseline=args.baseline,
        unified_system_mb=args.unified_system_mb,
    )

    # Compute optimal ${env.*} prefixes from the raw paths actually emitted,
    # then substitute them into the binary path and post-process the config.
    raw_paths = [llama_bin]
    if sd_bin:
        raw_paths.append(sd_bin)
    if whisper_bin:
        raw_paths.append(whisper_bin)
    for _m in models:
        if getattr(_m, "_override_error", None):
            continue
        if _m.gguf_path:
            raw_paths.append(str(_m.gguf_path))
        if _m.mmproj and _m.mmproj.gguf_path:
            raw_paths.append(str(_m.mmproj.gguf_path))
        if _m.mtp and _m.mtp.gguf_path:
            raw_paths.append(str(_m.mtp.gguf_path))
        ct = _m.resolved_chat_template
        if ct is not None:
            raw_paths.append(str(ct))
        for _lora in _m.resolved_loras:
            raw_paths.append(str(_lora))
    prefix_to_var, var_to_value = compute_env_prefixes(raw_paths, project_hint=llama_bin, hf_home=hf_home)
    sub = make_subst(prefix_to_var)
    template_vars["llama_bin"] = sub(llama_bin)
    if sd_bin:
        template_vars["sd_bin"] = sub(sd_bin)
    if whisper_bin:
        template_vars["whisper_bin"] = sub(whisper_bin)

    # vLLM backend defaults: already resolved above (CLI > profiles.yaml >
    # built-in constants) for backend inference.
    template_vars["vllm_image"] = vllm_image
    template_vars["vllm_bin"] = vllm_bin
    template_vars.setdefault("sd_bin", "sd-server")
    template_vars.setdefault("whisper_bin", "whisper-server")
    template_vars["audio_cpp_bin"] = audio_cpp_bin or AUDIO_CPP_DEFAULT_BIN
    template_vars["audio_cpp_backend"] = audio_cpp_backend
    for _knob in AUDIO_CPP_SERVER_KNOBS:
        _value = audio_cpp_cfg.get(_knob)
        if _value not in (None, ""):
            template_vars[f"audio_cpp_{_knob}"] = str(_value)

    template_vars["docker_args"] = str(vllm_cfg.get("docker_args") or VLLM_DEFAULT_DOCKER_ARGS)
    # GPU vendor for container device flags (docker --runtime/--gpus vs
    # podman --device); overridable per run via profiles.yaml.
    template_vars["container_vendor"] = str(
        vllm_cfg.get("container_vendor") or detect_gpu_vendor())
    template_vars["container_port"] = str(vllm_cfg.get("container_port") or VLLM_DEFAULT_CONTAINER_PORT)
    # Host HF_HOME root bind-mounted into vllm-docker containers
    # (/root/.cache/huggingface).  HF_HUB_OFFLINE forbids downloads, so this
    # must cover every repo-id model served via docker.  Falls back to the
    # top-level `hf_home:`.
    template_vars["hf_cache"] = str(vllm_cfg.get("hf_cache")
                                    or profiles_cfg.get("hf_home") or "")
    template_vars["vllm_args"] = backend_args(vllm_cfg, "vllm")
    template_vars["sd_args"] = backend_args(sd_cfg, "sd")
    template_vars["whisper_args"] = backend_args(whisper_cfg, "whisper")
    template_vars["llama_args"] = backend_args(profiles_cfg.get("llama_server"), "llama_server")
    # The named batch/ubatch keys render after the global args and win per
    # flag: args -b/-ub would be silently shadowed — point at the keys.
    _llama_args_toks = set(template_vars["llama_args"].split())
    if "-b" in _llama_args_toks or "-ub" in _llama_args_toks:
        logger.warning("llama_server.args carries -b/-ub — shadowed by the "
                       "rendered batch/ubatch keys; move them to "
                       "llama_server: batch:/ubatch:")
    # gpu_mem_util: explicit profiles.yaml value wins; otherwise derive the
    # fraction from the same reserve/spare budget llama.cpp uses, so vLLM's
    # --max-model-len and --gpu-memory-utilization describe one consistent pool.
    if vllm_cfg.get("gpu_mem_util") is not None:
        template_vars["gpu_mem_util"] = str(vllm_cfg["gpu_mem_util"])
    else:
        spare_mb = Profiles(profiles_cfg).global_spare_mb(args.spare, gpu.vram_mb)
        reserve = _RESERVE_SYSTEM + max(_RESERVE_VIDEO, gpu.baseline_mb)
        available = gpu.vram_mb - reserve - spare_mb
        if gpu.vram_mb > 0:
            util = max(0.0, min(1.0, available / gpu.vram_mb))
        else:
            util = VLLM_DEFAULT_GPU_MEM_UTIL
        template_vars["gpu_mem_util"] = str(round(util, 3))
        logger.info("vllm gpu-memory-utilization: %s (derived from budget)",
                    template_vars["gpu_mem_util"])

    # Detect matrix configuration before build_config
    (matrix_cfg, embed_model, rerank_model, matrix_categories,
     matrix_fixed) = _detect_matrix(profiles_cfg, models, args, logger)

    # Build config (progress bar appears only once the denominator is
    # known — total=len(models); without rich / non-TTY it is a no-op).
    # Memory margin: inflate every measured term so the affine residual
    # errs toward reserving more (profiles.yaml hardware.memory_margin).
    margin = _MEMORY_MARGIN
    raw_margin = (profiles_cfg.get("hardware") or {}).get("memory_margin")
    if raw_margin is not None:
        try:
            margin = float(raw_margin)
            if margin < 0:
                raise ValueError
        except (TypeError, ValueError):
            fatal("profiles.yaml hardware.memory_margin: %r is not a "
                  "non-negative number", raw_margin)
    progress = PackerProgress(enabled=sys.stderr.isatty())
    progress.start(len(models), "budgeting")
    try:
        config = build_config(
            models, Profiles(profiles_cfg), template_vars, fit_bin, gpu.vram_mb,
            spare=args.spare, max_context=max_ctx,
            matrix_cfg=matrix_cfg, embed_model=embed_model, rerank_model=rerank_model,
            fixed_categories=matrix_fixed,
            baseline_mb=gpu.baseline_mb,
            min_context=min_ctx if min_ctx is not None else _MIN_AGENTIC_CTX,
            min_context_explicit=min_ctx is not None,
            memory_margin=margin,
            progress_cb=lambda stem: progress.advance(stem),
        )
    except ValueError as e:
        fatal("%s", e)
    except RuntimeError as e:
        # Planning must never traceback: a stuck point (unmeasurable model,
        # broken binary) degrades to a clean fatal with the reason.
        fatal("planning failed: %s", e)
    finally:
        progress.stop()
    # Prepare flag macros (placeholder domain) — auto on unless --no-macros
    flag_macros: dict[str, str] = {}
    if not args.no_macros:
        from llama_packer.macros import Macro, Macros
        Macro.clear()
        Macros(profiles_cfg, Profiles(profiles_cfg), models_dirs, sub)
        # Apply env substitution to flag macro definitions as well (they were
        # built with placeholder-aware sub, but ensure consistency)
        flag_macros = Macro.definitions()
        logger.info("flag macros: %d registered (%s)", len(flag_macros), ", ".join(sorted(flag_macros)) if flag_macros else "none")
    config = _apply_env_subst(config, sub, raw_paths)
    # Apply flag macros to every cmd (post-env, placeholder domain)
    if flag_macros:
        from llama_packer.macros import Macro
        for entry in config.get("models", {}).values():
            cmd = entry.get("cmd")
            if cmd:
                entry["cmd"] = Macro.apply(cmd)
    if not config.get("models"):
        fatal("no model entries generated")
    logger.info("entries: %d generated", len(config["models"]))

    # Resolve ${VAR} macros to absolute paths in the config itself so that
    # -watch-config reloads pick up new paths (e.g. a new llama-server version)
    # without requiring a llama-swap service restart. Merge path + flag macros.
    # Preserve creation order so a flag macro that references a path macro
    # (e.g. MODELS_CHAT_QWEN3 → ${MODELS_DIR}/...) is defined after its
    # dependency — alphabetical sorting breaks nested substitution in llama-swap.
    merged_macros: dict[str, str] = dict(var_to_value)
    # Flag macros may collide with path macros — new replaces old with warning
    for k, v in flag_macros.items():
        if k in merged_macros:
            logger.warning("macros: flag macro %r collides with path macro %r; flag wins", k, merged_macros[k])
        merged_macros[k] = v
    config["macros"] = merged_macros

    # ── Swap matrix: build matrix vars if configured ──
    if matrix_cfg and embed_model and rerank_model:
        vars_, coload_vars = _build_matrix_vars(
            models, embed_model, rerank_model, matrix_categories,
            config.coload_stems, config.entry_ids_by_stem, logger)
        chat_var_names = [k for k in vars_ if re.fullmatch(r"c\d+", k)]
        # Parenthesized OR-lists: '&' binds tighter than '|' in the DSL.
        chat_expr = "(" + " | ".join(chat_var_names) + ")"
        sets_cfg = matrix_cfg.get("sets") or {}
        sets = _expand_matrix_sets(sets_cfg, chat_expr, coload_vars, logger)
        if coload_vars:
            joined = " ".join(str(s) for s in sets_cfg.values())
            if "__COLOAD_VARS__" not in joined:
                logger.info(
                    "matrix: co-loads %s included but no set references "
                    "__COLOAD_VARS__; they stay outside the co-loading sets",
                    coload_vars)
        # llama-swap schema: matrix lives under routing.router.settings.matrix,
        # not at the top level.
        config["routing"] = {
            "router": {
                "use": "matrix",
                "settings": {"matrix": {
                    "vars": vars_,
                    "evict_costs": matrix_cfg.get("evict_costs", {}),
                    "sets": sets,
                }},
            },
        }

    # Top-level llama-swap settings
    config["healthCheckTimeout"] = hct

    output_path = Path(args.output).absolute()

    if args.dry_run:
        payload = config.plain() if isinstance(config, EmittedConfig) else config
        sys.stdout.write(yaml.dump(payload, default_flow_style=False, sort_keys=False, allow_unicode=True))
        for _name in sorted(var_to_value):
            logger.info("env %s=%s", _name, var_to_value[_name])
    else:
        write_yaml(config, output_path)
        logger.info("written: %s", output_path)
        if not args.no_env:
            env_path = output_path.with_name("config.env")
            env_lines = [
                "# Generated by llama-packer",
                "# Source via systemd EnvironmentFile= or docker --env-file.",
                "# Values double as docker bind-mount sources, e.g. -v ${MODELS_DIR}:/models",
            ]
            for _name in sorted(var_to_value):
                env_lines.append(f"{_name}={var_to_value[_name]}")
            env_path.write_text("\n".join(env_lines) + "\n")
            logger.info("env written: %s", env_path)


if __name__ == "__main__":
    main()