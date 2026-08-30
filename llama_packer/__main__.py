# llama_packer/__main__.py
"""CLI entry point for llama-packer."""

from __future__ import annotations

import argparse
import importlib.resources
import logging
import os
import shutil
import sys
import textwrap
from pathlib import Path

import yaml

from llama_packer import Model, __version__, find_bin_dir
from llama_packer.hardware import GpuProfile, detect_gpu_vendor
from llama_packer.profiles import Profiles
from llama_packer.scope import ScopeStack
from llama_packer.discover import discover
from llama_packer.utils import (
    _MIN_USEFUL_CTX, compute_env_prefixes, make_subst, _detect_drive_speed,
    _RESERVE_SYSTEM, _RESERVE_VIDEO,
    VLLM_DEFAULT_IMAGE, VLLM_DEFAULT_BIN, VLLM_DEFAULT_DOCKER_ARGS,
    VLLM_DEFAULT_CONTAINER_PORT, VLLM_DEFAULT_GPU_MEM_UTIL,
    validate_dir_roles, NON_CHAT_ROLES,
)
from llama_packer.writer import build_config, write_yaml, EmittedConfig
from llama_packer.backends import (SD_BACKENDS, VLLM_BACKENDS,
                                   validate_backend_names)
from llama_packer.backends.kokoro import KOKORO_DEFAULT_IMAGES, KOKORO_CONTAINER_PORT




_LOG_LEVELS = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}

logger = logging.getLogger(__name__)


def setup_logging(verbosity: int = 0) -> None:
    level = _LOG_LEVELS.get(verbosity, logging.DEBUG)
    logging.basicConfig(
        level=level,
        format="%(levelname).1s | %(message)s",
        stream=sys.stderr,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate llama-swap config.yaml from model metadata and profiles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              llama-packer                              # defaults
              llama-packer --dry-run                    # preview
              llama-packer -v 8929                      # specific llama-server version
              llama-packer --llama-server /opt/lsrv     # explicit binary path
              llama-packer --output /etc/ls/config.yaml
        """),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print config to stdout instead of writing")
    parser.add_argument("--output", default="config.yaml", help="Output path (default: config.yaml)")
    parser.add_argument("-v", "--llama-version", default="latest", dest="version",
                        help="llama-server version (default: latest)")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}",
                        help="Show llama-packer version and exit")
    parser.add_argument("--llama-server", help="Explicit path to llama-server binary")
    parser.add_argument("--models-dir", nargs="+", default=None,
                        help="Model directories (default: profiles.yaml models_dirs, "
                             "else ./models); pass multiple to scan several")
    parser.add_argument("--hf-home", help="HF cache root for hub snapshot resolution and path grouping "
                        "(overrides profiles.yaml hf_home / $HF_HOME / $HUGGINGFACE_HUB_CACHE)")
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
    parser.add_argument("--verbose", "-V", action="count", default=0, help="Increase verbosity (-V: info, -VV: debug)")
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
    parser.add_argument("--kokoro-image", help="container image for `kokoro-podman` backend "
                         "(overrides profiles.yaml t2s.image; default: vendor-detected upstream image)")
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


def _apply_env_subst(config: dict, sub, raw_paths: list[str]) -> dict:
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
            logger.error("selector %r matched %d %s models (need exactly 1): %s",
                         selector, len(hits), type_name,
                         [h.stem for h in hits])
            sys.exit(1)
        return hits[0]
    return min(cands, key=lambda m: m.vram_mb)


def _build_matrix_vars(models: list, embed_model, rerank_model,
                       entry_ids_by_stem: dict[str, list[str]], logger) -> dict:
    """Auto-collect matrix vars: chat entries + selected embed/rerank.

    Each chat model contributes a var per emitted entry id (the bare id and,
    when present, its text-only variant — see writer.TEXT_SUFFIX), so text
    variants join the same co-loading sets as their parent entry.  The
    stem → ids mapping comes from the emitter (EmittedConfig) so naming is
    owned in exactly one place.
    """
    vars_: dict[str, str] = {}
    chat_idx = 0
    for m in models:
        if m.role in NON_CHAT_ROLES:
            continue
        for eid in entry_ids_by_stem.get(m.stem, []):
            chat_idx += 1
            vars_[f"c{chat_idx}"] = eid
    if embed_model is not None:
        vars_["emb"] = embed_model.template_id
    if rerank_model is not None:
        vars_["rnk"] = rerank_model.template_id
    logger.info("matrix vars: %d chat + emb + rnk", chat_idx)
    return vars_


def _health_check_timeout(models, args) -> int:
    """Auto-calculated healthCheckTimeout when not set explicitly.

    max(120, 1.2 * largest_model_mb / drive_speed_mb), raised to a 300s floor
    when any model uses a vLLM backend (docker pull / HF download / model load
    exceed the llama.cpp load path by minutes).
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
    if any(m.backend in VLLM_BACKENDS for m in models):
        hct = max(hct, 300)
    # sd-server models also need generous timeout (diffusion weights load minutes)
    if any(m.backend in SD_BACKENDS for m in models):
        hct = max(hct, 300)
    logger.info("healthCheckTimeout: %ds (largest=%dMB, drive=%dMB/s)",
                hct, largest_mb, drive_speed)
    return hct


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
            logger.error("profiles file not found: %s (see profiles.yaml.example)",
                         profiles_path)
            sys.exit(1)

    with open(profiles_path) as f:
        profiles_cfg = yaml.safe_load(f) or {}

    if not Profiles(profiles_cfg).profile_list:
        logger.error("no profiles defined in profiles.yaml")
        sys.exit(1)

    # Models dirs: CLI --models-dir > profiles.yaml models_dirs: > ./models.
    if args.models_dir:
        models_dirs = [Path(d).absolute() for d in args.models_dir]
    else:
        models_dirs = [Path(d).absolute() for d in (profiles_cfg.get("models_dirs") or ["models"])]
    missing = [d for d in models_dirs if not d.is_dir()]
    if missing:
        for d in missing:
            logger.error("models directory not found: %s", d)
        sys.exit(1)

    if args.agents:
        for d in models_dirs:
            write_agents_md(d)

    # HF cache root: CLI > profiles.yaml > env (used for hub snapshot
    # resolution and ${HF_HOME} path grouping).
    hf_home = args.hf_home or profiles_cfg.get("hf_home")

    # Directory-name → role map extension from profiles.yaml `dirs:`.
    dir_roles = profiles_cfg.get("dirs") or {}
    if not isinstance(dir_roles, dict):
        logger.error("profiles.yaml dirs: must be a mapping of directory name to role")
        sys.exit(1)
    err = validate_dir_roles(dir_roles)
    if err:
        logger.error("profiles.yaml %s", err)
        sys.exit(1)

    # Backend enable/prefer list (ordered; absent = all registered).
    backends_cfg = profiles_cfg.get("backends") or []
    if not isinstance(backends_cfg, list):
        logger.error("profiles.yaml backends: must be a list of backend names")
        sys.exit(1)
    err = validate_backend_names(backends_cfg)
    if err:
        logger.error("profiles.yaml %s", err)
        sys.exit(1)

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

    # kokoro-podman resource configuration (CLI > profiles.yaml `t2s:` section >
    # vendor-detected upstream image).  `vendor:` (auto|nvidia|amd|cpu) picks
    # the default image tag AND device flags; `image:`/--kokoro-image overrides
    # the tag only; `podman_args:` replaces the auto device flags entirely.
    t2s_cfg = profiles_cfg.get("t2s") or {}
    kokoro_vendor = str(t2s_cfg.get("vendor") or detect_gpu_vendor())
    if kokoro_vendor not in KOKORO_DEFAULT_IMAGES:
        logger.warning("profiles.yaml t2s.vendor: %r unknown (auto/nvidia/amd/cpu); "
                       "using cpu defaults", kokoro_vendor)
        kokoro_vendor = "cpu"
    kokoro_image = str(args.kokoro_image or t2s_cfg.get("image")
                       or KOKORO_DEFAULT_IMAGES[kokoro_vendor])

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
            "kokoro_image": kokoro_image,
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
        logger.error("no models found (create a .md sidecar file)")
        sys.exit(1)
    logger.info("models: %d found", len(models))

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
    raw_paths = [llama_bin, fit_bin]
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
    template_vars["kokoro_image"] = kokoro_image
    template_vars["kokoro_vendor"] = kokoro_vendor
    template_vars["podman_args"] = str(t2s_cfg.get("podman_args") or "")
    template_vars["kokoro_container_port"] = str(
        t2s_cfg.get("container_port") or KOKORO_CONTAINER_PORT)
    if t2s_cfg.get("voices_dir"):
        template_vars["voices_dir"] = str(t2s_cfg["voices_dir"])

    template_vars["docker_args"] = str(vllm_cfg.get("docker_args") or VLLM_DEFAULT_DOCKER_ARGS)
    template_vars["container_port"] = str(vllm_cfg.get("container_port") or VLLM_DEFAULT_CONTAINER_PORT)
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
    matrix_cfg = profiles_cfg.get("matrix")
    embed_model = None
    rerank_model = None
    if matrix_cfg:
        embed_model = _select_model(models, "embeddings", args.embed, logger)
        rerank_model = _select_model(models, "rerank", args.rerank, logger)
        if embed_model is None:
            logger.warning("no embeddings model found; skipping matrix")
            matrix_cfg = None
        elif rerank_model is None:
            logger.warning("no rerank model found; skipping matrix")
            matrix_cfg = None
        else:
            logger.info("matrix embed: %s", embed_model.stem)
            logger.info("matrix rerank: %s", rerank_model.stem)

    # Build config
    try:
        config = build_config(
            models, Profiles(profiles_cfg), template_vars, fit_bin, gpu.vram_mb,
            spare=args.spare, max_context=max_ctx,
            matrix_cfg=matrix_cfg, embed_model=embed_model, rerank_model=rerank_model,
            baseline_mb=gpu.baseline_mb,
            min_context=min_ctx if min_ctx is not None else _MIN_USEFUL_CTX,
        )
    except ValueError as e:
        logger.error("%s", e)
        sys.exit(1)
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
        logger.error("no model entries generated")
        sys.exit(1)
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
        vars_ = _build_matrix_vars(models, embed_model, rerank_model,
                                   config.entry_ids_by_stem, logger)
        # Expand the __CHAT_VARS__ placeholder in each set with the chat VAR
        # NAMES (c1 | c2 | ...), not the model IDs — llama-swap sets DSL
        # references var names, which map to model IDs via `vars`.
        chat_var_names = [k for k in vars_ if k.startswith("c")]
        # Parenthesize the OR-list: '&' binds tighter than '|' in the DSL.
        chat_expr = "(" + " | ".join(chat_var_names) + ")"
        sets = {}
        for sname, sexpr in (matrix_cfg.get("sets") or {}).items():
            sets[sname] = sexpr.replace("__CHAT_VARS__", chat_expr)
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