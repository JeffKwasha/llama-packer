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

from llama_packer import Model, find_bin_dir
from llama_packer.hardware import GpuProfile
from llama_packer.utils import (
    _MIN_USEFUL_CTX, compute_env_prefixes, make_subst, _detect_drive_speed,
    VLLM_DEFAULT_IMAGE, VLLM_DEFAULT_DOCKER_ARGS, VLLM_DEFAULT_CONTAINER_PORT,
    VLLM_DEFAULT_GPU_MEM_UTIL,
)
from llama_packer.writer import build_config, write_yaml


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
              gen-config                              # defaults
              gen-config --dry-run                    # preview
              gen-config -v 8929                      # specific version
              gen-config --llama-server /opt/lsrv     # explicit binary path
              gen-config --output /etc/ls/config.yaml
        """),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print config to stdout instead of writing")
    parser.add_argument("--output", default="config.yaml", help="Output path (default: config.yaml)")
    parser.add_argument("-v", "--version", default="latest", help="llama-server version (default: latest)")
    parser.add_argument("--llama-server", help="Explicit path to llama-server binary")
    parser.add_argument("--models-dir", default="models", help="Model directory (default: ./models)")
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
    parser.add_argument("--vllm-image", help="vLLM docker image for `template: vllm-docker` models "
                         "(overrides profiles.yaml vllm.image)")
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
    """Replace each raw emitted path in the generated cmds with its ${env.*} form."""
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


def _build_matrix_vars(models: list, embed_model, rerank_model, logger) -> dict:
    """Auto-collect matrix vars: chat models + selected embed/rerank."""
    vars_: dict[str, str] = {}
    chat_idx = 0
    for m in models:
        if m.role in ("embeddings", "rerank"):
            continue
        chat_idx += 1
        vars_[f"c{chat_idx}"] = m.template_id
    if embed_model is not None:
        vars_["emb"] = embed_model.template_id
    if rerank_model is not None:
        vars_["rnk"] = rerank_model.template_id
    return vars_


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    models_dir = Path(args.models_dir).absolute()
    if not models_dir.is_dir():
        logger.error("models directory not found: %s", models_dir)
        sys.exit(1)

    if args.agents:
        write_agents_md(models_dir)

    profiles_path = Path(args.profiles).absolute()
    if not profiles_path.is_file():
        bundled = importlib.resources.files("llama_packer").joinpath("profiles.yaml")
        if bundled.is_file():
            profiles_path = Path(str(bundled))
            logger.info("using bundled profiles: %s", profiles_path)
        else:
            logger.error("profiles file not found: %s", profiles_path)
            sys.exit(1)

    with open(profiles_path) as f:
        profiles_cfg = yaml.safe_load(f) or {}

    if not profiles_cfg.get("templates"):
        logger.error("no templates defined in profiles.yaml")
        sys.exit(1)
    if not profiles_cfg.get("profiles"):
        logger.error("no profiles defined in profiles.yaml")
        sys.exit(1)

    if args.llama_server:
        llama_bin = str(Path(args.llama_server).resolve())
        bin_dir = str(Path(args.llama_server).parent)
    else:
        bin_dir = find_bin_dir(args.version, Path.cwd())
        llama_bin = str(Path.cwd() / bin_dir / "llama-server")

    fit_bin = str(Path.cwd() / bin_dir / "llama-fit-params")

    template_vars = {"llama_bin": llama_bin, "models_dir": str(models_dir)}

    max_ctx = None
    if args.max_context:
        from llama_packer.utils import parse_context_length
        max_ctx = parse_context_length(args.max_context)

    min_ctx = None
    if args.min_context:
        from llama_packer.utils import parse_context_length
        min_ctx = parse_context_length(args.min_context)

    # Discover models
    models = Model.from_dir(models_dir, generate_stubs=not args.no_stubs, extra_dirs=args.extra_dirs)
    if not models:
        logger.error("no models found (create a .md sidecar file)")
        sys.exit(1)
    logger.info("models: %d found", len(models))

    # Auto-calculate healthCheckTimeout: max(120, 1.2 * largest_model_mb / drive_speed_mb)
    if args.health_check_timeout is None:
        largest_mb = max(
            (m.gguf_path.stat().st_size // (1024 * 1024) for m in models if m.gguf_path and m.gguf_path.is_file()),
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
        logger.info("healthCheckTimeout: %ds (largest=%dMB, drive=%dMB/s)", hct, largest_mb, drive_speed)
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
    for _m in models:
        raw_paths.append(str(_m.gguf_path))
        if _m.mmproj and _m.mmproj.gguf_path:
            raw_paths.append(str(_m.mmproj.gguf_path))
        if _m.mtp and _m.mtp.gguf_path:
            raw_paths.append(str(_m.mtp.gguf_path))
    prefix_to_var, var_to_value = compute_env_prefixes(raw_paths, project_hint=llama_bin)
    sub = make_subst(prefix_to_var)
    template_vars["llama_bin"] = sub(llama_bin)

    # vLLM docker backend defaults: CLI --vllm-image > profiles.yaml `vllm:`
    # section > built-in constants.
    vllm_cfg = profiles_cfg.get("vllm") or {}
    if args.vllm_image:
        template_vars["vllm_image"] = args.vllm_image
    else:
        template_vars["vllm_image"] = (
            vllm_cfg.get("image") or VLLM_DEFAULT_IMAGE
        )
    template_vars["docker_args"] = str(vllm_cfg.get("docker_args") or VLLM_DEFAULT_DOCKER_ARGS)
    template_vars["container_port"] = str(vllm_cfg.get("container_port") or VLLM_DEFAULT_CONTAINER_PORT)
    template_vars["gpu_mem_util"] = str(vllm_cfg.get("gpu_mem_util") or VLLM_DEFAULT_GPU_MEM_UTIL)

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
    config = build_config(
        models, profiles_cfg, template_vars, fit_bin, gpu.vram_mb,
        spare=args.spare, max_context=max_ctx,
        matrix_cfg=matrix_cfg, embed_model=embed_model, rerank_model=rerank_model,
        baseline_mb=gpu.baseline_mb,
        min_context=min_ctx if min_ctx is not None else _MIN_USEFUL_CTX,
    )
    config = _apply_env_subst(config, sub, raw_paths)
    if not config.get("models"):
        logger.error("no model entries generated")
        sys.exit(1)
    logger.info("entries: %d generated", len(config["models"]))

    # Resolve ${VAR} macros to absolute paths in the config itself so that
    # -watch-config reloads pick up new paths (e.g. a new llama-server version)
    # without requiring a llama-swap service restart.
    config["macros"] = {name: value for name, value in sorted(var_to_value.items())}

    # ── Swap matrix: build matrix vars if configured ──
    if matrix_cfg and embed_model and rerank_model:
        vars_ = _build_matrix_vars(models, embed_model, rerank_model, logger)
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
        sys.stdout.write(yaml.dump(config, default_flow_style=False, sort_keys=False, allow_unicode=True))
        for _name in sorted(var_to_value):
            logger.info("env %s=%s", _name, var_to_value[_name])
    else:
        write_yaml(config, output_path)
        logger.info("written: %s", output_path)
        if not args.no_env:
            env_path = output_path.with_name("config.env")
            env_lines = [
                "# Generated by gen-config.py",
                "# Source via systemd EnvironmentFile= or docker --env-file.",
                "# Values double as docker bind-mount sources, e.g. -v ${MODELS_DIR}:/models",
            ]
            for _name in sorted(var_to_value):
                env_lines.append(f"{_name}={var_to_value[_name]}")
            env_path.write_text("\n".join(env_lines) + "\n")
            logger.info("env written: %s", env_path)


if __name__ == "__main__":
    main()