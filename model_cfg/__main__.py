# model_cfg/__main__.py
"""CLI entry point for model-cfg."""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
from pathlib import Path

import yaml

from model_cfg import Model, get_vram_mb, find_bin_dir
from model_cfg.utils import compute_env_prefixes, make_subst
from model_cfg.output import build_config, write_yaml, write_ini


SCRIPT_DIR = Path(__file__).resolve().parent.parent  # project root (one level up from model_cfg/)

_LOG_LEVELS = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}


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
    parser.add_argument("--format", choices=["yaml", "ini"], default="yaml", help="Output format (default: yaml)")
    parser.add_argument("-v", "--version", default="latest", help="llama-server version (default: latest)")
    parser.add_argument("--llama-server", help="Explicit path to llama-server binary")
    parser.add_argument("--models-dir", default="models", help="Model directory (default: ./models)")
    parser.add_argument("--profiles", default="profiles.yaml", help="Profiles file (default: profiles.yaml)")
    parser.add_argument("--no-stubs", action="store_true", help="Skip generating stub .md files")
    parser.add_argument("--extra-dirs", nargs="*", default=["embed"], help="Extra subdirectories of --models-dir to scan for orphan GGUFs (default: embed)")
    parser.add_argument("--spare", help="Additional VRAM to reserve: suffixed (2G, 512m, 64k) or bare number (auto: GB if < 3×VRAM, else MB)")
    parser.add_argument("--vram", help="Total GPU VRAM: suffixed (32G, 24576m) or bare MB (overrides auto-detection)")
    parser.add_argument("--max-context", help="Hard cap on context length for all models (e.g. 128k, 65536)")
    parser.add_argument("--no-env", action="store_true", help="Do not write the sibling config.env file")
    parser.add_argument("--verbose", "-V", action="count", default=0, help="Increase verbosity (-V: info, -VV: debug)")
    return parser.parse_args(argv[1:] if argv else None)


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


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    models_dir = (SCRIPT_DIR / args.models_dir).absolute()
    if not models_dir.is_dir():
        logger.error("models directory not found: %s", models_dir)
        sys.exit(1)

    profiles_path = (SCRIPT_DIR / args.profiles).absolute()
    if not profiles_path.is_file():
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
        bin_dir = find_bin_dir(args.version, SCRIPT_DIR)
        llama_bin = str(SCRIPT_DIR / bin_dir / "llama-server")

    fit_bin = str(SCRIPT_DIR / bin_dir / "llama-fit-params")

    template_vars = {"llama_bin": llama_bin, "models_dir": str(models_dir)}

    max_ctx = None
    if args.max_context:
        from model_cfg.utils import parse_context_length
        max_ctx = parse_context_length(args.max_context)

    # Discover models
    models = Model.from_dir(models_dir, generate_stubs=not args.no_stubs, extra_dirs=args.extra_dirs)
    if not models:
        logger.error("no models found (create a .md sidecar file)")
        sys.exit(1)
    logger.info("models: %d found", len(models))

    # Get VRAM
    if args.vram:
        from model_cfg.utils import resolve_spare_mb
        vram_total = resolve_spare_mb(args.vram, 0)
    else:
        vram_total = get_vram_mb()

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

    # Build config
    config = build_config(models, profiles_cfg, template_vars, fit_bin, vram_total, spare=args.spare, max_context=max_ctx)
    config = _apply_env_subst(config, sub, raw_paths)
    if not config.get("models"):
        logger.error("no model entries generated")
        sys.exit(1)
    logger.info("entries: %d generated", len(config["models"]))

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = SCRIPT_DIR / args.output

    writers = {"yaml": write_yaml, "ini": write_ini}
    writer = writers[args.format]

    if args.dry_run:
        sys.stdout.write(yaml.dump(config, default_flow_style=False, sort_keys=False, allow_unicode=True))
        for _name in sorted(var_to_value):
            logger.info("env %s=%s", _name, var_to_value[_name])
    else:
        writer(config, output_path)
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