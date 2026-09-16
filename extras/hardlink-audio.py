#!/usr/bin/env python3
"""hardlink-audio — ensure audio.cpp model weights are hard links, not symlinks.

audio.cpp's engine detects a GGUF's tensor-source format from the *resolved*
path extension. HF cache snapshot entries are symlinks into ``blobs/<sha256>``
(extensionless), so the engine follows the link and rejects the file with
``unsupported tensor source format``.  A hard link in the snapshot dir keeps a
real ``.gguf`` path, costs zero extra bytes, and stays inside the repo dir —
so ``hf cache rm <repo>`` still frees the space (verified 2026-09-15).

Scope: only blobs referenced by audio.cpp sidecars in the audio role dirs
(``s2t``/``t2s``) — i.e. sidecars with an ``hf_repo:`` in an audio role.
Whisper ``.bin`` sidecars (no ``hf_repo``) and every other role are untouched.
Non-LFS files (README etc.) are never symlinks, so they're untouched too.

Idempotent: a symlink is converted once (symlink → hard link via tmp + rename);
an already-real file is left alone.  Re-run freely; it is also the fix to apply
after any future ``hf download`` pulls a new revision (new snapshots arrive as
symlinks again).

Usage (from the llama repo root, mirrors llama-packer's CLI):

    ./extras/hardlink-audio.py [--dry-run] [--models-dir DIR ...] [--hf-home DIR]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from llama_packer.discover import discover  # noqa: E402
from llama_packer.utils import hf_hub_cache  # noqa: E402

logger = logging.getLogger("hardlink-audio")

AUDIO_ROLES = frozenset({"s2t", "t2s"})
REPO_ROOT = Path(__file__).resolve().parent.parent


def load_profiles(path: Path) -> dict:
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data if isinstance(data, dict) else {}


def audio_cpp_models(models_dirs, dir_roles, hf_home):
    """Yield Models whose weight is an audio.cpp file (hf_repo + audio role)."""
    for m in discover(models_dirs, generate_stubs=False,
                      dir_roles=dir_roles, hf_home=hf_home):
        if m.role in AUDIO_ROLES and m.hf_repo:
            yield m


def convert(snap_path: Path, dry_run: bool) -> str:
    """Make *snap_path* a real file if it is a symlink into its repo blobs/.

    Returns an action label: already-linked | converted | skipped:<why>.
    """
    if not snap_path.is_symlink():
        if snap_path.is_file():
            return "already-linked"
        return "skipped:resolved-path-missing"
    target = os.path.realpath(snap_path)
    # Safety: only convert links that point into the repo's own blobs/ dir —
    # never a link escaping the cache (that would be someone else's file).
    if "/blobs/" not in target or not Path(target).is_file():
        return f"skipped:target-outside-blobs ({target})"
    if dry_run:
        return "would-convert"
    tmp = snap_path.with_name(snap_path.name + ".tmp")
    try:
        os.link(target, tmp)
        os.replace(tmp, snap_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return "converted"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ensure audio.cpp model weights are hard links, not symlinks.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change; touch nothing")
    parser.add_argument("--models-dir", nargs="+", default=None,
                        help="Model dirs (default: profiles.yaml models_dirs, else ./models)")
    parser.add_argument("--hf-home", default=None,
                        help="HF_HOME root containing hub/ (default: profiles.yaml hf_home, else $HF_HOME)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")
    # discover() walks every models dir and logs pre-existing issues in
    # unrelated (chat) models; this utility only cares about audio, so mute
    # the packer's own logger and let ours carry the signal.
    logging.getLogger("llama_packer").setLevel(logging.CRITICAL)

    profiles = load_profiles(REPO_ROOT / "profiles.yaml")
    models_dirs = [Path(d).absolute() for d in (
        args.models_dir or profiles.get("models_dirs") or ["models"])]
    dir_roles = profiles.get("dirs") or {}
    hf_home = args.hf_home or profiles.get("hf_home")
    hub = hf_hub_cache(hf_home)
    if hub is None:
        logger.error("no HF hub cache found (set --hf-home / $HF_HOME)")
        return 2

    counts: dict[str, int] = {}
    for m in audio_cpp_models(models_dirs, dir_roles, hf_home):
        p = m.gguf_path
        if p is None:
            counts["unresolved"] = counts.get("unresolved", 0) + 1
            logger.error("UNRESOLVED  %s  (%s: %s — hf download it first)",
                         m.label, m.hf_repo, m.frontmatter.get("model"))
            continue
        action = convert(p, args.dry_run)
        counts[action] = counts.get(action, 0) + 1
        logger.info("%-14s  %s  %s", action, m.label, p)

    if not counts:
        logger.warning("no audio.cpp models found in the audio role dirs")
        return 0
    total = sum(counts.values())
    logger.info("%d audio.cpp model(s): %s", total,
                ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
