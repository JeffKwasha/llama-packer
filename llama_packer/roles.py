"""Roles: operator-defined purposes and allocation policies.

A role is a label, a behaviour kind, and a memory policy. The code ships no
role names: the packaged ``profiles.yaml`` provides the default set as data,
and ``configure()`` replaces it from the operator's file.

``kind`` selects the behaviour implementation (chat solve, helper context,
media placement, audio). It is data, so a role may rename or duplicate a kind
without changing the code. Everything else is allocation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from llama_packer.utils import parse_device

log = logging.getLogger(__name__)

_ROLE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
LIVE_VALUES = frozenset({"1", "per-device", "capacity"})
KINDS = frozenset({"chat", "embeddings", "rerank", "image", "s2t", "t2s"})
#: Old role keys. A file that still sets them is told they do nothing.
RETIRED_KEYS = frozenset({"min", "preferred", "cap", "weight", "exclusive",
                          "persistent", "ctx_cap"})


@dataclass(frozen=True)
class Role:
    """One operator-defined purpose."""

    name: str
    #: Models-root-relative trees this role owns; ``"."`` is the root.
    dir: tuple[str, ...] = ()
    kind: str = "chat"
    #: Fraction of slack this role may claim; None when it fills instead.
    share: float | None = None
    fill: bool = False
    live: str = "1"
    #: Where every model of this role runs: ``cpu`` or a GPU index; None
    #: leaves placement to the sidecar/task.  Same vocabulary as sidecars.
    device: str | int | None = None


def _parse_share(value: object) -> float | None:
    """``40%`` / ``40`` / ``0.4`` → 0.4; None when unreadable."""
    if value is None:
        return None
    text = str(value).strip()
    try:
        if text.endswith("%"):
            return float(text[:-1]) / 100.0
        share = float(text)
    except ValueError:
        return None
    return share / 100.0 if share > 1.0 else share


def _default_specs() -> dict:
    """The packaged default role table, as data."""
    try:
        import importlib.resources

        import yaml

        path = importlib.resources.files("llama_packer").joinpath(
            "profiles.yaml")
        cfg = yaml.safe_load(path.read_text()) or {}
        return cfg.get("roles") or {}
    except Exception:
        return {}


def _parse_dirs(name: str, value: object) -> tuple[tuple[str, ...], list[str]]:
    """``dir:`` is one path or a list of them, relative to the models root."""
    if value is None:
        return (), []
    raw = value if isinstance(value, (list, tuple)) else [value]
    out: list[str] = []
    problems: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            problems.append(f"roles.{name}: dir entries must be paths")
            continue
        d = item.strip().replace("\\", "/")
        while d.startswith("./"):
            d = d[2:]
        d = d.rstrip("/") or "."
        if d.startswith("/") or ".." in d.split("/"):
            problems.append(f"roles.{name}: dir {item!r} must be relative "
                            "to the models root")
            continue
        out.append(d.lower())
    return tuple(dict.fromkeys(out)), problems


def _overlaps(a: str, b: str) -> bool:
    """Two owned trees overlap when one is the other or contains it."""
    return (a == b or a == "." or b == "."
            or a.startswith(b + "/") or b.startswith(a + "/"))


def _parse_one(name: object, spec: object,
               ) -> tuple[Role | None, list[str]]:
    if not isinstance(name, str) or not _ROLE_ID.match(name.strip()):
        return None, [f"roles.{name}: not a role id"]
    name = name.strip()
    if name == "ram" or name == "default" or re.fullmatch(r"gpu\d+", name):
        return None, [f"roles.{name}: reserved pool name"]
    if not isinstance(spec, dict):
        return None, [f"roles.{name}: must be a mapping"]
    known = {"dir", "kind", "share", "fill", "live",
             "device"} | RETIRED_KEYS
    problems = [f"roles.{name}: unknown key {key!r}"
                for key in sorted(set(spec) - known)]
    problems.extend(f"roles.{name}: {key!r} is retired"
                    for key in sorted(set(spec) & RETIRED_KEYS))
    dirs, dir_problems = _parse_dirs(name, spec.get("dir"))
    problems.extend(dir_problems)
    kind = str(spec.get("kind", "chat")).strip().lower()
    if kind not in KINDS:
        problems.append(f"roles.{name}: kind={kind!r} is not one of "
                        f"{', '.join(sorted(KINDS))}")
        kind = "chat"
    live = str(spec.get("live", "1")).strip().lower()
    if live not in LIVE_VALUES:
        problems.append(f"roles.{name}: live={spec.get('live')!r} is not one "
                        f"of {', '.join(sorted(LIVE_VALUES))}")
        live = "1"
    share = _parse_share(spec.get("share"))
    if spec.get("share") is not None and (
            share is None or not 0.0 < share <= 1.0):
        problems.append(f"roles.{name}: share={spec.get('share')!r} is not a "
                        "percentage in (0, 100]")
        share = None
    if share is not None and spec.get("fill"):
        problems.append(f"roles.{name}: share and fill are mutually exclusive")
        share = None
    device, device_problem = parse_device(spec.get("device"))
    if device_problem:
        problems.append(f"roles.{name}: device={spec.get('device')!r} is "
                        f"{device_problem}")
        device = None
    return Role(
        name=name,
        dir=dirs,
        kind=kind,
        share=share,
        fill=bool(spec.get("fill", False)),
        live=live,
        device=device,
    ), problems


def parse_roles(cfg: dict | None) -> tuple[dict[str, Role], list[str]]:
    """Parse ``profiles.yaml roles:``; never raises."""
    raw = (cfg or {}).get("roles")
    if raw is None:
        raw = _default_specs()
    if not isinstance(raw, dict):
        return {}, ["roles: must be a mapping of name to spec"]
    out: dict[str, Role] = {}
    problems: list[str] = []
    for name, spec in raw.items():
        role, issues = _parse_one(name, spec)
        problems.extend(issues)
        if role is not None:
            out[role.name] = role
    fills = [r.name for r in out.values() if r.fill]
    if len(fills) > 1:
        problems.append(f"roles: {', '.join(sorted(fills))} all fill; at most "
                        "one role may")
    total = sum(r.share or 0.0 for r in out.values())
    if total > 1.0 + 1e-9:
        problems.append(f"roles: shares sum to {total:.0%}; at most 100%")
    owned: dict[str, str] = {}
    for role in out.values():
        for d in role.dir:
            for other_dir, other in owned.items():
                if other == role.name:
                    continue
                if _overlaps(d, other_dir):
                    problems.append(
                        f"roles.{role.name} and roles.{other} both own "
                        f"{d!r}/{other_dir!r}; a tree belongs to one role")
            owned[d] = role.name
    return out, problems


ROLES: dict[str, Role] = {}
BY_NAME: dict[str, Role] = ROLES


def configure(cfg: dict | None) -> None:
    """Replace the active role table from ``profiles.yaml roles:``."""
    global ROLES, BY_NAME
    roles, problems = parse_roles(cfg)
    for problem in problems:
        log.warning("profiles.yaml: %s; ignored", problem)
    ROLES = roles
    BY_NAME = ROLES


def names() -> tuple[str, ...]:
    return tuple(ROLES)


def non_chat() -> frozenset[str]:
    return frozenset(n for n, r in ROLES.items() if r.kind != "chat")


def get(name: object) -> Role | None:
    return ROLES.get(name) if isinstance(name, str) else None


def base_of(name: str) -> str:
    """Behaviour kind for *name* (itself when unknown)."""
    role = ROLES.get(name)
    return role.kind if role is not None else name


def _key(rel_parts) -> str:
    parts = [str(p).lower() for p in rel_parts]
    return "/".join(parts) or "."


def owner(rel_parts) -> str | None:
    """Role owning the models-root-relative directory *rel_parts*."""
    key = _key(rel_parts)
    best: tuple[int, str] | None = None
    for role in ROLES.values():
        for d in role.dir:
            if d == "." or key == d or key.startswith(d + "/"):
                depth = 0 if d == "." else d.count("/") + 1
                if best is None or depth > best[0]:
                    best = (depth, role.name)
    return best[1] if best else None


def covers(rel_parts) -> bool:
    """True when some role owns this directory or one below it."""
    key = _key(rel_parts)
    for role in ROLES.values():
        for d in role.dir:
            if (d == "." or key == "."
                    or key == d or key.startswith(d + "/")
                    or d.startswith(key + "/")):
                return True
    return False


configure(None)
