"""Roles: operator-defined purposes and allocation policies.

A role is a label, the trees it owns, and a memory policy. The code ships no
role names: the packaged ``profiles.yaml`` provides the default set as data,
and ``configure()`` replaces it from the operator's file.

A role has no behaviour. A model declares ``tasks:``; each task names the
endpoint it is served at, and that endpoint selects the implementation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from llama_packer.utils import parse_device

log = logging.getLogger(__name__)

_ROLE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
#: Old role keys. A file that still sets them is told they do nothing.
RETIRED_KEYS = frozenset({"min", "preferred", "cap", "weight", "exclusive",
                          "persistent", "ctx_cap", "kind"})


@dataclass(frozen=True)
class Share:
    """A role's request in a set's auction."""

    #: ``weight``, ``fill``, ``min``, or ``zero``.
    kind: str = "zero"
    weight: float = 0.0

    @property
    def requests(self) -> bool:
        return self.kind in ("weight", "fill")

    @property
    def accepts(self) -> bool:
        return self.kind != "min"


@dataclass(frozen=True)
class Role:
    """One operator-defined purpose."""

    name: str
    #: Models-root-relative trees this role owns; ``"."`` is the root.
    dir: tuple[str, ...] = ()
    #: Request weight in each set's auction.
    share: Share = Share()
    #: Where every model of this role runs: ``cpu`` or a GPU index; None
    #: leaves placement to the sidecar/task.  Same vocabulary as sidecars.
    device: str | int | None = None
    #: Idle-unload lifetime for every serving of the role; None inherits the
    #: top-level ``ttl``.
    ttl: int | None = None


def _parse_share(value: object) -> Share | None:
    """``40``, ``40%``, ``fill``, ``min``, or ``0``; None when unreadable."""
    if value is None:
        return Share()
    if isinstance(value, bool):
        return None
    text = str(value).strip().lower()
    if text == "fill":
        return Share("fill", 100.0)
    if text == "min":
        return Share("min", 0.0)
    try:
        weight = float(text[:-1]) if text.endswith("%") else float(text)
    except ValueError:
        return None
    if weight < 0:
        return None
    if weight == 0.0:
        return Share("zero", 0.0)
    return Share("weight", weight)


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
    known = {"dir", "share", "device", "ttl"} | RETIRED_KEYS
    problems = [f"roles.{name}: unknown key {key!r}"
                for key in sorted(set(spec) - known)]
    problems.extend(f"roles.{name}: {key!r} is retired"
                    for key in sorted(set(spec) & RETIRED_KEYS))
    dirs, dir_problems = _parse_dirs(name, spec.get("dir"))
    problems.extend(dir_problems)
    share = _parse_share(spec.get("share"))
    if share is None:
        problems.append(f"roles.{name}: share={spec.get('share')!r} is not a "
                        "weight, 'fill', 'min', or 0")
        share = Share()
    device, device_problem = parse_device(spec.get("device"))
    if device_problem:
        problems.append(f"roles.{name}: device={spec.get('device')!r} is "
                        f"{device_problem}")
        device = None
    ttl = spec.get("ttl")
    if ttl is not None and (isinstance(ttl, bool)
                            or not isinstance(ttl, int) or ttl < 0):
        problems.append(f"roles.{name}: ttl={ttl!r} is not a non-negative "
                        "integer")
        ttl = None
    return Role(name=name, dir=dirs, share=share, device=device,
                ttl=ttl), problems


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


def get(name: object) -> Role | None:
    return ROLES.get(name) if isinstance(name, str) else None


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
