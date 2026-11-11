# llama_packer/profiles.py
"""profiles.yaml as a value object.

Single home for reading the input config: baseline defaults, sampling-key
expression resolution, ``allow_profiles`` filtering, per-model variant
grouping, and spare-VRAM parsing.  Nothing else should reach into the raw
``defaults`` / ``profiles`` dicts — go through this class so precedence rules
(sidecar > profile > built-in) exist in exactly one place.
"""

from __future__ import annotations


import logging
import re

from llama_packer import roles, utils
from llama_packer.endpoints import EMBEDDINGS, RERANK
from llama_packer.consts import DEFAULT_SACRIFICE_ORDER, _MIN_AGENTIC_CTX

logger = logging.getLogger(__name__)

#: Every top-level ``profiles.yaml`` key something reads.  Unknown keys warn
#: (a typo such as ``device_role`` must not silently do nothing).
PROFILES_KEYS: frozenset[str] = frozenset({
    "defaults", "tasks", "models_dirs", "hf_home", "backends",
    "profiles", "llama_server", "vllm", "sd", "whisper", "audio_cpp",
    "hardware", "overrides", "matrix", "pools", "macros",
    "shed", "capacity", "roles", "ttl", "min_context",
})


#: Retired top-level keys: reported by name, not as unknown typos.
RETIRED_TOP_KEYS = frozenset({"routing", "device_roles"})


def validate_profiles_keys(cfg: dict, log=logger) -> None:
    """Warn on a top-level profiles key no component reads."""
    if not isinstance(cfg, dict):
        return
    for key in cfg:
        if key in PROFILES_KEYS:
            continue
        import difflib
        close = difflib.get_close_matches(str(key), sorted(PROFILES_KEYS),
                                          n=1, cutoff=0.7)
        hint = f" (did you mean {close[0]!r}?)" if close else ""
        log.warning("profiles.yaml: unknown key %r%s; ignored", key, hint)


_PROFILE_KEYS = utils.SAMPLING_KEYS | {
    "cache_type", "parallel", "spare", "batch", "ubatch",
}
_TASK_KEYS = {
    "endpoint", "intent", "device", "min_context", "max_context",
    "min_parallel", "parallel_power", "ttl", "require", "keep", "sacrifice",
    "role",  # retired; accepted with a warning for old files
}
_MATRIX_KEYS = {
    "tools_min_ctx", "auto_parallel_max",
    "vllm_auto_parallel_max", "estimate_headroom", "auto_parallel",
    "parallel_power",
}
_POOL_KEYS = {"vram", "spare"}
_CAPACITY_KEYS = {"pp_tps", "drive_mbps"}
_HARDWARE_KEYS = {
    "vram", "baseline_mb", "unified_system_mb", "gpu_family",
    "memory_margin", "fast_storage", "host_pool", "system_ram",
    "host_pressure", "cgroup_dir",
}
_BACKEND_KEYS = {
    "llama_server": {"args", "batch", "ubatch"},
    "vllm": {"image", "bin", "docker_args", "podman_args", "container_args",
             "container_port", "container_vendor", "gpu_mem_util", "hf_cache",
             "args"},
    "sd": {"bin", "args"},
    "whisper": {"bin", "args"},
    "audio_cpp": {"bin", "backend", "device", "threads", "tmpdir",
                  "max_loaded_models", "idle_unload_ms", "busy_timeout_ms",
                  "min_free_memory_mb", "voice_dir", "model_spec_override",
                  "args"},
}


def _did_you_mean(name: str, known) -> str:
    import difflib
    close = difflib.get_close_matches(str(name), sorted(known), n=1,
                                      cutoff=0.7)
    return f" (did you mean {close[0]!r}?)" if close else ""


def _check_mapping(problems: list[str], label: str, value, *,
                   known: set[str] | frozenset[str] | None = None) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        problems.append(f"{label}: must be a mapping, got "
                        f"{type(value).__name__}")
        return {}
    if known is not None:
        for key in value:
            if key not in known:
                problems.append(f"{label}: unknown key {key!r}"
                                f"{_did_you_mean(key, known)}")
    return value


def check_profiles(cfg: dict) -> list[str]:
    """Return every profiles.yaml problem this validator can name.

    Collects instead of stopping at the first, so a single run explains the
    whole file.  Loading is ``--check``: no model discovery, no server.
    """
    problems: list[str] = []
    if not isinstance(cfg, dict):
        return [f"top level: must be a mapping, got {type(cfg).__name__}"]

    for key in cfg:
        if key in RETIRED_TOP_KEYS:
            continue
        if key not in PROFILES_KEYS:
            problems.append(f"unknown top-level key {key!r}"
                            f"{_did_you_mean(key, PROFILES_KEYS)}")

    profiles = cfg.get("profiles")
    if profiles is None:
        problems.append("profiles: missing; at least one profile is required")
    elif not isinstance(profiles, dict):
        problems.append("profiles: must be a mapping of name to values")
    elif not profiles:
        problems.append("profiles: empty; at least one profile is required")
    else:
        for name, body in profiles.items():
            _check_mapping(problems, f"profiles.{name}", body,
                           known=_PROFILE_KEYS | {"description"})

    _check_mapping(problems, "defaults", cfg.get("defaults"),
                   known=_PROFILE_KEYS)

    role_table, role_problems = roles.parse_roles(cfg)
    problems.extend(role_problems)

    tasks = _check_mapping(problems, "tasks", cfg.get("tasks"))
    for name, body in tasks.items():
        spec = _check_mapping(problems, f"tasks.{name}", body,
                              known=_TASK_KEYS)
        if name and "endpoint" not in spec:
            problems.append(f"tasks.{name}: endpoint is required")
        _device, device_problem = utils.parse_device(spec.get("device"))
        if device_problem:
            problems.append(f"tasks.{name}.device: {spec.get('device')!r} "
                            f"is {device_problem}")

    if cfg.get("models_dirs") is not None:
        roots = cfg["models_dirs"]
        if not isinstance(roots, (list, tuple)):
            problems.append("models_dirs: must be a list of directories")

    if cfg.get("dirs") is not None:
        problems.append("dirs: retired; a role owns its trees with "
                        "roles.<name>.dir")

    backends = cfg.get("backends")
    if backends is not None:
        if not isinstance(backends, (list, tuple)):
            problems.append("backends: must be a list of backend names")
        else:
            from llama_packer.backends import validate_backend_names
            err = validate_backend_names(backends)
            if err:
                problems.append(f"backends: {err}")

    matrix = _check_mapping(problems, "matrix", cfg.get("matrix"),
                            known=_MATRIX_KEYS | {
                                "sets", "evict_costs",
                                "embed_context", "rerank_context"})
    sets = matrix.get("sets")
    if sets is not None:
        if not isinstance(sets, dict):
            problems.append("matrix.sets: must be a mapping of name to "
                            "expression")
        else:
            for name, expr in sets.items():
                if not isinstance(expr, str) or not expr.strip():
                    problems.append(f"matrix.sets.{name}: must be an "
                                    "expression string")
                elif ("__CHAT_VARS__" in expr
                      or "__COLOAD_VARS__" in expr):
                    problems.append(
                        f"matrix.sets.{name}: __CHAT_VARS__/"
                        "__COLOAD_VARS__ are retired; name the roles "
                        "(e.g. 'chat & s2t')")
    if matrix.get("evict_costs"):
        problems.append("matrix.evict_costs: retired; remove it (eviction "
                        "cost is derived)")
    for key in ("embed_context", "rerank_context"):
        if matrix.get(key) is not None:
            problems.append(f"matrix.{key}: retired; set max_context on the "
                            "embeddings/rerank task")
    ttl = cfg.get("ttl")
    if ttl is not None and (isinstance(ttl, bool)
                            or not isinstance(ttl, int) or ttl < 0):
        problems.append("ttl: must be a non-negative integer")

    routing = _check_mapping(problems, "routing", cfg.get("routing"))
    if routing:
        problems.append("routing: retired; remove it (matrix.sets selects "
                        "the router)")

    top_raw = cfg.get("min_context")
    top_min = (_MIN_AGENTIC_CTX if top_raw is None
               else utils.parse_ctx_size(str(top_raw)))
    if top_raw is not None and (top_min is None or top_min < 0):
        problems.append("min_context: must be a non-negative integer")
    for name, body in tasks.items():
        if not isinstance(body, dict):
            continue
        mn_raw = body.get("min_context")
        mx_raw = body.get("max_context")
        mn = (top_min if mn_raw is None
              else utils.parse_ctx_size(str(mn_raw)))
        mx = (None if mx_raw is None
              else utils.parse_ctx_size(str(mx_raw)))
        if mn is not None and mx is not None and mn > mx:
            problems.append(
                f"tasks.{name}: min_context {mn_raw or top_raw} exceeds "
                f"max_context {mx_raw}")

    pools = _check_mapping(problems, "pools", cfg.get("pools"))
    for name, spec in pools.items():
        body = _check_mapping(problems, f"pools.{name}", spec,
                              known=_POOL_KEYS)

    _check_mapping(problems, "capacity", cfg.get("capacity"),
                   known=_CAPACITY_KEYS)
    _check_mapping(problems, "hardware", cfg.get("hardware"),
                   known=_HARDWARE_KEYS)

    if cfg.get("device_roles") is not None:
        problems.append("device_roles: retired; put device: on the role "
                        "(roles.<name>.device: cpu)")

    overrides = cfg.get("overrides")
    if overrides is not None:
        if not isinstance(overrides, (list, tuple)):
            problems.append("overrides: must be a list of rules")
        else:
            for i, rule in enumerate(overrides):
                if not isinstance(rule, dict):
                    problems.append(f"overrides[{i}]: must be a mapping")
                elif "when" not in rule:
                    problems.append(f"overrides[{i}]: when is required")

    for section, known in _BACKEND_KEYS.items():
        _check_mapping(problems, section, cfg.get(section), known=known)

    macros = _check_mapping(problems, "macros", cfg.get("macros"))
    for name, value in macros.items():
        if not isinstance(value, str):
            problems.append(f"macros.{name}: must be a string of flags")

    shed = cfg.get("shed")
    if shed is not None and not isinstance(shed, (list, tuple)):
        problems.append("shed: must be a list of capabilities")

    return problems


def parse_spare_mb(spare_str: str | None, vram_total: int) -> int:
    """Parse a spare string ("4G", "512m", bare MB); empty/None -> 0."""
    return utils.parse_mem_mb(str(spare_str), vram_total) if spare_str else 0


class Profiles:
    """Typed view over the profiles.yaml mapping."""

    def __init__(self, cfg: dict | None):
        self._cfg = cfg or {}
        self.defaults: dict = self._cfg.get("defaults", {}) or {}
        self.profile_list: dict = self._cfg.get("profiles", {}) or {}
        self.tasks_list: dict = self._cfg.get("tasks", {}) or {}
        #: Order in which capabilities a category did not name are shed.
        #: Configuration, not code: what "unnamed" means is the operator's
        #: call, and every category describes itself completely.
        self.shed_order: tuple[str, ...] = tuple(
            self._cfg.get("shed") or ()) or DEFAULT_SACRIFICE_ORDER

    @property
    def tasks_cfg(self) -> dict:
        """Raw ``tasks:`` section — per-task demand overrides.

        The catalogue owns the ids and their defaults; this section states
        only the demand keys the operator changes.  See
        :class:`llama_packer.tasks.TaskCatalogue`.
        """
        return self.tasks_list

    # ── fleet-wide defaults ──

    @property
    def default_cache_type(self) -> str:
        return str(self.defaults.get("cache_type", "q8_0"))

    @property
    def default_parallel(self) -> int:
        return int(self.defaults.get("parallel", 1))

    def global_spare_mb(self, cli_override: str | None = None,
                        vram_total: int = 0) -> int:
        """Fleet-wide spare VRAM in MB: ``defaults.spare`` > ``--spare`` > 0."""
        return parse_spare_mb(self.defaults.get("spare") or cli_override, vram_total)

    def spare_mb(self, preferred: str | None = None, cli_override: str | None = None,
                 vram_total: int = 0) -> int:
        """Spare VRAM in MB for one resolution: *preferred* (profile value)
        > *cli_override* (global ``--spare``) > 0."""
        return parse_spare_mb(preferred or cli_override, vram_total)

    @property
    def capacity_cfg(self) -> dict:
        """Raw ``capacity:`` section — capacity-router cost knobs.

        ``pp_tps`` is the fleet default prompt-processing rate (tokens/s)
        used to price a chat serving's lost prefill in ``evict_cost``;
        a sidecar ``pp_tps:`` overrides it. ``drive_mbps`` overrides the
        detected read speed used to price reload time.
        """
        raw = self._cfg.get("capacity")
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            logger.warning("capacity: section is not a mapping; ignoring")
            return {}
        return raw

    @property
    def pools_cfg(self) -> dict:
        """Raw ``pools:`` section: ``{pool_id: {vram, spare}}`` with
        non-dict garbage dropped (warned, never fatal).

        Pools are VRAM ledgers keyed by id (``default``, ``ram``, or a
        device id). ``vram:`` declares a pool's size (otherwise the detected
        total); ``spare:`` overrides the global spare for that pool.
        Resolution against pool sizes happens in the ledger
        (:class:`writer.PoolLedger`), not here.
        """
        raw = self._cfg.get("pools")
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            logger.warning("pools: section is not a mapping; ignoring")
            return {}
        pools = {}
        for pid, spec in raw.items():
            if not isinstance(spec, dict):
                logger.warning("pools: %r is not a mapping; ignoring", pid)
                continue
            key = {"gpu0": "default", "system": "ram"}.get(str(pid),
                                                           str(pid))
            pools[key] = {
                "vram": spec.get("vram"),
                "spare": spec.get("spare"),
            }
        return pools

    @property
    def llama_server_cfg(self) -> dict:
        """Raw ``llama_server:`` section (``args``, ``batch``, ``ubatch``,
        ``env``).  ``batch``/``ubatch`` are the fleet tier of the named
        batch-key cascade (sidecar > profile > fleet > role)."""
        raw = self._cfg.get("llama_server")
        return dict(raw) if isinstance(raw, dict) else {}

    # ── per-model selection ──

    def matched_for(self, model) -> list[tuple[str, dict]]:
        """(name, resolved-profile) pairs this model allows.

        Honors the model's ``allow_profiles`` regex/list/false gate; each
        profile is layered over ``defaults`` with ``base * N`` expressions
        resolved.
        """
        return [
            (pname, utils.resolve_params(pover, self.defaults))
            for pname, pover in _filter_profiles(self.profile_list,
                                                 model.allow_profiles,
                                                 stem=model.stem)
        ]

    def groups_for(self, model, vram_total: int,
                   spare_override: str | None = None) -> dict[tuple, list[tuple[str, dict]]]:
        """Group the model's allowed profiles by
        (parallel, cache_type, spare_mb, batch, ubatch).

        Each group shares one VRAM solve and one llama-swap entry; the profile
        names within a group become ``setParamsByID`` keys.  ``batch``/``ubatch``
        are part of the key so per-profile values cannot collide inside one
        entry's command (they render per entry and stamp the measurement
        shape); ``None`` is a group key like any other and means "no flag
        rendered".  When no profile matches, a single group derived from
        ``defaults`` is returned.

        Embed/rerank always serve single-slot: their parallel is pinned to 1
        (declared values are ignored with a note) — a fixed-overhead role's
        parallelism must never buy context away from the main chat it serves.
        """
        rag_role = model.serves_endpoint(EMBEDDINGS) or \
            model.serves_endpoint(RERANK)
        groups: dict[tuple, list] = {}
        for pname, resolved in self.matched_for(model):
            parallel = model.parallel_for(resolved.get("parallel", 1))
            if rag_role:
                parallel = _rag_parallel(model, parallel)
            cache_type = model.cache_type_for(
                str(resolved.get("cache_type", self.default_cache_type)))
            spare = self.spare_mb(resolved.get("spare"), spare_override, vram_total)
            batch, ubatch = model.batch_ubatch_for(
                resolved, self.llama_server_cfg)
            groups.setdefault(
                (int(parallel), cache_type, spare, batch, ubatch), []
            ).append((pname, resolved))

        if not groups:
            batch, ubatch = model.batch_ubatch_for(
                None, self.llama_server_cfg)
            groups = {
                (1 if rag_role else model.parallel_for(1),
                 model.cache_type_for(self.default_cache_type),
                 self.global_spare_mb(spare_override, vram_total), batch, ubatch): [
                    ("default", dict(self.defaults)),
                ],
            }
        return groups


# Embed/rerank serve single-slot: their parallel is a chat knob, never a
# fixed-overhead one — a note once per model when a declaration is ignored.
_rag_parallel_warned: set[str] = set()


def _rag_parallel(model, declared: int) -> int:
    """Pin an embeddings/rerank model to parallel 1.

    A fixed-overhead role's parallelism must never buy context away from the
    main chat it serves ("main chat slightly better" beats "embed/rerank
    parallel > 1").
    A declared parallel > 1 is ignored with a once-per-model note.
    """
    if declared == 1:
        return 1
    if model.stem not in _rag_parallel_warned:
        _rag_parallel_warned.add(model.stem)
        logger.warning(
            "%s: embed/rerank serve single-slot — parallel %d ignored "
            "(fixed-overhead parallelism must not shrink the main chat "
            "context it serves)", model.stem, declared)
    return 1


def _filter_profiles(profile_list: dict, allow_profiles,
                     stem: str) -> list[tuple[str, dict]]:
    """Filter profiles according to allow_profiles frontmatter value."""
    if allow_profiles is False:
        return []
    if allow_profiles is None or allow_profiles is True:
        return list(profile_list.items())
    if isinstance(allow_profiles, str):
        try:
            pattern = re.compile(allow_profiles)
        except re.error:
            logger.warning("%s: invalid allow_profiles regex %r, returning "
                           "all profiles", stem, allow_profiles)
            return list(profile_list.items())
        return [(pname, pover) for pname, pover in profile_list.items()
                if pattern.search(pname)]
    return list(profile_list.items())
