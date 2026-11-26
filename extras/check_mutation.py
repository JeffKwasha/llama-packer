#!/usr/bin/env python3
"""Prove each traced requirement fails when its implementation is broken.

    python extras/check_mutation.py

A test that passes whether or not the code is there does not hold the
requirement up -- it only appears to. For every requirement this inverts the
named implementation in a reflinked scratch copy of the tree and asserts the
named tests then FAIL. The working tree is never written, so a crash cannot
leave it modified; the copy is removed on exit.

This is the check that would have caught `met_demands`: removing it left
`require` gating nothing, and no test noticed until I happened to go looking.
"""
import fnmatch
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRACE = ROOT / "docs/plans/requirement-trace.md"
# requirement -> (file, substring[, replacement]) whose removal/replacement
# should break the tests
MUTATIONS = {
    "test_every_category_gets_an_entry": ("llama_packer/writer.py", "for demand in (*self.demands_for(model), fallback):"),
    "test_untagged_gets_a_bare_id": (
        "llama_packer/writer.py",
        'tail = "" if not cats else',
        'tail = "x" if not cats else'),
    "test_rung_drops_vision_when_it_buys_context": ("llama_packer/writer.py", "                best = rung\n"),
    "test_fallback_keeps_capability_until_the_minimum_forces_it": ("llama_packer/writer.py", "                return best\n"),
    "test_require_gates_existence": ("llama_packer/writer.py", "def can_meet(self, model, demand"),
    "test_sidecar_ceiling_binds_in_the_squeeze_branch": ("llama_packer/vram.py", "    def _sidecar_ceiling(self) -> int | None:"),
    "test_solve_context_image_floor_native_vision": ("llama_packer/vram.py", "            image = self.model.image_max_tokens\n"),
    "test_min_image_megapixels_converts_through_geometry": ("llama_packer/model.py", "        return round(mp * _PIXELS_PER_MEGAPIXEL / geo.pixels_per_token)\n"),
    "test_min_image_megapixels_floor_includes_output_reserve": ("llama_packer/vram.py", "        return image + self.model.image_output_reserve_tokens\n"),
    "test_min_image_megapixels_ignored_on_non_proportional": ("llama_packer/model.py", "        if geo is None or not geo.proportional:\n"),
    "test_min_image_megapixels_too_large_warns": ("llama_packer/vram.py", "        if cap is not None and image > cap:\n"),
    "test_min_image_megapixels_keeps_vision": ("llama_packer/writer.py", 'demands = tuple(replace(d, keep=d.keep | {"vision"}) for d in demands)\n'),
    "test_serving_pin_ignores_the_design_max": ("llama_packer/writer.py", "        return min(pins) if pins else None\n"),
    "test_duplicate_ids_do_not_move_when_context_moves": ("llama_packer/writer.py", "        while cand in taken:"),
    "test_fallback_slot_exponent_is_zero": ("llama_packer/writer.py", "power=0.0 if is_null else demand.parallel_power,"),
    "test_capabilities_context_is_served_window": (
        "llama_packer/writer.py", '"context": ctx_size,',
        '"context": context_length,'),
    "test_sidecar_lowers_max_output_tokens": (
        "llama_packer/writer.py", 'ceiling = metadata.get("max_output_tokens")',
        'ceiling = None'),
    "test_max_output_tokens_never_exceeds_the_served_window": (
        "llama_packer/writer.py",
        'metadata["max_output_tokens"] = min(ceiling or ctx_size, ctx_size)',
        'metadata["max_output_tokens"] = ceiling or ctx_size'),
    "test_pack_run_is_quiet_by_default": (
        "llama_packer/__main__.py",
        "floor=logging.INFO if (args.dry_run or args.check) else None",
        "floor=logging.INFO"),
    "test_dry_run_preview_is_visible_without_a_flag": (
        "llama_packer/__main__.py",
        "floor=logging.INFO if (args.dry_run or args.check) else None",
        "floor=None"),
    "test_verbosity_ladder_and_floor": (
        "llama_packer/__main__.py",
        "level = max(0, logging.WARNING - 10 * max(0, verbosity))",
        "level = logging.WARNING"),
    # ── CPU-residency measurement ──
    "test_parse_host_buffers_reads_final_pass": (
        "llama_packer/vram.py",
        "    if final_pass:\n        log_text = _final_pass_scope(log_text)\n"),
    "test_parse_host_buffers_headerless_log_parses_whole": (
        "llama_packer/vram.py",
        "            host[comp] += mib"),
    "test_run_measure_server_cpu_shape_returns_host_buffers": (
        "llama_packer/vram.py",
        '                cmd += ["--n-gpu-layers", "0"]'),
    "test_measurement_shape_stamps_cpu_device": (
        "llama_packer/writer.py",
        '    if model.on_cpu:\n        base = f"{base} --n-gpu-layers 0".strip()\n'),
    "test_run_measure_server_gpu_refuses_vulkanless_fullcpu": (
        "llama_packer/vram.py",
        '                else:\n                    outcome = "spill"'),
    "test_parse_fit_log_host_row_fallback": (
        "llama_packer/vram.py",
        "    row = device_row or next(\n        (m for m in rows if _is_host_report_name(m.group(1))), None)",
        "    row = device_row"),
    "test_measure_affine_trio_cpu_shape_persists_host_quad": (
        "llama_packer/vram.py",
        '        kv_per_token = (r1["kv"] - r3["kv"]) / (design - r3_ctx)'),
    "test_parse_device_buffers_whole_log_device_path": (
        "llama_packer/vram.py",
        ('    return not (dev.endswith("_Host") or dev.startswith("CPU")\n'
         '                or "Mapped" in dev)'),
        '    return True'),
    "test_usage_warnings_lead_with_the_sidecar_name": (
        "llama_packer/tasks.py",
        '    who = f"{stem}: tasks:" if stem else "tasks:"',
        '    who = "tasks:"'),
    "test_warn_unhandled_names_the_sidecar": (
        "llama_packer/backends/base.py",
        '        who = f"{stem}: " if stem else ""',
        '        who = ""'),
    "test_allow_profiles_bad_regex_names_the_sidecar": (
        "llama_packer/profiles.py",
        '"all profiles", stem, allow_profiles)',
        '"all profiles", allow_profiles)'),
    "test_merge_bad_base_expression_names_the_origin": (
        "llama_packer/utils.py",
        'logger.warning("%s: failed to eval %r with base=%s, using base",',
        'logger.warning("failed to eval %r with base=%s, using base",'),
    "test_host_pool_clamps_to_zero": (
        "llama_packer/hardware.py",
        "pool_mb=max(0, pool_mb)",
        "pool_mb=pool_mb"),
    "test_pressure_zone_fallback_is_logged": (
        "llama_packer/hardware.py",
        "           pressure_mb = int((total_mb or 0) * frac)",
        "           pressure_mb = total_mb or 0"),
    "test_cpu_variant_refused_on_small_pool": (
        "llama_packer/writer.py",
        "                    if floor > cap:",
        "                    if False:"),
    "test_cpu_parallel_charge_flips_the_served_context": (
        "llama_packer/writer.py",
        ("        total = (self.host_pool_mb\n"
         "                 if model.on_cpu and self.host_pool_mb else self.vram_total)"),
        "        total = self.vram_total"),
    "test_no_host_pool_never_gates": (
        "llama_packer/writer.py",
        ("        total = (self.host_pool_mb\n"
         "                 if model.on_cpu and self.host_pool_mb else self.vram_total)"),
        ("        total = ((self.host_pool_mb or 0)\n"
         "                 if model.on_cpu else self.vram_total)")),
    "test_metadata_packing_cpu_entry": (
        "llama_packer/writer.py",
        'packing: dict = {"device": "cpu" if model.on_cpu else "gpu"}',
        "packing: dict = {}"),
    "test_cpu_solve_measures_through_allow_cpu": (
        "llama_packer/writer.py",
        ("                llama_args=measurement_args(self.profiles, view,\n"
         "                                            self.llama_args),\n"
         "                allow_cpu=True)"),
        ("                llama_args=measurement_args(self.profiles, view,\n"
         "                                            self.llama_args),\n"
         "                allow_cpu=False)")),
    "test_cpu_chat_entry_inherits_the_global_ttl": (
        "llama_packer/writer.py",
        ("    if role_ttl is not None:\n"
         "        return role_ttl\n"
         "    return None"),
        ("    if role_ttl is not None:\n"
         "        return role_ttl\n"
         "    return 3600 if model.on_cpu else None")),
    "test_gpu_entry_has_no_ttl": (
        "llama_packer/writer.py",
        '    if ttl is not None:\n        entry["ttl"] = ttl',
        '    entry["ttl"] = ttl'),
    "test_host_measurement_does_not_leak_into_vram_pricing": (
        "llama_packer/vram.py",
        '        if getattr(self.model, "on_cpu", False) and not allow_cpu:',
        "        if False:"),
    # ── set composition: pricing, worst set, per-role pin ──
    "test_unknown_device_value_warns": (
        "llama_packer/discover.py",
        "    if problem:",
        "    if False:"),
    "test_role_registry_derives_its_sets": (
        "llama_packer/roles.py",
        "return tuple(ROLES)",
        'return tuple(ROLES) + ("bogus",)'),
    "test_unknown_profiles_key_warns_with_a_hint": (
        "llama_packer/profiles.py",
        "        if key in PROFILES_KEYS:",
        "        if True:"),
    "test_engines_refusal_names_every_engine": (
        "llama_packer/backends/__init__.py",
        ("        reasons[name] = why\n"
         "    return None, reasons"),
        ("        reasons[name] = \"\"\n"
         "    return None, reasons")),
    "test_a_declared_task_id_is_arbitrary": (
        "llama_packer/tasks.py",
        '_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")',
        '_ID_RE = re.compile(r"^[a-z]+$")'),
    "test_resolve_tasks_ignores_unblessed_tag_with_warning": (
        "llama_packer/tasks.py",
        "        if task is None:",
        "        if task is None and False:"),
    "test_registry_docs_agree": (
        "llama_packer/endpoints.py",
        '    "vllm": frozenset({CHAT, EMBEDDINGS, RERANK}),',
        '    "vllm": frozenset({CHAT, EMBEDDINGS}),'),
    "test_sidecar_without_engines_is_refused": (
        "llama_packer/scope.py",
        "        elif not model.stub:",
        "        elif False:"),
    "test_fit_crash_records_argv_and_stderr": (
        "llama_packer/vram.py",
        '            artifact.write_text("\\n".join(header + body) + "\\n")',
        "            pass"),
}


def tests_for(req: str, trace: str) -> list[str]:
    row = re.search(rf"^\| {re.escape(req)} \|(.+)$", trace, re.MULTILINE)
    return re.findall(r"`(test_[A-Za-z0-9_]+)`", row.group(1)) if row else []


#: Entries a test run never reads: VCS metadata, caches, and the dev engine
#: artifacts in a checkout that doubles as a deployment.
SKIP_NAMES = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache",
              ".mypy_cache", "node_modules"}
SKIP_GLOBS = ("llama-swap*", "llama-b*", "*.sqlite", "*.sqlite-*")


def scratch_copy(dest: Path) -> None:
    """Copy the tree with reflinks, so the working tree is never written.

    ``cp --reflink=always`` makes this metadata-only on BTRFS; on a filesystem
    without reflinks it fails and a normal copy is used instead.
    """
    sources = [str(e) for e in ROOT.iterdir()
               if e.name not in SKIP_NAMES
               and not any(fnmatch.fnmatch(e.name, g) for g in SKIP_GLOBS)]
    base = ["cp", "-a", *sources, str(dest)]
    try:
        subprocess.run([base[0], "--reflink=always", *base[1:]], check=True,
                       capture_output=True)
    except subprocess.CalledProcessError:
        shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir()
        subprocess.run(base, check=True)


def test_index(root: Path) -> dict[str, list[str]]:
    """``test name -> files defining it``, read once for the whole run."""
    index: dict[str, list[str]] = {}
    for path in sorted((root / "tests").rglob("test_*.py")):
        try:
            text = path.read_text()
        except OSError:
            continue
        for name in re.findall(r"^def (test_[A-Za-z0-9_]+)\(", text,
                               re.MULTILINE):
            index.setdefault(name, []).append(str(path.relative_to(root)))
    return index


def outcomes(output: str) -> dict[str, set[str]]:
    """Per-test PASSED/FAILED/ERROR from a pytest run."""
    seen: dict[str, set[str]] = {}
    for line in output.splitlines():
        match = re.search(r"::(test_[A-Za-z0-9_]+)(?:\[[^\]]*\])?\s+"
                          r"(PASSED|FAILED|ERROR)", line)
        if match:
            seen.setdefault(match.group(1), set()).add(match.group(2))
    return seen


#: Fixed name, so a copy left by a crashed run is removed by the next one.
#: Dot-prefixed and ``.tmp``-suffixed, which matches the backup and VCS
#: excludes (``.*`` and ``*.tmp``); same filesystem as the repo, so the copy
#: is a reflink rather than a full write.
WORK = ROOT.parent / ".mutation-check.tmp"


def main() -> int:
    trace = TRACE.read_text()
    index = test_index(ROOT)
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir()
    try:
        scratch_copy(WORK)
        return _run(trace, index, WORK)
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


def _run(trace: str, index: dict[str, list[str]], work: Path) -> int:
    bad = 0
    for test_name, (rel, needle, *replacement) in sorted(MUTATIONS.items()):
        row = next((ln for ln in trace.splitlines() if f"`{test_name}`" in ln), "")
        names = [test_name] + [n for n in re.findall(r"`(test_[A-Za-z0-9_]+)`", row) if n != test_name]
        missing = [n for n in names if n not in index]
        if missing:
            print(f"  {test_name} no tests named: {', '.join(missing)}")
            bad += 1
            continue
        original = (ROOT / rel).read_text()
        if needle not in original:
            print(f"  {test_name} MUTATION STALE - anchor not found in {rel}")
            bad += 1
            continue
        (work / rel).write_text(original.replace(
            needle, replacement[0] if replacement else "", 1))
        files = sorted({f for n in names for f in index[n]})
        run = subprocess.run(
            [sys.executable, "-m", "pytest", "-v", "--no-header",
             "--tb=no", "-p", "no:cacheprovider", *files,
             "-k", " or ".join(names)],
            cwd=work, capture_output=True, text=True, check=False)
        # A mutation may also break collection outright; that is still the
        # mutation being detected, so fall back to the exit code.
        seen = outcomes(run.stdout)
        results = []
        for n in names:
            states = seen.get(n, set())
            results.append((n, bool(states & {"FAILED", "ERROR"})
                            if states else run.returncode not in (0, 5)))
        # A requirement is held if *some* named test fails without the code.
        # Naming several tests does not mean they share a code path --
        # --max-context, say, is read both through _serving_pin() and directly
        # in the slot solve -- so demanding that every one of them catch one
        # mutation just measures which accessor got deleted. Per-test detail
        # is printed regardless, so a test that never catches anything stays
        # visible rather than quietly riding along.
        caught = any(ok for _, ok in results)
        print(f"  {test_name[:52]:<52} {'CAUGHT' if caught else 'MISSED'}")
        for n, ok in results:
            if not ok:
                print(f"          - {n}: other code path")
        if not caught:
            bad += 1
    print("\nall requirements hold" if not bad else f"\n{bad} not holding")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
