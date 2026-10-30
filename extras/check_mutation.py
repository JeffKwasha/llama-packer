#!/usr/bin/env python3
"""Prove each traced requirement fails when its implementation is broken.

    python extras/check_mutation.py

A test that passes whether or not the code is there does not hold the
requirement up -- it only appears to. For every requirement this inverts the
named implementation in a scratch copy of the tree and asserts the named
tests then FAIL. Restored afterwards, so the working tree is untouched.

This is the check that would have caught `met_demands`: removing it left
`require` gating nothing, and no test noticed until I happened to go looking.
"""
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRACE = ROOT / "docs/plans/requirement-trace.md"
# requirement -> (file, substring[, replacement]) whose removal/replacement
# should break the tests
MUTATIONS = {
    "test_every_category_gets_an_entry": ("llama_packer/writer.py", "for demand in (*self.demands_for(model), FALLBACK):"),
    "test_untagged_gets_a_bare_id": ("llama_packer/writer.py", "if demand is FALLBACK:\n            self.set_admission"),
    "test_rung_drops_vision_when_it_buys_context": ("llama_packer/writer.py", "                best = rung\n"),
    "test_fallback_keeps_capability_until_the_minimum_forces_it": ("llama_packer/writer.py", "                return best\n"),
    "test_require_gates_existence": ("llama_packer/writer.py", "def can_meet(self, model, demand"),
    "test_sidecar_ceiling_binds_in_the_squeeze_branch": ("llama_packer/vram.py", "    def _sidecar_ceiling(self) -> int | None:"),
    "test_calc_ctx_image_floor_native_vision": ("llama_packer/vram.py", "            image = self.model.image_max_tokens\n"),
    "test_min_image_megapixels_converts_through_geometry": ("llama_packer/model.py", "        return round(mp * _PIXELS_PER_MEGAPIXEL / geo.pixels_per_token)\n"),
    "test_min_image_megapixels_floor_includes_output_reserve": ("llama_packer/vram.py", "        return image + self.model.image_output_reserve_tokens\n"),
    "test_min_image_megapixels_ignored_on_non_proportional": ("llama_packer/model.py", "        if geo is None or not geo.proportional:\n"),
    "test_min_image_megapixels_too_large_warns": ("llama_packer/vram.py", "        if cap is not None and image > cap:\n"),
    "test_min_image_megapixels_keeps_vision": ("llama_packer/writer.py", 'demands = tuple(replace(d, keep=d.keep | {"vision"}) for d in demands)\n'),
    "test_serving_pin_ignores_the_design_max": ("llama_packer/writer.py", "        return min(pins) if pins else None\n"),
    "test_duplicate_ids_do_not_move_when_context_moves": ("llama_packer/writer.py", 'tail = ".".join(x for x in (cat_id, flags) if x)'),
    "test_set_admission_is_per_entry_not_per_stem": ("llama_packer/writer.py", "def record_admission(self, model, demand"),
    "test_fallback_slot_exponent_is_zero": ("llama_packer/writer.py", "power=0.0 if is_null else None,"),
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
        "floor=logging.INFO if args.dry_run else None",
        "floor=logging.INFO"),
    "test_dry_run_preview_is_visible_without_a_flag": (
        "llama_packer/__main__.py",
        "floor=logging.INFO if args.dry_run else None",
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
        "llama_packer/writer.py",
        '    who = f"{stem}: usage:" if stem else "usage:"',
        '    who = "usage:"'),
    "test_warn_same_role_names_the_pair": (
        "llama_packer/__main__.py",
        "                if len(names) < 2:",
        "                if False:"),
    "test_resident_options_branch_alternatives_not_co_resident": (
        "llama_packer/writer.py",
        '                elif op == "&":',
        "                elif False:"),
    "test_prune_leaves_removes_refused_members": (
        "llama_packer/writer.py",
        'kept.append(pruned if other not in pruned',
        "                                    kept.append(pruned if other in pruned"),
    "test_drop_refused_categories_prunes_sets_and_pins": (
        "llama_packer/__main__.py",
        "    return kept, [p for p in cpu_pins if p not in refused]",
        "    return kept, list(cpu_pins)"),
    "test_category_ctx_caps_at_one_seam": (
        "llama_packer/writer.py",
        "    ctx = (model.design_context if model.role not in (",
        "    ctx = (model.design_context if True not in ("),
    "test_host_unit_refusals_drop_the_unit": (
        "llama_packer/writer.py",
        "            if available < total:",
        "            if False:"),
    "test_host_unit_refusals_survive_one_fitting_set": (
        "llama_packer/writer.py",
        "                if all(s in fails for s in named):",
        "                if True:"),
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
        ("                    host_free = self.host_ledger.available() \\\n"
         "                        if serve_view.on_cpu else None"),
        "                    host_free = None"),
    "test_cpu_parallel_charge_flips_refusal": (
        "llama_packer/writer.py",
        ("                    host_free = self.host_ledger.available() \\\n"
         "                        if serve_view.on_cpu else None"),
        "                    host_free = None"),
    "test_cpu_resident_charges_host_not_vram": (
        "llama_packer/writer.py",
        "mb = self._host_resident_mb(model, ctx)",
        "mb = self._resident_overhead(model, ctx)"),
    "test_no_host_pool_never_gates": (
        "llama_packer/writer.py",
        "self.host_ledger = HostLedger(host_pool_mb)",
        "self.host_ledger = HostLedger(host_pool_mb or 0)"),
    "test_metadata_packing_cpu_entry": (
        "llama_packer/writer.py",
        'packing: dict = {"device": "cpu" if model.on_cpu else "gpu"}',
        "packing: dict = {}"),
    "test_host_resident_sizing_lifts_cpu_gate": (
        "llama_packer/writer.py",
        "                          allow_cpu=allow_cpu)",
        "                          allow_cpu=False)"),
    "test_select_model_prefers_cpu_candidate": (
        "llama_packer/__main__.py",
        'cpu = [m for m in cands if getattr(m, "on_cpu", False)]',
        "cpu = []"),
    "test_select_model_smallest_among_cpu_candidates": (
        "llama_packer/__main__.py",
        "    return min(pool, key=lambda m: m.vram_mb)",
        "    return min(pool, key=lambda m: -m.vram_mb)"),
    "test_select_model_no_cpu_keeps_size_rule": (
        "llama_packer/__main__.py",
        "    return min(pool, key=lambda m: m.vram_mb)",
        "    return pool[0]"),
    "test_select_model_selector_overrides_cpu_preference": (
        "llama_packer/__main__.py",
        "        return hits[0]",
        "        return min(pool, key=lambda m: m.vram_mb)"),
    "test_selection_reason_names_the_arm": (
        "llama_packer/__main__.py",
        '        return "device: cpu — explicit placement"',
        '        return "explicit placement"'),
    "test_device_roles_place_every_role_model": (
        "llama_packer/__main__.py",
        '            m.frontmatter["device"] = "cpu"',
        '            m.frontmatter["device"] = "gpu"'),
    "test_device_roles_absent_and_unmatched": (
        "llama_packer/__main__.py",
        '            logger.warning("device_roles: role %r matched no model", role)',
        '            pass'),
    "test_device_roles_rejects_unknown_role": (
        "llama_packer/__main__.py",
        '        if role not in roles.DEVICE_ROLES:',
        '        if False:'),
    "test_device_roles_rejects_a_non_cpu_value": (
        "llama_packer/__main__.py",
        '            if str(v).strip().lower() == "cpu":',
        "            if True:"),
    "test_device_roles_rejects_a_scalar": (
        "llama_packer/__main__.py",
        "    elif isinstance(roles_cfg, (list, tuple)):",
        "    elif True:"),
    "test_cpu_helpers_pinned_into_every_set": (
        "llama_packer/__main__.py",
        '            expr = f"{expr.strip()} & {name}"',
        '            expr = expr.strip()'),
    "test_cpu_entry_ttl_zero": (
        "llama_packer/writer.py",
        "    if model.on_cpu:\n        entry[\"ttl\"] = 0",
        "    if model.on_cpu:\n        pass"),
    "test_gpu_entry_has_no_ttl": (
        "llama_packer/writer.py",
        "    if model.on_cpu:\n        entry[\"ttl\"] = 0",
        '    if model.on_cpu: \n        entry["ttl"] = 0\n    entry["ttl"] = 0'),
    "test_select_model_picks_measured_cheapest_over_file_size": (
        "llama_packer/__main__.py",
        "            return min(priced, key=lambda t: t[1])[0]",
        "            return priced[0][0]"),
    "test_select_model_falls_back_to_size_when_unmeasurable": (
        "llama_packer/__main__.py",
        "    return min(pool, key=lambda m: m.vram_mb)",
        "    return pool[0]"),
    "test_select_model_cpu_preference_beats_cheaper_device": (
        "llama_packer/__main__.py",
        "    pool = cpu or cands",
        "    pool = cands"),
    "test_select_model_selector_prices_only_the_pick": (
        "llama_packer/__main__.py",
        "        if cost is not None and logger is not None:",
        "        if False:"),
    "test_detect_matrix_category_selector_defines_the_pair": (
        "llama_packer/__main__.py",
        '        embed_model = categories["emb"]',
        "        embed_model = embed_model"),
    "test_cpu_fixed_category_resident_charged_host": (
        "llama_packer/writer.py",
        "                if model is None or not model.on_cpu:\n"
        "                    continue\n"
        "                if name in self.host_unit_refused_vars:",
        "                if True:\n"
        "                    continue\n"
        "                if name in self.host_unit_refused_vars:"),
    "test_host_measurement_does_not_leak_into_vram_pricing": (
        "llama_packer/vram.py",
        '        if getattr(self.model, "on_cpu", False) and not allow_cpu:',
        "        if False:"),
    # ── set composition: pricing, worst set, per-role pin ──
    "test_set_overhead_prices_nested_branches": (
        "llama_packer/writer.py",
        '    alternatives = _split_top(expr, "|")\n'
        "    if len(alternatives) > 1:",
        '    alternatives = [expr]\n'
        "    if len(alternatives) > 1:"),
    "test_solve_matrix_prices_a_nested_and_branch": (
        "llama_packer/writer.py",
        '    alternatives = _split_top(expr, "|")\n'
        "    if len(alternatives) > 1:",
        '    alternatives = [expr]\n'
        "    if len(alternatives) > 1:"),
    "test_set_overhead_expands_named_sets": (
        "llama_packer/writer.py",
        "        if sets_cfg and name in sets_cfg and name not in _seen:\n"
        "            return set_overhead_mb(",
        "        if False:\n"
        "            return set_overhead_mb("),
    "test_expr_leaves_walks_nested_branches": (
        "llama_packer/writer.py",
        "        for part in parts:\n            out |= _expr_leaves(part, sets_cfg, _seen)",
        "        return set(parts)"),
    "test_solve_matrix_reserves_worst_set_not_category_union": (
        "llama_packer/writer.py",
        "        return max(_set_costs(embed_ctx_, rerank_ctx_, extra_mb).values(),\n"
        "                   default=0)",
        "        return sum(_set_costs(embed_ctx_, rerank_ctx_, extra_mb).values())"),
    "test_solve_matrix_unreferenced_category_reserves_nothing": (
        "llama_packer/writer.py",
        "        return {name: set_overhead_mb(expr, var_mb, frozenset(), frozenset(),\n"
        "                                      sets_cfg=raw_sets, coload_mb=coload_mb)\n"
        "                for name, expr in chat_sets.items()}",
        "        return {name: set_overhead_mb(expr, var_mb, frozenset(), frozenset(),\n"
        "                                      sets_cfg=raw_sets, coload_mb=coload_mb)\n"
        "                       + sum(v for n, v in fixed_mb.items()\n"
        "                             if n not in _expr_leaves(expr, raw_sets))\n"
        "                for name, expr in chat_sets.items()}"),
    "test_unknown_role_value_warns_with_a_hint": (
        "llama_packer/discover.py",
        '    if role and role not in utils.SERVED_ROLES:',
        "    if False:"),
    "test_unknown_device_value_warns": (
        "llama_packer/discover.py",
        "    if problem:",
        "    if False:"),
    "test_role_registry_derives_its_sets": (
        "llama_packer/roles.py",
        "if r.device_role)",
        "if True)"),
    "test_every_task_maps_to_a_served_endpoint": (
        "llama_packer/endpoints.py",
        '    "s2t": TRANSCRIPTIONS,',
        '    "s2t": "openai.nonexistent",'),
    "test_unknown_profiles_key_warns_with_a_hint": (
        "llama_packer/profiles.py",
        "        if key in PROFILES_KEYS:",
        "        if True:"),
    "test_cpu_pin_is_one_canonical_category_per_role": (
        "llama_packer/__main__.py",
        '        if role not in canonical or name in ("emb", "rnk"):',
        "        if True:"),
    "test_cpu_pin_skips_a_set_that_chose_the_role": (
        "llama_packer/__main__.py",
        "            if role and any(role_of.get(leaf) == role for leaf in leaves):\n"
        "                continue",
        "            if False:\n                continue"),
    "test_selection_reason_names_the_cost_terms": (
        "llama_packer/__main__.py",
        '    suffix = f" = {detail}" if detail else ""',
        '    suffix = ""'),
    "test_resident_cost_detail_names_the_terms": (
        "llama_packer/writer.py",
        "    kv_mb = int(kv_factor * ctx + slot_mib)",
        "    kv_mb = 0"),
}


def tests_for(req: int, trace: str) -> list[str]:
    row = re.search(rf"^\| {req} \|(.+)$", trace, re.MULTILINE)
    return re.findall(r"`(test_[A-Za-z0-9_]+)`", row.group(1)) if row else []


def main() -> int:
    trace = TRACE.read_text()
    bad = 0
    for test_name, (rel, needle, *replacement) in sorted(MUTATIONS.items()):
        row = next((ln for ln in trace.splitlines() if f"`{test_name}`" in ln), "")
        names = [test_name] + [n for n in re.findall(r"`(test_[A-Za-z0-9_]+)`", row) if n != test_name]
        if not names:
            print(f"  {test_name} no tests named")
            bad += 1
            continue
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "tree"
            shutil.copytree(ROOT, work, ignore=shutil.ignore_patterns(
                ".git", "__pycache__", ".venv", "*.pyc", ".ruff_cache", ".pytest_cache", ".claude", "node_modules"))
            target = work / rel
            text = target.read_text()
            if needle not in text:
                print(f"  {test_name} MUTATION STALE - anchor not found in {rel}")
                bad += 1
                continue
            target.write_text(text.replace(needle, replacement[0] if replacement else "", 1))
            # Map each test name to the file that defines it.
            files = set()
            for n in names:
                hit = subprocess.run(
                    ["grep", "-rlE", rf"^def {n}\(", "tests"],
                    cwd=work, capture_output=True, text=True, check=False).stdout.split()
                if not hit:
                    print(f"          - {n} not found in tests/")
                    bad += 1
                    break
                files.update(hit)
            else:
                # Each named test is run on its own: a mutation is only
                # caught if *that* test fails, not if some sibling in the
                # same file happens to. A rule with two tests where only
                # one holds it is a rule half-tested.
                results = []
                for n in names:
                    f = subprocess.run(
                        ["grep", "-rlE", rf"^def {n}\(", "tests"],
                        cwd=work, capture_output=True,
                        text=True, check=False).stdout.split()
                    r = subprocess.run(
                        ["python", "-m", "pytest", "-q", "-p",
                         "no:cacheprovider", *f, "-k", n],
                        cwd=work, capture_output=True, text=True,
                        check=False)
                    # returncode is the reliable signal: 0 = passed, 5 = no
                    # tests collected, anything else means the test (or its
                    # collection) failed. Matching words in the last line
                    # missed an IndentationError, whose usual summary line
                    # never arrives -- the traceback is the last thing on
                    # stdout, and "IndentationError" is not the string
                    # "error".
                    results.append((n, r.returncode not in (0, 5)))
                # A requirement is held if *some* named test fails without
                # the code. Naming several tests does not mean they share a
                # code path -- --max-context, say, is read both through
                # _serving_pin() and directly in the slot solve -- so
                # demanding that every one of them catch one mutation just
                # measures which accessor got deleted. Per-test detail is
                # printed regardless, so a test that never catches anything
                # stays visible rather than quietly riding along.
                caught = any(ok for _, ok in results)
                print(f"  {test_name[:52]:<52} "
                      f"{'CAUGHT' if caught else 'MISSED'}")
                for n, ok in results:
                    if not ok:
                        print(f"          - {n}: other code path")
                if not caught:
                    bad += 1
    print("\nall requirements hold" if not bad else f"\n{bad} not holding")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
