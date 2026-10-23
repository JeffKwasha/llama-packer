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
