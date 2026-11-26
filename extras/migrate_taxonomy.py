#!/usr/bin/env python3
"""One-time migration from numbered requirements to named taxonomy IDs.

    python extras/migrate_taxonomy.py

Idempotent: once the registry has no numbered rows it exits without changes.
The number -> name map lives here, never in the registry or the docs.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRACE = ROOT / "docs/plans/requirement-trace.md"
TESTS = ROOT / "tests"

NAMES = {
    1: "FLOW_TASK_DECLARED_ONLY", 2: "FLOW_TASK_UNKNOWN_WARNS_IGNORED",
    3: "FLOW_TASK_EMITS_ONE_ENTRY", 4: "FLOW_TASK_UNSERVABLE_EMITS_NOTHING",
    5: "FLOW_MODEL_FALLBACK_ENTRY", 6: "FLOW_FALLBACK_CAN_EMIT_NOTHING",
    7: "FLOW_VISION_SHED_BUYS_CONTEXT", 8: "FLOW_VISION_KEPT_WHEN_FREE",
    9: "FLOW_INFEASIBLE_RUNG_SKIPPED", 10: "FLOW_CATEGORY_MAX_CONTEXT_RUNG",
    11: "FLOW_FALLBACK_CAPABILITY_FIRST",
    12: "FLOW_FALLBACK_SHEDS_ONLY_WHEN_FORCED",
    13: "FLOW_SACRIFICE_ORDER_DRIVES_LADDER",
    14: "FLOW_UNNAMED_CAPABILITIES_SHED_LAST",
    15: "FLOW_REQUIRED_CAPABILITY_NEVER_SHED",
    16: "FLOW_REQUIRED_CAPABILITY_GATES_EXISTENCE",
    17: "FLOW_KEEP_FORBIDS_SHED_ONLY",
    18: "FLOW_KEEP_ENFORCED_ON_DRAFT_AXIS",
    19: "FLOW_CONTEXT_LENGTH_IS_CEILING",
    20: "FLOW_NATIVE_VISION_IMAGE_FLOOR",
    21: "FLOW_MAX_CONTEXT_ONLY_PIN",
    22: "FLOW_CATEGORY_BEATS_TOOLS_HEURISTIC",
    23: "FLOW_OPERATOR_FLOORS_OUTRANK_CATEGORY",
    24: "FLOW_MEGAPIXEL_CONVERTS_THROUGH_GEOMETRY",
    25: "FLOW_IMAGE_FLOOR_INCLUDES_OUTPUT_RESERVE",
    26: "FLOW_NONPROPORTIONAL_PROJECTOR_WARNED",
    27: "FLOW_IMAGE_BUDGET_CLAMPED_WITH_WARNING",
    28: "FLOW_ZERO_IMAGE_BUDGET_DROPS_REQUIREMENT",
    29: "FLOW_IMAGE_BUDGET_KEEPS_VISION",
    30: "FLOW_NO_BUDGET_LETS_VISION_SHED",
    31: "FLOW_ENTRY_ID_IGNORES_CONTEXT",
    32: "FLOW_UNTAGGED_BARE_STEM_ID",
    33: "FLOW_UNSLUGGABLE_NAME_DROPS_SEGMENT",
    34: "FLOW_ENTRY_ID_STABLE_ACROSS_REPACKS",
    35: "FLOW_NO_ENTRY_EXCEEDS_VRAM",
    36: "FLOW_OVERALLOCATION_GUARD_QUIET_WHEN_SANE",
    37: "FLOW_CPU_CHAT_SLOTS_USE_HOST_BUDGET",
    38: "FLOW_TOOLS_DEMOTED_ON_OWN_CONTEXT",
    39: "FLOW_FIXED_SERVING_TASK_CONTEXT_CAP",
    40: "FLOW_FIXED_ROLE_PARALLEL_IGNORED",
    43: "FLOW_FALLBACK_SLOT_EXPONENT_ZERO",
    44: "FLOW_TIE_PREFERS_MORE_SLOTS",
    45: "DEPLOY_INSTALL_ATOMIC",
    46: "DEPLOY_PRUNE_KEEPS_SYMLINK_TARGET",
    47: "DEPLOY_AMBIGUOUS_SELECTOR_EXITS_NONZERO",
    48: "DEPLOY_CPU_CUDA_NEVER_VULKAN",
    49: "DEPLOY_VALIDATE_AS_SERVICE_USER",
    50: "DEPLOY_UNIT_VERIFIED_BEFORE_INSTALL",
    51: "DEPLOY_UNREADABLE_HF_HOME_WARNS",
    52: "DEPLOY_UNIT_IS_ENABLEABLE",
    53: "DEPLOY_INACCESSIBLE_PATHS_EXIST_ONLY",
    54: "DEPLOY_PROBE_WAITS_FOR_SOCKET",
    55: "DEPLOY_PROBE_DEADLINE_WITH_JOURNAL_HINT",
    56: "FLOW_CAPABILITIES_CONTEXT_SERVED_WINDOW",
    57: "FLOW_MAX_OUTPUT_TOKENS_CANNOT_RAISE_WINDOW",
    58: "FLOW_PACK_LOG_FLOOR_WARNING",
    59: "FLOW_HOST_PARSE_READS_FINAL_PASS",
    60: "FLOW_HOST_PARSE_REPORTS_COMPONENT_SUMS",
    61: "FLOW_CPU_SHAPE_CARRIES_DEVICE_PIN",
    62: "FLOW_CPU_SHAPE_SKIPS_SPILL_REJECTION",
    63: "FLOW_FIT_LOG_HOST_ROW_FALLBACK",
    64: "FLOW_CPU_QUAD_PERSISTED_UNDER_SHAPE",
    65: "FLOW_DEVICE_PARSE_EXCLUDES_HOST_SUMS",
    94: "FLOW_FIT_FAILURE_RECORDS_EVIDENCE",
    66: "FLOW_WARNING_NAMES_SIDECAR",
    67: "FLOW_UNHANDLED_WARNING_NAMES_SIDECAR",
    68: "FLOW_REGEX_WARNING_NAMES_SIDECAR",
    69: "FLOW_BASE_ERROR_NAMES_ORIGIN",
    70: "FLOW_HOST_POOL_CLAMPS_TO_ZERO",
    71: "FLOW_PRESSURE_ZONE_SOURCE_PRECEDENCE",
    72: "FLOW_HOST_POOL_MIXED_TERMS",
    73: "FLOW_SERVICE_CGROUP_CAPS_HOST_POOL",
    74: "FLOW_CPU_SERVING_CHARGES_RAM_ONLY",
    75: "FLOW_CPU_SOLVES_INSIDE_RAM_POOL",
    76: "FLOW_NO_HOST_POOL_NO_GATE",
    77: "FLOW_ENTRY_PACKING_METADATA",
    78: "FLOW_ROLE_FLOOR_AND_WEIGHTED_AUCTION",
    79: "FLOW_ROLE_DEVICE_PLACEMENT",
    80: "FLOW_TTL_MOST_SPECIFIC_WINS",
    88: "FLOW_ROLE_KEY_AND_DEVICE_VOCABULARY",
    89: "FLOW_UNKNOWN_PROFILES_KEY_HINT",
    106: "FLOW_PROFILES_VALIDATION_NAMES_ERRORS",
    107: "ARCH_ROLE_IS_OPERATOR_DATA",
    108: "FLOW_POOL_SPARE_REDUCES_POOL",
    109: "FLOW_MATRIX_SETS_SELECT_ROUTER",
    110: "FLOW_SHARE_WEIGHTS_UNCAPPED",
    90: "ARCH_ROLE_REGISTRY_CLOSED_ENDPOINTS",
    95: "ARCH_ENGINES_VOCABULARY_AND_RESOLUTION",
    96: "ARCH_AUTHORED_SIDECAR_DECLARES_ENGINES",
    97: "ARCH_ENDPOINT_REGISTRY_SINGLE_OWNER",
    98: "ARCH_TASK_CATALOGUE_ONLY_VOCABULARY",
    99: "ARCH_TASK_ID_IS_DECLARED_DATA",
    100: "FLOW_EMITTED_POOLS_ARE_AVAILABLE",
    101: "FLOW_ENTRY_POOL_CHARGES_WITH_MARGIN",
    102: "FLOW_CPU_ENTRY_CHARGES_RAM_ONLY",
    103: "FLOW_EVICT_COST_RELOAD_AND_PREFILL",
}

OWNERS = {
    "The task is a declaration": "SPEC.md §Tasks",
    "Vision and the ladder": "SPEC.md §Context Tiers",
    "Context floors": "SPEC.md §Context calculation formula",
    "Image resolution budget": "docs/models/vision-context.md",
    "Entry ids": "SPEC.md §Output Format",
    "VRAM safety": "SPEC.md §VRAM Allocation",
    "Allocation": "docs/allocation.md §The auction",
    "Deploy": "docs/install.md",
    "Model listing limits": "SPEC.md §Model Metadata",
    "Log levels": "SPEC.md §Log Levels",
    "CPU-residency measurement (SPEC.md §Host pool)": "SPEC.md §Host pool",
    "Measurement failure evidence": "SPEC.md §Measurement via `llama-fit-params`",
    "A diagnostic names its subject": "docs/development.md §Claims",
    "Host budget (SPEC.md §Host pool)": "SPEC.md §Host pool",
    "Alien keys": "docs/profiles-authoring.md",
    "Roles (docs/architecture.md §Models, tasks, engines and endpoints)":
        "docs/architecture.md §Models, tasks, engines and endpoints",
    "Engine declaration (SPEC.md §Backend Selection)": "SPEC.md §Backend Selection",
    "Endpoint registry (SPEC.md §Backend support matrix)":
        "SPEC.md §Backend Selection",
    "Task catalogue (SPEC.md §Tasks)": "SPEC.md §Tasks",
    "Declared task vocabulary (SPEC.md §Tasks)": "SPEC.md §Tasks",
    "Per-category context cap (SPEC.md §VRAM Allocation)":
        "SPEC.md §VRAM Allocation",
    "Capacity routing (SPEC.md §Capacity routing)": "SPEC.md §Capacity routing",
}

ROW = re.compile(r"^\| (\d+) \| (.+?) \| (.+?) \|$")
TEST = re.compile(r"`(test_[A-Za-z0-9_]+)`")


def migrate_trace() -> dict[str, str]:
    text = TRACE.read_text()
    sections = dict(OWNERS)
    by_number: dict[int, tuple[str, str]] = {}
    out: list[str] = []
    section = ""
    for line in text.splitlines():
        if line.startswith("## "):
            section = line[3:].strip()
        if line.startswith("| "):
            match = ROW.match(line)
            if match:
                number = int(match.group(1))
                name = NAMES[number]
                owner = sections.get(section, OWNERS.get(section, "SPEC.md"))
                by_number[number] = (name, owner)
                out.append(f"| {name} | {owner} |  | {match.group(2)} | "
                           f"{match.group(3)} |")
                continue
            if line.startswith("| # | Requirement | Test |"):
                out.append("| ID | Owner | Satisfies | Requirement | Tests |")
                continue
            if re.fullmatch(r"\|[-| ]+\|", line):
                out.append("|---|---|---|---|---|")
                continue
        out.append(line)
    header = (
        "Every traced rule has a named ID, an owner doc section and a test.\n"
        "See docs/taxonomy.md for the ID scheme; `python extras/check_trace.py`\n"
        "verifies the registry, citations and generated index."
    )
    body = "\n".join(out)
    body = re.sub(r"Every rule the design states.*?carries a `Validates:` citation\.",
                  header, body, count=1, flags=re.S)
    TRACE.write_text(body.rstrip() + "\n")
    return {name: owner for name, owner in by_number.values()}


def migrate_tests(by_number: dict[int, tuple[str, str]],
                  by_test: dict[str, tuple[str, str]]) -> None:
    unresolved: list[str] = []
    for path in sorted(TESTS.rglob("test_*.py")):
        lines = path.read_text().splitlines(keepends=True)
        current = ""
        changed = False
        for i, line in enumerate(lines):
            if line.startswith("def test_"):
                current = line.split("(")[0][4:]
            if "requirement #" not in line:
                continue
            resolved = None
            for raw in re.findall(r"(\d+)", line.split("requirement #", 1)[1]):
                number = int(raw)
                if number in by_number:
                    resolved = by_number[number]
                    break
                if current in by_test:
                    resolved = by_test[current]
                    break
            if resolved is None:
                unresolved.append(f"{path.name}:{i + 1}")
                continue
            name, owner = resolved
            indent = line[:len(line) - len(line.lstrip())]
            marker = "# Validates:" if line.lstrip().startswith("#") \
                else "Validates:"
            lines[i] = f"{indent}{marker} {name} — {owner}\n"
            changed = True
        if changed:
            path.write_text("".join(lines))
    if unresolved:
        raise SystemExit(f"unresolved numeric citations: {unresolved}")


def main() -> int:
    if not re.search(r"^\| \d+ \|", TRACE.read_text(), re.MULTILINE):
        print("taxonomy already migrated")
        return 0
    missing = set(ROW.findall(TRACE.read_text())) - set(map(str, NAMES))
    if missing:
        raise SystemExit(f"no name for requirements: {sorted(missing)}")
    by_number: dict[int, tuple[str, str]] = {}
    sections = dict(OWNERS)
    section = ""
    for line in TRACE.read_text().splitlines():
        if line.startswith("## "):
            section = line[3:].strip()
        match = ROW.match(line)
        if match:
            number = int(match.group(1))
            by_number[number] = (
                NAMES[number], sections.get(section, "SPEC.md"))
    by_test: dict[str, tuple[str, str]] = {}
    for line in TRACE.read_text().splitlines():
        if line.startswith("## "):
            section = line[3:].strip()
        match = ROW.match(line)
        if match:
            for test in TEST.findall(match.group(3)):
                by_test.setdefault(test, by_number[int(match.group(1))])
    migrate_trace()
    migrate_tests(by_number, by_test)
    print(f"migrated {len(by_number)} requirements")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
