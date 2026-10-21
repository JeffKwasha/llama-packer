#!/usr/bin/env python3
"""Find tests that hold up nothing.

    python extras/find_decorated_tests.py [--out report.json]

A test that passes whether or not the code is there enforces nothing. This
finds those tests by mutation, module by module:

1. Measure which package functions each test file actually executes
   (coverage, one run per test file).
2. Mutate each of those functions -- its body returns None instead of its
   result -- and run the tests that touch it. Every test that fails under
   some mutation *detects* that mutation; every test that fails under none
   of them is decoration.

A test that detects nothing is a deletion candidate. A test that detects
something is holding a rule up, however narrow that rule is.
"""
from __future__ import annotations

import argparse
import ast
import itertools
import json
import os
import re
import time
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"
EXTRAS = ("install.py", "update.py", "build_engine.py", "bench.py")
FAILED_PREFIX = "FAILED "
ERROR_PREFIX = "ERROR "


def target_files() -> list[Path]:
    files = sorted((ROOT / "llama_packer").rglob("*.py"))
    files += [ROOT / "extras" / name for name in EXTRAS]
    return [f for f in files if f.exists()]


MUTANTS = {
    int: "-1",
    float: "-1.0",
    str: '"\\x00MUTANT"',
    bool: "False",
    dict: '{"\\x00": "\\x00MUTANT"}',
    list: '["\\x00MUTANT"]',
    tuple: '("\\x00MUTANT",)',
}


def mutant_value(node) -> str:
    """A wrong value of the same shape as what the function returns.

    Same type keeps the mutation from exploding downstream (a string where
    an int was expected raises instead of failing an assertion), and the
    value is one nothing legitimate returns.
    """
    types = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Return) and n.value is not None:
            try:
                types.add(type(ast.literal_eval(n.value)))
            except (ValueError, SyntaxError, TypeError):
                continue
    for t in (bool, int, float, str, dict, list, tuple):
        if t in types:
            return MUTANTS[t]
    return MUTANTS[str]


def functions(path: Path) -> list[dict]:
    """Every function in *path* that can be mutated to change its result."""
    tree = ast.parse(path.read_text())
    out: list[dict] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        name = node.name
        if name.startswith("__") and name.endswith("__"):
            continue
        returns = [n for n in ast.walk(node)
                   if isinstance(n, ast.Return) and n.value is not None]
        yields = [n for n in ast.walk(node) if isinstance(n, (ast.Yield, ast.YieldFrom))]
        if not node.body:
            continue
        first = node.body[0]
        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            if len(node.body) < 2:
                continue  # a docstring-only body: nothing to mutate
            first = node.body[1]
        if yields and not returns:
            kind = "empty_gen"  # a generator that yields nothing
        elif not returns:
            kind = "noop"       # a side-effecting function: make it do nothing
        else:
            kind = "value"
        out.append({
            "name": name,
            # the body only: a def line runs at import time, so counting it
            # would make every function in an imported module look covered
            "start": first.lineno,
            "end": node.end_lineno or node.lineno,
            "insert_at": first.lineno,
            "kind": kind,
            "mutant": mutant_value(node) if kind == "value" else "",
        })
    return out


def covered_functions(executed: dict[str, set[int]], index: dict[str, list[dict]]) -> dict[str, list[str]]:
    """Map target file -> the function names whose lines were executed."""
    out: dict[str, list[str]] = {}
    for rel, lines in executed.items():
        for fn in index.get(rel, []):
            if any(fn["start"] <= ln <= fn["end"] for ln in lines):
                out.setdefault(rel, []).append(fn["name"])
    return out


def run_coverage(test_file: Path, tmp: Path) -> tuple[dict[str, set[int]], dict[str, float]]:
    data = tmp / ".coverage"
    report = tmp / "cov.json"
    env = dict(os.environ, COVERAGE_FILE=str(data))
    subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-q", "--no-header",
         "--cov=llama_packer", "--cov=extras", f"--cov-report=json:{report}",
         "--cov-report=", "-p", "no:cacheprovider"],
        cwd=ROOT, env=env, capture_output=True, text=True, check=False)
    if not report.exists():
        return {}, {}
    data_json = json.loads(report.read_text())
    out = {}
    percent = {}
    for rel, info in data_json.get("files", {}).items():
        lines = {int(x) for x in info.get("executed_lines", [])}
        if lines:
            out[rel] = lines
        summary = info.get("summary") or {}
        if summary.get("num_statements"):
            percent[rel] = round(summary["percent_covered"], 1)
    return out, percent


_MUTATION_SEQ = itertools.count()


def _bump_mtime(path: Path) -> None:
    """Make the bytecode cache miss for this file.

    CPython keys a .pyc on the source's mtime (whole seconds) and its size.
    Two mutations of the same size inside one second would therefore share
    a cached module, and the second mutant would silently run the first
    one's code. Each write gets its own timestamp.
    """
    stamp = time.time() + next(_MUTATION_SEQ)
    os.utime(path, (stamp, stamp))


def mutate(path: Path, fn: dict) -> str:
    original = path.read_text()
    lines = original.splitlines(keepends=True)
    kind = fn.get("kind", "value")
    if kind == "noop":
        # the whole body goes: a side-effecting function that does nothing
        start = fn["insert_at"] - 1
        end = fn["end"]
        indent = re.match(r"\s*", lines[start]).group(0)
        lines[start:end] = [f"{indent}pass  # MUTANT\n"]
    elif kind == "empty_gen":
        idx = fn["insert_at"] - 1
        indent = re.match(r"\s*", lines[idx]).group(0)
        lines.insert(idx, f"{indent}return  # MUTANT\n")
    else:
        idx = fn["insert_at"] - 1
        indent = re.match(r"\s*", lines[idx]).group(0)
        lines.insert(idx, f"{indent}return {fn.get('mutant', 'None')}  # MUTANT\n")
    path.write_text("".join(lines))
    _bump_mtime(path)
    return original


def run_tests(files: list[Path]) -> tuple[set[str], int]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *[str(f) for f in files], "-q",
         "--no-header", "--tb=no", "-rfE", "--continue-on-collection-errors",
         "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True, check=False)
    failed = set()
    errored_files = set()
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith(FAILED_PREFIX):
            prefix = FAILED_PREFIX
        elif line.startswith(ERROR_PREFIX):
            prefix = ERROR_PREFIX
        else:
            continue
        # the id may contain spaces (parametrised values), so cut the
        # summary's " - <message>" tail rather than splitting on whitespace
        node = line[len(prefix):]
        if " - " in node:
            node = node.split(" - ", 1)[0]
        node = node.strip()
        if prefix is ERROR_PREFIX and "::" not in node:
            # a collection error: the whole file could not even be imported
            errored_files.add(node)
        else:
            failed.add(node)
    return failed, errored_files, proc.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "extras" / "decorated-tests.json"))
    ap.add_argument("--limit", type=int, default=0, help="stop after N mutants")
    ap.add_argument("--test", default="", help="restrict to one test file")
    args = ap.parse_args()

    targets = target_files()
    index = {str(p.relative_to(ROOT)): functions(p) for p in targets}
    total_fns = sum(len(v) for v in index.values())
    print(f"{len(targets)} target files, {total_fns} mutable functions", flush=True)

    test_files = sorted(TESTS.rglob("test_*.py"))
    if args.test:
        test_files = [ROOT / args.test] if (ROOT / args.test).exists() else [
            p for p in test_files if p.name == args.test]
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # 1. which functions does each test file exercise?
        file_fns: dict[Path, list[tuple[str, str]]] = {}
        coverage: dict[str, float] = {}
        for i, tf in enumerate(test_files, 1):
            executed, percent = run_coverage(tf, tmp)
            for rel, names in covered_functions(executed, index).items():
                for n in names:
                    file_fns.setdefault(tf, []).append((rel, n))
            for rel, pct in percent.items():
                if rel in index:
                    coverage[rel] = max(coverage.get(rel, 0.0), pct)
            print(f"  coverage {i}/{len(test_files)} {tf.name}", flush=True)

        # 2. mutate each function and see which tests notice.
        detected: dict[str, set[str]] = {}
        errored_files: set[str] = set()
        mutants = 0
        for rel, names in index.items():
            path = ROOT / rel
            for fn in names:
                covering = [tf for tf, fns in file_fns.items()
                            if (rel, fn["name"]) in fns]
                if not covering:
                    continue
                original = mutate(path, fn)
                try:
                    failed, errored, _ = run_tests(covering)
                finally:
                    path.write_text(original)
                    _bump_mtime(path)
                mutants += 1
                for t in failed:
                    detected.setdefault(t, set()).add(f"{rel}::{fn['name']}")
                for ef in errored:
                    errored_files.add(ef)
                if mutants % 25 == 0:
                    print(f"  mutants {mutants}: {len(detected)} tests detect something",
                          flush=True)
                if args.limit and mutants >= args.limit:
                    print(f"  stopped at {mutants} mutants (--limit)", flush=True)
                    break
            if args.limit and mutants >= args.limit:
                break

    all_tests = set()
    for tf in test_files:
        out = subprocess.run(
            [sys.executable, "-m", "pytest", str(tf), "-q", "--no-header",
             "--collect-only", "-p", "no:cacheprovider"],
            cwd=ROOT, capture_output=True, text=True, check=False)
        for line in out.stdout.splitlines():
            if "::" in line and line.startswith(str(tf.relative_to(ROOT))):
                all_tests.add(line.strip())

    # a file that could not even be imported under some mutation noticed
    # the mutation too: its tests could not run against the broken code
    unproven = sorted(t for t in all_tests
                      if t not in detected
                      and t.split("::")[0] not in errored_files)
    report = {
        "mutants": mutants,
        "tests_collected": len(all_tests),
        "tests_detecting_something": len(detected),
        "tests_detecting_nothing": unproven,
        "collection_errors": sorted(errored_files),
        "coverage": dict(sorted(coverage.items())),
        "detections": {k: sorted(v) for k, v in sorted(detected.items())},
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\n{len(all_tests)} tests, {mutants} mutants")
    print(f"{len(detected)} tests detect at least one mutation")
    print(f"{len(unproven)} tests detect nothing:")
    for t in unproven:
        print(f"  {t}")
    print(f"\nreport: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
