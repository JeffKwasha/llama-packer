#!/usr/bin/env python3
"""Verify the requirement trace: every named test exists and cites a doc.

    python extras/check_trace.py

The point is that "implemented" is checkable. A requirement with no test is
unimplemented no matter what the code looks like, and a test without a
`Validates:` citation cannot be traced back to the rule it holds up.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRACE = ROOT / "docs/plans/requirement-trace.md"


def defined_tests() -> set[str]:
    out = subprocess.run(
        ["grep", "-rhoE", r"^def (test_[A-Za-z0-9_]+)", str(ROOT / "tests")],
        capture_output=True, text=True, check=False).stdout
    return {line.split("def ", 1)[1].strip() for line in out.splitlines()
            if line.strip()}


def cited_tests() -> set[str]:
    """Test functions whose own body carries a ``Validates:`` line.

    Scoped to the function rather than the file: a citation elsewhere in
    the module does not hold up the test that merely sits near it.
    """
    import ast
    cited = set()
    for path in (ROOT / "tests").rglob("test_*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if not node.name.startswith("test_"):
                continue
            body = ast.get_source_segment(path.read_text(), node) or ""
            if "Validates:" in body:
                cited.add(node.name)
    return cited


def main() -> int:
    if not TRACE.exists():
        print(f"missing {TRACE}", file=sys.stderr)
        return 2
    trace = TRACE.read_text()
    want = set(re.findall(r"`(test_[A-Za-z0-9_]+)`", trace))
    have = defined_tests()

    missing = sorted(want - have)
    requirements = len(re.findall(r"^\| \d+ \|", trace, re.MULTILINE))

    print(f"{requirements} requirements, {len(want)} tests named")
    if missing:
        print(f"\nNAMED BUT NOT DEFINED ({len(missing)}):")
        for name in missing:
            print(f"  {name}")
    uncited = sorted(want - cited_tests())
    if uncited:
        print(f"\nDEFINED BUT NO `Validates:` CITATION ({len(uncited)}):")
        for name in uncited:
            print(f"  {name}")
    if not missing and not uncited:
        print("\ntrace is complete: every requirement has a test, "
              "every test cites its rule.")
    return 1 if (missing or uncited) else 0


if __name__ == "__main__":
    sys.exit(main())
