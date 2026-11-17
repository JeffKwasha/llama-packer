#!/usr/bin/env python3
"""Verify the requirement trace: every named test exists and cites a doc.

    python extras/check_trace.py

The point is that "implemented" is checkable. A requirement with no test is
unimplemented no matter what the code looks like, and a test without a
`Validates:` citation cannot be traced back to the rule it holds up.
"""
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRACE = ROOT / "docs/plans/requirement-trace.md"


def scan_tests() -> tuple[set[str], set[str]]:
    """``(defined, cited)`` in one pass over the test tree.

    A test is *cited* when its own body carries a ``Validates:`` line, so the
    citation is read from the function's source segment rather than the file:
    a line elsewhere in the module does not hold up the test that merely sits
    near it.
    """
    defined: set[str] = set()
    cited: set[str] = set()
    for path in (ROOT / "tests").rglob("*.py"):
        text = path.read_text()
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node, ast.FunctionDef):
                continue
            if not node.name.startswith("test_"):
                continue
            defined.add(node.name)
            if "Validates:" in (ast.get_source_segment(text, node) or ""):
                cited.add(node.name)
    return defined, cited


def main() -> int:
    if not TRACE.exists():
        print(f"missing {TRACE}", file=sys.stderr)
        return 2
    trace = TRACE.read_text()
    want = set(re.findall(r"`(test_[A-Za-z0-9_]+)`", trace))
    have, cited = scan_tests()

    missing = sorted(want - have)
    requirements = len(re.findall(r"^\| \d+ \|", trace, re.MULTILINE))

    print(f"{requirements} requirements, {len(want)} tests named")
    if missing:
        print(f"\nNAMED BUT NOT DEFINED ({len(missing)}):")
        for name in missing:
            print(f"  {name}")
    uncited = sorted(want - cited)
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
