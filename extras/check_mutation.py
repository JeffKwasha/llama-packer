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
# requirement -> (file, substring) whose deletion should break the tests
MUTATIONS = {
    3: ("llama_packer/writer.py", "for demand in (*self.demands_for(model), FALLBACK):"),
    5: ("llama_packer/writer.py", "if demand is FALLBACK:\n            self.set_admission"),
    7: ("llama_packer/writer.py", "                best = rung\n"),
    20: ("llama_packer/writer.py", "        return min(pins) if pins else None\n"),
    11: ("llama_packer/writer.py", "                return best\n"),
    16: ("llama_packer/writer.py", "def can_meet(self, model, demand"),
    19: ("llama_packer/vram.py", "    def _sidecar_ceiling(self) -> int | None:"),
    23: ("llama_packer/writer.py", 'tail = ".".join(x for x in (cat_id, flags) if x)'),
    33: ("llama_packer/writer.py", "def record_admission(self, model, demand"),
    35: ("llama_packer/writer.py", "power=0.0 if is_null else None,"),
}


def tests_for(req: int, trace: str) -> list[str]:
    row = re.search(rf"^\| {req} \|(.+)$", trace, re.MULTILINE)
    return re.findall(r"`(test_[A-Za-z0-9_]+)`", row.group(1)) if row else []


def main() -> int:
    trace = TRACE.read_text()
    bad = 0
    for req, (rel, needle) in sorted(MUTATIONS.items()):
        names = tests_for(req, trace)
        if not names:
            print(f"  #{req:<3} no tests named")
            bad += 1
            continue
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "tree"
            shutil.copytree(ROOT, work, ignore=shutil.ignore_patterns(
                ".git", "__pycache__", ".venv", "*.pyc", ".ruff_cache", ".pytest_cache", ".claude", "node_modules"))
            target = work / rel
            text = target.read_text()
            if needle not in text:
                print(f"  #{req:<3} MUTATION STALE - anchor not found in {rel}")
                bad += 1
                continue
            target.write_text(text.replace(needle, "", 1))
            # Map each test name to the file that defines it.
            files = set()
            for n in names:
                hit = subprocess.run(
                    ["grep", "-rlE", rf"^def {n}\(", "tests"],
                    cwd=work, capture_output=True, text=True, check=False).stdout.split()
                if not hit:
                    print(f"  #{req:<3} {n} not found in tests/")
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
                    out = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
                    results.append((n, "failed" in out or "error" in out))
                # A requirement is held if *some* named test fails without
                # the code. Naming several tests does not mean they share a
                # code path -- --max-context, say, is read both through
                # _serving_pin() and directly in the slot solve -- so
                # demanding that every one of them catch one mutation just
                # measures which accessor got deleted. Per-test detail is
                # printed regardless, so a test that never catches anything
                # stays visible rather than quietly riding along.
                caught = any(ok for _, ok in results)
                print(f"  #{req:<3} {'CAUGHT ' if caught else 'MISSED '} "
                      f"{', '.join(names)[:58]}")
                for n, ok in results:
                    if not ok:
                        print(f"          - {n}: other code path")
                if not caught:
                    bad += 1
    print("\nall requirements hold" if not bad else f"\n{bad} not holding")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
