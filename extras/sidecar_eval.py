#!/usr/bin/env python3
"""Evaluate sidecar co-residency across capacity-routing strategies.

Runs llama-packer over a base ``profiles.yaml`` once per strategy, then judges
each emitted config with :mod:`check_sidecar_budget`: can a chat serving be
resident beside the sidecars that matter during a session?

Strategies are declared as small transformations of the base profile, so the
same evaluation runs against any fleet and any future packer:

    python extras/sidecar_eval.py --profiles /opt/llama/profiles.yaml
    python extras/sidecar_eval.py --profiles P --only host,voice --json

Exit 0 when every strategy satisfies its own requirement; 1 otherwise.
See ``docs/plans/sidecar-residency-eval.md`` for what the strategies mean.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_sidecar_budget import evaluate  # noqa: E402


@dataclass
class Strategy:
    name: str
    what: str
    require: list[str] = field(default_factory=list)
    min_chats: int = 0
    #: Roles to place on the host, None = leave the base profile alone.
    host_roles: list[str] | None = None
    #: pool id -> extra pool keys merged into the base ``pools:`` section.
    pools: dict[str, dict] = field(default_factory=dict)


STRATEGIES = [
    Strategy(
        name="host",
        what="deployed profile: RAG and dictation on the host, voice on GPU "
             "without a reservation",
        require=[],
    ),
    Strategy(
        name="voice",
        what="reserve room for one voice sidecar in every chat solve",
        require=["t2s"],
        pools={"default": {"reserve_extra": "4G"}},
    ),
    Strategy(
        name="rag-hot",
        what="dictation to GPU, so the RAG pair is the only host resident",
        require=["embeddings", "rerank"],
        host_roles=["s2t"],
    ),
    Strategy(
        name="rag-voice",
        what="dictation to GPU plus a voice reservation",
        require=["embeddings", "rerank", "t2s"],
        host_roles=["s2t"],
        pools={"default": {"reserve_extra": "4G"}},
    ),
    Strategy(
        name="all-hot",
        what="every sidecar role on GPU, nothing left on the host",
        require=["embeddings", "rerank", "s2t", "t2s"],
        host_roles=[],
        pools={"default": {"reserve_extra": "4G"}},
    ),
]


def apply_strategy(base: dict, strategy: Strategy) -> dict:
    cfg = yaml.safe_load(yaml.safe_dump(base))
    if strategy.host_roles is not None:
        for name, spec in (cfg.get("roles") or {}).items():
            if name in strategy.host_roles:
                spec["device"] = "cpu"
            else:
                spec.pop("device", None)
    if strategy.pools:
        pools = cfg.setdefault("pools", {}) or {}
        for pool_id, extra in strategy.pools.items():
            pools.setdefault(pool_id, {}).update(extra)
        cfg["pools"] = pools
    return cfg


def run_packer(profile_path: Path, workdir: Path,
               repo: Path) -> tuple[str, str, int]:
    out = workdir / "config.yaml"
    log = workdir / "packer.log"
    proc = subprocess.run(
        [sys.executable, "-m", "llama_packer", "--dry-run",
         "--profiles", str(profile_path)],
        capture_output=True, text=True, cwd=str(repo),
    )
    out.write_text(proc.stdout)
    log.write_text(proc.stderr)
    return str(out), str(log), proc.returncode


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profiles", required=True)
    ap.add_argument("--only", default="",
                    help="comma-separated strategy names")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--keep", action="store_true",
                    help="keep the generated profiles and configs")
    args = ap.parse_args(argv)

    base = yaml.safe_load(Path(args.profiles).read_text())
    wanted = {s.strip() for s in args.only.split(",") if s.strip()}
    chosen = [s for s in STRATEGIES if not wanted or s.name in wanted]
    unknown = wanted - {s.name for s in STRATEGIES}
    if unknown:
        raise SystemExit(f"unknown strategy: {', '.join(sorted(unknown))}")

    repo = Path(__file__).resolve().parent.parent
    workroot = Path(tempfile.mkdtemp(prefix="sidecar-eval-", dir=repo))
    results = []
    for strategy in chosen:
        work = workroot / strategy.name
        work.mkdir()
        profile = work / "profiles.yaml"
        profile.write_text(yaml.safe_dump(apply_strategy(base, strategy),
                                          sort_keys=False))
        cfg_path, log_path, rc = run_packer(profile, work, repo)
        if rc != 0:
            results.append({"strategy": strategy.name, "what": strategy.what,
                            "packer_exit": rc, "ok": False,
                            "problems": [f"packer exited {rc}; see {log_path}"]})
            continue
        report = evaluate(yaml.safe_load(open(cfg_path)),
                          {r: None for r in strategy.require},
                          strategy.min_chats)
        report |= {"strategy": strategy.name, "what": strategy.what,
                   "packer_exit": rc}
        results.append(report)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"base profile: {args.profiles}")
        print(f"work dir:     {workroot if args.keep else '(removed)'}")
        print()
        head = (f"{'strategy':10s} {'max chat':>8s} {'min free':>8s} "
                f"{'entries':>7s} {'med ctx':>8s}  {'required sidecars':34s} result")
        print(head)
        for r in results:
            if "chats" not in r:
                print(f"{r['strategy']:10s} {'-':>8s} {'-':>8s} {'-':>7s} "
                      f"{'-':>8s}  {'':34s} FAIL")
                continue
            chats = r["chats"]
            max_chat = max((c["caps"].get("default", 0) for c in chats),
                           default=0)
            free = min((c["headroom"].get("default", 0) for c in chats),
                       default=0)
            ctxs = sorted(c["ctx"] for c in chats if c.get("ctx"))
            med = ctxs[len(ctxs) // 2] if ctxs else 0
            marks = " ".join(
                f"{role}:{sum(1 for c in chats if c['sidecars'][role]['fits'])}"
                f"/{len(chats)}" for role in (r["chats"][0]["sidecars"]
                                              if chats else []))
            print(f"{r['strategy']:10s} {max_chat:8d} {free:8d} "
                  f"{r['chat_entries']:7d} {med:8d}  {marks:34s} "
                  f"{'pass' if r['ok'] else 'FAIL'}")
            for problem in r["problems"]:
                print(f"{'':10s} ! {problem}")
        print()
        for r in results:
            print(f"{r['strategy']:10s} {r['what']}")

    ok = all(r["ok"] for r in results)
    if not args.keep:
        import shutil
        shutil.rmtree(workroot, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
