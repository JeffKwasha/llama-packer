#!/usr/bin/env python3
"""Judge a capacity-routing config: what can actually live beside a chat.

Reads an emitted llama-swap config (or a dry-run log with ``I |``/``W |``
prefixes) and answers one question per chat entry: given the pools the router
will enforce, which sidecars can be resident at the same time?

The router admits a set of entries when, for every pool, the summed
``capacities`` of the set is <= the pool size.  Role pools are size 1, so a
set holds at most one serving per role.  This tool applies that rule
literally; it never second-guesses the planner.

Usage:
    check_sidecar_budget.py CONFIG [--require ROLE[=ENTRY][,...]]
                                 [--min-chat-entries N] [--json]

Examples:
    # Can every chat serve beside one TTS and the RAG pair?
    check_sidecar_budget.py run.yaml --require t2s,embeddings,rerank

    # Pin the exact sidecars a deployment intends to use.
    check_sidecar_budget.py run.yaml --require t2s=chatterbox.s,embeddings=jina-v5-small-retrieval

Exit 0 when every check passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict

import yaml

LOG_PREFIX = re.compile(r"^[IWED] \| ")
NON_POOL_KEYS = ("default", "ram")


def load_config(path: str) -> dict:
    lines = open(path).read().splitlines()
    clean = "\n".join(ln for ln in lines if not LOG_PREFIX.match(ln))
    cfg = yaml.safe_load(clean)
    if not isinstance(cfg, dict) or "models" not in cfg:
        raise SystemExit(f"{path}: not an emitted config (no `models:`)")
    return cfg


def pools_of(cfg: dict) -> dict[str, int]:
    try:
        return dict(
            cfg["routing"]["router"]["settings"]["capacity"]["pools"])
    except (KeyError, TypeError):
        raise SystemExit("not a capacity-routing config (no pools at "
                         "routing.router.settings.capacity.pools)")


def role_of(caps: dict) -> str | None:
    for key in caps:
        if key not in NON_POOL_KEYS:
            return key
    return None


def inventory(cfg: dict) -> dict[str, dict]:
    """entry id -> {caps, role, evict_cost, ctx}."""
    out = {}
    for entry_id, entry in cfg["models"].items():
        caps = entry.get("capacities") or {}
        meta = entry.get("metadata") or {}
        out[entry_id] = {
            "caps": {k: int(v) for k, v in caps.items()},
            "role": role_of(caps),
            "evict_cost": entry.get("evict_cost"),
            "ctx": meta.get("ctx_size"),
        }
    return out


def fits(entries: list[dict], pools: dict[str, int]) -> bool:
    totals: dict[str, int] = defaultdict(int)
    for e in entries:
        for pool, mb in e["caps"].items():
            totals[pool] += mb
    return all(mb <= pools.get(pool, 0) for pool, mb in totals.items())


def headroom(entry: dict, pools: dict[str, int]) -> dict[str, int]:
    return {pool: pools.get(pool, 0) - mb
            for pool, mb in entry["caps"].items()}


def parse_require(spec: str) -> dict[str, str | None]:
    """`role` or `role=entry`, comma separated."""
    wanted: dict[str, str | None] = {}
    if not spec:
        return wanted
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            role, entry = part.split("=", 1)
            wanted[role.strip()] = entry.strip()
        else:
            wanted[part] = None
    return wanted


def pick(role: str, inventory_: dict[str, dict], wanted: str | None,
         ) -> list[str]:
    members = [k for k, v in inventory_.items() if v["role"] == role]
    if wanted is not None:
        return [wanted] if wanted in members else []
    return sorted(members, key=lambda k: sum(
        v for p, v in inventory_[k]["caps"].items() if p != role))


def evaluate(cfg: dict, require: dict[str, str | None],
             min_chat_entries: int) -> dict:
    pools = pools_of(cfg)
    inv = inventory(cfg)
    chats = sorted((k for k, v in inv.items() if v["role"] == "chat"),
                   key=lambda k: -inv[k]["caps"].get("default", 0))
    report: dict = {
        "pools": pools,
        "entries": len(inv),
        "chat_entries": len(chats),
        "by_role": {},
        "problems": [],
        "chats": [],
    }

    for role in sorted({v["role"] for v in inv.values() if v["role"]}):
        ids = sorted(k for k, v in inv.items() if v["role"] == role)
        report["by_role"][role] = ids

    for entry_id, entry in inv.items():
        for pool, mb in entry["caps"].items():
            if mb > pools.get(pool, 0):
                report["problems"].append(
                    f"{entry_id}: {pool}={mb} exceeds pool {pools.get(pool, 0)}")
        if entry["role"] is None:
            report["problems"].append(f"{entry_id}: declares no role pool")

    required_ids: dict[str, list[str]] = {}
    for role, wanted in require.items():
        ids = pick(role, inv, wanted)
        required_ids[role] = ids
        if not ids:
            report["problems"].append(
                f"required role {role!r} has no entry"
                + (f" named {wanted!r}" if wanted else ""))

    for chat_id in chats:
        chat = inv[chat_id]
        row = {"entry": chat_id,
               "caps": dict(chat["caps"]),
               "ctx": chat["ctx"],
               "headroom": headroom(chat, pools),
               "sidecars": {}}
        chosen: list[dict] = []
        for role, ids in required_ids.items():
            ok = [i for i in ids
                  if fits([chat] + chosen + [inv[i]], pools)]
            row["sidecars"][role] = {"fits": ok, "tried": ids}
            if ok:
                chosen.append(inv[ok[0]])
        row["all_required"] = all(row["sidecars"][r]["fits"]
                                  for r in required_ids)
        report["chats"].append(row)

    report["all_chats_satisfy"] = all(r["all_required"] for r in report["chats"]) \
        if required_ids else None
    if min_chat_entries and len(chats) < min_chat_entries:
        report["problems"].append(
            f"only {len(chats)} chat entries, expected >= {min_chat_entries}")
    report["ok"] = (not report["problems"]
                    and report["all_chats_satisfy"] is not False)
    return report


def summarize(report: dict) -> str:
    pools = report["pools"]
    out = ["pools: " + "  ".join(f"{k}={v}" for k, v in pools.items()),
           f"entries: {report['entries']}  chats: {report['chat_entries']}",
           ""]
    for role, ids in report["by_role"].items():
        if role == "chat":
            out.append(f"  {role:11s} {len(ids):2d}")
        else:
            out.append(f"  {role:11s} {len(ids):2d}: {', '.join(ids)}")
    out.append("")
    required = list(report["chats"][0]["sidecars"]) if report["chats"] else []
    if report["chats"]:
        head = f"{'chat entry':46s} {'VRAM':>6s} {'freeV':>6s} {'freeR':>6s}"
        for role in required:
            head += f"  {role:>10s}"
        out.append(head)
        for row in report["chats"]:
            line = (f"{row['entry']:46s} "
                    f"{row['caps'].get('default', 0):6d} "
                    f"{row['headroom'].get('default', 0):6d} "
                    f"{row['headroom'].get('ram', pools.get('ram', 0)):6d}")
            for role in required:
                fits_ = row["sidecars"][role]["fits"]
                line += f"  {'ok' if fits_ else 'NO':>10s}"
            out.append(line)
    if report["problems"]:
        out.append("")
        out.append("problems:")
        out.extend(f"  - {p}" for p in report["problems"])
    if report["all_chats_satisfy"] is not None:
        ok = {}
        for role in required:
            ok[role] = sum(1 for row in report["chats"]
                           if row["sidecars"][role]["fits"])
        out.append("")
        out.append("chats that can host one of each required role:")
        out.extend(f"  {role:11s} {ok[role]:2d}/{len(report['chats'])}"
                   for role in required)
        out.append("")
        out.append("every chat can host the whole required set: "
                   + ("yes" if report["all_chats_satisfy"] else "NO"))
    ctxs = sorted(r["ctx"] for r in report["chats"] if r.get("ctx"))
    if ctxs:
        out.append("chat served ctx: min %s  median %s  max %s"
                   % (ctxs[0], ctxs[len(ctxs) // 2], ctxs[-1]))
    out.append("")
    out.append("RESULT: " + ("pass" if report["ok"] else "FAIL"))
    return "\n".join(out)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--require", default="",
                    help="ROLE[=ENTRY][,...] that must co-reside with every chat")
    ap.add_argument("--min-chat-entries", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    report = evaluate(load_config(args.config), parse_require(args.require),
                      args.min_chat_entries)
    print(json.dumps(report, indent=2) if args.json else summarize(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
