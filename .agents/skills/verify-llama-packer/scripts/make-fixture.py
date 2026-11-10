#!/usr/bin/env python3
"""Write a minimal model tree for verification.

    make-fixture.py <dir> [--layout chat] [--tasks chat] [--engines llama-server]

The GGUF is a real header with one zero-filled tensor, so the packer can name
the architecture.  llama-fit-params may still refuse to load it; the packer
then emits the entry as `estimated: false`, which is expected.
"""
import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--layout", default="chat",
                    help="tree under models/; 'root' puts the model at the root")
    ap.add_argument("--tasks", default="chat")
    ap.add_argument("--engines", default="llama-server")
    args = ap.parse_args()
    try:
        import numpy as np
        from gguf import GGUFWriter
    except ImportError as e:
        print(f"make-fixture: {e}; run under the repo's dev environment "
              "(uv run / the interpreter with gguf installed)", file=sys.stderr)
        return 2

    root = Path(args.dir) / "models"
    tree = root if args.layout == "root" else root / args.layout
    tree.mkdir(parents=True, exist_ok=True)
    gguf = tree / "m.gguf"
    writer = GGUFWriter(str(gguf), arch="qwen3")
    writer.add_uint32("qwen3.context_length", 32768)
    writer.add_tensor("blk.0.attn_q.weight",
                      np.zeros((16, 16), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    # Seed a measured block under the shape the packer derives at the
    # bundled defaults (`-ub 512`, no fleet args).  Without it the packer
    # runs llama-fit-params on a synthetic header, which crashes and costs
    # seconds.  A profile that changes ubatch/flash re-measures by design.
    (tree / "m.md").write_text(
        f"---\nname: Fixture\ntasks: [{args.tasks}]\n"
        f"engines: [{args.engines}]\n"
        "derived:\n"
        "  model_mib: 1000\n"
        "  kv_per_token_mib: 0.05\n"
        "  slot_mib: 0\n"
        "  compute_mib: 100\n"
        "  source: llama-server\n"
        "  cache_type: q8_0\n"
        '  shape: "-ub 512"\n'
        "---\n")
    print(gguf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
