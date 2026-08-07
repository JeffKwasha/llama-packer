#!/usr/bin/env python3
"""Generate llama-swap config.yaml from model metadata and profiles.

Usage:
  ./gen-config.py                        # defaults
  ./gen-config.py --dry-run              # preview to stdout
  ./gen-config.py -v 8929                # specific llama-server version
  ./gen-config.py --llama-server /opt/lsrv  # explicit binary path
  ./gen-config.py --output /etc/ls/config.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from source without install
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llama_packer.__main__ import main

if __name__ == "__main__":
    main()