# Installing llama-packer Without PyPI

`llama-packer` is **not published on PyPI**. The README's `uvx llama-packer --dry-run` command is incorrect — `uvx` resolves packages from PyPI, and this package does not exist there. The project lives at [github.com/JeffKwasha/llama-packer](https://github.com/JeffKwasha/llama-packer) and must be installed directly from the git repository or local source.

This document covers every installation method, a Linux setup script with full environment-variable customization, prerequisites, and a comparison of PyPI vs. GitHub-direct distribution with security analysis.

---

## Prerequisites

| Requirement | Details |
|---|---|
| **Python** | 3.10 or newer |
| **uv** | The Astral `uv` package manager (`pip install uv` or follow [uv.getting-started](https://docs.astral.sh/uv/getting-started/installation/)) |
| **llama.cpp builds** | Prebuilt binaries in `llama-b*/` directories (created by `extras/update`), OR set `LLAMA_BIN_DIR` to an existing directory containing `llama-server` and `llama-fit-params` |
| **GPU** | NVIDIA (CUDA) or AMD (ROCm) GPU for VRAM detection and model serving |
| **git** | Required for cloning the repository |
| **Linux** | This guide targets Linux; macOS should work similarly with path adjustments |

### Assumptions

- The project is cloned or will be cloned to a known directory.
- `llama-b*/` build directories exist in the repo root, created by running `./extras/update` from the repo root. If absent, the tool requires `LLAMA_BIN_DIR` to point at a directory containing `llama-server` and `llama-fit-params`.
- `models/` directory exists (or is configured via `profiles.yaml` `models_dirs:`) containing GGUF files and `.md` sidecar metadata.
- `profiles.yaml` exists (either `./profiles.yaml` or the bundled default in `llama_packer/profiles.yaml`).

---

## Installation Methods

### Method 1: One-Shot Run (No Persistent Install)

Run the tool directly from the git repo without installing it into your environment. This is the closest equivalent to the (broken) `uvx` claim:

```bash
uv run --with git+https://github.com/JeffKwasha/llama-packer.git \
  llama-packer --dry-run
```

**What it does**: `uv run --with` creates an ephemeral virtual environment, resolves all dependencies from the `pyproject.toml` (including the git-hosted `vllm-memory-estimator`), downloads the package from GitHub, and executes `llama-packer`. The `--dry-run` flag prints the generated YAML to stdout without writing any files.

**Caveat**: Dependencies are re-resolved on every invocation. Slow first-run; fast subsequent runs via uv's cache.

### Method 2: Editable Install From Git (Recommended)

Install the package in editable mode so it tracks source changes and is available as a command on `PATH`:

```bash
git clone https://github.com/JeffKwasha/llama-packer.git
cd llama-packer
uv pip install -e .
llama-packer --dry-run
```

**What it does**: `uv pip install -e .` reads `pyproject.toml`, creates a virtual environment, installs `pyyaml` and `ruamel-yaml`, installs the `vllm-memory-estimator` from its git source, and creates the `llama-packer` console script pointing at the local source tree. Edits to `.py` files are immediately reflected — no reinstall needed.

### Method 3: Install From a Specific Commit or Tag

For reproducibility, pin to a specific commit or tag:

```bash
git clone https://github.com/JeffKwasha/llama-packer.git --branch v0.3.0
cd llama-packer
uv pip install -e .
```

Or install directly without cloning:

```bash
uv pip install git+https://github.com/JeffKwasha/llama-packer.git@v0.3.0
```

### Using llama.cpp From an Arbitrary Location

The tool resolves the `llama-server` and `llama-fit-params` binaries via `find_bin_dir()` (`llama_packer/utils.py:403`). Resolution order:

1. **`LLAMA_BIN_DIR` env var** — if set, used directly as the directory containing the binaries. Bypasses all auto-detection.
2. **`--llama-server /path/to/llama-server`** CLI flag — sets both the binary and its parent dir explicitly.
3. **Auto-detect `llama-b*/` dirs** in the current working directory — picks the highest version number.

If your llama.cpp build lives at `/opt/llama.cpp/llama-b9123/` (containing `llama-server`, `llama-fit-params`, etc.):

```bash
# Option A: env var (applies to every invocation)
export LLAMA_BIN_DIR=/opt/llama.cpp/llama-b9123
llama-packer --dry-run

# Option B: per-invocation CLI flag
llama-packer --llama-server /opt/llama.cpp/llama-b9123/llama-server --dry-run

# Option C: symlink from the repo root (makes auto-detect work)
ln -s /opt/llama.cpp/llama-b9123 ./llama-b9123
llama-packer --dry-run   # finds llama-b9123 in cwd
```

All three are equivalent. Option A is the cleanest for persistent setups (add it to your shell profile or `config.env`). Option C is useful if you want `extras/update`'s symlink management to track your build alongside any others it downloads.

#### How `extras/update` Relates to This

`extras/update` always installs into the **repo root** as `llama-bXXXXX/` directories (e.g., `./llama-b10819/`). It has no env var or flag to change the install target — the path is hardcoded:

```bash
# From extras/update line 19-20:
INSTALL_DIR="$SCRIPT_DIR"
[[ "$(basename "$SCRIPT_DIR")" == "extras" ]] && INSTALL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
```

So if you want llama.cpp in `/opt/llama.cpp/` rather than the repo root, you have two paths:

1. **Run `extras/update` normally, then relocate**:
   ```bash
   ./extras/update --cpp-only b9123    # installs to ./llama-b9123/
   mv llama-b9123 /opt/llama.cpp/      # move it where you want
   export LLAMA_BIN_DIR=/opt/llama.cpp/llama-b9123
   ```

2. **Skip `extras/update` entirely** — manually download the release tarball from [github.com/ggml-org/llama.cpp/releases](https://github.com/ggml-org/llama.cpp/releases) and extract it to your chosen location, then point `LLAMA_BIN_DIR` at it. The script is a convenience wrapper around GitHub API calls + `curl` + `tar`; nothing in it is required for the tool to function.

The key invariant: wherever the binaries end up, `llama-packer` just needs a directory containing both `llama-server` and `llama-fit-params`. The directory name doesn't matter when using `LLAMA_BIN_DIR` or `--llama-server`; it only matters for auto-detection (which expects the `llama-b####` naming convention).

---

## Linux Install Script

Save the following as `install_llama_packer.sh` and customize the environment variables at the top. The script handles all prerequisites, clones the repo, installs the tool, sets up llama.cpp binaries, and configures environment paths.

```bash
#!/usr/bin/env bash
set -euo pipefail
# =============================================================================
# llama-packer Linux Install Script
# =============================================================================
# Installs llama-packer from GitHub, sets up llama.cpp build binaries,
# configures environment variables, and verifies the installation.
#
# Customize the ENV variables below before running.
# Usage: ./install_llama_packer.sh
# =============================================================================

# --- User-Customizable Environment Variables ---

# Installation root: where the repo is cloned and the tool lives.
# Default: $HOME/llama-packer
: "${INSTALL_ROOT:=${HOME}/llama-packer}"
export INSTALL_ROOT

# llama.cpp build directory: path containing llama-server and llama-fit-params.
# If set, overrides the auto-detected llama-b#### directories.
# Leave empty to use the llama-b*/ dirs under INSTALL_ROOT (created by extras/update).
# Example: "/opt/llama/cpp/build" or "~/llama-builds"
export LLAMA_BIN_DIR="${LLAMA_BIN_DIR:-}"

# Models directory: root containing GGUF models and .md sidecars.
# This is the default for --models-dir when not passed on the CLI.
# Must exist before running llama-packer.
: "${MODELS_DIR:=${HOME}/models}"
export MODELS_DIR

# HF cache root: Hugging Face cache for hub-snapshot resolution.
# Also sets the ${HF_HOME} path macro in generated config.env.
# Defaults to ~/.cache/huggingface if not set and $HF_HOME is unset.
: "${HF_HOME:=${HOME}/.cache/huggingface}"
export HF_HOME

# Python version to use (must be >= 3.10).
# Only relevant if multiple Python versions are installed.
: "${PYTHON_VERSION:=3.12}"
export PYTHON_VERSION

# Whether to run ./extras/update to fetch llama.cpp binaries.
# Set to "1" (default) to attempt the download. Requires git + curl.
# Set to "0" to skip — you must provide llama-b*/ dirs manually or set LLAMA_BIN_DIR.
: "${RUN_UPDATER:=1}"
export RUN_UPDATER

# Path to uv binary. Leave empty to use uv from PATH.
: "${UV_BINARY:=uv}"
export UV_BINARY

# uv lockfile resolution: if "1", uv will sync dependencies from uv.lock.
# If "0", uv resolves fresh from pyproject.toml.
: "${USE_LOCKFILE:=1}"
export USE_LOCKFILE

# Directory for uv's cache and virtual environments.
# Defaults to ~/.cache/uv.
: "${UV_CACHE_DIR:=${HOME}/.cache/uv}"
export UV_CACHE_DIR

# Verbosity level for the install script itself (0, 1, 2).
: "${VERBOSITY:=1}"
export VERBOSITY

# --- Derived Paths ---

export REPO_DIR="${INSTALL_ROOT}"
export SCRIPTS_DIR="${INSTALL_ROOT}/extras"

# --- Logging ---

log()  { echo "[llama-packer-install] $*"; }
warn() { echo "[llama-packer-install] WARN: $*" >&2; }
die()  { echo "[llama-packer-install] ERROR: $*" >&2; exit 1; }

# --- Pre-flight Checks ---

check_prereqs() {
    log "Checking prerequisites..."

    # Python version
    local py_ver
    py_ver=$("${UV_BINARY}" run --with python="${PYTHON_VERSION}" python -c \
        'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if [ "$(printf '%s\n%s' "3.10" "$py_ver" | sort -V -C)" = "0" ]; then
        if [ "$(printf '%s\n%s' "$py_ver" "3.10" | sort -V -C)" = "0" ]; then
            : # ok
        else
            die "Python ${PYTHON_VERSION}+ required, found ${py_ver}"
        fi
    fi
    log "  Python: ${py_ver}"

    # uv
    if ! command -v "${UV_BINARY}" &>/dev/null; then
        die "uv not found at ${UV_BINARY}. Install: pip install uv or see https://docs.astral.sh/uv/getting-started/installation/"
    fi
    log "  uv: found"

    # git
    if ! command -v git &>/dev/null; then
        die "git not found. Install via your package manager."
    fi
    log "  git: found"

    # curl (for extras/update)
    if [ "${RUN_UPDATER}" = "1" ] && ! command -v curl &>/dev/null; then
        warn "curl not found; skipping llama.cpp download via extras/update"
        export RUN_UPDATER=0
    fi

    # GPU detection
    if command -v nvidia-smi &>/dev/null; then
        local vram
        vram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
        log "  GPU (NVIDIA): ${vram} MB VRAM detected"
    elif command -v rocminfo &>/dev/null; then
        log "  GPU (AMD/ROCm): detected via rocminfo"
    else
        warn "No NVIDIA or AMD GPU detected. llama-packer may still work for CPU-only scanning, but model serving will not be configured."
    fi

    log "Prerequisites passed."
}

# --- Clone Repository ---

clone_repo() {
    if [ -d "${REPO_DIR}/.git" ]; then
        log "Repository already exists at ${REPO_DIR}. Pulling latest..."
        git -C "${REPO_DIR}" fetch --all
        git -C "${REPO_DIR}" reset --hard origin/main
    else
        log "Cloning llama-packer to ${REPO_DIR}..."
        mkdir -p "${REPO_DIR}"
        git clone https://github.com/JeffKwasha/llama-packer.git "${REPO_DIR}"
    fi
}

# --- Set Up llama.cpp Binaries ---

setup_llama_cpp() {
    if [ "${LLAMA_BIN_DIR}" != "" ]; then
        log "Using LLAMA_BIN_DIR=${LLAMA_BIN_DIR} (explicit override)"
        if [ ! -f "${LLAMA_BIN_DIR}/llama-server" ]; then
            die "LLAMA_BIN_DIR=${LLAMA_BIN_DIR} is set but llama-server not found there."
        fi
        return 0
    fi

    if [ "${RUN_UPDATER}" = "1" ]; then
        log "Running extras/update to fetch llama.cpp binaries..."
        if [ -x "${SCRIPTS_DIR}/update" ]; then
            bash "${SCRIPTS_DIR}/update" --cpp-only
        else
            warn "extras/update not found at ${SCRIPTS_DIR}/update. Skipping llama.cpp download."
            warn "Set LLAMA_BIN_DIR to point at a directory with llama-server and llama-fit-params, or run ./extras/update manually."
        fi
    else
        log "Run_Updater=0 — skipping llama.cpp download."
        warn "You must ensure llama-b*/ directories exist under ${REPO_DIR} or set LLAMA_BIN_DIR."
    fi
}

# --- Install llama-packer ---

install_packer() {
    log "Installing llama-packer from ${REPO_DIR}..."
    cd "${REPO_DIR}"

    if [ "${USE_LOCKFILE}" = "1" ] && [ -f "uv.lock" ]; then
        log "Resolving dependencies from uv.lock..."
        "${UV_BINARY}" pip install -e . --locked
    else
        log "Resolving dependencies from pyproject.toml..."
        "${UV_BINARY}" pip install -e .
    fi

    log "Verifying installation..."
    local version
    version=$(llama-packer --version 2>&1 || true)
    if [ -n "${version}" ]; then
        log "  Installed: ${version}"
    else
        die "Installation failed: llama-packer --version returned empty."
    fi
}

# --- Environment File Setup ---

setup_env() {
    local env_file="${REPO_DIR}/config.env"
    log "Writing environment file: ${env_file}"

    cat > "${env_file}" <<EOF
# Generated by install_llama_packer.sh
# Source via systemd EnvironmentFile= or docker --env-file.
# Values double as docker bind-mount sources, e.g. -v \${MODELS_DIR}:/models
HF_HOME=${HF_HOME}
LLAMA_DIR=${LLAMA_BIN_DIR:-$(ls -d "${REPO_DIR}"/llama-b[0-9]* 2>/dev/null | head -1 || echo "NOT_SET")}
MODELS_DIR=${MODELS_DIR}
EOF

    log "Environment file written. Source it with: source ${env_file}"
}

# --- Verification ---

verify_install() {
    log "Verifying installation..."
    cd "${REPO_DIR}"

    # Check that the tool runs
    if llama-packer --dry-run >/dev/null 2>&1; then
        log "  llama-packer --dry-run: OK"
    else
        warn "llama-packer --dry-run failed. Check that models/ and profiles.yaml exist."
    fi

    # Check llama-server binary availability
    if [ -n "${LLAMA_BIN_DIR}" ]; then
        if [ -f "${LLAMA_BIN_DIR}/llama-server" ]; then
            log "  llama-server found at ${LLAMA_BIN_DIR}/llama-server"
        fi
    else
        local bin_dir
        bin_dir=$(llama-packer --dry-run 2>&1 | grep -o 'llama-b[0-9]*' | head -1 || true)
        if [ -n "${bin_dir}" ]; then
            log "  Auto-detected llama.cpp build: ${bin_dir}"
        else
            warn "No llama-server binary auto-detected. Set LLAMA_BIN_DIR or run ./extras/update."
        fi
    fi

    log "Installation complete."
    echo ""
    echo "==========================================================="
    echo " llama-packer installed at ${REPO_DIR}"
    echo ""
    echo " Usage:"
    echo "   cd ${REPO_DIR}"
    echo "   llama-packer --dry-run                    # preview config"
    echo "   llama-packer                              # generate config.yaml"
    echo "   llama-packer --models-dir \${MODELS_DIR}  # specify models dir"
    echo ""
    echo " Environment variables to customize:"
    echo "   INSTALL_ROOT     = ${INSTALL_ROOT}"
    echo "   LLAMA_BIN_DIR    = ${LLAMA_BIN_DIR:-<unset>}"
    echo "   MODELS_DIR       = ${MODELS_DIR}"
    echo "   HF_HOME          = ${HF_HOME}"
    echo "   UV_CACHE_DIR     = ${UV_CACHE_DIR}"
    echo ""
    echo " Run ./extras/update to update llama.cpp and llama-swap binaries."
    echo "==========================================================="
}

# --- Main ---

main() {
    log "llama-packer installer starting (verbosity=${VERBOSITY})"
    check_prereqs
    clone_repo
    setup_llama_cpp
    install_packer
    setup_env
    verify_install
}

main "$@"
```

### How to Use the Script

1. **Save** it as `install_llama_packer.sh`.
2. **Make it executable**: `chmod +x install_llama_packer.sh`
3. **Customize** the env vars at the top (or override them on the command line):
   ```bash
   INSTALL_ROOT=/opt/llama-packer MODELS_DIR=~/models ./install_llama_packer.sh
   ```
4. **Run**: `./install_llama_packer.sh`

### Customization Guide

| Variable | Default | What It Controls |
|---|---|---|
| `INSTALL_ROOT` | `~/llama-packer` | Where the repo is cloned; all paths are derived from this |
| `LLAMA_BIN_DIR` | _(auto)_ | Points directly at a directory with `llama-server`/`llama-fit-params`. Bypasses `llama-b*/` detection. |
| `MODELS_DIR` | `~/models` | Default model directory for `llama-packer --models-dir` |
| `HF_HOME` | `~/.cache/huggingface` | Hugging Face cache root; also becomes `${HF_HOME}` macro in generated config |
| `RUN_UPDATER` | `1` | Whether to auto-run `extras/update` to download llama.cpp binaries |
| `USE_LOCKFILE` | `1` | If `1`, uses `uv.lock` for deterministic installs; if `0`, resolves fresh |
| `UV_CACHE_DIR` | `~/.cache/uv` | uv's cache directory for downloaded packages |
| `UV_BINARY` | `uv` | Path to the `uv` binary if not on PATH |

---

## What Each Command Does

### `uv run --with git+https://github.com/JeffKwasha/llama-packer.git llama-packer --dry-run`

Creates a temporary virtual environment, installs `llama-packer` and all its dependencies from GitHub, and runs the tool. No persistent installation. Best for one-off use or testing. The `--dry-run` flag prints the generated YAML to stdout instead of writing files.

### `uv pip install -e .`

Installs the package in **editable** mode. The console script `llama-packer` is created on `PATH` and points to the source tree. Any edits to `.py` files take effect immediately without reinstalling. Dependencies (`pyyaml`, `ruamel-yaml`, `vllm-memory-estimator`) are installed into a virtual environment. This is the recommended persistent installation method.

### `./extras/update`

Downloads and installs prebuilt `llama.cpp` and `llama-swap` binaries from GitHub releases into the `llama-b*/` directories. Required for `llama-packer` to find the `llama-server` binary (via `find_bin_dir`). Uses `GITHUB_TOKEN` or `GITHUB_PERSONAL_ACCESS_TOKEN` environment variables for higher API rate limits. Supports `--cpp-only`, `--swap-only`, `--vulkan`, `--rocm`, and specific version tags.

### `llama-packer --dry-run`

Scans the `models/` directory (or configured `models_dirs`), reads `.md` sidecar files, detects GPU VRAM, budgets memory across models, and prints the generated `config.yaml` to stdout. Does not write any files. Use to verify configuration before deploying.

### `llama-packer --agents`

Writes an `AGENTS.md` guide into each models directory from the bundled template, only if the file doesn't already exist. Documents discovery and sidecar conventions for AI coding agents.

---

## What Is Needed To Put llama-packer on PyPI

### Current Obstacles

1. **Dynamic version with no source**: `pyproject.toml` has `dynamic = ["version"]` resolved via `attr = "llama_packer.__version__"`. This works locally but PyPI requires a static version or a proper build backend that can compute it during build.

2. **Git dependency not on PyPI**: `vllm-memory-estimator` is declared as a git source in `[tool.uv.sources]`. PyPI cannot resolve git dependencies. This must either be published to PyPI or replaced with a `requirements.txt`-style mechanism.

3. **No publishing infrastructure**: No `.github/` workflows, no `twine` configuration, no `~/.pypirc`, no `build` tool configuration.

4. **No CI/CD pipeline**: Without automated testing and publishing, every release would be manual.

### Step-by-Step PyPI Publishing Checklist

1. **Fix the version mechanism**: Add a static version or use `setuptools-scm` / `hatchling` to compute it during build. The current `dynamic = ["version"]` with `attr = "llama_packer.__version__"` may work with setuptools but needs testing under `python -m build`.

2. **Resolve the git dependency**: Publish `vllm-memory-estimator` to PyPI, or add it as a regular PyPI dependency. Alternatively, use `[project.optional-dependencies]` with a `pip`-installable fallback.

3. **Add build tooling**:
   ```bash
   pip install build twine
   ```

4. **Create a `.pypirc`** (or use environment variables):
   ```ini
   [distutils]
   index-servers = pypi

   [pypi]
   username = __token__
   password = pypi-XXXXXXXXXXXXXXXX
   ```

5. **Build and publish**:
   ```bash
   python -m build
   twine check dist/*
   twine upload dist/*
   ```

6. **Add GitHub Actions workflow** for automated publishing on tags.

---

## Comparison: PyPI vs. GitHub-Direct Installation

### Distribution Comparison

| Aspect | PyPI | GitHub-Direct |
|---|---|---|
| **Installation command** | `uvx llama-packer` | `uv pip install -e git+https://...` or `uv run --with git+...` |
| **Version pinning** | `uvx llama-packer==0.3.0` | `git+https://...@v0.3.0` |
| **Dependency resolution** | Automatic via PyPI | Automatic via `pyproject.toml` + `uv.lock` |
| **Binary artifacts** | Wheels uploaded to PyPI | No wheels; source + deps resolved |
| **Update mechanism** | `uv pip install --upgrade llama-packer` | `git pull && uv pip install -e .` |
| **Offline capability** | Cache wheels locally | Clone repo once; no network for subsequent runs |
| **Build from source** | Wheels are pre-built | Must build from source on every install (or use `uv.lock`) |

### Security Analysis

#### PyPI Security Model

| Factor | Assessment |
|---|---|
| **Trust model** | Trust PyPI infrastructure (warehouse.pypa.io), the package maintainer's PyPI account, and PyPI's 2FA enforcement. |
| **Supply chain** | PyPI supports publishing via trusted publishing (OIDC) from GitHub Actions, eliminating credential exposure. |
| **Tamper resistance** | Once uploaded, package files are immutable. PyPI has logged-in-only uploads, 2FA enforcement for organizations, and project-level role management. |
| **Provenance** | PyPI supports `sigstore`/`cosign` attestations (PEP 458) and the PyPI Attestations feature for verifiable builds. |
| **Key risk** | Credential theft: if a PyPI account is compromised, malicious packages can be uploaded. Mitigated by 2FA and IP whitelisting. |
| **Key risk** | Dependency confusion: package names can be squatted. `llama-packer` is available since it's not on PyPI, but future squatters could register it. |
| **Mitigation** | Use `--require-hashes` or `uv`'s lockfile verification to pin exact versions and hashes. |

#### GitHub-Direct Security Model

| Factor | Assessment |
|---|---|
| **Trust model** | Trust the GitHub repository maintainer, GitHub's infrastructure, and the absence of malicious forks. |
| **Supply chain** | No provenance. Anyone with write access to the repo can push changes that immediately affect `git+https://...` installs. No attestation, no review gate. |
| **Tamper resistance** | Git commits are cryptographically signed only if the maintainer configures GPG signing. Unsigned commits are common. Tags can be forged if GitHub credentials are compromised. |
| **Provenance** | None by default. No SBOM, no build attestations, no checksum verification for the installed package. |
| **Key risk** | Repository compromise: if the GitHub account is hacked, malicious code can be pushed and immediately pulled by all users installing from `main`. |
| **Key risk** | No access control on the `git+https` URL — anyone can clone, but only collaborators can push. Public repos allow anyone to create issues, open PRs, and potentially inject code via merge. |
| **Key risk** | `extras/update` downloads binaries from GitHub releases without signature verification (the script explicitly notes this gap: "PGP signature check: not implemented"). |
| **Mitigation** | Pin to a specific commit hash or tag, not `main`. Verify commit signatures with `git verify-commit`. Use `uv.lock` to pin exact dependency versions. |

#### Side-by-Side Security Summary

| Security Dimension | PyPI (if published) | GitHub-Direct |
|---|---|---|
| **Code provenance** | Upload attestation, signed wheels | No provenance by default |
| **Immutable releases** | Once uploaded, cannot be changed | Git history is append-only but refs can be force-pushed |
| **2FA enforcement** | PyPI enforces 2FA for publishing | GitHub 2FA optional; depends on maintainer |
| **Dependency integrity** | `uv.lock` pins hashes | `uv.lock` pins hashes |
| **Binary safety** | Wheel is pre-built, uploaded | `extras/update` downloads binaries with no signature check |
| **Supply chain attacks** | Lower risk: trusted publisher, PyPI infrastructure | Higher risk: depends on GitHub account security, no provenance |
| **Transparency** | All uploads logged, public audit trail | Git history visible, but no formal attestation |
| **Speed of compromise** | Slow (requires account takeover + PyPI 2FA bypass) | Fast (single compromised credential pushes to main) |

### Recommendation

- **For personal/development use**: GitHub-direct is fine. Pin to a specific tag, verify commits if security is a concern, and use `uv.lock`.
- **For production/deployment**: PyPI would be preferable if published, due to its stronger provenance and immutability guarantees. Alternatively, pin GitHub installs to a signed tag and verify with `git verify-commit`.
- **For the `extras/update` script**: Add GPG signature verification for downloaded binaries. The script currently acknowledges this gap explicitly.

---

## Path Macros in Generated Config

When `llama-packer` runs, it generates `config.env` with path macros used by `config.yaml` commands. These macros are derived from the actual file paths:

- `${MODELS_DIR}` — the models directory path
- `${HF_HOME}` — Hugging Face cache root
- `${LLAMA_DIR}` — the `llama-b####` directory containing `llama-server`
- Custom macros for each model file, chat template, LoRA, etc.

These are referenced in the generated `config.yaml` `cmd` fields and in the `docker-compose.yml` volume bindings (e.g., `-v ${MODELS_DIR}:/models`). The `config.env` file is designed for use with `systemd EnvironmentFile=` or Docker `--env-file`.

---

## Troubleshooting

### `uvx llama-packer` fails with "Package not found"

Expected. The package is not on PyPI. Use `uv run --with git+https://github.com/JeffKwasha/llama-packer.git` or `uv pip install -e .` instead.

### `llama-packer` exits with "no llama-b#### directory found"

The tool needs `llama-server` and `llama-fit-params` binaries. Either:
1. Run `./extras/update` to download them.
2. Set `LLAMA_BIN_DIR` to an existing directory containing both binaries.
3. Pass `--llama-server /path/to/llama-server` on the CLI.

### `uv pip install -e .` fails on `vllm-memory-estimator`

The `vllm-memory-estimator` dependency is fetched from a git repository. Ensure network access to GitHub. If the git source is unavailable, install it manually:
```bash
uv pip install git+https://github.com/ashishkamra/vllm-memory-estimator.git
```

### No models found

Ensure `models/` exists and contains `.gguf` files with `.md` sidecar files, or create sidecar stubs with `llama-packer` (stubs are auto-generated unless `--no-stubs` is passed).
