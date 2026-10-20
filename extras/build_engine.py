#!/usr/bin/env python3
"""build — compile a serving engine from source into its install directory.

``extras/update.py`` installs prebuilt binaries; this compiles one. The two
differ in every way that matters, and pretending otherwise is how a fleet ends
up serving a binary nobody can account for:

* **Verification.** Upstream publishes sha256 manifests for two of the five
  engines and nothing at all for audio.cpp's source tarball. A download can be
  checked; a source tree cannot. What stands in for a checksum here is the
  compiler plus a smoke test — the build must produce a binary that enumerates
  the backend it was built for, or it never becomes the live symlink.
* **Time.** A build is minutes, not seconds, and it needs a toolchain the
  download path never touches. So it is a separate command, never something
  ``update.py`` does behind your back, and ``--dry-run`` prints the whole plan
  (fetch URL, cmake invocation, where the artifacts land) before spending it.
* **Shape.** The result is installed exactly like a download: a versioned entry
  beside the engine's stable symlink, in the same directory ``engines.ini``
  names, pruned by the same rules. A source build and a prebuilt one are
  interchangeable from every other tool's point of view.

Why an engine needs this at all: the R9700 is **gfx1201**, which the stock
ROCm packages do not carry an offload binary for, so the ``hip`` build
silently produces a server that sees no device. Building for the local arch is
the only way to get one — ``--gpu-targets gfx1201``, or let
:func:`detect_gpu_targets` read it off ``amdgpu-arch``/``rocminfo``.

Usage:
    ./extras/build_engine.py audio                       # rocm/hip, local arch
    ./extras/build_engine.py audio --backend vulkan      # portable AMD path
    ./extras/build_engine.py audio --dry-run             # print the plan only
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path

import update as upd

HERE = Path(__file__).resolve().parent
CONFIG_NAME = upd.CONFIG_NAME


class BuildError(RuntimeError):
    """A build that cannot proceed, or that produced something unusable."""


@dataclass(frozen=True)
class SourceEngine:
    """An engine whose server is built from a source tarball."""

    key: str
    label: str
    repo: str
    script: str                  # upstream's own build wrapper
    primary: str                 # the binary this installs (stable symlink)
    artifact_root: str           # where the built tree lands, under build/
    backends: tuple[str, ...]    # what --backend accepts
    default_backend: str


#: audio.cpp is the only engine here, and upstream ships the build wrapper that
#: already knows how to hand ROCm its own clang and to turn a GPU name into a
#: ``gfxXXXX`` target. Re-implementing either would be a worse version of the
#: same logic. ``artifact_root`` is ``CMAKE_RUNTIME_OUTPUT_DIRECTORY``: upstream
#: puts binaries *and* libraries in ``build/bin``, which is why the whole
#: directory is collected rather than one file — a deployment build resolves
#: its ``libggml-hip``/``libggml-vulkan`` from there.
AUDIO = SourceEngine(
    key="audio",
    label="audio.cpp",
    repo="0xShug0/audio.cpp",
    script="scripts/build_linux.sh",
    primary="audiocpp_server",
    artifact_root="bin",
    backends=("hip", "rocm", "vulkan", "cpu", "cuda"),
    default_backend="hip",
)

ENGINES = {e.key: e for e in (AUDIO,)}


# --- toolchain -------------------------------------------------------------


def rocm_home() -> Path:
    return Path(os.environ.get("ROCM_PATH")
                or os.environ.get("HIP_PATH") or "/opt/rocm")


def rocm_clangxx() -> Path | None:
    """ROCm's own clang++, resolved the way upstream's script resolves it.

    That order matters and is not obvious from a directory listing:
    ``$ROCM_HOME/bin/clang++`` first, then ``$(hipconfig -l)/clang++``, then
    ``$ROCM_HOME/llvm/bin/clang++``. A ROCm install laid out by version
    (``/opt/rocm/core-10.1/lib/llvm/bin/clang++``, which is where the only
    ROCm on this box puts it) has *no* ``bin/clang++`` at the top level — so a
    check that only looked there would declare a working toolchain missing and
    refuse a build that would have succeeded.

    One deliberate difference from upstream: when ``ROCM_PATH`` is set it is
    honoured exclusively. Upstream's ``$(hipconfig -l)`` follows whatever
    ``hipconfig`` is on ``PATH``, which on a box with more than one ROCm can
    silently validate one toolchain and build against another. Naming an
    install has to mean that install.
    """
    home = rocm_home()
    candidates = [home / "bin" / "clang++"]
    own_hipconfig = home / "bin" / "hipconfig"
    if os.environ.get("ROCM_PATH") or os.environ.get("HIP_PATH"):
        hipconfigs = [str(own_hipconfig)] if own_hipconfig.is_file() else []
    else:
        hipconfigs = [p for p in (shutil.which("hipconfig"),
                                  str(own_hipconfig) if own_hipconfig.is_file()
                                  else "") if p]
    for hipconfig in hipconfigs:
        try:
            libdir = subprocess.run([hipconfig, "-l"], capture_output=True,
                                    text=True, timeout=60,
                                    check=False).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            continue
        if libdir:
            candidates.append(Path(libdir) / "clang++")
    candidates.append(home / "llvm" / "bin" / "clang++")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def toolchain_problems(engine: SourceEngine, backend: str) -> list[str]:
    """What is missing for this build, as sentences. Empty means ready.

    Checked *before* a source tree is fetched so the failure costs seconds.
    Deliberately not fatal on its own: a caller may be building for another
    machine, and the build itself will say the same things with better
    context. ``--dry-run`` prints them as a warning.
    """
    problems: list[str] = []
    if shutil.which("cmake") is None:
        problems.append("cmake is not on PATH")
    if backend == "hip":
        home = rocm_home()
        if not home.is_dir():
            problems.append(f"no ROCm at {home} (set ROCM_PATH)")
        elif rocm_clangxx() is None:
            problems.append(
                f"no ROCm clang++ under {home}, via hipconfig, or at "
                f"{home / 'llvm' / 'bin' / 'clang++'} (set ROCM_PATH)")
    return problems


def detect_gpu_targets() -> str:
    """``gfx1201`` from the local AMD device(s), semicolon-separated.

    Mirrors upstream's ``build_linux.sh`` detection order — ``amdgpu-arch``
    first, ``rocminfo`` second — and its refusal to accept anything weaker:
    only a full ``Name: gfxXXXX`` agent line counts, because a loose grep also
    matches ``gfx11`` fragments in ISA-description prose and building for a
    prefix produces a binary that loads and then faults.
    """
    home = rocm_home()
    amd_arch = home / "bin" / "amdgpu-arch"
    if amd_arch.is_file() and os.access(amd_arch, os.X_OK):
        found = subprocess.run([str(amd_arch)], capture_output=True, text=True,
                               timeout=120).stdout
        targets = sorted({t for t in found.split() if re.fullmatch(r"gfx[0-9a-f]{3,}", t)})
        if targets:
            return ";".join(targets)
    rocminfo = home / "bin" / "rocminfo"
    if not rocminfo.is_file():
        rocminfo = shutil.which("rocminfo") or ""
    if rocminfo:
        found = subprocess.run([rocminfo], capture_output=True, text=True,
                               timeout=180).stdout
        targets = sorted(set(re.findall(r"^\s*Name:\s*(gfx[0-9a-f]{3,})\s*$",
                                        found, re.M)))
        if targets:
            return ";".join(targets)
    return ""


# --- the plan --------------------------------------------------------------


def source_url(engine: SourceEngine, ref: str) -> str:
    """The source tarball URL for *ref*.

    ``codeload`` rather than the ``archive/refs/tags`` redirect: it is a plain
    200 with the tarball, so a failed download is a curl error instead of an
    HTML login page landing in the staging directory.
    """
    return f"https://codeload.github.com/{engine.repo}/tar.gz/refs/tags/{ref}"


def versioned_name(engine: SourceEngine, ref: str, backend: str,
                   targets: str = "") -> str:
    """``audiocpp_server-src-v0.9.2-hip-gfx1201``.

    Marked ``src`` so a built engine is never mistaken for a verified download
    by whoever reads the directory later, and carries the backend and arch it
    was built for: ``audiocpp_server-v0.9.2`` from the prebuilt path and this
    are different artifacts of the same version, and conflating them is how a
    box ends up serving the wrong one after a rollback.
    """
    name = f"{engine.primary}-src-{ref}-{backend}"
    if targets:
        name += "-" + targets.replace(";", "+")
    return name


def build_argv(engine: SourceEngine, backend: str, targets: str, jobs: int,
               source_dir: Path, build_dir: Path,
               extra: list[str] | None = None) -> list[str]:
    """The upstream build invocation, verbatim in spirit.

    Delegated rather than reconstructed: the wrapper handles ROCm's own clang,
    the sticky-cache ``-U`` on GPU targets, and defaulting HIP builds to
    deployment builds. Re-implementing that here would be a second, worse copy
    of upstream's build policy.
    """
    source_dir = Path(source_dir)
    argv = [str(source_dir / engine.script), "--backend", backend,
            "--build-dir", str(build_dir), "--deployment-build"]
    if targets:
        argv += ["--gpu-targets", targets]
    if jobs:
        argv += ["--jobs", str(jobs)]
    if extra:
        argv += list(extra)
    return argv


def collect_artifact(build_dir: Path, engine: SourceEngine) -> Path:
    """The built tree to install, verified to contain the server.

    Raises rather than returning a best guess: an install directory without
    the binary is a dead symlink, and the promotion that follows would make it
    live.
    """
    root = build_dir / engine.artifact_root
    binary = root / engine.primary
    if not binary.is_file():
        raise BuildError(
            f"build produced no {engine.primary} in {root} "
            f"(found: {sorted(p.name for p in root.glob('*'))[:12] if root.is_dir() else 'no directory'})")
    return root


# --- smoke test ------------------------------------------------------------

#: ``--list-devices`` output differs by backend *and* by engine, and both
#: spellings turn up in practice:
#:
#:     Vulkan0: AMD Radeon AI Pro R9700 (RADV GFX1201) (32624 MiB, ...)  llama-server
#:     ROCm:0 "AMD Radeon AI PRO R9700" [GPU]                            audio.cpp, HIP
#:     CPU:0 "AMD Ryzen 7 5700X3D" [CPU]
#:
#: so the separator can sit on either side of the index. Accepting only one
#: form reported a working HIP build as seeing no devices — which is the
#: failure this script exists to catch, produced by the check itself.
_DEVICE_LINE = re.compile(
    r"^\s*(Vulkan|HIP|CUDA|ROCm|CPU|OpenCL)\s*:?\s*(\d+)\s*[:\s]", re.M)

#: ``available_devices=2`` — audio.cpp's own count. Checked alongside the
#: per-device lines because it is the one unambiguous signal that at least
#: one device was found, whichever spelling the engine prefers.
_DEVICE_COUNT = re.compile(r"^available_devices\s*=\s*(\d+)", re.M)

#: What each ``--backend`` value must be able to enumerate, as the name the
#: probe prints and as a word an error message can use.
_BACKEND_PROBE_NAME = {"hip": "hip", "vulkan": "vulkan", "cuda": "cuda"}


def probe_backends(binary: Path, timeout: int = 120) -> set[str]:
    """Backend names the binary can actually enumerate a device on.

    This is the check that makes a source build safe to trust as an install: a
    binary that compiles and then sees nothing — the gfx1201 failure this whole
    script exists for — fails here instead of in a serving request.
    """
    try:
        result = subprocess.run([str(binary), "--list-devices"],
                                capture_output=True, text=True, timeout=timeout,
                                check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BuildError(f"{binary.name} --list-devices failed: {exc}") from exc
    out = f"{result.stdout}\n{result.stderr}"
    found = {m.group(1).lower() for m in _DEVICE_LINE.finditer(out)}
    # "ROCm" is how audio.cpp names HIP; the caller asks in HIP terms.
    if "rocm" in found:
        found.add("hip")
    if _DEVICE_COUNT.search(out) and not found:
        # A count with no recognisable per-device line: better to say so than
        # to report an empty set that reads as "no devices".
        raise BuildError(
            f"{binary.name} reported devices but none in a form this script "
            f"recognises; the output format has probably changed:\n"
            + "\n".join(out.splitlines()[:12]))
    return found


def check_backends(binary: Path, backend: str, found: set[str]) -> None:
    """Refuse a build that cannot serve the backend it was built for.

    ``cpu`` is exempt: it always enumerates something, and that something is
    the fallback the engine uses when nothing else is present — a cpu build is
    not a mistake, it is the portable one.
    """
    want = _BACKEND_PROBE_NAME.get(backend)
    if want is None or want in found:
        return
    raise BuildError(
        f"built {binary.name} enumerates no {want.upper()} device "
        f"(reports: {', '.join(sorted(found)) or 'nothing'}). This is what a "
        "build for the wrong GPU architecture looks like: it compiles, and it "
        "cannot see the card. Check --gpu-targets against "
        "`rocminfo | grep gfx`, and note that stock ROCm packages carry "
        "no offload binary for recent gfx targets.")


# --- patching the extracted source ----------------------------------------


@dataclass(frozen=True)
class Patch:
    """One recorded edit to an extracted source tree.

    A patched tree is no longer upstream's tree, so the rule here is that the
    patch is never invisible: it is announced before the build, written into
    the log, and recorded next to the installed engine. Someone reading the
    directory later has to be able to tell that the binary they are running is
    not bit-for-bit what the tag says it is.
    """

    name: str
    what: str
    why: str
    path: str                                   # relative to the source root
    find: str                                   # regex, first match only
    replace: str


#: audio.cpp v0.9.2 (and main) compiles its app code as C++17 with
#: ``-Wpedantic -pedantic-errors``, while ``app/server/parallel_runtime.cpp``
#: captures structured bindings inside a lambda — a C++20 extension. GCC
#: tolerates that; ROCm's clang++ treats it as an error, so a HIP build of the
#: tag fails at ``parallel_runtime.cpp:2559`` with ``captured structured
#: bindings are a C++20 extension``.
#:
#: Raising the standard instead — the obvious move, and the first thing tried
#: here — is *worse*: the same tree compares ``std::string_view`` against ``u8``
#: literals, and in C++20 those are ``const char8_t[]``, which no longer
#: converts. Bumping the standard trades one error for seven. So the standard
#: stays at 17 and the single diagnostic that flags the C++20-ism is disabled;
#: the code is otherwise valid C++17, which is what every other compiler on
#: this box already builds it as.
#:
#: Clang-only, and deliberately narrow: ``-Wno-c++20-extensions`` removes the
#: warning that ``-Werror`` was tripping on, and nothing else.
SOURCE_PATCHES = (
    Patch(
        name="clang-cxx20-diagnostic",
        what="CMakeLists.txt: -Wno-c++20-extensions for Clang",
        why="app/server/parallel_runtime.cpp captures structured bindings in "
            "a lambda (a C++20 extension) while the project builds that file "
            "as C++17 with -pedantic-errors; ROCm's clang++ rejects it and "
            "GCC does not. Raising CMAKE_CXX_STANDARD to 20 does not work — "
            "the same tree compares std::string_view against u8 literals, "
            "which only converts before C++20",
        path="CMakeLists.txt",
        find=r"(set\(CMAKE_CXX_STANDARD 17\))",
        replace=r"\1\n# added by llama-packer's build_engine.py: see the\n"
                r"# .patched record beside the installed engine\n"
                r"add_compile_options("
                r"$<$<COMPILE_LANG_AND_ID:CXX,Clang>:-Wno-c++20-extensions>)",
    ),
)


def apply_patches(source_dir: Path,
                  patches: tuple[Patch, ...] = SOURCE_PATCHES) -> list[str]:
    """Apply *patches*, returning one line per patch actually applied.

    A patch whose pattern does not match is **not** an error: upstream fixing
    their own tree should make this a no-op, not a build that refuses to run.
    That case is worth saying out loud, though, since it means the version of
    the flag this recorded was written against has changed.
    """
    applied: list[str] = []
    for patch in patches:
        target = source_dir / patch.path
        if not target.is_file():
            applied.append(f"{patch.name}: SKIPPED ({patch.path} is gone)")
            continue
        text = target.read_text(errors="replace")
        new_text, count = re.subn(patch.find, patch.replace, text, count=1)
        if not count:
            applied.append(f"{patch.name}: SKIPPED (pattern no longer matches "
                           f"in {patch.path} — upstream may have changed it)")
            continue
        target.write_text(new_text)
        applied.append(f"{patch.name}: {patch.what}\n"
                       f"    because: {patch.why}")
    return applied


# --- fetch + build ---------------------------------------------------------


def fetch_source(engine: SourceEngine, ref: str, dest: Path) -> Path:
    """Download and unpack the source tree into *dest*; return its root.

    No checksum step, because there is nothing to check against — upstream
    publishes no manifest for the source tarball. The pinned tag URL over TLS
    is the same guarantee the prebuilt path offers for audio.cpp, and no more.
    """
    url = source_url(engine, ref)
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "source.tar.gz"
    print(f"Fetching {engine.label} {ref} source...")
    try:
        upd.download(url, archive, quiet=False)
    except subprocess.CalledProcessError as exc:
        raise BuildError(f"download failed: {url} ({exc})") from exc
    tree = dest / "tree"
    try:
        upd.extract_archive(archive, tree, "tar.gz")
    except (tarfile_errors(), OSError) as exc:      # noqa: F821 - see below
        raise BuildError(f"could not unpack {archive.name}: {exc}") from exc
    entries = [p for p in tree.iterdir() if p.is_dir()]
    if len(entries) != 1:
        raise BuildError(f"source tarball has {len(entries)} top-level "
                         f"directories, expected 1")
    return entries[0]


def tarfile_errors() -> tuple[type[BaseException], ...]:
    import tarfile
    return (tarfile.TarError,)


def run_build(argv: list[str], log: Path, cwd: Path) -> int:
    """Run the upstream build script, streaming its output to *log*.

    Long by nature: a full audio.cpp build is minutes. The log is the record —
    a failed configure prints its actual complaint there and nothing in this
    script can improve on it — so its path is printed before the build starts
    and the last lines are echoed on failure.
    """
    print(f"Building: {' '.join(argv)}")
    print(f"Log: {log}")
    started = time.time()
    with log.open("w") as handle:
        handle.write(f"$ {' '.join(argv)}\n\n")
        handle.flush()
        proc = subprocess.run(argv, stdout=handle, stderr=subprocess.STDOUT,
                              cwd=str(cwd), check=False)
    took = time.time() - started
    print(f"Build {'succeeded' if proc.returncode == 0 else 'FAILED'} "
          f"in {took:.0f}s")
    if proc.returncode:
        for line in log.read_text(errors="replace").splitlines()[-25:]:
            print(f"  | {line}")
    return proc.returncode


# --- install ---------------------------------------------------------------


def install_built(engine: SourceEngine, install_dir: Path, ref: str,
                  backend: str, targets: str, jobs: int,
                  rebuild: bool = False, keep_build: bool = False,
                  extra: list[str] | None = None,
                  from_build: Path | None = None) -> Path:
    """Build, verify, and publish. The live symlink moves only at the end.

    Every step happens in a staging directory on the destination filesystem,
    for the reason :func:`update.staging` gives: promotion is then a rename
    rather than a copy across filesystems, so the versioned name is never
    observed half-populated, and the compiler writes into the filesystem that
    will hold the result rather than a tmpfs.

    A build that fails leaves the live symlink exactly where it was. That is
    the whole reason the smoke test runs before the promote and not after: a
    binary that compiles for the wrong GPU architecture is a perfectly good
    binary that cannot see the card, and installing it is worse than not
    building it.

    ``from_build`` installs a tree that is already compiled — the case where
    the compile took twenty minutes and only the smoke test needs re-running,
    which happens every time the check itself turns out to be wrong.
    """
    versioned = versioned_name(engine, ref, backend, targets)
    final = install_dir / versioned
    if final.is_dir() and not rebuild:
        print(f"{engine.label} {versioned} is already installed at "
              f"{final.name} (--rebuild to replace it)")
        return final

    with upd.staging(install_dir, f"build-{engine.primary}") as stage:
        if from_build is not None:
            artifact = Path(from_build)
            if not (artifact / engine.primary).is_file():
                raise BuildError(
                    f"{artifact} has no {engine.primary} to install")
            print(f"Using the existing build at {artifact} (not recompiling)")
            patched = [f"installed from a prebuilt tree: {artifact}"]
        else:
            source_dir = fetch_source(engine, ref, stage / "src")
            patched = apply_patches(source_dir)
            print("Source patches applied: "
                  f"{len([p for p in patched if 'SKIPPED' not in p])}"
                  f"/{len(SOURCE_PATCHES)}")
            for line in patched:
                print(f"  - {line}")

            build_dir = source_dir / "build"
            log = stage / "build.log"
            argv = build_argv(engine, backend, targets, jobs, source_dir,
                              build_dir, extra)
            code = run_build(argv, log, cwd=source_dir)
            # The log outlives the staging directory on purpose: a build that
            # failed is the case where someone needs to read it, and the
            # staging tree is removed on exit.
            saved_log = install_dir / f"{versioned}.build.log"
            shutil.copyfile(log, saved_log)
            print(f"Build log: {saved_log}")
            if code != 0:
                raise BuildError(f"build failed (exit {code}); see {saved_log}")
            if keep_build:
                kept = install_dir / f"{versioned}.sourcetree"
                print(f"Keeping source and build tree at {kept}")
                shutil.copytree(source_dir, kept, symlinks=True)
            artifact = collect_artifact(build_dir, engine)

        binary = artifact / engine.primary
        found = probe_backends(binary)
        print(f"Device probe: {sorted(found) or 'no devices'}")
        check_backends(binary, backend, found)

        installed = stage / "payload"
        shutil.copytree(artifact, installed)
        # ...and the list of edits, so the installed tree is never mistaken
        # for an unmodified checkout of the tag.
        (install_dir / f"{versioned}.patched").write_text(
            f"{engine.label} {ref} built from source with these edits:\n\n"
            + "\n".join(patched) + "\n")
        upd.promote(installed, final, stage)

    link = install_dir / engine.primary
    if link.exists() and not link.is_symlink():
        raise BuildError(
            f"{link} is a real file, not a symlink; refusing to replace it "
            "(move it aside first)")
    upd.replace_symlink(link, f"./{versioned}/{engine.primary}")
    print(f"{engine.primary} -> ./{versioned}/{engine.primary}")
    return final


# --- CLI -------------------------------------------------------------------


def resolve_dir(args: argparse.Namespace) -> Path:
    """Install directory, from the same rules update.py uses.

    ``--dir`` here names a bare path rather than update.py's ``engine=path``:
    the engine is already the positional argument, so repeating it would be
    noise. It is turned into update.py's form and handed to the same resolver,
    which is what keeps one description of "where does this engine live" in
    the codebase instead of two that drift.
    """
    overrides = upd.parse_dir_overrides(
        [f"{args.engine}={args.dir}"] if args.dir else [])
    root, per_engine = upd.load_engines_config()
    return upd.resolve_install_dirs(overrides, (root, per_engine))[args.engine][0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\nUsage")[0],
        epilog=(
            "Example:\n"
            "  ./extras/build_engine.py audio --dry-run\n"
            "  ./extras/build_engine.py audio --backend hip --gpu-targets gfx1201\n"
            "\n"
            "Environment:\n"
            "  ROCM_PATH / HIP_PATH   ROCm root (default /opt/rocm)\n"
            "  KEEP_BUILD             Set to keep the source tree for debugging\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("engine", choices=sorted(ENGINES),
                        help="engine to build from source (currently: audio)")
    parser.add_argument("--backend", default="",
                        choices=ENGINES[AUDIO.key].backends,
                        help="hip/rocm (default), vulkan, cuda or cpu. HIP "
                             "is the one that needs a source build on recent "
                             "AMD cards: the R9700 is gfx1201, which stock "
                             "ROCm packages ship no offload binary for.")
    parser.add_argument("--gpu-targets", default="",
                        help="semicolon-separated gfx targets "
                             "(default: detected from amdgpu-arch/rocminfo)")
    parser.add_argument("--ref", default="",
                        help="tag to build (default: latest release tag)")
    parser.add_argument("--jobs", type=int, default=0,
                        help="parallel build jobs (default: all cores)")
    parser.add_argument("--dir", metavar="PATH",
                        help="install directory (default: what "
                             f"{CONFIG_NAME} and discovery resolve)")
    parser.add_argument("--rebuild", action="store_true",
                        help="replace an existing build of the same name")
    parser.add_argument("--from-build", metavar="DIR",
                        help="install an already-compiled build/bin tree "
                             "instead of fetching and compiling again — for "
                             "re-running the device check without paying "
                             "for a rebuild")
    parser.add_argument("--keep-build", action="store_true",
                        help="keep the source and build trees")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and exit")
    parser.add_argument("--build-flag", action="append", default=[],
                        metavar="FLAG",
                        help="extra flag for the upstream build script "
                             "(repeatable)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    engine = ENGINES[args.engine]
    backend = args.backend or engine.default_backend
    if backend == "rocm":
        backend = "hip"
    targets = args.gpu_targets
    if not targets and backend == "hip":
        targets = detect_gpu_targets()

    ref = args.ref
    if not ref:
        try:
            ref = latest_tag(engine)
        except upd.ApiError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    problems = toolchain_problems(engine, backend)
    if backend == "hip" and not targets:
        problems.append("could not detect a GPU target; pass "
                        "--gpu-targets gfx1201 (rocminfo | grep gfx)")

    install_dir = resolve_dir(args)
    versioned = versioned_name(engine, ref, backend, targets)

    print(f"Engine:    {engine.label} {ref} ({backend}"
          f"{' ' + targets if targets else ''})")
    print(f"Install:   {install_dir} / {versioned}")
    print(f"Source:    {source_url(engine, ref)}")
    if args.from_build:
        print(f"Command:   (none — installing {args.from_build})")
    else:
        print(f"Command:   {' '.join(build_argv(engine, backend, targets, args.jobs, '<source>', '<build>', args.build_flag))}")
    for problem in problems:
        print(f"Warning:   {problem}")

    if args.dry_run:
        return 1 if problems else 0
    if problems:
        print("Refusing to start a build that cannot succeed.", file=sys.stderr)
        return 1

    try:
        install_built(engine, install_dir, ref, backend, targets, args.jobs,
                      rebuild=args.rebuild,
                      keep_build=args.keep_build
                      or bool(os.environ.get("KEEP_BUILD")),
                      extra=args.build_flag,
                      from_build=Path(args.from_build)
                      if args.from_build else None)
    except (BuildError, upd.ApiError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


def latest_tag(engine: SourceEngine) -> str:
    payload = upd.gh_api(f"https://api.github.com/repos/{engine.repo}/releases")
    for release in payload:
        tag = release.get("tag_name", "")
        if re.fullmatch(engine_tag_re(engine), tag):
            return tag
    raise upd.ApiError(f"no release tag matching {engine_tag_re(engine)}")


def engine_tag_re(engine: SourceEngine) -> str:
    return r"v[0-9]+(\.[0-9]+)*"


if __name__ == "__main__":
    sys.exit(main())