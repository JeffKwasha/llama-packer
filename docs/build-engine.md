# Building an engine from source

`extras/build_engine.py` builds an engine from its upstream source and
installs it beside the prebuilt one. It exists for the engines whose prebuilt
binaries do not fit this box — a build for *this* card, with the libraries it
needs. The rules below are the ones its tests hold up.

## Naming

- **A built engine is named apart from a downloaded one**: the versioned name
  carries `src` (`audiocpp_server-src-v0.9.2-hip-gfx1201`). A built artifact
  and a verified download of the same version are different things, and
  conflating them is how a box ends up serving the wrong one after a
  rollback.
- **The architecture is in the name only when there is one**: a build with no
  GPU target carries no target segment.
- **A patched engine says so in its name**, for the same reason.

## The plan

- **The plan pins the architecture and asks for a deployment build.** The
  same plan is what `--dry-run` prints, and a dry run downloads nothing.
- **The source URL is the codeload endpoint** for the requested ref.
- **The build runs the upstream script**, not a reimplementation of it.

## Finding the target

- **The targets come from the local AMD device** (`amdgpu arch`); `rocminfo`
  is the fallback, and **only full lines count**.
- **No device means no target**, not a guess.
- **The binary's own backends are probed**: a build for the wrong
  architecture is refused, and `vulkan` and `hip` are not interchangeable.
- **A CPU build is exempt from the device check** — it is not going to use
  the card.

## The toolchain

- **The toolchain is checked before anything is fetched**, so a missing
  compiler costs a message rather than a download.
- **A ROCm that keeps clang under LLVM is accepted.**
- **`hipconfig` is consulted before giving up** on the compiler's location.

## Patching

- **The patches are announced before the build** and **recorded next to the
  installed engine**: a patched tree is no longer upstream's tree, and the
  patch must never be invisible.
- **A patch whose pattern no longer matches is reported, not fatal** — and
  neither is a patch file that has gone missing. Upstream fixing the problem
  is the good outcome, not a failure.

## Installing

- **The whole build tree is collected**, because the libraries have to travel
  with the binary.
- **A build tree without the binary is an error.**
- **A good build installs beside the prebuilt one and moves the link**; a
  **failed build leaves the live symlink alone.**
- **A build that cannot see the card is never installed.**
- **An existing build is not replaced without asking**; `--rebuild` replaces
  the tree.
- **A real file where the symlink belongs is not overwritten.**
- **Installing an existing build tree does not recompile it**, and
  `--from-build` still refuses a cardless binary.
