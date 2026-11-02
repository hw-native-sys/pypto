# Development Container

Pull a prebuilt image that already carries CANN, PyPTO, and the runtime, and run a kernel
on a real Ascend card without assembling a toolchain.

This guide's introductory commands and image facts describe the legacy ARM A2/A3
`a3-dist` image with CANN 9.0.0. For the CANN 9.2 ARM A2/A3, ARM A5 and x86 A5
images, use the tags, drivers and validation cases in
[Maintaining published images in CI](#maintaining-published-images-in-ci).
Those tags become available after their first successful publishing run.

## Concept

[Installation](01-installation.md) builds PyPTO from source and gives you the **compiler
front end** — enough to write kernels and read the IR they lower to. Running one needs
more: a matched CANN, the `simpler` runtime built against it, ptoas, and a device. The
development container is that whole stack, prebuilt.

The split between image and host is the thing to understand before anything else:

| Layer | Who provides it | Notes |
| ----- | --------------- | ----- |
| NPU hardware | Host | Ascend A2 / A3, at least one free card |
| Kernel driver | **Host** | Not in the image, and not a free choice — see below |
| Ascend Docker Runtime | Host | Injects the driver libraries and device nodes into the container |
| CANN user space | Image | 9.0.0 |
| CPython 3.10, torch, ptoas | Image | `/opt/pypto/venv`, `/opt/pypto/ptoas` |
| PyPTO + `simpler`, built | Image | `/workspace/pypto`, already compiled, reachable as `$PYPTO_SRC` |
| `pypto-lib` models and examples | Image | `/workspace/pypto-lib`, reachable as `$PYPTO_LIB_SRC` |

**The host driver version is not a free choice.** The image carries CANN user space but
no driver. The AICPU device-side package is pushed to the device by the container's CANN
at run time, over TSD, and the driver version-checks it before accepting it. That check
is on the AICPU path only — so with a mismatched driver, compilation, memory allocation,
and `aclrtSetDevice` all succeed, and the first AICPU initialization fails with `507018`.
The supported combination is `driver 26.0.rc1 + CANN 9.0.0`, matching the supported
environment the runtime declares in `runtime/docs/install.md`; it is also the only
combination this image has been verified on. [Triaging `507018`](#triaging-507018) covers
what to do when you hit it.

### Image facts

```text
swr.cn-east-3.myhuaweicloud.com/cloud-pypto/pypto-dev:a3-dist
```

| Property | Value |
| -------- | ----- |
| Architecture | ARM64 (`linux/arm64`) — will not run on an x86_64 host |
| Target device | Ascend A2 / A3 (PyPTO platform name `a2a3`) |
| User space | Ubuntu 22.04, CANN 9.0.0 |
| ptoas | 0.61 — what the bundled pypto commit pins; `main` has since moved past it |
| Size, uncompressed | ~16.7 GB |
| Working directory | `/workspace` |
| Entry point | Interactive bash with the PyPTO environment already active |
| Verified digest | `sha256:ffa6bef8ed34f565a2331a66369902303ff957e832476fa1abaad2be435b82b5` |

**The image's ptoas lags `main`.** It matches `toolchain/versions.env` in the pypto commit
the image was built from — the `pypto` line of `dist-manifest` below — not the repository
today. Codegen rejects a ptoas older than the source's pin, so as soon as the source in
the container pins a newer ptoas than the image ships — pulled, or a current checkout
mounted over `/workspace` — every codegen fails until ptoas is reinstalled; see
[Updating the source and rebuilding](#updating-the-source-and-rebuilding). Compare the two:

```bash
grep '^PTOAS_VERSION=' "$PYPTO_SRC/toolchain/versions.env"
/opt/pypto/ptoas/bin/ptoas --version
```

`/opt/pypto/dist-manifest` records exactly what went into the image — the pypto, runtime,
pto-isa, and pypto-lib commits, and when it was built. Quote it when reporting a problem:

```bash
cat /opt/pypto/dist-manifest
```

```text
pypto     6fdd989fdf45b10011c4bd6b0291a7e6c18fd7e4
runtime   39ce891dbb3f665e72e99b4a1387b47012fb18bd
pto-isa   5a4f74cbf627d4aac2e0ce10d5e0d8b118343265
pypto-lib c3f0dea274f55d9648920f17c968e8564ae9fcdc
built     2026-09-14T09:19:43Z
```

Three commands are on `PATH` inside the container, from `/opt/pypto/bin`:

| Command | What it does |
| ------- | ------------ |
| `pypto-doctor` | Checks the toolchain, driver, device access, source tree, and build artifacts, and says what to do about anything that fails |
| `pypto-build` | Rebuilds the checkout at `$PYPTO_SRC` with native CMake, then installs `./runtime` (`simpler`) |
| `pypto-activate.sh` | The environment itself — sourced as the shell's rcfile and as `BASH_ENV`, so `docker exec … bash -c '…'` sees the same environment as a login shell |

The repository now maintains the updater and installer alongside these helpers. Images
rebuilt from it install them on `PATH`; the verified older image above may still carry
its updater only inside the source tree. See
[Updating the source and rebuilding](#updating-the-source-and-rebuilding).

The image also ships a `pto-isa` checkout matching the pinned runtime, so a first run does
not clone it over the network.

## Quickstart

This assumes the host already carries the NPU driver, Docker Engine, and Ascend Docker
Runtime. [Host prerequisites](#host-prerequisites) is the checklist for all three, and
installs whichever is missing.

On the host, find a free card:

```bash
npu-smi info
```

Assuming physical card 1 is free:

```bash
export PYPTO_IMAGE=swr.cn-east-3.myhuaweicloud.com/cloud-pypto/pypto-dev:a3-dist
docker pull "$PYPTO_IMAGE"

docker run --rm -it \
  --runtime=ascend \
  -e ASCEND_VISIBLE_DEVICES=1 \
  --shm-size=16g \
  "$PYPTO_IMAGE"
```

Inside the container, check the environment before running anything:

```bash
pypto-doctor
```

The checks that matter are:

```text
ok    device access: aclrtSetDevice(0) ok, 62748MB HBM visible
ok    import pypto -> /workspace/pypto/python/pypto/__init__.py
ok    simpler binding matches its source tree
environment looks good.
```

The HBM figure depends on the card. Two lines come back as `warn` and are expected under
`--runtime=ascend` — neither means the environment is broken:

```text
warn  driver libs present but version.info is not exposed (normal under --runtime=ascend)
warn  npu-smi: npu get board type failed. ret is -9005 — expected under --runtime=ascend
```

The driver's `version.info` is readable on the host, not in the container, and `npu-smi`
fails because it enumerates every board while only the cards you exposed are present.
`device access: aclrtSetDevice(0) ok` confirms device access; it does not establish
driver compatibility for AICPU initialization. A missing CANN interpreter or an ACL probe
that cannot execute is a blocking doctor failure. When `version.info` is readable,
doctor enforces the runtime's supported pairs: CANN `9.0.0` with driver `26.0.rc1`,
or CANN `9.1.T500` with driver `25.6.rc1.b108`. CANN 9.2 also accepts the
architecture-specific deployment configurations in the CI image table below;
those still require hardware validation. Unknown pairs fail validation.
`ASCEND_DRIVER_ROOT` overrides the default driver directory `/usr/local/Ascend/driver`.

Then run a kernel on the card:

```bash
python $PYPTO_SRC/examples/beginner/01_hello_world.py
```

The final line is:

```text
OK
```

That single line covers the whole path: the parser built IR, the pass pipeline ran, ptoas
assembled the kernel, the runtime dispatched it, and the result matched torch.

`--rm` deletes the container's writable layer on exit. The image stays; anything you
edited inside the container does not. [Keeping your source](#keeping-your-source) fixes
that.

## Mechanics

### Host prerequisites

| Requirement | Expected | Verify with |
| ----------- | -------- | ----------- |
| Architecture | `aarch64` | `uname -m` |
| NPU driver | `Version=26.0.rc1` | `cat /usr/local/Ascend/driver/version.info` |
| Device nodes | `davinci0`, `davinci_manager`, `devmm_svm`, `hisi_hdc` | `ls /dev/davinci*` |
| Device health | `OK` | `npu-smi info` |
| Docker Engine | 27.2.0 verified; others must satisfy the driver and Ascend Docker Runtime | `docker version` |
| Ascend Docker Runtime | `ascend` listed | `docker info --format '{{json .Runtimes}}'` |
| Free disk | ≥ 30 GB for image layers, containers, and build cache | `df -h /var/lib/docker` |

In `version.info`, `Version=` is the public driver version. `compatible_version=` lists
internal release identifiers such as `V100R001C10`, not public CANN version numbers.
Check the supported public driver/CANN pair in `runtime/docs/install.md`.

Install what is missing, in this order — each step depends on the one before it:

| Missing | How to install |
| ------- | -------------- |
| NPU driver | Install the Ascend driver matching the server model and OS, then reboot. It is the one component the image cannot supply, and its version is fixed at `26.0.rc1` by the image's CANN 9.0.0 |
| Docker Engine | Follow the [official Docker Engine installation guide](https://docs.docker.com/engine/install/) for the distribution |
| Ascend Docker Runtime | [Installing Ascend Docker Runtime](#installing-ascend-docker-runtime), below |

### Installing Ascend Docker Runtime

Ascend Docker Runtime is an OCI runtime plugin, analogous to NVIDIA Container Runtime: it
injects the selected NPUs, the driver libraries, and the device nodes into a container.

Download the `.run` matching the host architecture and driver from the Ascend channel,
then install it as root:

```bash
RUNTIME_RUN=Ascend-docker-runtime_<paste the version here>_linux-aarch64.run

chmod +x "$RUNTIME_RUN"
sudo "./$RUNTIME_RUN" --check
sudo "./$RUNTIME_RUN" --install
sudo systemctl daemon-reload
sudo systemctl restart docker
```

Fill the version into the first line before running the rest — unquoted `<` and `>` are
redirection to the shell, so a pasted placeholder truncates the filename instead of
failing loudly.

Restarting Docker interrupts every running container on the host — do it in a maintenance
window. Then confirm registration:

```bash
docker info --format '{{json .Runtimes}}'
```

`ascend` must appear in the output; without it, `--runtime=ascend` will not work. See the
vendor's [manual installation guide](https://www.hiascend.com/document/detail/zh/mindcluster/72rc1/clustersched/dlug/dlug_installation_017.html)
and [Docker client usage](https://www.hiascend.com/document/detail/zh/mindx-dl/500/dockerruntime/dockerruntimeug/dlruntime_ug_013.html).

### Pulling the image

```bash
export PYPTO_IMAGE=swr.cn-east-3.myhuaweicloud.com/cloud-pypto/pypto-dev:a3-dist
docker pull "$PYPTO_IMAGE"
```

The registry allows anonymous pulls, so this needs no `docker login`. It is the slowest
step on this page: the image unpacks to about 16 GB on disk, which is what the 30 GB free
space in the prerequisites is for. You pay it once per host — every later `docker run`,
and every container you create, reuses the local copy.

In the tag `a3-dist`, `a3` is the chip series and `dist` is the fully built variant, the
one this page documents.

Confirm what arrived:

```bash
docker image inspect "$PYPTO_IMAGE" --format 'arch={{.Architecture}} os={{.Os}} size={{.Size}}'
docker image inspect "$PYPTO_IMAGE" --format '{{index .RepoDigests 0}}'
```

For a reproducible environment, address the image by digest instead of by tag — a tag can
be moved. Rebind `PYPTO_IMAGE` rather than pulling the digest separately, so every later
`docker run` on this page uses it too:

```bash
export PYPTO_IMAGE=swr.cn-east-3.myhuaweicloud.com/cloud-pypto/pypto-dev@sha256:ffa6bef8ed34f565a2331a66369902303ff957e832476fa1abaad2be435b82b5
docker pull "$PYPTO_IMAGE"
```

### Selecting cards

`ASCEND_VISIBLE_DEVICES` names **host physical** cards. Ascend Docker Runtime renumbers
them inside the container, always starting at 0:

| Host | `ASCEND_VISIBLE_DEVICES` | Inside the container | PyPTO argument |
| ---- | ------------------------ | -------------------- | -------------- |
| card 1 | `1` | card 0 | `-d 0` |
| cards 2, 3 | `2,3` | cards 0, 1 | `-d 0,1` |
| cards 4-7 | `4,5,6,7` | cards 0-3 | `-d 0,1,2,3` |

So PyPTO commands always count from 0 regardless of which physical cards you exposed.

Never hand the same physical NPU to two containers. Check with `npu-smi info` and
`docker ps` before starting one.

### Multi-card

Expose two free cards and run a minimal collective:

```bash
docker run --rm -it \
  --runtime=ascend \
  -e ASCEND_VISIBLE_DEVICES=2,3 \
  --shm-size=16g \
  "$PYPTO_IMAGE"
```

```bash
python $PYPTO_SRC/examples/distributed/08_allreduce_mesh.py -p a2a3 -d 0,1
```

The final line is `OK`. This exercises HCCL and the cross-rank path, which a single-card
run does not.

Some HCCL / driver / Ascend Docker Runtime combinations misbehave for multi-card programs
inside a container. `driver 26.0.rc1` with this image is verified; if another combination
exits silently, fall back to running on the host as project CI does, and re-check the
driver / CANN / HCCL compatibility matrix.

### Keeping your source

The image already contains built source, so the quickstart needs no mount at all — but
everything you change is lost with the container. Two ways to keep it:

**Named volumes.** On first mount of an empty volume, Docker copies the image's content
into it; the source and caches then survive container deletion.

```bash
docker volume create pypto-a3-src
docker volume create pypto-a3-lib-src
docker volume create pypto-a3-cache

docker run -it \
  --name pypto-a3-dev \
  --runtime=ascend \
  -e ASCEND_VISIBLE_DEVICES=1 \
  --shm-size=16g \
  -v pypto-a3-src:/workspace/pypto \
  -v pypto-a3-lib-src:/workspace/pypto-lib \
  -v pypto-a3-cache:/opt/pypto/cache \
  "$PYPTO_IMAGE"
```

Re-enter the same container with `docker start -ai pypto-a3-dev`. If you delete the
container, a new one over the same three volumes picks up where you left off.

A volume is never refreshed by pulling a newer image: the **environment** moves with the
image, the **source** moves with `git pull` inside the volume.

**Bind mount.** To work on a checkout that lives on the host, the host directory must
already have this shape:

```text
/path/to/workspace/
├── pypto/
└── pypto-lib/
```

```bash
export PYPTO_HOST_WORKSPACE=/path/to/workspace

docker run --rm -it \
  --runtime=ascend \
  -e ASCEND_VISIBLE_DEVICES=1 \
  --shm-size=16g \
  -v "$PYPTO_HOST_WORKSPACE:/workspace" \
  "$PYPTO_IMAGE"
```

Mount the directory that *contains* `pypto/` and `pypto-lib/`, not its parent — the
container needs `/workspace/pypto` and `/workspace/pypto-lib` to exist directly. A bind
mount hides the image's own `/workspace`, so the source you mounted is the only source
there is, and it is almost certainly not the one `simpler` was built from. Run
`pypto-build` before anything else.
The startup message always directs you to check the current checkout; an image's
`dist-manifest` records what shipped and does not certify a bind-mounted tree as built.

Model suggestions use `PYPTO_PLATFORM`: the launcher sets `a2a3` for `run a3` and
`a5` for `run a5`; direct image runs derive it from the image's CANN tag. In a custom
environment, set `PYPTO_PLATFORM=a2a3` or `PYPTO_PLATFORM=a5` explicitly. An unknown
tag produces setup guidance instead of guessing a model target.

### Updating the source and rebuilding

The maintained helpers are in [`docker/scripts/`](../../../docker/scripts/). Images
rebuilt from this Dockerfile install them under `/opt/pypto/bin`, which remains visible
when a bind mount covers `/workspace`. Existing published images may still have the old
updater only in their bundled checkout; use the repository copy below in those images.

For an existing container whose checkout still has the helpers directly under
`.github/docker/` or `docker/`, first run `git -C "$PYPTO_SRC" pull --ff-only`, then
use the command below. This also recovers an old installed updater that pulled the
directory move and then reported a
missing helper. The repository updater refreshes `/opt/pypto/bin` to use the new layout.

For a checkout with a tracking branch and no local changes:

```bash
bash "$PYPTO_SRC/docker/scripts/pypto-update.sh"
```

This fast-forwards each existing repository on its current tracking branch, updates
pinned submodules, installs the PTOAS release in `toolchain/versions.env` if necessary,
rebuilds PyPTO and simpler, and runs `pypto-doctor`. It never switches branches or resets
commits. A detached checkout, divergent branch, or local changes require you to resolve
that state yourself or use `--no-pull`.
After pulling (or immediately with `--no-pull`), the updater uses the helpers from
`$PYPTO_SRC/docker/scripts`, including when launched from `/opt/pypto/bin`. The selected
checkout must contain those helpers; missing helpers fail instead of using stale image copies.
Every update also refreshes the installed helpers and toolchain inputs atomically,
including without `--reinstall-deps`. Reload an existing shell with
`source "$PYPTO_HOME/bin/pypto-activate.sh"`, or re-enter it, to use the new activation.
`pypto-setup.sh --helpers-only` performs just this helper refresh without installing dependencies.

For edits in a bind-mounted checkout, preserve the source and rebuild:

```bash
bash "$PYPTO_SRC/docker/scripts/pypto-update.sh" --no-pull --jobs 2
```

| Flag | Effect |
| ---- | ------ |
| `--no-pull` | Keep source and submodules unchanged; align PTOAS and rebuild |
| `--reinstall-deps` | Refresh Python dependencies using the shared setup script |
| `--proxy http://host:port` | Set HTTP and HTTPS proxies for this invocation |
| `--remote-submodules` | Track submodule remotes across repeated updates; commit drift is allowed, local edits are rejected |
| `--jobs N` | Build job count capped by machine limits; otherwise use the repository resource loader |
| `--skip-doctor` | Skip the closing checks when no device is exposed |

To fix only a stale PTOAS, without updating source or rebuilding:

```bash
bash "$PYPTO_SRC/docker/scripts/pypto-install-ptoas.sh"
source "$PYPTO_SRC/docker/scripts/pypto-activate.sh"
"$PTOAS_ROOT/bin/ptoas" --version
```

The installer reads the checkout's version and architecture-specific SHA256, verifies
the release wheel before changing the installation, and installs into
`${PTOAS_ROOT:-/opt/pypto/ptoas}`. It accepts `--force` to reinstall the same version.
Both dedicated environments must contain a working CPython 3.10 interpreter,
`pyvenv.cfg`, and `bin/activate`; an incompatible or incomplete PTOAS prefix is
recreated after the downloaded wheel passes checksum verification. Activation keeps
PyPTO's venv first on PATH while exposing the pinned PTOAS ahead of CANN's assembler;
runtime packages are installed explicitly into `$PYPTO_HOME/venv`.
In an older image, source the checkout's activation helper in each new shell before
using PTOAS; selecting its executable alone does not prevent CANN's Python package from
shadowing the pinned wheel. The installer isolates its own version probes automatically.
The environment must already contain uv and CPython 3.10, as the published image does.
Run `pypto-setup.sh` from a complete checkout to provision a different CANN container;
it installs the same helper sources and uses `build-constraints.txt` for build tools.
Its dedicated `$PYPTO_HOME/venv` must use CPython 3.10 and provide readable
`Python.h`, `pyconfig.h`, and a Python library for the native build. Incompatible or
incomplete environments are recreated with uv-managed Python before dependencies
are installed; an existing complete build environment is preserved. PTOAS's separate
runtime environment does not require development files.
After setup (including `--helpers-only`), copy the printed activation command into
your shell. It exports the selected installation and checkout paths before sourcing
the helper, including paths supplied with `--pypto-home` or `--pypto-src`.
The updater's reload command preserves these paths as well.
Setup copies a discovered uv executable into `$PYPTO_HOME/bin`, so checkout-owner
builds can use it even when the original executable is under root's private home.
To pin uv, pass `--uv-version TAG` or set `UV_VERSION=TAG`; a concrete release tag
always downloads that release, even if uv is already installed. The command-line
option takes precedence over the environment variable. With the default `latest`
setting, setup reuses an existing uv or downloads the latest release if none exists.

`pypto-build` uses native CMake, installs the simpler matching the checkout, and follows
its PTO-ISA pin. It rebuilds and reinstalls the runtime wheel on every invocation,
including after uncommitted source edits; native build and ccache reuse remain enabled.
It deliberately avoids `pip install -e .`: a scikit-build-configured
`build/` can place the extension outside the source import path. Build parallelism comes
from `.claude/skills/testing/load-env.sh`, with two jobs on unclassified machines.
An explicit job request can lower that cap but cannot raise it.
Existing native builds keep their CMake generator; fresh builds use Ninja. Caches from
a different path, environment or toolchain are discarded together with their objects,
including host caches mounted at the same path. The helper records build state after
each successful build; unchanged builds remain incremental, while unrecorded builds
or artifacts modified outside the helper are rebuilt. Container hostnames, path-list
ordering, ownership and filesystem change times are excluded from that record, so
recreating an equivalent container preserves incremental builds.
`pypto-doctor` checks the installed simpler build revision against the mounted `runtime/`.
It warns when `runtime/` has uncommitted changes, because a matching revision cannot
prove that the installed binding includes those edits. A known revision mismatch fails
even with `PYPTO_SKIP_RUNTIME_PIN_CHECK=1`, as does a binding that predates revision
stamping. An empty build stamp or unavailable checkout revision is reported as
unverified, never as a verified match.

### Building the development image

Build from the repository root on the target architecture. `deps` provides the
environment, `sim` adds GCC/G++ 15, and `dist` also clones and builds PyPTO and pypto-lib:

```bash
docker buildx build --load --target dist \
  --file docker/pypto_dev.Dockerfile \
  --build-arg DIST_BUILD_JOBS=2 \
  --build-arg PYPTO_REF=<pypto-commit> \
  --build-arg PYPTO_LIB_REF=<pypto-lib-commit> \
  --tag pypto-dev:a3-dist .
```

The default `main` refs stay on tracking branches so `pypto-update.sh` can pull updates.
Tag and commit refs produce detached checkouts; use `--no-pull` for those pinned builds.
The selected PyPTO ref must contain the dev-container helpers, `toolchain/versions.env`
and `build-constraints.txt`. The `dist` stage runs that checkout's setup and build
helpers, refreshes dependencies from its pins and constraints, and installs its helpers
for subsequent sessions. It does not fall back to the Docker context's helper copies
when a selected ref lacks them.

For an AMD64 image, build on an x86_64 host with a CANN base tag supporting
`linux/amd64`; confirm that support before building. Publish to a separate architecture
tag first, and validate both `pypto-doctor` and a model on the matching NPU host before
combining architecture manifests. A successful build without an NPU verifies the image
build only.

Use `docker/scripts/pypto-dev.sh build a3` for a local environment image and
`docker/scripts/pypto-dev.sh run a3 0` to create or re-enter a container. The launcher
supports CANN tag, image, cache-volume, and proxy overrides; see its source for details.
Proxy values are forwarded from the environment without printing credentials in command
traces or passing them as command arguments. Re-entry retains the container's existing
environment. The launcher checks the checkout paths, image (including a rebuilt tag),
cache volume, and device configuration before reusing a container. A mismatch, or an older
container without configuration metadata, stops with recreation instructions; preserve
container-only changes before explicitly removing it.
With Ascend Docker Runtime, the selected physical card is logical device `0` inside the
container, so the launcher exports `PYPTO_DEVICE=0`. Privileged access keeps the physical
card index. `pypto-doctor` probes that same selected device. For linked worktrees, the
launcher preserves the original paths of both the checkout and common Git metadata.
It sets `PYPTO_SRC`/`PYPTO_LIB_SRC` accordingly and starts in `PYPTO_SRC`; this keeps
relative submodule gitdir paths and worktree back-references valid without rewriting
host metadata. Regular checkouts still use `/workspace/pypto` and `/workspace/pypto-lib`.
The launcher supplies scoped Git `safe.directory` entries for the mounted checkouts and
their initialized submodules, including nested ones, so container root can access
host-owned trees. Adding initialized submodules changes the container configuration
and requires recreation.
Git updates run as each checkout's numeric UID/GID. Native compilation and runtime
wheel creation run as the PyPTO checkout owner, so new sources, Git metadata and build
outputs remain writable on the host. Rootful containers require `setpriv` from
util-linux; no matching account in `/etc/passwd` is needed. Source-writing commands use
a separate HOME at `$PYPTO_HOME/users/<uid>` and ccache directory at
`$PYPTO_HOME/cache/ccache/<uid>`. Configure private-repository Git authentication for
that HOME or supply an SSH agent; root's credentials are not copied. Installation of
the completed wheel retains the caller's privileges and leaves the existing Python
environment's ownership intact. Existing host files and linked-worktree metadata are
never recursively chowned. Shell/device diagnostics keep their original privileges;
checkout diagnostics run as the source owner. Activation disables Python bytecode
writes and optional Git index refreshes during read-only checks.
When the primary PyPTO checkout has `.claude/skills/testing/testing.env`, the launcher
mounts that file read-only at its original path too, preserving the machine's job limits.

### Maintaining published images in CI

[`docker/images.json`](../../../docker/images.json) is the maintenance entrypoint.
The `Maintain Development Images` workflow runs daily at 19:37 UTC (03:37 Beijing),
on changes to `docker/`, toolchain pins, build constraints or its workflow on `main`,
and through **Run workflow**. PRs touching those paths run configuration and
offline script tests on GitHub-hosted runners. Only PRs whose source repository is
`hw-native-sys/pypto` or the explicitly trusted fork `luohuan19/pypto` also build
the image and validate doctor, single-card decode and a two-card distributed case on the
configured NPU runner. All other forks skip that job. The allowlist lives in the
workflow's `image.if` condition; adding a repository grants its code execution on
the persistent host and requires a maintainer trust review.
Because PRs can edit workflow YAML, this allowlist alone is not a security boundary.
Before enabling shared-runner PR builds, an administrator must configure **Settings
→ Actions → General → Require approval for all external contributors** and ensure
that maintainers approve only trusted sources and reviewed code, including workflow
changes. This repository-wide policy affects all external PR workflows; pending
approval also prevents their configuration checks from running. It does not isolate
approved code. See [GitHub's fork workflow approval settings](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/enabling-features-for-your-repository/managing-github-actions-settings-for-a-repository#controlling-changes-from-forks-to-workflows-in-public-repositories).
Scheduled runs become active only after this workflow is merged into the default
branch (`main`). A matching change merged into `main` also triggers an immediate
build and publication; configure the SWR secrets before merging.
PRs do not publish or receive SWR credentials. Hardware builds run only in `hw-native-sys/pypto`.
Manual runs validate without publishing by default; publication also requires `main`.

The configuration enables three native images, all using CANN 9.2.0:

| Image | Runner labels | Host driver (`Version=`) | Published tag |
| ----- | ------------- | ------------------------ | ------------- |
| `arm-a2a3` | `self-hosted`, `linux`, `ARM64`, `npu-xp` | `26.0.rc1` | `arm-a2a3-cann9.2-dist` |
| `arm-a5` | `self-hosted`, `Linux`, `ARM64`, `npu-a5` | `25.6.rc2.b023` | `arm-a5-cann9.2-dist` |
| `x86-a5` | `self-hosted`, `Linux`, `npu-a5`, `X64`, `pr` | `25.7.rc1.12` | `x86-a5-cann9.2-dist` |

The A5 base is `9.2.0-950-ubuntu22.04-py3.11-devel`, pinned separately to its ARM64
and AMD64 manifests. Architecture labels select the matching native A5 runner.
Doctor restricts each CANN 9.2 A5 driver pair to its configured host architecture.
The `pr` label selects the x86 runner; it does not restrict the workflow event type.
The remaining planned combination is `x86-a2a3`: it needs a native X64 A2/A3 runner,
an AMD64 CANN base digest, a verified driver pair and single-/two-card validation.
The A2/A3 preflight additionally requires runner `227-npu-a2a3-XP-1`.
If that label pool grows, assign the host a dedicated label and update `runners`.
Each runner needs the `pypto-toolchain` bundle with Python 3.10, Git, Docker, Ascend Docker Runtime and
`task-submit` with `--find`/`--kill` support. No host PyPTO installation is used.
Host maintenance commands enter that bundle explicitly; the system `python3` may be older.
Its verification output reports the host's CANN version; the image's version is
selected by `cann_tag`/`cann_digest` and checked inside the container by strict doctor.
The workflow installs Buildx v0.35.0 in its isolated Docker configuration and uses
the host's default Docker builder, preserving the shared build cache.

Configure these repository settings before the first publishing run:

| Kind | Name | Value |
| ---- | ---- | ----- |
| Secret | `SWR_USERNAME` | Username from the long-term SWR login command |
| Secret | `SWR_PASSWORD` | Password from that command |

Credentials are supplied only to the publishing step with `--password-stdin`.
Each attempt has a private temporary Docker configuration directory, removed by
an `always()` step. The temporary login token must never be committed.
The workflow sets `PYPTO_DOCKER_PROXY=http://127.0.0.1:7892` to use the CI runner's
local proxy; no GitHub Actions proxy variable is required. The proxy is passed to
Git and Docker build arguments, not baked into the image environment.
Buildx installation and GitHub artifact uploads also use this proxy.
For manual script invocation, set the `PYPTO_DOCKER_PROXY` environment variable
to a proxy reachable from that machine. The initial configuration sets
`proxy_required: true`, so a missing environment variable fails preflight.
`--network=host` affects build steps; registry pulls still use the Docker daemon's
network/proxy configuration. SWR authentication and manifest operations connect
directly without a proxy. Login/push subprocesses also clear proxy variables;
if the Docker daemon has its own proxy, its `NO_PROXY` must include the SWR host.

Before the first build, make the configured CANN base accessible. The default
`cann_repository` is `quay.io/ascend/cann`; if the daemon cannot reach it, mirror
the exact image for the configured architecture to an anonymously readable SWR repository and update that
field. Each architecture digest is pinned; replace `cann_digest` with the verified
mirror digest when changing repositories. A null digest
means pull the configured tag and resolve its digest on every run. Both modes
build from the resolved digest and record it in the artifact; a stale local cache
cannot hide a failed pull. Changing CANN requires editing the tag, digest, driver
expectation and published tag together.

Each run freezes the PyPTO and pypto-lib revisions before building. With PyPTO
`ref: main`, it uses the workflow checkout's commit so the Docker recipe and helper
scripts belong to the same event snapshot. For PRs, this is the exact PR head commit
cloned from its source repository, including forks, rather than a temporary merge
commit. Explicitly configured PyPTO refs keep their configured source repository.
Other refs and pypto-lib refs resolve
to full commit IDs. Runtime always follows PyPTO's gitlink; PTO-ISA follows that
runtime's `pto_isa.pin`, and PTOAS follows the selected PyPTO toolchain pin.
Build cache invalidation uses these SHAs instead of a mutable `main` argument.
Compilation is capped by both `build_jobs` and the resource-limit loader.
CI selects simpler's `build_package_<validation_platform>` and `build_package_sim`
targets: each image prebuilds its selected onboard platform and both CPU simulators.
`DIST_RUNTIME_TARGETS` stores this selection as
`PYPTO_RUNTIME_BUILD_TARGETS` for later `pypto-build` calls; an empty value uses
simpler's default targets. The override applies only to the runtime wheel build.
Selected PyPTO refs must provide `pypto-doctor --strict`; validation rejects older
helpers that would silently ignore the strict option.

The `cann92-a2a3-aicpu-class-memaccess` and `cann92-a5-aicpu-class-memaccess` patches
add only the selected platform's onboard AICPU warning exception needed by CANN 9.2's
GNU compiler. Each image applies its own patch before building; it is skipped
if its exact change is already upstream, and rejected if its context no longer
matches. `/opt/pypto/dist-state.json` records the runtime SHA, diff hash, patch hash
and installed extension hash after a successful build. `pypto-doctor --strict`
accepts only this recorded patch; other edits, missing stamps or unverified
bindings fail. The driver pairs in the table are deployment validation configurations;
publication still requires strict doctor and both hardware tests to pass. They are
not general compatibility guarantees for other hosts.

Validation uses fresh containers without source/cache volumes and sets
`PYPTO_BENCH=0` for correctness checks. Login and successful-build hints select
DeepSeek SWA for `a5` and Qwen3 decode for `a2a3`. Each image config selects a single-card
`decode` script and a two-card `distributed` script; optional `distributed_args`
are passed as individual arguments. A2/A3 uses Qwen3-14B decode and pypto-lib's
all-reduce example. A5 uses `models/deepseek_v4_1_flash/decode_swa.py` for SWA,
KV cache and MXFP8 validation, and `models/deepseek_v4_1_flash/decode_moe.py`
with `--ep 2` for expert computation and cross-card dispatch/combine validation.
Both A5 cases generate inputs and compare results against their golden references.
A single-card `task-submit` reservation runs strict doctor before decode;
a separate reservation runs the two-card case. Physical cards come from
`TASK_DEVICE`, while model arguments use container-local IDs `0` or `0,1`.
Driver metadata is mounted read-only from the host. ARM A5 containers also receive
`/dev/ummu` and `/dev/uburma` for URMA communication. Both models must exit zero
and produce `[RUN] PASS`; missing PASS, timeouts and test failures prevent
publication. `task_timeout_seconds` bounds the total wait for queueing and execution;
`task_max_seconds` independently caps device execution. The total timeout must
exceed the execution cap; the initial limits allow 3600 seconds total and 1800
seconds of device execution for each case. Cleanup steps remove
only this run's containers/tasks, without pruning shared caches.

Before uploading and again immediately before promotion, publication verifies that
the recipe commit is still the current upstream `main`. A stale rerun or an
unavailable upstream check fails without changing current/previous. This check
uses the recipe commit, so an explicitly pinned PyPTO source remains supported.
To retry, build the latest `main`; rollback is a separate, explicit operation.

After validation, the exact tested image ID is pushed to a temporary
`arm-a2a3-cann9.2-dist-<run_id>-<attempt>` tag. Its manifest and config digest must
match the tested local image before promotion. The current
`arm-a2a3-cann9.2-dist` moves to `arm-a2a3-cann9.2-dist-previous`, then the new
image becomes current. Retrying the same digest preserves the previous version.
The first publication creates only current; the second distinct publication
creates previous. Temporary tags are removed by tag name, never by shared digest.
Successful publication retains only these two tags for each configured image.
Configuration validation reserves both current and generated `-previous` tags across
all images, rejecting collisions before any native job is scheduled.

Legacy tags such as `a3-dist` and `a3-dist-20260918` are neither updated nor deleted
by this workflow. Keep them until two new builds have passed hardware validation,
been uploaded, and are available through current/previous; remove the old tags
in a separate migration cleanup afterward. Do not relabel old images as CANN 9.2
without verifying their contents.

The Actions summary and 14-day artifacts contain revisions, digests, build logs,
validation logs and pre-promotion manifests for recovery. Repoint current to the
previous digest to roll back without rebuilding. Promotion spans multiple registry
requests: if it fails midway, inspect both tags and the saved manifests before
retrying. A failed temporary-tag deletion fails the job and may require removing
that specific run tag manually; never delete its digest or shared layers.

These images contain detached source snapshots. `pypto-update.sh` will not pull
them, and the managed patch also makes runtime dirty. Pull a newer image and
create a fresh container/source volume to consume a new snapshot. Existing named
volumes keep their own source. Developers choosing to track branches must handle
their checkout and any local patch explicitly before using the updater.

To add `x86-a2a3` or images using another CANN version, add a configuration entry with matching
native runner labels, CANN base, driver and hardware validation paths. Keep it
disabled until that combination has been provisioned. Publish independent tags;
Docker architecture manifests cannot select images based on the attached NPU.

### Without Ascend Docker Runtime

When no Ascend runtime is registered, the launcher's default `auto` mode selects
`privileged`. This requires permission to run privileged Docker containers and exposes
all host devices and cards. The launcher prints this behavior before creating or
re-entering the container. Choose it explicitly when that access is acceptable:

```bash
PYPTO_DEVICE_ACCESS=privileged docker/scripts/pypto-dev.sh run a3 1
```

The launcher mounts the host driver tree and management tools, and sets
`PYPTO_DEVICE=1` to select physical card 1 as the default compute device. This setting
does not restrict which other cards or host devices the container can access.
For A5, use `run a5` on an appropriate host; the launcher also supplies its URMA devices.

Passing only four device nodes and driver mounts is not a portable substitute for
this fallback: device enumeration can fail with `drvGetDevNum`/`507899`, including on
driver `26.0.rc1`. The guide therefore makes no claim that this restricted form works
with that driver. Use the registered `--runtime=ascend` path when device isolation is
required; it injects the driver and exposes only the selected cards. If privileged
containers are not permitted, arrange that runtime with the host administrator.

## Edge Cases

> **Fatal pitfall:** the host driver must be `26.0.rc1` for the image's CANN 9.0.0.
> A mismatch does not fail at startup — it fails at the first AICPU initialization, long
> after compilation and `aclrtSetDevice` have succeeded, and it looks like a kernel bug.

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| `unknown or invalid runtime name: ascend` | Ascend Docker Runtime missing, unregistered, or Docker not restarted after install | `sudo systemctl daemon-reload && sudo systemctl restart docker`, then check `docker info --format '{{json .Runtimes}}'`; or use the [privileged fallback](#without-ascend-docker-runtime) |
| `exec format error` | ARM64 image on an x86_64 host | Run it on an `aarch64` host — there is no workaround |
| `aclrtSetDevice` returns `507899` or `Resource_Busy` | The same physical card is held by another container or process | Find it with `npu-smi info` and `docker ps`, stop it or pick another card. The ACL context may be in a sticky error state afterwards — recreate the container before retesting |
| `simpler_init failed with code 507018` | Almost always a driver / CANN mismatch | [Triaging `507018`](#triaging-507018) |
| `npu-smi` reports `-9005`, or `DrvMngGetConsoleLogLevel failed (ret=4)` | `npu-smi` tries to enumerate the whole machine while only some cards are exposed | Harmless. Trust `pypto-doctor`'s `device access: aclrtSetDevice(0) ok` instead |
| `/workspace/pypto` does not exist | Bind mount one level too high | Mount the directory containing `pypto/`: `-v "$PWD/workspace:/workspace"`, not `-v "$PWD:/workspace"` |
| `pypto-doctor` says the simpler binding does not match its source | The mounted source differs from what the image built | `pypto-build && pypto-doctor` |
| Codegen fails with `ptoas at '...' is version X, but PyPTO requires PTOAS >= vY` | PTOAS is older than the checkout requires | Run the repository installer in [Updating the source and rebuilding](#updating-the-source-and-rebuilding); it also works with bind mounts and older images |
| `pto-isa` tries to clone over the network | The mounted source bumped `runtime/pto_isa.pin`, or the managed checkout was modified | Expected after a pin bump. If GitHub HTTP/2 is unstable, `git config --global http.version HTTP/1.1` — the resolver retries GitHub and then falls back to the GitCode mirror |

### Triaging `507018`

The failure looks like this — compilation, memory allocation, and opening the device all
succeed, and only the runtime stage fails:

```text
[RUN] runtime ...
[ERROR] ensure_aicpu_init_launched: [device_runner_base.cpp:532]
        ensure_aicpu_init_launched: stream sync failed: 507018 (device_id=0)
RuntimeError: simpler_init failed with code 507018
```

`507018` is `ACL_ERROR_RT_AICPU_EXCEPTION`, and it is a **generic code**: several
unrelated device-side mechanisms all funnel into this one number, so it alone does not
prove a deadlock, an out-of-memory, or a faulty operator. The full taxonomy is in the
runtime's own `runtime/docs/troubleshooting/device-error-codes.md`.

The call stack narrows it sharply, though. `ensure_aicpu_init_launched` is AICPU
*initialization*, which happens before any user kernel runs: PyPTO has just pushed the
AICPU runtime SO to the device and brought it up for the first time. The failure is in the
environment, not in the model you were running.

**First, check the host driver version.** This is the most common cause in a container.
Run it on the **host** — `version.info` is not exposed inside a `--runtime=ascend`
container:

```bash
cat /usr/local/Ascend/driver/version.info
```

For the A3 image's CANN 9.0.0, `Version=` must be `26.0.rc1`.
`compatible_version=` uses internal identifiers, so do not search it for `9.0.0`.
Other public version pairs are unsupported for this image (the runtime lists what it supports in
`runtime/docs/install.md`); upgrade the driver and firmware to the matching pair before
retesting.

**Second, check that the chip series matches `-p`.** The wrong arch also surfaces as
`507018` and looks convincingly like a compiler bug. `a2a3` covers `Ascend910*` (A2) and
`Ascend910_93*` (A3); `a5` covers `Ascend950*`:

```bash
npu-smi info -t board -i 0 -c 0 | grep -iE 'Chip Name|NPU Name'
```

**Third, check that two containers do not share one chip's die pair.** On a2a3, adjacent
`npu-smi` Phy-IDs are die0 and die1 of the same `Ascend910`, and they share the
device-side directory that stages the AICPU SO. Concurrent bring-up has corrupted that
file and raised an AICPU exception. Allocate cards a whole chip at a time.

**Fourth, capture device-side logs.** The host log shows only the cascade; the cause is in
CANN's device slog:

```bash
CARD_ID=0   # a free physical card

docker run --rm \
  --runtime=ascend -e ASCEND_VISIBLE_DEVICES="$CARD_ID" \
  --shm-size=16g \
  -e ASCEND_SLOG_PRINT_TO_STDOUT=1 \
  -e ASCEND_GLOBAL_LOG_LEVEL=1 \
  "$PYPTO_IMAGE" \
  -c 'python $PYPTO_SRC/examples/beginner/01_hello_world.py' \
  > slog.txt 2>&1
```

In `slog.txt`, find the **first** `PrintAicpuErrorInfo` or `ProcessStarsAicpuErrorInfo`
and note its `soName`, `funcName`, `errorCode`, and `chipId/dieId`; every `507899` and
`507901` after it is cascade noise. Also check lines mentioning `version_verify` and
`package_process_config`, where a failed version check or AICPU package push leaves a
record.

When reporting the problem, attach the host's `version.info`, `npu-smi info`, the exact
`docker run` command, the full `pypto-doctor` output, and `slog.txt`.

## See Also

- [Installation](01-installation.md) — the from-source path, and what an install does and
  does not give you.
- [Quickstart](02-quickstart.md) — your first kernels, once you have an environment.
- [PTO Project Ecosystem](../dev/00-ecosystem.md) — how PyPTO, PTOAS, pto-isa, and the
  runtime relate, and why their versions move together.
- [Runtime documentation](https://hw-native-sys.github.io/simpler/) — installing and
  operating the runtime that executes compiled programs.
