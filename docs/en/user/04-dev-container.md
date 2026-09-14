# Development Container

Pull a prebuilt image that already carries CANN, PyPTO, and the runtime, and run a kernel
on a real Ascend card without assembling a toolchain.

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
| PyPTO + `simpler`, built | Image | `/workspace/pypto`, already compiled |
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
| ptoas | 0.61 — the version `toolchain/versions.env` pins |
| Size, uncompressed | ~16.7 GB |
| Working directory | `/workspace` |
| Entry point | Interactive bash with the PyPTO environment already active |
| Verified digest | `sha256:ffa6bef8ed34f565a2331a66369902303ff957e832476fa1abaad2be435b82b5` |

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

A fourth script, `pypto-update.sh`, ships inside the source tree rather than on `PATH` —
see [Updating the source and rebuilding](#updating-the-source-and-rebuilding).

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
`device access: aclrtSetDevice(0) ok` is the check that settles whether the card works.

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

In `version.info`, `Version=` is the driver and `compatible_version=` is the range of CANN
versions it accepts. Those two lines settle every version argument.

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

### Updating the source and rebuilding

Pulling the latest main, editing C++ in a volume, or mounting a different checkout — all
three end in a rebuild. The image ships a script that does the whole sequence:

```bash
/workspace/pypto/.github/docker/pypto-update.sh
```

It pulls pypto (submodules included) and pypto-lib, reinstalls ptoas if
`toolchain/versions.env` has moved, runs `pypto-build`, and closes with `pypto-doctor`.
Unlike the other three commands it is **not** on `PATH` — invoke it by the full path above.

| Flag | Effect |
| ---- | ------ |
| `--no-pull` | Rebuild only, keeping local edits |
| `--reinstall-deps` | Reinstall the pip dependencies too (torch, numpy, clang-tidy, …) |
| `--proxy http://host:port` | Proxy for the git pulls and the ptoas download |
| `--remote-submodules` | Track each submodule's own remote instead of the pinned commits |
| `--jobs N` | Build parallelism |
| `--skip-doctor` | Skip the closing `pypto-doctor` |

> **It discards local work, committed included.** The pull is
> `git reset --hard origin/main`, which moves the branch itself — committing first does
> not save you, since the commit simply stops being reachable from the branch. Either use
> `--no-pull`, or leave a reference behind before running it:
>
> ```bash
> git -C /workspace/pypto branch backup/before-update
> ```
>
> Untracked files are left alone, which is also why the script does not delete itself: it
> is untracked in that checkout.

**Only two of the three cases can run it.** The script lives in the image's own source
tree, not in the repository and not on `PATH`, so a bind mount over `/workspace` hides it
along with everything else the image had there. A named volume is fine — an empty one is
pre-populated from the image, the script included. With a bind-mounted checkout, rebuild
by hand:

```bash
git -C /workspace/pypto pull
git -C /workspace/pypto submodule update --init --recursive
git -C /workspace/pypto-lib pull
pypto-build
pypto-doctor
```

These steps are a **partial** stand-in, not an equivalent. Two things the script does are
missing: it reinstalls ptoas when `toolchain/versions.env` moves the pin, and it passes
proxy and job-count settings through to both. Check the assembler by hand after a pin
bump:

```bash
grep '^PTOAS_VERSION=' /workspace/pypto/toolchain/versions.env
/opt/pypto/ptoas/bin/ptoas --version
```

`pypto-build` rebuilds PyPTO with native CMake, installs the `simpler` matching the
current `runtime/` source, and resolves the `pto-isa` pin that source names. It
deliberately does not use `pip install -e .`: a scikit-build-configured `build/` puts the
extension under `build/python/bindings/` instead of `python/pypto/`, after which every
later build succeeds while Python keeps importing a stale `.so` — C++ edits appear to do
nothing at all.

What `pypto-build` does **not** do is move ptoas — that is the one step only
`pypto-update.sh` performs.

### Without Ascend Docker Runtime

If you cannot install the runtime — no root, no permission to restart Docker, maintenance
window not open — pass the device nodes and driver in yourself:

```bash
docker run --rm -it \
  --device /dev/davinci1 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  --shm-size=16g \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  "$PYPTO_IMAGE"
```

`davinci1` is the card you want; the other three nodes are driver management nodes and
must be passed together with it. The `driver` mount supplies the driver libraries. The
`dcmi`, `npu-smi`, and `ascend_install.info` mounts only decide whether `npu-smi` works
inside the container and can be dropped.

**Card numbers are not remapped here.** The container still sees `/dev/davinci1`, but ACL
enumerates only that one card, so PyPTO still takes `-d 0`; passing `davinci2` and
`davinci3` gives `-d 0,1`. To select by physical number when several are visible, use
CANN's own variable, whose logical ids start at 0 in the order listed:

```bash
-e ASCEND_RT_VISIBLE_DEVICES=2,3
```

This path is verified on `driver 26.0.rc1` with this image: `pypto-doctor` passes every
check, and both the single-card and two-card runs pass. It needs no `--privileged`, and
`--privileged` is not a substitute — that hands the container every card on the machine.
Reach for it only as a last resort, if the above still reports `507899`.

`--runtime=ascend` remains the recommended path: it injects driver and device nodes,
renumbers cards, and exposes only the ones you named.

## Edge Cases

> **Fatal pitfall:** the host driver must be `26.0.rc1` for the image's CANN 9.0.0.
> A mismatch does not fail at startup — it fails at the first AICPU initialization, long
> after compilation and `aclrtSetDevice` have succeeded, and it looks like a kernel bug.

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| `unknown or invalid runtime name: ascend` | Ascend Docker Runtime missing, unregistered, or Docker not restarted after install | `sudo systemctl daemon-reload && sudo systemctl restart docker`, then check `docker info --format '{{json .Runtimes}}'`; or use the [`--device` fallback](#without-ascend-docker-runtime) |
| `exec format error` | ARM64 image on an x86_64 host | Run it on an `aarch64` host — there is no workaround |
| `aclrtSetDevice` returns `507899` or `Resource_Busy` | The same physical card is held by another container or process | Find it with `npu-smi info` and `docker ps`, stop it or pick another card. The ACL context may be in a sticky error state afterwards — recreate the container before retesting |
| `simpler_init failed with code 507018` | Almost always a driver / CANN mismatch | [Triaging `507018`](#triaging-507018) |
| `npu-smi` reports `-9005`, or `DrvMngGetConsoleLogLevel failed (ret=4)` | `npu-smi` tries to enumerate the whole machine while only some cards are exposed | Harmless. Trust `pypto-doctor`'s `device access: aclrtSetDevice(0) ok` instead |
| `/workspace/pypto` does not exist | Bind mount one level too high | Mount the directory containing `pypto/`: `-v "$PWD/workspace:/workspace"`, not `-v "$PWD:/workspace"` |
| `pypto-doctor` says the simpler binding does not match its source | The mounted source differs from what the image built | `pypto-build && pypto-doctor` |
| ptoas rejects an instruction form in generated code | The assembler has fallen behind `toolchain/versions.env` after a hand-run source update | `/workspace/pypto/.github/docker/pypto-update.sh` — it is the only step that reinstalls ptoas. The image as published is aligned (0.61) |
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

`Version=` must be `26.0.rc1`, and `compatible_version=` must include the image's CANN
9.0.0. Anything else is an unsupported environment (the runtime lists what it supports in
`runtime/docs/install.md`); upgrade the driver and firmware to the matching pair before
retesting.

**Second, check that the chip series matches `-p`.** The wrong arch also surfaces as
`507018` and looks convincingly like a compiler bug. `a2a3` covers `Ascend910B*` (A2) and
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
