# ---------------------------------------------------------------------------
# PyPTO on-device development image, on top of the official CANN base.
#
# Goal: `docker run` -> build PyPTO from a bind-mounted checkout -> run
# pypto-lib models on a real card, on A2/A3 (Ascend910B) and A5 (Ascend950)
# alike. One Dockerfile; the chip series is the CANN base tag.
#
#   A2/A3  --build-arg CANN_TAG=9.0.0-a3-ubuntu22.04-py3.11-devel
#   A5     --build-arg CANN_TAG=9.1.T500-950-ubuntu22.04-py3.11-devel
#
# Three deliberate choices, each measured rather than assumed:
#
# 1. NO apt, and no PPA. The on-device path needs no gcc-15: simpler builds its
#    runtime at C++17 (runtime/CMakeLists.txt), orchestration .so with plain
#    `g++`, in-core kernels with CANN's own CCEC/BiSheng, and device code with
#    CANN's aarch64-target-linux-gnu-g++. g++-15 is reached for only by
#    Gxx15Toolchain (simulation kernels, -std=c++23) and sanitizer builds — see
#    the `sim` target at the bottom for those. Dropping apt also makes the build
#    work behind a proxy that allowlists github but not the Ubuntu mirrors.
#
# 2. Python 3.10 from uv, not the base image's interpreter. CANN 9.0.0 images
#    ship 3.11 and 9.0.1+ ship 3.12, while ptoas is released as a cp310 wheel
#    only. The distro's /usr/bin/python3.10 is present but unusable (no venv
#    module, no headers, and installing those needs apt). uv brings its own
#    standalone CPython 3.10 with headers, which also puts this image on the
#    exact interpreter CI tests against.
#
# 3. Nothing pypto-shaped is baked in. simpler's _task_interface stamps
#    SIMPLER_BUILD_COMMIT at compile time and refuses a binding built from a
#    different tree, so a preinstalled simpler would break the moment the
#    mounted checkout's `runtime` submodule pin differs — as an ImportError
#    that looks nothing like a version problem. The image ships the
#    environment; `pypto-build` builds the tree you mounted.
#
# Not included, by construction: the NPU driver. It comes from the host at
# `docker run` time — see docker/scripts/pypto-dev.sh.
#
# The build reaches github.com (uv, CPython, ptoas), PyPI and
# download.pytorch.org. Build it where those are reachable and distribute the
# result through a registry; a proxy that allowlists only github cannot build
# this image.
#
# No `# syntax=` directive on purpose: it makes BuildKit pull a frontend image
# from docker.io before anything else runs, which is one more registry to be
# reachable for no feature this file uses.
# ---------------------------------------------------------------------------
ARG CANN_REPO=quay.io/ascend/cann
ARG CANN_TAG=9.0.0-a3-ubuntu22.04-py3.11-devel

FROM ${CANN_REPO}:${CANN_TAG} AS deps
ARG CANN_TAG

# Pinned to what the CI fleet carries. Left to resolve `torch>=2.0.0` itself,
# pip picks an aarch64 wheel that drags in the whole CUDA stack (torch 427 MB +
# cudnn 444 MB + ...), none of it usable here; and a torch bump alone moves
# tests/st/codegen/torch/test_torch_codegen_paged_attention.py off its golden.
ARG TORCH_VERSION=2.10.0
# 'latest' tracks the newest uv release. Pin a tag (e.g. 0.9.2) for a
# reproducible image.
ARG UV_VERSION=latest

ENV PYPTO_HOME=/opt/pypto \
    PYPTO_CANN_TAG=${CANN_TAG} \
    UV_PYTHON_INSTALL_DIR=/opt/pypto/python \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# The one thing this image must inherit from the base rather than install:
# ptoas needs GLIBCXX_3.4.29. Ubuntu 22.04's libstdc++ carries 3.4.30, but an
# openEuler-based tag may not — fail here rather than at the first assemble.
RUN set -eu; \
    libcxx="$(ls /usr/lib/*-linux-gnu/libstdc++.so.6 2>/dev/null | head -1)"; \
    strings "$libcxx" | grep -q GLIBCXX_3.4.29 \
      || { echo "$libcxx lacks GLIBCXX_3.4.29 — ptoas will not load" >&2; exit 1; }

# uv, straight from its GitHub release: a static binary, no package index
# involved, and the only tool needed to bootstrap everything below.
RUN set -eu; \
    case "$(uname -m)" in \
      aarch64) asset=uv-aarch64-unknown-linux-gnu.tar.gz ;; \
      x86_64)  asset=uv-x86_64-unknown-linux-gnu.tar.gz ;; \
      *) echo "unsupported architecture: $(uname -m)" >&2; exit 1 ;; \
    esac; \
    if [ "$UV_VERSION" = latest ]; then \
      url="https://github.com/astral-sh/uv/releases/latest/download/$asset"; \
    else \
      url="https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/$asset"; \
    fi; \
    curl --fail --location --retry 3 --retry-all-errors "$url" -o /tmp/uv.tar.gz; \
    tar -xzf /tmp/uv.tar.gz -C /tmp; \
    install -m 0755 /tmp/uv-*/uv /tmp/uv-*/uvx /usr/local/bin/; \
    rm -rf /tmp/uv.tar.gz /tmp/uv-*; \
    uv --version

# Keep dependencies, PTOAS installation, and helper sources identical to the
# in-container setup path. These remain reachable when /workspace is mounted.
COPY toolchain/versions.env build-constraints.txt ${PYPTO_HOME}/toolchain/
COPY docker/scripts/pypto-owner.sh docker/scripts/pypto-venv.sh \
     docker/scripts/pypto-activate.sh docker/scripts/pypto-build docker/scripts/pypto-doctor \
     docker/scripts/pypto-update.sh docker/scripts/pypto-setup.sh \
     docker/scripts/pypto-install-ptoas.sh ${PYPTO_HOME}/bin/
RUN chmod 0755 "$PYPTO_HOME/bin/pypto-build" "$PYPTO_HOME/bin/pypto-doctor" \
      "$PYPTO_HOME/bin/pypto-update.sh" "$PYPTO_HOME/bin/pypto-setup.sh" \
      "$PYPTO_HOME/bin/pypto-install-ptoas.sh" \
 && chmod 0644 "$PYPTO_HOME/bin/pypto-activate.sh" "$PYPTO_HOME/bin/pypto-venv.sh" \
 && TORCH_VERSION="$TORCH_VERSION" "$PYPTO_HOME/bin/pypto-setup.sh"

# The source mounts are the developer-selected Git workspaces in this image.
RUN git config --global --add safe.directory /workspace/pypto \
 && git config --global --add safe.directory /workspace/pypto/runtime \
 && git config --global --add safe.directory /workspace/pypto-lib

# Bind-mount the checkout here. /workspace is a plain dir, not a VOLUME:
# declaring it would make an un-mounted `docker run` silently produce an empty
# anonymous volume instead of the actionable error pypto-doctor gives.
WORKDIR /workspace
ENV BASH_ENV=${PYPTO_HOME}/bin/pypto-activate.sh \
    PATH=${PYPTO_HOME}/bin:${PATH}
ENTRYPOINT ["/bin/bash", "--rcfile", "/opt/pypto/bin/pypto-activate.sh"]
CMD ["-i"]

# ---------------------------------------------------------------------------
# sim — optional: adds g++-15 so the CPU simulator platforms (--platform=a5sim
# / a2a3sim) and sanitizer builds work too. Split out because it is the only
# layer that needs apt, and therefore the only one that fails behind a proxy
# which allowlists github but not the Ubuntu mirrors.
#
#   docker buildx build --target sim -t pypto-dev:a3-sim ...
# ---------------------------------------------------------------------------
FROM deps AS sim
# The PPA's signing key fingerprint, as published by Launchpad's own API for this
# archive (`api.launchpad.net/1.0/~ubuntu-toolchain-r/+archive/ubuntu/test`).
ARG GCC_PPA_FINGERPRINT=C8EC952E2A0E1FBDC5090F6A2C277A0A352154E5
# The repository is added by hand rather than with `add-apt-repository`, for two
# reasons found by running it: it imports the key through dirmngr's keyserver
# protocol, which is not routable behind an ordinary CONNECT proxy, and it shells
# out to gpg, which this base image does not ship at all. Fetching the same key
# over HTTPS avoids both, names the fingerprint explicitly, and drops the need
# for software-properties-common.
#
# The key is stored ASCII-armored as `.asc` and referenced directly: apt 2.4
# reads armored keyrings, so nothing here needs gpg to dearmor it.
RUN set -eu; \
    . /etc/os-release; \
    install -d /etc/apt/keyrings; \
    curl --fail --location --retry 3 --retry-all-errors \
      "https://keyserver.ubuntu.com/pks/lookup?op=get&options=mr&search=0x${GCC_PPA_FINGERPRINT}" \
      -o /etc/apt/keyrings/ubuntu-toolchain-r.asc; \
    grep -q 'BEGIN PGP PUBLIC KEY BLOCK' /etc/apt/keyrings/ubuntu-toolchain-r.asc; \
    echo "deb [signed-by=/etc/apt/keyrings/ubuntu-toolchain-r.asc]" \
         "https://ppa.launchpadcontent.net/ubuntu-toolchain-r/test/ubuntu" \
         "${VERSION_CODENAME} main" \
      > /etc/apt/sources.list.d/ubuntu-toolchain-r.list; \
    export DEBIAN_FRONTEND=noninteractive; \
    apt-get update; \
    apt-get install -y --no-install-recommends gcc-15 g++-15 ccache; \
    rm -rf /var/lib/apt/lists/*; \
    gxx_major="$(g++-15 -dumpversion | cut -d. -f1)"; \
    [ "$gxx_major" -ge 15 ] || { echo "g++-15 reports $gxx_major, need >= 15" >&2; exit 1; }

# ---------------------------------------------------------------------------
# dist — the image to hand to a user. `deps` deliberately ships no pypto, so a
# user landing in it has to build first; this stage clones the two public repos
# and builds them, so `docker run` lands somewhere a model can be run
# immediately.
#
#   docker buildx build --target dist -t pypto-dev:a3-dist ...
#
# Cloned rather than COPYed from the build context, for three reasons: the
# context has no pypto-lib in it; a COPYed `.git` would carry this repo's ssh
# remote, which a user without a deploy key cannot pull from; and with no
# .dockerignore in the repo, `COPY .` would drag the 1.2 GB build/ tree in.
#
# The build products stay in the image (~2 GB). Dropping them would save that
# much and cost the user a full rebuild — 263 objects, minutes on a large
# machine and far worse on a small one — the first time they touch a line of
# C++. Keeping them makes that edit an incremental relink instead.
#
# Intended run shape, where the named volumes let the user's edits outlive the
# container (Docker pre-populates an empty named volume from the image):
#
#   docker run -it --runtime=ascend -e ASCEND_VISIBLE_DEVICES=0 \
#     -v pypto-src:/workspace/pypto -v pypto-lib-src:/workspace/pypto-lib IMAGE
#
# Code is updated with `git pull` inside that volume, NOT by pulling a newer
# image: a named volume that already has content is never re-populated, so a
# fresh image would leave the old tree in place. Pull a new image for a new
# environment (CANN, torch, ptoas); use git for code.
# ---------------------------------------------------------------------------
# Built on `sim`, not `deps`, so a user gets both paths: the on-device platforms
# and the CPU simulator ones (--platform=a2a3sim / a5sim), which need g++-15.
#
# That choice moves the host compiler: pypto-activate.sh prefers gcc-15/g++-15
# whenever they exist, deliberately, so the sanitizer runtime cannot be mixed
# between the host build and the simulator kernels. So everything here is built
# with g++-15 rather than the base image's g++-11 — worth knowing when comparing
# against a `deps` build. It also brings ccache, which leaves this image with a
# warm compile cache for the user's first edit.
FROM sim AS dist

ARG PYPTO_REPO=https://github.com/hw-native-sys/pypto.git
ARG PYPTO_LIB_REPO=https://github.com/hw-native-sys/pypto-lib.git
ARG PYPTO_REF=main
ARG PYPTO_LIB_REF=main
# Explicit, never `nproc`: this builds on whatever machine CI hands it, and the
# repo's own rule is that the job count is stated rather than inferred.
ARG DIST_BUILD_JOBS=2

# Branch refs retain their upstream so the shipped updater can fast-forward.
# Git detaches tags and commit IDs, preserving reproducible pinned builds.
RUN set -eu; \
    git clone "$PYPTO_REPO" /workspace/pypto; \
    git -C /workspace/pypto checkout "$PYPTO_REF" --; \
    git config --global --add safe.directory /workspace/pypto; \
    git -C /workspace/pypto submodule update --init --recursive; \
    git clone "$PYPTO_LIB_REPO" /workspace/pypto-lib; \
    git -C /workspace/pypto-lib checkout "$PYPTO_LIB_REF" --

# The selected ref owns its setup inputs and helpers. Refresh the environment
# before building so a context/ref mismatch cannot silently use stale logic.
# Do not activate the context's BASH_ENV while bootstrapping the selected setup.
RUN set -eu; \
    export PYPTO_SRC=/workspace/pypto; \
    helpers="$PYPTO_SRC/docker/scripts"; \
    for required in docker/scripts/pypto-setup.sh docker/scripts/pypto-build \
        docker/scripts/pypto-activate.sh docker/scripts/pypto-venv.sh \
        toolchain/versions.env build-constraints.txt; do \
        test -r "$PYPTO_SRC/$required" || { \
            echo "Selected PYPTO_REF lacks $required; choose a ref with the dev-container helpers" >&2; \
            exit 1; \
        }; \
    done; \
    BASH_ENV= "$helpers/pypto-setup.sh"; \
    BASH_ENV="$helpers/pypto-activate.sh" PYPTO_BUILD_JOBS="$DIST_BUILD_JOBS" "$helpers/pypto-build"

# A wheel install resolves PTO-ISA below simpler_setup/_assets, but the runtime
# package intentionally installs only build/lib and the pin — not the managed
# git checkout. Seed that exact installed-package location from the checkout
# pypto-build already fetched, so a dist image needs no network on first use.
# Keep .git: the resolver verifies both the pinned HEAD and a clean worktree
# before it reuses the checkout.
RUN set -eu; \
    managed_pto_isa="$("$PYPTO_HOME/venv/bin/python" -c \
      'from simpler_setup.pto_isa import get_pto_isa_clone_path; print(get_pto_isa_clone_path())')"; \
    rm -rf "$managed_pto_isa"; \
    mkdir -p "$(dirname "$managed_pto_isa")"; \
    cp -a /workspace/pypto/runtime/build/pto-isa "$managed_pto_isa"; \
    test "$(git -C "$managed_pto_isa" rev-parse HEAD)" = \
      "$(cat /workspace/pypto/runtime/pto_isa.pin)"; \
    test -z "$(git -C "$managed_pto_isa" status --porcelain)"

# What this image shipped with. Not for version comparison — the user tracks
# main with `git pull` — but so a support question can be answered: the tree in
# a long-lived volume has moved on, and this records where it started.
RUN set -eu; \
    { echo "pypto     $(git -C /workspace/pypto rev-parse HEAD)"; \
      echo "runtime   $(git -C /workspace/pypto rev-parse HEAD:runtime)"; \
      echo "pto-isa   $(cat /workspace/pypto/runtime/pto_isa.pin)"; \
      echo "pypto-lib $(git -C /workspace/pypto-lib rev-parse HEAD)"; \
      echo "built     $(date -u +%Y-%m-%dT%H:%M:%SZ)"; \
    } > "$PYPTO_HOME/dist-manifest"; \
    cat "$PYPTO_HOME/dist-manifest"
