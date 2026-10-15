#!/bin/bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Build and run the PyPTO on-device development image.
#
#   .github/docker/pypto-dev.sh build a3        # Ascend910B (A2/A3)
#   .github/docker/pypto-dev.sh build a5        # Ascend950  (A5)
#   .github/docker/pypto-dev.sh run   a3 [device-id]   # create or re-enter
#
# Env overrides: CANN_REPO, CANN_VERSION, CANN_PY, TARGET, IMAGE,
# PYPTO_SRC_HOST, PYPTO_LIB_SRC_HOST, PYPTO_CACHE_VOLUME, PYPTO_DEVICE_ACCESS.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$here/../.." && pwd)"

# quay.io is the registry the Ascend images are published to; the Huawei Cloud
# SWR mirror (swr.cn-south-1.myhuaweicloud.com/ascendhub/cann) carries the same
# tags and is the one to use where quay is unreachable.
CANN_REPO="${CANN_REPO:-quay.io/ascend/cann}"
# CANN is not a free choice: it is a function of the chip series, so it is
# derived per chip below rather than defaulting globally. runtime/docs/install.md
# names A3 -> CANN 9.0.0 and A5 -> CANN 9.1.T500, and the a2a3 runtime carries
# workarounds written against exactly 9.0.0 + driver 26.0.rc1 (see
# kCANN900FatalHandoffMs in a2a3/platform/onboard/host/device_runner.cpp and the
# ACL exception noted in common/platform/include/common/acl_hal_device.h).
# Overriding these is for trying a new CANN, not for routine use.
#
# The tags also differ in Python: 9.0.0 ships py3.11 and py3.12 both. It makes no
# difference to PyPTO, which builds its own CPython 3.10 regardless.
# 'deps' is the lean on-device image; 'sim' adds g++-15 for --platform=a5sim /
# a2a3sim and needs an apt mirror to be reachable at build time.
TARGET="${TARGET:-deps}"
IMAGE="${IMAGE:-pypto-dev}"
# How the card gets into the container. Two paths, and the first is much better:
#
#   runtime    Ascend-Docker-Runtime (`--runtime=ascend`, card chosen with
#              ASCEND_VISIBLE_DEVICES). It injects the driver, the character
#              devices and the libraries itself, needs no extra privilege, and
#              exposes *only* the requested card.
#   privileged Fallback for hosts without that runtime registered. Measured on
#              driver 26.0.rc1: plain `--device` passthrough is not enough there
#              — the driver refuses device enumeration (drvGetDevNum ->
#              drvRet=87), so aclrtSetDevice fails with 507899. Every narrower
#              relaxation was ruled out: --cap-add=ALL, an all-devices cgroup
#              rule, seccomp/apparmor/systempaths unconfined, a read-write /sys,
#              and bind-mounting the host's entire /dev. Only full --privileged
#              worked, and it hands the container every card on the machine.
#
# Left as 'auto' this picks `runtime` when dockerd reports the ascend runtime.
# Force one with PYPTO_DEVICE_ACCESS=runtime|privileged.
PYPTO_DEVICE_ACCESS="${PYPTO_DEVICE_ACCESS:-auto}"

usage() {
    echo 'Usage: pypto-dev.sh {build|run} {a3|a5} [device-id]' >&2
    exit 2
}
[ $# -ge 2 ] || usage

cmd="$1"; chip="$2"; shift 2

case "$chip" in
    a3)
        cann_chip=a3
        extra_devices=()
        # Verified end to end on this combination: on-device build plus a
        # pypto-lib decode model, against host driver 26.0.rc1.
        cann_version=9.0.0
        cann_py=py3.11
        ;;
    a5)
        # Ascend950. On aarch64 these images also carry URMA for RoCE, whose two
        # extra character devices have to be passed in as well.
        cann_chip=950
        extra_devices=(--device /dev/ummu --device /dev/uburma)
        # runtime/docs/install.md names 9.1.T500 for A5. Unverified here — no
        # Ascend950 card was available — including whether the CANN registry
        # publishes a matching tag at all.
        cann_version=9.1.T500
        cann_py=py3.11
        ;;
    *)  echo "unknown chip '$chip' (expected a3 or a5)" >&2; usage ;;
esac
CANN_VERSION="${CANN_VERSION:-$cann_version}"
CANN_PY="${CANN_PY:-$cann_py}"
tag="${CANN_VERSION}-${cann_chip}-ubuntu22.04-${CANN_PY}-devel"

case "$cmd" in
build)
    # The build reaches github.com, PyPI and download.pytorch.org; forward the
    # caller's proxy rather than making every developer remember to.
    proxy_args=()
    for v in HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy; do
        [ -n "${!v:-}" ] && proxy_args+=(--build-arg "$v=${!v}")
    done
    set -x
    docker buildx build \
        --file "$here/pypto_dev.Dockerfile" \
        --target "$TARGET" \
        --build-arg "CANN_REPO=$CANN_REPO" \
        --build-arg "CANN_TAG=$tag" \
        "${proxy_args[@]}" \
        --tag "$IMAGE:$chip" \
        --load \
        "$repo_root"
    ;;
run)
    device_id="${1:-0}"
    src="${PYPTO_SRC_HOST:-$repo_root}"
    lib="${PYPTO_LIB_SRC_HOST:-$(dirname "$repo_root")/pypto-lib}"
    lib_mount=()
    if [ -d "$lib" ]; then
        lib_mount=(-v "$lib:/workspace/pypto-lib")
    else
        echo "note: no pypto-lib checkout at $lib — set PYPTO_LIB_SRC_HOST to mount one" >&2
    fi
    # A *named* volume, not a bind mount, and not --rm:
    #  - Docker pre-populates a fresh named volume from the image, so anything
    #    the image put under /opt/pypto/cache survives the mount rather than
    #    being hidden by it (a bind mount would hide it).
    #  - The cache must outlive the container but must NOT land in the source
    #    tree, where it would show up as untracked files and be shared with a
    #    host-side build that has a different toolchain.
    cache_volume="${PYPTO_CACHE_VOLUME:-pypto-dev-cache}"
    docker volume create "$cache_volume" >/dev/null

    access="$PYPTO_DEVICE_ACCESS"
    if [ "$access" = auto ]; then
        if docker info --format '{{range $name, $_ := .Runtimes}}{{$name}}
{{end}}' 2>/dev/null | grep -qx ascend; then
            access=runtime
        else
            access=privileged
        fi
    fi
    case "$access" in
    runtime)
        # The runtime injects the driver, the character devices, npu-smi and the
        # libraries. ASCEND_VISIBLE_DEVICES selects the card, and it is the only
        # one the container can see — nothing else has to be named here.
        #
        # `npu-smi info` fails inside such a container (-9005): it enumerates
        # every board on the machine and the runtime deliberately exposes one.
        # That is the isolation working, not a fault; aclrtSetDevice succeeds.
        device_args=(--runtime=ascend -e "ASCEND_VISIBLE_DEVICES=$device_id")
        ;;
    privileged)
        # No runtime: name the devices, mount the driver tree *whole* (with only
        # lib64 + version.info npu-smi cannot reach the board at all, -9005),
        # and take the privilege hit. This hands over every card on the host.
        device_args=(--privileged
            --device "/dev/davinci$device_id"
            --device /dev/davinci_manager
            --device /dev/devmm_svm
            --device /dev/hisi_hdc
            "${extra_devices[@]}"
            -v /usr/local/dcmi:/usr/local/dcmi
            -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi
            -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro
            -v /etc/ascend_install.info:/etc/ascend_install.info)
        ;;
    *)
        echo "unknown PYPTO_DEVICE_ACCESS '$access' (expected auto, runtime or privileged)" >&2
        exit 2
        ;;
    esac
    echo "device access: $access (card $device_id)"

    # The container needs the proxy too, not just the build: the first
    # pypto-build clones the pinned pto-isa over HTTPS from inside it.
    proxy_env=()
    for v in HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy; do
        [ -n "${!v:-}" ] && proxy_env+=(-e "$v=${!v}")
    done

    container="pypto-dev-$chip-$device_id"
    # Re-enter rather than recreate: `uv pip install ./runtime` writes simpler
    # into the container's own venv, and a --rm container would throw that away
    # after every session, making each `docker run` recompile it.
    if docker container inspect "$container" >/dev/null 2>&1; then
        echo "re-entering $container (docker rm -f $container to start over)"
        exec docker start -ai "$container"
    fi

    set -x
    docker run -it \
        --name "$container" \
        --shm-size=16g \
        "${device_args[@]}" \
        -v "$src:/workspace/pypto" \
        "${lib_mount[@]}" \
        -v "$cache_volume:/opt/pypto/cache" \
        "${proxy_env[@]}" \
        -e "PYPTO_DEVICE=$device_id" \
        "$IMAGE:$chip"
    ;;
*)
    usage
    ;;
esac
