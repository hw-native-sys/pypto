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
#   docker/scripts/pypto-dev.sh build a3        # Ascend910B (A2/A3)
#   docker/scripts/pypto-dev.sh build a5        # Ascend950  (A5)
#   docker/scripts/pypto-dev.sh run   a3 [device-id]   # create or re-enter
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
        platform=a2a3
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
        platform=a5
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
        [ -n "${!v:-}" ] && proxy_args+=(--build-arg "$v")
    done
    docker buildx build \
        --file "$repo_root/docker/pypto_dev.Dockerfile" \
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
    src="$(cd "$src" && pwd -P)"
    if [ -d "$lib" ]; then lib="$(cd "$lib" && pwd -P)"; fi
    container_src=/workspace/pypto
    container_lib=/workspace/pypto-lib
    # Linked worktrees store an absolute gitdir outside their source directory.
    # Keep both source and metadata at their original paths: submodule gitdir
    # files may be relative, while core.worktree points back from the metadata.
    git_mount=()
    for checkout in "$src" "$lib"; do
        if [ -f "$checkout/.git" ]; then
            git_common="$(git -C "$checkout" rev-parse --path-format=absolute --git-common-dir)"
            git_mount+=(-v "$git_common:$git_common")
            # load-env.sh reads the ignored profile beside the primary .git,
            # outside the linked source mount. Preserve that cap, even when
            # --jobs requests more workers, without exposing the primary tree.
            if [ "$checkout" = "$src" ]; then
                container_src="$src"
                profile="$(dirname "$git_common")/.claude/skills/testing/testing.env"
                if [ -f "$profile" ]; then
                    git_mount+=(--mount "type=bind,src=$profile,dst=$profile,readonly")
                fi
            else
                container_lib="$lib"
            fi
        fi
    done
    lib_mount=()
    if [ -d "$lib" ]; then
        lib_mount=(-v "$lib:$container_lib")
    else
        echo "note: no pypto-lib checkout at $lib — set PYPTO_LIB_SRC_HOST to mount one" >&2
    fi
    # Root in the container does not own host checkouts. Supply protected Git
    # configuration for exactly the mounted roots and initialized submodules,
    # including nested ones. Do not change the host's global Git configuration.
    git_env=()
    safe_count=0
    for checkout in "$src" "$lib"; do
        [ -d "$checkout" ] || continue
        destination="$container_src"
        if [ "$checkout" = "$lib" ]; then destination="$container_lib"; fi
        safe_paths=("$destination")
        if [ -e "$checkout/.git" ]; then
            # foreach visits initialized submodules only; any subsequently
            # initialized modules will be owned by the container's own user.
            submodules="$(git -C "$checkout" submodule foreach --quiet --recursive 'printf "%s\n" "$displaypath"')"
            while IFS= read -r submodule; do
                [ -n "$submodule" ] && safe_paths+=("$destination/$submodule")
            done <<< "$submodules"
        fi
        for safe_path in "${safe_paths[@]}"; do
            git_env+=(-e "GIT_CONFIG_KEY_$safe_count=safe.directory" -e "GIT_CONFIG_VALUE_$safe_count=$safe_path")
            safe_count=$((safe_count + 1))
        done
    done
    git_env+=(-e "GIT_CONFIG_COUNT=$safe_count")
    # A *named* volume, not a bind mount, and not --rm:
    #  - Docker pre-populates a fresh named volume from the image, so anything
    #    the image put under /opt/pypto/cache survives the mount rather than
    #    being hidden by it (a bind mount would hide it).
    #  - The cache must outlive the container but must NOT land in the source
    #    tree, where it would show up as untracked files and be shared with a
    #    host-side build that has a different toolchain.
    cache_volume="${PYPTO_CACHE_VOLUME:-pypto-dev-cache}"

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
        # A single selected physical card is logical device 0 inside the container.
        container_device=0
        ;;
    privileged)
        container_device="$device_id"
        echo 'Privileged fallback exposes every host device/card; the selected card is only the default compute device.' >&2
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
        # Docker reads the value from its environment; keep credentials out of
        # command arguments and shell traces.
        [ -n "${!v:-}" ] && proxy_env+=(-e "$v")
    done

    container="pypto-dev-$chip-$device_id"
    # Include effective mounts and device settings, but never proxy credentials.
    # The version marker makes future changes to this contract explicit.
    config="$(printf '%s\0' pypto-dev-v3 "$src" "$container_src" "$container_lib" "$IMAGE:$chip" "$cache_volume" \
        "$container_device" "$platform" "${device_args[@]}" "${git_mount[@]}" "${lib_mount[@]}" "${git_env[@]}" | sha256sum)"
    config="${config%% *}"
    # Re-enter rather than recreate: `uv pip install ./runtime` writes simpler
    # into the container's own venv, and a --rm container would throw that away
    # after every session, making each `docker run` recompile it.
    if existing="$(docker container inspect "$container" \
        --format '{{index .Config.Labels "org.pypto.dev.config"}} {{.Image}}' 2>/dev/null)"; then
        read -r existing_config existing_image <<< "$existing"
        image_id="$(docker image inspect "$IMAGE:$chip" --format '{{.Id}}')" || {
            echo "cannot verify the image for $container; restore $IMAGE:$chip before re-entering" >&2
            exit 1
        }
        if [ "$existing_config" != "$config" ] || [ "$existing_image" != "$image_id" ]; then
            echo "$container uses a different checkout, image, cache or device configuration, or predates configuration tracking." >&2
            echo "Preserve any container-only changes, then remove it explicitly with: docker rm $container" >&2
            echo "Run this launcher again to create the requested environment." >&2
            exit 1
        fi
        echo "re-entering $container (docker rm -f $container to start over)"
        exec docker start -ai "$container"
    fi

    docker volume create "$cache_volume" >/dev/null
    docker run -it \
        --name "$container" \
        --label "org.pypto.dev.config=$config" \
        --shm-size=16g \
        "${device_args[@]}" \
        -v "$src:$container_src" \
        "${git_mount[@]}" \
        "${lib_mount[@]}" \
        -v "$cache_volume:/opt/pypto/cache" \
        "${proxy_env[@]}" \
        "${git_env[@]}" \
        -e "PYPTO_SRC=$container_src" \
        -e "PYPTO_LIB_SRC=$container_lib" \
        -e "PYPTO_DEVICE=$container_device" \
        -e "PYPTO_PLATFORM=$platform" \
        --workdir "$container_src" \
        "$IMAGE:$chip"
    ;;
*)
    usage
    ;;
esac
