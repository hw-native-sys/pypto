#!/bin/bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# Fast-forward source, align PTOAS with the checkout, and rebuild the environment.
set -euo pipefail
export PYPTO_HOME="${PYPTO_HOME:-/opt/pypto}"
export PYPTO_SRC="${PYPTO_SRC:-/workspace/pypto}"
export PYPTO_LIB_SRC="${PYPTO_LIB_SRC:-/workspace/pypto-lib}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/pypto-owner.sh"
# Even read-only Git commands can refresh the index. Run all Git access with
# the corresponding repository's owner; the two checkouts may have different IDs.
checkout_git() {
    local checkout="$1"; shift
    pypto_as_checkout_owner "$checkout" git -C "$checkout" "$@"
}
pull=true
reinstall=false
doctor=true
remote_submodules=false
jobs=""
branch=""
setup_args=()

usage() {
    cat <<'USAGE'
Usage: pypto-update.sh [OPTIONS]
  --no-pull                 Keep source and submodules unchanged
  --branch NAME             Require this current branch in both repositories
  --pypto-src PATH          PyPTO checkout (including linked worktrees)
  --pypto-lib-src PATH      Optional pypto-lib checkout
  --reinstall-deps          Refresh dependencies with pypto-setup.sh
  --remote-submodules       Track submodule remotes instead of pinned commits
  --jobs N                  Explicit build parallelism
  --skip-doctor             Skip the closing device/environment checks
  --proxy URL               Proxy for HTTP and HTTPS
  --http-proxy URL          HTTP proxy
  --https-proxy URL         HTTPS proxy
  --no-proxy HOSTS          Hosts which bypass the proxy
  --pypi URL                Python package index
  --torch-index URL         CPU torch wheel index
  --uv-mirror URL           CPython download mirror
USAGE
}
while [ $# -gt 0 ]; do
    case "$1" in
        --no-pull) pull=false; shift ;;
        --reinstall-deps) reinstall=true; shift ;;
        --remote-submodules) remote_submodules=true; shift ;;
        --skip-doctor) doctor=false; shift ;;
        -h|--help) usage; exit 0 ;;
        --branch|--pypto-src|--pypto-lib-src|--jobs|--proxy|--http-proxy|--https-proxy|--no-proxy|--pypi|--pip-index|--torch-index|--uv-mirror)
            [ $# -ge 2 ] && [ -n "$2" ] || { echo "$1 requires a value" >&2; exit 2; }
            case "$1" in
                --branch) branch="$2" ;;
                --pypto-src) export PYPTO_SRC="$2" ;;
                --pypto-lib-src) export PYPTO_LIB_SRC="$2" ;;
                --jobs) jobs="$2" ;;
                --proxy) export HTTP_PROXY="$2" HTTPS_PROXY="$2" http_proxy="$2" https_proxy="$2" ;;
                --http-proxy) export HTTP_PROXY="$2" http_proxy="$2" ;;
                --https-proxy) export HTTPS_PROXY="$2" https_proxy="$2" ;;
                --no-proxy) export NO_PROXY="$2" no_proxy="$2" ;;
                --pypi|--pip-index) export PIP_INDEX_URL="$2" UV_INDEX_URL="$2" ;;
                --torch-index|--uv-mirror) setup_args+=("$1" "$2") ;;
            esac
            shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
if [ -n "$jobs" ]; then
    [[ "$jobs" =~ ^[1-9][0-9]*$ ]] || { echo "--jobs must be a positive integer" >&2; exit 2; }
fi

# Validate both repositories before updating either. Never change branches or reset
# commits implicitly; detached checkouts can still use --no-pull.
repos=("$PYPTO_SRC")
if [ -d "$PYPTO_LIB_SRC" ]; then
    repos+=("$PYPTO_LIB_SRC")
else
    echo "No pypto-lib checkout at $PYPTO_LIB_SRC; skipping its update"
fi
if $pull; then
    status_args=(--porcelain)
    if $remote_submodules; then status_args+=(--ignore-submodules=all); fi
    for repo in "${repos[@]}"; do
        checkout_git "$repo" rev-parse --is-inside-work-tree >/dev/null
        current_branch="$(checkout_git "$repo" symbolic-ref --quiet --short HEAD)" || {
            echo "$repo has detached HEAD; select a branch or use --no-pull" >&2; exit 1;
        }
        if [ -n "$branch" ] && [ "$branch" != "$current_branch" ]; then
            echo "$repo is on $current_branch, expected $branch; select the branch explicitly" >&2
            exit 1
        fi
        repo_status="$(checkout_git "$repo" status "${status_args[@]}")" || exit 1
        [ -z "$repo_status" ] || {
            echo "$repo has local changes; commit/stash them or use --no-pull" >&2; exit 1;
        }
        if $remote_submodules; then
            # An explicitly staged gitlink is a user edit, not the unstaged drift
            # produced by --remote. Keep it protected even when status hides links.
            checkout_git "$repo" diff --cached --quiet --ignore-submodules=none || {
                echo "$repo has local changes; commit/stash them or use --no-pull" >&2; exit 1;
            }
            # Remote tracking deliberately moves submodule HEADs off their gitlinks.
            # Ignore only that expected drift: check every populated submodule for
            # tracked/untracked edits separately, including nested submodules.
            checkout_git "$repo" submodule foreach --quiet --recursive '
                git diff --cached --quiet --ignore-submodules=none || {
                    echo "submodule $displaypath has local changes; commit/stash them or use --no-pull" >&2
                    exit 1
                }
                submodule_status="$(git status --porcelain --ignore-submodules=all)" || exit 1
                test -z "$submodule_status" || {
                    echo "submodule $displaypath has local changes; commit/stash them or use --no-pull" >&2
                    exit 1
                }
            ' || exit 1
        fi
        checkout_git "$repo" rev-parse --verify '@{upstream}' >/dev/null || {
            echo "$repo needs a tracking branch or --no-pull" >&2; exit 1;
        }
    done
    for repo in "${repos[@]}"; do
        checkout_git "$repo" pull --ff-only
    done
    submodule_args=(--init --recursive)
    if $remote_submodules; then submodule_args+=(--remote); fi
    checkout_git "$PYPTO_SRC" submodule update "${submodule_args[@]}"
fi

# Installed launchers can outlive the image's helper copies. Once the source is
# current, use its helpers together so build, activation and pin handling agree.
here="$PYPTO_SRC/docker/scripts"
for helper in pypto-build pypto-install-ptoas.sh pypto-setup.sh pypto-doctor; do
    [ -x "$here/$helper" ] || {
        echo "Missing checkout helper $here/$helper; select a checkout containing the container scripts" >&2
        exit 1
    }
done
if $reinstall; then
    BASH_ENV= "$here/pypto-setup.sh" "${setup_args[@]}"
else
    BASH_ENV= "$here/pypto-setup.sh" --helpers-only
    "$here/pypto-install-ptoas.sh"
fi
if [ -n "$jobs" ]; then export PYPTO_BUILD_JOBS="$jobs"; fi
"$here/pypto-build"
if $doctor; then "$here/pypto-doctor"; fi
echo "PyPTO update complete"
printf 'Reload this shell with: export PYPTO_HOME=%q PYPTO_SRC=%q PTOAS_ROOT=%q PYPTO_LIB_SRC=%q; source %q\n' \
    "$PYPTO_HOME" "$PYPTO_SRC" "${PTOAS_ROOT:-$PYPTO_HOME/ptoas}" \
    "$PYPTO_LIB_SRC" "$PYPTO_HOME/bin/pypto-activate.sh"
