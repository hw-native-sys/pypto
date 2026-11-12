#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Build and install the Rust frontend tools from source.
#
# Source rather than a prebuilt binary, deliberately: the tools are a few
# seconds of stable-Rust compilation with no third-party dependencies, so a
# release-artefact pipeline would add a supply chain to save almost nothing.
#
#   curl -fsSL <raw url>/rust/install.sh | bash
#
# Honours PREFIX (default ~/.local/bin) and PYPTO_BUILD_JOBS (default 2).

set -euo pipefail

PREFIX="${PREFIX:-$HOME/.local/bin}"
JOBS="${PYPTO_BUILD_JOBS:-2}"
REPO_URL="${PYPTO_REPO:-https://github.com/hw-native-sys/pypto}"
BRANCH="${PYPTO_BRANCH:-main}"

say() { printf 'install: %s\n' "$*" >&2; }
die() { printf 'install: %s\n' "$*" >&2; exit 1; }

command -v cargo >/dev/null 2>&1 || die \
  "cargo not found. Install Rust from https://rustup.rs and re-run. Stable is
   enough -- these tools need no nightly toolchain."

# Run from a checkout when there is one, so a contributor installs what they
# have rather than what is on a branch somewhere.
if [ -f "$(dirname "$0")/Cargo.toml" ]; then
    WORKSPACE="$(cd "$(dirname "$0")" && pwd)"
    say "building from $WORKSPACE"
    CLEANUP=""
else
    command -v git >/dev/null 2>&1 || die "git not found, and no local checkout to build from"
    TMP="$(mktemp -d)"
    CLEANUP="$TMP"
    say "cloning $REPO_URL ($BRANCH)"
    git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$TMP/pypto" >/dev/null 2>&1 \
        || die "clone failed: $REPO_URL ($BRANCH)"
    WORKSPACE="$TMP/pypto/rust"
fi

say "building with $JOBS job(s)"
( cd "$WORKSPACE" && cargo build --release --locked --jobs "$JOBS" )

mkdir -p "$PREFIX"
installed=0
for bin in rspto ptoas-rs pto-diff pto-to-ascendc tile-search; do
    src="$WORKSPACE/target/release/$bin"
    [ -x "$src" ] || { say "skipping $bin: not built"; continue; }
    install -m 0755 "$src" "$PREFIX/$bin"
    say "installed $PREFIX/$bin"
    installed=$((installed + 1))
done
[ "$installed" -gt 0 ] || die "nothing was installed"

[ -n "$CLEANUP" ] && rm -rf "$CLEANUP"

case ":$PATH:" in
    *":$PREFIX:"*) ;;
    *) say "note: $PREFIX is not on PATH" ;;
esac

say "done. Try: rspto --help"
say "note: the ds4-rs-ascend serving utility is a separate workspace; build it"
say "      with: cd rust/ds4-rs-ascend && cargo build --release"
