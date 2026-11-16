# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Shared helpers for the tool demos.
#
# The hard part of a narrated terminal demo is not running the commands, it is
# PACING. A script that runs as fast as it can produces a ten-second recording
# that no one can narrate; a script padded with guesses produces six minutes of
# dead air in the wrong places. So pacing here is explicit and accountable:
#
#   * every pause goes through `pause`, which adds to a running total;
#   * `DEMO_DRY=1` runs no commands and sleeps not at all, and instead prints
#     the projected duration and the narration cue sheet. That is what makes a
#     five-minute target checkable BEFORE a recording is made rather than after;
#   * `say` marks a narration beat with the words that belong there, so the
#     voice-over has timestamps instead of a transcript to guess from.
#
# Typing is simulated rather than instant, because a viewer needs to read a
# command before its output replaces the screen. `DEMO_SPEED` scales every
# duration at once, so one number moves the whole runtime.

set -uo pipefail

DEMO_SPEED="${DEMO_SPEED:-1.0}"
DEMO_DRY="${DEMO_DRY:-0}"
DEMO_COLS="${DEMO_COLS:-100}"

_DEMO_T=0          # projected seconds so far
_DEMO_CUES=""      # "mm:ss<TAB>text" per narration beat
_DEMO_FAILED=0

# Colours, but only when the output is a terminal. A cast recorded without them
# is unreadable; a cue sheet piped to a file should not carry escape codes.
if [ -t 1 ] && [ "$DEMO_DRY" = 0 ]; then
    C_HEAD=$'\033[1;36m'; C_CMD=$'\033[1;32m'; C_DIM=$'\033[2m'
    C_WARN=$'\033[1;33m'; C_OK=$'\033[1;32m'; C_OFF=$'\033[0m'
else
    C_HEAD=""; C_CMD=""; C_DIM=""; C_WARN=""; C_OK=""; C_OFF=""
fi

_scaled() { awk -v d="$1" -v s="$DEMO_SPEED" 'BEGIN{printf "%.2f", d*s}'; }

_mmss() {
    awk -v t="$1" 'BEGIN{printf "%02d:%02d", int(t/60), int(t)%60}'
}

# pause SECONDS -- hold the screen so narration can land.
pause() {
    local d; d="$(_scaled "${1:-1}")"
    _DEMO_T="$(awk -v a="$_DEMO_T" -v b="$d" 'BEGIN{printf "%.2f", a+b}')"
    [ "$DEMO_DRY" = 1 ] || sleep "$d"
}

# say WORDS... -- a narration beat. Records the cue AND the time it falls at,
# which is the whole point: a voice-over needs to know when, not only what.
# Records the cue and its timestamp. Prints NOTHING: the callers below own what
# reaches the screen, and an earlier version printed here as well, so every
# `note` appeared twice.
say_beat() {
    _DEMO_CUES="${_DEMO_CUES}$(_mmss "$_DEMO_T")	$*
"
}

# beat TITLE -- a section header the viewer can orient by.
beat() {
    local bar; bar="$(printf '%*s' "$DEMO_COLS" '' | tr ' ' '-')"
    if [ "$DEMO_DRY" = 0 ]; then
        printf '\n%s%s%s\n%s  %s%s\n%s%s%s\n\n' \
            "$C_HEAD" "$bar" "$C_OFF" "$C_HEAD" "$*" "$C_OFF" "$C_HEAD" "$bar" "$C_OFF"
    fi
    say_beat "SECTION: $*"
    pause 2
}

# type_cmd CMD -- show a command as if typed, without running it.
type_cmd() {
    local cmd="$*"
    if [ "$DEMO_DRY" = 0 ]; then
        printf '%s$ %s' "$C_CMD" ""
        local i ch
        for (( i=0; i<${#cmd}; i++ )); do
            ch="${cmd:$i:1}"
            printf '%s' "$ch"
            # 55 chars/second is brisk but followable; a long command would
            # otherwise eat the whole segment.
            sleep "$(_scaled 0.018)"
        done
        printf '%s\n' "$C_OFF"
    fi
    local n=${#cmd}
    _DEMO_T="$(awk -v a="$_DEMO_T" -v n="$n" -v s="$DEMO_SPEED" 'BEGIN{printf "%.2f", a+n*0.018*s}')"
}

# run CMD... -- type it, run it, show what happened.
#
# A non-zero exit is NOT treated as a demo failure, because several of these
# tools refuse by design and the refusal is the thing worth showing. Use
# `expect_refusal` to say so out loud, so a viewer does not read an intended
# refusal as a broken demo.
run() {
    type_cmd "$@"
    if [ "$DEMO_DRY" = 0 ]; then
        "$@" || true
    fi
    pause 1.5
}

# expect_refusal WHY -- label the refusal that just happened as intended.
expect_refusal() {
    if [ "$DEMO_DRY" = 0 ]; then
        printf '%s  ^ this refusal is the point: %s%s\n' "$C_WARN" "$*" "$C_OFF"
    fi
    say_beat "That refusal is intended: $*"
    pause 3
}

# note WORDS... -- an on-screen aside, and a narration cue with it.
note() {
    if [ "$DEMO_DRY" = 0 ]; then
        printf '%s  %s%s\n' "$C_DIM" "$*" "$C_OFF"
    fi
    say_beat "$*"
    pause 2
}

# demo_end -- print the accounting. In dry mode this is the whole output.
demo_end() {
    if [ "$DEMO_DRY" = 1 ]; then
        printf 'projected_seconds\t%.0f\n' "$_DEMO_T"
        printf 'projected_mmss\t%s\n' "$(_mmss "$_DEMO_T")"
        printf -- '--- narration cues ---\n%s' "$_DEMO_CUES"
    else
        printf '\n%s%s%s\n' "$C_OK" "demo segment complete." "$C_OFF"
        pause 2
    fi
}

# The repository root, so a scene can find the bench and the binaries without
# depending on where it was invoked from.
demo_root() {
    # demo lives at rust/demo, so the REPOSITORY root -- which is what a scene
    # needs for bench/ -- is two levels up. The binaries are under rust/.
    cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd
}

# A tool's path, preferring the built release binary and falling back to PATH so
# the demo also works for someone who installed with install.sh.
tool() {
    local name="$1" root; root="$(demo_root)"
    for p in "$root/rust/target/release/$name" \
             "$root/rust/ds4-rs-ascend/target/release/$name"; do
        [ -x "$p" ] && { printf '%s' "$p"; return 0; }
    done
    if command -v "$name" >/dev/null 2>&1; then
        command -v "$name"
        return 0
    fi
    # Fail HARD rather than returning a sentinel. An earlier version printed
    # `MISSING:<name>` and returned non-zero, and the scene then ran every
    # command against that string, failed every one, and still printed "demo
    # segment complete". A recording of ten failed commands that calls itself
    # complete is the exact artefact this whole project refuses to ship.
    printf '\ndemo: %s is not built.\n' "$name" >&2
    printf 'demo: build it first:\n' >&2
    printf 'demo:   cd rust            && cargo build --release\n' >&2
    printf 'demo:   cd ds4-rs-ascend   && cargo build --release\n' >&2
    exit 127
}
