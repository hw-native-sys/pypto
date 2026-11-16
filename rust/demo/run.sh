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
# Drive the tool demos, and turn them into something narratable.
#
#   rust/demo/run.sh plan              what each segment will do, and how long it runs
#   rust/demo/run.sh cues              the narration cue sheets, with timestamps
#   rust/demo/run.sh record [scene]    record the terminal session(s) to .cast
#   rust/demo/run.sh gif    [scene]    .cast -> .gif
#   rust/demo/run.sh mp4    [scene]    .gif -> .mp4 (silent)
#   rust/demo/run.sh voice  [scene]    narration -> .aiff with a male voice
#   rust/demo/run.sh mux    [scene]    .mp4 + .aiff -> narrated .mp4
#   rust/demo/run.sh publish[scene]   copy the gif and mp4 into demo/media/ for the READMEs
#   rust/demo/run.sh all               every step, every scene
#
# On the five-minute target: each segment is PACED for roughly five minutes of
# narrated runtime, and the recorder MEASURES what it actually produced and says
# so. It does not pad a short recording to look like a long one, and it does not
# claim five minutes for a three-minute file -- `record` prints the measured
# length against the target and warns when they disagree. A number nobody
# checked is the one thing a demo must not ship.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
OUT="${DEMO_OUT:-$HERE/out}"
MEDIA="${DEMO_MEDIA:-$HERE/media}"
TARGET_S="${DEMO_TARGET_S:-300}"
TOLERANCE="${DEMO_TOLERANCE:-45}"
GEOMETRY="${DEMO_GEOMETRY:-100x30}"
VOICE="${DEMO_VOICE:-Daniel}"     # a male voice; `say -v '?'` lists the rest

SCENES=(rspto ptoas-rs tile-search pto-to-ascendc ds4-rs-ascend)

# Per-scene pacing, calibrated from a measured recording rather than guessed.
# A scene that measures 190s needs about 1.6 to reach five minutes; raising this
# stretches every pause AND the typing, which is what keeps the two in step.
#
# A case statement rather than an associative array, because macOS still ships
# bash 3.2 and `declare -A` is a bash 4 feature -- the first version of this
# script died on `unbound variable` on the machine it was written on.
speed_for() {
  case "$1" in
    rspto)          printf '1.60' ;;
    ptoas-rs)       printf '1.55' ;;
    tile-search)    printf '1.60' ;;
    pto-to-ascendc) printf '1.90' ;;
    ds4-rs-ascend)  printf '1.50' ;;
    *)              printf '1.00' ;;
  esac
}

die() { printf 'demo: %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

cast_seconds() {
  # Cast v3 stores each event's time as a DELTA from the previous one, not as an
  # absolute offset. Reading the last field gives the final gap -- about two
  # seconds -- and makes a three-minute recording look empty. Sum them.
  python3 - "$1" <<'PY'
import json, sys
total = 0.0
with open(sys.argv[1]) as f:
    for i, line in enumerate(f):
        if i == 0:
            continue
        try:
            total += float(json.loads(line)[0])
        except Exception:
            pass
print(f"{total:.1f}")
PY
}

mmss() { awk -v t="$1" 'BEGIN{printf "%02d:%02d", int(t/60), int(t)%60}'; }

scene_list() { [ "$#" -gt 0 ] && [ -n "${1:-}" ] && printf '%s\n' "$1" || printf '%s\n' "${SCENES[@]}"; }

cmd_plan() {
  printf '%-18s %-10s %-10s %s\n' SCENE PROJECTED SPEED "AT THAT SPEED"
  for s in "${SCENES[@]}"; do
    local p sp est
    p="$(DEMO_DRY=1 bash "$HERE/scenes/$s.sh" | awk -F'\t' '$1=="projected_seconds"{print $2}')"
    sp="$(speed_for "$s")"
    est="$(awk -v p="$p" -v s="$sp" 'BEGIN{printf "%.0f", p*s}')"
    printf '%-18s %-10s %-10s %s\n' "$s" "$(mmss "$p")" "$sp" "$(mmss "$est")"
  done
  printf '\nProjected excludes real command runtime, so a recording runs longer.\n'
  printf 'Target %ss with +/-%ss tolerance; `record` measures and reports.\n' "$TARGET_S" "$TOLERANCE"
}

cmd_cues() {
  mkdir -p "$HERE/narration"
  for s in $(scene_list "${1:-}"); do
    local f="$HERE/narration/$s.txt"
    {
      printf 'Narration cue sheet -- %s\n' "$s"
      printf 'Timestamps are at DEMO_SPEED=%s and shift with it.\n' "$(speed_for "$s")"
      printf 'Read each line at its mark; the terminal holds for it.\n\n'
      DEMO_DRY=1 DEMO_SPEED="$(speed_for "$s")" bash "$HERE/scenes/$s.sh" \
        | sed -n '/--- narration cues ---/,$p' | tail -n +2
    } > "$f"
    printf 'wrote %s (%s cues)\n' "${f#"$ROOT"/}" "$(grep -c $'\t' "$f" || true)"
  done
}

cmd_record() {
  have asciinema || die "asciinema is not installed"
  mkdir -p "$OUT"
  for s in $(scene_list "${1:-}"); do
    local cast="$OUT/$s.cast"
    printf '\n=== recording %s (speed %s) ===\n' "$s" "$(speed_for "$s")"
    DEMO_SPEED="$(speed_for "$s")" asciinema rec "$cast" --overwrite -q \
      --window-size "$GEOMETRY" -c "bash $HERE/scenes/$s.sh" \
      || die "recording $s failed"
    local d; d="$(cast_seconds "$cast")"
    local delta; delta="$(awk -v d="$d" -v t="$TARGET_S" 'BEGIN{printf "%.0f", d-t}')"
    printf '%-18s %s  (target %s, off by %ss)' "$s" "$(mmss "$d")" "$(mmss "$TARGET_S")" "$delta"
    if awk -v d="$d" -v t="$TARGET_S" -v tol="$TOLERANCE" \
         'BEGIN{exit !(d < t-tol || d > t+tol)}'; then
      printf '  <-- OUTSIDE TOLERANCE: adjust speed_for() for %s in this script\n' "$s"
    else
      printf '  ok\n'
    fi
  done
}

cmd_gif() {
  have agg || die "agg is not installed (asciinema gif generator)"
  for s in $(scene_list "${1:-}"); do
    [ -f "$OUT/$s.cast" ] || die "no recording for $s -- run `record` first"
    # --idle-time-limit is the one flag that matters here and its default is 5
    # SECONDS. In a narrated demo the pauses ARE the content -- the holds are
    # where the voice-over goes -- and the default silently clipped every one of
    # them, so a 290-second recording rendered as a 233-second GIF and the
    # narration marks no longer lined up with the picture. Set it longer than
    # the longest hold any scene takes.
    agg --font-size 16 --theme asciinema --idle-time-limit 120 \
      "$OUT/$s.cast" "$OUT/$s.gif" || die "gif for $s failed"
    printf '%-18s %s\n' "$s.gif" "$(du -h "$OUT/$s.gif" | cut -f1)"
  done
  printf '\nA five-minute terminal GIF is tens of megabytes; the mp4 below is the\n'
  printf 'one to hand to a player, and is what the voice-over attaches to.\n'
}

cmd_mp4() {
  have ffmpeg || die "ffmpeg is not installed"
  for s in $(scene_list "${1:-}"); do
    [ -f "$OUT/$s.gif" ] || die "no gif for $s -- run `gif` first"
    # fps_mode passthrough keeps the GIF's own variable frame delays. Without
    # it ffmpeg re-times the stream and the video comes out shorter than the
    # recording, which puts the narration out of step all over again.
    ffmpeg -y -loglevel error -i "$OUT/$s.gif" \
      -movflags +faststart -pix_fmt yuv420p -fps_mode passthrough \
      -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" "$OUT/$s.silent.mp4" \
      || die "mp4 for $s failed"
    # And verify it, because this is exactly the step that lost time before.
    local gd vd
    gd="$(cast_seconds "$OUT/$s.cast")"
    vd="$(ffprobe -v error -select_streams v:0 -show_entries stream=duration \
           -of csv=p=0 "$OUT/$s.silent.mp4" 2>/dev/null)"
    awk -v g="$gd" -v v="$vd" 'BEGIN{
      d=v-g; if (d<0) d=-d;
      if (d > 0.06*g) printf "  WARNING: mp4 is %.0fs against a %.0fs recording\n", v, g
      else printf "  mp4 %.0fs matches the %.0fs recording\n", v, g }'
    printf '%-18s %s\n' "$s.silent.mp4" "$(du -h "$OUT/$s.silent.mp4" | cut -f1)"
  done
}

cmd_voice() {
  have say || die "`say` is only available on macOS; record the voice-over by hand from demo/narration/"
  mkdir -p "$OUT"
  cmd_cues "${1:-}" >/dev/null
  for s in $(scene_list "${1:-}"); do
    # A DRAFT voice-over, from the cue sheet, so the timing can be judged before
    # anyone books a studio. It is not a finished read: `say` ignores the marks,
    # so the lines run back to back and will drift from the picture. The cue
    # sheet carries the marks a human read would hit.
    local txt="$HERE/narration/$s.txt"
    cut -f2- "$txt" | grep -v '^SECTION:' | sed '/^$/d' > "$OUT/$s.lines.txt"
    say -v "$VOICE" -o "$OUT/$s.draft.aiff" -f "$OUT/$s.lines.txt" \
      || die "voice for $s failed"
    printf '%-18s %s with voice %s\n' "$s.draft.aiff" \
      "$(du -h "$OUT/$s.draft.aiff" | cut -f1)" "$VOICE"
  done
  printf '\nDRAFT only: these lines run back to back and WILL drift from the\n'
  printf 'picture. demo/narration/<scene>.txt carries the timestamps a read hits.\n'
}

cmd_mux() {
  have ffmpeg || die "ffmpeg is not installed"
  for s in $(scene_list "${1:-}"); do
    [ -f "$OUT/$s.silent.mp4" ] || die "no silent mp4 for $s"
    local audio="$OUT/$s.aiff"
    [ -f "$audio" ] || audio="$OUT/$s.draft.aiff"
    [ -f "$audio" ] || die "no audio for $s -- run `voice`, or drop $s.aiff in $OUT"
    # The PICTURE is authoritative, so the audio is padded with silence to its
    # length. `-shortest` alone truncates the VIDEO to the audio, and a draft
    # voice-over is far shorter than its segment -- that silently cut a
    # 306-second recording to a 178-second file, after the mp4 step had already
    # verified the silent version as correct. The published artefact was the
    # muxed one, so the check passed and the output was still wrong.
    ffmpeg -y -loglevel error -i "$OUT/$s.silent.mp4" -i "$audio" \
      -filter_complex "[1:a]apad[padded]" -map 0:v -map "[padded]" \
      -c:v copy -c:a aac -b:a 160k -shortest "$OUT/$s.mp4" \
      || die "mux for $s failed"
    # Verify the thing that will actually be PUBLISHED, not an intermediate.
    local cd_ vd_
    cd_="$(cast_seconds "$OUT/$s.cast")"
    vd_="$(ffprobe -v error -select_streams v:0 -show_entries stream=duration \
            -of csv=p=0 "$OUT/$s.mp4" 2>/dev/null)"
    awk -v c="$cd_" -v v="$vd_" 'BEGIN{
      d=v-c; if (d<0) d=-d;
      if (d > 0.08*c) printf "  WARNING: narrated mp4 is %.0fs against a %.0fs recording\n", v, c
      else printf "  narrated mp4 %.0fs matches the %.0fs recording\n", v, c }'
    printf '%-18s %s\n' "$s.mp4" "$(du -h "$OUT/$s.mp4" | cut -f1)"
  done
}

cmd_publish() {
  # out/ is the working directory and is gitignored: casts, intermediate mp4s,
  # draft audio. media/ is what the READMEs reference, so copying into it is a
  # deliberate step rather than a side effect of recording.
  mkdir -p "$MEDIA"
  for s in $(scene_list "${1:-}"); do
    for ext in gif mp4; do
      [ -f "$OUT/$s.$ext" ] || { printf 'skip %-18s no %s yet\n' "$s" "$ext"; continue; }
      cp "$OUT/$s.$ext" "$MEDIA/$s.$ext"
      printf '%-28s %s\n' "media/$s.$ext" "$(du -h "$MEDIA/$s.$ext" | cut -f1)"
    done
  done
  printf '\ntotal published: %s\n' "$(du -sh "$MEDIA" | cut -f1)"
}

main() {
  local what="${1:-plan}"; shift || true
  case "$what" in
    plan)   cmd_plan ;;
    cues)   cmd_cues "${1:-}" ;;
    record) cmd_record "${1:-}" ;;
    gif)    cmd_gif "${1:-}" ;;
    mp4)    cmd_mp4 "${1:-}" ;;
    voice)  cmd_voice "${1:-}" ;;
    mux)    cmd_mux "${1:-}" ;;
    publish) cmd_publish "${1:-}" ;;
    all)    cmd_cues ""; cmd_record ""; cmd_gif ""; cmd_mp4 ""; cmd_voice ""; cmd_mux ""; cmd_publish "" ;;
    *)      die "unknown command '$what' (plan|cues|record|gif|mp4|voice|mux|publish|all)" ;;
  esac
}

main "$@"
