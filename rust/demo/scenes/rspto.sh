# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Scene 1 of 5: rspto -- a tile-rs kernel to PTO assembly.
#
# The spine of the segment is the claim that matters and the evidence for it:
# two spellings of the same input cannot produce different PTO, and a kernel the
# hardware cannot run is refused with the bound named rather than emitted.

source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
ROOT="$(demo_root)"; cd "$ROOT"
RSPTO="$(tool rspto)"
W="$(mktemp -d)"; trap 'rm -rf "$W"' EXIT
B=rust/bench/playground16

beat "rspto -- lowering a kernel to PTO"
note "A Rust path into PTO, alongside the Python one. Stable Rust, no LLVM, no third-party crates."
run "$RSPTO" --version
run "$RSPTO" --help
pause 6

beat "The input is chosen by extension, and nothing else"
note ".tile is tile-dialect MLIR. It needs nothing at all -- no backend, no download."
run head -6 $B/tile/rms_norm.tile
pause 3
run "$RSPTO" $B/tile/rms_norm.tile -o "$W/rms_norm.pto"
note "That is PTO assembly, for the same assembler the Python backend feeds."
run head -12 "$W/rms_norm.pto"
pause 6

beat "The two spellings cannot disagree"
note ".mlir is the spelling the codegen backend writes. Accepted so an artefact needs no renaming."
run cp $B/tile/rms_norm.tile "$W/same.mlir"
run "$RSPTO" "$W/same.mlir" -o "$W/from_mlir.pto"
note "Same emitter, so the question is whether the bytes agree. Not an opinion -- a diff."
run cmp "$W/rms_norm.pto" "$W/from_mlir.pto"
note "No output from cmp means byte-identical. A test in the suite asserts this on every run."
pause 6

beat "Stopping at the intermediate representation"
note "--emit tile writes the MLIR instead of the PTO, which is what a build system wants mid-pipeline."
run "$RSPTO" $B/tile/vec_add.tile --emit tile -o "$W/vec_add.out.tile"
run head -8 "$W/vec_add.out.tile"
pause 5

beat "A refusal is a result"
note "Two of the sixteen bench kernels cannot run on this part. Watch what the tool does about it."
run "$RSPTO" $B/tile/gate_up_silu.tile -o "$W/x.pto"
expect_refusal "the live tile set is larger than the unified buffer, and the message says by how much"
run "$RSPTO" $B/tile/argmax.tile -o "$W/y.pto"
expect_refusal "a different bound -- the row-major tile bound for this architecture"
note "Neither is emitted and left to fail on a card. The bound is named, with the numbers."
pause 5

beat "The reshaped variants, which do run"
note "The corpus carries the same computations at widths that fit, so a refusal is not a dead end."
run "$RSPTO" $B/tile/fixed/gate_up_silu.tile -o "$W/fixed.pto"
run "$RSPTO" $B/tile/fixed/argmax.tile -o "$W/fixed2.pto"
pause 4

beat "The whole corpus, in one pass"
note "Sixteen kernels. Fourteen emit; two are refused for the reasons just shown."
if [ "$DEMO_DRY" = 0 ]; then
  ok=0; no=0
  for f in $B/tile/*.tile; do
    if "$RSPTO" "$f" -o "$W/$(basename "$f" .tile).pto" >/dev/null 2>&1; then
      ok=$((ok+1)); printf '  %-22s %semits%s\n' "$(basename "$f" .tile)" "$C_OK" "$C_OFF"
    else
      no=$((no+1)); printf '  %-22s %srefused%s\n' "$(basename "$f" .tile)" "$C_WARN" "$C_OFF"
    fi
    sleep 0.25
  done
  printf '\n  %d emit, %d refused\n' "$ok" "$no"
fi
pause 8

beat "What the Rust input mode costs, and why it is absent"
note "A tile-rs .rs kernel needs a codegen backend loaded into rustc, plus a nightly toolchain."
run "$RSPTO" --check
note "Exit code 4, not 2. The mode is understood; the half that serves it is not built in."
run "$RSPTO" $B/tile-rs/vec_add.rs
expect_refusal "it names the missing half rather than reporting a bad filename"
pause 6

demo_end
