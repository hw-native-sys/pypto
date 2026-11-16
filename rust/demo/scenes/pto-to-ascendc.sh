# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Scene 4 of 5: pto-to-ascendc -- the AscendC half of the comparison.

source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
ROOT="$(demo_root)"; cd "$ROOT"
ACC="$(tool pto-to-ascendc)"; RSPTO="$(tool rspto)"
W="$(mktemp -d)"; trap 'rm -rf "$W"' EXIT
B=rust/bench/playground16; V=rust/bench/deepseek_v41

beat "pto-to-ascendc -- the fourth representation"
note "A kernel exists here as tile-rs, as tile-dialect MLIR, as PTO, and as AscendC C++."
note "The point is comparison: which representation makes a thing easy to SEE."
run "$ACC" --help
pause 8

beat "One kernel, two representations, side by side"
run "$RSPTO" $B/tile/rms_norm.tile -o "$W/rms.pto"
run "$ACC" $B/tile/rms_norm.tile -o "$W/rms.cpp"
note "The same computation. First as PTO, which is close to the machine."
run sed -n '3,12p' "$W/rms.pto"
pause 5
note "And now as AscendC, which is C++ a kernel engineer reads."
run sed -n '12,26p' "$W/rms.cpp"
pause 8

beat "Generated code carries the licence header, emitted by the emitter"
note "Not bolted on by a script afterwards -- that would break every byte-for-byte comparison."
run head -4 "$W/rms.cpp"
pause 5

beat "The part is not optional, and the default is a real one"
note "--soc picks which Ascend part to emit for: 910b by default, or the 950 family."
run "$ACC" $B/tile/vec_add.tile --soc 950 -o "$W/v950.cpp"
note "Why it cannot be left blank: a blank target has NO unified buffer."
note "Every budget check then answers 'fits', and every kernel appears to overflow."
note "Measured before the default was made a real part: zero of sixteen emitted,"
note "with vec_add 'overflowing' at 1024 bytes. So the default is a part, not an absence."
pause 8

beat "The two emitters check different things"
note "Over the sixteen playground kernels, both emit fourteen. On real model shapes they diverge."
if [ "$DEMO_DRY" = 0 ]; then
  p=0; a=0; n=0
  for f in $V/tile/*.tile; do
    n=$((n+1))
    "$RSPTO" "$f" -o "$W/t.pto" >/dev/null 2>&1 && p=$((p+1))
    "$ACC"   "$f" -o "$W/t.cpp" >/dev/null 2>&1 && a=$((a+1))
    printf '  %-38s pto=%s ascendc=%s\n' "$(basename "$f" .tile)" \
      "$([ -f "$W/t.pto" ] && echo ok || echo no)" \
      "$([ -f "$W/t.cpp" ] && echo ok || echo no)"
    rm -f "$W/t.pto" "$W/t.cpp"; sleep 0.3
  done
  printf '\n  of %d V4.1 kernels: PTO %d, AscendC %d\n' "$n" "$p" "$a"
fi
note "AscendC covering less than PTO is a measured fact about the two emitters,"
note "not a gap in the corpus. The bench records a verdict per representation for that reason."
pause 10

beat "On the name, and what it does not do yet"
note "PTO input is the direction this tool is named for, and it is not implemented."
run "$ACC" $B/pto/vec_add.pto -o "$W/z.cpp"
expect_refusal "two routes lead there and neither is in this tree yet"
note "The pinned assembler does it directly and is the reference answer."
note "The other lifts the PTO back to tile-rs and lowers that. Having BOTH is the point:"
note "the two outputs can be diffed, and that diff says more than either route alone."
pause 8

demo_end
