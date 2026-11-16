# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Scene 3 of 5: tile-search -- enumerate the tilings a kernel admits.

source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
ROOT="$(demo_root)"; cd "$ROOT"
TS="$(tool tile-search)"; RSPTO="$(tool rspto)"
W="$(mktemp -d)"; trap 'rm -rf "$W"' EXIT
B=rust/bench/playground16; V=rust/bench/deepseek_v41; G=rust/bench/glm_5_3

beat "tile-search -- which tilings does this kernel admit?"
note "Knobs are read OUT of the kernel, not supplied. A value typed by hand is a guess."
run "$TS" --help
pause 8

beat "The case that matters: a kernel the hardware refuses"
note "We saw rspto refuse this one. Its live tile set is larger than the unified buffer."
run "$RSPTO" $B/tile/gate_up_silu.tile -o "$W/no.pto"
expect_refusal "so as written, this kernel cannot run at all"
note "Now ask the search what widths the target WOULD accept."
run "$TS" $B/tile/gate_up_silu.tile --dry-run
pause 10

beat "Read what just happened"
note "Eight candidates. Seven accepted. The captured width is the one that is refused."
note "And the oracle is the REAL emitter -- every candidate was compiled, not scored by a model."
note "A model of the bounds would be a second place for them to live, and the two would drift."
pause 8

beat "The ranking is a declared model, and says so"
note "Nothing here ran on hardware, so this cannot know which accepted tiling is faster."
note "It orders them by a stated rule -- largest accepted extent -- and prints that it is a model."
note "Its own failure mode is written down too: the largest extent may leave no room to double-buffer."
pause 8

beat "Writing the chosen kernel out"
run "$TS" $B/tile/gate_up_silu.tile -o "$W/rescued.tile"
note "And the choice has to survive the emitter that refused the original."
run "$RSPTO" "$W/rescued.tile" -o "$W/rescued.pto"
note "It emits. A kernel that could not run now can."
pause 6

beat "Doing nothing has to be reachable"
note "A search that cannot choose the input it was given is not a search."
run "$TS" $B/tile/down_proj.tile --dry-run
note "Already the largest accepted extent, so it says there is nothing to change."
pause 6

beat "Real model shapes push harder"
note "DeepSeek V4.1's own dimensions, not teaching sizes. Seven of fourteen are refused."
run "$TS" $V/tile/rmsnorm_v41_8x5120.tile --dry-run
pause 8

beat "And the limit of a tiling search, shown rather than claimed"
note "GLM 5.3 routes over 288 experts. Watch the search try, and fail honestly."
run "$TS" $G/tile/router_topk_glm53_1x288_k8.tile --dry-run
expect_refusal "every candidate refused -- and the arithmetic says why"
note "A row must span whole 512-byte fractals, so an f32 extent must be a multiple of 128."
note "288 has no divisor that is. The expert count is the obstruction, not the tiling."
note "No choice of tiling can move that, and a search claiming otherwise would be lying."
pause 10

beat "Lifting the winner back to tile-rs"
note "--emit rs runs the chosen tiling through the lift, so the output is a kernel, not a number."
run "$TS" $B/tile/down_proj.tile --emit rs -o "$W/opt.rs"
run head -8 "$W/opt.rs"
pause 6

demo_end
