# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Scene 2 of 5: ptoas-rs -- the pipeline run backwards, and what rustc can prove.

source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
ROOT="$(demo_root)"; cd "$ROOT"
LIFT="$(tool ptoas-rs)"; DIFF="$(tool pto-diff)"; RSPTO="$(tool rspto)"
W="$(mktemp -d)"; trap 'rm -rf "$W"' EXIT
B=rust/bench/playground16

beat "ptoas-rs -- PTO back to tile-rs"
note "The pipeline backwards. PTO in, Rust out, and a report on what the compiler can prove about it."
run "$LIFT" --help
pause 8

beat "Why lift at all"
note "Compiling the result puts two invariants in front of a checker the assembler does not run."
note "Const-generic Tile<R,C,T> makes a matmul with disagreeing inner dimensions a TYPE error."
note "And a tile is deliberately not Copy, so a double store is a borrow-check error."
run "$LIFT" $B/pto/down_proj.pto -o "$W/down_proj.rs"
run head -14 "$W/down_proj.rs"
pause 8

beat "Read the coverage line before trusting an exit of zero"
note "An op the lift does not model is carried through verbatim. That keeps a round trip faithful"
note "and says nothing about whether the lift understood the kernel. So every run prints it."
run "$LIFT" $B/pto/k_proj.pto --stdout
pause 6

beat "Four refusals, four different causes"
note "Ten of the fourteen bench artefacts lift. The four that do not are not one problem."
run "$LIFT" $B/pto/rms_norm.pto -o "$W/a.rs"
expect_refusal "a scalar literal is not threaded into the emitter -- wiring, nothing deeper"
run "$LIFT" $B/pto/matmul.pto -o "$W/b.rs"
expect_refusal "there is no tile-to-tile copy wrapper to call: this one needs a new intrinsic"
run "$LIFT" $B/pto/rope.pto -o "$W/c.rs"
expect_refusal "an operation that is not modelled at all"
run "$LIFT" $B/pto/attn_gqa.pto -o "$W/d.rs"
expect_refusal "a strided view. THIS one must stay refused -- lifting it would read the wrong elements"
note "That last distinction is the useful one. Three are work; one is a correctness boundary."
pause 8

beat "The safety report"
note "Aliasing, capacity, operand constraints, matmul bounds, dead tiles, linear use. Advisory, on the console."
run "$LIFT" $B/pto/down_proj.pto --stdout --safety
note "The address-shape checks read an assembler memory-plan dump, not a bare .pto, and it says so."
pause 6

beat "Handing the kernel to the Rust compiler"
note "--verify compiles what was lifted. This is the difference between 'rustc could' and 'rustc did'."
run "$LIFT" $B/pto/down_proj.pto --stdout --verify rustc
note "Unavailable, not clean -- and exit code 4, not 0. Those are not the same answer,"
note "and a caller reading only a status must not be able to confuse them."
pause 8

beat "The intermediate representation"
note "--emit tile would write the tile-dialect MLIR. That needs the Rust-to-MLIR half."
run "$LIFT" $B/pto/down_proj.pto --emit tile
expect_refusal "an accepted flag that silently did nothing would be worse than a missing one"
pause 5

beat "pto-diff -- what moved between two memory plans"
note "A different question, useful without lifting anything: which tiles moved, and where to."
run "$DIFF" --help
pause 6

beat "The round trip, closed"
note "Lift a kernel to Rust, and separately re-emit its PTO from the MLIR. Both directions, one corpus."
run "$RSPTO" $B/tile/down_proj.tile -o "$W/re.pto"
run cmp $B/pto/down_proj.pto "$W/re.pto"
note "Byte-identical. Every generated file in the bench is re-derived on every test run, never trusted."
pause 6

demo_end
