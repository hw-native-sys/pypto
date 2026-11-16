// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! `mlir_to_pto`'s PASSTHROUGH FAMILY, pinned arm by arm.
//!
//! Nine arms of `mlir_to_pto` lower an op to something that is not that op and
//! write a comment saying so. No corpus kernel reaches any of them, so no gate in
//! this repository had ever looked at one: the three-column byte-identity gates
//! compare stored artefacts, and nothing stored exercises these arms. This file
//! exercises each of them directly, from a synthetic `.tile`, so the family has a
//! state that is written down and cannot drift unnoticed.
//!
//! # What was measured, 2026-11-08
//!
//! Each arm's claim about PTO was checked against the assembler instead of being
//! read. **Eight of the nine assert a capability gap that does not exist.**
//!
//! | arm | its own claim | the op that does exist | downstream verdict |
//! |---|---|---|---|
//! | transpose | "PTO lacks native transpose" | `pto.ttrans` -> `TTRANS` | REFUSED |
//! | slice | "TODO ... when supported" | `pto.textract` -> `TEXTRACT` | REFUSED |
//! | concat | "PTO lacks native concat" | `pto.tinsert` -> `TINSERT` | REFUSED |
//! | rsqrt | "PTO lacks native rsqrt" | `pto.trsqrt` -> `TRSQRT` | accepted |
//! | log | "PTO lacks native log op" | `pto.tlog` -> `TLOG` | accepted |
//! | cast | "PTO does not have a native `pto.tcast`" | `pto.tcvt` -> `TCVT` | accepted |
//! | gather | "no PTO equivalent" | `pto.tgather` | accepted |
//! | scatter | "no PTO equivalent" | `pto.tscatter` | accepted |
//! | rope (STUB) | "PTO lacks sin/cos/pow" | **nothing** -- `pto.tsin` and `pto.tcos` are unknown ops | accepted |
//!
//! The rope stub is the ONE honest disclaimer in the family, and it is the only
//! one with a corpus artefact behind it.
//!
//! # The split that matters: three are loud, five are silent
//!
//! REFUSED above means the emitted `.pto` does not assemble. All three fail for
//! one reason, and it is the passthrough itself: `pto.tmov` between tiles of
//! different SHAPES is rejected -- `'pto.tmov' op expects A2/A3 non-mat tmov to
//! use matching src/dst shapes` -- and transpose, slice and concat all change
//! shape. So those three cannot ship a wrong artefact; they are broken loudly, by
//! luck rather than by design.
//!
//! The other five keep the shape and only change the MEANING, so nothing stops
//! them: each assembles at rc=0 and emits a bare `TMOV` where `TRSQRT`, `TLOG`,
//! `TCVT`, a gather or a scatter belongs. Those five are the dangerous set, and
//! the earlier severity claim in this campaign -- that slice and concat shipped "a
//! clean-assembling wrong answer" -- was wrong about which arms do that.
//!
//! # Why this gate asserts TEXT and not the verdict
//!
//! The accept/refuse column above needs an assembler. `pto-to-ascendc` answers it
//! for a `.pto` input only by SHELLING OUT to `ptoas`, which is machine-local, so
//! asserting the verdict here would make this file skip wherever `ptoas` is
//! absent, and a skip that returns success is how a gate stops being one. What is
//! CI-safe is the emitted text: an arm that is still a passthrough emits
//! `pto.tmov` (or, for rope, an identity `pto.tmuls` by 1.0) and does NOT emit the
//! real op. That is asserted below, per arm.
//!
//! Version caveat, stated because the repository disagrees with this machine: all
//! of the above was measured with **ptoas 0.68**, and the repository PINS v0.70
//! (`pto-to-ascendc` prints that mismatch as a note on every run). A later
//! assembler may accept or refuse differently. The emitted-text assertions below
//! do not depend on the assembler at all.
//!
//! # This list is a ratchet toward zero, not a defence of the bug
//!
//! Each row asserts that an arm is STILL a passthrough. Fixing one therefore turns
//! this red, and the message says to delete the row rather than to restore the
//! passthrough. Eight of the nine are fixable today -- the op exists and the arm
//! does not reach for it -- and each is additionally pinned by a test in
//! `mlir_to_pto.rs` asserting `pto.contains("pto.tmov")`, which is why they still
//! emit: `.claude/rules/first-principles.md` section 3 forbids changing a test
//! expectation before raising it with the owner.

use std::path::{Path, PathBuf};
use std::process::Command;

/// One arm: a synthetic kernel that reaches it, what a passthrough looks like,
/// and the op that should be there instead.
struct Arm {
    name: &'static str,
    tile: &'static str,
    /// Text the passthrough emits today. Present == still a passthrough.
    passthrough: &'static str,
    /// The real op. Its appearance means the arm was fixed.
    real_op: &'static str,
    /// What the remedy is, for the failure message.
    remedy: &'static str,
}

fn tmp(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("passthru_{}_{name}", std::process::id()));
    p
}

fn emit(tile: &str, name: &str) -> String {
    let src = tmp(&format!("{name}.tile"));
    let dst = tmp(&format!("{name}.pto"));
    std::fs::write(&src, tile).expect("write the synthetic .tile");
    let run = Command::new(env!("CARGO_BIN_EXE_rspto"))
        .arg(&src)
        .arg("-o")
        .arg(&dst)
        .output()
        .expect("rspto runs");
    assert!(
        run.status.success(),
        "{name}: the emitter refused a kernel this gate needs in order to see the \
         arm at all. If the arm now REFUSES on purpose, that is the fix landing -- \
         delete this arm's row from ARMS and say so.\n{}",
        String::from_utf8_lossy(&run.stderr)
    );
    let text = std::fs::read_to_string(&dst).expect("the emitter wrote a .pto");
    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&dst);
    text
}

/// The emitted OPS, with every comment line dropped.
///
/// Needed because these arms write the remedy into the artefact as a comment, so
/// the slice arm's own disclaimer names `pto.textract` -- and a whole-text search
/// for the real op then reports the arm as fixed by reading the note that says it
/// is not. Measured the hard way: the first run of this gate failed on exactly
/// that, with "slice: the emitted PTO contains `pto.textract`, so this arm appears
/// to have been fixed".
fn ops_only(pto: &str) -> String {
    pto.lines()
        .map(str::trim_start)
        .filter(|l| !l.starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n")
}

const ARMS: &[Arm] = &[
    Arm {
        name: "transpose",
        tile: r#"
module {
  llvm.func @transpose_k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c8 = llvm.mlir.constant(8 : i32) : i32
    %c32 = llvm.mlir.constant(32 : i32) : i32
    %t0 = llvm.call @__tile_load_f32(%arg0, %c8, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @__tile_transpose_f32(%c0, %t0, %c8, %c32) : (i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg1, %t1, %c32, %c8) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#,
        passthrough: "pto.tmov",
        real_op: "pto.ttrans",
        remedy: "emit pto.ttrans, which this file already uses for the attention \
                 score transposes and which assembles an 8x32 -> 32x8 as TTRANS",
    },
    Arm {
        name: "slice",
        tile: r#"
module {
  llvm.func @slice_k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c8 = llvm.mlir.constant(8 : i32) : i32
    %c16 = llvm.mlir.constant(16 : i32) : i32
    %c32 = llvm.mlir.constant(32 : i32) : i32
    %t0 = llvm.call @__tile_load_f32(%arg0, %c32, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @__tile_slice_f32(%c0, %t0, %c4, %c8, %c32, %c32, %c16, %c16) : (i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg1, %t1, %c16, %c16) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#,
        passthrough: "pto.tmov",
        real_op: "pto.textract",
        remedy: "emit pto.textract at (row_off, col_off); both offsets reach the \
                 emitted TEXTRACT call and are bounds-checked",
    },
    Arm {
        name: "concat",
        tile: r#"
module {
  llvm.func @concat_k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c16 = llvm.mlir.constant(16 : i32) : i32
    %c32 = llvm.mlir.constant(32 : i32) : i32
    %t0 = llvm.call @__tile_load_f32(%arg0, %c32, %c16) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @__tile_load_f32(%arg1, %c32, %c16) : (!llvm.ptr<1>, i32, i32) -> i32
    %t2 = llvm.call @__tile_concat_f32(%c0, %t0, %t1, %c32, %c16, %c16) : (i32, i32, i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg2, %t2, %c32, %c32) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#,
        passthrough: "pto.tmov",
        real_op: "pto.tinsert",
        remedy: "emit two pto.tinsert, at column 0 and at column cols_a; today the \
                 second operand is not copied at all",
    },
    Arm {
        name: "rsqrt",
        tile: r#"
module {
  llvm.func @rsqrt_k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c64 = llvm.mlir.constant(64 : i32) : i32
    %t0 = llvm.call @__tile_load_f32(%arg0, %c1, %c64) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @__tile_rsqrt_f32(%c0, %t0, %c1, %c64) : (i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg1, %t1, %c1, %c64) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#,
        passthrough: "pto.tmov",
        real_op: "pto.trsqrt",
        remedy: "emit pto.trsqrt, which assembles and lowers to TRSQRT",
    },
    Arm {
        name: "log",
        tile: r#"
module {
  llvm.func @log_k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c64 = llvm.mlir.constant(64 : i32) : i32
    %t0 = llvm.call @__tile_load_f32(%arg0, %c1, %c64) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @__tile_log_f32(%c0, %t0, %c1, %c64) : (i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg1, %t1, %c1, %c64) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#,
        passthrough: "pto.tmov",
        real_op: "pto.tlog",
        remedy: "emit pto.tlog, which assembles and lowers to TLOG",
    },
    Arm {
        name: "cast",
        tile: r#"
module {
  llvm.func @cast_k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c32 = llvm.mlir.constant(32 : i32) : i32
    %t0 = llvm.call @__tile_load_f32(%arg0, %c1, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @__tile_cast_f32_f16(%c0, %t0, %c1, %c32) : (i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f16(%arg1, %t1, %c1, %c32) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#,
        passthrough: "pto.tmov",
        real_op: "pto.tcvt",
        remedy: "emit pto.tcvt, which this file already uses in seven places and \
                 which lowers to CAST_RINT + saturation + TCVT",
    },
    Arm {
        name: "gather",
        tile: r#"
module {
  llvm.func @gather_k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c8 = llvm.mlir.constant(8 : i32) : i32
    %c32 = llvm.mlir.constant(32 : i32) : i32
    %t0 = llvm.call @__tile_load_f32(%arg0, %c8, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @__tile_gather_f32(%c0, %t0, %arg1, %c8, %c32, %c1) : (i32, i32, !llvm.ptr<1>, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg2, %t1, %c8, %c32) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#,
        passthrough: "pto.tmov",
        real_op: "pto.tgather",
        remedy: "emit pto.tgather, which exists in the assembler and which this \
                 file already uses on the top-k path",
    },
    Arm {
        name: "scatter",
        tile: r#"
module {
  llvm.func @scatter_k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c8 = llvm.mlir.constant(8 : i32) : i32
    %c32 = llvm.mlir.constant(32 : i32) : i32
    %t0 = llvm.call @__tile_load_f32(%arg0, %c8, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @__tile_scatter_f32(%c0, %t0, %arg1, %c8, %c32, %c1) : (i32, i32, !llvm.ptr<1>, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg2, %t1, %c8, %c32) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#,
        passthrough: "pto.tmov",
        real_op: "pto.tscatter",
        remedy: "emit pto.tscatter, which the assembler knows -- a single-operand \
                 probe is refused for its operand list, not as an unknown op",
    },
];

#[test]
fn every_passthrough_arm_is_still_a_passthrough_and_says_which() {
    assert_eq!(
        ARMS.len(),
        8,
        "the arm list changed size; a row leaving means an arm was fixed (good -- \
         record it) and a row arriving means a new passthrough was added (not good)"
    );
    for arm in ARMS {
        let pto = ops_only(&emit(arm.tile, arm.name));
        assert!(
            pto.contains(arm.passthrough),
            "{}: the emitted PTO no longer contains `{}`. If this arm was FIXED, \
             delete its row from ARMS and update the table in the module note -- do \
             not restore the passthrough to make this pass.\n{pto}",
            arm.name,
            arm.passthrough
        );
        assert!(
            !pto.contains(arm.real_op),
            "{}: the emitted PTO contains `{}`, so this arm appears to have been \
             fixed. Delete its row from ARMS and update the module note's table. \
             The remedy recorded for it was: {}.",
            arm.name,
            arm.real_op,
            arm.remedy
        );
    }
}

/// The rope stub is held apart because it is the family's ONE honest member.
///
/// `pto.tsin` and `pto.tcos` really are unknown ops -- a probe of each is refused
/// with `custom op 'pto.tsin' is unknown` -- so "PTO lacks sin/cos/pow" is true
/// and the arm has nothing better to emit.
///
/// AMENDED 2026-11-09, AND THE FIXTURE IS WHAT MOVED. This test used to drive the
/// arm at `%p = constant(0)` and assert the stub comment, and its own failure
/// message said "if the arm now computes a rope, this gate should go with the
/// stub". The arm does now, at that one position: rope rotates each pair by
/// theta = pos * freq, so at pos 0 every theta is 0, cos = 1, sin = 0, and the
/// rotation matrix is I, which `src * 1.0` computes EXACTLY in IEEE-754. So pos 0
/// was the one position at which the stub comment was a FALSE label on a correct
/// artefact -- and it was also the position the corpus kernel declares, which is
/// why `deepseek_v41/rope_v41_8x64.pto` no longer carries the comment and is no
/// longer withheld by the scorer.
///
/// The assertion below is UNCHANGED. Only the position moved, to 42, where the
/// rotation is real and the arm genuinely has nothing better to emit. A second
/// case pins pos 0 to the exactness wording, so neither branch is now unpinned,
/// and after this change these two cases are the ONLY cover for the stub wording:
/// no corpus artefact carries it any more.
///
/// What this asserts is that the stub stays SELF-DECLARING. The identity is
/// `out = src * 1.0`, which is invisible in the emitted ops -- a `pto.tmuls` is a
/// perfectly ordinary scale -- so the comment is the only thing in the artefact
/// that distinguishes an identity from a rope, and a scorer reads that comment.
#[test]
fn the_rope_stub_declares_itself_in_the_artefact() {
    let tile = r#"
module {
  llvm.func @rope_k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    %r = llvm.mlir.constant(8 : i32) : i32
    %cl = llvm.mlir.constant(64 : i32) : i32
    %p = llvm.mlir.constant(42 : i32) : i32
    %x = llvm.call @__tile_load_f32(%arg0, %r, %cl) : (!llvm.ptr<1>, i32, i32) -> i32
    %o = llvm.call @__tile_rope_f32(%x, %x, %p, %r, %cl) : (i32, i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg1, %o, %r, %cl) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
    let pto = emit(tile, "rope");
    assert!(
        pto.contains("STUB:") && pto.contains("PTO lacks sin/cos/pow"),
        "the rope stub stopped declaring itself in the artefact. The identity it \
         emits is `src * 1.0`, which is an ordinary scale in the emitted ops, so \
         this comment is the ONLY thing distinguishing the artefact from a real \
         rope -- and the campaign's scorer reads it out of these bytes to withhold \
         the kernel. If the arm now computes a rope, this gate should go with the \
         stub.\n{pto}"
    );
    assert!(
        pto.contains("pto.tmuls"),
        "the rope stub no longer emits the identity scale it documents:\n{pto}"
    );
}

/// The other side of the same guard: at pos 0 the passthrough is the ANSWER.
///
/// This is the position the corpus kernel declares, and it is the only position at
/// which the stub wording would be a false label. If this regressed to emitting the
/// stub comment, `deepseek_v41/rope_v41_8x64` would be withheld from the campaign's
/// score again for arithmetic that is exact -- and if it regressed the other way, by
/// dropping the comment for EVERY position, the sibling test above fails. The pair
/// is what makes either one mean anything.
#[test]
fn the_rope_arm_declares_pos_zero_exact_rather_than_a_stub() {
    let tile = r#"
module {
  llvm.func @rope_k0(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    %r = llvm.mlir.constant(8 : i32) : i32
    %cl = llvm.mlir.constant(64 : i32) : i32
    %p = llvm.mlir.constant(0 : i32) : i32
    %x = llvm.call @__tile_load_f32(%arg0, %r, %cl) : (!llvm.ptr<1>, i32, i32) -> i32
    %o = llvm.call @__tile_rope_f32(%x, %x, %p, %r, %cl) : (i32, i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg1, %o, %r, %cl) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
    let pto = emit(tile, "rope_pos0");
    assert!(
        !pto.contains("PTO lacks sin/cos/pow"),
        "at pos 0 the rotation is the identity and `src * 1.0` computes it exactly, \
         so the stub wording is a FALSE label on a correct artefact -- and the \
         campaign's scorer reads it out of these bytes to withhold the kernel:\n{pto}"
    );
    assert!(
        pto.contains("pos=0: rotation by angle 0 is the identity"),
        "pos 0 must still SAY why its passthrough is exact. The arithmetic is an \
         ordinary `pto.tmuls`, so without the comment a reader cannot tell a \
         deliberate exact-at-zero emission from an arm that forgot to rotate:\n{pto}"
    );
    assert!(
        pto.contains("pto.tmuls") && pto.contains("pto.tstore"),
        "the guard changes the LABEL, never the arithmetic; the copy and its store \
         must both survive it:\n{pto}"
    );
}

/// Guard against this file passing because the emitter binary moved.
#[test]
fn the_emitter_under_test_exists() {
    let bin = Path::new(env!("CARGO_BIN_EXE_rspto"));
    assert!(bin.exists(), "{} is not there", bin.display());
}
