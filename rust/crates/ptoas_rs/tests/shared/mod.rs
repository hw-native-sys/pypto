// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Fixtures and corpus discovery shared by this crate's integration tests.
//!
//! Extracted from `roundtrip.rs` when a second test needed the same per-op
//! case list. Copying it would have been the cheaper edit and the wrong one:
//! the list's whole job is to fail when `MODELLED_OPS` grows without a case,
//! and a second copy would have let one of them go stale unnoticed.
//!
//! `#![allow(dead_code)]` because each test binary links the module whole and
//! uses only part of it.

#![allow(dead_code)]

/// The root to walk for `.pto` artifacts: the REPOSITORY root, not `rust/`.
///
/// This distinction decides the denominator of every corpus sweep, and it was
/// got wrong in four of the five call sites. The bench corpus -- the artifacts
/// the tools are actually meant to handle -- lives in `bench/` at the
/// repository root, OUTSIDE the cargo workspace. A walk rooted at `rust/`
/// therefore sees 27 distinct artifacts, almost all of them this crate's own
/// fixtures, while the real corpus is 60. A sweep whose denominator is its own
/// fixtures measures nothing, and a size assertion calibrated against it fails
/// for a reason that has nothing to do with what it was testing.
///
/// Centralised rather than copied: `roundtrip.rs` had worked this out and
/// carried the comment explaining it, and the four sites that had not were
/// each one edit away from the same mistake.
pub fn corpus_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .expect("the repository root must be reachable")
}

pub fn collect_pto_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        // Skip build artifacts and heavy vendored trees.
        if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
            // `.claude` holds other lanes' git worktrees -- 126 of the 173
            // .pto files this walk reached were COPIES living there, so the
            // corpus was mostly not this revision's artifacts and its size
            // depended on which worktrees happened to exist on the machine.
            if name == "target"
                || name == ".git"
                || name == ".claude"
                || name == "node_modules"
                || name == ".docker"
                || name == ".slicer-cache"
            {
                continue;
            }
        }
        if path.is_dir() {
            collect_pto_files(&path, out);
        } else if path.extension().and_then(|s| s.to_str()) == Some("pto") {
            out.push(path);
        }
    }
}

    // 8 round-tripping artifacts uses `tmov`. This count is the only
pub enum Case {
    /// Exercised by the preamble every case shares (views, allocs, load/store,
    /// and the constants that feed them).
    InPreamble,
    Line(String),
    Missing,
}

pub fn op_case(op: &str) -> Case {
    const T32: &str = "!pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32>";
    const T1: &str = "!pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=1>";
    const TA: &str = "!pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=64>";
    const TB: &str = "!pto.tile_buf<loc=vec, dtype=f16, rows=64, cols=32>";
    const TC: &str = "!pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32>";
    const TIX: &str = "!pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=32>";
    const TH: &str = "!pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=32>";

    match op {
        "arith.constant"
        | "pto.make_tensor_view"
        | "pto.partition_view"
        | "pto.alloc_tile"
        | "pto.tload"
        | "pto.tstore" => Case::InPreamble,

        "pto.tadd" | "pto.tsub" | "pto.tmul" | "pto.tdiv" | "pto.tmax" | "pto.tmin" => {
            Case::Line(format!("{op} ins(%t0, %t1 : {T32}, {T32}) outs(%t2 : {T32})"))
        }

        // The scalar's operand type is `f32`, not a tile -- the distinction the
        // prefix dispatch erased for 277 op instances.
        // `pto.tsubs` joined 2026-11-09: its `ScalarBinKind` variant and its
        // `WRAPPERS` stem both existed from the day `Adds` was added, but the
        // parser's dispatch table never gained the row, so the lift refused it as
        // "not modelled" while the model was already there. Same op SHAPE as the
        // four beside it, so it belongs in this arm rather than its own.
        "pto.tmuls" | "pto.tadds" | "pto.tsubs" | "pto.tmaxs" | "pto.tmins" => {
            Case::Line(format!("{op} ins(%t0, %sc : {T32}, f32) outs(%t2 : {T32})"))
        }

        // The shift/mask family is the same op SHAPE at a different width: an
        // i32 tile and an i32 scalar, which is why `scalar_ty` has to be
        // carried rather than assumed to be f32.
        "pto.tshls" | "pto.tshrs" | "pto.tands" => {
            Case::Line(format!("{op} ins(%ix, %si : {TIX}, i32) outs(%ix2 : {TIX})"))
        }

        // src and dst dtypes DIFFER, so both ends come from the shapes map.
        "pto.tcvt" => Case::Line(format!(
            "pto.tcvt ins(%t0 {{sat_mode = #pto<saturation_mode ON>}} : {T32}) \
             outs(%h0 : {TH})"
        )),

        // `pto.trsqrt` joined 2026-11-09. Same op SHAPE as the three beside it
        // -- one tile in, one tile out -- which is why it belongs in this arm.
        // It is the FUSED rsqrt spelling, and unlike `pto.tsqrt`/`pto.trecip` it
        // lifts, because `tile_std::tile_rsqrt_f32` is a safe wrapper taking and
        // returning `Tile<R, C, f32>` while `__tile_sqrt_f32` is a bare
        // intrinsic and nothing in the tile API reciprocates.
        "pto.texp" | "pto.tneg" | "pto.tabs" | "pto.tmov" | "pto.trsqrt" => {
            Case::Line(format!("{op} ins(%t0 : {T32}) outs(%t1 : {T32})"))
        }

        // Second `ins` operand is the scratch tile the parser discards.
        "pto.trowmax" | "pto.trowsum" => {
            Case::Line(format!("{op} ins(%t0, %t1 : {T32}, {T32}) outs(%r : {T1})"))
        }

        "pto.trowexpandsub" | "pto.trowexpanddiv" | "pto.trowexpandmul" => {
            Case::Line(format!("{op} ins(%t0, %r : {T32}, {T1}) outs(%t2 : {T32})"))
        }

        // One operand and TWO shapes: 1x1 in, 1x32 out. The shapes must differ
        // or the case would not exercise the thing that makes this its own
        // variant rather than a `UnOp` kind.
        "pto.trowexpand" => Case::Line(format!("pto.trowexpand ins(%r : {T1}) outs(%t2 : {T32})")),

        // The column-direction pair, added 2026-11-10 with the 1-query sink
        // attention PTO. `tcolexpand` is 1x32 -> 32x32 and `tcolsum` is the
        // reverse, so each end's shape has to come from the shapes map.
        "pto.tcolexpand" => {
            Case::Line(format!("pto.tcolexpand ins(%t0 : {T32}) outs(%mc : {TC})"))
        }
        "pto.tcolsum" => {
            Case::Line(format!("pto.tcolsum ins(%mc : {TC}) outs(%t2 : {T32})"))
        }

        "pto.tsqrt" | "pto.trecip" => {
            Case::Line(format!("{op} ins(%t0 : {T32}) outs(%t1 : {T32})"))
        }

        // No operands, no result: the pipe is the whole op.
        "pto.barrier" => Case::Line("pto.barrier <PIPE_ALL>".to_string()),

        // Two regions, and the `} else {` line that both closes one and opens
        // the other -- which is the part a line-based reader gets wrong: it
        // reads that line as the end of the ENCLOSING body and silently drops
        // everything after the conditional. Each branch holds a real op so both
        // recursions are exercised.
        "scf.if" => Case::Line(format!(
            "scf.if %cnd {{\n      \
             pto.texp ins(%t0 : {T32}) outs(%t1 : {T32})\n                 }} else {{\n      \
             pto.tneg ins(%t0 : {T32}) outs(%t1 : {T32})\n    }}"
        )),

        // The one op with a REGION. The body holds a real op so the recursion
        // and the re-indentation are both exercised.
        "scf.for" => Case::Line(format!(
            "scf.for %r = %c0 to %c32 step %c1 {{\n      \
             pto.texp ins(%t0 : {T32}) outs(%t1 : {T32})\n    }}"
        )),

        // MLIR generic form with an empty operand list and a scalar result.
        "pto.get_block_idx" | "pto.get_block_num" => {
            Case::Line(format!("%bq = \"{op}\"() : () -> i64"))
        }
        // Carries its own `pto.get_block_idx` rather than leaning on one in
        // the preamble. A block query in the preamble is in EVERY case's
        // module, and the Rust lift refuses it (`tile_std::get_block_idx`
        // sits at the crate root, outside the `tile_std::tile` glob the
        // emitter imports) -- so every per-op module was refused on a line
        // that had nothing to do with the op under test, and a gate that
        // lifts these cases measured zero while reporting nothing wrong.
        "arith.index_cast" => Case::Line(
            "%bq0 = \"pto.get_block_idx\"() : () -> i64\n    \
             %bqi = arith.index_cast %bq0 : i64 to index"
                .to_string(),
        ),

        // MLIR generic form; the result type is where the shape comes from.
        "pto.subview" => Case::Line(format!(
            "%sv = \"pto.subview\"(%t0, %c0, %c0) {{operandSegmentSizes = \
             array<i32: 1, 2, 0, 0>, sizes = [1, 16]}} : ({T32}, index, index) -> \
             !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16>"
        )),

        // The index operand is an i32 tile, not an f32 one.
        "pto.tgatherb" => Case::Line(format!(
            "pto.tgatherb ins(%t0, %ix : {T32}, {TIX}) outs(%t2 : {T32})"
        )),

        // Two ins operands: the source and the workspace tile PTO threads
        // through. Shapes are the transpose of each other, which the LIFT
        // checks; the round trip only has to put the two-operand form back.
        "pto.ttrans" => Case::Line(format!(
            "pto.ttrans ins(%mc, %mc2 : {TC}, {TC}) outs(%mc3 : {TC})"
        )),

        "pto.tmatmul" => Case::Line(format!(
            "pto.tmatmul ins(%ma, %mb : {TA}, {TB}) outs(%mc : {TC})"
        )),

        _ => Case::Missing,
    }
}

/// The shared preamble: pointers in and out, the views and tiles every case
/// refers to, and a typed scalar constant for the tile-by-scalar forms.
pub fn module_with(op_line: &str) -> String {
    format!(
        r#"module {{
  func.func @k(%a: !pto.ptr<f32>, %b: !pto.ptr<f32>) {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %sc = arith.constant 2.5 : f32
    %v0 = pto.make_tensor_view %a, shape = [%c1, %c32], strides = [%c32, %c1] : !pto.tensor_view<?x?xf32>
    %p0 = pto.partition_view %v0, offsets = [%c0, %c0], sizes = [%c1, %c32] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<1x32xf32>
    %t0 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32>
    %t1 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32>
    %t2 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32>
    %r = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=1>
    %ix = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=32>
    %ix2 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=32>
    %h0 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=32>
    %si = arith.constant 3 : i32
    %cnd = arith.constant 1 : i1
    %ma = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=64>
    %mb = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=64, cols=32>
    %mc = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32>
    %mc2 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32>
    %mc3 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32>
    pto.tload ins(%p0 : !pto.partition_tensor_view<1x32xf32>) outs(%t0 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32>)
    {op_line}
    %v1 = pto.make_tensor_view %b, shape = [%c1, %c32], strides = [%c32, %c1] : !pto.tensor_view<?x?xf32>
    %p1 = pto.partition_view %v1, offsets = [%c0, %c0], sizes = [%c1, %c32] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<1x32xf32>
    pto.tstore ins(%t2 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32>) outs(%p1 : !pto.partition_tensor_view<1x32xf32>)
    return
  }}
}}
"#
    )
}

// --- toolchain probes, shared because they were drifting ------------------
//
// Four test files need to know whether the kernel-side toolchain and Miri are
// installed, and the first three grew their own copy. One of those copies read
// the hardcoded toolchain while the code under test honoured the
// `PTOAS_RS_KERNEL_TOOLCHAIN` override, so the probe reported on a DIFFERENT
// toolchain than the one that would run -- and the test failed inside the
// runner it had just certified as available.
//
// They matter more than a test helper usually does. With no nightly installed,
// the gates in this crate FAILED rather than skipping: measured by stripping
// `ptoas` from PATH and pointing the override at an absent toolchain, which is
// what a GitHub-hosted runner looks like. The `rust` CI job would have gone red
// on a tree where nothing was wrong.

/// The toolchain the kernel-side DSL needs: nightly, permanently, because it is
/// `#![no_core]`. One spelling, overridable for a bisect.
pub const KERNEL_TOOLCHAIN: &str = "nightly-2025-08-04";

pub fn kernel_toolchain() -> String {
    std::env::var("PTOAS_RS_KERNEL_TOOLCHAIN")
        .unwrap_or_else(|_| KERNEL_TOOLCHAIN.to_string())
}

/// Can that toolchain compile anything here?
pub fn have_kernel_toolchain() -> bool {
    std::process::Command::new("rustup")
        .args(["run", &kernel_toolchain(), "rustc", "--version"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Is Miri installed ON THAT TOOLCHAIN -- not on some other one.
pub fn have_miri() -> bool {
    std::process::Command::new("cargo")
        .args(["miri", "--version"])
        .env("RUSTUP_TOOLCHAIN", kernel_toolchain())
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Is the PTO assembler reachable?
pub fn have_ptoas() -> bool {
    std::process::Command::new("sh")
        .arg("-c")
        .arg("command -v ptoas")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Say, loudly, that a gate did not run, and why.
///
/// Returns true so a caller can write `if skip_unless(c, name) { return; }`.
/// NOT silent: a skip that returns `Ok` is how a suite comes to certify a run
/// that never happened, and a CI log is the only place anyone would notice.
pub fn skip_unless(condition: bool, name: &str, need: &str) -> bool {
    if condition {
        return false;
    }
    eprintln!(
        "SKIPPED {name}: {need} is not available (toolchain `{}`), so this gate ran \
         NOTHING. It did not pass -- it did not execute.",
        kernel_toolchain()
    );
    true
}
