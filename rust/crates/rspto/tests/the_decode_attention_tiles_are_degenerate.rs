// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! NONE of the three `attention_decode_*` kernels loads Q, K and V at the same
//! extent any more. `deepseek_v41` moved to the sink form (ORACLE 53,
//! 5.976e-08), `glm_5_3` to the blocked sink-free form at its sourced
//! index_topk=2048 (ORACLE 56, 1.277e-06), and `kimi_k3` to the RUNTIME-S form
//! (ORACLE 57, 1.014e-07) -- all on 2026-11-10. The zero-count assertion below
//! now pins that none of them comes back to the degenerate name.
//!
//! The original note follows, because it is the reason the remaining two are
//! still refused rather than "fixed" with an arm that would return V.
//!
//! Both emitters refuse these three with "no arm for `__tile_attention_decode_f32`",
//! and that refusal is true about the intrinsic NAME and misleading about
//! everything else. Measured 2026-11-08, reading the `.tile` sources rather than
//! the refusals:
//!
//!   * Every one of the three loads Q, K and V as `1 x cl` -- the same shape, with
//!     `cl` the head dim (512 for V4.1, 256 for GLM 5.3, 128 for K3). Attention
//!     needs `Q[1 x D]`, `K[S x D]`, `V[S x D]`. With `S = 1` the score matrix is
//!     a single element, `softmax` of one element is exactly 1, and the kernel
//!     computes **V**. Giving the emitter an arm for this name would bank three
//!     kernels that are not attention -- the same shape of defect as a
//!     `TILE_SUBSTITUTION`, arriving through the source instead of the emitter.
//!
//!   * `mlir_to_pto` ALREADY HAS the arm these kernels want, under
//!     `__tile_attention_f32`: "a3-native vector-only DECODE attention (1-query):
//!     out[D] = softmax(Q.K^T.scale).V, Q[1xD], K[SxD], V[SxD]", validated on a
//!     910c at max_rel_err ~1e-6 against a CPU oracle. So the capability is
//!     present, device-measured, and unreachable -- blocked by an intrinsic name
//!     and a shape.
//!
//!   * The shape is missing because `S` is not in the registry. `registry.json`
//!     sources `D`, `HEAD_DIM`, `VOCAB`, `T`, `ROPE`, `TOPK` and the MoE widths,
//!     and no context or KV length at all. The generators' own doctrine forbids
//!     inventing one -- "an invented middle configuration measures nothing and
//!     looks exactly like a real one afterwards" -- so emitting `S = 1` is what
//!     refusing to invent looks like when the refusal is not stated.
//!
//! # What this gate is for
//!
//! It pins the degeneracy so that the cheap-looking fix cannot land quietly. Three
//! of the fourteen kernels with no campaign record are these, which makes "add an
//! arm for `__tile_attention_decode_f32`" the largest single-mechanism group in the
//! remaining work and the most tempting thing in the backlog. Anyone who takes it
//! will find this test red, with the reason.
//!
//! The honest sequence is the other way round: pin `S` in the registry, have the
//! generators emit `__tile_attention_f32` at `K[S x D]` / `V[S x D]`, and the arm
//! that already exists takes them. That is an OWNER decision -- a shape set is a
//! pinned benchmark input -- which is why this file records the finding instead of
//! changing the generators.
//!
//! Deliberately NOT done here: making the generators refuse. The campaign reports
//! `T = tiled / K` as a secondary target at 1.0, so withdrawing three `.tile`
//! sources would take a reported indicator below its target. Whether three kernels
//! should have no tile source at all is the same owner decision as the shape.

use std::path::{Path, PathBuf};

fn bench() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../bench")
        .canonicalize()
        .expect("the bench must be reachable")
}

/// `(rows, cols)` for each `__tile_load_f32` in order, resolving the constants.
fn loads(text: &str) -> Vec<(String, String)> {
    let mut consts = std::collections::HashMap::new();
    for line in text.lines() {
        // `%r = llvm.mlir.constant(1 : i32) : i32`
        let Some((lhs, rhs)) = line.split_once(" = llvm.mlir.constant(") else {
            continue;
        };
        let Some(v) = rhs.split(' ').next() else { continue };
        consts.insert(lhs.trim().to_string(), v.to_string());
    }
    let mut out = Vec::new();
    for line in text.lines() {
        let Some(args) = line.split_once("@__tile_load_f32(") else {
            continue;
        };
        let args = args.1.split(')').next().unwrap_or("");
        let parts: Vec<&str> = args.split(',').map(str::trim).collect();
        if parts.len() < 3 {
            continue;
        }
        let r = consts.get(parts[1]).cloned().unwrap_or_else(|| parts[1].to_string());
        let c = consts.get(parts[2]).cloned().unwrap_or_else(|| parts[2].to_string());
        out.push((r, c));
    }
    out
}

#[test]
fn every_decode_attention_tile_loads_q_k_and_v_at_one_extent() {
    let mut seen = 0usize;
    for model in ["deepseek_v41", "glm_5_3", "kimi_k3", "engine"] {
        let dir = bench().join(model).join("tile");
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        let mut paths: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
        paths.sort();
        for path in paths {
            if path.extension().and_then(|e| e.to_str()) != Some("tile") {
                continue;
            }
            let text = std::fs::read_to_string(&path).unwrap();
            if !text.contains("__tile_attention_decode_f32") {
                continue;
            }
            seen += 1;
            let l = loads(&text);
            assert_eq!(
                l.len(),
                3,
                "{}: expected three loads (Q, K, V), found {}: {l:?}",
                path.display(),
                l.len()
            );
            assert!(
                l[0] == l[1] && l[1] == l[2],
                "{}: Q, K and V no longer load at the same extent -- Q {:?}, K {:?}, \
                 V {:?}. If this kernel now declares a real sequence length, the \
                 degeneracy this file documents is GONE: delete this test, point the \
                 generator at `__tile_attention_f32` (whose arm already handles \
                 Q[1xD], K[SxD], V[SxD] and is device-validated at ~1e-6), and record \
                 the sourced S in the registry.",
                path.display(),
                l[0],
                l[1],
                l[2]
            );
            // The one that matters: S == 1, so softmax runs over a single score.
            assert_eq!(
                l[1].0, "1",
                "{}: K loads {} rows. This gate asserts the DEGENERATE state (S = 1); \
                 a K with more than one row means the kernel became real attention \
                 and this file is stale.",
                path.display(),
                l[1].0
            );
        }
    }
    assert_eq!(
        seen, 0,
        "expected no degenerate decode-attention tile left: deepseek_v41 moved to \
         the sink form, glm_5_3 to the blocked sourced-S form, and kimi_k3 to the \
         runtime-S form, all on 2026-11-10; found {seen}. A non-zero count means \
         one came back to the degenerate name, which is what this file pins."
    );
}

/// The deepseek_v41 tile is the FIXED one, and this pins the contract that
/// replaced the degenerate stub.
#[test]
fn deepseek_v41_now_declares_real_sink_attention() {
    let path = bench()
        .join("deepseek_v41")
        .join("tile")
        .join("attention_decode_v41_1x512.tile");
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(
        text.contains("__tile_attention_sink_f32"),
        "{}: no longer emits the sink intrinsic; the device record at \
         iteration 117 rests on this name",
        path.display()
    );
    let l = loads(&text);
    assert_eq!(l.len(), 3, "expected Q, K, V loads, found {l:?}");
    assert_eq!(l[1].0, "8", "K must load S=8 rows, found {:?}", l[1]);
    assert_eq!(l[2].0, "8", "V must load S=8 rows, found {:?}", l[2]);
    assert!(
        text.contains("__tile_store_f32(%arg3, %o, %cl, %r)"),
        "{}: the blocked lowering returns a [D x 1] tile; the store must be \
         (D, 1), not (1, D)",
        path.display()
    );
}

/// The capability these kernels need is present, under another name.
///
/// Asserted against the emitter source rather than described, because the whole
/// point of the finding is that the refusal names a missing arm while the arm
/// exists. If `__tile_attention_f32`'s decode path is ever removed, the remedy
/// recorded above stops being the remedy and this says so.
#[test]
fn the_arm_these_kernels_want_already_exists_under_another_name() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../rspto_emit/src/mlir_to_pto.rs")
        .canonicalize()
        .expect("the PTO emitter source must be reachable");
    let text = std::fs::read_to_string(&src).unwrap();
    assert!(
        text.contains("\"__tile_attention_f32\""),
        "mlir_to_pto no longer lists __tile_attention_f32. The decode-attention \
         finding in this file rests on that arm existing; re-price it."
    );
    assert!(
        text.contains("DECODE attention (1-query)"),
        "the a3-native 1-query decode-attention arm is no longer documented in \
         mlir_to_pto. Three corpus kernels are refused on the grounds that an arm is \
         missing while this one is present; if it really went away, that refusal \
         becomes accurate and this file should go with it."
    );
}
