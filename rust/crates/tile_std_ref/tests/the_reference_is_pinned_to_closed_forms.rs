// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! The reference, checked against closed forms worked out by hand.
//!
//! Without this the crate would only ever be checked against itself: a lifted
//! kernel that loads, adds and stores agrees with this crate's `add` whatever
//! either of them does. So each op is pinned to a value computed independently
//! of the implementation -- by hand where the arithmetic is short, and by an
//! INVARIANT where it is not.
//!
//! Invariants are the better instrument for the reductions. `softmax` summing
//! to 1 per row, and being invariant under adding a constant to a row, are
//! properties no plausible wrong implementation satisfies by accident, while a
//! hand-computed softmax of eight values is a transcription exercise that tests
//! arithmetic rather than code.

use tile_std_ref::tile::*;

fn load<const R: usize, const C: usize>(v: &[f32]) -> Tile<R, C, f32> {
    assert_eq!(v.len(), R * C);
    unsafe { tile_load_f32::<R, C>(v.as_ptr()) }
}

#[test]
fn load_and_store_are_inverses_and_row_major() {
    let src: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let t = load::<2, 3>(&src);
    // Row-major: index 4 is (row 1, col 1).
    assert_eq!(t.at(1, 1), 4.0, "the tile is not row-major");
    let mut dst = vec![0.0f32; 6];
    unsafe { tile_store_f32::<2, 3>(dst.as_mut_ptr(), t) };
    assert_eq!(dst, src, "store did not reproduce what load read");
}

#[test]
fn elementwise_ops_are_elementwise() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![10.0, 20.0, 30.0, 40.0];
    let sum = tile_add_f32::<1, 4>(load::<1, 4>(&a), load::<1, 4>(&b));
    assert_eq!(sum.to_vec(), vec![11.0, 22.0, 33.0, 44.0]);
    let diff = tile_sub_f32::<1, 4>(load::<1, 4>(&b), load::<1, 4>(&a));
    assert_eq!(diff.to_vec(), vec![9.0, 18.0, 27.0, 36.0]);
    let prod = tile_mul_f32::<1, 4>(load::<1, 4>(&a), load::<1, 4>(&b));
    assert_eq!(prod.to_vec(), vec![10.0, 40.0, 90.0, 160.0]);
}

#[test]
fn matmul_is_m_k_n_and_not_m_n_k() {
    // 2x3 @ 3x2 -> 2x2, worked out by hand:
    //   A = [[1,2,3],[4,5,6]]   B = [[7,8],[9,10],[11,12]]
    //   C = [[1*7+2*9+3*11, 1*8+2*10+3*12],
    //        [4*7+5*9+6*11, 4*8+5*10+6*12]]
    //     = [[58, 64], [139, 154]]
    //
    // The shape is the assertion that matters: if the const generics were read
    // as <M, N, K> this would not even typecheck, which is the property the
    // real API has and an emitter bypassed by passing extents positionally.
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let c = tile_matmul_f32::<2, 3, 2>(load::<2, 3>(&a), load::<3, 2>(&b));
    assert_eq!(c.to_vec(), vec![58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn softmax_rows_sum_to_one_and_are_shift_invariant() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 0.5, 0.5, 0.5, 0.5];
    let y = tile_softmax_f32::<2, 4>(load::<2, 4>(&x));
    for r in 0..2 {
        let s: f32 = (0..4).map(|c| y.at(r, c)).sum();
        assert!((s - 1.0).abs() < 1e-6, "row {r} sums to {s}, not 1");
    }
    // A uniform row is uniform.
    for c in 0..4 {
        assert!((y.at(1, c) - 0.25).abs() < 1e-6);
    }
    // Shift invariance, which is what the max-subtraction is for.
    let shifted: Vec<f32> = x.iter().map(|v| v + 100.0).collect();
    let z = tile_softmax_f32::<2, 4>(load::<2, 4>(&shifted));
    for r in 0..2 {
        for c in 0..4 {
            assert!(
                (y.at(r, c) - z.at(r, c)).abs() < 1e-6,
                "softmax is not shift-invariant at ({r}, {c})"
            );
        }
    }
}

#[test]
fn softmax_does_not_overflow_on_a_large_logit() {
    // The reason the max shift is not optional. Without it `exp(800)` is `inf`
    // and every output becomes NaN.
    let x = vec![800.0, 0.0, 0.0, 0.0];
    let y = tile_softmax_f32::<1, 4>(load::<1, 4>(&x));
    assert!(
        y.to_vec().iter().all(|v| v.is_finite()),
        "softmax overflowed: {:?}",
        y.to_vec()
    );
    assert!((y.at(0, 0) - 1.0).abs() < 1e-6);
}

#[test]
fn rms_norm_scales_a_row_to_unit_rms() {
    // mean(x^2) for [3,4] is 12.5, sqrt is ~3.5355, so x/3.5355.
    let x = vec![3.0, 4.0];
    let y = tile_rms_norm_f32::<1, 2>(load::<1, 2>(&x), 0.0);
    let expected = [3.0 / 12.5f32.sqrt(), 4.0 / 12.5f32.sqrt()];
    for c in 0..2 {
        assert!(
            (y.at(0, c) - expected[c]).abs() < 1e-6,
            "rms_norm[{c}] = {}, expected {}",
            y.at(0, c),
            expected[c]
        );
    }
    // And the invariant: the result's RMS is 1.
    let rms = (y.to_vec().iter().map(|v| v * v).sum::<f32>() / 2.0).sqrt();
    assert!((rms - 1.0).abs() < 1e-6, "the normalised row's RMS is {rms}");
}

#[test]
fn exp_is_exp() {
    let x = vec![0.0, 1.0];
    let y = tile_exp_f32::<1, 2>(load::<1, 2>(&x));
    assert!((y.at(0, 0) - 1.0).abs() < 1e-6);
    assert!((y.at(0, 1) - std::f32::consts::E).abs() < 1e-6);
}

#[test]
fn a_tile_is_not_copy() {
    // Compile-time property, asserted as a fact about the source because a
    // negative trait bound cannot be written. The real `Tile` is not `Copy` so
    // that each tile is consumed once; a reference that allowed copying would
    // accept lifted kernels the device rejects.
    let src = include_str!("../src/lib.rs");
    assert!(
        !src.contains("derive(Debug, Copy")
            && !src.contains("derive(Copy")
            && !src.contains("impl<const ROWS: usize, const COLS: usize, T> Copy"),
        "Tile has gained Copy, which makes this reference MORE permissive than the \
         device API it stands in for"
    );
    assert!(
        src.contains("NOT `Copy`"),
        "the reason Tile is not Copy is no longer written down"
    );
}
