// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Hyper-connection: the Sinkhorn-normalised mixing matrix.
//!
//! DS4-Flash carries `hyper_connection.count = 4` parallel hidden streams. Each sublayer reads a
//! `[24]` mix vector (`4 + 4 + 4*4`) whose last 16 entries are a `4x4` combination matrix, and
//! that matrix is Sinkhorn-normalised before use.
//!
//! The spec is transcribed from `engine_core`'s CPU dispatcher, and it has an asymmetry that the
//! phrase "20 Sinkhorn iterations" hides:
//!
//!   * iteration 1 does a numerically-stable row **softmax**, then `+eps`, then a column scale;
//!   * iterations 2..N do a row **sum-normalisation**, then a column scale.
//!
//! So it is not twenty identical passes, and implementing it as twenty identical passes yields a
//! plausible matrix that is wrong. The tests pin that asymmetry directly rather than trusting
//! the transcription.

/// The `eps` the reference uses. Small but load-bearing: added after the row softmax and inside
/// every column divisor, it is what keeps a degenerate column from dividing by zero.
pub const HC_EPS: f32 = 1e-8;

/// Sinkhorn-normalise an `n x n` combination matrix in place.
///
/// `iters` is `hyper_connection.sinkhorn_iter` (20 for DS4-Flash). `iters == 0` still runs the
/// first pass, because that pass is what turns logits into a distribution at all — skipping it
/// would leave raw logits, which no caller wants.
pub fn sinkhorn_comb(comb: &mut [f32], n: usize, iters: u32, eps: f32) {
    assert_eq!(comb.len(), n * n, "comb must be n*n");
    if n == 0 {
        return;
    }
    // Pass 1: stable row softmax, then eps.
    for r in 0..n {
        let row = &mut comb[r * n..(r + 1) * n];
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - m).exp();
            sum += *v;
        }
        let inv = 1.0 / sum;
        for v in row.iter_mut() {
            *v = *v * inv + eps;
        }
    }
    col_scale(comb, n, eps);
    // Passes 2..iters: row SUM-normalise, then column scale.
    for _ in 1..iters.max(1) {
        for r in 0..n {
            let row = &mut comb[r * n..(r + 1) * n];
            let s: f32 = row.iter().sum();
            let inv = 1.0 / (s + eps);
            for v in row.iter_mut() {
                *v *= inv;
            }
        }
        col_scale(comb, n, eps);
    }
}

fn col_scale(comb: &mut [f32], n: usize, eps: f32) {
    let mut col = vec![0.0f32; n];
    for r in 0..n {
        for c in 0..n {
            col[c] += comb[r * n + c];
        }
    }
    for c in 0..n {
        let inv = 1.0 / (col[c] + eps);
        for r in 0..n {
            comb[r * n + c] *= inv;
        }
    }
}

/// Split a `[2n + n*n]` mix vector into its three sections.
///
/// `pre[n]` weights the previous streams into this sublayer's input, `post[n]` weights the
/// sublayer's output back out, and `comb[n*n]` mixes the streams. Returned as slices of the
/// caller's buffer so the layout stays visible at the call site.
pub fn split_mix(mix: &[f32], n: usize) -> (&[f32], &[f32], &[f32]) {
    assert_eq!(mix.len(), 2 * n + n * n, "mix must be 2n + n*n");
    let (pre, rest) = mix.split_at(n);
    let (post, comb) = rest.split_at(n);
    (pre, post, comb)
}

/// `cur = sum_h pre[h] * prev_hc[h]` — collapse the streams into this sublayer's input.
pub fn collapse_streams(prev_hc: &[f32], pre: &[f32], n: usize, d: usize) -> Vec<f32> {
    assert_eq!(prev_hc.len(), n * d);
    assert_eq!(pre.len(), n);
    let mut cur = vec![0.0f32; d];
    for h in 0..n {
        let w = pre[h];
        for j in 0..d {
            cur[j] += w * prev_hc[h * d + j];
        }
    }
    cur
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: usize = 4; // hyper_connection.count for DS4-Flash

    fn logits(seed: u32) -> Vec<f32> {
        let mut s = seed | 1;
        (0..N * N)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                ((s & 0xFFFF) as f32 / 65536.0 - 0.5) * 6.0
            })
            .collect()
    }

    /// One pass, computed by hand: stable row softmax, `+eps`, then column scale. This is the
    /// independent check on the FIRST-pass asymmetry — nothing here reuses `sinkhorn_comb`.
    #[test]
    fn first_pass_is_a_row_softmax_not_a_sum_normalise() {
        let l = logits(1);
        let mut got = l.clone();
        sinkhorn_comb(&mut got, N, 1, HC_EPS);

        let mut want = l.clone();
        for r in 0..N {
            let row = &mut want[r * N..(r + 1) * N];
            let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = row.iter().map(|v| (v - m).exp()).sum();
            for v in row.iter_mut() {
                *v = (*v - m).exp() / sum + HC_EPS;
            }
        }
        let mut col = [0.0f32; N];
        for r in 0..N {
            for c in 0..N {
                col[c] += want[r * N + c];
            }
        }
        for r in 0..N {
            for c in 0..N {
                want[r * N + c] /= col[c] + HC_EPS;
            }
        }
        for i in 0..N * N {
            assert!(
                (got[i] - want[i]).abs() <= 1e-6,
                "elem {i}: {} vs {}",
                got[i],
                want[i]
            );
        }

        // And it must NOT equal the sum-normalise-first variant, or the asymmetry is untested.
        let mut sumfirst = l.clone();
        for r in 0..N {
            let row = &mut sumfirst[r * N..(r + 1) * N];
            let s: f32 = row.iter().sum();
            let inv = 1.0 / (s + HC_EPS);
            for v in row.iter_mut() {
                *v *= inv;
            }
        }
        col_scale(&mut sumfirst, N, HC_EPS);
        let diff: f32 = (0..N * N)
            .map(|i| (got[i] - sumfirst[i]).abs())
            .fold(0.0, f32::max);
        assert!(
            diff > 1e-3,
            "softmax-first and sum-normalise-first agree to {diff:e} — the asymmetry is untested"
        );
    }

    /// The iteration count is load-bearing: 1 and 20 must differ.
    #[test]
    fn iteration_count_changes_the_result() {
        let l = logits(2);
        let mut a = l.clone();
        let mut b = l.clone();
        sinkhorn_comb(&mut a, N, 1, HC_EPS);
        sinkhorn_comb(&mut b, N, 20, HC_EPS);
        let diff: f32 = (0..N * N).map(|i| (a[i] - b[i]).abs()).fold(0.0, f32::max);
        assert!(diff > 1e-4, "1 vs 20 iterations differ by only {diff:e}");
    }

    /// The LAST operation is a column scale, so columns sum to ~1. Rows do not, and asserting
    /// they do would be wrong — which is why the row property is deliberately not claimed.
    #[test]
    fn columns_are_normalised_after_the_iterations() {
        let l = logits(3);
        let mut m = l.clone();
        sinkhorn_comb(&mut m, N, 20, HC_EPS);
        for c in 0..N {
            let s: f32 = (0..N).map(|r| m[r * N + c]).sum();
            assert!((s - 1.0).abs() < 1e-4, "column {c} sums to {s}");
        }
        assert!(
            m.iter().all(|v| v.is_finite() && *v >= 0.0),
            "must stay a nonnegative matrix"
        );
    }

    /// `eps` must keep a degenerate input finite, and symmetry must give 1/N.
    #[test]
    fn degenerate_input_stays_finite() {
        let mut m = vec![0.0f32; N * N];
        sinkhorn_comb(&mut m, N, 20, HC_EPS);
        assert!(m.iter().all(|v| v.is_finite()), "all-zero logits gave {m:?}");
        for v in &m {
            assert!((v - 1.0 / N as f32).abs() < 1e-4, "expected 1/N, got {v}");
        }
        let mut big = vec![-1e30f32; N * N];
        sinkhorn_comb(&mut big, N, 20, HC_EPS);
        assert!(big.iter().all(|v| v.is_finite()), "very negative logits gave {big:?}");
    }

    /// `iters == 0` must still produce a distribution rather than leaving raw logits.
    #[test]
    fn zero_iterations_still_runs_the_first_pass() {
        let l = logits(4);
        let mut z = l.clone();
        let mut one = l.clone();
        sinkhorn_comb(&mut z, N, 0, HC_EPS);
        sinkhorn_comb(&mut one, N, 1, HC_EPS);
        assert_eq!(z, one);
        assert_ne!(z, l, "raw logits must not survive");
    }

    /// The `[2n + n*n]` split, and that `comb` starts at `2n` rather than `n`.
    #[test]
    fn mix_splits_into_pre_post_comb() {
        let mix: Vec<f32> = (0..2 * N + N * N).map(|i| i as f32).collect();
        let (pre, post, comb) = split_mix(&mix, N);
        assert_eq!(pre, &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(post, &[4.0, 5.0, 6.0, 7.0]);
        assert_eq!(comb.len(), N * N);
        assert_eq!(comb[0], 8.0, "comb starts at 2n, not at n");
    }

    /// Collapsing weights STREAMS, not embedding dimensions — an axis mix-up that would still
    /// produce a `d`-long vector and so would not fail on shape.
    #[test]
    fn collapse_weights_streams_not_dimensions() {
        let d = 3usize;
        let prev: Vec<f32> = (0..N).flat_map(|h| vec![(h + 1) as f32; d]).collect();
        let pre = [0.0f32, 0.0, 1.0, 0.0];
        let cur = collapse_streams(&prev, &pre, N, d);
        assert_eq!(cur, vec![3.0, 3.0, 3.0], "should select stream index 2");
        let uni = [0.25f32; N];
        let avg = collapse_streams(&prev, &uni, N, d);
        assert_eq!(avg, vec![2.5, 2.5, 2.5]);
    }
}
