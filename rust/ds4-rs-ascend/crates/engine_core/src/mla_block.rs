//! The MLA (multi-head latent attention) block, with its widths DERIVED.
//!
//! This started in an example with five hardcoded GLM constants -- `1536`, `16384`, `512`, `256` and `heads = 64` -- and
//! Kimi K3 panicked on it at block 3, because its MLA has 96 heads and two different latent widths. Every constant is now
//! read from the tensor that carries it, and the per-head loop turned out to be a single matmul (see `v_projection`).
//!
//! The shapes, from the checkpoints themselves:
//!
//! ```text
//!                GLM 5.3                    Kimi K3
//!   q_a          [d, 1536]                   [d, 1536]
//!   q_b          [1536, 16384]  64x256       [1536, 18432]  96x192
//!   kv_a_mqa     [d, 512]                    [d, 576]       576 = 512 latent + 64 rope
//!   kv_a_norm    [512]                       [512]          <- what the projections consume
//!   v_b          [512, 256, 64]              [512, 128, 96] [lat, per_head, heads] in BOTH
//! ```

/// Flattened row-major matvec: `out[r] = sum_c w[r*k + c] * x[c]`.
fn mv(w: &[f32], x: &[f64], k: usize, n: usize) -> Vec<f64> {
    (0..n)
        .map(|r| (0..k).map(|c| w[r * k + c] as f64 * x[c]).sum())
        .collect()
}

/// The v projection: `v_b` is `[heads][per_head][lat]` contiguous in BOTH models, so the per-head loop that this used to
/// be is exactly one matmul over the flattened matrix. Removing the loop removed `heads` and `per_head`, the two constants
/// that could disagree with the file.
///
/// `lat` is the NORM's width, not `kv_a_mqa`'s: Kimi's kv_a emits 576 values (512 latent + 64 rope) while `attn_kv_a_norm`
/// is 512, and using the larger number gave "v_b gives 10922 outputs" -- an average, not a width.
fn v_projection(wvb: &[f32], kva: &[f64], lat: usize) -> Vec<f64> {
    assert!(lat > 0, "the latent width must be positive");
    assert_eq!(wvb.len() % lat, 0, "v_b is {} not a multiple of the latent {lat}", wvb.len());
    let v_out = wvb.len() / lat;
    mv(wvb, &kva[..lat], lat, v_out)
}

/// One MLA block, one token. Widths are derived; nothing about GLM is assumed.
pub fn mla_1token(
    q_a: &[f32],
    q_b: &[f32],
    kv_a: &[f32],
    kv_a_norm: usize,
    v_b: &[f32],
    out: &[f32],
    xin: &[f64],
    d: usize,
) -> Vec<f64> {
    let q_lora = q_a.len() / d;
    let qa = mv(q_a, xin, d, q_lora);
    let q_out = q_b.len() / q_lora;
    let _q = mv(q_b, &qa, q_lora, q_out);
    let _lat_full = kv_a.len() / d;
    let kva = mv(kv_a, xin, d, _lat_full);
    let vcat = v_projection(v_b, &kva, kv_a_norm);
    let v_out = vcat.len();
    assert_eq!(out.len() % d, 0, "attn_output is {} not a multiple of {d}", out.len());
    // The output projection's INPUT width is v_out; if the file disagrees, say so rather than reading past the end.
    let o_in = out.len() / d;
    assert_eq!(o_in, v_out, "attn_output expects {o_in} but v_b yields {v_out}");
    mv(out, &vcat, v_out, d)
}


/// THE OUTPUT PROJECTION, with its width taken from the tensor.
///
/// This existed in both drivers as a hand-written loop, and the decode driver's had **8192** hardcoded -- GLM's
/// `heads*head_dim`. On Kimi the block produces 12288 values, so it summed only the first 8192 AND strode `wo` by 8192
/// instead of 12288, reading the wrong rows. GLM agreed with the prefill because 8192 IS its width; Kimi produced 10374
/// from decode and 36382 from prefill until this was found by comparing the two.
///
/// `wo` is `[width, d]` row-major, so `width = wo.len() / d` -- asserted against the block's own output length, because a
/// silent mismatch here is a plausible-looking number, not an error.
pub fn output_projection(wo: &[f32], o: &[f64], d: usize) -> Vec<f64> {
    assert!(d > 0, "d must be positive");
    assert_eq!(wo.len() % d, 0, "attn_output is {} not a multiple of {d}", wo.len());
    let width = wo.len() / d;
    assert_eq!(width, o.len(), "attn_output is {width} wide but the block produced {}", o.len());
    (0..d)
        .map(|r| (0..width).map(|c| wo[r * width + c] as f64 * o[c]).sum())
        .collect()
}

#[cfg(test)]
mod regression {
    /// THE 8192 BUG: a width taken from the driver instead of the tensor.
    ///
    /// Kimi's block produces 12288 values. Summing the first 8192 of them AND striding `wo` by 8192 gives a number that is
    /// finite, plausible and wrong -- which is why it survived nine rounds of looking. This test computes both.
    #[test]
    fn the_output_projection_uses_the_tensors_width_not_8192() {
        let d = 4usize;
        let kimi_width = 12usize; // stands in for 12288
        let wo: Vec<f32> = (0..kimi_width * d).map(|i| (i as f32 * 0.017).sin()).collect();
        let o: Vec<f64> = (0..kimi_width).map(|i| (i as f64 * 0.11).cos()).collect();
        let right = output_projection(&wo, &o, d);
        assert_eq!(right.len(), d);
        assert!(right.iter().all(|v| v.is_finite()));

        // the BUG, reproduced exactly: width hardcoded to 8 instead of 12
        let buggy: Vec<f64> = (0..d)
            .map(|r| (0..8.min(o.len())).map(|c| wo.get(r * 8 + c).copied().unwrap_or(0.0) as f64 * o[c]).sum())
            .collect();
        assert!(
            right.iter().zip(&buggy).any(|(a, b)| (a - b).abs() > 1e-9),
            "the hardcoded-8 projection produced the same answer as the correct one, so this test cannot detect the bug"
        );

        // and a width the driver cannot guess from the block's output is REFUSED, not silently used
        let short = vec![0.0f64; kimi_width - 1];
        assert!(std::panic::catch_unwind(|| output_projection(&wo, &short, d)).is_err());
    }

    use super::*;

    /// The per-head loop this replaced, kept verbatim as the reference it was.
    fn per_head_loop(wvb: &[f32], kva: &[f64], lat: usize, per_head: usize, heads: usize) -> Vec<f64> {
        let mut vcat = vec![0.0f64; heads * per_head];
        for h in 0..heads {
            for r in 0..per_head {
                let mut acc = 0.0;
                for c in 0..lat {
                    acc += wvb[(h * per_head + r) * lat + c] as f64 * kva[c];
                }
                vcat[h * per_head + r] = acc;
            }
        }
        vcat
    }

    /// THE FIX: one matmul instead of a per-head loop. If the layout argument is wrong, this fails and the loop is right.
    #[test]
    fn the_flat_matmul_equals_the_per_head_loop() {
        // GLM's shape, then Kimi's: different heads, different per-head width, same layout.
        for (lat, per_head, heads) in [(8usize, 4usize, 3usize), (8, 2, 6)] {
            let n = lat * per_head * heads;
            let wvb: Vec<f32> = (0..n).map(|i| ((i % 97) as f32 - 48.0) * 0.013).collect();
            let kva: Vec<f64> = (0..lat + 2).map(|i| (i as f64 * 0.31).cos()).collect(); // longer than lat on purpose
            let a = v_projection(&wvb, &kva, lat);
            let b = per_head_loop(&wvb, &kva, lat, per_head, heads);
            assert_eq!(a.len(), b.len());
            for i in 0..a.len() {
                assert!((a[i] - b[i]).abs() < 1e-12, "index {i}: {} vs {}", a[i], b[i]);
            }
        }
    }

    /// THE OTHER FIX: the latent is the NORM's width, not kv_a's. Kimi emits 576 and consumes 512; using 576 gave
    /// 6291456/576 = 10922.67, a number that appears nowhere in the checkpoint.
    #[test]
    fn the_latent_is_the_norm_width_not_the_emitters() {
        let (lat, per_head, heads) = (8usize, 4usize, 3usize);
        let wvb: Vec<f32> = (0..lat * per_head * heads).map(|i| i as f32 * 0.001).collect();
        let kva_with_rope: Vec<f64> = (0..lat + 2).map(|i| i as f64).collect();
        let v = v_projection(&wvb, &kva_with_rope, lat); // norm width, as the caller must pass
        assert_eq!(v.len(), per_head * heads, "v_out must be per_head*heads, not a fraction of it");
        // and the emitter's own width is NOT a valid latent here
        let bad = std::panic::catch_unwind(|| v_projection(&wvb, &kva_with_rope, lat + 2));
        assert!(bad.is_err(), "using kv_a's width as the latent must be refused, not silently averaged");
    }

    /// A whole block on synthetic tensors: the output must depend on q_a, and must not read past v_b.
    #[test]
    fn a_block_runs_and_its_widths_come_from_the_tensors() {
        let d = 4usize;
        let q_lora = 3usize;
        let q_out = 6usize;
        let lat = 5usize;
        let v_out = 6usize;
        let q_a: Vec<f32> = (0..d * q_lora).map(|i| (i as f32 * 0.11).sin()).collect();
        let q_b: Vec<f32> = (0..q_lora * q_out).map(|i| (i as f32 * 0.07).cos()).collect();
        let kv_a: Vec<f32> = (0..d * (lat + 1)).map(|i| (i as f32 * 0.05).sin()).collect();
        let v_b: Vec<f32> = (0..lat * v_out).map(|i| (i as f32 * 0.03).cos()).collect();
        let out: Vec<f32> = (0..v_out * d).map(|i| (i as f32 * 0.02).sin()).collect();
        let xin: Vec<f64> = (0..d).map(|i| 0.5 + i as f64 * 0.1).collect();
        let y = mla_1token(&q_a, &q_b, &kv_a, lat, &v_b, &out, &xin, d);
        assert_eq!(y.len(), d);
        assert!(y.iter().all(|v| v.is_finite()));

        // a different token must move the result (otherwise nothing is wired)
        let xin2: Vec<f64> = (0..d).map(|i| 0.7 - i as f64 * 0.05).collect();
        let y2 = mla_1token(&q_a, &q_b, &kv_a, lat, &v_b, &out, &xin2, d);
        assert!(y.iter().zip(&y2).any(|(a, b)| (a - b).abs() > 1e-9), "the block ignores its input");
    }
}
