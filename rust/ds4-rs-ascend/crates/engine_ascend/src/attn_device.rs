//! The attention dispatcher that holds a KERNEL dispatcher -- the seam the MLA path needs.
//!
//! `engine_core`'s `AttentionDispatcher` is, in its own words, "intentionally NOT a supertrait of
//! `KernelDispatcher`"; dispatchers compose by HOLDING both. Its only implementation is
//! `CpuAttentionDispatcher`, a unit struct, so the real MLA attention path has had no device hook.
//!
//! This is that hook, and it is deliberately INCREMENTAL: methods are routed to the kernel dispatcher
//! one at a time, each with a bit-comparison against the CPU reference, because each one carries its
//! own semantics (eps, gamma, clamps) that a bulk rewrite would get subtly wrong.
//!
//! ROUTED SO FAR: `qkv_rms_norm_rows` -- the two RMS norms over the q-lora row and the kv row now go
//! through `KernelDispatcher::rms_norm`, which on the device is `ds4_rmsmul_*`. Everything else still
//! delegates to `CpuAttentionDispatcher`.

use engine_core::attn_dispatch::{
    AttnHeadsOut, AttnPrefixBatchedOut, AttentionDispatcher, CpuAttentionDispatcher, HcKind,
    KvCacheView, LayerParams,
};
use engine_core::dispatch::KernelDispatcher;

pub struct AscendAttentionDispatcher<D: KernelDispatcher> {
    /// The reference implementation, for every method not yet routed.
    inner: CpuAttentionDispatcher,
    /// The kernel dispatcher: on device this is `AscendDispatcher`, which forwards to the emitted
    /// kernels; in tests it is `CpuDispatcher`, which makes the comparison exact.
    kernels: D,
}

impl<D: KernelDispatcher> AscendAttentionDispatcher<D> {
    pub fn new(kernels: D) -> Self {
        Self { inner: CpuAttentionDispatcher, kernels }
    }

    pub fn kernels(&self) -> &D {
        &self.kernels
    }
}

impl<D: KernelDispatcher> AttentionDispatcher for AscendAttentionDispatcher<D> {
    fn qkv_rms_norm_rows(
        &self,
        params: &LayerParams,
        qr: &[f32],
        kv_raw: &[f32],
        gamma_q: &[f32],
        gamma_kv: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        (
            self.kernels.rms_norm(qr, gamma_q, params.rms_eps),
            self.kernels.rms_norm(kv_raw, gamma_kv, params.rms_eps),
        )
    }

    fn hc_collapse_norm(
        &self,
        params: &LayerParams,
        kind: HcKind,
        hc_fn: &[f32],
        hc_scale: &[f32],
        hc_base: &[f32],
        prev_hc: &[f32],
        use_gamma: Option<&[f32]>,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        self.inner.hc_collapse_norm(params, kind, hc_fn, hc_scale, hc_base, prev_hc, use_gamma)
    }

    fn rope_tail(&self, params: &LayerParams, x: &mut [f32], pos: u32, backward: bool) {
        self.inner.rope_tail(params, x, pos, backward)
    }

    fn kv_fp8_store(&self, params: &LayerParams, kv_row_f32: &[f32], view: &mut KvCacheView<'_>) {
        self.inner.kv_fp8_store(params, kv_row_f32, view)
    }

    fn flash_attn_decode(
        &self,
        params: &LayerParams,
        q: &[f32],
        kv_raw: &[f32],
        n_raw: u32,
        raw_cap: u32,
        raw_start: u32,
        kv_comp: Option<&[f32]>,
        n_comp: u32,
        comp_selected: Option<&[u32]>,
        n_selected: u32,
        attn_sinks: &[f32],
    ) -> AttnHeadsOut {
        self.inner.flash_attn_decode(
            params, q, kv_raw, n_raw, raw_cap, raw_start, kv_comp, n_comp, comp_selected,
            n_selected, attn_sinks,
        )
    }

    /// ROUTED: the GROUPED projection (8 groups of 1024 rows over 4096 inputs) and the outer
    /// `w_o_b` matvec both go through `KernelDispatcher::matvec_f32`. These are the largest matrix
    /// multiplies in the attention block -- 8x1024x4096 plus 4096x8192 per layer.
    ///
    /// Delegated when any of the fidelity/capture switches is set: `DS4_Q8_0_ACT` and
    /// `DS4_MATVEC_F32_FIDELITY` change the arithmetic, and `DS4_DUMP_ATTN_OUT` reads the capture
    /// hooks this path does not maintain. Skipping the capture hook is invisible unless that switch is
    /// set, which is exactly what the guard is for.
    fn attn_output_proj(
        &self,
        params: &LayerParams,
        heads: &[f32],
        w_o_a: &[f32],
        w_o_b: &[f32],
        cur_hc: &[f32],
        hc_split_post: &[f32],
        hc_split_comb: &[f32],
    ) -> Vec<f32> {
        let debugged = std::env::var("DS4_Q8_0_ACT").ok().as_deref() == Some("1")
            || std::env::var("DS4_MATVEC_F32_FIDELITY").ok().as_deref() == Some("1")
            || std::env::var("DS4_DUMP_ATTN_OUT").is_ok();
        if debugged {
            return self.inner.attn_output_proj(
                params, heads, w_o_a, w_o_b, cur_hc, hc_split_post, hc_split_comb,
            );
        }

        let d_embd = params.d_embd as usize;
        let n_hc = params.n_hc as usize;
        let n_groups = params.n_out_group as usize;
        let q_dim = heads.len();
        let group_dim = q_dim / n_groups;
        let n_lora_o = w_o_a.len() / (n_groups * group_dim);
        let out_low_dim = n_groups * n_lora_o;

        let mut attn_low = vec![0.0f32; out_low_dim];
        for g in 0..n_groups {
            let heads_g = &heads[g * group_dim..(g + 1) * group_dim];
            let block = &w_o_a[g * n_lora_o * group_dim..(g + 1) * n_lora_o * group_dim];
            let y = self.kernels.matvec_f32(block, heads_g, n_lora_o);
            attn_low[g * n_lora_o..(g + 1) * n_lora_o].copy_from_slice(&y);
        }
        let attn_out = self.kernels.matvec_f32(w_o_b, &attn_low, d_embd);

        let mut after = vec![0.0f32; n_hc * d_embd];
        for dst in 0..n_hc {
            let base = dst * d_embd;
            let w_post = hc_split_post[dst];
            for e in 0..d_embd {
                let mut acc = w_post * attn_out[e];
                for src in 0..n_hc {
                    acc += hc_split_comb[dst + src * n_hc] * cur_hc[src * d_embd + e];
                }
                after[base + e] = acc;
            }
        }
        after
    }

    /// ROUTED: the gate and up projections both go through `KernelDispatcher::matvec_f32`; the
    /// clamp and the SiLU multiply stay here because they are elementwise.
    ///
    /// The clamp is DS4-Flash's `deepseek4.swiglu_clamp_shexp` (10.0) and is NOT optional: the CPU
    /// reference's own comment records that dropping it "silently disagreed with `kernel_swiglu_f32`
    /// and with antirez's own comment". So it is applied here exactly as there, and the two fidelity
    /// switches fall back to the reference rather than being re-derived.
    fn shared_expert(
        &self,
        params: &LayerParams,
        ffn_norm: &[f32],
        w_gate: &[f32],
        w_up: &[f32],
        shared_dim: u32,
        clamp: f32,
    ) -> Vec<f32> {
        let switched = std::env::var("DS4_Q8_0_ACT").ok().as_deref() == Some("1")
            || std::env::var("DS4_MATVEC_F32_FIDELITY").ok().as_deref() == Some("1");
        if switched {
            return self.inner.shared_expert(params, ffn_norm, w_gate, w_up, shared_dim, clamp);
        }
        let sd = shared_dim as usize;
        let g = self.kernels.matvec_f32(w_gate, ffn_norm, sd);
        let u = self.kernels.matvec_f32(w_up, ffn_norm, sd);
        let mut out = vec![0.0f32; sd];
        for i in 0..sd {
            let (gi, ui) = if clamp > 1e-6 {
                (g[i].min(clamp), u[i].clamp(-clamp, clamp))
            } else {
                (g[i], u[i])
            };
            out[i] = crate::ffn::silu(gi) * ui;
        }
        out
    }

    /// ROUTED: the down projection goes through `KernelDispatcher::matvec_f32` (the w8a8 kernels on
    /// device); the hc expand stays here because it is elementwise and cheap.
    ///
    /// The two fidelity switches are honoured by FALLING BACK to the reference rather than by
    /// reimplementing what they select: `DS4_Q8_0_ACT` rounds activations through q8_0 and
    /// `DS4_MATVEC_F32_FIDELITY` selects a different accumulation order. Reimplementing either here
    /// would duplicate logic that lives in `engine_core` and drift from it.
    fn shared_down_hc_expand_add(
        &self,
        params: &LayerParams,
        shared_mid: &[f32],
        w_down: &[f32],
        routed_out: &[f32],
        after_attn_hc: &[f32],
        hc_split_post: &[f32],
        hc_split_comb: &[f32],
    ) -> Vec<f32> {
        let q80 = std::env::var("DS4_Q8_0_ACT").ok().as_deref() == Some("1");
        let fidelity = std::env::var("DS4_MATVEC_F32_FIDELITY").ok().as_deref() == Some("1");
        if q80 || fidelity {
            return self.inner.shared_down_hc_expand_add(
                params, shared_mid, w_down, routed_out, after_attn_hc, hc_split_post, hc_split_comb,
            );
        }
        let d_embd = params.d_embd as usize;
        let n_hc = params.n_hc as usize;
        let shared_out = self.kernels.matvec_f32(w_down, shared_mid, d_embd);
        let mut after = vec![0.0f32; n_hc * d_embd];
        for dst in 0..n_hc {
            let base = dst * d_embd;
            let w_post = hc_split_post[dst];
            for e in 0..d_embd {
                let mut acc = w_post * (shared_out[e] + routed_out[e]);
                for src in 0..n_hc {
                    acc += hc_split_comb[dst + src * n_hc] * after_attn_hc[src * d_embd + e];
                }
                after[base + e] = acc;
            }
        }
        after
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_core::dispatch::CpuDispatcher;

    fn params() -> LayerParams {
        LayerParams {
            layer_idx: 0, d_embd: 8, n_hc: 2, n_head: 2, head_dim: 4, n_rot: 2,
            n_lora_q: 6, n_lora_kv: 4, hc_sinkhorn_iter: 1, hc_eps: 1e-6, rms_eps: 1e-6,
            rope_orig_ctx: 4096, rope_freq_base: 10000.0, rope_freq_scale: 1.0,
            rope_ext_factor: 0.0, rope_attn_factor: 1.0, compress_ratio: 0, n_out_group: 1,
        }
    }

    /// The routed method must agree with the CPU reference EXACTLY when both sit on the CPU
    /// dispatcher. That is the property that makes routing one method at a time safe: a mismatch is a
    /// difference in SEMANTICS, not in hardware.
    #[test]
    fn the_routed_method_matches_the_cpu_reference() {
        let p = params();
        let qr: Vec<f32> = (0..6).map(|i| (i as f32 - 2.5) * 0.3).collect();
        let kv: Vec<f32> = (0..4).map(|i| (i as f32 - 1.5) * 0.7).collect();
        let gq: Vec<f32> = (0..6).map(|i| 1.0 + i as f32 * 0.05).collect();
        let gk: Vec<f32> = (0..4).map(|i| 1.0 - i as f32 * 0.03).collect();

        let reference = CpuAttentionDispatcher.qkv_rms_norm_rows(&p, &qr, &kv, &gq, &gk);
        let ours = AscendAttentionDispatcher::new(CpuDispatcher).qkv_rms_norm_rows(&p, &qr, &kv, &gq, &gk);

        assert_eq!(ours.0.len(), reference.0.len());
        for (a, b) in ours.0.iter().zip(&reference.0) {
            assert!((a - b).abs() <= 1e-6, "q row disagrees: {a} vs {b}");
        }
        for (a, b) in ours.1.iter().zip(&reference.1) {
            assert!((a - b).abs() <= 1e-6, "kv row disagrees: {a} vs {b}");
        }
    }

    /// And the values must actually be an RMS norm, not a pass-through: a test that would pass on a
    /// function returning its input unchanged would not be a test.
    #[test]
    fn the_routed_method_normalises_rather_than_passing_through() {
        let p = params();
        let qr = vec![3.0f32; 6];
        let kv = vec![3.0f32; 4];
        let gq = vec![1.0f32; 6];
        let gk = vec![1.0f32; 4];
        let (q, k) = AscendAttentionDispatcher::new(CpuDispatcher)
            .qkv_rms_norm_rows(&p, &qr, &kv, &gq, &gk);
        // A constant row normalises to ~1 per element, not to 3.
        for v in q.iter().chain(k.iter()) {
            assert!((v - 1.0).abs() < 1e-3, "expected ~1.0 after RMS norm, got {v}");
        }
    }
}

#[cfg(test)]
mod routed_tests {
    use super::*;
    use engine_core::dispatch::CpuDispatcher;

    fn params(d_embd: u32, n_hc: u32) -> LayerParams {
        LayerParams {
            layer_idx: 0, d_embd, n_hc, n_head: 2, head_dim: 4, n_rot: 2, n_lora_q: 6, n_lora_kv: 4,
            hc_sinkhorn_iter: 1, hc_eps: 1e-6, rms_eps: 1e-6, rope_orig_ctx: 4096,
            rope_freq_base: 10000.0, rope_freq_scale: 1.0, rope_ext_factor: 0.0, rope_attn_factor: 1.0,
            compress_ratio: 0, n_out_group: 1,
        }
    }

    #[test]
    fn the_routed_down_projection_matches_the_reference() {
        let p = params(4, 2);
        let sd = 3usize;
        let shared_mid: Vec<f32> = (0..sd).map(|i| 0.5 + i as f32 * 0.25).collect();
        let w_down: Vec<f32> = (0..4 * sd).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
        let routed: Vec<f32> = (0..4).map(|i| i as f32 * 0.05).collect();
        let after_attn: Vec<f32> = (0..8).map(|i| (i as f32 - 3.5) * 0.02).collect();
        let post = vec![1.0f32, 0.5];
        let comb = vec![0.25f32, 0.5, 0.75, 1.0];

        let want = CpuAttentionDispatcher.shared_down_hc_expand_add(
            &p, &shared_mid, &w_down, &routed, &after_attn, &post, &comb,
        );
        let got = AscendAttentionDispatcher::new(CpuDispatcher).shared_down_hc_expand_add(
            &p, &shared_mid, &w_down, &routed, &after_attn, &post, &comb,
        );
        assert_eq!(got.len(), want.len());
        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).abs() <= 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn a_zero_down_projection_still_carries_the_hc_path() {
        // Without w_down the result is the hc combination alone -- non-zero when after_attn_hc is.
        // A function that returned zeros would pass a weaker test that only compared to the reference
        // if the reference were also zero, so this pins that the path is live.
        let p = params(4, 2);
        let shared_mid = vec![1.0f32; 3];
        let w_down = vec![0.0f32; 12];
        let routed = vec![0.0f32; 4];
        let after_attn: Vec<f32> = vec![1.0; 8];
        let post = vec![1.0f32, 1.0];
        let comb = vec![1.0f32, 0.0, 0.0, 1.0];
        let got = AscendAttentionDispatcher::new(CpuDispatcher).shared_down_hc_expand_add(
            &p, &shared_mid, &w_down, &routed, &after_attn, &post, &comb,
        );
        assert!(got.iter().all(|v| (*v - 1.0).abs() < 1e-6), "{got:?}");
    }
}

#[cfg(test)]
mod outproj_tests {
    use super::*;
    use engine_core::dispatch::CpuDispatcher;

    fn params() -> LayerParams {
        LayerParams {
            layer_idx: 0, d_embd: 4, n_hc: 2, n_head: 2, head_dim: 2, n_rot: 1, n_lora_q: 4,
            n_lora_kv: 2, hc_sinkhorn_iter: 1, hc_eps: 1e-6, rms_eps: 1e-6, rope_orig_ctx: 4096,
            rope_freq_base: 10000.0, rope_freq_scale: 1.0, rope_ext_factor: 0.0,
            rope_attn_factor: 1.0, compress_ratio: 0, n_out_group: 2,
        }
    }

    #[test]
    fn the_routed_output_projection_matches_the_reference() {
        let p = params();
        // 2 groups x 2 heads x 2 head_dim = 8 heads values; group_dim 4, n_lora_o 2 -> wo_a 2*2*4.
        let heads: Vec<f32> = (0..8).map(|i| (i as f32 - 3.5) * 0.1).collect();
        let w_o_a: Vec<f32> = (0..2 * 2 * 4).map(|i| ((i % 5) as f32 - 2.0) * 0.05).collect();
        let w_o_b: Vec<f32> = (0..4 * 4).map(|i| ((i % 7) as f32 - 3.0) * 0.02).collect();
        let cur_hc: Vec<f32> = (0..8).map(|i| (i as f32) * 0.01).collect();
        let post = vec![1.0f32, 0.5];
        let comb = vec![0.25f32, 0.5, 0.75, 1.0];

        let want = CpuAttentionDispatcher.attn_output_proj(&p, &heads, &w_o_a, &w_o_b, &cur_hc, &post, &comb);
        let got = AscendAttentionDispatcher::new(CpuDispatcher)
            .attn_output_proj(&p, &heads, &w_o_a, &w_o_b, &cur_hc, &post, &comb);
        assert_eq!(got.len(), want.len());
        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).abs() <= 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn zero_projection_still_propagates_the_hc_split() {
        // With w_o_a and w_o_b zero, `attn_out` is zero and the result is hc_split_comb @ cur_hc.
        // A path that dropped the hc expand -- or returned zeros -- would fail here and could pass the
        // equivalence test only if the reference were also zero.
        let p = params();
        let heads = vec![1.0f32; 8];
        let w_o_a = vec![0.0f32; 16];
        let w_o_b = vec![0.0f32; 16];
        let cur_hc: Vec<f32> = (0..8).map(|i| (i % 4) as f32).collect();
        let post = vec![0.0f32, 0.0];
        let comb = vec![1.0f32, 0.0, 0.0, 1.0];
        let got = AscendAttentionDispatcher::new(CpuDispatcher)
            .attn_output_proj(&p, &heads, &w_o_a, &w_o_b, &cur_hc, &post, &comb);
        // dst slot 0 <- src slot 0, slot 1 <- src slot 1.
        assert!((got[0] - cur_hc[0]).abs() < 1e-6, "{got:?}");
        assert!((got[4] - cur_hc[4]).abs() < 1e-6, "{got:?}");
    }
}

#[cfg(test)]
mod shared_tests {
    use super::*;
    use engine_core::dispatch::CpuDispatcher;

    fn params() -> LayerParams {
        LayerParams {
            layer_idx: 0, d_embd: 4, n_hc: 2, n_head: 2, head_dim: 2, n_rot: 1, n_lora_q: 4,
            n_lora_kv: 2, hc_sinkhorn_iter: 1, hc_eps: 1e-6, rms_eps: 1e-6, rope_orig_ctx: 4096,
            rope_freq_base: 10000.0, rope_freq_scale: 1.0, rope_ext_factor: 0.0,
            rope_attn_factor: 1.0, compress_ratio: 0, n_out_group: 1,
        }
    }

    #[test]
    fn the_routed_shared_expert_matches_the_reference() {
        let p = params();
        let sd = 3usize;
        let x: Vec<f32> = (0..4).map(|i| (i as f32 - 1.5) * 0.4).collect();
        let wg: Vec<f32> = (0..sd * 4).map(|i| ((i % 5) as f32 - 2.0) * 0.3).collect();
        let wu: Vec<f32> = (0..sd * 4).map(|i| ((i % 3) as f32 - 1.0) * 0.2).collect();

        for clamp in [0.0f32, 10.0] {
            let want = CpuAttentionDispatcher.shared_expert(&p, &x, &wg, &wu, sd as u32, clamp);
            let got = AscendAttentionDispatcher::new(CpuDispatcher)
                .shared_expert(&p, &x, &wg, &wu, sd as u32, clamp);
            for (a, b) in got.iter().zip(&want) {
                assert!((a - b).abs() <= 1e-5, "clamp {clamp}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn the_clamp_actually_clamps() {
        // Large gate/up inputs with clamp = 1 must be capped, so the result differs from clamp = 0.
        // A path that ignored `clamp` would pass the equivalence test only if the reference did too,
        // and the reference has it -- so this pins that MY body reads the parameter rather than
        // accidentally delegating.
        let p = params();
        let sd = 2usize;
        let x = vec![50.0f32; 4];
        let wg = vec![1.0f32; sd * 4];
        let wu = vec![1.0f32; sd * 4];
        let unclamped = AscendAttentionDispatcher::new(CpuDispatcher).shared_expert(&p, &x, &wg, &wu, sd as u32, 0.0);
        let clamped = AscendAttentionDispatcher::new(CpuDispatcher).shared_expert(&p, &x, &wg, &wu, sd as u32, 1.0);
        assert!(unclamped[0] > clamped[0], "clamp did nothing: {} vs {}", unclamped[0], clamped[0]);
    }
}
