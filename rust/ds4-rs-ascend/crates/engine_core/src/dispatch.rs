// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Kernel dispatch trait + CPU reference implementations.
//!
//! The DS4 decode pipeline calls a small set of math primitives (rms_norm,
//! matvec, softplus-sqrt, router-finalize, moe_routed_step) once per layer.
//! On Linux we run scalar Rust references; on macOS the same call sites will
//! dispatch to generated Metal kernels via the kernel registry.
//!
//! `KernelDispatcher` is the seam: a trait object pointer (or generic) lets
//! `decode_layer` stay shape-stable while we swap the back end. The CPU
//! `CpuDispatcher` preserves the current antirez-compatible decode contract.
//! `MathReferenceDispatcher` is the stricter semantic oracle for cases where
//! antirez compatibility is not the same thing as mathematical correctness.

use crate::forward::{matvec_f32, rms_norm};
use crate::moe::{moe_routed_step, router_finalize, router_finalize_math_reference, softplus_sqrt};

/// Decode-time math primitives. Each method has the same signature as the
/// corresponding free function in `forward`/`moe`; the Metal impl will
/// dispatch through a kernel registry instead of running scalar Rust.
pub trait KernelDispatcher {
    fn rms_norm(&self, x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32>;

    fn matvec_f32(&self, w: &[f32], x: &[f32], d_out: usize) -> Vec<f32>;

    /// Quantized matvec: `w_q8` is GGUF `block_q8_0` row-major `[d_out, d_in]`
    /// (34 B per 32 weights = f16 scale + 32×int8); `x` is f32 `[d_in]`; returns
    /// `[d_out]`. The DEFAULT dequantizes q8→f32 then calls `matvec_f32` (the CPU
    /// reference). GPU backends OVERRIDE this with a native q8 kernel that reads
    /// the 1-byte/weight blocks directly (no f32 weight materialized) — what lets
    /// the DSpark drafter run on ~1 B/weight quantized weights so it co-resides
    /// with (and runs faster than) the f32 path.
    fn matvec_q8(&self, w_q8: &[u8], x: &[f32], d_out: usize) -> Vec<f32> {
        let d_in = x.len();
        assert_eq!(d_in % 32, 0, "matvec_q8: d_in {d_in} % 32 != 0");
        let row_bytes = (d_in / 32) * 34;
        assert_eq!(
            w_q8.len(),
            d_out * row_bytes,
            "matvec_q8: w bytes {} != d_out*row_bytes {}",
            w_q8.len(),
            d_out * row_bytes
        );
        let mut w_f32 = vec![0.0f32; d_out * d_in];
        for o in 0..d_out {
            let row = &w_q8[o * row_bytes..(o + 1) * row_bytes];
            for (b, blk) in row.chunks_exact(34).enumerate() {
                let d = crate::layer_view::f16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                for i in 0..32 {
                    w_f32[o * d_in + b * 32 + i] = d * (blk[2 + i] as i8 as f32);
                }
            }
        }
        self.matvec_f32(&w_f32, x, d_out)
    }

    fn softplus_sqrt(&self, logits: &[f32]) -> Vec<f32>;

    fn router_finalize(&self, probs: &[f32], bias: &[f32], k: usize) -> (Vec<usize>, Vec<f32>);

    /// `layer_idx` is the 0-based MoE layer index — backends that route
    /// dispatch through preloaded per-layer quantized tables (e.g.
    /// `MetalDispatcher` with `QuantizedExpertWeights`) need it to pick
    /// the right table. The `experts_w_*` slices remain the CPU-reference
    /// inputs; backends that consult preloaded tables may ignore them.
    fn moe_routed_step(
        &self,
        layer_idx: u32,
        x: &[f32],
        selected: &[usize],
        weights: &[f32],
        experts_w_gate: &[f32],
        experts_w_up: &[f32],
        experts_w_down: &[f32],
        d_ffn: usize,
    ) -> Vec<f32>;

    /// Phase C.3a. Batched form of the per-layer Q-projection
    /// head chain: `matvec_f32(w_qa, x, n_lora_q) → rms_norm(qr, gamma_q, eps)`.
    /// The default impl runs the two underlying trait methods sequentially,
    /// so the contract is invariant across backends. `MetalDispatcher`
    /// overrides this to pack both ops into one `MTLCommandBuffer` with a
    /// single readback (saves 1 commit+wait+readback per layer per token).
    /// Bit-identity of the override vs the default is gated by
    /// the `layer_qa_rms_batched_smoke` smoke test.
    fn layer_qa_rms_batched(
        &self,
        x: &[f32],
        w_qa: &[f32],
        gamma_q: &[f32],
        n_lora_q: usize,
        eps_rms: f32,
    ) -> Vec<f32> {
        let qr = self.matvec_f32(w_qa, x, n_lora_q);
        self.rms_norm(&qr, gamma_q, eps_rms)
    }

    /// Phase C.3b. Batched form of the shared-expert FFN body:
    /// `matvec(w_gate, ffn_norm) → matvec(w_up, ffn_norm) → swiglu →
    /// (optional q8_0_round_trip) → matvec(w_down, mid) → [d_embd]`.
    /// Returns the `shared_out` vector that the caller adds into the
    /// HC expand-add fold.
    ///
    /// `want_q80` mirrors `DS4_Q8_0_ACT=1` — when set, the SwiGLU output is
    /// q8_0-roundtripped before the down matvec (matches antirez's
    /// `matvec_q8_0` activation-side rounding, ds4.c:4905).
    ///
    /// The default impl runs the chain sequentially through the trait
    /// methods + `crate::forward::silu` + `crate::forward::q8_0_round_trip`,
    /// so it remains a no-op replacement for CPU paths.  `MetalDispatcher`
    /// overrides to pack the five passes into one `MTLCommandBuffer` with
    /// a single readback (saves 2–3 commit+wait+readbacks per layer per
    /// token, up to 129 cwr/token at 43 layers when q80=on).  Bit-identity
    /// of the override vs the default is gated by
    /// the `shared_chain_batched_smoke` smoke test and by
    /// `cpu_dispatcher_shared_chain_batched_matches_sequential`.
    ///
    /// Limitation: the override is only safe under default-branch silu
    /// (matches `ds4_silu_default_f32` MSL and antirez
    /// `kernel_swiglu_f32`). Callers under `DS4_SILU_FIDELITY=1` must
    /// use the sequential trait path instead.
    fn shared_chain_batched(
        &self,
        ffn_norm: &[f32],
        w_gate: &[f32],
        w_up: &[f32],
        w_down: &[f32],
        shared_dim: u32,
        want_q80: bool,
    ) -> Vec<f32> {
        let sd = shared_dim as usize;
        let d_embd = ffn_norm.len();
        debug_assert_eq!(w_gate.len(), sd * d_embd);
        debug_assert_eq!(w_up.len(), sd * d_embd);
        debug_assert_eq!(w_down.len(), d_embd * sd);
        let g = self.matvec_f32(w_gate, ffn_norm, sd);
        let u = self.matvec_f32(w_up, ffn_norm, sd);
        let mid: Vec<f32> = g
            .iter()
            .zip(u.iter())
            .map(|(&gi, &ui)| crate::forward::silu(gi) * ui)
            .collect();
        let mid_in = if want_q80 {
            crate::forward::q8_0_round_trip(&mid)
        } else {
            mid
        };
        self.matvec_f32(w_down, &mid_in, d_embd)
    }

    /// Phase C.3c. Batched form of the once-per-token output-HC
    /// head pre-amble: `rms_norm(inp_hc, unit_gamma, eps) → matvec_f32(fn_w,
    /// flat, n_hc)`. Returns `pre[n_hc]`; callers run the trivial sigmoid +
    /// weighted-sum post-amble on CPU (`n_hc = 8` for production DS4).
    ///
    /// The default impl runs `rms_norm` on an all-ones gamma and then
    /// `matvec_f32` through the trait, matching the sequential CPU path.
    /// `MetalDispatcher` overrides to pack both passes into one
    /// `MTLCommandBuffer` with a single readback (saves 1 commit+wait+
    /// readback per token). Bit-identity of the override vs the default
    /// is gated by the `output_hc_head_batched_smoke` smoke test and by
    /// `cpu_dispatcher_output_hc_head_batched_matches_sequential`.
    ///
    /// No fidelity branch — `output_hc_head_one`'s sigmoid is already
    /// branched and matches antirez `ds4.c:7660-7674` under any setting.
    fn output_hc_head_batched(
        &self,
        inp_hc: &[f32],
        fn_w: &[f32],
        n_hc: usize,
        d_embd: usize,
        eps_rms: f32,
    ) -> Vec<f32> {
        let hc_dim = n_hc * d_embd;
        debug_assert_eq!(inp_hc.len(), hc_dim);
        debug_assert_eq!(fn_w.len(), hc_dim * n_hc);
        let unit_gamma = vec![1.0f32; hc_dim];
        let flat = self.rms_norm(inp_hc, &unit_gamma, eps_rms);
        self.matvec_f32(fn_w, &flat, n_hc)
    }

    /// Phase C.3d.1. Per-head RMS norm on Q heads after the
    /// `attn_q_b` matvec — antirez `head_rms_norm_inplace` (ds4.c:4511).
    /// Splits `x` into `n_head` chunks of `head_dim`, normalises each
    /// chunk against unit gamma with `eps`, and returns the result. Input
    /// is `[n_head * head_dim]` flattened.
    ///
    /// The default impl runs the antirez f32 chain (matches the Metal
    /// kernel `ds4_head_rms_norm_f32` and ds4.c default). When
    /// `DS4_HEAD_RMS_F64_FIDELITY=1` callers MUST take an inline f64
    /// reduction path instead — the override only matches default-OFF.
    /// `MetalDispatcher` overrides to dispatch through the existing
    /// `ds4_head_rms_norm_f32` Metal pipeline (Phase A.1).
    fn head_rms_norm(&self, x: &[f32], n_head: usize, head_dim: usize, eps: f32) -> Vec<f32> {
        debug_assert_eq!(x.len(), n_head * head_dim);
        let mut out = x.to_vec();
        let hd = head_dim as f32;
        for h in 0..n_head {
            let chunk = &mut out[h * head_dim..(h + 1) * head_dim];
            let ss: f32 = chunk.iter().map(|&v| v * v).sum();
            let scale = 1.0 / ((ss / hd) + eps).sqrt();
            for v in chunk.iter_mut() {
                *v *= scale;
            }
        }
        out
    }

    /// Phase C.3d.3. Batched per-layer Q/K/V split chain:
    ///   q_heads_raw = matvec_f32(w_q_b,  qr_normed_q,  q_dim)
    ///   q_heads     = head_rms_norm(q_heads_raw, n_head, head_dim, eps_rms)
    ///   kv_raw_row  = matvec_f32(w_kv,   normed_kv,    kv_row)
    /// Returns `(q_heads, kv_raw_row)`. The default impl runs the three
    /// trait methods sequentially so semantics are invariant on every
    /// backend. `MetalDispatcher` overrides to pack all three passes
    /// into one `MTLCommandBuffer` (saves 2 commit+wait+readback per
    /// layer per token). Bit-identity of the override vs the default
    /// is gated by the `qkv_b_head_rms_batched_smoke` smoke test.
    ///
    /// Callers under `DS4_HEAD_RMS_F64_FIDELITY=1` MUST NOT use this
    /// method — the default impl runs the f32 head_rms_norm chain and
    /// the Metal override matches it. Take the trait methods
    /// separately and run an inline f64 reduction for `head_rms_norm`
    /// instead (see decode_step.rs:747-771).
    #[allow(clippy::too_many_arguments)]
    fn qkv_b_head_rms_batched(
        &self,
        qr_normed_q: &[f32],
        w_q_b: &[f32],
        q_dim: usize,
        n_head: usize,
        head_dim: usize,
        eps_rms: f32,
        normed_kv: &[f32],
        w_kv: &[f32],
        kv_row: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        debug_assert_eq!(q_dim, n_head * head_dim);
        let q_raw = self.matvec_f32(w_q_b, qr_normed_q, q_dim);
        let q_heads = self.head_rms_norm(&q_raw, n_head, head_dim, eps_rms);
        let kv_raw_row = self.matvec_f32(w_kv, normed_kv, kv_row);
        (q_heads, kv_raw_row)
    }

    /// Phase C.3e. Pack the per-layer MoE router head into one
    /// dispatcher call:
    ///   logits = matvec_f32(w_router, h_norm, n_experts)
    ///   probs  = softplus_sqrt(logits)
    /// Returns `probs`. The default impl runs the two trait methods
    /// sequentially so semantics are invariant on every backend.
    /// `MetalDispatcher` overrides to pack both passes into ONE
    /// `MTLCommandBuffer` (saves 1 commit+wait+readback per layer per
    /// token — 43 cwr/token on DS4 V4 Flash). Bit-identity of the
    /// override vs the default is gated by
    /// the `router_logits_batched_smoke` smoke test.
    fn router_logits_batched(
        &self,
        w_router: &[f32],
        h_norm: &[f32],
        n_experts: usize,
    ) -> Vec<f32> {
        let logits = self.matvec_f32(w_router, h_norm, n_experts);
        self.softplus_sqrt(&logits)
    }

    /// B Slice 5-redo. Fuse `moe_routed_step` and
    /// `shared_chain_batched` into one dispatcher call so the Metal
    /// backend can pack both into a single `MTLCommandBuffer` with one
    /// `commit + wait_until_completed`. The two ops have no data
    /// dependency on each other (both consume `ffn_normed`/`x`
    /// independently and their outputs are combined on CPU by
    /// `hc_expand_add_only`); fusion saves 1 wait per layer per token.
    ///
    /// Default impl runs the two methods sequentially so CPU backends
    /// keep working unchanged. `MetalDispatcher` overrides to share a
    /// cb. Empirical `DS4_MOE_TRACE` measurement showed
    /// `wait_until_completed` is ~25-100 ms per call while the GPU
    /// compute itself is ~300-400 us — so each saved wait recovers
    /// real wall time.
    #[allow(clippy::too_many_arguments)]
    fn moe_and_shared_chain_batched(
        &self,
        layer_idx: u32,
        // moe inputs
        moe_x: &[f32],
        moe_selected: &[usize],
        moe_weights: &[f32],
        moe_w_gate: &[f32],
        moe_w_up: &[f32],
        moe_w_down: &[f32],
        d_ffn: usize,
        // shared_chain inputs
        sh_ffn_norm: &[f32],
        sh_w_gate: &[f32],
        sh_w_up: &[f32],
        sh_w_down: &[f32],
        shared_dim: u32,
        want_q80: bool,
    ) -> (Vec<f32>, Vec<f32>) {
        let moe_out = self.moe_routed_step(
            layer_idx,
            moe_x,
            moe_selected,
            moe_weights,
            moe_w_gate,
            moe_w_up,
            moe_w_down,
            d_ffn,
        );
        let sh_out = self.shared_chain_batched(
            sh_ffn_norm,
            sh_w_gate,
            sh_w_up,
            sh_w_down,
            shared_dim,
            want_q80,
        );
        (moe_out, sh_out)
    }

    /// B. Fuse the attention-half QKV chain into one
    /// dispatcher call:
    ///   qr         = matvec_f32(attn_q_a, normed)
    ///   qr_normed  = rms_norm(qr, gamma_q)
    ///   q_raw      = matvec_f32(attn_q_b, qr_normed)
    ///   q_heads    = head_rms_norm(q_raw, n_head, head_dim)
    ///   kv_raw_row = matvec_f32(attn_kv, normed)
    /// Returns `(qr_normed, q_heads, kv_raw_row)` — `qr_normed` is
    /// surfaced because the indexer (compress_ratio==4 layers)
    /// consumes it on host.
    ///
    /// Default impl chains `layer_qa_rms_batched` + `qkv_b_head_rms_batched`
    /// so semantics are invariant on every backend. `MetalDispatcher`
    /// overrides to pack all 5 ops into ONE `MTLCommandBuffer` — saves
    /// 1 commit+wait + 2 readbacks per layer per token (43 cb commits
    /// per token saved across DS4's 43 layers). Bit-identity of the
    /// override vs the default is gated by
    /// the `attn_qkv_chain_batched_smoke` smoke test.
    #[allow(clippy::too_many_arguments)]
    fn attn_qkv_chain_batched(
        &self,
        normed: &[f32],
        attn_q_a: &[f32],
        gamma_q: &[f32],
        n_lora_q: usize,
        attn_q_b: &[f32],
        n_head: usize,
        head_dim: usize,
        eps_rms: f32,
        attn_kv: &[f32],
        kv_row: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let q_dim = n_head * head_dim;
        let qr_normed = self.layer_qa_rms_batched(normed, attn_q_a, gamma_q, n_lora_q, eps_rms);
        let (q_heads, kv_raw_row) = self.qkv_b_head_rms_batched(
            &qr_normed, attn_q_b, q_dim, n_head, head_dim, eps_rms, normed, attn_kv, kv_row,
        );
        (qr_normed, q_heads, kv_raw_row)
    }
}

/// Scalar Rust reference dispatcher. Always available, no device required.
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuDispatcher;

impl KernelDispatcher for CpuDispatcher {
    fn rms_norm(&self, x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
        rms_norm(x, gamma, eps)
    }

    fn matvec_f32(&self, w: &[f32], x: &[f32], d_out: usize) -> Vec<f32> {
        matvec_f32(w, x, d_out)
    }

    fn softplus_sqrt(&self, logits: &[f32]) -> Vec<f32> {
        softplus_sqrt(logits)
    }

    fn router_finalize(&self, probs: &[f32], bias: &[f32], k: usize) -> (Vec<usize>, Vec<f32>) {
        router_finalize(probs, bias, k)
    }

    fn moe_routed_step(
        &self,
        _layer_idx: u32,
        x: &[f32],
        selected: &[usize],
        weights: &[f32],
        experts_w_gate: &[f32],
        experts_w_up: &[f32],
        experts_w_down: &[f32],
        d_ffn: usize,
    ) -> Vec<f32> {
        moe_routed_step(
            x,
            selected,
            weights,
            experts_w_gate,
            experts_w_up,
            experts_w_down,
            d_ffn,
        )
    }
}

/// Pure math-reference dispatcher for tests and quality experiments.
///
/// This intentionally omits antirez compatibility guards when the underlying
/// operation has a clearer mathematical definition. Today that distinction is
/// visible in router normalization: `CpuDispatcher` preserves antirez's tiny
/// denominator floor, while this dispatcher normalizes by the actual positive
/// selected-probability sum.
#[derive(Debug, Default, Clone, Copy)]
pub struct MathReferenceDispatcher;

impl KernelDispatcher for MathReferenceDispatcher {
    fn rms_norm(&self, x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
        rms_norm(x, gamma, eps)
    }

    fn matvec_f32(&self, w: &[f32], x: &[f32], d_out: usize) -> Vec<f32> {
        matvec_f32(w, x, d_out)
    }

    fn softplus_sqrt(&self, logits: &[f32]) -> Vec<f32> {
        softplus_sqrt(logits)
    }

    fn router_finalize(&self, probs: &[f32], bias: &[f32], k: usize) -> (Vec<usize>, Vec<f32>) {
        router_finalize_math_reference(probs, bias, k)
    }

    fn moe_routed_step(
        &self,
        _layer_idx: u32,
        x: &[f32],
        selected: &[usize],
        weights: &[f32],
        experts_w_gate: &[f32],
        experts_w_up: &[f32],
        experts_w_down: &[f32],
        d_ffn: usize,
    ) -> Vec<f32> {
        moe_routed_step(
            x,
            selected,
            weights,
            experts_w_gate,
            experts_w_up,
            experts_w_down,
            d_ffn,
        )
    }
}

// ---------------------------------------------------------------------------
// RecordingDispatcher — wraps another dispatcher, counts/logs each call.
// ---------------------------------------------------------------------------

/// Summary of kernel-method invocations seen by a `RecordingDispatcher`.
/// Per decode_step the engine emits a fixed pattern of calls; comparing
/// these counts across Cpu/Metal dispatchers proves both paths exercise
/// the same trait surface.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct DispatchCounts {
    pub rms_norm: usize,
    pub matvec_f32: usize,
    pub softplus_sqrt: usize,
    pub router_finalize: usize,
    pub moe_routed_step: usize,
}

impl DispatchCounts {
    pub fn total(&self) -> usize {
        self.rms_norm
            + self.matvec_f32
            + self.softplus_sqrt
            + self.router_finalize
            + self.moe_routed_step
    }
}

/// Wraps another dispatcher and records every call. Useful for:
/// (a) layer-by-layer oracle comparison between Cpu/Metal back ends,
/// (b) confirming the decode pipeline exercises exactly the expected
///     trait methods.
pub struct RecordingDispatcher<'a, D: KernelDispatcher> {
    inner: &'a D,
    counts: std::cell::RefCell<DispatchCounts>,
}

impl<'a, D: KernelDispatcher> RecordingDispatcher<'a, D> {
    pub fn new(inner: &'a D) -> Self {
        Self {
            inner,
            counts: std::cell::RefCell::new(DispatchCounts::default()),
        }
    }

    pub fn counts(&self) -> DispatchCounts {
        *self.counts.borrow()
    }
}

impl<'a, D: KernelDispatcher> KernelDispatcher for RecordingDispatcher<'a, D> {
    fn rms_norm(&self, x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
        self.counts.borrow_mut().rms_norm += 1;
        self.inner.rms_norm(x, gamma, eps)
    }

    fn matvec_f32(&self, w: &[f32], x: &[f32], d_out: usize) -> Vec<f32> {
        self.counts.borrow_mut().matvec_f32 += 1;
        self.inner.matvec_f32(w, x, d_out)
    }

    fn softplus_sqrt(&self, logits: &[f32]) -> Vec<f32> {
        self.counts.borrow_mut().softplus_sqrt += 1;
        self.inner.softplus_sqrt(logits)
    }

    fn router_finalize(&self, probs: &[f32], bias: &[f32], k: usize) -> (Vec<usize>, Vec<f32>) {
        self.counts.borrow_mut().router_finalize += 1;
        self.inner.router_finalize(probs, bias, k)
    }

    fn moe_routed_step(
        &self,
        layer_idx: u32,
        x: &[f32],
        selected: &[usize],
        weights: &[f32],
        experts_w_gate: &[f32],
        experts_w_up: &[f32],
        experts_w_down: &[f32],
        d_ffn: usize,
    ) -> Vec<f32> {
        self.counts.borrow_mut().moe_routed_step += 1;
        self.inner.moe_routed_step(
            layer_idx,
            x,
            selected,
            weights,
            experts_w_gate,
            experts_w_up,
            experts_w_down,
            d_ffn,
        )
    }
}

// ---------------------------------------------------------------------------
// TracingDispatcher — captures per-call activation vectors for replay.
// ---------------------------------------------------------------------------

/// One recorded dispatch call. Each variant carries the bare minimum needed
/// to replay or validate the call: activation vectors + scalars. Static
/// weight tensors (w_attn, expert stacks) are passed in by reference each
/// call but are identical across CPU/Metal back ends; we capture only the
/// data that varies per dispatch.
///
/// The Mac validation harness will run `decode_step_with(&MetalDispatcher,
/// ...)` under a TracingDispatcher and compare the resulting `Vec<TraceEvent>`
/// against the Cpu-side trace, event by event.
#[derive(Debug, Clone)]
pub enum TraceEvent {
    RmsNorm {
        x: Vec<f32>,
        gamma_len: usize,
        eps: f32,
        output: Vec<f32>,
    },
    MatVecF32 {
        x: Vec<f32>,
        w_len: usize,
        d_out: usize,
        output: Vec<f32>,
    },
    SoftplusSqrt {
        logits: Vec<f32>,
        output: Vec<f32>,
    },
    RouterFinalize {
        probs: Vec<f32>,
        bias_len: usize,
        k: usize,
        selected: Vec<usize>,
        weights: Vec<f32>,
    },
    MoeRoutedStep {
        layer_idx: u32,
        x: Vec<f32>,
        selected: Vec<usize>,
        weights: Vec<f32>,
        d_ffn: usize,
        output: Vec<f32>,
    },
}

impl TraceEvent {
    /// Short stable identifier for the event variant (handy for assertions).
    pub fn kind(&self) -> &'static str {
        match self {
            TraceEvent::RmsNorm { .. } => "rms_norm",
            TraceEvent::MatVecF32 { .. } => "matvec_f32",
            TraceEvent::SoftplusSqrt { .. } => "softplus_sqrt",
            TraceEvent::RouterFinalize { .. } => "router_finalize",
            TraceEvent::MoeRoutedStep { .. } => "moe_routed_step",
        }
    }

    /// Compare two events for numerical agreement. Inputs must be identical
    /// (the two dispatchers should see the same activations); outputs may
    /// drift within `tol` (max element-wise abs error). Returns the first
    /// mismatch as a human-readable string, or `None` if the events agree.
    pub fn check_close(a: &TraceEvent, b: &TraceEvent, tol: f32) -> Option<String> {
        if a.kind() != b.kind() {
            return Some(format!("kind drift: {} vs {}", a.kind(), b.kind()));
        }
        let max_abs = |x: &[f32], y: &[f32]| -> f32 {
            x.iter()
                .zip(y.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max)
        };
        match (a, b) {
            (
                TraceEvent::RmsNorm {
                    x: xa, output: oa, ..
                },
                TraceEvent::RmsNorm {
                    x: xb, output: ob, ..
                },
            ) => {
                if xa != xb {
                    return Some("rms_norm: input x drift".into());
                }
                let err = max_abs(oa, ob);
                if err > tol {
                    return Some(format!("rms_norm: output max_abs={err} > {tol}"));
                }
                None
            }
            (
                TraceEvent::MatVecF32 {
                    x: xa,
                    output: oa,
                    d_out: da,
                    ..
                },
                TraceEvent::MatVecF32 {
                    x: xb,
                    output: ob,
                    d_out: db,
                    ..
                },
            ) => {
                if xa != xb || da != db {
                    return Some("matvec_f32: input drift".into());
                }
                let err = max_abs(oa, ob);
                if err > tol {
                    return Some(format!("matvec_f32: output max_abs={err} > {tol}"));
                }
                None
            }
            (
                TraceEvent::SoftplusSqrt {
                    logits: la,
                    output: oa,
                },
                TraceEvent::SoftplusSqrt {
                    logits: lb,
                    output: ob,
                },
            ) => {
                if la != lb {
                    return Some("softplus_sqrt: input drift".into());
                }
                let err = max_abs(oa, ob);
                if err > tol {
                    return Some(format!("softplus_sqrt: output max_abs={err} > {tol}"));
                }
                None
            }
            (
                TraceEvent::RouterFinalize {
                    probs: pa,
                    selected: sa,
                    weights: wa,
                    ..
                },
                TraceEvent::RouterFinalize {
                    probs: pb,
                    selected: sb,
                    weights: wb,
                    ..
                },
            ) => {
                if pa != pb {
                    return Some("router_finalize: input drift".into());
                }
                if sa != sb {
                    return Some(format!("router_finalize: selected drift {sa:?} vs {sb:?}"));
                }
                let err = max_abs(wa, wb);
                if err > tol {
                    return Some(format!("router_finalize: weights max_abs={err} > {tol}"));
                }
                None
            }
            (
                TraceEvent::MoeRoutedStep {
                    layer_idx: la,
                    x: xa,
                    selected: sa,
                    weights: wa,
                    output: oa,
                    ..
                },
                TraceEvent::MoeRoutedStep {
                    layer_idx: lb,
                    x: xb,
                    selected: sb,
                    weights: wb,
                    output: ob,
                    ..
                },
            ) => {
                if la != lb {
                    return Some(format!("moe_routed_step: layer_idx drift {la} vs {lb}"));
                }
                if xa != xb || sa != sb || wa != wb {
                    return Some("moe_routed_step: input drift".into());
                }
                let err = max_abs(oa, ob);
                if err > tol {
                    return Some(format!("moe_routed_step: output max_abs={err} > {tol}"));
                }
                None
            }
            _ => unreachable!("kind() match already ruled this out"),
        }
    }
}

/// Compare two traces. Returns `Ok(())` on full agreement; otherwise the
/// `Err` describes the first divergence (index + reason).
pub fn check_traces_close(
    expected: &[TraceEvent],
    actual: &[TraceEvent],
    tol: f32,
) -> Result<(), String> {
    if expected.len() != actual.len() {
        return Err(format!(
            "trace length drift: expected {}, actual {}",
            expected.len(),
            actual.len()
        ));
    }
    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        if let Some(reason) = TraceEvent::check_close(e, a, tol) {
            return Err(format!("event {i} ({}): {reason}", e.kind()));
        }
    }
    Ok(())
}

/// Wraps another dispatcher and captures every call as a `TraceEvent`.
pub struct TracingDispatcher<'a, D: KernelDispatcher> {
    inner: &'a D,
    events: std::cell::RefCell<Vec<TraceEvent>>,
}

impl<'a, D: KernelDispatcher> TracingDispatcher<'a, D> {
    pub fn new(inner: &'a D) -> Self {
        Self {
            inner,
            events: std::cell::RefCell::new(Vec::new()),
        }
    }

    pub fn events(&self) -> Vec<TraceEvent> {
        self.events.borrow().clone()
    }

    pub fn len(&self) -> usize {
        self.events.borrow().len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.borrow().is_empty()
    }
}

impl<'a, D: KernelDispatcher> KernelDispatcher for TracingDispatcher<'a, D> {
    fn rms_norm(&self, x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
        let output = self.inner.rms_norm(x, gamma, eps);
        self.events.borrow_mut().push(TraceEvent::RmsNorm {
            x: x.to_vec(),
            gamma_len: gamma.len(),
            eps,
            output: output.clone(),
        });
        output
    }

    fn matvec_f32(&self, w: &[f32], x: &[f32], d_out: usize) -> Vec<f32> {
        let output = self.inner.matvec_f32(w, x, d_out);
        self.events.borrow_mut().push(TraceEvent::MatVecF32 {
            x: x.to_vec(),
            w_len: w.len(),
            d_out,
            output: output.clone(),
        });
        output
    }

    fn softplus_sqrt(&self, logits: &[f32]) -> Vec<f32> {
        let output = self.inner.softplus_sqrt(logits);
        self.events.borrow_mut().push(TraceEvent::SoftplusSqrt {
            logits: logits.to_vec(),
            output: output.clone(),
        });
        output
    }

    fn router_finalize(&self, probs: &[f32], bias: &[f32], k: usize) -> (Vec<usize>, Vec<f32>) {
        let (selected, weights) = self.inner.router_finalize(probs, bias, k);
        self.events.borrow_mut().push(TraceEvent::RouterFinalize {
            probs: probs.to_vec(),
            bias_len: bias.len(),
            k,
            selected: selected.clone(),
            weights: weights.clone(),
        });
        (selected, weights)
    }

    fn moe_routed_step(
        &self,
        layer_idx: u32,
        x: &[f32],
        selected: &[usize],
        weights: &[f32],
        experts_w_gate: &[f32],
        experts_w_up: &[f32],
        experts_w_down: &[f32],
        d_ffn: usize,
    ) -> Vec<f32> {
        let output = self.inner.moe_routed_step(
            layer_idx,
            x,
            selected,
            weights,
            experts_w_gate,
            experts_w_up,
            experts_w_down,
            d_ffn,
        );
        self.events.borrow_mut().push(TraceEvent::MoeRoutedStep {
            layer_idx,
            x: x.to_vec(),
            selected: selected.to_vec(),
            weights: weights.to_vec(),
            d_ffn,
            output: output.clone(),
        });
        output
    }
}

// ── PerKernelTimingDispatcher ─────────────────────────────────────────
//
// Records per-call latency for each KernelDispatcher method. Useful both
// as a CPU-side sanity floor and as the future Mac per-kernel µs harness
// without touching benchmark scaffolding.

/// Accumulated nanosecond latency per kernel method, separated from call
/// count so callers can derive mean/total trivially.
#[derive(Debug, Default, Clone)]
pub struct DispatchTimings {
    pub rms_norm_ns: u128,
    pub matvec_f32_ns: u128,
    pub softplus_sqrt_ns: u128,
    pub router_finalize_ns: u128,
    pub moe_routed_step_ns: u128,
    pub rms_norm_calls: u64,
    pub matvec_f32_calls: u64,
    pub softplus_sqrt_calls: u64,
    pub router_finalize_calls: u64,
    pub moe_routed_step_calls: u64,
}

impl DispatchTimings {
    pub fn total_ns(&self) -> u128 {
        self.rms_norm_ns
            + self.matvec_f32_ns
            + self.softplus_sqrt_ns
            + self.router_finalize_ns
            + self.moe_routed_step_ns
    }

    pub fn total_calls(&self) -> u64 {
        self.rms_norm_calls
            + self.matvec_f32_calls
            + self.softplus_sqrt_calls
            + self.router_finalize_calls
            + self.moe_routed_step_calls
    }
}

/// Decorator that times every call into the inner dispatcher.
// ---------------------------------------------------------------------------
// Co-execution across two engines
// ---------------------------------------------------------------------------

/// Statistics from a [`CoExecDispatcher`] run.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CoExecStats {
    /// Experts evaluated on the primary engine.
    pub experts_primary: usize,
    /// Experts evaluated on the secondary engine.
    pub experts_secondary: usize,
    /// Calls to `moe_routed_step`.
    pub calls: usize,
}

/// Splits routed-MoE expert evaluation across two engines.
///
/// # Why this is exact
///
/// A routed MoE step computes `Σ_e w_e · f_e(x)` over the selected experts.
/// That is a sum, so partitioning the experts between two engines and adding
/// the partial sums is mathematically identical to evaluating them all in one
/// place. There is no approximation and no cross-expert dependency, which is
/// what makes MoE unusually well suited to heterogeneous execution.
///
/// Floating-point addition is not associative, so the split result can differ
/// from the unsplit one in the last ULP or two — the partial sums are
/// accumulated in a different order. It is not bit-identical, and the tests
/// assert closeness rather than equality.
///
/// # Choosing the share
///
/// `secondary_share` is the fraction of experts routed to the secondary
/// engine. It should reflect measured throughput, not core counts: on an AMD
/// Strix Halo the iGPU streams expert weights at ~93 GB/s and the NPU at
/// ~12.6 GB/s, so a share of ~0.12 balances them. Measured on that hardware,
/// the two engines do not contend for memory — driving the NPU at 9.45 GB/s
/// changed iGPU decode by 0.01 tok/s — so the bandwidths add.
pub struct CoExecDispatcher<'a, P: KernelDispatcher, S: KernelDispatcher> {
    primary: &'a P,
    secondary: &'a S,
    secondary_share: f32,
    stats: std::cell::RefCell<CoExecStats>,
}

impl<'a, P: KernelDispatcher, S: KernelDispatcher> CoExecDispatcher<'a, P, S> {
    /// `secondary_share` is clamped to `0.0..=1.0`.
    pub fn new(primary: &'a P, secondary: &'a S, secondary_share: f32) -> Self {
        Self {
            primary,
            secondary,
            secondary_share: secondary_share.clamp(0.0, 1.0),
            stats: std::cell::RefCell::new(CoExecStats::default()),
        }
    }

    pub fn stats(&self) -> CoExecStats {
        *self.stats.borrow()
    }

    /// How many of `n` experts go to the secondary engine.
    ///
    /// Rounds to nearest, so a share below `0.5/n` sends none — with 6 experts
    /// a 12% share yields 1 (0.72 rounds to 1), matching the bandwidth ratio
    /// about as closely as an integer split permits.
    fn n_secondary(&self, n: usize) -> usize {
        ((n as f32) * self.secondary_share).round().min(n as f32) as usize
    }
}

impl<'a, P: KernelDispatcher, S: KernelDispatcher> KernelDispatcher
    for CoExecDispatcher<'a, P, S>
{
    fn moe_routed_step(
        &self,
        layer_idx: u32,
        x: &[f32],
        selected: &[usize],
        weights: &[f32],
        experts_w_gate: &[f32],
        experts_w_up: &[f32],
        experts_w_down: &[f32],
        d_ffn: usize,
    ) -> Vec<f32> {
        assert_eq!(selected.len(), weights.len());
        let n_sec = self.n_secondary(selected.len());
        let split = selected.len() - n_sec;

        {
            let mut st = self.stats.borrow_mut();
            st.calls += 1;
            st.experts_primary += split;
            st.experts_secondary += n_sec;
        }

        // Primary takes the leading experts, secondary the trailing ones. The
        // engines are independent here; a real deployment would run them
        // concurrently rather than in sequence.
        let mut acc = if split > 0 {
            self.primary.moe_routed_step(
                layer_idx,
                x,
                &selected[..split],
                &weights[..split],
                experts_w_gate,
                experts_w_up,
                experts_w_down,
                d_ffn,
            )
        } else {
            vec![0.0f32; x.len()]
        };

        if n_sec > 0 {
            let tail = self.secondary.moe_routed_step(
                layer_idx,
                x,
                &selected[split..],
                &weights[split..],
                experts_w_gate,
                experts_w_up,
                experts_w_down,
                d_ffn,
            );
            debug_assert_eq!(acc.len(), tail.len());
            for (a, &t) in acc.iter_mut().zip(tail.iter()) {
                *a += t;
            }
        }
        acc
    }

    // Everything else runs on the primary engine: the secondary is only worth
    // using for low-arithmetic-intensity, bandwidth-bound work, which routed
    // expert evaluation is and the rest of the layer is not.
    fn rms_norm(&self, x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
        self.primary.rms_norm(x, gamma, eps)
    }
    fn matvec_f32(&self, w: &[f32], x: &[f32], d_out: usize) -> Vec<f32> {
        self.primary.matvec_f32(w, x, d_out)
    }
    fn softplus_sqrt(&self, logits: &[f32]) -> Vec<f32> {
        self.primary.softplus_sqrt(logits)
    }
    fn router_finalize(&self, probs: &[f32], bias: &[f32], k: usize) -> (Vec<usize>, Vec<f32>) {
        self.primary.router_finalize(probs, bias, k)
    }
}

pub struct PerKernelTimingDispatcher<'a, D: KernelDispatcher> {
    inner: &'a D,
    timings: std::cell::RefCell<DispatchTimings>,
}

impl<'a, D: KernelDispatcher> PerKernelTimingDispatcher<'a, D> {
    pub fn new(inner: &'a D) -> Self {
        Self {
            inner,
            timings: std::cell::RefCell::new(DispatchTimings::default()),
        }
    }

    pub fn timings(&self) -> DispatchTimings {
        self.timings.borrow().clone()
    }
}

impl<'a, D: KernelDispatcher> KernelDispatcher for PerKernelTimingDispatcher<'a, D> {
    fn rms_norm(&self, x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
        let t0 = std::time::Instant::now();
        let out = self.inner.rms_norm(x, gamma, eps);
        let dt = t0.elapsed().as_nanos();
        let mut t = self.timings.borrow_mut();
        t.rms_norm_ns += dt;
        t.rms_norm_calls += 1;
        out
    }

    fn matvec_f32(&self, w: &[f32], x: &[f32], d_out: usize) -> Vec<f32> {
        let t0 = std::time::Instant::now();
        let out = self.inner.matvec_f32(w, x, d_out);
        let dt = t0.elapsed().as_nanos();
        let mut t = self.timings.borrow_mut();
        t.matvec_f32_ns += dt;
        t.matvec_f32_calls += 1;
        out
    }

    fn softplus_sqrt(&self, logits: &[f32]) -> Vec<f32> {
        let t0 = std::time::Instant::now();
        let out = self.inner.softplus_sqrt(logits);
        let dt = t0.elapsed().as_nanos();
        let mut t = self.timings.borrow_mut();
        t.softplus_sqrt_ns += dt;
        t.softplus_sqrt_calls += 1;
        out
    }

    fn router_finalize(&self, probs: &[f32], bias: &[f32], k: usize) -> (Vec<usize>, Vec<f32>) {
        let t0 = std::time::Instant::now();
        let out = self.inner.router_finalize(probs, bias, k);
        let dt = t0.elapsed().as_nanos();
        let mut t = self.timings.borrow_mut();
        t.router_finalize_ns += dt;
        t.router_finalize_calls += 1;
        out
    }

    fn moe_routed_step(
        &self,
        layer_idx: u32,
        x: &[f32],
        selected: &[usize],
        weights: &[f32],
        experts_w_gate: &[f32],
        experts_w_up: &[f32],
        experts_w_down: &[f32],
        d_ffn: usize,
    ) -> Vec<f32> {
        let t0 = std::time::Instant::now();
        let out = self.inner.moe_routed_step(
            layer_idx,
            x,
            selected,
            weights,
            experts_w_gate,
            experts_w_up,
            experts_w_down,
            d_ffn,
        );
        let dt = t0.elapsed().as_nanos();
        let mut t = self.timings.borrow_mut();
        t.moe_routed_step_ns += dt;
        t.moe_routed_step_calls += 1;
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_dispatcher_rms_norm_matches_free_function() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY).
        // Gate-flipping tests take write(); we take read() so the gate cannot change
        // mid-test, while staying parallel with other readers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        let d = CpuDispatcher;
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let gamma = vec![1.0f32; 4];
        let eps = 1e-6;
        let via_trait = d.rms_norm(&x, &gamma, eps);
        let via_free = rms_norm(&x, &gamma, eps);
        assert_eq!(via_trait, via_free);
    }

    #[test]
    fn cpu_dispatcher_layer_qa_rms_batched_matches_sequential() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY).
        // Gate-flipping tests take write(); we take read() so the gate cannot change
        // mid-test, while staying parallel with other readers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        // C.3a.2: the trait default impl runs matvec_f32 +
        // rms_norm sequentially. Output must be bit-identical to the
        // manual two-step composition so that `MetalDispatcher`'s
        // batched override (gated by tail_lm_head_batched_smoke +
        // layer_qa_rms_batched_smoke on macOS) has a stable contract.
        let d = CpuDispatcher;
        let d_in = 16usize;
        let n_lora_q = 8usize;
        let x: Vec<f32> = (0..d_in).map(|i| 0.1 * i as f32 - 0.5).collect();
        let w_qa: Vec<f32> = (0..n_lora_q * d_in)
            .map(|i| 0.05 * ((i as f32 * 0.013).sin()))
            .collect();
        let gamma_q: Vec<f32> = (0..n_lora_q).map(|i| 1.0 + 0.01 * (i as f32)).collect();
        let eps = 1e-5_f32;

        let batched = d.layer_qa_rms_batched(&x, &w_qa, &gamma_q, n_lora_q, eps);
        let qr = d.matvec_f32(&w_qa, &x, n_lora_q);
        let qr_normed = d.rms_norm(&qr, &gamma_q, eps);

        assert_eq!(batched.len(), qr_normed.len());
        for (i, (a, b)) in batched.iter().zip(&qr_normed).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "default impl diverged from sequential at i={i}: batched={a} sequential={b}"
            );
        }
    }

    #[test]
    fn cpu_dispatcher_shared_chain_batched_matches_sequential() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY).
        // Gate-flipping tests take write(); we take read() so the gate cannot change
        // mid-test, while staying parallel with other readers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        // C.3b.2: the trait default impl chains matvec(w_gate) +
        // matvec(w_up) + silu(g)*u + (optional q8_0_round_trip) +
        // matvec(w_down). Output must be bit-identical to the manual
        // composition so the `MetalDispatcher`'s batched override
        // (gated by `tests/shared_chain_batched_smoke.rs` on macOS) has
        // a stable contract on every backend. Asserts both `want_q80`
        // branches independently.
        let d = CpuDispatcher;
        let d_embd = 16usize;
        let sd = 8usize;
        let ffn_norm: Vec<f32> = (0..d_embd)
            .map(|i| ((i as f32 * 0.019).sin() * 0.5 + 0.1).clamp(-2.0, 2.0))
            .collect();
        let w_gate: Vec<f32> = (0..sd * d_embd)
            .map(|i| (i as f32 * 0.007).cos() * 0.2)
            .collect();
        let w_up: Vec<f32> = (0..sd * d_embd)
            .map(|i| (i as f32 * 0.013).sin() * 0.2)
            .collect();
        let w_down: Vec<f32> = (0..d_embd * sd)
            .map(|i| (i as f32 * 0.005).cos() * 0.15)
            .collect();

        for &want_q80 in &[false, true] {
            let batched =
                d.shared_chain_batched(&ffn_norm, &w_gate, &w_up, &w_down, sd as u32, want_q80);
            let g = d.matvec_f32(&w_gate, &ffn_norm, sd);
            let u = d.matvec_f32(&w_up, &ffn_norm, sd);
            let mid: Vec<f32> = g
                .iter()
                .zip(u.iter())
                .map(|(&gi, &ui)| crate::forward::silu(gi) * ui)
                .collect();
            let mid_in = if want_q80 {
                crate::forward::q8_0_round_trip(&mid)
            } else {
                mid
            };
            let manual = d.matvec_f32(&w_down, &mid_in, d_embd);

            assert_eq!(batched.len(), manual.len());
            for (i, (a, b)) in batched.iter().zip(&manual).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "default impl diverged from sequential at want_q80={want_q80} i={i}: batched={a} sequential={b}"
                );
            }
        }
    }

    #[test]
    fn cpu_dispatcher_output_hc_head_batched_matches_sequential() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY).
        // Gate-flipping tests take write(); we take read() so the gate cannot change
        // mid-test, while staying parallel with other readers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        // C.3c.2: the trait default impl runs rms_norm(unit gamma)
        // + matvec_f32 sequentially. Output must be bit-identical to the
        // manual two-step composition so that `MetalDispatcher`'s batched
        // override (gated by `tests/output_hc_head_batched_smoke.rs` on
        // macOS) has a stable contract on every backend.
        let d = CpuDispatcher;
        let n_hc = 4usize;
        let d_embd = 16usize;
        let hc_dim = n_hc * d_embd;
        let inp_hc: Vec<f32> = (0..hc_dim)
            .map(|i| ((i as f32 * 0.013).sin() * 0.5 + 0.07).clamp(-2.0, 2.0))
            .collect();
        let fn_w: Vec<f32> = (0..hc_dim * n_hc)
            .map(|i| ((i as f32 * 0.0071).cos() * 0.18))
            .collect();
        let eps = 1e-5_f32;

        let batched = d.output_hc_head_batched(&inp_hc, &fn_w, n_hc, d_embd, eps);
        let unit_gamma = vec![1.0f32; hc_dim];
        let flat = d.rms_norm(&inp_hc, &unit_gamma, eps);
        let manual = d.matvec_f32(&fn_w, &flat, n_hc);

        assert_eq!(batched.len(), manual.len());
        for (i, (a, b)) in batched.iter().zip(&manual).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "default impl diverged from sequential at i={i}: batched={a} sequential={b}"
            );
        }
    }

    #[test]
    fn cpu_dispatcher_qkv_b_head_rms_batched_matches_sequential() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY).
        // Gate-flipping tests take write(); we take read() so the gate cannot change
        // mid-test, while staying parallel with other readers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        // C.3d.3: the trait default impl runs matvec_f32 +
        // head_rms_norm + matvec_f32 sequentially. Output must be
        // bit-identical to the manual three-step composition so that
        // `MetalDispatcher`'s batched override (gated by
        // `tests/qkv_b_head_rms_batched_smoke.rs` on macOS) has a
        // stable contract on every backend.
        let d = CpuDispatcher;
        let n_head = 4usize;
        let head_dim = 16usize;
        let q_dim = n_head * head_dim;
        let d_qb = 32usize;
        let kv_row = 32usize;
        let d_kv = 32usize;
        let eps = 1e-5_f32;

        let qr: Vec<f32> = (0..d_qb)
            .map(|i| ((i as f32 * 0.017).sin() * 0.42 + 0.03).clamp(-2.0, 2.0))
            .collect();
        let w_q_b: Vec<f32> = (0..q_dim * d_qb)
            .map(|i| ((i as f32 * 0.0053).cos() * 0.21))
            .collect();
        let normed_kv: Vec<f32> = (0..d_kv)
            .map(|i| ((i as f32 * 0.029).cos() * 0.37 + 0.05).clamp(-2.0, 2.0))
            .collect();
        let w_kv: Vec<f32> = (0..kv_row * d_kv)
            .map(|i| ((i as f32 * 0.0083).sin() * 0.17))
            .collect();

        let (q_h, kv_r) = d.qkv_b_head_rms_batched(
            &qr, &w_q_b, q_dim, n_head, head_dim, eps, &normed_kv, &w_kv, kv_row,
        );

        let q_raw = d.matvec_f32(&w_q_b, &qr, q_dim);
        let q_h_manual = d.head_rms_norm(&q_raw, n_head, head_dim, eps);
        let kv_r_manual = d.matvec_f32(&w_kv, &normed_kv, kv_row);

        assert_eq!(q_h.len(), q_h_manual.len());
        for (i, (a, b)) in q_h.iter().zip(&q_h_manual).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "q_heads default impl diverged at i={i}: batched={a} manual={b}"
            );
        }
        assert_eq!(kv_r.len(), kv_r_manual.len());
        for (i, (a, b)) in kv_r.iter().zip(&kv_r_manual).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "kv_raw_row default impl diverged at i={i}: batched={a} manual={b}"
            );
        }
    }

    #[test]
    fn cpu_dispatcher_router_logits_batched_matches_sequential() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY).
        // Gate-flipping tests take write(); we take read() so the gate cannot change
        // mid-test, while staying parallel with other readers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        // C.3e: the trait default impl runs matvec_f32 +
        // softplus_sqrt sequentially. Output must be bit-identical to
        // the manual two-step composition so the Metal override's
        // single-MTLCommandBuffer pack has a stable contract.
        let d = CpuDispatcher;
        let n_experts = 32usize;
        let d_in = 64usize;
        let h_norm: Vec<f32> = (0..d_in)
            .map(|i| ((i as f32 * 0.041).sin() * 0.31 + 0.07).clamp(-2.0, 2.0))
            .collect();
        let w_router: Vec<f32> = (0..n_experts * d_in)
            .map(|i| ((i as f32 * 0.0067).cos() * 0.19))
            .collect();

        let probs = d.router_logits_batched(&w_router, &h_norm, n_experts);
        let logits_manual = d.matvec_f32(&w_router, &h_norm, n_experts);
        let probs_manual = d.softplus_sqrt(&logits_manual);

        assert_eq!(probs.len(), probs_manual.len());
        for (i, (a, b)) in probs.iter().zip(&probs_manual).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "router_logits_batched default impl diverged at i={i}: batched={a} manual={b}"
            );
        }
    }

    #[test]
    fn cpu_dispatcher_head_rms_norm_matches_inline_f32() {
        // C.3d.1: the trait default impl runs the antirez f32
        // chain per head — bit-identical to an inline f32 implementation.
        // Under DS4_HEAD_RMS_F64_FIDELITY=1 the helper at decode_step.rs
        // must take a separate inline f64 path; this test locks the
        // default-OFF semantic only.
        let d = CpuDispatcher;
        let n_head = 4usize;
        let head_dim = 16usize;
        let eps = 1e-5_f32;
        let x: Vec<f32> = (0..n_head * head_dim)
            .map(|i| ((i as f32 * 0.027).sin() * 1.3 + 0.07).clamp(-3.0, 3.0))
            .collect();

        let via_trait = d.head_rms_norm(&x, n_head, head_dim, eps);

        // Manual f32 oracle: same chain as the default impl.
        let mut manual = x.clone();
        let hd = head_dim as f32;
        for h in 0..n_head {
            let chunk = &mut manual[h * head_dim..(h + 1) * head_dim];
            let ss: f32 = chunk.iter().map(|&v| v * v).sum();
            let scale = 1.0 / ((ss / hd) + eps).sqrt();
            for v in chunk.iter_mut() {
                *v *= scale;
            }
        }

        assert_eq!(via_trait.len(), manual.len());
        for (i, (a, b)) in via_trait.iter().zip(&manual).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "default impl diverged from inline f32 at i={i}: via_trait={a} manual={b}"
            );
        }
    }

    #[test]
    fn cpu_dispatcher_matvec_matches_free_function() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY).
        // Gate-flipping tests take write(); we take read() so the gate cannot change
        // mid-test, while staying parallel with other readers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        let d = CpuDispatcher;
        let w = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // 3x2
        let x = vec![2.0f32, 3.0];
        let via_trait = d.matvec_f32(&w, &x, 3);
        let via_free = matvec_f32(&w, &x, 3);
        assert_eq!(via_trait, via_free);
    }

    #[test]
    fn cpu_dispatcher_softplus_sqrt_matches_free_function() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY).
        // Gate-flipping tests take write(); we take read() so the gate cannot change
        // mid-test, while staying parallel with other readers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        let d = CpuDispatcher;
        let logits = vec![-5.0f32, 0.0, 1.5, 25.0];
        assert_eq!(d.softplus_sqrt(&logits), softplus_sqrt(&logits));
    }

    #[test]
    fn cpu_dispatcher_router_finalize_matches_free_function() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY).
        // Gate-flipping tests take write(); we take read() so the gate cannot change
        // mid-test, while staying parallel with other readers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        let d = CpuDispatcher;
        let probs = vec![0.1f32, 0.7, 0.2, 0.4, 0.5];
        let bias = vec![0.0f32; 5];
        let (idx_t, w_t) = d.router_finalize(&probs, &bias, 2);
        let (idx_f, w_f) = router_finalize(&probs, &bias, 2);
        assert_eq!(idx_t, idx_f);
        assert_eq!(w_t, w_f);
    }

    #[test]
    fn math_reference_dispatcher_router_matches_cpu_above_antirez_floor() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY).
        // Gate-flipping tests take write(); we take read() so the gate cannot change
        // mid-test, while staying parallel with other readers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        let cpu = CpuDispatcher;
        let math = MathReferenceDispatcher;
        let probs = vec![0.1f32, 0.7, 0.2, 0.4, 0.5];
        let bias = vec![0.0f32; 5];
        let (idx_cpu, w_cpu) = cpu.router_finalize(&probs, &bias, 3);
        let (idx_math, w_math) = math.router_finalize(&probs, &bias, 3);
        assert_eq!(idx_cpu, idx_math);
        for (a, b) in w_cpu.iter().zip(w_math.iter()) {
            assert!((a - b).abs() < 1e-6, "cpu {a} vs math {b}");
        }
    }

    #[test]
    fn math_reference_dispatcher_router_exposes_tiny_sum_policy() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY).
        // Gate-flipping tests take write(); we take read() so the gate cannot change
        // mid-test, while staying parallel with other readers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        let cpu = CpuDispatcher;
        let math = MathReferenceDispatcher;
        let probs = vec![0.5f32, 0.4, 1e-10, 1e-10];
        let bias = vec![0.0f32, 0.0, 100.0, 100.0];
        let (idx_cpu, w_cpu) = cpu.router_finalize(&probs, &bias, 2);
        let (idx_math, w_math) = math.router_finalize(&probs, &bias, 2);
        assert_eq!(idx_cpu, idx_math);
        let sum_cpu: f32 = w_cpu.iter().sum();
        let sum_math: f32 = w_math.iter().sum();
        assert!(sum_cpu < 1e-3, "cpu parity sum={sum_cpu:e}");
        assert!((sum_math - 1.5).abs() < 1e-6, "math sum={sum_math:e}");
    }

    #[test]
    fn cpu_dispatcher_moe_routed_step_matches_free_function() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY).
        // Gate-flipping tests take write(); we take read() so the gate cannot change
        // mid-test, while staying parallel with other readers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        let d = CpuDispatcher;
        let n_experts = 2;
        let d_in = 3;
        let d_ffn = 3;
        let identity = |k: usize| {
            let mut m = vec![0.0f32; k * k];
            for i in 0..k {
                m[i * k + i] = 1.0;
            }
            m
        };
        let mut gate = Vec::new();
        let mut up = Vec::new();
        let mut down = Vec::new();
        for _ in 0..n_experts {
            gate.extend(identity(d_ffn));
            up.extend(identity(d_ffn));
            down.extend(identity(d_in));
        }
        let x = vec![0.5f32, -0.1, 0.7];
        let selected = vec![0usize, 1];
        let weights = vec![0.6f32, 0.4];
        let via_trait = d.moe_routed_step(0, &x, &selected, &weights, &gate, &up, &down, d_ffn);
        let via_free = moe_routed_step(&x, &selected, &weights, &gate, &up, &down, d_ffn);
        assert_eq!(via_trait, via_free);
    }

    #[test]
    fn recording_dispatcher_counts_single_calls() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY).
        // Gate-flipping tests take write(); we take read() so the gate cannot change
        // mid-test, while staying parallel with other readers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        let inner = CpuDispatcher;
        let rec = RecordingDispatcher::new(&inner);
        let _ = rec.rms_norm(&[1.0, 2.0], &[1.0, 1.0], 1e-6);
        let _ = rec.matvec_f32(&[1.0, 0.0, 0.0, 1.0], &[2.0, 3.0], 2);
        let _ = rec.softplus_sqrt(&[0.5, -1.0]);
        let _ = rec.router_finalize(&[0.4, 0.6, 0.1], &[0.0; 3], 1);
        let counts = rec.counts();
        assert_eq!(counts.rms_norm, 1);
        assert_eq!(counts.matvec_f32, 1);
        assert_eq!(counts.softplus_sqrt, 1);
        assert_eq!(counts.router_finalize, 1);
        assert_eq!(counts.moe_routed_step, 0);
        assert_eq!(counts.total(), 4);
    }

    #[test]
    fn recording_dispatcher_forwards_results_identically() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY).
        // Gate-flipping tests take write(); we take read() so the gate cannot change
        // mid-test, while staying parallel with other readers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        let inner = CpuDispatcher;
        let rec = RecordingDispatcher::new(&inner);
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let gamma = vec![1.0f32; 4];
        let direct = inner.rms_norm(&x, &gamma, 1e-6);
        let recorded = rec.rms_norm(&x, &gamma, 1e-6);
        assert_eq!(direct, recorded);
    }

    // ── TracingDispatcher coverage ──────────────────────────────────────

    #[test]
    fn tracing_dispatcher_records_event_shape_per_call() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY).
        // Gate-flipping tests take write(); we take read() so the gate cannot change
        // mid-test, while staying parallel with other readers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        let inner = CpuDispatcher;
        let tracer = TracingDispatcher::new(&inner);
        let _ = tracer.rms_norm(&[1.0, 2.0, 3.0, 4.0], &[1.0; 4], 1e-6);
        let _ = tracer.matvec_f32(&[1.0, 0.0, 0.0, 1.0], &[5.0, 6.0], 2);
        let _ = tracer.softplus_sqrt(&[0.5, -1.0, 2.0]);

        let events = tracer.events();
        assert_eq!(events.len(), 3);

        match &events[0] {
            TraceEvent::RmsNorm { x, output, eps, .. } => {
                assert_eq!(x.len(), 4);
                assert_eq!(output.len(), 4);
                assert_eq!(*eps, 1e-6);
            }
            other => panic!("expected RmsNorm, got {other:?}"),
        }
        match &events[1] {
            TraceEvent::MatVecF32 {
                x, output, d_out, ..
            } => {
                assert_eq!(x.len(), 2);
                assert_eq!(output.len(), 2);
                assert_eq!(*d_out, 2);
            }
            other => panic!("expected MatVecF32, got {other:?}"),
        }
        match &events[2] {
            TraceEvent::SoftplusSqrt { logits, output } => {
                assert_eq!(logits.len(), 3);
                assert_eq!(output.len(), 3);
            }
            other => panic!("expected SoftplusSqrt, got {other:?}"),
        }
    }

    #[test]
    fn check_traces_close_returns_ok_for_two_cpu_dispatchers() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY).
        // Gate-flipping tests take write(); we take read() so the gate cannot change
        // mid-test, while staying parallel with other readers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        let inner_a = CpuDispatcher;
        let inner_b = CpuDispatcher;
        let tracer_a = TracingDispatcher::new(&inner_a);
        let tracer_b = TracingDispatcher::new(&inner_b);

        for tr in [&tracer_a, &tracer_b] {
            let _ = tr.rms_norm(&[1.0, 2.0, 3.0, 4.0], &[1.0; 4], 1e-6);
            let _ = tr.matvec_f32(&[1.0, 0.0, 0.0, 1.0], &[5.0, 6.0], 2);
            let _ = tr.softplus_sqrt(&[0.5, -1.0, 2.0]);
            let _ = tr.router_finalize(&[0.4, 0.6, 0.1], &[0.0; 3], 1);
        }

        let a = tracer_a.events();
        let b = tracer_b.events();
        assert!(check_traces_close(&a, &b, 1e-7).is_ok());
    }

    #[test]
    fn check_traces_close_catches_length_mismatch() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY).
        // Gate-flipping tests take write(); we take read() so the gate cannot change
        // mid-test, while staying parallel with other readers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        let inner = CpuDispatcher;
        let tr_short = TracingDispatcher::new(&inner);
        let tr_long = TracingDispatcher::new(&inner);

        let _ = tr_short.rms_norm(&[1.0, 2.0], &[1.0; 2], 1e-6);
        let _ = tr_long.rms_norm(&[1.0, 2.0], &[1.0; 2], 1e-6);
        let _ = tr_long.softplus_sqrt(&[0.0]);

        let res = check_traces_close(&tr_short.events(), &tr_long.events(), 1e-6);
        let err = res.unwrap_err();
        assert!(err.contains("length drift"), "got: {err}");
    }

    #[test]
    fn check_traces_close_catches_kind_drift() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY).
        // Gate-flipping tests take write(); we take read() so the gate cannot change
        // mid-test, while staying parallel with other readers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        let inner = CpuDispatcher;
        let tr_a = TracingDispatcher::new(&inner);
        let tr_b = TracingDispatcher::new(&inner);

        let _ = tr_a.rms_norm(&[1.0, 2.0], &[1.0; 2], 1e-6);
        let _ = tr_b.softplus_sqrt(&[1.0, 2.0]);

        let res = check_traces_close(&tr_a.events(), &tr_b.events(), 1e-6);
        let err = res.unwrap_err();
        assert!(err.contains("kind drift"), "got: {err}");
    }

    #[test]
    fn check_traces_close_catches_output_numerical_drift() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY).
        // Gate-flipping tests take write(); we take read() so the gate cannot change
        // mid-test, while staying parallel with other readers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        let inner = CpuDispatcher;
        let real = inner.rms_norm(&[1.0, 2.0, 3.0, 4.0], &[1.0; 4], 1e-6);
        let mut drifted = real.clone();
        drifted[0] += 0.1;

        let event_a = TraceEvent::RmsNorm {
            x: vec![1.0, 2.0, 3.0, 4.0],
            gamma_len: 4,
            eps: 1e-6,
            output: real,
        };
        let event_b = TraceEvent::RmsNorm {
            x: vec![1.0, 2.0, 3.0, 4.0],
            gamma_len: 4,
            eps: 1e-6,
            output: drifted,
        };

        let res = check_traces_close(&[event_a], &[event_b], 1e-5);
        let err = res.unwrap_err();
        assert!(err.contains("rms_norm"), "got: {err}");
        assert!(err.contains("max_abs"), "got: {err}");
    }

    #[test]
    fn check_traces_close_catches_input_drift() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY).
        // Gate-flipping tests take write(); we take read() so the gate cannot change
        // mid-test, while staying parallel with other readers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        let inner = CpuDispatcher;
        let tr_a = TracingDispatcher::new(&inner);
        let tr_b = TracingDispatcher::new(&inner);

        let _ = tr_a.rms_norm(&[1.0, 2.0, 3.0, 4.0], &[1.0; 4], 1e-6);
        let _ = tr_b.rms_norm(&[1.0, 2.0, 3.0, 5.0], &[1.0; 4], 1e-6);

        let res = check_traces_close(&tr_a.events(), &tr_b.events(), 1e-5);
        let err = res.unwrap_err();
        assert!(err.contains("input x drift"), "got: {err}");
    }

    // ── PerKernelTimingDispatcher coverage ────────────────────────────

    #[test]
    fn timing_dispatcher_counts_match_recording_dispatcher() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY).
        // Gate-flipping tests take write(); we take read() so the gate cannot change
        // mid-test, while staying parallel with other readers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        let inner = CpuDispatcher;
        let timing = PerKernelTimingDispatcher::new(&inner);
        for _ in 0..3 {
            let _ = timing.rms_norm(&[1.0, 2.0], &[1.0; 2], 1e-6);
        }
        let _ = timing.matvec_f32(&[1.0, 0.0, 0.0, 1.0], &[2.0, 3.0], 2);
        let _ = timing.softplus_sqrt(&[0.5]);

        let t = timing.timings();
        assert_eq!(t.rms_norm_calls, 3);
        assert_eq!(t.matvec_f32_calls, 1);
        assert_eq!(t.softplus_sqrt_calls, 1);
        assert_eq!(t.router_finalize_calls, 0);
        assert_eq!(t.moe_routed_step_calls, 0);
        assert_eq!(t.total_calls(), 5);
    }

    #[test]
    fn timing_dispatcher_records_nonzero_latency_for_real_work() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY).
        // Gate-flipping tests take write(); we take read() so the gate cannot change
        // mid-test, while staying parallel with other readers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        let inner = CpuDispatcher;
        let timing = PerKernelTimingDispatcher::new(&inner);
        let big_x: Vec<f32> = (0..4096).map(|i| i as f32 * 0.001).collect();
        let big_g: Vec<f32> = vec![1.0; 4096];
        let _ = timing.rms_norm(&big_x, &big_g, 1e-6);

        let t = timing.timings();
        assert!(
            t.rms_norm_ns > 0,
            "expected non-zero ns, got {}",
            t.rms_norm_ns
        );
        assert_eq!(t.total_ns(), t.rms_norm_ns);
    }

    #[test]
    fn timing_dispatcher_results_match_inner_dispatcher() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY).
        // Gate-flipping tests take write(); we take read() so the gate cannot change
        // mid-test, while staying parallel with other readers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        let inner = CpuDispatcher;
        let timing = PerKernelTimingDispatcher::new(&inner);
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let g = vec![1.0f32; 4];
        let direct = inner.rms_norm(&x, &g, 1e-6);
        let timed = timing.rms_norm(&x, &g, 1e-6);
        assert_eq!(direct, timed);
    }

    // ── Decorator composition (Recording⊕Tracing⊕Timing) ───────────────

    #[test]
    fn all_three_decorators_compose() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY).
        // Gate-flipping tests take write(); we take read() so the gate cannot change
        // mid-test, while staying parallel with other readers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        // Triple-stacked decorators must all observe the same calls,
        // forward results unchanged, and not interfere with each other.
        let inner = CpuDispatcher;
        let timing = PerKernelTimingDispatcher::new(&inner);
        let tracing = TracingDispatcher::new(&timing);
        let recording = RecordingDispatcher::new(&tracing);

        let _ = recording.rms_norm(&[1.0, 2.0, 3.0, 4.0], &[1.0; 4], 1e-6);
        let _ = recording.matvec_f32(&[1.0, 0.0, 0.0, 1.0], &[5.0, 6.0], 2);
        let _ = recording.softplus_sqrt(&[0.5, -1.0]);
        let _ = recording.router_finalize(&[0.4, 0.6, 0.1], &[0.0; 3], 2);

        let counts = recording.counts();
        assert_eq!(counts.total(), 4);
        assert_eq!(counts.rms_norm, 1);
        assert_eq!(counts.matvec_f32, 1);
        assert_eq!(counts.softplus_sqrt, 1);
        assert_eq!(counts.router_finalize, 1);

        let events = tracing.events();
        assert_eq!(events.len(), 4);
        assert_eq!(events[0].kind(), "rms_norm");
        assert_eq!(events[3].kind(), "router_finalize");

        let t = timing.timings();
        assert_eq!(t.total_calls(), 4);
        assert_eq!(t.rms_norm_calls, 1);
        assert_eq!(t.matvec_f32_calls, 1);
    }

    #[test]
    fn composed_result_matches_inner() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY).
        // Gate-flipping tests take write(); we take read() so the gate cannot change
        // mid-test, while staying parallel with other readers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        // The result returned through the full decorator stack must be
        // bit-identical to the inner dispatcher's direct output.
        let inner = CpuDispatcher;
        let timing = PerKernelTimingDispatcher::new(&inner);
        let tracing = TracingDispatcher::new(&timing);
        let recording = RecordingDispatcher::new(&tracing);

        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let gamma = vec![1.0f32; 4];

        let direct = inner.rms_norm(&x, &gamma, 1e-6);
        let stacked = recording.rms_norm(&x, &gamma, 1e-6);
        assert_eq!(direct, stacked);
    }

    // -- co-execution -------------------------------------------------------

    /// Build a small but non-trivial routed-MoE problem.
    fn coexec_fixture() -> (Vec<f32>, Vec<usize>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, usize) {
        let d_in = 16usize;
        let d_ffn = 8usize;
        let n_experts = 6usize;
        // deterministic pseudo-random, no rng dependency
        let g = |i: usize| ((i as f32 * 12.9898).sin() * 43758.547).fract() - 0.5;
        let x: Vec<f32> = (0..d_in).map(g).collect();
        let selected: Vec<usize> = (0..n_experts).collect();
        let weights: Vec<f32> = (0..n_experts).map(|i| 0.1 + 0.03 * i as f32).collect();
        let n_gu = n_experts * d_ffn * d_in;
        let n_dn = n_experts * d_in * d_ffn;
        let wg: Vec<f32> = (0..n_gu).map(|i| g(i + 7)).collect();
        let wu: Vec<f32> = (0..n_gu).map(|i| g(i + 101)).collect();
        let wd: Vec<f32> = (0..n_dn).map(|i| g(i + 907)).collect();
        (x, selected, weights, wg, wu, wd, d_ffn)
    }

    /// Splitting experts across two engines must reproduce the unsplit result.
    ///
    /// A routed MoE step is a sum over experts, so any partition of that sum is
    /// mathematically identical. It is not *bit*-identical: the partial sums
    /// accumulate in a different order and float addition is not associative,
    /// hence the tolerance rather than `assert_eq!`.
    #[test]
    fn coexec_split_matches_unsplit_for_every_share() {
        let (x, selected, weights, wg, wu, wd, d_ffn) = coexec_fixture();
        let cpu = CpuDispatcher;
        let reference =
            cpu.moe_routed_step(0, &x, &selected, &weights, &wg, &wu, &wd, d_ffn);

        for share in [0.0f32, 1.0 / 6.0, 0.12, 0.5, 5.0 / 6.0, 1.0] {
            let co = CoExecDispatcher::new(&cpu, &cpu, share);
            let got = co.moe_routed_step(0, &x, &selected, &weights, &wg, &wu, &wd, d_ffn);
            assert_eq!(got.len(), reference.len());
            for (i, (&a, &b)) in got.iter().zip(reference.iter()).enumerate() {
                assert!(
                    (a - b).abs() <= 1e-5 * b.abs().max(1.0),
                    "share {share}: element {i} diverged: {a} vs {b}"
                );
            }
            let st = co.stats();
            assert_eq!(st.experts_primary + st.experts_secondary, selected.len());
            assert_eq!(st.calls, 1);
        }
    }

    /// The expert split should track the measured bandwidth ratio, not core
    /// counts. On Strix Halo the iGPU streams ~93 GB/s and the NPU ~12.6 GB/s,
    /// so ~12% of six experts is one.
    #[test]
    fn coexec_share_maps_to_expert_counts() {
        let cpu = CpuDispatcher;
        let cases = [(0.0f32, 0usize), (0.12, 1), (0.5, 3), (1.0, 6)];
        for (share, want_secondary) in cases {
            let co = CoExecDispatcher::new(&cpu, &cpu, share);
            assert_eq!(
                co.n_secondary(6),
                want_secondary,
                "share {share} should route {want_secondary} of 6 experts to the secondary engine"
            );
        }
    }

    /// Out-of-range shares are clamped rather than panicking or over-routing.
    #[test]
    fn coexec_share_is_clamped() {
        let cpu = CpuDispatcher;
        assert_eq!(CoExecDispatcher::new(&cpu, &cpu, -1.0).n_secondary(6), 0);
        assert_eq!(CoExecDispatcher::new(&cpu, &cpu, 9.0).n_secondary(6), 6);
    }
}
