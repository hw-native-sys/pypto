// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! DS4 V4 Flash MTP (Multi-Token Prediction) drafter weights loader.
//!
//! antirez publishes a small auxiliary GGUF (~3.5 GB) alongside the main DS4
//! V4 Flash model. The MTP file contains a one-layer drafter network used
//! for speculative decoding: given the last emitted token + the previous
//! token's HC residual, it predicts the NEXT token's logits, producing a
//! cheap K-position draft that the main model then verifies.
//!
//! ## File contents
//!
//! 31 tensors with `mtp.0.*.weight` (+1 bias) name prefix. Two groups:
//!
//! **MTP-specific** (8 tensors): embedding-mixer head + final norm + output
//! HC head.
//!
//! ```text
//!   mtp.0.e_proj.weight        Q8_0  [n_embd, n_embd]   token-embed → mtp_input
//!   mtp.0.h_proj.weight        Q8_0  [n_embd, n_embd]   prev_hc-row → mtp_input
//!   mtp.0.enorm.weight         F32   [n_embd]
//!   mtp.0.hnorm.weight         F32   [n_embd]
//!   mtp.0.norm.weight          F32   [n_embd]           final norm before output
//!   mtp.0.hc_head_base.weight  F32   [n_hc]
//!   mtp.0.hc_head_fn.weight    F16   [hc_dim, n_hc]     final HC head
//!   mtp.0.hc_head_scale.weight F32   [1]
//! ```
//!
//! **Embedded layer block** (23 tensors, +1 bias): a full DS4 attn+ffn
//! decode layer in the same shape as `attn_dispatch::AttnLayerWeights`.
//! Same names as `blk.N.*` in the main model but with `mtp.0.*` prefix.
//!
//! Reference: antirez `mtp_weights_bind` in `ds4.c:2489` and
//! `mtp_weights_validate_layout` (ds4.c:2207).
//!
//! ## Usage
//!
//! ```ignore
//! use engine_core::mtp::MtpWeights;
//! let mtp = MtpWeights::from_path("path/to/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf")?;
//! eprintln!("MTP block: n_experts={} d_ffn={}", mtp.n_experts, mtp.d_ffn);
//! ```

use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};

use crate::gguf::{GgmlType, GgufFile, TensorInfo};

/// Expected DS4 V4 Flash MTP architecture constants (mirror of antirez
/// `DS4_N_*` in `ds4.c`).
pub mod consts {
    // Mirror antirez DS4_N_* in ds4.c:86-97 — DS4 V4 Flash production constants.
    pub const N_EMBD: u64 = 4096;
    pub const N_HC: u64 = 4;
    pub const N_HEAD: u64 = 64;
    pub const N_HEAD_DIM: u64 = 512;
    pub const N_OUT_GROUP: u64 = 8; // DS4_N_OUT_GROUP
    pub const N_LORA_Q: u64 = 1024; // DS4_N_LORA_Q
    pub const N_LORA_O: u64 = 1024; // DS4_N_LORA_O
    pub const N_EXPERT: u64 = 256;
    pub const N_FF_EXP: u64 = 2048;
}

/// A located tensor in the MTP GGUF: name, info, and absolute offset of its
/// data in the mmap. Callers dequantize on demand.
#[derive(Clone)]
pub struct MtpTensor {
    pub info: TensorInfo,
    /// Absolute byte offset (tensor_data_offset + relative offset) — ready
    /// to feed into `LayerViews::dequant_*` style callers.
    pub abs_offset: u64,
}

impl MtpTensor {
    fn new(info: TensorInfo, tensor_data_offset: u64) -> Self {
        let abs_offset = tensor_data_offset + info.offset;
        Self { info, abs_offset }
    }

    /// Phase 3 Step 4 — raw GGUF bytes for this tensor as a borrow into the
    /// MTP file's mmap. Caller uses this for Q8_0 paths where
    /// `BatchScope::weight_q8_0_raw` consumes block-packed bytes directly.
    ///
    /// For Q4_K / IQ2_XXS expert tensors (loaded via
    /// `load_mtp_expert_weights` into `state.expert_weights`) you don't
    /// need this — the dispatcher pulls bytes through `QuantTensor.bytes`
    /// at that point.
    pub fn raw_bytes<'a>(&self, mtp_gguf_bytes: &'a [u8]) -> Result<&'a [u8]> {
        let n_bytes = self.byte_size()? as usize;
        let start = self.abs_offset as usize;
        let end = start
            .checked_add(n_bytes)
            .ok_or_else(|| anyhow!("MtpTensor {}: byte range overflow", self.info.name))?;
        if end > mtp_gguf_bytes.len() {
            bail!(
                "MtpTensor {}: byte range [{}, {}) exceeds GGUF size {}",
                self.info.name,
                start,
                end,
                mtp_gguf_bytes.len()
            );
        }
        Ok(&mtp_gguf_bytes[start..end])
    }

    /// Phase 3 Step 4 — dequantize this tensor to f32. Handles F32, F16,
    /// BF16, and Q8_0 inline; bails for the heavier MoE quants (Q4_K,
    /// IQ2_XXS, Q2_K) which are loaded directly into Metal buffers via
    /// `MetalDispatcher::load_mtp_expert_weights`.
    pub fn dequant_f32(&self, mtp_gguf_bytes: &[u8]) -> Result<Vec<f32>> {
        let raw = self.raw_bytes(mtp_gguf_bytes)?;
        let n = self.n_elements() as usize;
        match self.info.ttype {
            GgmlType::F32 => {
                anyhow::ensure!(
                    raw.len() == n * 4,
                    "MtpTensor {}: F32 bytes {} != 4*{}",
                    self.info.name,
                    raw.len(),
                    n
                );
                let mut out = vec![0.0f32; n];
                for (i, c) in raw.chunks_exact(4).enumerate() {
                    out[i] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                }
                Ok(out)
            }
            GgmlType::F16 => {
                anyhow::ensure!(
                    raw.len() == n * 2,
                    "MtpTensor {}: F16 bytes {} != 2*{}",
                    self.info.name,
                    raw.len(),
                    n
                );
                let mut out = vec![0.0f32; n];
                for (i, c) in raw.chunks_exact(2).enumerate() {
                    out[i] = f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]]));
                }
                Ok(out)
            }
            GgmlType::BF16 => {
                anyhow::ensure!(
                    raw.len() == n * 2,
                    "MtpTensor {}: BF16 bytes {} != 2*{}",
                    self.info.name,
                    raw.len(),
                    n
                );
                let mut out = vec![0.0f32; n];
                for (i, c) in raw.chunks_exact(2).enumerate() {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    out[i] = f32::from_bits((bits as u32) << 16);
                }
                Ok(out)
            }
            GgmlType::Q8_0 => {
                anyhow::ensure!(
                    n % 32 == 0,
                    "MtpTensor {}: Q8_0 n_elems {} not %32",
                    self.info.name,
                    n
                );
                let n_blocks = n / 32;
                anyhow::ensure!(
                    raw.len() == n_blocks * 34,
                    "MtpTensor {}: Q8_0 bytes {} != 34*{}",
                    self.info.name,
                    raw.len(),
                    n_blocks
                );
                let mut out = Vec::with_capacity(n);
                for ib in 0..n_blocks {
                    let blk = &raw[ib * 34..(ib + 1) * 34];
                    let d = f16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                    for k in 0..32 {
                        let q = blk[2 + k] as i8;
                        out.push(d * q as f32);
                    }
                }
                Ok(out)
            }
            other => bail!(
                "MtpTensor {}: dequant_f32 doesn't handle {:?} — for routed-expert quants \
                 (Q4_K/IQ2_XXS/Q2_K), load via MetalDispatcher::load_mtp_expert_weights",
                self.info.name,
                other
            ),
        }
    }

    /// Element count = product of dimensions.
    pub fn n_elements(&self) -> u64 {
        self.info.dims.iter().product()
    }

    /// Storage byte size for this tensor in the GGUF.
    fn byte_size(&self) -> Result<u64> {
        let n = self.n_elements();
        match self.info.ttype {
            GgmlType::F32 => Ok(n * 4),
            GgmlType::F16 | GgmlType::BF16 => Ok(n * 2),
            GgmlType::Q8_0 => {
                anyhow::ensure!(n % 32 == 0, "Q8_0 n_elems {} not %32", n);
                Ok((n / 32) * 34)
            }
            GgmlType::Q4_K | GgmlType::Q2_K => {
                anyhow::ensure!(n % 256 == 0, "Q4_K/Q2_K n_elems {} not %256", n);
                let block_bytes = if self.info.ttype == GgmlType::Q4_K {
                    144
                } else {
                    84
                };
                Ok((n / 256) * block_bytes)
            }
            GgmlType::IQ2_XXS => {
                anyhow::ensure!(n % 256 == 0, "IQ2_XXS n_elems {} not %256", n);
                Ok((n / 256) * 66)
            }
            other => bail!("byte_size: unhandled ttype {:?}", other),
        }
    }
}

/// IEEE 754 binary16 → binary32. Local copy of LayerViews' f16_bits_to_f32.
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 0x1;
    let exp = (bits >> 10) & 0x1f;
    let mant = bits & 0x3ff;
    let f = if exp == 0 {
        if mant == 0 {
            0.0
        } else {
            (mant as f32) * 2f32.powi(-24)
        }
    } else if exp == 31 {
        if mant == 0 {
            f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        ((mant as f32 + 1024.0) * 2f32.powi(exp as i32 - 25)) as f32
    };
    if sign != 0 {
        -f
    } else {
        f
    }
}

/// The full MTP weight table. Owned `MtpTensor` views into the mmap; the
/// caller holds `gguf.bytes` alive and dequantizes on demand.
#[derive(Clone)]
pub struct MtpWeights {
    // MTP-specific (8 tensors).
    pub e_proj: MtpTensor,
    pub h_proj: MtpTensor,
    pub enorm: MtpTensor,
    pub hnorm: MtpTensor,
    pub norm: MtpTensor,
    pub hc_head_base: MtpTensor,
    pub hc_head_fn: MtpTensor,
    pub hc_head_scale: MtpTensor,

    // Embedded layer block (23 + 1 bias).
    pub hc_attn_fn: MtpTensor,
    pub hc_attn_scale: MtpTensor,
    pub hc_attn_base: MtpTensor,
    pub attn_norm: MtpTensor,
    pub attn_q_a: MtpTensor,
    pub attn_q_a_norm: MtpTensor,
    pub attn_q_b: MtpTensor,
    pub attn_kv: MtpTensor,
    pub attn_kv_a_norm: MtpTensor,
    pub attn_sinks: MtpTensor,
    pub attn_output_a: MtpTensor,
    pub attn_output_b: MtpTensor,
    pub hc_ffn_fn: MtpTensor,
    pub hc_ffn_scale: MtpTensor,
    pub hc_ffn_base: MtpTensor,
    pub ffn_norm: MtpTensor,
    pub ffn_gate_inp: MtpTensor,
    pub ffn_exp_probs_b: MtpTensor,
    pub ffn_gate_exps: MtpTensor,
    pub ffn_up_exps: MtpTensor,
    pub ffn_down_exps: MtpTensor,
    pub ffn_gate_shexp: MtpTensor,
    pub ffn_up_shexp: MtpTensor,
    pub ffn_down_shexp: MtpTensor,

    // Derived metadata from the experts tensor (gate/up/down share n_expert,
    // d_in, d_ffn — validated for consistency).
    pub n_experts: u64,
    pub d_ffn: u64,
}

impl MtpWeights {
    /// Open an MTP GGUF, locate all 32 tensors by canonical name, and
    /// validate per-tensor shape + type against the antirez layout.
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let gguf = GgufFile::open(path.as_ref())
            .with_context(|| format!("MTP GGUF open: {}", path.as_ref().display()))?;
        Self::from_gguf(&gguf)
    }

    /// Parse a previously-opened MTP GGUF.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let mut by_name: std::collections::HashMap<&str, &TensorInfo> =
            std::collections::HashMap::with_capacity(gguf.tensors.len());
        for t in &gguf.tensors {
            by_name.insert(t.name.as_str(), t);
        }

        let find = |name: &str| -> Result<MtpTensor> {
            let info = by_name
                .get(name)
                .copied()
                .cloned()
                .ok_or_else(|| anyhow!("MTP GGUF missing required tensor: {}", name))?;
            Ok(MtpTensor::new(info, gguf.tensor_data_offset))
        };

        let w = Self {
            e_proj: find("mtp.0.e_proj.weight")?,
            h_proj: find("mtp.0.h_proj.weight")?,
            enorm: find("mtp.0.enorm.weight")?,
            hnorm: find("mtp.0.hnorm.weight")?,
            norm: find("mtp.0.norm.weight")?,
            hc_head_base: find("mtp.0.hc_head_base.weight")?,
            hc_head_fn: find("mtp.0.hc_head_fn.weight")?,
            hc_head_scale: find("mtp.0.hc_head_scale.weight")?,
            hc_attn_fn: find("mtp.0.hc_attn_fn.weight")?,
            hc_attn_scale: find("mtp.0.hc_attn_scale.weight")?,
            hc_attn_base: find("mtp.0.hc_attn_base.weight")?,
            attn_norm: find("mtp.0.attn_norm.weight")?,
            attn_q_a: find("mtp.0.attn_q_a.weight")?,
            attn_q_a_norm: find("mtp.0.attn_q_a_norm.weight")?,
            attn_q_b: find("mtp.0.attn_q_b.weight")?,
            attn_kv: find("mtp.0.attn_kv.weight")?,
            attn_kv_a_norm: find("mtp.0.attn_kv_a_norm.weight")?,
            attn_sinks: find("mtp.0.attn_sinks.weight")?,
            attn_output_a: find("mtp.0.attn_output_a.weight")?,
            attn_output_b: find("mtp.0.attn_output_b.weight")?,
            hc_ffn_fn: find("mtp.0.hc_ffn_fn.weight")?,
            hc_ffn_scale: find("mtp.0.hc_ffn_scale.weight")?,
            hc_ffn_base: find("mtp.0.hc_ffn_base.weight")?,
            ffn_norm: find("mtp.0.ffn_norm.weight")?,
            ffn_gate_inp: find("mtp.0.ffn_gate_inp.weight")?,
            ffn_exp_probs_b: find("mtp.0.exp_probs_b.bias")?,
            ffn_gate_exps: find("mtp.0.ffn_gate_exps.weight")?,
            ffn_up_exps: find("mtp.0.ffn_up_exps.weight")?,
            ffn_down_exps: find("mtp.0.ffn_down_exps.weight")?,
            ffn_gate_shexp: find("mtp.0.ffn_gate_shexp.weight")?,
            ffn_up_shexp: find("mtp.0.ffn_up_shexp.weight")?,
            ffn_down_shexp: find("mtp.0.ffn_down_shexp.weight")?,

            n_experts: 0,
            d_ffn: 0,
        };

        // Validate per-tensor layouts against antirez expectations
        // (ds4.c:2207 `mtp_weights_validate_layout`).
        Self::validate(&w)?;

        // Routed-expert tensor shape: [n_experts, d_inner, d_outer]. For
        // gate/up the dims are [n_experts, d_ffn, n_embd], for down it's
        // [n_experts, n_embd, d_ffn].
        let n_experts = w.ffn_gate_exps.info.dims[2];
        let d_ffn = w.ffn_gate_exps.info.dims[1];

        if w.ffn_up_exps.info.dims != w.ffn_gate_exps.info.dims {
            bail!(
                "MTP: ffn_up_exps dims {:?} != ffn_gate_exps dims {:?}",
                w.ffn_up_exps.info.dims,
                w.ffn_gate_exps.info.dims
            );
        }
        if w.ffn_down_exps.info.dims[0] != consts::N_FF_EXP
            || w.ffn_down_exps.info.dims[1] != consts::N_EMBD
            || w.ffn_down_exps.info.dims[2] != n_experts
        {
            bail!(
                "MTP: ffn_down_exps dims {:?} unexpected (want [{}, {}, {}])",
                w.ffn_down_exps.info.dims,
                consts::N_FF_EXP,
                consts::N_EMBD,
                n_experts
            );
        }
        if w.ffn_gate_exps.info.ttype != w.ffn_up_exps.info.ttype {
            bail!(
                "MTP routed gate/up experts use different quant types: gate={:?} up={:?}",
                w.ffn_gate_exps.info.ttype,
                w.ffn_up_exps.info.ttype
            );
        }

        Ok(MtpWeights {
            n_experts,
            d_ffn,
            ..w
        })
    }

    /// Per-tensor shape + ttype assertions, mirroring antirez
    /// `mtp_weights_validate_layout` (ds4.c:2207).
    fn validate(w: &Self) -> Result<()> {
        let hc_dim = consts::N_EMBD * consts::N_HC;
        let hc_mix_dim = 2 * consts::N_HC + consts::N_HC * consts::N_HC;
        let q_dim = consts::N_HEAD * consts::N_HEAD_DIM;
        let out_low_dim = consts::N_OUT_GROUP * consts::N_LORA_O;

        check(&w.hc_head_base, GgmlType::F32, &[consts::N_HC])?;
        check_plain(&w.hc_head_fn, &[hc_dim, consts::N_HC])?;
        check(&w.hc_head_scale, GgmlType::F32, &[1])?;
        check(&w.e_proj, GgmlType::Q8_0, &[consts::N_EMBD, consts::N_EMBD])?;
        check(&w.h_proj, GgmlType::Q8_0, &[consts::N_EMBD, consts::N_EMBD])?;
        check(&w.enorm, GgmlType::F32, &[consts::N_EMBD])?;
        check(&w.hnorm, GgmlType::F32, &[consts::N_EMBD])?;
        check(&w.norm, GgmlType::F32, &[consts::N_EMBD])?;

        check_plain(&w.hc_attn_fn, &[hc_dim, hc_mix_dim])?;
        check(&w.hc_attn_scale, GgmlType::F32, &[3])?;
        check(&w.hc_attn_base, GgmlType::F32, &[hc_mix_dim])?;
        check(&w.attn_norm, GgmlType::F32, &[consts::N_EMBD])?;
        check(
            &w.attn_q_a,
            GgmlType::Q8_0,
            &[consts::N_EMBD, consts::N_LORA_Q],
        )?;
        check(&w.attn_q_a_norm, GgmlType::F32, &[consts::N_LORA_Q])?;
        check(&w.attn_q_b, GgmlType::Q8_0, &[consts::N_LORA_Q, q_dim])?;
        check(
            &w.attn_kv,
            GgmlType::Q8_0,
            &[consts::N_EMBD, consts::N_HEAD_DIM],
        )?;
        check(&w.attn_kv_a_norm, GgmlType::F32, &[consts::N_HEAD_DIM])?;
        check(&w.attn_sinks, GgmlType::F32, &[consts::N_HEAD])?;
        check(
            &w.attn_output_a,
            GgmlType::Q8_0,
            &[
                consts::N_HEAD_DIM * (consts::N_HEAD / consts::N_OUT_GROUP),
                out_low_dim,
            ],
        )?;
        check(
            &w.attn_output_b,
            GgmlType::Q8_0,
            &[out_low_dim, consts::N_EMBD],
        )?;

        check_plain(&w.hc_ffn_fn, &[hc_dim, hc_mix_dim])?;
        check(&w.hc_ffn_scale, GgmlType::F32, &[3])?;
        check(&w.hc_ffn_base, GgmlType::F32, &[hc_mix_dim])?;
        check(&w.ffn_norm, GgmlType::F32, &[consts::N_EMBD])?;
        check_plain(&w.ffn_gate_inp, &[consts::N_EMBD, consts::N_EXPERT])?;
        check(&w.ffn_exp_probs_b, GgmlType::F32, &[consts::N_EXPERT])?;

        check(
            &w.ffn_gate_shexp,
            GgmlType::Q8_0,
            &[consts::N_EMBD, consts::N_FF_EXP],
        )?;
        check(
            &w.ffn_up_shexp,
            GgmlType::Q8_0,
            &[consts::N_EMBD, consts::N_FF_EXP],
        )?;
        check(
            &w.ffn_down_shexp,
            GgmlType::Q8_0,
            &[consts::N_FF_EXP, consts::N_EMBD],
        )?;

        Ok(())
    }
}

fn check(t: &MtpTensor, want_type: GgmlType, want_dims: &[u64]) -> Result<()> {
    if t.info.ttype != want_type {
        bail!(
            "MTP tensor {} type mismatch: got {:?}, want {:?}",
            t.info.name,
            t.info.ttype,
            want_type
        );
    }
    if t.info.dims != want_dims {
        bail!(
            "MTP tensor {} shape mismatch: got {:?}, want {:?}",
            t.info.name,
            t.info.dims,
            want_dims
        );
    }
    Ok(())
}

/// "Plain" means antirez accepts F16/F32/BF16 (any unquantized type) for
/// these HC-mixer tensors. Match the same flexibility.
fn check_plain(t: &MtpTensor, want_dims: &[u64]) -> Result<()> {
    let plain = matches!(t.info.ttype, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16);
    if !plain {
        bail!(
            "MTP tensor {} expected plain (F32/F16/BF16), got {:?}",
            t.info.name,
            t.info.ttype
        );
    }
    if t.info.dims != want_dims {
        bail!(
            "MTP tensor {} shape mismatch: got {:?}, want {:?}",
            t.info.name,
            t.info.dims,
            want_dims
        );
    }
    Ok(())
}
