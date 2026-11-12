// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! The seam where a dispatcher method stops delegating and starts launching.
//!
//! Everything up to here runs on the CPU by delegation. Making a method run on the NPU needs
//! three things this crate does not have yet:
//!
//!   1. `ascend_sys` cannot build off-device — its build script requires `ASCEND_HOME_PATH`, so
//!      the FFI compiles only where CANN is installed;
//!   2. the symbols are missing anyway: `ascend_sys` carries blas/dvpp/hccl/prof but no
//!      `aclrtMalloc`, `aclrtLaunchKernel` or `aclrtGetMemInfo`;
//!   3. no kernel has ever been launched from Rust here. Every device result in this project came
//!      from a C++ runner with the kernel compiled into the binary.
//!
//! So this module defines the shape of the hole rather than pretending to fill it. A
//! [`KernelLaunch`] implementation owns one method's device round-trip; the dispatcher calls it
//! when present and delegates when absent. That makes the remaining work a single well-defined
//! implementation rather than a rewrite, and lets the marshalling and validation be tested now.
//!
//! # Why [`KernelLaunch::is_device`] exists
//!
//! A host-side fake is necessary to test the plumbing and is dangerous to have around: it makes
//! the pipeline look complete. `is_device` forces every implementation to declare whether it
//! actually reaches an NPU, and `offloaded_by` keys on that — so a fake can never be counted as
//! NPU coverage no matter how much of the path it exercises. This project has repeatedly found
//! claims that drifted from what was running; this is the cheapest available guard against
//! adding another.

use anyhow::{bail, Result};

/// One dispatcher method, executed somewhere other than the delegating CPU path.
pub trait KernelLaunch: Send + Sync {
    /// Identifier for reporting, e.g. `"rms_norm"`.
    fn name(&self) -> &str;

    /// True only if this executes on an NPU. A host fake MUST return false.
    fn is_device(&self) -> bool;

    /// `rms_norm(x, gamma, eps)`. Returns `x.len()` values.
    fn rms_norm(&self, x: &[f32], gamma: &[f32], eps: f32) -> Result<Vec<f32>>;

    /// Q8_0 matvec on SPLIT planes: `codes` is `[d_out x d_in]` int8 in row-block order,
    /// `scales` is `[d_out x d_in]` f32 — the per-32-block scale EXPANDED PER ELEMENT, which is
    /// the `.sd` plane the emitted `q8mv` kernels read (verified against the kernel source: its
    /// scale tensor is viewed `[1 x d_in]` per output row). `x` is `[d_in]`. Returns `[d_out]`.
    ///
    /// Split planes rather than the GGUF's interleaved 34-byte blocks, because that is what the
    /// emitted `q8mv` kernels read — the dispatcher owns the split (and caches it per weight),
    /// the launcher owns the launch. The default REFUSES: a launcher that cannot run it must say
    /// so, not approximate it.
    fn matvec_q8(
        &self,
        codes: &[i8],
        scales: &[f32],
        x: &[f32],
        d_out: usize,
    ) -> Result<Vec<f32>> {
        let _ = (codes, scales, x, d_out);
        bail!("{}: matvec_q8 is not implemented by this launcher", self.name())
    }

    /// Drop any device-side buffers pinned under host `(ptr, len)` identities.
    ///
    /// The launcher may cache uploads of long-lived weight planes. The identities are HOST
    /// pointers, so the moment a caller drops the backing allocations this cache can go stale
    /// the ABA way — a fresh Vec landing on a recycled address would silently reuse the OLD
    /// device bytes. The contract: whoever drops pinned host memory calls this first. Default
    /// no-op for launchers that cache nothing.
    fn invalidate_pins(&self) {}

    /// w8a8 expert matvec: `out[r] = sum_c (codes[r,c] - offset[r]) * scale[r] * x[c]`.
    ///
    /// The package's own layout -- int8 codes with a per-output-channel scale and offset -- so nothing is
    /// dequantised before the launch. This is the launch the engine's two `w8a8_experts_*` slots need;
    /// the argument order is the KERNEL's: (x, w, scale, offset, out).
    ///
    /// Default refuses, like the two below: a launcher with no device path must fall back to the host
    /// rather than compute something that is not this function.
    fn matvec_w8a8_experts(
        &self,
        name: &str,
        codes: &[i8],
        scale: &[f32],
        offset: &[f32],
        x: &[f32],
        d_out: usize,
    ) -> Result<Vec<f32>> {
        let _ = (name, codes, scale, offset, x, d_out);
        bail!("{}: matvec_w8a8_experts is not implemented by this launcher", self.name())
    }

    /// Packed-MXFP4 matvec (`mxfp4_repack` contract: nibble plane, halved even|odd scales,
    /// PERMUTED activation). Default refuses, same reasoning as `matvec_q8`.
    fn matvec_mx4pk(
        &self,
        w: &[u8],
        scales: &[f32],
        x_perm: &[f32],
        d_out: usize,
    ) -> Result<Vec<f32>> {
        let _ = (w, scales, x_perm, d_out);
        bail!("{}: matvec_mx4pk is not implemented by this launcher", self.name())
    }
}

/// Validate a launcher's output before it reaches the model.
///
/// A device kernel that returns the wrong number of elements, or a non-finite value, must fail
/// loudly here. Propagating it produces a model that generates plausible tokens from a broken
/// kernel — the most expensive failure mode available, and one this project has already met in
/// the form of kernels that compiled and computed nothing.
pub fn check_rms_norm_output(out: &[f32], expected_len: usize, who: &str) -> Result<()> {
    if out.len() != expected_len {
        bail!(
            "{who}: rms_norm returned {} values, expected {expected_len} — a length mismatch here \
             would otherwise be reshaped silently downstream",
            out.len()
        );
    }
    if let Some((i, v)) = out.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        bail!("{who}: rms_norm produced a non-finite value {v} at index {i}");
    }
    Ok(())
}

/// A launcher that computes on the host, for testing the plumbing without a device.
///
/// Deliberately NOT named anything suggesting a device, and `is_device()` returns false, so it
/// cannot be counted as offloaded. It exists to prove marshalling, validation and the swap
/// mechanism — not to stand in for the NPU.
pub struct HostFakeLaunch {
    name: &'static str,
    /// When set, corrupt the output so the validation path itself can be tested.
    pub corrupt: Option<Corruption>,
}

#[derive(Debug, Clone, Copy)]
pub enum Corruption {
    /// Return one element too few.
    ShortOutput,
    /// Return a NaN in the first position.
    NonFinite,
}

impl HostFakeLaunch {
    pub fn new(name: &'static str) -> Self {
        Self { name, corrupt: None }
    }
    pub fn with_corruption(name: &'static str, c: Corruption) -> Self {
        Self {
            name,
            corrupt: Some(c),
        }
    }
}

impl KernelLaunch for HostFakeLaunch {
    fn name(&self) -> &str {
        self.name
    }

    /// Always false. See the module note: a fake that reported true would make the pipeline look
    /// finished while nothing had run on hardware.
    fn is_device(&self) -> bool {
        false
    }

    fn rms_norm(&self, x: &[f32], gamma: &[f32], eps: f32) -> Result<Vec<f32>> {
        // Same arithmetic as the CPU dispatcher, reached through the launcher interface so the
        // marshalling is what is under test rather than the maths.
        let n = x.len();
        let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        let mut out: Vec<f32> = (0..n).map(|i| x[i] * inv * gamma[i % gamma.len()]).collect();
        match self.corrupt {
            Some(Corruption::ShortOutput) => {
                out.pop();
            }
            Some(Corruption::NonFinite) => {
                if let Some(v) = out.first_mut() {
                    *v = f32::NAN;
                }
            }
            None => {}
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_host_fake_is_never_reported_as_a_device() {
        let f = HostFakeLaunch::new("rms_norm");
        assert!(!f.is_device(), "a host fake must never claim to be a device");
        assert_eq!(f.name(), "rms_norm");
    }

    #[test]
    fn output_validation_rejects_a_short_buffer() {
        let f = HostFakeLaunch::with_corruption("rms_norm", Corruption::ShortOutput);
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let g = vec![1.0f32; 4];
        let out = f.rms_norm(&x, &g, 1e-6).unwrap();
        let e = check_rms_norm_output(&out, x.len(), "test").unwrap_err().to_string();
        assert!(e.contains("expected 4"), "got: {e}");
    }

    #[test]
    fn output_validation_rejects_a_non_finite_value() {
        let f = HostFakeLaunch::with_corruption("rms_norm", Corruption::NonFinite);
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let g = vec![1.0f32; 4];
        let out = f.rms_norm(&x, &g, 1e-6).unwrap();
        let e = check_rms_norm_output(&out, x.len(), "test").unwrap_err().to_string();
        assert!(e.contains("non-finite"), "got: {e}");
    }

    #[test]
    fn a_correct_launcher_passes_validation() {
        let f = HostFakeLaunch::new("rms_norm");
        let x = vec![0.5f32, -1.5, 2.0, 0.25];
        let g = vec![1.0f32; 4];
        let out = f.rms_norm(&x, &g, 1e-6).unwrap();
        check_rms_norm_output(&out, x.len(), "test").unwrap();
    }
}
