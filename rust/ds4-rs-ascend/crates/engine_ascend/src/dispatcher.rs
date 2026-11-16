// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! The Ascend backend as a `engine_core` DISPATCHER, rather than a second engine.
//!
//! `engine_core`'s layer stack is generic over two traits — `decode_step_with_attn<K, A>` takes a
//! `KernelDispatcher` and an `AttentionDispatcher` — so the layer sequencing, KV cache, MoE
//! routing and lm_head do not need porting to reach this target. They need these two traits
//! implemented.
//!
//! This is the skeleton: every method DELEGATES to an inner dispatcher, so the whole model runs
//! through `engine_ascend` from the first commit and each method can be swapped to a generated PTO
//! kernel one at a time, with the inner implementation as that method's oracle. The pattern is
//! the one `engine_core` already uses for `RecordingDispatcher` and `TracingDispatcher`.
//!
//! Two things this buys that a from-scratch engine would not:
//!
//!   * a working baseline at every point, so a regression is always attributable to the one
//!     method just swapped;
//!   * an oracle per method that is not a transcription — it is the implementation the rest of
//!     the project already trusts.
//!
//! CORRECTED: this note used to say "nothing here launches a kernel yet — the `ascend` feature is
//! still an empty flag with no ACL linkage, and every device result in this project has come from
//! C++ runners with kernels compiled in". Two of those three claims are now false. The feature links
//! `libruntime` and `libascendcl`, and a kernel HAS been launched from Rust — the e8m0 expansion at
//! 0/262144 wrong bitwise, and `rms_norm` at 9.344e-7 against an f64 replay. [`crate::device::DeviceLaunch`]
//! is a real [`KernelLaunch`], and [`AscendDispatcher::with_device`] installs it in one call.
//!
//! What remains true, and is the honest state: nothing in the DECODE PATH installs one. [`offloaded`]
//! reports which methods run on the NPU for a dispatcher built by [`AscendDispatcher::new`], and that
//! is still none — a launcher has to be installed for the answer to change, and
//! [`AscendDispatcher::offloaded_by`] is what reports it per instance. That distinction is why the
//! two functions exist separately, and why a stale module note like the one above is worth
//! correcting rather than leaving to be read as current.

use crate::launch::{check_rms_norm_output, KernelLaunch};
use engine_core::dispatch::KernelDispatcher;

/// Which dispatcher methods are served by generated PTO kernels rather than delegated.
///
/// Exists so "is this on the NPU yet?" has an answer in code instead of in a document. An empty
/// slice is the honest state with no launcher installed.
pub fn offloaded() -> &'static [&'static str] {
    &[]
}

/// Wraps an inner dispatcher and forwards everything.
///
/// Generic over the inner type so tests can wrap `CpuDispatcher` while production wraps whatever
/// the server already builds, and so swapping one method never silently changes the others.
/// One expert's weights in the checkpoint's own layout: int8 codes with a per-output-channel
/// scale and offset, i.e. exactly what the emitted `w8a8_experts_*` kernels take. Nothing is
/// dequantised before the launch.
#[derive(Clone, Debug, Default)]
pub struct W8a8Expert {
    pub codes: Vec<i8>,
    pub scale: Vec<f32>,
    pub offset: Vec<f32>,
}

pub struct AscendDispatcher<D: KernelDispatcher> {
    inner: D,
    /// Installed per method. `None` delegates, which is what keeps a working baseline at every
    /// point during the swap. `Arc` rather than `Box` because one opened device serves several
    /// methods.
    rms_norm_launch: Option<std::sync::Arc<dyn KernelLaunch>>,
    matvec_q8_launch: Option<std::sync::Arc<dyn KernelLaunch>>,
    matvec_mx4pk_launch: Option<std::sync::Arc<dyn KernelLaunch>>,
    /// The w8a8 expert matvec launch. A SEPARATE field from `matvec_mx4pk_launch` because it is a
    /// separate socket: five arguments (x, w, per-output-channel scale, offset, out) against the
    /// packed form's four. Keeping one field for both would be the same "same extent, different
    /// function" mistake the KPI reader was fixed for twice.
    matvec_w8a8_experts_launch: Option<std::sync::Arc<dyn KernelLaunch>>,
    /// One launcher PER CHIP, for the routed-expert (MXFP4) matvecs. `moe_routed_step` picks
    /// `moe_devices[layer_idx % n]`, so each chip pins the expert weights for ITS layers — the
    /// 155 GB of MXFP4 experts (what SSD-streams today) then spread across chips' HBM instead of
    /// one chip re-streaming them every token. Empty = no device; length 1 = the single-chip path
    /// (identical routing to before). `matvec_mx4pk_launch` above stays as `moe_devices[0]` for the
    /// capability gate; the per-layer routing reads this vector.
    moe_devices: Vec<std::sync::Arc<dyn KernelLaunch>>,
    /// Split Q8_0 planes, keyed by the WEIGHT's `(ptr, len)`. The GGUF interleaves a 2-byte f16
    /// scale with each 32-code block; the emitted `q8mv` kernels read separate planes. Weights
    /// are long-lived mmaps with stable addresses, so the split is paid once per tensor — the
    /// same `(ptr, len)` identity cache the Metal side settled on for exactly this problem.
    q8_planes: std::sync::Mutex<
        std::collections::HashMap<(usize, usize), std::sync::Arc<(Vec<i8>, Vec<f32>)>>,
    >,
    /// f32-plane identity -> raw Q8_0 bytes for the SAME weight, as `(ptr, len)` pairs.
    ///
    /// The composed model already retains the GGUF Q8_0 bytes beside each dequanted f32
    /// projection (`attn_*_q8`, kept for Metal's 1-byte matvec). Registering the pair lets
    /// `matvec_f32` recognise a plane it was handed BY VALUE and run the device q8 kernel on
    /// the original bytes instead — which is how the provided fused methods
    /// (`qkv_b_head_rms_batched` et al.) pick up the NPU without being overridden: their
    /// defaults compose `self.matvec_f32`, and `self` is this wrapper.
    ///
    /// Raw parts rather than owned copies: registration is `unsafe`, with the contract that
    /// both slices outlive this dispatcher (they live in the runner's composed model, which
    /// outlives the session that owns this dispatcher).
    q8_alias: std::sync::Mutex<std::collections::HashMap<(usize, usize), (usize, usize)>>,
    /// Per-layer ROUTED-EXPERT tables: raw quantised bytes for the stacked
    /// `ffn_{gate,up,down}_exps` tensors, as `(ptr, len)` with shape/type. The composed CPU
    /// model deliberately does NOT materialise routed experts (600+ GB as f32); this is the
    /// provider that makes them real for the CPU/NPU chain — dequant the SELECTED experts per
    /// call, never the stack. Same `unsafe` lifetime contract as the q8 aliases.
    expert_tables: std::sync::Mutex<std::collections::HashMap<u32, ExpertTable>>,
    /// Dequanted expert triples, keyed by (layer, expert). Greedy decode re-selects the same
    /// experts constantly; a bounded map keeps the hot ones. Cleared wholesale past the cap —
    /// crude, correct, and measurable later.
    expert_cache: std::sync::Mutex<
        std::collections::HashMap<(u32, usize), std::sync::Arc<(Vec<f32>, Vec<f32>, Vec<f32>)>>,
    >,
    /// Repacked (device-format) expert triples, keyed like `expert_cache`. ~40 MB per triple
    /// (0.625 B/weight) versus the f32 cache's ~100 — the device path is also the smaller one.
    repack_cache: std::sync::Mutex<
        std::collections::HashMap<
            (u32, usize),
            std::sync::Arc<(
                crate::mxfp4_repack::PackedWeight,
                crate::mxfp4_repack::PackedWeight,
                crate::mxfp4_repack::PackedWeight,
            )>,
        >,
    >,
}

/// One layer's routed-expert source: raw stacked quantised tensors.
#[derive(Clone, Copy)]
pub struct ExpertTable {
    pub ttype: engine_core::gguf::GgmlType,
    gate: (usize, usize),
    up: (usize, usize),
    down: (usize, usize),
    pub n_experts: usize,
    pub d_ffn: usize,
    pub d_model: usize,
}

impl<D: KernelDispatcher> AscendDispatcher<D> {
    pub fn new(inner: D) -> Self {
        Self {
            inner,
            rms_norm_launch: None,
            matvec_q8_launch: None,
            matvec_mx4pk_launch: None,
            matvec_w8a8_experts_launch: None,
            moe_devices: Vec::new(),
            q8_planes: std::sync::Mutex::new(std::collections::HashMap::new()),
            q8_alias: std::sync::Mutex::new(std::collections::HashMap::new()),
            expert_tables: std::sync::Mutex::new(std::collections::HashMap::new()),
            expert_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
            repack_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// The device-format triple for one expert: repacked nibble planes + halved scales.
    fn expert_triple_packed(
        &self,
        t: &ExpertTable,
        layer_idx: u32,
        e: usize,
    ) -> Option<
        std::sync::Arc<(
            crate::mxfp4_repack::PackedWeight,
            crate::mxfp4_repack::PackedWeight,
            crate::mxfp4_repack::PackedWeight,
        )>,
    > {
        use engine_core::gguf::GgmlType;
        if e >= t.n_experts || t.ttype != GgmlType::MXFP4 {
            return None;
        }
        let key = (layer_idx, e);
        if let Some(hit) = self.repack_cache.lock().unwrap().get(&key) {
            return Some(hit.clone());
        }
        let stride_gu = t.d_ffn * t.d_model / 32 * 17;
        let stride_dn = t.d_model * t.d_ffn / 32 * 17;
        let slice = |src: (usize, usize), stride: usize| -> Option<&'static [u8]> {
            let (p, l) = src;
            if (e + 1) * stride > l {
                return None;
            }
            Some(unsafe { std::slice::from_raw_parts((p + e * stride) as *const u8, stride) })
        };
        let g = crate::mxfp4_repack::repack_weight(slice(t.gate, stride_gu)?, t.d_model, t.d_ffn)
            .ok()?;
        let u = crate::mxfp4_repack::repack_weight(slice(t.up, stride_gu)?, t.d_model, t.d_ffn)
            .ok()?;
        let d = crate::mxfp4_repack::repack_weight(slice(t.down, stride_dn)?, t.d_ffn, t.d_model)
            .ok()?;
        let arc = std::sync::Arc::new((g, u, d));
        let mut cache = self.repack_cache.lock().unwrap();
        if cache.len() >= 96 {
            // The launcher pins device twins under these Arcs' HOST addresses; dropping the
            // Arcs without invalidating would leave the pins ABA-stale (a fresh allocation on
            // a recycled address would reuse the wrong device bytes). Contract from
            // KernelLaunch::invalidate_pins. With N chips a given expert plane may be pinned on
            // ANY of them (whichever chip owns that layer), so every chip is invalidated.
            for l in &self.moe_devices {
                l.invalidate_pins();
            }
            if let Some(l) = &self.matvec_q8_launch {
                l.invalidate_pins();
            }
            cache.clear();
        }
        cache.insert(key, arc.clone());
        Some(arc)
    }

    /// Register a layer's routed-expert table.
    ///
    /// # Safety
    /// The three byte slices must outlive this dispatcher (they are views into the runner's
    /// mmap, which outlives the session owning this dispatcher).
    pub unsafe fn register_expert_table(
        &self,
        layer_idx: u32,
        ttype: engine_core::gguf::GgmlType,
        gate: &[u8],
        up: &[u8],
        down: &[u8],
        n_experts: usize,
        d_ffn: usize,
        d_model: usize,
    ) {
        self.expert_tables.lock().unwrap().insert(
            layer_idx,
            ExpertTable {
                ttype,
                gate: (gate.as_ptr() as usize, gate.len()),
                up: (up.as_ptr() as usize, up.len()),
                down: (down.as_ptr() as usize, down.len()),
                n_experts,
                d_ffn,
                d_model,
            },
        );
    }

    /// Layers with a registered routed-expert table (the non-vacuity signal `step` checks).
    pub fn expert_tables_registered(&self) -> usize {
        self.expert_tables.lock().unwrap().len()
    }

    /// Dequant one expert's (gate, up, down) from a table. MXFP4 only today; other types are
    /// REFUSED by `expert_triple` returning None so the caller can fail loudly.
    fn expert_triple(
        &self,
        t: &ExpertTable,
        layer_idx: u32,
        e: usize,
    ) -> Option<std::sync::Arc<(Vec<f32>, Vec<f32>, Vec<f32>)>> {
        use engine_core::gguf::GgmlType;
        if e >= t.n_experts || t.ttype != GgmlType::MXFP4 {
            return None;
        }
        let key = (layer_idx, e);
        if let Some(hit) = self.expert_cache.lock().unwrap().get(&key) {
            return Some(hit.clone());
        }
        // MXFP4: 17 bytes per 32 weights.
        let stride_gu = t.d_ffn * t.d_model / 32 * 17;
        let stride_dn = t.d_model * t.d_ffn / 32 * 17;
        let slice = |src: (usize, usize), stride: usize| -> Option<&'static [u8]> {
            let (p, l) = src;
            if (e + 1) * stride > l {
                return None;
            }
            Some(unsafe {
                std::slice::from_raw_parts((p + e * stride) as *const u8, stride)
            })
        };
        let g = engine_core::glm5_stream::dequant_mxfp4(slice(t.gate, stride_gu)?);
        let u = engine_core::glm5_stream::dequant_mxfp4(slice(t.up, stride_gu)?);
        let d = engine_core::glm5_stream::dequant_mxfp4(slice(t.down, stride_dn)?);
        let arc = std::sync::Arc::new((g, u, d));
        let mut cache = self.expert_cache.lock().unwrap();
        // ~100 MB per triple at real dims; 48 entries ≈ 5 GB, fine on the 2 TB box and
        // irrelevant at test dims. Wholesale clear keeps the code obviously correct.
        if cache.len() >= 48 {
            cache.clear();
        }
        cache.insert(key, arc.clone());
        Some(arc)
    }

    /// Register `q8` as the raw Q8_0 source of the dequanted `f32_plane`.
    ///
    /// # Safety
    /// Both slices must outlive this dispatcher: only `(ptr, len)` is stored. The intended
    /// caller is the session, registering planes owned by the runner it borrows.
    pub unsafe fn register_q8_alias(&self, f32_plane: &[f32], q8: &[u8]) {
        self.q8_alias.lock().unwrap().insert(
            (f32_plane.as_ptr() as usize, f32_plane.len()),
            (q8.as_ptr() as usize, q8.len()),
        );
    }

    /// Aliases registered (for reporting/tests).
    pub fn q8_aliases(&self) -> usize {
        self.q8_alias.lock().unwrap().len()
    }
    pub fn inner(&self) -> &D {
        &self.inner
    }

    /// Build a dispatcher with the NPU installed for every method that has a kernel.
    ///
    /// One call, because the alternative — open a device, register a binary, wrap it, install it —
    /// is four steps that a caller could get half-right. Today that is `rms_norm` only; each further
    /// method joins here as its kernel is verified, and [`Self::offloaded_by`] is the report of what
    /// actually landed rather than what was intended.
    #[cfg(feature = "ascend")]
    pub fn with_device(self, elf_path: &std::path::Path, device_id: i32) -> anyhow::Result<Self> {
        self.with_devices(elf_path, &[device_id])
    }

    /// Open `elf_path` on EVERY chip in `device_ids` and install them.
    ///
    /// The device binary registers once, process-wide (see `DeviceBinary`), so the second and later
    /// chips reuse that registration and only cost a context + stream. `rms_norm` and `matvec_q8`
    /// are layer-agnostic (their dispatch methods never see a layer), so they run on the first chip.
    /// The routed-expert (MXFP4) matvecs ARE layer-aware — `moe_devices` holds all chips and
    /// `moe_routed_step` sends layer `L`'s experts to `moe_devices[L % n]`, so each chip pins the
    /// expert weights for its own layers instead of one chip re-streaming all 155 GB every token.
    #[cfg(feature = "ascend")]
    pub fn with_devices(
        mut self,
        elf_path: &std::path::Path,
        device_ids: &[i32],
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(!device_ids.is_empty(), "with_devices needs at least one chip");
        // Open the first chip and read the ELF's capabilities (identical across chips).
        let d0 = crate::device::DeviceLaunch::open(elf_path, device_ids[0])?;
        let (has_rms, has_q8, has_mx4pk) =
            (d0.has_any("rms_norm"), d0.has_any("matvec_"), d0.has_any("mx4pk"));
        let dev0: std::sync::Arc<dyn KernelLaunch> = std::sync::Arc::new(d0);
        // Layer-agnostic kernels: the first chip. An ELF that lacks a kind stays on CPU for it
        // rather than panicking at an unregistered launch.
        if has_rms {
            self.rms_norm_launch = Some(dev0.clone());
        }
        if has_q8 {
            self.matvec_q8_launch = Some(dev0.clone());
        }
        // Only the routed-expert (MXFP4) matvecs can use more than one chip, so extra chips are
        // opened only when the ELF carries those kernels. Without them the request degrades to the
        // first chip rather than opening chips it would immediately drop.
        if has_mx4pk {
            self.matvec_mx4pk_launch = Some(dev0.clone());
            self.matvec_w8a8_experts_launch = Some(dev0.clone());
            let mut moe = vec![dev0];
            for &id in &device_ids[1..] {
                moe.push(std::sync::Arc::new(crate::device::DeviceLaunch::open(elf_path, id)?));
            }
            self.moe_devices = moe;
        }
        Ok(self)
    }

    /// Install a launcher for `rms_norm`.
    pub fn with_rms_norm(mut self, l: Box<dyn KernelLaunch>) -> Self {
        self.rms_norm_launch = Some(std::sync::Arc::from(l));
        self
    }

    /// Install a launcher for `matvec_q8`.
    pub fn with_matvec_q8(mut self, l: Box<dyn KernelLaunch>) -> Self {
        self.matvec_q8_launch = Some(std::sync::Arc::from(l));
        self
    }

    /// Install a launcher for `matvec_mx4pk` (the routed-expert projections).
    pub fn with_matvec_mx4pk(mut self, l: Box<dyn KernelLaunch>) -> Self {
        self.matvec_mx4pk_launch = Some(std::sync::Arc::from(l));
        self
    }

    /// Route the w8a8 expert matvec to a launcher. Separate from `with_matvec_mx4pk` on purpose: the
    /// two sockets take different arguments, so a caller that wants w8a8 experts must say so.
    /// One routed-expert step whose planes come STRAIGHT FROM THE PACKAGE.
    ///
    /// This is the wiring item (3) names: `moe_routed_step_w8a8` takes planes, and the checkpoint already
    /// stores them as int8 codes with a per-output-channel scale and offset -- so this reads them, hands
    /// them over, and dequantises nothing. The package's `(q - offset) * scale` is exactly the kernel's
    /// arithmetic.
    pub fn moe_routed_step_from_package(
        &self,
        pkg: &engine_core::safetensors::Package,
        layer: usize,
        expert: usize,
        x: &[f32],
        weight: f32,
        d_ffn: usize,
    ) -> anyhow::Result<Vec<f32>> {
        let prefix = format!("layers.{layer}.ffn.experts.{expert}");
        let one = |name: &str| -> anyhow::Result<W8a8Expert> {
            let raw = pkg.raw(&format!("{prefix}.{name}.weight"))?.1;
            let scale = pkg.f32_tensor(&format!("{prefix}.{name}.weight_scale"))?;
            let offset = pkg.f32_tensor(&format!("{prefix}.{name}.weight_offset"))?;
            anyhow::ensure!(
                scale.len() == offset.len(),
                "{name}: scale {} != offset {}",
                scale.len(),
                offset.len()
            );
            Ok(W8a8Expert { codes: raw.iter().map(|&b| b as i8).collect(), scale, offset })
        };
        let planes = [one("w1")?, one("w3")?, one("w2")?];
        self.moe_routed_step_w8a8(x, &[expert], &[weight], &planes, d_ffn)
    }

    /// The w8a8 expert launcher, or a refusal naming what is missing.
    fn w8a8_launcher(&self) -> anyhow::Result<&std::sync::Arc<dyn KernelLaunch>> {
        self.matvec_w8a8_experts_launch.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "no w8a8 expert launcher: register one with `with_matvec_w8a8_experts` (or \
                 `with_devices`), otherwise the MoE must stay on the host"
            )
        })
    }

    /// One routed expert's three projections, in the package's OWN w8a8 layout.
    ///
    /// No dequantisation on this side: the checkpoint stores int8 codes with a per-output-channel
    /// scale and offset, which is exactly what the emitted `w8a8_experts_*` kernels take.
    pub fn moe_routed_step_w8a8(
        &self,
        x: &[f32],
        selected: &[usize],
        weights: &[f32],
        planes: &[W8a8Expert; 3],
        d_ffn: usize,
    ) -> anyhow::Result<Vec<f32>> {
        let l = self.w8a8_launcher()?;
        let (gate_p, up_p, down_p) = (&planes[0], &planes[1], &planes[2]);
        // The engine's own fold, kept identical to `engine_core::moe::moe_routed_step`:
        // gate clamps upper-only, up clamps symmetrically, and the routing weight is folded
        // into `mid` BEFORE the down projection.
        const CLAMP: f32 = 10.0;
        let mut acc = vec![0.0f32; x.len()];
        for (&_e, &w_e) in selected.iter().zip(weights.iter()) {
            let gate = l.matvec_w8a8_experts(
                "w8a8_experts_4096_2048", &gate_p.codes, &gate_p.scale, &gate_p.offset, x, d_ffn)?;
            let up = l.matvec_w8a8_experts(
                "w8a8_experts_4096_2048", &up_p.codes, &up_p.scale, &up_p.offset, x, d_ffn)?;
            let mid: Vec<f32> = gate
                .iter()
                .zip(&up)
                .map(|(&g, &u)| {
                    let g = g.clamp(f32::NEG_INFINITY, CLAMP);
                    let u = u.clamp(-CLAMP, CLAMP);
                    let s = g / (1.0 + (-g).exp());
                    s * u * w_e
                })
                .collect();
            let down = l.matvec_w8a8_experts(
                "w8a8_experts_2048_4096", &down_p.codes, &down_p.scale, &down_p.offset, &mid, x.len())?;
            for (a, d) in acc.iter_mut().zip(down) {
                *a += d;
            }
        }
        Ok(acc)
    }

    pub fn with_matvec_w8a8_experts(mut self, l: Box<dyn KernelLaunch>) -> Self {
        self.matvec_w8a8_experts_launch = Some(std::sync::Arc::from(l));
        self
    }

    /// Methods that actually run on an NPU — keyed on `KernelLaunch::is_device`, so a host fake
    /// exercising the whole path still reports nothing. See `launch`'s module note.
    pub fn offloaded_by(&self) -> Vec<&str> {
        let mut v = Vec::new();
        if let Some(l) = &self.rms_norm_launch {
            if l.is_device() {
                v.push("rms_norm");
            }
        }
        if let Some(l) = &self.matvec_q8_launch {
            if l.is_device() {
                v.push("matvec_q8");
            }
        }
        if let Some(l) = &self.matvec_w8a8_experts_launch {
            if l.is_device() {
                v.push("matvec_w8a8_experts");
            }
        }
        if let Some(l) = &self.matvec_mx4pk_launch {
            if l.is_device() {
                v.push("matvec_mx4pk");
            }
        }
        v
    }

    /// The split planes for a Q8_0 weight, computed once per `(ptr, len)` identity.
    fn q8_split(&self, w_q8: &[u8], d_in: usize, d_out: usize) -> std::sync::Arc<(Vec<i8>, Vec<f32>)> {
        let key = (w_q8.as_ptr() as usize, w_q8.len());
        if let Some(hit) = self.q8_planes.lock().unwrap().get(&key) {
            return hit.clone();
        }
        let nb = d_in / 32;
        let row_bytes = nb * 34;
        let mut codes = vec![0i8; d_out * d_in];
        // Per-ELEMENT expanded scales — the layout the emitted kernel reads (its scale tensor is
        // a [1 x d_in] view per output row). 4x the bytes of a per-block plane, paid once per
        // weight; keeping the kernel byte-identical to its proven form is worth more than the
        // memory, the same trade the e8m0 expansion made.
        let mut scales = vec![0f32; d_out * d_in];
        for o in 0..d_out {
            let row = &w_q8[o * row_bytes..(o + 1) * row_bytes];
            for (b, blk) in row.chunks_exact(34).enumerate() {
                let sc =
                    engine_core::layer_view::f16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                for i in 0..32 {
                    codes[o * d_in + b * 32 + i] = blk[2 + i] as i8;
                    scales[o * d_in + b * 32 + i] = sc;
                }
            }
        }
        let arc = std::sync::Arc::new((codes, scales));
        self.q8_planes.lock().unwrap().insert(key, arc.clone());
        arc
    }
}

impl<D: KernelDispatcher> KernelDispatcher for AscendDispatcher<D> {
    // Only the trait's REQUIRED methods are forwarded. The provided ones are deliberately left
    // to their defaults: those defaults compose these primitives, so a kernel swapped in below
    // is picked up by every chain built on it. Overriding a chain as well would mean two places
    // to keep in sync, and a fused chain kernel is worth having only once its parts are proven.
    /// Launches when a launcher is installed, delegates otherwise.
    ///
    /// A launcher failure or a malformed result PANICS rather than silently falling back to the
    /// CPU. Falling back would turn a broken kernel into a slow-but-correct model, which is the
    /// one outcome that guarantees nobody notices — and the trait returns `Vec<f32>`, so there is
    /// no error channel to use instead.
    fn rms_norm(&self, x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
        // Shape routing is POLICY, like matvec_q8's: the emitted kernels are
        // shape-specialised (4096/1024/512) and bake their eps in, so a width or eps they
        // were never emitted for goes to the CPU rather than to a refusal-turned-panic.
        // The first full engine run found this the honest way: the HC-wide norm asks for
        // 16384, which no kernel serves. A launcher FAILURE on a shape it accepted still
        // panics — that is a broken kernel, not a routing decision.
        let device_serves = crate::device::select_rms_kernel(x.len()).is_ok()
            && crate::device::check_rms_eps(eps).is_ok();
        match &self.rms_norm_launch {
            Some(l) if device_serves => {
                let out = l
                    .rms_norm(x, gamma, eps)
                    .unwrap_or_else(|e| panic!("{}: rms_norm launch failed: {e}", l.name()));
                check_rms_norm_output(&out, x.len(), l.name()).unwrap_or_else(|e| panic!("{e}"));
                out
            }
            _ => self.inner.rms_norm(x, gamma, eps),
        }
    }

    fn matvec_f32(&self, w: &[f32], x: &[f32], d_out: usize) -> Vec<f32> {
        // Stays on the CPU BY DECISION for arbitrary f32 weights — the proven device kernels
        // are quantised. EXCEPT when this exact plane was registered as the dequant of a
        // retained Q8_0 tensor: then the original bytes exist, the device kernel applies, and
        // the call reroutes through the q8 seam (which still gates on the four ELF shapes and
        // still panics on a launcher failure). This is how the chain's fused projections reach
        // the NPU without any change to engine_core.
        if self.matvec_q8_launch.is_some()
            && crate::device::q8mv_kernel_for(x.len(), d_out).is_some()
        {
            let key = (w.as_ptr() as usize, w.len());
            let hit = self.q8_alias.lock().unwrap().get(&key).copied();
            if let Some((p, l)) = hit {
                let q8 = unsafe { std::slice::from_raw_parts(p as *const u8, l) };
                return self.matvec_q8(q8, x, d_out);
            }
        }
        self.inner.matvec_f32(w, x, d_out)
    }

    /// Offloads the four shapes the ELF carries; every other shape delegates. That routing is
    /// POLICY, stated here: the device kernels are shape-specialised (their unrolling and
    /// blockDim were tuned per shape), so an unknown shape goes to the CPU rather than to a
    /// kernel that was never validated for it. A launcher failure on a KNOWN shape panics, same
    /// as rms_norm — falling back would turn a broken kernel into a slow-but-correct model.
    fn matvec_q8(&self, w_q8: &[u8], x: &[f32], d_out: usize) -> Vec<f32> {
        let d_in = x.len();
        let launchable = self
            .matvec_q8_launch
            .as_ref()
            .filter(|_| crate::device::q8mv_kernel_for(d_in, d_out).is_some());
        match launchable {
            None => self.inner.matvec_q8(w_q8, x, d_out),
            Some(l) => {
                let planes = self.q8_split(w_q8, d_in, d_out);
                let out = l
                    .matvec_q8(&planes.0, &planes.1, x, d_out)
                    .unwrap_or_else(|e| panic!("{}: matvec_q8 launch failed: {e}", l.name()));
                check_rms_norm_output(&out, d_out, l.name()).unwrap_or_else(|e| panic!("{e}"));
                out
            }
        }
    }

    fn softplus_sqrt(&self, logits: &[f32]) -> Vec<f32> {
        self.inner.softplus_sqrt(logits)
    }

    fn router_finalize(&self, probs: &[f32], bias: &[f32], k: usize) -> (Vec<usize>, Vec<f32>) {
        self.inner.router_finalize(probs, bias, k)
    }

    /// The routed MoE step, made REAL for this backend. The composed CPU model deliberately
    /// leaves the stacked expert f32 slices empty (materialising them is 600+ GB); when a
    /// layer's expert table is registered, the SELECTED experts are dequanted from the raw
    /// quantised bytes and the exact `engine_core::moe::moe_routed_step` math runs on a
    /// selected-only stack — zero transcription of the clamp/weight-fold semantics. Without a
    /// table the call delegates unchanged (synthetic tests pass real slices). The batched
    /// variant needs no override: its default composes `self.moe_routed_step`.

    #[allow(clippy::too_many_arguments)]
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
        let table = self.expert_tables.lock().unwrap().get(&layer_idx).copied();
        // This layer's chip: with N chips, layer L's experts run on chip L%N, so each chip pins
        // (and re-uses across tokens) only its own layers' MXFP4 planes. One chip => index 0, i.e.
        // exactly the previous single-chip path.
        let dev_for_layer = (!self.moe_devices.is_empty())
            .then(|| &self.moe_devices[layer_idx as usize % self.moe_devices.len()]);
        // DEVICE path: the two expert projections run as packed-MXFP4 kernels on the NPU; the
        // swiglu/clamp/weight-fold glue stays on the CPU, transcribed from the free function
        // it must match. Falls through to the CPU dequant path when no launcher is present.
        if let (Some(t), Some(l)) = (table.as_ref(), dev_for_layer) {
            if t.ttype == engine_core::gguf::GgmlType::MXFP4
                && crate::device::mx4pk_kernel_for(t.d_model, t.d_ffn).is_some()
                && crate::device::mx4pk_kernel_for(t.d_ffn, t.d_model).is_some()
            {
                let xperm = crate::mxfp4_repack::permute_activation(x, t.d_model)
                    .expect("x length is d_model by contract");
                let mut acc = vec![0.0f32; t.d_model];
                const CLAMP: f32 = 10.0;
                for (&e, &w_e) in selected.iter().zip(weights.iter()) {
                    let pk = self.expert_triple_packed(t, layer_idx, e).unwrap_or_else(|| {
                        panic!(
                            "routed expert {e} of layer {layer_idx} is not repackable: \
                             refusing to decode without it"
                        )
                    });
                    let gate = l
                        .matvec_mx4pk(&pk.0.w, &pk.0.d, &xperm, t.d_ffn)
                        .unwrap_or_else(|err| panic!("{}: expert gate launch: {err}", l.name()));
                    let up = l
                        .matvec_mx4pk(&pk.1.w, &pk.1.d, &xperm, t.d_ffn)
                        .unwrap_or_else(|err| panic!("{}: expert up launch: {err}", l.name()));
                    // Same fold as engine_core::moe::moe_routed_step: gate clamps upper-only,
                    // up symmetrically, weight folded into mid BEFORE the down matvec.
                    let mid: Vec<f32> = gate
                        .iter()
                        .zip(up.iter())
                        .map(|(&g, &u)| {
                            let g_c = if g > CLAMP { CLAMP } else { g };
                            let u_c = u.clamp(-CLAMP, CLAMP);
                            (g_c / (1.0 + (-g_c).exp())) * u_c * w_e
                        })
                        .collect();
                    let midperm = crate::mxfp4_repack::permute_activation(&mid, t.d_ffn)
                        .expect("mid length is d_ffn");
                    let y = l
                        .matvec_mx4pk(&pk.2.w, &pk.2.d, &midperm, t.d_model)
                        .unwrap_or_else(|err| panic!("{}: expert down launch: {err}", l.name()));
                    for (av, &yv) in acc.iter_mut().zip(y.iter()) {
                        *av += yv;
                    }
                }
                return acc;
            }
        }
        if let Some(t) = table {
            let d_ffn_real = t.d_ffn;
            let mut g_stack = Vec::with_capacity(selected.len() * d_ffn_real * t.d_model);
            let mut u_stack = Vec::with_capacity(selected.len() * d_ffn_real * t.d_model);
            let mut d_stack = Vec::with_capacity(selected.len() * t.d_model * d_ffn_real);
            for &e in selected {
                let triple = self.expert_triple(&t, layer_idx, e).unwrap_or_else(|| {
                    panic!(
                        "routed expert {e} of layer {layer_idx} is not providable \
                         (type {:?}, n_experts {}): refusing to decode without it",
                        t.ttype, t.n_experts
                    )
                });
                g_stack.extend_from_slice(&triple.0);
                u_stack.extend_from_slice(&triple.1);
                d_stack.extend_from_slice(&triple.2);
            }
            let local: Vec<usize> = (0..selected.len()).collect();
            return engine_core::moe::moe_routed_step(
                x, &local, weights, &g_stack, &u_stack, &d_stack, d_ffn_real,
            );
        }
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

/// The dispatcher on REAL hardware, end to end.
///
/// Gated on `DS4_DEVICE_ELF` rather than on the feature alone, so it is skipped — loudly — where
/// there is no device, and cannot pass by not running. This is the first test in the project that
/// makes [`offloaded_by`] report something other than nothing.
#[cfg(all(test, feature = "ascend"))]
mod device_tests {
    use super::*;
    use engine_core::dispatch::{CpuDispatcher, KernelDispatcher};

    #[test]
    fn the_dispatcher_offloads_rms_norm_to_the_npu() {
        let Ok(elf) = std::env::var("DS4_DEVICE_ELF") else {
            eprintln!("SKIPPED: set DS4_DEVICE_ELF to the 11-kernel device object");
            return;
        };
        let d = AscendDispatcher::new(CpuDispatcher)
            .with_device(std::path::Path::new(&elf), 0)
            .expect("install the NPU");

        // The claim this whole seam exists to make honest.
        assert_eq!(
            d.offloaded_by(),
            vec!["rms_norm", "matvec_q8"],
            "the NPU must be reported as covering both installed methods"
        );

        // And it must COMPUTE, not merely be installed: compare against engine_core's own CPU
        // implementation of the same method, which is the oracle the plan named.
        let n = 4096usize;
        let mut s: u32 = 0x9E37_79B9;
        let mut rnd = || { s ^= s << 13; s ^= s >> 17; s ^= s << 5; ((s & 0xFFFF) as f32 / 65536.0 - 0.5) * 3.0 };
        let x: Vec<f32> = (0..n).map(|_| rnd()).collect();
        let g: Vec<f32> = (0..n).map(|_| 0.75 + rnd().abs()).collect();

        let got = d.rms_norm(&x, &g, crate::device::EMITTED_RMS_EPS);
        // The oracle is engine_core's OWN CPU implementation of the same method — the comparison the
        // corrected plan named, rather than a replay written here.
        let cpu = CpuDispatcher.rms_norm(&x, &g, crate::device::EMITTED_RMS_EPS);
        assert_eq!(got.len(), cpu.len());
        let worst = (0..n)
            .map(|i| ((got[i] - cpu[i]).abs() as f64) / (cpu[i].abs() as f64).max(1e-3))
            .fold(0.0f64, f64::max);
        eprintln!("dispatcher rms_norm on NPU vs engine_core CPU: worst rel {worst:.3e}");
        assert!(got.iter().any(|v| *v != 0.0), "an all-zero result would agree with nothing");
        assert!(worst < 1e-5, "NPU disagrees with the CPU oracle at {worst:.3e}");

        // matvec_q8 at a KNOWN shape (4096 -> 1024): the dispatcher must route to the NPU and
        // agree with engine_core's CPU dequant reference on the SAME GGUF-format bytes.
        let d_in = 4096usize;
        let d_out = 1024usize;
        let nb = d_in / 32;
        let mut w_q8 = vec![0u8; d_out * nb * 34];
        for blk in w_q8.chunks_exact_mut(34) {
            // scale: a small positive f16; codes: full int8 range.
            let sc = 0.5 + ((rnd().abs() * 0.5) as f32);
            let bits = {
                // f32 -> f16 via the round trip the loader itself uses.
                let f = sc;
                let b = f.to_bits();
                let sign = ((b >> 16) & 0x8000) as u16;
                let exp = (((b >> 23) & 0xff) as i32) - 127 + 15;
                let man = ((b >> 13) & 0x3ff) as u16;
                if exp <= 0 { sign } else { sign | ((exp as u16) << 10) | man }
            };
            blk[0] = (bits & 0xff) as u8;
            blk[1] = (bits >> 8) as u8;
            for i in 0..32 {
                blk[2 + i] = ((rnd() * 80.0) as i32).clamp(-127, 127) as u8;
            }
        }
        let x: Vec<f32> = (0..d_in).map(|_| rnd()).collect();
        let got = d.matvec_q8(&w_q8, &x, d_out);
        let cpu = CpuDispatcher.matvec_q8(&w_q8, &x, d_out);
        let worst = (0..d_out)
            .map(|i| ((got[i] - cpu[i]).abs() as f64) / (cpu[i].abs() as f64).max(1e-3))
            .fold(0.0f64, f64::max);
        eprintln!("dispatcher matvec_q8 on NPU vs engine_core CPU: worst rel {worst:.3e}");
        assert!(got.iter().any(|v| *v != 0.0), "an all-zero result would agree with nothing");
        assert!(worst < 1e-4, "NPU disagrees with the CPU q8 oracle at {worst:.3e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_core::dispatch::CpuDispatcher;

    fn synth(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                ((s & 0xFFFF) as f32 / 65536.0 - 0.5) * 2.0
            })
            .collect()
    }

    /// The wrapper must be transparent while it delegates: BIT-IDENTICAL to the inner
    /// dispatcher, not merely close. Anything less and the first swapped kernel cannot be
    /// attributed, because the baseline would already differ.
    #[test]
    fn delegating_wrapper_is_bit_identical_to_the_inner_dispatcher() {
        let cpu = CpuDispatcher;
        let asc = AscendDispatcher::new(CpuDispatcher);
        let x = synth(256, 1);
        let gamma = synth(256, 2);
        assert_eq!(asc.rms_norm(&x, &gamma, 1e-6), cpu.rms_norm(&x, &gamma, 1e-6));

        let (d_in, d_out) = (64usize, 8usize);
        let w = synth(d_in * d_out, 3);
        let v = synth(d_in, 4);
        assert_eq!(asc.matvec_f32(&w, &v, d_out), cpu.matvec_f32(&w, &v, d_out));

        let logits = synth(32, 5);
        assert_eq!(asc.softplus_sqrt(&logits), cpu.softplus_sqrt(&logits));

        let probs = synth(16, 6).iter().map(|v| v.abs()).collect::<Vec<_>>();
        let bias = synth(16, 7);
        assert_eq!(
            asc.router_finalize(&probs, &bias, 4),
            cpu.router_finalize(&probs, &bias, 4)
        );
    }

    /// The DEFAULT-provided methods must also come out identical. They compose the primitives
    /// above, so this is what makes "swap one primitive and every chain follows" true rather
    /// than hoped — and it would fail if the wrapper accidentally overrode a chain.
    #[test]
    fn default_provided_methods_follow_the_primitives() {
        let cpu = CpuDispatcher;
        let asc = AscendDispatcher::new(CpuDispatcher);
        let x = synth(128, 11);
        let n_head = 4usize;
        let head_dim = 32usize;
        assert_eq!(
            asc.head_rms_norm(&x, n_head, head_dim, 1e-6),
            cpu.head_rms_norm(&x, n_head, head_dim, 1e-6),
            "head_rms_norm is default-provided over rms_norm and must track it"
        );

        // A batched CHAIN, which is what decode_step actually calls. This is the real risk the
        // wrapper carries: if a required method were left unforwarded, or if the inner dispatcher
        // overrode a default the wrapper does not, the chains would diverge here while the
        // primitives above still agreed.
        let (d_in, n_lora) = (64usize, 16usize);
        let xx = synth(d_in, 21);
        let w_qa = synth(d_in * n_lora, 22);
        let gamma_q = synth(n_lora, 23);
        assert_eq!(
            asc.layer_qa_rms_batched(&xx, &w_qa, &gamma_q, n_lora, 1e-6),
            cpu.layer_qa_rms_batched(&xx, &w_qa, &gamma_q, n_lora, 1e-6),
            "layer_qa_rms_batched composes matvec_f32 + rms_norm and must track both"
        );
    }

    /// Every method the trait REQUIRES must be forwarded, or the wrapper silently falls back to a
    /// default while the inner dispatcher uses its own implementation. There is no reflection to
    /// assert that with, so this records the count the wrapper was written against: if the trait
    /// gains a required method, this comment is the only warning, and the compiler is the actual
    /// enforcement (a missing required method fails to build, as it did while writing this).
    #[test]
    fn wrapper_forwards_every_required_method() {
        // rms_norm, matvec_f32, softplus_sqrt, router_finalize, moe_routed_step.
        const FORWARDED: usize = 5;
        assert_eq!(FORWARDED, 5, "update the wrapper when the trait's required set changes");
    }

    /// With a launcher installed, the whole path runs through it AND still matches the oracle.
    /// This is what the launcher seam exists to prove: marshalling, validation and the swap
    /// mechanism, with the arithmetic held constant so any difference is the plumbing.
    #[test]
    fn an_installed_launcher_is_used_and_still_matches_the_oracle() {
        use crate::launch::HostFakeLaunch;
        let cpu = CpuDispatcher;
        let asc = AscendDispatcher::new(CpuDispatcher)
            .with_rms_norm(Box::new(HostFakeLaunch::new("rms_norm")));
        let x = synth(256, 31);
        let gamma = synth(256, 32);
        let got = asc.rms_norm(&x, &gamma, 1e-6);
        let want = cpu.rms_norm(&x, &gamma, 1e-6);
        assert_eq!(got.len(), want.len());
        for i in 0..want.len() {
            assert!(
                (got[i] - want[i]).abs() <= 1e-5 * want[i].abs().max(1.0),
                "elem {i}: {} vs {}",
                got[i],
                want[i]
            );
        }
    }

    /// A HOST fake must never be counted as NPU coverage, however much of the path it exercises.
    /// Without this the pipeline reads as finished while nothing has run on hardware.
    #[test]
    fn a_host_fake_launcher_is_not_reported_as_offloaded() {
        use crate::launch::HostFakeLaunch;
        let asc = AscendDispatcher::new(CpuDispatcher)
            .with_rms_norm(Box::new(HostFakeLaunch::new("rms_norm")));
        assert!(
            asc.offloaded_by().is_empty(),
            "a host fake reported as offloaded: {:?}",
            asc.offloaded_by()
        );
    }

    /// A launcher returning a malformed result must PANIC, not fall back to the CPU. A silent
    /// fallback turns a broken kernel into a slow-but-correct model, which is the outcome nobody
    /// notices.
    #[test]
    #[should_panic(expected = "expected 4")]
    fn a_malformed_launcher_result_panics_rather_than_falling_back() {
        use crate::launch::{Corruption, HostFakeLaunch};
        let asc = AscendDispatcher::new(CpuDispatcher).with_rms_norm(Box::new(
            HostFakeLaunch::with_corruption("rms_norm", Corruption::ShortOutput),
        ));
        // A SERVED width (4096): the shape gate routes unemitted widths to the CPU by policy,
        // so a 4-wide fixture would delegate and never reach the corrupt launcher.
        let _ = asc.rms_norm(&vec![1.0f32; 4096], &vec![1.0f32; 4096], 1e-6);
    }

    /// `offloaded()` is the honest answer to "is any of this on the NPU?". It must stay in step
    /// with reality: today nothing launches, so it must be empty. When a method is swapped this
    /// test is the reminder to say so.
    #[test]
    fn offloaded_reports_nothing_until_a_kernel_is_actually_launched() {
        assert!(
            offloaded().is_empty(),
            "offloaded() lists {:?} — if a method now launches a PTO kernel, this test should be \
             updated deliberately, not deleted",
            offloaded()
        );
    }
}

#[cfg(test)]
mod q8_seam_tests {
    use super::*;
    use engine_core::dispatch::{CpuDispatcher, KernelDispatcher};

    /// A host launcher that computes the split-plane matvec the way the DEVICE kernel defines
    /// it: `out[o] = Σ_b scale[o,b] · Σ_i codes[o,b,i] · x[b*32+i]`. Comparing the dispatcher
    /// through this against `CpuDispatcher::matvec_q8` on the SAME GGUF bytes validates the
    /// SPLIT and the routing — if the plane layout drifted from the interleaved format, the two
    /// would disagree.
    struct HostQ8Fake;
    impl crate::launch::KernelLaunch for HostQ8Fake {
        fn name(&self) -> &str {
            "host-q8-fake"
        }
        fn is_device(&self) -> bool {
            false
        }
        fn rms_norm(&self, _x: &[f32], _g: &[f32], _e: f32) -> anyhow::Result<Vec<f32>> {
            anyhow::bail!("not under test")
        }
        fn matvec_q8(
            &self,
            codes: &[i8],
            scales: &[f32],
            x: &[f32],
            d_out: usize,
        ) -> anyhow::Result<Vec<f32>> {
            let d_in = x.len();
            let mut out = vec![0f32; d_out];
            for o in 0..d_out {
                let mut acc = 0f32;
                for j in 0..d_in {
                    acc += scales[o * d_in + j] * codes[o * d_in + j] as f32 * x[j];
                }
                out[o] = acc;
            }
            Ok(out)
        }
    }

    fn synth_q8(d_in: usize, d_out: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
        let nb = d_in / 32;
        let mut s = seed | 1;
        let mut rnd = move || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s & 0xFFFF) as f32 / 65536.0 - 0.5
        };
        let mut w = vec![0u8; d_out * nb * 34];
        for blk in w.chunks_exact_mut(34) {
            let sc = 0.25 + rnd().abs();
            let bits = {
                let b = sc.to_bits();
                let sign = ((b >> 16) & 0x8000) as u16;
                let exp = (((b >> 23) & 0xff) as i32) - 127 + 15;
                let man = ((b >> 13) & 0x3ff) as u16;
                if exp <= 0 { sign } else { sign | ((exp as u16) << 10) | man }
            };
            blk[0] = (bits & 0xff) as u8;
            blk[1] = (bits >> 8) as u8;
            for i in 0..32 {
                blk[2 + i] = ((rnd() * 200.0) as i32).clamp(-127, 127) as u8;
            }
        }
        let x = (0..d_in).map(|_| rnd()).collect();
        (w, x)
    }

    #[test]
    fn split_planes_agree_with_the_interleaved_reference() {
        // A KNOWN device shape, so the route goes through the launcher.
        let (w, x) = synth_q8(4096, 1024, 0xBEEF);
        let d = AscendDispatcher::new(CpuDispatcher).with_matvec_q8(Box::new(HostQ8Fake));
        let got = d.matvec_q8(&w, &x, 1024);
        let want = CpuDispatcher.matvec_q8(&w, &x, 1024);
        let worst = (0..1024)
            .map(|i| ((got[i] - want[i]).abs() as f64) / (want[i].abs() as f64).max(1e-3))
            .fold(0.0f64, f64::max);
        assert!(got.iter().any(|v| *v != 0.0), "vacuous agreement");
        // The fake accumulates per block in f32 like the kernel; the CPU reference dequantises
        // then dots. Same values, different order: tolerance, not equality. Measured 3.2e-4 at
        // this seed — reorder noise over 128 cancelling blocks. A LAYOUT bug (a scale plane off
        // by one block, codes misaligned) measures O(1), so 1e-3 still separates the failure
        // this test exists to catch.
        assert!(worst < 1e-3, "split planes disagree with the interleaved bytes: {worst:.3e}");
    }

    #[test]
    fn unknown_shapes_delegate_even_with_a_launcher_installed() {
        // 64 -> 8 is no ELF kernel; the fake would return garbage for it, so agreement with the
        // CPU proves DELEGATION, not luck.
        let (w, x) = synth_q8(64, 8, 0xACE1);
        let d = AscendDispatcher::new(CpuDispatcher).with_matvec_q8(Box::new(HostQ8Fake));
        let got = d.matvec_q8(&w, &x, 8);
        let want = CpuDispatcher.matvec_q8(&w, &x, 8);
        assert_eq!(got, want, "unknown shape must take the CPU path bit-for-bit");
    }

    #[test]
    fn a_host_fake_is_not_reported_as_offload() {
        let d = AscendDispatcher::new(CpuDispatcher).with_matvec_q8(Box::new(HostQ8Fake));
        assert!(d.offloaded_by().is_empty(), "is_device()=false must not be reported");
    }
}

#[cfg(test)]
mod q8_alias_tests {
    use super::*;
    use engine_core::dispatch::{CpuDispatcher, KernelDispatcher};

    /// The alias route: a matvec_f32 call on a REGISTERED plane must produce the q8 kernel's
    /// result (via a host fake computing from the split planes), and agree with the f32 CPU
    /// matvec on the dequant of the same bytes — proving the alias maps the right weight.
    #[test]
    fn aliased_plane_reroutes_matvec_f32_through_q8() {
        struct Fake;
        impl crate::launch::KernelLaunch for Fake {
            fn name(&self) -> &str {
                "host-q8-fake"
            }
            fn is_device(&self) -> bool {
                false
            }
            fn rms_norm(&self, _: &[f32], _: &[f32], _: f32) -> anyhow::Result<Vec<f32>> {
                anyhow::bail!("not under test")
            }
            fn matvec_q8(
                &self,
                codes: &[i8],
                scales: &[f32],
                x: &[f32],
                d_out: usize,
            ) -> anyhow::Result<Vec<f32>> {
                let d_in = x.len();
                let mut out = vec![0f32; d_out];
                for o in 0..d_out {
                    let mut acc = 0f32;
                    for j in 0..d_in {
                        acc += scales[o * d_in + j] * codes[o * d_in + j] as f32 * x[j];
                    }
                    out[o] = acc;
                }
                Ok(out)
            }
        }
        // A served shape: 4096 -> 1024.
        let (d_in, d_out) = (4096usize, 1024usize);
        let nb = d_in / 32;
        let mut s = 0xACE1u32 | 1;
        let mut rnd = move || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s & 0xFFFF) as f32 / 65536.0 - 0.5
        };
        let mut q8 = vec![0u8; d_out * nb * 34];
        for blk in q8.chunks_exact_mut(34) {
            let sc = 0.25 + rnd().abs();
            let bits = {
                let b = sc.to_bits();
                let sign = ((b >> 16) & 0x8000) as u16;
                let exp = (((b >> 23) & 0xff) as i32) - 127 + 15;
                let man = ((b >> 13) & 0x3ff) as u16;
                if exp <= 0 { sign } else { sign | ((exp as u16) << 10) | man }
            };
            blk[0] = (bits & 0xff) as u8;
            blk[1] = (bits >> 8) as u8;
            for i in 0..32 {
                blk[2 + i] = ((rnd() * 200.0) as i32).clamp(-127, 127) as u8;
            }
        }
        // Dequant twin, same expression the loader uses.
        let mut f32_plane = vec![0f32; d_out * d_in];
        for (b, blk) in q8.chunks_exact(34).enumerate() {
            let d = engine_core::layer_view::f16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
            for i in 0..32 {
                f32_plane[b * 32 + i] = d * (blk[2 + i] as i8) as f32;
            }
        }
        let x: Vec<f32> = (0..d_in).map(|_| rnd()).collect();

        let disp = AscendDispatcher::new(CpuDispatcher).with_matvec_q8(Box::new(Fake));
        // Unregistered: bit-identical CPU delegation.
        let plain = disp.matvec_f32(&f32_plane, &x, d_out);
        assert_eq!(plain, CpuDispatcher.matvec_f32(&f32_plane, &x, d_out));
        // Registered: reroutes through the q8 seam.
        unsafe { disp.register_q8_alias(&f32_plane, &q8) };
        assert_eq!(disp.q8_aliases(), 1);
        let via_q8 = disp.matvec_f32(&f32_plane, &x, d_out);
        let want = CpuDispatcher.matvec_f32(&f32_plane, &x, d_out);
        let worst = (0..d_out)
            .map(|i| ((via_q8[i] - want[i]).abs() as f64) / (want[i].abs() as f64).max(1e-3))
            .fold(0.0f64, f64::max);
        assert!(via_q8.iter().any(|v| *v != 0.0), "vacuous");
        assert!(worst < 1e-3, "aliased q8 route disagrees with the dequant twin: {worst:.3e}");
        // An UNSERVED shape with an alias still delegates (policy).
        let f32_small = vec![0.5f32; 64 * 8];
        let q8_small = vec![0u8; 64 / 32 * 34 * 8];
        unsafe { disp.register_q8_alias(&f32_small, &q8_small) };
        let xs = vec![0.25f32; 64];
        assert_eq!(
            disp.matvec_f32(&f32_small, &xs, 8),
            CpuDispatcher.matvec_f32(&f32_small, &xs, 8)
        );
    }
}

#[cfg(test)]
mod expert_provider_tests {
    use super::*;
    use engine_core::dispatch::{CpuDispatcher, KernelDispatcher};

    /// The provider's selected-only stack must be BIT-IDENTICAL to the free
    /// `moe_routed_step` on a full pre-dequanted stack: same dequant, same math, only the
    /// indexing differs — so any divergence is an indexing bug, exactly what this catches.
    #[test]
    fn provided_experts_match_the_full_stack_bit_for_bit() {
        let (d_model, d_ffn, n_experts) = (64usize, 32usize, 4usize);
        let blk_gu = d_ffn * d_model / 32 * 17;
        let blk_dn = d_model * d_ffn / 32 * 17;
        let mut s = 0x5EED_u32 | 1;
        let mut rnd = move || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        };
        let mut synth = |n: usize| -> Vec<u8> {
            let mut v = vec![0u8; n];
            for (i, b) in v.iter_mut().enumerate() {
                *b = if i % 17 == 0 {
                    120 + (rnd() % 16) as u8 // e8m0 exponents near 1.0
                } else {
                    (rnd() & 0xFF) as u8
                };
            }
            v
        };
        let gate = synth(n_experts * blk_gu);
        let up = synth(n_experts * blk_gu);
        let down = synth(n_experts * blk_dn);

        let disp = AscendDispatcher::new(CpuDispatcher);
        unsafe {
            disp.register_expert_table(
                7,
                engine_core::gguf::GgmlType::MXFP4,
                &gate,
                &up,
                &down,
                n_experts,
                d_ffn,
                d_model,
            )
        };
        let x: Vec<f32> = (0..d_model).map(|i| (i as f32 * 0.37).sin() * 0.5).collect();
        let selected = [2usize, 0, 3];
        let weights = [0.5f32, 0.3, 0.2];
        // Chain-shaped call: EMPTY f32 slices, d_ffn 0 — what the composed model passes.
        let got = disp.moe_routed_step(7, &x, &selected, &weights, &[], &[], &[], 0);
        // Reference: full dequanted stack through the free function.
        let g_full = engine_core::glm5_stream::dequant_mxfp4(&gate);
        let u_full = engine_core::glm5_stream::dequant_mxfp4(&up);
        let d_full = engine_core::glm5_stream::dequant_mxfp4(&down);
        let want = engine_core::moe::moe_routed_step(
            &x, &selected, &weights, &g_full, &u_full, &d_full, d_ffn,
        );
        assert!(got.iter().any(|v| *v != 0.0), "vacuous — routed output must be non-zero");
        assert_eq!(got, want, "selected-only stack diverged from the full stack");
        // An UNREGISTERED layer still delegates to the inner (which needs real slices).
        let got2 = disp.moe_routed_step(9, &x, &selected, &weights, &g_full, &u_full, &d_full, d_ffn);
        assert_eq!(got2, want);
    }
}

#[cfg(test)]
mod expert_device_path_tests {
    use super::*;
    use engine_core::dispatch::{CpuDispatcher, KernelDispatcher};

    /// The DEVICE expert path (repack + permute + kernel-contract matvec + CPU glue) must agree
    /// with the CPU dequant path on the same synthetic MXFP4 bytes. The fake launcher computes
    /// `mxfp4_repack::matvec_ref` — the same oracle the device kernel was proven against — so
    /// what this test pins is the DISPATCHER's assembly: repack slicing, permutation, clamp and
    /// weight-fold order.
    #[test]
    fn device_expert_path_matches_cpu_dequant_path() {
        struct Fake;
        impl crate::launch::KernelLaunch for Fake {
            fn name(&self) -> &str {
                "host-mx4-fake"
            }
            fn is_device(&self) -> bool {
                false
            }
            fn rms_norm(&self, _: &[f32], _: &[f32], _: f32) -> anyhow::Result<Vec<f32>> {
                anyhow::bail!("not under test")
            }
            fn matvec_mx4pk(
                &self,
                w: &[u8],
                scales: &[f32],
                x_perm: &[f32],
                d_out: usize,
            ) -> anyhow::Result<Vec<f32>> {
                let pk = crate::mxfp4_repack::PackedWeight { w: w.to_vec(), d: scales.to_vec() };
                Ok(crate::mxfp4_repack::matvec_ref(&pk, x_perm, x_perm.len(), d_out))
            }
        }
        // REAL shapes — the device path's shape gate demands them.
        let (d_model, d_ffn, n_experts) = (4096usize, 2048usize, 2usize);
        let blk_gu = d_ffn * d_model / 32 * 17;
        let blk_dn = d_model * d_ffn / 32 * 17;
        let mut s = 0x00D5_u32 | 1;
        let mut rnd = move || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        };
        let mut synth = |n: usize| -> Vec<u8> {
            let mut v = vec![0u8; n];
            for (i, b) in v.iter_mut().enumerate() {
                *b = if i % 17 == 0 { 121 + (rnd() % 12) as u8 } else { (rnd() & 0xFF) as u8 };
            }
            v
        };
        let gate = synth(n_experts * blk_gu);
        let up = synth(n_experts * blk_gu);
        let down = synth(n_experts * blk_dn);
        let x: Vec<f32> = (0..d_model).map(|i| ((i as f32) * 0.013).sin() * 0.3).collect();
        let selected = [1usize, 0];
        let weights = [0.6f32, 0.4];

        let reg = |d: &AscendDispatcher<CpuDispatcher>| unsafe {
            d.register_expert_table(
                3,
                engine_core::gguf::GgmlType::MXFP4,
                &gate,
                &up,
                &down,
                n_experts,
                d_ffn,
                d_model,
            )
        };
        // CPU dequant path (no launcher).
        let cpu = AscendDispatcher::new(CpuDispatcher);
        reg(&cpu);
        let want = cpu.moe_routed_step(3, &x, &selected, &weights, &[], &[], &[], 0);
        // Device path (fake launcher, matvec_ref semantics).
        let dev = AscendDispatcher::new(CpuDispatcher).with_matvec_mx4pk(Box::new(Fake));
        reg(&dev);
        let got = dev.moe_routed_step(3, &x, &selected, &weights, &[], &[], &[], 0);
        assert!(got.iter().any(|v| *v != 0.0), "vacuous");
        let worst = (0..d_model)
            .map(|i| ((got[i] - want[i]).abs() as f64) / (want[i].abs() as f64).max(1e-4))
            .fold(0.0f64, f64::max);
        // matvec_ref accumulates in f64, the CPU path in f32 — tolerance, not equality; a
        // repack/permute/fold mistake measures O(1) here.
        assert!(worst < 1e-3, "device expert path diverged: {worst:.3e}");
    }
}

#[cfg(test)]
mod w8a8_moe_tests {
    use super::*;
    use engine_core::dispatch::CpuDispatcher;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// Records how many w8a8 expert launches it saw, and returns a fixed vector so the
    /// composition can be predicted exactly.
    struct Counting {
        calls: Arc<AtomicUsize>,
        out: Vec<f32>,
    }

    impl crate::launch::KernelLaunch for Counting {
        fn name(&self) -> &str { "w8a8-counting" }
        fn is_device(&self) -> bool { true }
        fn rms_norm(&self, x: &[f32], w: &[f32], eps: f32) -> anyhow::Result<Vec<f32>> {
            // required by the trait, unused here: the w8a8 path never calls it
            Ok(engine_core::forward::rms_norm(x, w, eps))
        }
        fn matvec_w8a8_experts(
            &self,
            _name: &str,
            codes: &[i8],
            scale: &[f32],
            offset: &[f32],
            x: &[f32],
            d_out: usize,
        ) -> anyhow::Result<Vec<f32>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            // refuse quietly if the caller handed us the wrong planes, so a wiring mistake shows up
            assert_eq!(scale.len(), d_out, "scale plane must be one per output");
            assert_eq!(offset.len(), d_out);
            assert!(!codes.is_empty() && !x.is_empty());
            Ok(self.out.clone())
        }
    }

    /// THE POINT OF THIS TEST: a w8a8 MoE step must go THROUGH the launcher. If the method
    /// silently fell back to the host, the count would be zero and the assertion would fail --
    /// which is the difference between "it computes" and "it computes on the device".
    #[test]
    fn the_moe_step_actually_calls_the_w8a8_launcher_three_times_per_expert() {
        let calls = Arc::new(AtomicUsize::new(0));
        let l = Counting { calls: calls.clone(), out: vec![1.0f32; 8] };
        let dev = AscendDispatcher::new(CpuDispatcher).with_matvec_w8a8_experts(Box::new(l));

        let plane = W8a8Expert { codes: vec![1i8; 16], scale: vec![0.5f32; 8], offset: vec![0.0f32; 8] };
        let out = dev
            .moe_routed_step_w8a8(&[0.1f32; 8], &[0, 1], &[0.5, 0.5], &[plane.clone(), plane.clone(), plane], 8)
            .expect("the launcher is registered, so this must not refuse");

        assert_eq!(out.len(), 8);
        // three projections per expert, two experts selected
        assert_eq!(calls.load(Ordering::SeqCst), 6, "gate, up and down for each selected expert");
    }

    /// And with no launcher it REFUSES rather than quietly computing on the host.
    #[test]
    fn without_a_launcher_the_step_refuses_instead_of_falling_back() {
        let dev = AscendDispatcher::new(CpuDispatcher);
        let plane = W8a8Expert { codes: vec![1i8; 16], scale: vec![0.5f32; 8], offset: vec![0.0f32; 8] };
        let err = dev
            .moe_routed_step_w8a8(&[0.1f32; 8], &[0], &[1.0], &[plane.clone(), plane.clone(), plane], 8)
            .expect_err("no launcher registered");
        assert!(format!("{err}").contains("no w8a8 expert launcher"), "got: {err}");
    }
}
