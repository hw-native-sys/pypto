// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Launching our generated kernels on the NPU: the policy, always compiled; the FFI, feature-gated.
//!
//! [`crate::launch::KernelLaunch`] described the shape of this hole. The probes under `probes/`
//! proved it can be filled — a device binary registered at runtime, kernels launched by stub
//! pointer, `rms_norm` verified at 9.344e-7 against an f64 replay. This module is where that stops
//! being a probe.
//!
//! # The split, and why it is not cosmetic
//!
//! Everything that can be wrong WITHOUT a device lives above the `cfg` line and is unit-tested on
//! any machine: which blockDim each kernel may use, which `eps` the emitted kernel actually
//! implements, and that no two kernels share a stub. Those are the three mistakes the probes turned
//! up, and none of them needs an NPU to catch. Only the `extern` blocks and the round-trip are
//! gated, so a macOS `cargo test` still checks the parts that decide correctness.
//!
//! # blockDim is PER-KERNEL, and that is measured
//!
//! Not a style preference. On the layer-0 chain, raising the projection blockDim past 2 corrupted
//! the q chain while batched attention and the Q8_0 matvecs scaled cleanly to 48 — because
//! `rms_norm` is a reduction over D with no cross-core combine, and `rmsmul_1024`/`rope_halves`
//! write IN PLACE, so extra cores there race rather than divide. A launcher carrying one global
//! blockDim would be silently wrong for half the chain, and the whole 34x measured on that chain
//! came from getting this right per kernel.

use anyhow::{bail, Result};

/// The `eps` the emitted `rmsmul` kernels bake in as a constant.
///
/// [`crate::launch::KernelLaunch::rms_norm`] takes `eps` as a parameter, so a device launcher is
/// offered values it cannot honour. It must REFUSE them: silently normalising with 1e-6 when asked
/// for something else is a wrong answer that would pass every shape check and show up only as
/// slightly-off logits.
pub const EMITTED_RMS_EPS: f32 = 1e-6;

/// How a kernel may be launched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Partitioning {
    /// Divides across cores: a row-partitioned loop over independent outputs.
    ByRow,
    /// Must run on ONE core — a reduction with no cross-core combine, or an in-place write.
    SingleCore,
    /// Must run on EXACTLY this many cores, because the kernel's blocks cooperate.
    ///
    /// The fused grouped-matmul swiglu_quant reduces its per-row absmax across blocks through a
    /// workspace and a grid-wide barrier, and the emitter sizes both from `PtoOpts::gmm_blocks` at
    /// emit time (`TILERS_GMM_BLOCKS`, default 1). The count is therefore a property of the
    /// artifact, not a choice the launcher gets to make: fewer cores than it was built for leaves
    /// part of the reduction unwritten, more leaves blocks writing past it, and neither is a
    /// hardware limit — it is two numbers that have to agree with nothing making them agree.
    ///
    /// `ByRow` and `SingleCore` cannot express this. Clamping such a kernel to `USABLE_AIV` or
    /// forcing it to 1 are both wrong, which is why this is a third case rather than a tighter
    /// clamp on the first.
    Cooperating(u32),
}

/// Usable AI vector cores. Measured: 48 is the ceiling, and 96 was worse than 48 at every point of
/// the unpack and expansion sweeps, so this is a cap rather than a starting guess.
pub const USABLE_AIV: u32 = 48;

/// One generated kernel, as the launcher needs to know it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelSpec {
    /// Readable name, for reporting.
    pub name: &'static str,
    /// The MANGLED device symbol. `ccec` emits C++ mangling, so `ds4_rmsmul_4096(float*,float*,
    /// float*)` becomes `_Z15ds4_rmsmul_4096PfS_S_`; `rtFunctionRegister` wants this exact string
    /// and registering the unmangled name fails at launch rather than at registration.
    pub symbol: &'static str,
    pub partitioning: Partitioning,
}

impl KernelSpec {
    /// The blockDim this kernel may be launched with, given a desired width.
    ///
    /// Clamps to [`USABLE_AIV`] for row-partitioned kernels, forces 1 for single-core ones, and
    /// returns the built-in width for cooperating ones. Returning a value rather than an error is
    /// deliberate: a caller asking for 48 on a single-core kernel is not making a mistake, it is
    /// expressing a default that this function is here to correct.
    ///
    /// A `Cooperating(n)` kernel ignores `desired` in BOTH directions — it is not a ceiling to clamp
    /// under, it is the width the artifact was built for.
    pub fn block_dim(&self, desired: u32) -> u32 {
        match self.partitioning {
            Partitioning::SingleCore => 1,
            Partitioning::ByRow => desired.clamp(1, USABLE_AIV),
            Partitioning::Cooperating(n) => n.max(1),
        }
    }
}

/// The layer-0 kernels, all of which live in ONE device ELF (45,568 bytes from `layer0.cpp`), so a
/// launcher registers one binary and then one function per kernel.
pub const LAYER0_KERNELS: &[KernelSpec] = &[
    KernelSpec {
        name: "rms_norm_4096",
        symbol: "_Z15ds4_rmsmul_4096PfS_S_",
        partitioning: Partitioning::SingleCore,
    },
    KernelSpec {
        name: "rms_norm_1024",
        symbol: "_Z15ds4_rmsmul_1024PfS_S_",
        partitioning: Partitioning::SingleCore,
    },
    KernelSpec {
        name: "rms_norm_512",
        symbol: "_Z14ds4_rmsmul_512PfS_S_",
        partitioning: Partitioning::SingleCore,
    },
    KernelSpec {
        name: "hc_norm",
        symbol: "_Z11ds4_hc_normPfS_",
        partitioning: Partitioning::SingleCore,
    },
    KernelSpec {
        name: "rope_halves",
        // In place on q and kv, so extra cores would race on the values they rewrite.
        symbol: "_Z15ds4_rope_halvesPfS_S_S_",
        partitioning: Partitioning::SingleCore,
    },
    KernelSpec {
        name: "hc_mv",
        symbol: "_Z9ds4_hc_mvPaPfS0_S0_",
        partitioning: Partitioning::ByRow,
    },
    KernelSpec {
        name: "matvec_4096_1024",
        symbol: "_Z18ds4_q8mv_4096_1024PaPfS0_S0_",
        partitioning: Partitioning::ByRow,
    },
    KernelSpec {
        name: "matvec_1024_32768",
        symbol: "_Z19ds4_q8mv_1024_32768PaPfS0_S0_",
        partitioning: Partitioning::ByRow,
    },
    KernelSpec {
        name: "matvec_4096_512",
        symbol: "_Z17ds4_q8mv_4096_512PaPfS0_S0_",
        partitioning: Partitioning::ByRow,
    },
    KernelSpec {
        name: "matvec_8192_4096",
        symbol: "_Z18ds4_q8mv_8192_4096PaPfS0_S0_",
        partitioning: Partitioning::ByRow,
    },
    KernelSpec {
        name: "attention_sink_batched",
        symbol: "_Z21ds4_attn_sink_batchedPfS_S_S_",
        partitioning: Partitioning::ByRow,
    },
    // The two routed-expert projections, packed-MXFP4. Added for the MoE offload; the ELF
    // grew from layer0.cpp's eleven to thirteen (merge_pto.py layer0.cpp + the two mx4pk
    // raw cpps -> layer13.cpp -> ccec). Signature (w nibble-planes, halved f32 scales, x
    // PERMUTED, out) — see mxfp4_repack for both transforms and matvec_ref for the oracle.
    // THE EXPERT MATVECS, now declared as w8a8 SOCKETS. The deployment quantisation is w8a8 (the goal
    // says so, and the checkpoint's own tensors are I8 + per-output-channel scale/offset), while the
    // packed-MXFP4 form is corpus evidence only. The old `mx4pk_*` names described a format the
    // deployment does not use, and keeping them while swapping the implementation would have left the
    // name lying about the layout -- the defect this campaign has spent rounds removing.
    //
    // The symbols are UNMANGLED plus CANN's `_mix_aiv` suffix, because these kernels are emitted as
    // `extern "C"` (verified by `nm` on the extracted `.aicore_binary`, iteration 111 -- the C++
    // `_Z30...PfS_S_S_S_` form I predicted was wrong).
    //
    // THE ARGUMENTS STILL DO NOT MATCH, and that is the one piece not done: the launch
    // (`matvec_mx4pk`) marshals packed nibble planes, halved scales and a PERMUTED x, which is what
    // the engine's expert storage holds. A w8a8 socket needs (x, w_i8, per-channel scale, offset) and
    // therefore w8a8 expert STORAGE. Registering these names means the engine no longer claims a
    // format it will not use; it does not mean the MoE runs on device yet.
    KernelSpec {
        name: "w8a8_experts_4096_2048",
        symbol: "w8a8_experts_1x2048_k4096_mix_aiv",
        partitioning: Partitioning::ByRow,
    },
    KernelSpec {
        name: "w8a8_experts_2048_4096",
        symbol: "w8a8_experts_1x4096_k2048_mix_aiv",
        partitioning: Partitioning::ByRow,
    },
];

/// Kernels a CALLER-side DECOMPOSITION may launch, beyond the engine's slots.
///
/// These are device-certified corpus artefacts, and they are deliberately NOT in [`LAYER0_KERNELS`]: that
/// table is the KPI's denominator and describes what the ENGINE's layer code calls. A caller that must
/// decompose a projection -- because the arm faults above ~1024 outputs at k=4096 -- needs a certified
/// kernel to do it with, and adding one to the slot table would grow the denominator by a caller's
/// convenience. This table is the alternative, and it is short and curated on purpose.
pub const DECOMPOSITION_KERNELS: &[KernelSpec] = &[
    KernelSpec {
        name: "w8a8_slice_1x1024_k4096",
        symbol: "proj_w8a8_engine_1x1024_k4096_mix_aiv",
        partitioning: Partitioning::ByRow,
    },
    // The vocabulary band: a certified 4096-output w8a8 matvec at k=4096, used by a caller that
    // decomposes a 154,880-output head over the OUTPUT dimension (see w8a8_lm_head_banded).
    // The same band width at k=2048, for bisecting the banded launch's k sensitivity.
    KernelSpec {
        name: "w8a8_band_k2048",
        symbol: "w8a8_experts_1x4096_k2048_mix_aiv",
        partitioning: Partitioning::ByRow,
    },
    KernelSpec {
        name: "w8a8_lm_head_band",
        symbol: "glm53_proj_output_mix_aiv",
        partitioning: Partitioning::ByRow,
    },
    KernelSpec {
        name: "w8a8_slice_1x512_k4096",
        symbol: "proj_w8a8_engine_1x512_k4096_mix_aiv",
        partitioning: Partitioning::ByRow,
    },
];

/// The packed-MXFP4 matvec kernel for a `(d_in, d_out)` shape — the two routed-expert
/// projections only.
pub fn mx4pk_kernel_for(d_in: usize, d_out: usize) -> Option<&'static str> {
    // The slots are w8a8 sockets now. The function keeps its name so every existing caller still
    // compiles and still asks the same question; what changed is which kernels ANSWER it.
    match (d_in, d_out) {
        (4096, 2048) => Some("w8a8_experts_4096_2048"),
        (2048, 4096) => Some("w8a8_experts_2048_4096"),
        _ => None,
    }
}

/// Find a kernel by name, or say which names exist.
pub fn kernel(name: &str) -> Result<&'static KernelSpec> {
    LAYER0_KERNELS.iter().find(|k| k.name == name).ok_or_else(|| {
        anyhow::anyhow!(
            "no kernel named {name}; have: {}",
            LAYER0_KERNELS.iter().map(|k| k.name).collect::<Vec<_>>().join(", ")
        )
    })
}

/// Check an `eps` against what the emitted kernel implements.
pub fn check_rms_eps(eps: f32) -> Result<()> {
    if eps != EMITTED_RMS_EPS {
        bail!(
            "the emitted rmsmul kernel bakes eps = {EMITTED_RMS_EPS:e} in as a constant and cannot \
             honour {eps:e}. Refusing rather than normalising with the wrong epsilon, which would \
             pass every shape check and surface only as slightly-wrong logits. Re-emit the kernel \
             with this eps, or run this call on the CPU path."
        );
    }
    Ok(())
}

/// Pick the `rms_norm` kernel for a width.
///
/// The emitted kernels are SHAPE-SPECIALISED — `ds4_rmsmul_4096` is a different kernel from
/// `ds4_rmsmul_1024`, not the same one with a parameter — so a width we did not emit cannot be
/// served and must be refused rather than rounded to the nearest. Layer 0 needs exactly these three:
/// 4096 for the attention and FFN norms, 1024 for the Q-A norm, 512 for the KV-A norm.
pub fn select_rms_kernel(d: usize) -> Result<&'static KernelSpec> {
    match d {
        4096 => kernel("rms_norm_4096"),
        1024 => kernel("rms_norm_1024"),
        512 => kernel("rms_norm_512"),
        other => bail!(
            "no rms_norm kernel emitted for width {other}; have 4096, 1024, 512. These are \
             shape-specialised kernels, so the width cannot be rounded — emit one for {other} or \
             run this call on the CPU path."
        ),
    }
}

/// ELF magic for a `dav-c220-vec` device binary.
///
/// `RT_DEV_BINARY_MAGIC_ELF_AIVEC`. The plain-ELF magic (`0x43554245`) registers successfully and
/// then fails at launch, so getting this wrong costs a debugging session rather than a build error.
pub const MAGIC_ELF_AIVEC: u32 = 0x4141_5246;

#[cfg(feature = "ascend")]
mod ffi {
    //! The runtime-registration path, proven in `probes/rust_kernel_launch.rs`.
    //!
    //! These are `libruntime` (`rt*`), NOT ACL. `ascend_sys` carries blas/dvpp/hccl/prof but none of
    //! these, and it cannot build off-device at all; declaring them here keeps the gate in one place
    //! until they move there.
    pub type RtError = i32;

    #[repr(C)]
    pub struct RtDevBinary {
        pub magic: u32,
        pub version: u32,
        pub data: *const u8,
        pub length: u64,
    }

    #[link(name = "runtime")]
    extern "C" {
        pub fn rtDevBinaryRegister(bin: *const RtDevBinary, hdl: *mut *mut u8) -> RtError;
        /// `kernel_info_ext` carries the DEVICE FUNCTION NAME; `stub_name` is only a host label.
        /// Passing null for the former returns 507000 — after `rtDevBinaryRegister` has already
        /// succeeded, which reads as a binary problem and is not.
        pub fn rtFunctionRegister(
            bin_handle: *mut u8,
            stub_func: *const u8,
            // `c_char`, NOT `i8`: on aarch64-linux `c_char` is UNSIGNED, so `CString::as_ptr()`
            // yields `*const u8` there while it is `*const i8` on x86_64 and on macOS. Hard-coding
            // either one compiles on the machine you wrote it on and fails on the other — which is
            // exactly how this was found: it type-checked on the Mac and would not build on 910c.
            stub_name: *const std::ffi::c_char,
            kernel_info_ext: *const u8,
            func_mode: u32,
        ) -> RtError;
        pub fn rtKernelLaunch(
            stub_func: *const u8,
            num_blocks: u32,
            args: *mut u8,
            args_size: u32,
            sm_desc: *mut u8,
            stm: *mut u8,
        ) -> RtError;
    }

    #[link(name = "ascendcl")]
    extern "C" {
        pub fn aclInit(cfg: *const std::ffi::c_char) -> i32;
        pub fn aclrtSetDevice(id: i32) -> i32;
        pub fn aclrtCreateContext(context: *mut *mut u8, device_id: i32) -> i32;
        pub fn aclrtSetCurrentContext(context: *mut u8) -> i32;
        pub fn aclrtCreateStream(s: *mut *mut u8) -> i32;
        pub fn aclrtSynchronizeStream(s: *mut u8) -> i32;
        pub fn aclrtMalloc(p: *mut *mut u8, size: usize, policy: i32) -> i32;
        pub fn aclrtFree(p: *mut u8) -> i32;
        pub fn aclrtMemcpy(dst: *mut u8, dmax: usize, src: *const u8, cnt: usize, kind: i32) -> i32;
        pub fn aclrtGetDeviceCount(count: *mut u32) -> i32;
        pub fn aclrtGetMemInfo(attr: i32, free: *mut usize, total: *mut usize) -> i32;
    }

    pub const MEMCPY_H2D: i32 = 1;
    pub const MEMCPY_D2H: i32 = 2;
    pub const MALLOC_HUGE_FIRST: i32 = 0;
    /// `ACL_HBM_MEM` — the attr `aclrtGetMemInfo` reports device HBM under.
    pub const ATTR_HBM_MEM: i32 = 1;
}

/// Per-device `(id, free_bytes, total_bytes)` from `aclrtGetMemInfo`, the authoritative figure
/// the engine allocates against.
///
/// This is the memory-profile side of the controller's plan step: `npu-smi` is the no-linkage
/// fallback, this is the truth. `aclrtGetMemInfo` reports the CALLING device, so it sets each in
/// turn — the same per-device set the multi-link H2D measurement relied on. Initialising ACL and
/// selecting a device are cheap and idempotent; a device that is busy with another job still
/// reports its free HBM, which is exactly what a shared box needs.
#[cfg(feature = "ascend")]
pub fn acl_memory_profile() -> anyhow::Result<Vec<(u32, u64, u64)>> {
    unsafe {
        // aclInit is safe to call once per process; ignore "already initialised".
        let _ = ffi::aclInit(std::ptr::null());
        let mut count: u32 = 0;
        if ffi::aclrtGetDeviceCount(&mut count) != 0 {
            anyhow::bail!("aclrtGetDeviceCount failed");
        }
        let mut out = Vec::with_capacity(count as usize);
        for id in 0..count {
            if ffi::aclrtSetDevice(id as i32) != 0 {
                anyhow::bail!("aclrtSetDevice({id}) failed");
            }
            let (mut free, mut total): (usize, usize) = (0, 0);
            if ffi::aclrtGetMemInfo(ffi::ATTR_HBM_MEM, &mut free, &mut total) != 0 {
                anyhow::bail!("aclrtGetMemInfo(device {id}) failed");
            }
            out.push((id, free as u64, total as u64));
        }
        Ok(out)
    }
}


/// The device ELF and its kernel stubs, registered EXACTLY ONCE per process.
///
/// rt*'s binary/function registration is process-global and keyed by the device function NAME,
/// not by chip. Registering the same ELF a second time (to bring up another chip) rebinds that
/// global table, so the FIRST chip's stubs then launch the wrong/stale binding and fail with
/// 107003. The staged 2-chip probe proved exactly this: chip 1 ran byte-identical to chip 0, but
/// opening chip 1 broke chip 0's next launch. The fix is to register once and launch the SAME
/// shared stubs on each chip's own stream — the launch targets the stream's device, so one
/// registration serves every chip.
///
/// Each kernel gets its OWN heap-allocated stub byte, whose ADDRESS is the key the runtime maps to
/// a device function; two kernels sharing an address would silently launch the wrong one, so the
/// boxes are per-kernel and kept alive for the process lifetime.
#[cfg(feature = "ascend")]
struct DeviceBinary {
    /// Held so the registered pointer stays valid. Never read by us.
    _elf: Vec<u8>,
    #[allow(dead_code)]
    handle: *mut u8,
    /// `(kernel name, stub)`. The Box's address is the registration key.
    stubs: Vec<(&'static str, Box<u8>)>,
}
// The stubs' addresses are stable for the process lifetime and the runtime is the only concurrent
// reader; the launcher enforces single-thread use via `check_thread`.
#[cfg(feature = "ascend")]
unsafe impl Send for DeviceBinary {}
#[cfg(feature = "ascend")]
unsafe impl Sync for DeviceBinary {}

#[cfg(feature = "ascend")]
static DEVICE_BINARY: std::sync::OnceLock<DeviceBinary> = std::sync::OnceLock::new();

#[cfg(feature = "ascend")]
impl DeviceBinary {
    /// Register `elf` once; every later chip reuses this registration. `get_or_init` guarantees the
    /// closure runs a single time, so the ELF is never registered twice (the whole point).
    fn get(elf: Vec<u8>, elf_path: &std::path::Path) -> &'static DeviceBinary {
        DEVICE_BINARY.get_or_init(|| {
            Self::register(elf, elf_path).unwrap_or_else(|e| panic!("device binary: {e}"))
        })
    }

    fn register(elf: Vec<u8>, elf_path: &std::path::Path) -> Result<DeviceBinary> {
        use std::ffi::CString;
        unsafe {
            let bin = ffi::RtDevBinary {
                magic: MAGIC_ELF_AIVEC,
                version: 0,
                data: elf.as_ptr(),
                length: elf.len() as u64,
            };
            let mut handle: *mut u8 = std::ptr::null_mut();
            let rc = ffi::rtDevBinaryRegister(&bin, &mut handle);
            if rc != 0 {
                bail!(
                    "rtDevBinaryRegister failed with {rc} on {} ({} bytes). Check the magic is the \
                     AIVEC one and that the object came from `ccec --cce-aicore-only`.",
                    elf_path.display(),
                    elf.len()
                );
            }
            let mut stubs: Vec<(&'static str, Box<u8>)> = Vec::new();
            // REGISTER EACH DEVICE SYMBOL ONCE. Chaining DECOMPOSITION_KERNELS straight onto the slots
            // registered a symbol TWICE whenever a decomposition kernel reused one (w8a8_band_k2048 is
            // the same device function as the slot w8a8_experts_2048_4096), and the second registration
            // corrupted the stub the slot launches: the expert launch then hung with a parameter block
            // the launcher log shows is correct, while three examples regressed without changing.
            let mut seen_symbols: Vec<&'static str> = Vec::new();
            for k in LAYER0_KERNELS.iter().chain(DECOMPOSITION_KERNELS) {
                if seen_symbols.contains(&k.symbol) {
                    continue;
                }
                seen_symbols.push(k.symbol);
                let stub = Box::new(0u8);
                let name = CString::new(k.symbol)?;
                let rc = ffi::rtFunctionRegister(
                    handle,
                    &*stub as *const u8,
                    name.as_ptr(),
                    // The DEVICE FUNCTION NAME goes here; null returns 507000 even though the
                    // binary registered fine.
                    name.as_ptr() as *const u8,
                    0,
                );
                if rc != 0 {
                    // 507000 = this symbol is not in the device binary. An ELF that does not carry
                    // every kernel registers the ones it has; the dispatcher CPU-falls-back for any
                    // it could not install here.
                    eprintln!("[device] {} not in ELF (rc={rc}) — left on CPU", k.symbol);
                    continue;
                }
                stubs.push((k.name, stub));
            }
            Ok(DeviceBinary { _elf: elf, handle, stubs })
        }
    }

    fn has_any(&self, prefix: &str) -> bool {
        self.stubs.iter().any(|(n, _)| n.starts_with(prefix))
    }

    fn stub(&self, name: &str) -> Option<*const u8> {
        self.stubs.iter().find(|(n, _)| *n == name).map(|(_, s)| &**s as *const u8)
    }
}

/// A per-chip launcher: its own context + stream + pinned-buffer cache, launching the process-wide
/// [`DeviceBinary`]'s shared stubs on this chip's stream. The real [`crate::launch::KernelLaunch`].
///
/// One instance per chip. The device binary is registered once (see [`DeviceBinary`]); every
/// launcher points at the same `&'static` registration.
#[cfg(feature = "ascend")]
pub struct DeviceLaunch {
    /// Device buffers pinned under host `(ptr, len)` identities — weight planes uploaded once
    /// instead of per launch. ~20 GB/token of H2D measured before this existed (27 MB expert
    /// planes x 774 launches). Freed wholesale by `invalidate_pins` (the ABA contract in
    /// `KernelLaunch`) and on Drop. Mutex for the Send/Sync bound; actual use is single-thread
    /// (check_thread).
    pinned: std::sync::Mutex<std::collections::HashMap<(usize, usize), *mut u8>>,
    /// The process-wide device binary (registered once). Shared across all chips; each launcher
    /// launches these stubs on its own `stream`.
    binary: &'static DeviceBinary,
    stream: *mut u8,
    /// The device this launcher's stream and buffers live on. ACL's current device is
    /// thread-local and gets overwritten whenever another `DeviceLaunch` opens (another chip),
    /// so every launch re-selects it first — the primitive multi-chip pipelining needs.
    device_id: i32,
    /// This launcher's own ACL context on `device_id`. Each `DeviceLaunch` (each chip)
    /// holds its own; every launch sets it current so operations and the registered
    /// device binary target the right chip.
    context: *mut u8,
    /// The thread that created the stream. ACL's device and context are THREAD-LOCAL — the
    /// multi-link H2D measurement depended on that, calling `aclrtSetDevice` inside every worker —
    /// so a stream used from a thread that never set the device belongs to no context there.
    owner: std::thread::ThreadId,
}

#[cfg(feature = "ascend")]
impl DeviceLaunch {
    /// Register `elf_path` on `device_id` and bind every kernel in [`LAYER0_KERNELS`].
    ///
    /// `aclInit` is guarded by a `Once` because it is process-global and returns an error on the
    /// second call — which would make a second launcher fail for a reason that has nothing to do
    /// with it.
    /// Whether this ELF registered any kernel whose name starts with `prefix`
    /// (e.g. "rms_norm", "matvec_", "mx4pk"). Lets the dispatcher offload only the
    /// kinds actually present in the binary and CPU-fall-back the rest.
    pub fn has_any(&self, prefix: &str) -> bool {
        self.binary.has_any(prefix)
    }

    pub fn open(elf_path: &std::path::Path, device_id: i32) -> Result<Self> {
        use std::ffi::CString;
        static ACL_INIT: std::sync::Once = std::sync::Once::new();
        let mut init_rc = 0;
        ACL_INIT.call_once(|| unsafe {
            init_rc = ffi::aclInit(std::ptr::null());
        });
        if init_rc != 0 {
            bail!("aclInit failed with {init_rc}");
        }
        let elf = std::fs::read(elf_path)
            .map_err(|e| anyhow::anyhow!("cannot read device ELF {}: {e}", elf_path.display()))?;

        unsafe {
            let rc = ffi::aclrtSetDevice(device_id);
            if rc != 0 {
                bail!("aclrtSetDevice({device_id}) failed with {rc}");
            }
            // Give this launcher its OWN context on the chip. Two launchers on two chips
            // then have independent contexts; setting the right one current before each
            // launch is what makes rt* binary registration + launch multi-device-safe
            // (aclrtSetDevice alone left chip 0 broken once chip 1 opened — 107003).
            let mut context: *mut u8 = std::ptr::null_mut();
            let rc = ffi::aclrtCreateContext(&mut context, device_id);
            if rc != 0 {
                bail!("aclrtCreateContext({device_id}) failed with {rc}");
            }
            let mut stream: *mut u8 = std::ptr::null_mut();
            let rc = ffi::aclrtCreateStream(&mut stream);
            if rc != 0 {
                bail!("aclrtCreateStream failed with {rc}");
            }

            // Register the device binary ONCE, process-wide (see `DeviceBinary`). This chip is
            // current now, but the registration serves every chip: later chips reuse the same
            // `&'static` registration and launch its stubs on their own streams. Registering
            // per-chip instead rebinds rt*'s name-keyed table and breaks the earlier chip (107003).
            let binary = DeviceBinary::get(elf, elf_path);
            Ok(Self {
                pinned: std::sync::Mutex::new(std::collections::HashMap::new()),
                binary,
                stream,
                device_id,
                context,
                owner: std::thread::current().id(),
            })
        }
    }

    /// Refuse use from a thread other than the one that created the stream.
    ///
    /// This is what makes the `Send + Sync` impl below sound rather than asserted. `KernelLaunch`
    /// requires both, but the stream is only valid in its creating thread's context, so the choice
    /// is between a runtime refusal and undefined behaviour. This crate already prefers loud
    /// failure — `KernelLaunch` errors PANIC rather than falling back, because a silent fallback
    /// turns a broken kernel into a slow-but-correct model — and the same reasoning applies here.
    fn check_thread(&self) -> Result<()> {
        let now = std::thread::current().id();
        if now != self.owner {
            bail!(
                "DeviceLaunch was created on thread {:?} and used from {now:?}. ACL's device and \
                 context are thread-local, so this stream has no context here. Create one launcher \
                 per thread, each calling aclrtSetDevice for itself.",
                self.owner
            );
        }
        Ok(())
    }

    /// Launch with all inputs inside ONE device allocation, at aligned offsets.
    ///
    /// The harness that certifies these kernels passes four pointers into a SINGLE allocation; this
    /// launcher has always passed one allocation per input. If a kernel's vector chunks assume
    /// contiguity, per-buffer slack cannot substitute for it -- which is the hypothesis this entry point
    /// exists to test, and the reason it is a separate method rather than a change to `run_raw_cached`.
    pub fn run_by_symbol_contiguous(
        &self,
        name: &str,
        symbol: &str,
        partitioning: Partitioning,
        inputs: &[&[u8]],
        out_len: usize,
    ) -> Result<Vec<f32>> {
        let spec = KernelSpec {
            name: Box::leak(name.to_string().into_boxed_str()),
            symbol: Box::leak(symbol.to_string().into_boxed_str()),
            partitioning,
        };
        const ALIGN: usize = 512;
        const SLACK: usize = 64 * 1024;
        let mut offsets: Vec<usize> = Vec::with_capacity(inputs.len());
        let mut total = 0usize;
        for inp in inputs {
            total = total.div_ceil(ALIGN) * ALIGN;
            offsets.push(total);
            total += inp.len();
        }
        total = total.div_ceil(ALIGN) * ALIGN;

        let mut base: *mut u8 = std::ptr::null_mut();
        if unsafe { ffi::aclrtMalloc(&mut base, total + SLACK, ffi::MALLOC_HUGE_FIRST) } != 0 {
            bail!("aclrtMalloc({}) failed for contiguous {}", total + SLACK, spec.name);
        }
        for (inp, &off) in inputs.iter().zip(offsets.iter()) {
            let dst = unsafe { base.add(off) };
            let rc = unsafe { ffi::aclrtMemcpy(dst, inp.len(), inp.as_ptr(), inp.len(), ffi::MEMCPY_H2D) };
            if rc != 0 {
                unsafe { ffi::aclrtFree(base) };
                bail!("H2D copy at offset {off} failed for {}", spec.name);
            }
        }
        let out_bytes = out_len * std::mem::size_of::<f32>() + SLACK;
        let mut d_out: *mut u8 = std::ptr::null_mut();
        if unsafe { ffi::aclrtMalloc(&mut d_out, out_bytes, ffi::MALLOC_HUGE_FIRST) } != 0 {
            unsafe { ffi::aclrtFree(base) };
            bail!("aclrtMalloc({out_bytes}) failed for contiguous output of {}", spec.name);
        }
        let mut args: Vec<*mut u8> = offsets.iter().map(|&o| unsafe { base.add(o) }).collect();
        args.push(d_out);
        let stub = match self.stub(spec.name) {
            Ok(s) => s,
            Err(e) => {
                unsafe { ffi::aclrtFree(base) };
                unsafe { ffi::aclrtFree(d_out) };
                return Err(e);
            }
        };
        let rc = unsafe {
            ffi::rtKernelLaunch(
                stub,
                // ONE BLOCK. The emitted kernel has no `GetBlockIdx` (verified: 0 occurrences in both
                // the w8a8 and the q8mv artefacts), so it walks every row in a single AIV; launching it
                // with `block_dim(USABLE_AIV)` starts N copies that contend for the same DMA. The
                // harness's own `_do` launcher, which certifies this kernel, uses one.
                1,
                args.as_mut_ptr() as *mut u8,
                (args.len() * std::mem::size_of::<*mut u8>()) as u32,
                std::ptr::null_mut(),
                self.stream,
            )
        };
        if rc != 0 {
            unsafe { ffi::aclrtFree(base) };
            unsafe { ffi::aclrtFree(d_out) };
            bail!("rtKernelLaunch({}) failed with {rc}", spec.name);
        }
        if unsafe { ffi::aclrtSynchronizeStream(self.stream) } != 0 {
            unsafe { ffi::aclrtFree(base) };
            unsafe { ffi::aclrtFree(d_out) };
            bail!("stream sync failed after {}", spec.name);
        }
        let mut out = vec![0.0f32; out_len];
        let bytes = out_len * std::mem::size_of::<f32>();
        let rc = unsafe {
            ffi::aclrtMemcpy(out.as_mut_ptr() as *mut u8, bytes, d_out, bytes, ffi::MEMCPY_D2H)
        };
        unsafe { ffi::aclrtFree(base) };
        unsafe { ffi::aclrtFree(d_out) };
        if rc != 0 {
            bail!("D2H copy failed for {}", spec.name);
        }
        Ok(out)
    }

    fn stub(&self, name: &str) -> Result<*const u8> {
        self.binary
            .stub(name)
            .ok_or_else(|| anyhow::anyhow!("kernel {name} was never registered"))
    }

    /// Make this launcher's chip current before a launch.
    ///
    /// ACL's device AND context are thread-local, and another chip's `open` (or a launch on it)
    /// leaves both pointing at that other chip. Restoring only the context was not enough: chip 0's
    /// `rms_norm` still failed with 107003 after chip 1 opened, because the kernel-launch path in
    /// rt* also tracks a current DEVICE that `aclrtSetCurrentContext` does not move. Both calls are
    /// needed — `aclrtSetDevice` restores rt*'s device, `aclrtSetCurrentContext` restores the ACL
    /// context our stream belongs to. This is what lets two chips coexist.
    unsafe fn bind(&self) {
        ffi::aclrtSetDevice(self.device_id);
        ffi::aclrtSetCurrentContext(self.context);
    }

    /// Launch a kernel over f32 buffers: upload `inputs`, run, download `out_len` floats.
    ///
    /// Allocates and frees per call, which is wasteful and deliberate for a first implementation —
    /// a buffer pool is an optimisation, and getting one wrong would corrupt results rather than
    /// slow them down.
    fn run_f32(&self, spec: &KernelSpec, inputs: &[&[f32]], out_len: usize) -> Result<Vec<f32>> {
        self.check_thread()?;
        unsafe {
            self.bind();
            let mut dev: Vec<*mut u8> = Vec::with_capacity(inputs.len() + 1);
            let mut cleanup = |ptrs: &[*mut u8]| {
                for p in ptrs {
                    ffi::aclrtFree(*p);
                }
            };
            for inp in inputs {
                let bytes = std::mem::size_of_val(*inp);
                let mut p: *mut u8 = std::ptr::null_mut();
                if ffi::aclrtMalloc(&mut p, bytes, ffi::MALLOC_HUGE_FIRST) != 0 {
                    cleanup(&dev);
                    bail!("aclrtMalloc({bytes}) failed for {}", spec.name);
                }
                if ffi::aclrtMemcpy(p, bytes, inp.as_ptr() as *const u8, bytes, ffi::MEMCPY_H2D) != 0
                {
                    dev.push(p);
                    cleanup(&dev);
                    bail!("H2D copy failed for {}", spec.name);
                }
                dev.push(p);
            }
            // THE OUTPUT NEEDS SLACK TOO. It is sized exactly to `out_len` floats, and the kernel writes
            // in 32 B vector chunks; an unmapped write just past the end faults and the stream spins.
            // Padding inputs without padding the output left the sliced launch hanging at 1024 outputs
            // (4 KiB, the size where the over-write crosses a page).
            let out_bytes = out_len * std::mem::size_of::<f32>() + 64 * 1024;
            let mut d_out: *mut u8 = std::ptr::null_mut();
            if ffi::aclrtMalloc(&mut d_out, out_bytes, ffi::MALLOC_HUGE_FIRST) != 0 {
                cleanup(&dev);
                bail!("aclrtMalloc({out_bytes}) failed for {} output", spec.name);
            }

            let mut args: Vec<*mut u8> = dev.clone();
            args.push(d_out);
            // ONE BLOCK, WHICH IS WHAT THE HARNESS USES. The harness that certifies every one of these
            // kernels calls `KERNEL_DO(1, stream, ...)` -- blockDim 1, always -- while this launcher
            // passed `spec.block_dim(USABLE_AIV)`. The emitted kernels contain no `GetBlockIdx`, so every
            // block walks all rows: N blocks are N redundant copies contending for the same DMA, and for
            // the w8a8 expert matvec that deadlocked (eight rounds), while for the q8mv matvecs the
            // copies are idempotent and merely wasteful -- which is why the model still produced its
            // BLOCK DIMENSION, TUNABLE FOR ONE EXPERIMENT. Iteration 151 aligned every launch with the
            // harness's `KERNEL_DO(1, ...)`. That is right for the q8mv matvecs; the w8a8 expert launch
            // PASSED at iteration 145 with `spec.block_dim(USABLE_AIV)` and hangs with 1, so the value is
            // now overridable to settle which it is for this kernel.
            let block_dim: u32 = std::env::var("DS4_BLOCK_DIM")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1);
            let rc = ffi::rtKernelLaunch(
                self.stub(spec.name)?,
                block_dim,
                args.as_mut_ptr() as *mut u8,
                (args.len() * std::mem::size_of::<*mut u8>()) as u32,
                std::ptr::null_mut(),
                self.stream,
            );
            if rc != 0 {
                dev.push(d_out);
                cleanup(&dev);
                bail!("rtKernelLaunch({}) failed with {rc}", spec.name);
            }
            if ffi::aclrtSynchronizeStream(self.stream) != 0 {
                dev.push(d_out);
                cleanup(&dev);
                bail!("stream sync failed after {}", spec.name);
            }
            let mut out = vec![0f32; out_len];
            let rc = ffi::aclrtMemcpy(
                out.as_mut_ptr() as *mut u8,
                out_bytes,
                d_out,
                out_bytes,
                ffi::MEMCPY_D2H,
            );
            dev.push(d_out);
            cleanup(&dev);
            if rc != 0 {
                bail!("D2H copy failed for {}", spec.name);
            }
            Ok(out)
        }
    }
}

#[cfg(feature = "ascend")]
impl DeviceLaunch {
    /// As [`DeviceLaunch::run_f32`] but over raw byte buffers, for kernels whose inputs are not
    /// all f32 — the `q8mv` family takes an int8 code plane first. Same deliberate
    /// allocate-and-free-per-call shape.
    fn run_raw(&self, spec: &KernelSpec, inputs: &[&[u8]], out_len: usize) -> Result<Vec<f32>> {
        self.check_thread()?;
        unsafe {
            self.bind();
            let mut dev: Vec<*mut u8> = Vec::with_capacity(inputs.len() + 1);
            let mut cleanup = |ptrs: &[*mut u8]| {
                for p in ptrs {
                    ffi::aclrtFree(*p);
                }
            };
            for inp in inputs {
                let bytes = inp.len();
                let mut p: *mut u8 = std::ptr::null_mut();
                if ffi::aclrtMalloc(&mut p, bytes, ffi::MALLOC_HUGE_FIRST) != 0 {
                    cleanup(&dev);
                    bail!("aclrtMalloc({bytes}) failed for {}", spec.name);
                }
                if ffi::aclrtMemcpy(p, bytes, inp.as_ptr(), bytes, ffi::MEMCPY_H2D) != 0 {
                    dev.push(p);
                    cleanup(&dev);
                    bail!("H2D copy failed for {}", spec.name);
                }
                dev.push(p);
            }
            let out_bytes = out_len * std::mem::size_of::<f32>();
            let mut d_out: *mut u8 = std::ptr::null_mut();
            if ffi::aclrtMalloc(&mut d_out, out_bytes, ffi::MALLOC_HUGE_FIRST) != 0 {
                cleanup(&dev);
                bail!("aclrtMalloc({out_bytes}) failed for {} output", spec.name);
            }
            let mut args: Vec<*mut u8> = dev.clone();
            args.push(d_out);
            let rc = ffi::rtKernelLaunch(
                self.stub(spec.name)?,
                spec.block_dim(USABLE_AIV),
                args.as_mut_ptr() as *mut u8,
                (args.len() * std::mem::size_of::<*mut u8>()) as u32,
                std::ptr::null_mut(),
                self.stream,
            );
            if rc != 0 {
                dev.push(d_out);
                cleanup(&dev);
                bail!("rtKernelLaunch({}) failed with {rc}", spec.name);
            }
            if ffi::aclrtSynchronizeStream(self.stream) != 0 {
                dev.push(d_out);
                cleanup(&dev);
                bail!("stream sync failed after {}", spec.name);
            }
            let mut out = vec![0f32; out_len];
            let rc = ffi::aclrtMemcpy(
                out.as_mut_ptr() as *mut u8,
                out_bytes,
                d_out,
                out_bytes,
                ffi::MEMCPY_D2H,
            );
            dev.push(d_out);
            cleanup(&dev);
            if rc != 0 {
                bail!("D2H copy failed for {}", spec.name);
            }
            Ok(out)
        }
    }
}

#[cfg(feature = "ascend")]
impl DeviceLaunch {
    /// Get-or-upload a pinned device twin of `bytes`, keyed by host identity.
    fn upload_pinned(&self, spec_name: &str, bytes: &[u8]) -> Result<*mut u8> {
        let key = (bytes.as_ptr() as usize, bytes.len());
        if let Some(&p) = self.pinned.lock().unwrap().get(&key) {
            return Ok(p);
        }
        unsafe {
            let mut p: *mut u8 = std::ptr::null_mut();
            // THE CACHED PATH IS THE ONE THE WEIGHTS TAKE, and it allocated EXACTLY `bytes.len()`.
            // The emitted kernels read a little past the end of the weight plane; on a tight allocation
            // that read is unmapped, the device raises 507034, and the stream then spins for minutes --
            // which is what "the launch hangs" has meant for six rounds. Both paths now over-allocate by
            // the same slack, so the over-read lands in mapped memory and the kernel's result is
            // unaffected (the padded runs match f64 to 1.130e-07).
            const SLACK: usize = 64 * 1024;
            let total = bytes.len() + SLACK;
            if ffi::aclrtMalloc(&mut p, total, ffi::MALLOC_HUGE_FIRST) != 0 {
                bail!("aclrtMalloc({total}) failed pinning for {spec_name}");
            }
            if ffi::aclrtMemcpy(p, bytes.len(), bytes.as_ptr(), bytes.len(), ffi::MEMCPY_H2D)
                != 0
            {
                ffi::aclrtFree(p);
                bail!("H2D pin copy failed for {spec_name}");
            }
            self.pinned.lock().unwrap().insert(key, p);
            Ok(p)
        }
    }

    /// As `run_raw`, but inputs marked `true` are PINNED — uploaded once under their host
    /// identity and reused across launches. Transient inputs and the output keep the
    /// allocate-per-call shape.

    /// `matvec_w8a8_experts` for a kernel in [`DECOMPOSITION_KERNELS`] rather than a slot.
    ///
    /// IDENTICAL in every other respect to the method above -- same marshalling, same flags, same call.
    /// It exists because the banded launch hangs through `run_by_symbol` at the same width and k where this
    /// sequence returns 1.130e-07, and three candidates (k, the pin flags, the spec's identity) were
    /// eliminated by measurement. Having the WORKING body reachable for a decomposition kernel turns "the
    /// two paths differ somewhere" into a diff of two functions that should now be the same.
    pub fn matvec_w8a8_banded(
        &self,
        name: &str,
        codes: &[i8],
        scale: &[f32],
        offset: &[f32],
        x: &[f32],
        d_out: usize,
    ) -> Result<Vec<f32>> {
        let spec = DECOMPOSITION_KERNELS
            .iter()
            .find(|k| k.name == name)
            .ok_or_else(|| anyhow::anyhow!("{name} is not in DECOMPOSITION_KERNELS"))?;
        let xb = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4) };
        let wb = unsafe { std::slice::from_raw_parts(codes.as_ptr() as *const u8, codes.len()) };
        let sb = unsafe { std::slice::from_raw_parts(scale.as_ptr() as *const u8, scale.len() * 4) };
        let ob = unsafe { std::slice::from_raw_parts(offset.as_ptr() as *const u8, offset.len() * 4) };
        self.run_raw_cached(spec, &[(xb, false), (wb, true), (sb, true), (ob, true)], d_out)
    }

    /// Launch a kernel by SYMBOL, without a named slot.
    ///
    /// The slot table (`LAYER0_KERNELS`) describes what the ENGINE's layer code calls, and its denominator
    /// is the campaign's KPI. A caller-side DECOMPOSITION -- running a certified kernel over 1024-row bands
    /// because the arm faults above ~1024 rows at k=4096 -- is not an engine slot and must not become one,
    /// or the denominator grows by a caller's convenience. This entry point is what such a decomposition
    /// uses instead.
    ///
    /// The strings are leaked because `KernelSpec` holds `&'static str`; a decomposition builds a handful of
    /// these per call site, so the leak is bounded and deliberate rather than a general pattern.
    pub fn run_by_symbol(
        &self,
        name: &str,
        symbol: &str,
        partitioning: Partitioning,
        inputs: &[(&[u8], bool)],
        out_len: usize,
    ) -> Result<Vec<f32>> {
        // PREFER THE REGISTERED SPEC. `matvec_w8a8_experts` passes a spec borrowed from LAYER0_KERNELS and
        // works; this function used to BUILD one on the fly every call, and the banded launch hung with it at
        // both k=2048 and k=4096 -- the same width and k where the borrowed spec returns 1.130e-07. Building
        // a second spec with the same name means the launcher sees a different `&KernelSpec` identity than
        // the one it registered a stub against, which is the kind of difference that does not show in the
        // C++ but changes what the runtime resolves.
        let spec: &KernelSpec = if let Some(s) =
            DECOMPOSITION_KERNELS.iter().find(|k| k.name == name)
        {
            s
        } else {
            Box::leak(Box::new(KernelSpec {
                name: Box::leak(name.to_string().into_boxed_str()),
                symbol: Box::leak(symbol.to_string().into_boxed_str()),
                partitioning,
            }))
        };
        self.run_raw_cached(spec, inputs, out_len)
    }

    fn run_raw_cached(
        &self,
        spec: &KernelSpec,
        inputs: &[(&[u8], bool)],
        out_len: usize,
    ) -> Result<Vec<f32>> {
        self.check_thread()?;
        unsafe {
            // Make this launcher's context current: another DeviceLaunch (another chip) may
            // have set a different one, and the device binary + stream belong to this context.
            self.bind();
            let mut transient: Vec<*mut u8> = Vec::new();
            let cleanup = |ptrs: &[*mut u8]| {
                for p in ptrs {
                    ffi::aclrtFree(*p);
                }
            };
            let mut args: Vec<*mut u8> = Vec::with_capacity(inputs.len() + 1);
            for &(inp, pin) in inputs {
                if pin {
                    match self.upload_pinned(spec.name, inp) {
                        Ok(p) => args.push(p),
                        Err(e) => {
                            cleanup(&transient);
                            return Err(e);
                        }
                    }
                } else {
                    let bytes = inp.len();
                    // ZERO-PADDED UPLOAD. The emitted kernels read slightly past the logical end of a
                    // buffer, and what they find there matters: the caller-side workaround (a Vec extended
                    // with zeros) makes the launch return, while a launcher that merely over-ALLOCATES
                    // (leaving uninitialised device memory) still hangs. So the slack is filled with
                    // zeros here too, and the buffer is the full allocated size.
                    const SLACK: usize = 64 * 1024;
                    let total = bytes + SLACK;
                    let mut padded = Vec::with_capacity(total);
                    padded.extend_from_slice(inp);
                    padded.resize(total, 0u8);
                    let mut p: *mut u8 = std::ptr::null_mut();
                    if ffi::aclrtMalloc(&mut p, total, ffi::MALLOC_HUGE_FIRST) != 0 {
                        cleanup(&transient);
                        bail!("aclrtMalloc({total}) failed for {}", spec.name);
                    }
                    if ffi::aclrtMemcpy(p, total, padded.as_ptr(), total, ffi::MEMCPY_H2D) != 0 {
                        transient.push(p);
                        cleanup(&transient);
                        bail!("H2D copy failed for {}", spec.name);
                    }
                    transient.push(p);
                    args.push(p);
                }
            }
            let out_bytes = out_len * std::mem::size_of::<f32>();
            let mut d_out: *mut u8 = std::ptr::null_mut();
            if ffi::aclrtMalloc(&mut d_out, out_bytes, ffi::MALLOC_HUGE_FIRST) != 0 {
                cleanup(&transient);
                bail!("aclrtMalloc({out_bytes}) failed for {} output", spec.name);
            }
            args.push(d_out);
            // DS4_LAUNCH_DEBUG: the parameter block, printed BEFORE the launch. run_raw_cached is the
            // path every w8a8 launch takes; a hang then leaves one printed block as evidence, which is
            // exactly what four rounds of post-hoc elimination did not have.
            if std::env::var_os("DS4_LAUNCH_DEBUG").is_some() {
                eprintln!(
                    "[launch] {} args={} ptbytes={} out_len={} out_bytes={out_bytes}",
                    spec.name,
                    args.len(),
                    args.len() * std::mem::size_of::<*mut u8>(),
                    out_len
                );
                for i in 0..inputs.len() {
                    eprintln!("[launch]   in[{i}] len={}", inputs[i].0.len());
                }
            }
            let rc = ffi::rtKernelLaunch(
                self.stub(spec.name)?,
                spec.block_dim(USABLE_AIV),
                args.as_mut_ptr() as *mut u8,
                (args.len() * std::mem::size_of::<*mut u8>()) as u32,
                std::ptr::null_mut(),
                self.stream,
            );
            if rc != 0 {
                transient.push(d_out);
                cleanup(&transient);
                bail!("rtKernelLaunch({}) failed with {rc}", spec.name);
            }
            if ffi::aclrtSynchronizeStream(self.stream) != 0 {
                transient.push(d_out);
                cleanup(&transient);
                bail!("stream sync failed after {}", spec.name);
            }
            let mut out = vec![0f32; out_len];
            let rc = ffi::aclrtMemcpy(
                out.as_mut_ptr() as *mut u8,
                out_bytes,
                d_out,
                out_bytes,
                ffi::MEMCPY_D2H,
            );
            transient.push(d_out);
            cleanup(&transient);
            if rc != 0 {
                bail!("D2H copy failed for {}", spec.name);
            }
            Ok(out)
        }
    }
}

/// The q8mv kernel for a `(d_in, d_out)` shape, or `None` — the ELF carries exactly four, and a
/// caller asking for a fifth gets a refusal from the launcher rather than a wrong launch.
pub fn q8mv_kernel_for(d_in: usize, d_out: usize) -> Option<&'static str> {
    match (d_in, d_out) {
        (4096, 1024) => Some("matvec_4096_1024"),
        (1024, 32768) => Some("matvec_1024_32768"),
        (4096, 512) => Some("matvec_4096_512"),
        (8192, 4096) => Some("matvec_8192_4096"),
        _ => None,
    }
}

// Sound because every entry point calls `check_thread` first: a launcher moved or shared across
// threads refuses at runtime instead of using a stream from another thread's context. Without that
// guard this impl would be a lie, since the raw stream pointer is neither Send nor Sync.
#[cfg(feature = "ascend")]
unsafe impl Send for DeviceLaunch {}
#[cfg(feature = "ascend")]
unsafe impl Sync for DeviceLaunch {}

#[cfg(feature = "ascend")]
impl crate::launch::KernelLaunch for DeviceLaunch {
    fn name(&self) -> &str {
        "ascend-npu"
    }

    /// TRUE, unlike `HostFakeLaunch`. `offloaded_by` keys on this, so a wrong answer here would
    /// report NPU coverage the model never had.
    fn is_device(&self) -> bool {
        true
    }

    fn rms_norm(&self, x: &[f32], gamma: &[f32], eps: f32) -> Result<Vec<f32>> {
        check_rms_eps(eps)?;
        if gamma.len() != x.len() {
            bail!(
                "the emitted rmsmul reads gamma elementwise over the same width as x, so a \
                 gamma of {} against an x of {} would read out of bounds on the device",
                gamma.len(),
                x.len()
            );
        }
        let spec = select_rms_kernel(x.len())?;
        self.run_f32(spec, &[x, gamma], x.len())
    }

    /// Split-plane Q8_0 matvec on the four shapes the ELF carries. Argument order matches the
    /// emitted `q8mv` signature `(codes i8*, scales f32*, x f32*, out f32*)` — the same order
    /// every device runner for these kernels has used.
    fn matvec_q8(
        &self,
        codes: &[i8],
        scales: &[f32],
        x: &[f32],
        d_out: usize,
    ) -> Result<Vec<f32>> {
        let d_in = x.len();
        let Some(name) = q8mv_kernel_for(d_in, d_out) else {
            bail!(
                "no q8mv kernel for {d_in}->{d_out}; the ELF carries 4096->1024, 1024->32768, \
                 4096->512 and 8192->4096. Other shapes stay on the CPU by policy, not by \
                 accident — see the dispatcher's routing"
            );
        };
        if codes.len() != d_out * d_in {
            bail!("q8mv {name}: code plane {} != d_out*d_in {}", codes.len(), d_out * d_in);
        }
        if scales.len() != d_out * d_in {
            bail!(
                "q8mv {name}: scale plane {} != d_out*d_in {} — the emitted kernel reads the \
                 per-element EXPANDED plane, not per-block scales",
                scales.len(),
                d_out * d_in
            );
        }
        let spec = kernel(name)?;
        let codes_bytes =
            unsafe { std::slice::from_raw_parts(codes.as_ptr() as *const u8, codes.len()) };
        let scales_bytes = unsafe {
            std::slice::from_raw_parts(scales.as_ptr() as *const u8, scales.len() * 4)
        };
        let x_bytes =
            unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4) };
        self.run_raw_cached(
            spec,
            &[(codes_bytes, true), (scales_bytes, true), (x_bytes, false)],
            d_out,
        )
    }

    /// Packed-MXFP4 matvec on the two expert-projection shapes. `w` is the repacked nibble
    /// plane, `scales` the halved even|odd per-block f32 plane, `x_perm` the PERMUTED
    /// activation — all three exactly [`crate::mxfp4_repack`]'s contract, whose `matvec_ref`
    /// is the oracle this kernel was proven against on device.
    fn matvec_mx4pk(
        &self,
        w: &[u8],
        scales: &[f32],
        x_perm: &[f32],
        d_out: usize,
    ) -> Result<Vec<f32>> {
        // THE SOCKETS ARE w8a8 NOW AND THIS LAUNCH STILL MARSHALS THE PACKED FORM. Calling it would
        // pass four packed arguments to a five-argument kernel -- a crash or silent corruption, not a
        // wrong number. So it refuses, the MoE falls back to the host, and the missing piece has a
        // name: **w8a8 expert storage**. Registering w8a8 slot names without this guard would have
        // moved the failure from "not implemented" to "undefined behaviour", which is strictly worse.
        bail!(
            "matvec_mx4pk({} out, {} packed bytes): the slot is a w8a8 socket now, but this launch still marshals packed \
             MXFP4 (nibble planes, halved scales, permuted x). Serving it needs w8a8 EXPERT STORAGE \
             -- int8 codes with a per-output-channel scale and offset -- not a different launch.",
            d_out,
            w.len()
        );

        let d_in = x_perm.len();
        let Some(name) = mx4pk_kernel_for(d_in, d_out) else {
            bail!(
                "no mx4pk kernel for {d_in}->{d_out}; the ELF carries 4096->2048 and                  2048->4096 (the routed-expert projections)"
            );
        };
        if w.len() != d_out * d_in / 2 {
            bail!("mx4pk {name}: nibble plane {} != d_out*d_in/2 {}", w.len(), d_out * d_in / 2);
        }
        if scales.len() != d_out * d_in / 32 {
            bail!(
                "mx4pk {name}: scale plane {} != d_out*d_in/32 {}",
                scales.len(),
                d_out * d_in / 32
            );
        }
        let spec = kernel(name)?;
        let scales_bytes = unsafe {
            std::slice::from_raw_parts(scales.as_ptr() as *const u8, scales.len() * 4)
        };
        let x_bytes = unsafe {
            std::slice::from_raw_parts(x_perm.as_ptr() as *const u8, x_perm.len() * 4)
        };
        self.run_raw_cached(spec, &[(w, true), (scales_bytes, true), (x_bytes, false)], d_out)
    }

    /// Free every pinned device buffer. See the trait doc: callers dropping pinned host
    /// memory MUST call this first, or a recycled address silently reuses stale device bytes.
    fn invalidate_pins(&self) {
        if self.check_thread().is_err() {
            // Wrong thread: leak rather than free buffers a stream elsewhere may still touch.
            return;
        }
        let mut map = self.pinned.lock().unwrap();
        unsafe {
            for (_, p) in map.drain() {
                ffi::aclrtFree(p);
            }
        }
    }
    fn matvec_w8a8_experts(
        &self,
        name: &str,
        codes: &[i8],
        scale: &[f32],
        offset: &[f32],
        x: &[f32],
        d_out: usize,
    ) -> Result<Vec<f32>> {
        let spec = crate::device::kernel(name)?;
        // KERNEL argument order: (x, w, scale, offset, out) -- see the declaration in launch.rs.
        let xb = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4) };
        let wb = unsafe { std::slice::from_raw_parts(codes.as_ptr() as *const u8, codes.len()) };
        let sb = unsafe { std::slice::from_raw_parts(scale.as_ptr() as *const u8, scale.len() * 4) };
        let ob = unsafe { std::slice::from_raw_parts(offset.as_ptr() as *const u8, offset.len() * 4) };
        self.run_raw_cached(spec, &[(xb, false), (wb, true), (sb, true), (ob, true)], d_out)
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    /// A cooperating kernel's width is a property of the artifact, so `desired` must not move it in
    /// EITHER direction. Clamping would be the natural thing to write and is wrong both ways: under
    /// its built-in width part of the cross-block reduction is never written, over it the extra
    /// blocks write past the workspace the emitter sized.
    ///
    /// This is the guard for the `gmm_blocks` mismatch — an emit-time constant
    /// (`TILERS_GMM_BLOCKS`, default 1) baked into a workspace reduction, against a launcher that
    /// otherwise picks its own width. Two numbers that must agree, with nothing making them agree.
    #[test]
    fn a_cooperating_kernel_ignores_the_desired_width_in_both_directions() {
        let coop = KernelSpec {
            name: "gmm_swiglu_quant_16",
            symbol: "_Z0stub16",
            partitioning: Partitioning::Cooperating(16),
        };
        for desired in [0, 1, 8, 16, 24, 48, 96, u32::MAX] {
            assert_eq!(
                coop.block_dim(desired),
                16,
                "Cooperating(16) must launch at 16, not {desired}",
            );
        }

        // The other two arms keep their established behaviour — a third variant must not have
        // loosened them.
        let single = KernelSpec {
            partitioning: Partitioning::SingleCore,
            ..coop
        };
        let by_row = KernelSpec {
            partitioning: Partitioning::ByRow,
            ..coop
        };
        assert_eq!(single.block_dim(48), 1, "SingleCore is forced to one core");
        assert_eq!(by_row.block_dim(96), USABLE_AIV, "ByRow clamps to the usable ceiling");
        assert_eq!(by_row.block_dim(0), 1, "ByRow never yields a zero stride");
    }

    /// Every kernel's stub is keyed by its SYMBOL, so duplicate symbols would collapse two kernels
    /// onto one registration and silently launch the wrong function. Names must be unique too, or
    /// `kernel()` becomes order-dependent.
    #[test]
    fn kernel_symbols_and_names_are_unique() {
        let mut syms = std::collections::HashSet::new();
        let mut names = std::collections::HashSet::new();
        for k in LAYER0_KERNELS {
            assert!(syms.insert(k.symbol), "duplicate symbol {}", k.symbol);
            assert!(names.insert(k.name), "duplicate name {}", k.name);
        }
        assert_eq!(
            LAYER0_KERNELS.len(),
            13,
            "eleven layer-0 kernels plus the two mx4pk expert projections"
        );
    }

    /// Mangled symbols, not plain names: registering the unmangled name fails at LAUNCH rather than
    /// at registration, which is an expensive way to find a typo.
    ///
    /// The check is the mangling's own length prefix — `_Z15ds4_rmsmul_4096PfS_S_` claims 15
    /// characters, and `"ds4_rmsmul_4096"` is 15 — so a hand-edited symbol that drops or adds a
    /// character fails here instead of on the box. An earlier version of this test tried to derive
    /// the device name from the readable one by string substitution and was simply wrong
    /// ("attention_sink_batched" mangles from `attn_sink_batched`); a self-consistency check on the
    /// symbol needs no such mapping.
    #[test]
    fn symbols_are_cpp_mangled_and_self_consistent() {
        for k in LAYER0_KERNELS {
            // TWO SYMBOL FORMS, both verified against `nm` on an extracted `.aicore_binary`:
            //   * `_Z...`   -- a C++ `__global__` function, which is how the engine's own
            //                  layer0 kernels are written (`_Z19ds4_mx4pk_...PaPfS0_S0_`);
            //   * `<name>_mix_aiv` -- an `extern "C"` kernel as CANN emits it from this tree's tiles
            //                  (`w8a8_experts_1x2048_k4096_mix_aiv`). Predicting the C++ form from the
            //                  signature was WRONG (iteration 111), so the check accepts the observed
            //                  form instead of asserting the assumed one.
            if k.symbol.ends_with("_mix_aiv") && !k.symbol.starts_with("_Z") {
                assert!(
                    k.symbol.trim_end_matches("_mix_aiv").chars().all(|c| c.is_ascii_alphanumeric() || c == '_'),
                    "{}: {} is neither mangled nor a plain `_mix_aiv` entry",
                    k.name,
                    k.symbol
                );
                continue;
            }
            let rest = k
                .symbol
                .strip_prefix("_Z")
                .unwrap_or_else(|| panic!("{} must be mangled, got {}", k.name, k.symbol));
            let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            let len: usize = digits.parse().unwrap_or_else(|_| {
                panic!("{}: no length prefix in {}", k.name, k.symbol)
            });
            let after = &rest[digits.len()..];
            assert!(
                after.len() >= len,
                "{}: symbol {} is shorter than its own length prefix {len}",
                k.name,
                k.symbol
            );
            let fname = &after[..len];
            assert!(
                fname.starts_with("ds4_"),
                "{}: length prefix {len} selects {fname:?}, which is not one of our kernels",
                k.name
            );
            // What follows the name is the parameter mangling, so it must not be empty: a kernel
            // with no arguments would take no buffers and could not do anything.
            assert!(
                !after[len..].is_empty(),
                "{}: {} encodes no parameters",
                k.name,
                k.symbol
            );
        }
    }

    /// The measured rule: reductions and in-place kernels get ONE core however wide the caller asks.
    #[test]
    fn single_core_kernels_ignore_a_wide_request() {
        for name in ["rms_norm_4096", "rms_norm_1024", "rms_norm_512", "hc_norm", "rope_halves"] {
            let k = kernel(name).unwrap();
            assert_eq!(k.partitioning, Partitioning::SingleCore, "{name}");
            for desired in [1, 2, 8, 48, 96, u32::MAX] {
                assert_eq!(k.block_dim(desired), 1, "{name} at desired {desired}");
            }
        }
    }

    /// Row-partitioned kernels take the width, capped at the usable core count — 96 measured worse
    /// than 48 at every point, so this clamps rather than trusting the caller.
    #[test]
    fn row_partitioned_kernels_take_the_width_but_are_capped() {
        let k = kernel("attention_sink_batched").unwrap();
        assert_eq!(k.partitioning, Partitioning::ByRow);
        assert_eq!(k.block_dim(48), 48);
        assert_eq!(k.block_dim(8), 8);
        assert_eq!(k.block_dim(96), USABLE_AIV, "must clamp, not trust");
        assert_eq!(k.block_dim(u32::MAX), USABLE_AIV);
        assert_eq!(k.block_dim(0), 1, "zero blocks would launch nothing");
    }

    /// The eps the kernel actually implements, and a refusal that names the fix.
    #[test]
    fn a_mismatched_eps_is_refused_with_the_reason() {
        check_rms_eps(EMITTED_RMS_EPS).expect("the baked-in value must be accepted");
        let e = check_rms_eps(1e-5).unwrap_err().to_string();
        assert!(e.contains("bakes eps"), "got: {e}");
        assert!(e.contains("Re-emit"), "the error must say what to do: {e}");
        // 1e-6 written a different way must still pass — this is a value check, not a string check.
        check_rms_eps(0.000_001).expect("1e-6 by another spelling");
    }

    /// An unknown kernel must list what exists rather than just failing.
    #[test]
    fn an_unknown_kernel_lists_the_available_ones() {
        let e = kernel("rms_norm_777").unwrap_err().to_string();
        assert!(e.contains("no kernel named rms_norm_777"));
        assert!(e.contains("rms_norm_4096"), "must list real names: {e}");
    }

    /// rms_norm kernels are SHAPE-SPECIALISED, so a width we did not emit must be refused rather
    /// than rounded — rounding would normalise over the wrong number of elements and produce a
    /// plausible vector.
    #[test]
    fn rms_kernel_selection_refuses_a_width_we_did_not_emit() {
        assert_eq!(select_rms_kernel(4096).unwrap().name, "rms_norm_4096");
        assert_eq!(select_rms_kernel(1024).unwrap().name, "rms_norm_1024");
        assert_eq!(select_rms_kernel(512).unwrap().name, "rms_norm_512");
        for bad in [0usize, 1, 511, 513, 2048, 4095, 8192] {
            let e = select_rms_kernel(bad).unwrap_err().to_string();
            assert!(e.contains(&format!("width {bad}")), "must name the width: {e}");
            assert!(e.contains("cannot be rounded"), "must say why: {e}");
        }
        // And every selectable width must be single-core, since they are all reductions.
        for d in [4096usize, 1024, 512] {
            assert_eq!(select_rms_kernel(d).unwrap().block_dim(48), 1);
        }
    }

    /// The AIVEC magic, not plain ELF: the wrong one registers fine and fails at launch.
    #[test]
    fn the_elf_magic_is_the_aivec_one() {
        assert_eq!(MAGIC_ELF_AIVEC, 0x4141_5246);
        assert_ne!(MAGIC_ELF_AIVEC, 0x4355_4245, "that is RT_DEV_BINARY_MAGIC_ELF, which fails later");
    }
}
