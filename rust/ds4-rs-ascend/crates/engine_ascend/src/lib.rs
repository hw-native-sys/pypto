// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! `engine_ascend` — the Huawei Ascend backend for the DS4 inference engine.
//!
//! Shape mirrors `the accelerator host crate` and `ds4_cuda`: `DecodeRunner` / `DecodeSession`
//! (the `ds4_server` surface) plus the CPU `sampling` module, so the server
//! reaches it through the existing `crate::backend` cfg-alias seam and no
//! server file needs to know which backend is compiled in.
//!
//! Decode is REAL and end-to-end: `DecodeSession::step` runs the composed chain
//! through `engine_core::decode_step`, `prefill` loops it, and `bin/ds4_generate`
//! tokenizes, prefills, greedy-decodes and streams text. What varies is how much
//! of each step runs on the NPU rather than the CPU.
//!
//! The chain is a hybrid by design. Every dispatcher method starts delegated to
//! the CPU and is swapped to a device launcher as its kernel is proven, so there
//! is a working baseline at every point during the swap. Ask
//! [`dispatcher::AscendDispatcher::offloaded_by`] what a given instance actually
//! launched — it keys on `KernelLaunch::is_device`, so a host fake exercising the
//! whole path still reports nothing. Today that is the routed-expert MXFP4
//! matvecs (the load-bearing case: the experts do not fit one chip, so layer L
//! goes to chip `L % n`), plus `rms_norm` and the Q8_0 matvec. Attention is still
//! CPU-side.
//!
//! Do not read a claim about device coverage out of this comment — read
//! `offloaded_by`. An earlier version of this header said the crate could not
//! decode at all, and stayed that way through four commits that made it untrue.
//!
//! Kernel provenance: every op comes from `rustc_codegen_tile`'s `mlir_to_pto`
//! backend, through `ptoas --pto-arch=a3` and `ccec --cce-aicore-arch=
//! dav-c220-vec`. Attention is the S-blocked lowering, which is what makes the
//! model's 512-wide MLA latent reachable at all — the one-shot form admitted
//! only S <= 96 there.
//!
//! GATE: the device layer compiles only under the `ascend` feature or a Linux
//! target; a bare macOS build type-checks everything without needing CANN.

pub mod attn_device;
pub mod decode_runner;
pub mod expert_place;
pub mod dispatcher;

/// Weight placement: measured free HBM in, a contiguous layer-range plan out.
///
/// The plan this crate started from placed layers by a fixed range against a
/// fixed per-chip budget. That does not survive a shared box — see the module
/// docs for the measurements that killed it.
pub mod mxfp4_repack;
pub mod attention;
pub mod ffn;
pub mod hc;
pub mod launch;
pub mod decomposition;
pub mod device;
pub mod parallel;
pub mod placement;
pub mod seq_attention;
pub mod sweep;
pub mod weights;

/// Which Ascend generation the kernels were built for, and the guard that
/// refuses to run them on silicon they do not match.
pub mod soc;
pub mod sampling;

/// Speculative decoding (MTP / DSpark) priced before it is built — the verify-K
/// round economics on the measured per-layer figures, and the break-even the
/// k-row kernel measurement has to beat.
pub mod spec;
