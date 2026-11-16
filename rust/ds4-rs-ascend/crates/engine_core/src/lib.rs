// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! engine_core library surface.
//!
//! The crate is primarily a binary (`ds4-infer`). This `lib.rs` exposes the
//! decode-critical kernel registry to integration tests and external crates
//! (e.g. `codegen_tests` for the emit-vs-registry drift sentinel).
//!
//! The `#[path]` attribute reuses the same source file as the binary, so
//! there is one source of truth for `KERNELS` and the drift-sentinel tests.

// `kernel_registry` lived here. It was not a list with some Apple-GPU rows in
// it -- it was a Metal kernel registry outright, keyed by the `[[kernel]]`
// symbol in a generated `.metal` file. Nothing on the Ascend path referenced it,
// so it is removed rather than filtered, and this repository's licence is the
// reason it could not be carried either way.

pub mod kvstore_format;
pub mod snapshot_store;

#[path = "forward.rs"]
pub mod forward;
pub mod kda_block;
pub mod hyper_connection;
pub mod moe_route;
pub mod glm53_ffn;
pub mod glm53_plan;

pub mod mla_block;
pub mod model_names;
#[path = "moe.rs"]
pub mod moe;

#[path = "dispatch.rs"]
pub mod dispatch;

#[path = "attn_dispatch.rs"]
pub mod attn_dispatch;

#[path = "layer_view.rs"]
pub mod layer_view;

pub mod engram;
#[path = "gguf.rs"]
pub mod gguf;
pub mod safetensors;
pub mod w8a8_roles;

/// Value-aware DS4_* boolean flag parsing. `DS4_X=0` must DISABLE — see the module docs
/// for the presence-only bug this exists to prevent.
#[path = "env_flags.rs"]
pub mod env_flags;

// `policy` lived here: a signed key/value file that could ship threshold
// changes without shipping a binary. The idea is sound and carries knobs rather
// than code, so it cannot change what the engine computes.
//
// It is removed for a reason worth stating plainly: its signature verifier was
// implemented against Apple's Security.framework, and on every other target
// `verify` returned `UnverifiablePlatform` -- it refused rather than trusted,
// which was the right call, but it also means the feature was INERT on every
// machine this utility is for. Carrying a platform FFI for a feature that
// cannot run on the target is the wrong trade, and this repository's licence
// does not cover that platform anyway.
//
// If it returns it needs a portable verifier. `env_flags` below no longer
// consults it, so the precedence rule is simply the environment.

#[path = "kv_cache.rs"]
pub mod kv_cache;

#[path = "decode_step.rs"]
pub mod decode_step;

#[path = "batched_branching.rs"]
pub mod batched_branching;

// `prefill_ffi` lived here: hand-written bindings to a vendored third-party
// engine, compiled by a build script that linked the Apple Metal and Foundation
// frameworks. Its purpose was comparing this engine's Metal dispatcher against
// that one as a correctness oracle on macOS.
//
// It is gone, and so is the build script, because this repository's licence
// does not cover developing for processors other than Huawei AI Processors.
// Removing it also removed every third-party dependency that bridge needed.
//
// NOTE: that oracle was a real correctness instrument, and the Ascend side has
// no equivalent yet. A CPU-reference comparison is owed here, not optional --
// see the decode path's own tests for what currently stands in for it.

#[path = "bench.rs"]
pub mod bench;

#[path = "tokenizer.rs"]
pub mod tokenizer;

#[path = "op_timer.rs"]
pub mod op_timer;

#[path = "mtp.rs"]
pub mod mtp;

/// GLM-5.2 (`glm_moe_dsa` / DeepSeek-V3.2) CPU reference forward — Phase 0
/// architecture port, validated against mlx-lm as the oracle. Additive and
/// gated behind `glm5_ref::ModelVariant::Glm5Dsa`; does not touch the
/// DS4-V4-Flash decode path.
#[path = "glm5_ref.rs"]
pub mod glm5_ref;

/// GLM-5.2 (`glm-dsa`) streaming weight source + forward — Phase 1b. Runs the
/// real ~226 GB unsloth GGUF (UD-IQ1_S, 6 shards) by mmap-streaming weights
/// from SSD and dequantizing on demand (only the top-k MoE experts per layer).
/// Same math as `glm5_ref` (cos=1.0 vs mlx-lm); gated behind GLM-5.2, does not
/// touch DS4-V4-Flash.
#[path = "glm5_stream.rs"]
pub mod glm5_stream;

#[path = "dspark.rs"]
pub mod dspark;

/// Qwen2 **dense** (`qwen2` GGUF architecture) config — the small, runs-anywhere
/// model family (e.g. `DeepSeek-R1-Distill-Qwen-1.5B`). Provides
/// `DefaultsQwen2::from_gguf`; the Metal forward lives in the host-side `qwen_session` helper.
/// Additive; does not touch the DS4-V4-Flash or GLM decode paths.
#[path = "qwen2.rs"]
pub mod qwen2;

/// Serialises tests that mutate PROCESS-GLOBAL environment variables.
///
/// `std::env::set_var` is process-wide while `cargo test` runs tests on parallel threads, so a
/// test that sets `DS4_*_FIDELITY` changes what an unrelated test observes mid-run. That is not
/// theoretical: `forward::tests::silu_antirez_fidelity_gate_switches_implementation` failed
/// intermittently with a 1-ULP diff, because `attn_dispatch`'s gate test had set
/// `DS4_SILU_FIDELITY=1` while it was asserting the gate was OFF.
///
/// Every test that sets OR clears one of these gates must hold this lock for the whole span in
/// which it depends on the value — a lock taken by only some participants protects nothing.
/// `unwrap_or_else(|p| p.into_inner())` so one panicking test does not poison the rest.
///
/// **Readers count as participants.** The first version of this lock was a `Mutex` taken only by
/// the gate-flipping tests, which is the "protects nothing" failure it warns about: a mutator
/// held the lock while ~76 other tests called `silu`/`rms_norm`/`matvec_f32`/`softplus_sqrt`
/// — all of which read a `DS4_*_FIDELITY` gate on every call — with no lock at all. The gate
/// could therefore flip *between two calls inside one test*, which is how
/// `forward::tests::matvec_f32_deterministic_under_parallel` failed (it asserts three
/// consecutive `matvec_f32` calls agree; a concurrent flip of `DS4_MATVEC_F32_FIDELITY` swapped
/// the reducer underneath it, 8 runs in 15).
///
/// So this is an `RwLock`, not a `Mutex`:
/// - a test that **mutates** a gate takes [`RwLockWriteGuard`](std::sync::RwLock::write) —
///   exclusive, so no reader can observe a torn value;
/// - a test that merely **calls gated code** takes [`read`](std::sync::RwLock::read) — shared,
///   so the ~76 read-only tests still run concurrently with each other and the suite does not
///   serialise.
///
/// `std::sync::RwLock` is not reentrant: never take `read` in a test that already holds `write`.
#[cfg(test)]
pub(crate) static ENV_LOCK: std::sync::RwLock<()> = std::sync::RwLock::new(());
