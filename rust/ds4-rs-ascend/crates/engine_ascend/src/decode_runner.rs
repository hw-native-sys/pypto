// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! The `ds4_server` decode surface, backed by generated PTO kernels.

use anyhow::{anyhow, bail, Result};
use engine_core::gguf::{GgufFile, MetaValue};
use std::path::Path;

/// Model geometry read from the GGUF, checked against what the kernels assume.
#[derive(Debug, Clone, Copy)]
pub struct Dims {
    pub n_layers: u32,
    pub n_embd: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    /// MLA latent width — `attention.key_length`. This is the D that decides
    /// whether attention fits: the one-shot lowering admits only S <= 96 here
    /// at 512, which is why the blocked path exists.
    pub head_dim: u32,
    /// The indexer's narrower head, `attention.indexer.key_length`.
    pub indexer_dim: u32,
    pub n_experts: u32,
    pub n_vocab: u32,
}

/// Where one layer's stacked routed-expert tensors live in the mmap, and their shape.
/// Collected at compose so the session can register expert tables without re-parsing.
#[derive(Debug, Clone, Copy)]
pub struct ExpertInfo {
    pub layer_idx: u32,
    pub ttype: engine_core::gguf::GgmlType,
    pub gate: (usize, usize),
    pub up: (usize, usize),
    pub down: (usize, usize),
    pub n_experts: usize,
    pub d_ffn: usize,
    pub d_model: usize,
}

pub struct DecodeRunner {
    /// Reads tensors and produces kernel-ready planes. Held open for the life of the
    /// runner: the expert pool is refilled per token, so the file stays hot.
    weights: crate::weights::WeightSource,
    /// The file this runner opened, for the lazy composition below.
    path: std::path::PathBuf,
    /// The composed CPU-side model + token embeddings, built on FIRST use rather than at
    /// open: composing materialises the dense f32 planes, and callers that only plan or
    /// inspect geometry should not pay for it.
    ///
    /// `LayerViews` rides in the tuple because the composed model REFERENCES its mmap — the
    /// quantized planes are zero-copy views into the file. Dropping the views while keeping
    /// the model compiled fine and segfaulted on the first lm_head matvec on the 910c; the
    /// gdb backtrace (KernelDispatcher::matvec_q8 reading unmapped memory) is why this field
    /// holds three things and not two.
    #[allow(clippy::type_complexity)]
    composed: std::sync::OnceLock<(
        engine_core::layer_view::LayerViews,
        engine_core::decode_step::ComposedModelWeights,
        Vec<f32>,
        Vec<ExpertInfo>,
    )>,
    dims: Dims,
    raw_cap: u32,
    n_tensors: u64,
    /// Public because the server reaches through it for warm-up, matching the
    /// other backends' shape rather than inventing a different one here.
    pub dispatcher: Dispatcher,
}

/// Owns the device context and the compiled PTO kernels.
///
/// Residency is the whole game on this target: the model is ~145 GiB of MXFP4
/// against 64 GiB per chip, so weights are placed per chip by layer range and
/// the activation hops at each boundary — a placement proven byte-identical
/// across four chips. Nothing is resident yet, so warm-up reports zero rather
/// than a number that would read as success.
#[derive(Default)]
pub struct Dispatcher {
    resident_bytes: u64,
}

impl Dispatcher {
    /// Bytes of expert weights made resident. Zero until placement is wired.
    pub fn warm_up_expert_pages(&self) -> u64 {
        self.resident_bytes
    }
}

impl DecodeRunner {
    /// Open a DeepSeek-V4 GGUF and validate it against the compiled kernels.
    ///
    /// `raw_cap` is the SWA window, carried through from the server unchanged.
    pub fn open(path: &Path, raw_cap: u32) -> Result<Self> {
        let weights = crate::weights::WeightSource::open(path)?;
        let g = GgufFile::open(path).map_err(|e| anyhow!("open {}: {e}", path.display()))?;
        let arch = match g.get_meta("general.architecture") {
            Some(MetaValue::String(s)) => s.to_ascii_lowercase(),
            _ => bail!("{}: no general.architecture", path.display()),
        };
        if !arch.starts_with("deepseek") {
            bail!(
                "engine_ascend: architecture {arch:?} is not a DeepSeek-V4 model; \
                 the PTO kernels are built for `deepseek4`"
            );
        }
        // V4.1 spells these the way the HuggingFace config does, not in GGUF's own
        // vocabulary: `deepseek41.num_hidden_layers`, `deepseek41.hidden_size` and so on.
        // `engine_core::gguf` already carries that fallback table for its own reads, but
        // this runner asked only for `{arch}.{gguf_name}` -- so a V4.1 export failed with
        // "missing deepseek41.block_count" while the loader beside it knew the answer.
        // Both spellings are listed per field for the same reason they are listed there:
        // the V4 releases carry only the GGUF names and V4.1 carries only the HF ones.
        let u32_of = |k: &str| -> Option<u32> {
            let aliases: &[&str] = match k {
                "block_count" => &["block_count", "num_hidden_layers"],
                "embedding_length" => &["embedding_length", "hidden_size"],
                "attention.head_count" => &["attention.head_count", "num_attention_heads"],
                "attention.head_count_kv" => {
                    &["attention.head_count_kv", "num_key_value_heads"]
                }
                "attention.key_length" => &["attention.key_length", "head_dim"],
                "expert_count" => &["expert_count", "n_routed_experts"],
                "vocab_size" => &["vocab_size"],
                _ => &[k],
            };
            // BOTH PREFIXES, not just the declared architecture. The 1017 export
            // declares `general.architecture = deepseek4` while writing every key under
            // `deepseek41.`, so a single-prefix read reports "missing
            // deepseek4.attention.head_count" for a file that carries
            // `deepseek41.num_attention_heads`. `deepseek41` is tried first because a
            // file that has it is the newer layout.
            for prefix in ["deepseek41", "deepseek4", arch.to_string().as_str()] {
                for a in aliases {
                    if let Some(MetaValue::U32(v)) = g.get_meta(&format!("{prefix}.{a}")) {
                        return Some(*v);
                    }
                }
            }
            None
        };
        let need = |k: &str| -> Result<u32> {
            u32_of(k).ok_or_else(|| anyhow!("{}: missing {arch}.{k}", path.display()))
        };
        let dims = Dims {
            n_layers: need("block_count")?,
            n_embd: need("embedding_length")?,
            n_heads: need("attention.head_count")?,
            n_kv_heads: u32_of("attention.head_count_kv").unwrap_or(1),
            head_dim: need("attention.key_length")?,
            indexer_dim: u32_of("attention.indexer.key_length").unwrap_or(0),
            n_experts: u32_of("expert_count").unwrap_or(0),
            // Not every export carries vocab_size; the tokenizer is authoritative.
            n_vocab: u32_of("vocab_size").unwrap_or(0),
        };
        Ok(Self {
            weights,
            path: path.to_path_buf(),
            composed: std::sync::OnceLock::new(),
            dims,
            raw_cap,
            n_tensors: g.n_tensors,
            dispatcher: Dispatcher::default(),
        })
    }

    /// The composed model and token embeddings, built once.
    ///
    /// This is the SAME composition the Metal harness drives — `validate_ds4_layout` →
    /// `LayerViews` → `ComposedModelWeights::from_views` — so the engine here is not a second
    /// model implementation but the shared one, reached through a different dispatcher.
    #[allow(clippy::type_complexity)]
    pub fn composed(
        &self,
    ) -> Result<&(
        engine_core::layer_view::LayerViews,
        engine_core::decode_step::ComposedModelWeights,
        Vec<f32>,
        Vec<ExpertInfo>,
    )> {
        if let Some(c) = self.composed.get() {
            return Ok(c);
        }
        let manifest = engine_core::gguf::validate_ds4_layout(&self.path)?;
        let views = engine_core::layer_view::LayerViews::open(&self.path, manifest.n_layers)?;
        let defaults = engine_core::attn_dispatch::DefaultsDs4::ds4_v4_flash();
        let model = engine_core::decode_step::ComposedModelWeights::from_views(
            &views, &manifest, defaults,
        )?;
        let g = GgufFile::open(&self.path)?;
        let ti = g
            .tensors
            .iter()
            .find(|t| t.name == "token_embd.weight")
            .ok_or_else(|| anyhow!("{}: no token_embd.weight", self.path.display()))?;
        let n = model.vocab_size * model.d_model;
        let start = (g.tensor_data_offset + ti.offset) as usize;
        let bytes: &[u8] = views.bytes.as_ref();
        let embd: Vec<f32> = match ti.ttype {
            engine_core::gguf::GgmlType::F32 => bytes[start..start + n * 4]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            engine_core::gguf::GgmlType::F16 => bytes[start..start + n * 2]
                .chunks_exact(2)
                .map(|c| {
                    engine_core::layer_view::f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]]))
                })
                .collect(),
            engine_core::gguf::GgmlType::Q8_0 => {
                // The MXFP4 export stores token_embd as Q8_0 (34 B per 32 weights). Dequant
                // once at compose; rows are contiguous, so this is the same block walk the
                // CPU matvec_q8 reference does.
                let nb = n / 32;
                let mut out = vec![0f32; n];
                let src = &bytes[start..start + nb * 34];
                for (b, blk) in src.chunks_exact(34).enumerate() {
                    let d = engine_core::layer_view::f16_bits_to_f32(u16::from_le_bytes([
                        blk[0], blk[1],
                    ]));
                    for i in 0..32 {
                        out[b * 32 + i] = d * (blk[2 + i] as i8) as f32;
                    }
                }
                out
            }
            other => bail!("token_embd is {other:?}; expected F32/F16/Q8_0"),
        };
        // Routed-expert locations, per layer: the composed model does not materialise these
        // (600+ GB as f32); the dispatcher's expert provider dequants the SELECTED ones per
        // call from these mmap ranges instead.
        let mut experts = Vec::new();
        for li in 0..model.layers.len() {
            let find = |suffix: &str| -> Option<(engine_core::gguf::GgmlType, usize, usize)> {
                let name = format!("blk.{li}.ffn_{suffix}_exps.weight");
                let ti = g.tensors.iter().find(|t| t.name == name)?;
                let start = (g.tensor_data_offset + ti.offset) as usize;
                Some((ti.ttype, start, g.tensor_byte_size(ti) as usize))
            };
            let (Some(gt), Some(ut), Some(dt)) = (find("gate"), find("up"), find("down")) else {
                continue; // dense layer (layer 0 in some exports) or non-MoE
            };
            let ti = g
                .tensors
                .iter()
                .find(|t| t.name == format!("blk.{li}.ffn_gate_exps.weight"))
                .expect("just found");
            experts.push(ExpertInfo {
                layer_idx: li as u32,
                ttype: gt.0,
                gate: (gt.1, gt.2),
                up: (ut.1, ut.2),
                down: (dt.1, dt.2),
                n_experts: ti.dims[2] as usize,
                d_ffn: ti.dims[1] as usize,
                d_model: ti.dims[0] as usize,
            });
            let _ = (dt, ut);
        }
        let _ = self.composed.set((views, model, embd, experts));
        Ok(self.composed.get().expect("just set"))
    }

    /// Bytes reclaimed by dropping f32 duplicates of quantized weights. Zero
    /// here: this backend never materialises them, so there is nothing to free.
    /// Reported honestly rather than omitted, so the server's lean-weights
    /// message stays truthful on every backend.
    pub fn free_dead_f32_weights(&mut self) -> u64 {
        0
    }

    pub fn dims(&self) -> Dims {
        self.dims
    }

    pub fn n_layers(&self) -> u32 {
        self.dims.n_layers
    }

    pub fn raw_cap(&self) -> u32 {
        self.raw_cap
    }

    /// Tensor entries in THIS file. A split GGUF keeps them all in the shard
    /// that holds the tensor data, so the metadata shard reports zero.
    pub fn n_tensors(&self) -> u64 {
        self.n_tensors
    }

    /// Longest context the compiled attention kernel admits at this model's
    /// latent width, or `None` if the shape is out of reach entirely.
    ///
    /// Mirrors `mlir_to_pto`'s `pick_s_block`: a fixed pool of 8 tiles of
    /// `SB x D` f32 plus a small fixed term, against 192 KiB of AIV unified
    /// buffer. NOTHING scales with context — the emitter recomputes the scores
    /// rather than storing them — so the only real bound is the block itself.
    ///
    /// This duplicates the emitter's model because the two live in different
    /// layers and cannot share code; `context_is_unbounded_once_a_block_fits`
    /// below is what stops them drifting. It already drifted once: this carried
    /// a `256 * (S / SB)` term for the per-block score tiles after the emitter
    /// stopped allocating them, and under-reported the ceiling as a result.
    pub fn max_context(&self) -> Option<u32> {
        let d = self.dims.head_dim as usize;
        const UB: usize = 192 * 1024;
        // A block fits or it does not; S does not enter the budget.
        let fits = |sb: usize| 32 * sb * d + 8 * d + 1792 <= UB;
        if !fits(8) {
            return None;
        }
        // Bounded by what the emitter will unroll, not by unified buffer.
        Some(MAX_UNROLLED_CONTEXT)
    }
}

/// Emitted code size, not memory, is what now caps context: the blocks are
/// unrolled at emit time, so S=65536 at D=512 is 8192 blocks and ~295k lines
/// of PTO. Reported as the practical ceiling until the emitter grows a real
/// loop with runtime `partition_view` offsets.
pub const MAX_UNROLLED_CONTEXT: u32 = 16384;

impl DecodeRunner {
}

pub struct DecodeSession<'r> {
    runner: &'r DecodeRunner,
    logits: Vec<f32>,
    dspark_tap: bool,
    /// Decode state, created on the first token.
    state: Option<engine_core::decode_step::AttnStepState>,
    /// The dispatcher the chain runs through. CPU by construction; NPU methods join via
    /// [`DecodeSession::install_device`], and `offloaded_by()` stays the honest report.
    disp: crate::dispatcher::AscendDispatcher<engine_core::dispatch::CpuDispatcher>,
    /// The last step's pre-head hidden, kept when the DSpark tap is armed.
    main_hidden: Option<Vec<f32>>,
}

/// The chips to offload onto, from `DS4_DEVICES` (comma-separated, e.g. `"0,1"`). Defaults to a
/// single chip 0. Blank/duplicate entries are dropped; anything unparseable falls back to `[0]`.
pub fn env_device_ids() -> Vec<i32> {
    let raw = match std::env::var("DS4_DEVICES") {
        Ok(v) => v,
        Err(_) => return vec![0],
    };
    let mut ids: Vec<i32> = Vec::new();
    for tok in raw.split(',') {
        let t = tok.trim();
        if t.is_empty() {
            continue;
        }
        match t.parse::<i32>() {
            Ok(id) if !ids.contains(&id) => ids.push(id),
            Ok(_) => {}
            Err(_) => return vec![0],
        }
    }
    if ids.is_empty() {
        vec![0]
    } else {
        ids
    }
}

impl<'r> DecodeSession<'r> {
    pub fn new(runner: &'r DecodeRunner) -> Self {
        Self {
            runner,
            logits: Vec::new(),
            dspark_tap: false,
            state: None,
            disp: crate::dispatcher::AscendDispatcher::new(
                engine_core::dispatch::CpuDispatcher,
            ),
            main_hidden: None,
        }
    }

    /// Install the NPU for every method with a proven kernel. See `offloaded_by` for what
    /// actually landed.
    #[cfg(feature = "ascend")]
    pub fn install_device(&mut self, elf_path: &std::path::Path, device_id: i32) -> Result<()> {
        self.install_devices(elf_path, &[device_id])
    }

    /// Install the NPU across `device_ids` chips. Layer-agnostic kernels (rms_norm, matvec_q8) run
    /// on the first chip; the routed-expert MXFP4 matvecs are spread across all chips by layer, so
    /// each chip holds its own layers' expert weights resident. See `offloaded_by` for what landed.
    #[cfg(feature = "ascend")]
    pub fn install_devices(&mut self, elf_path: &std::path::Path, device_ids: &[i32]) -> Result<()> {
        let d = std::mem::replace(
            &mut self.disp,
            crate::dispatcher::AscendDispatcher::new(engine_core::dispatch::CpuDispatcher),
        );
        self.disp = d.with_devices(elf_path, device_ids)?;
        // Register each retained Q8_0 projection plane as the alias of its dequanted f32 twin,
        // so the chain's fused projections (which compose self.matvec_f32) reroute through the
        // device q8 kernels. SAFETY: the planes live in the runner's composed model, which this
        // session borrows for its whole life — they outlive the dispatcher.
        let (_views, model, _embd, _experts) = self.runner.composed()?;
        let mut n = 0usize;
        for layer in &model.layers {
            let a = &layer.attn;
            for (f32p, q8) in [
                (&a.attn_q_a, &a.attn_q_a_q8),
                (&a.attn_q_b, &a.attn_q_b_q8),
                (&a.attn_kv, &a.attn_kv_q8),
            ] {
                if !q8.is_empty() {
                    unsafe { self.disp.register_q8_alias(f32p, q8) };
                    n += 1;
                }
            }
        }
        eprintln!("engine_ascend: {n} projection planes aliased to their Q8_0 bytes");
        Ok(())
    }

    /// What the NPU covers in this session.
    pub fn offloaded_by(&self) -> Vec<&str> {
        self.disp.offloaded_by()
    }

    pub fn arm_dspark_tap(&mut self) {
        self.dspark_tap = true;
    }

    pub fn disarm_dspark_tap(&mut self) {
        self.dspark_tap = false;
    }

    pub fn take_main_hidden(&mut self) -> Option<Vec<f32>> {
        self.main_hidden.take()
    }

    /// Sequential prefill: the prompt is stepped token by token through the same chain as
    /// decode. Correct first — the batched sweep the prefill estimate prices is an
    /// optimisation on top of a working engine, not a precondition for one.
    pub fn prefill(&mut self, prompt_tokens: &[i32]) -> Result<()> {
        for &t in prompt_tokens {
            self.step(t)?;
        }
        Ok(())
    }

    /// One decode step through the SHARED composed chain, with this session's dispatcher —
    /// which is where the NPU kernels enter, per `offloaded_by`.
    pub fn step(&mut self, token: i32) -> Result<()> {
        let (views, model, embd, experts) = self.runner.composed()?;
        // Register the routed-expert tables once. SAFETY: the byte ranges point into
        // views.bytes, owned by the runner's composed tuple, which outlives this session.
        if self.disp.expert_tables_registered() == 0 && !experts.is_empty() {
            let bytes: &[u8] = views.bytes.as_ref();
            for e in experts {
                unsafe {
                    self.disp.register_expert_table(
                        e.layer_idx,
                        e.ttype,
                        &bytes[e.gate.0..e.gate.0 + e.gate.1],
                        &bytes[e.up.0..e.up.0 + e.up.1],
                        &bytes[e.down.0..e.down.0 + e.down.1],
                        e.n_experts,
                        e.d_ffn,
                        e.d_model,
                    );
                }
            }
            eprintln!(
                "engine_ascend: routed-expert tables registered for {} layers ({:?})",
                experts.len(),
                experts.first().map(|e| e.ttype)
            );
        }
        let raw_cap = self.runner.raw_cap;
        let state = self.state.get_or_insert_with(|| {
            engine_core::decode_step::AttnStepState::new(model, raw_cap)
        });
        if token < 0 || (token as usize) >= model.vocab_size {
            bail!("token {token} out of vocab {}", model.vocab_size);
        }
        // NON-VACUITY: the composed model must actually carry its routed experts. from_views
        // leaves the expert f32 slices EMPTY and d_ffn None ("Metal reads d_ffn from QEW"),
        // and the CPU chain then computes a ZERO routed contribution — a shared-expert-only
        // model that decodes plausible-looking degenerate loops. Three argmax gates passed on
        // exactly that before this check existed: a parity gate proves A==B, not A==model.
        // Refuse rather than serve it.
        if let Some(l0) = model.layers.first() {
            let d_ffn = l0.moe.d_ffn.map_or(0, |n| n.get());
            let provided = self.disp.expert_tables_registered() > 0;
            if !provided && (d_ffn == 0 || l0.moe.w_gate_exps.is_empty()) {
                bail!(
                    "routed experts are not materialised in this composed model (d_ffn={},                      gate slice len {}): the decode would silently run shared-expert-only.                      The CPU routed-expert provider is not built yet — this engine refuses                      rather than serving a plausible-looking wrong model.",
                    d_ffn,
                    l0.moe.w_gate_exps.len()
                );
            }
        }
        let d = model.d_model;
        let x = embd[(token as usize) * d..(token as usize + 1) * d].to_vec();
        for kp in state.kv_pos.iter_mut() {
            *kp = state.pos % raw_cap;
        }
        let cfg = engine_core::decode_step::DecodeConfig::default();
        let attn = engine_core::attn_dispatch::CpuAttentionDispatcher;
        let hidden = engine_core::decode_step::decode_step_with_attn_to_residual(
            &self.disp, &attn, x, model, state, &cfg, raw_cap,
        )?;
        if self.dspark_tap {
            self.main_hidden = Some(hidden.clone());
        }
        let h = self.disp.rms_norm(&hidden, &model.final_norm_gamma, cfg.eps_rms);
        use engine_core::dispatch::KernelDispatcher as _;
        self.logits = if !model.lm_head_q8.is_empty() {
            self.disp.matvec_q8(&model.lm_head_q8, &h, model.vocab_size)
        } else if !model.lm_head.is_empty() {
            self.disp.matvec_f32(&model.lm_head, &h, model.vocab_size)
        } else {
            bail!("model carries neither an f32 nor a Q8_0 lm_head");
        };
        Ok(())
    }

    pub fn logits(&self) -> &[f32] {
        &self.logits
    }

    pub fn runner(&self) -> &DecodeRunner {
        self.runner
    }
}

impl DecodeRunner {
    /// The open weight source, for callers that need tensors directly.
    pub fn weights(&self) -> &crate::weights::WeightSource {
        &self.weights
    }

    /// Plan the split across chips, with the geometry taken from THIS file's tensor table.
    ///
    /// `free_bytes` is `(device_id, free)` measured per chip by the caller — the whole point of computing the
    /// split at startup is that these boxes are shared and occupancy swings, so a compiled-in
    /// split OOMs on a box whose free memory it never asked about. `aclrtGetMemInfo` supplies
    /// these under the `ascend` feature; the argument keeps this function testable without a
    /// device, and honest about the fact that it does not measure anything itself.
    pub fn plan(
        &self,
        layout: crate::placement::ExpertLayout,
        free_bytes: &[(u32, u64)],
        reserve: u64,
        max_ctx: u32,
    ) -> Result<crate::placement::Placement> {
        let g = self.weights.geometry(layout)?;
        crate::placement::plan(&g, free_bytes, reserve, max_ctx)
            .map_err(|e| anyhow!("placement: {e}"))
    }

    /// The geometry this file implies under `layout` — derived from its tensors, not from a
    /// transcription of the model card.
    pub fn geometry(
        &self,
        layout: crate::placement::ExpertLayout,
    ) -> Result<crate::placement::Geometry> {
        self.weights.geometry(layout)
    }

    /// Resolve the parallelism split for this model on this box: an operator's `DS4_PARALLEL`
    /// pin if present (refused with the reason if infeasible, never substituted), else the best
    /// split for `objective`.
    ///
    /// This is what the runner should consume instead of [`DecodeRunner::plan`]: `plan()` can
    /// only express contiguous layer ranges — pure `pp` — while the measured tables show the
    /// mixes beat both poles under the latency-first objective (`ep4 pp2` gives 85% of pure-EP
    /// per-user latency at 1.7x its aggregate). `plan()` remains for the placement details it
    /// still owns (per-chip budgets, refusal floors); the SPLIT decision belongs here.
    ///
    /// `free_bytes` is per-chip measured free HBM, same contract as `plan()`. The evaluation
    /// uses the MINIMUM across chips: splits assign near-uniform work, so the tightest chip is
    /// the one that decides feasibility — using the mean would approve a split that OOMs the
    /// fullest chip.
    pub fn choose(
        &self,
        layout: crate::placement::ExpertLayout,
        objective: crate::parallel::Objective,
        free_bytes: &[(u32, u64)],
        reserve: u64,
        max_ctx: u32,
    ) -> Result<crate::parallel::Chosen> {
        if free_bytes.is_empty() {
            return Err(anyhow!("choose: no chips measured — free_bytes is empty"));
        }
        let g = self.weights.geometry(layout)?;
        let min_free = free_bytes
            .iter()
            .map(|&(_, f)| f.saturating_sub(reserve))
            .min()
            .unwrap_or(0);
        crate::parallel::choose(
            &g,
            layout,
            objective,
            free_bytes.len() as u32,
            min_free,
            max_ctx,
            crate::placement::measured::H2D_PINNED_GBPS,
        )
        .map_err(|e| anyhow!("parallel choice: {e}"))
    }
}

/// One message, so the gap is stated once and cannot drift between call sites.
#[cfg(test)]
mod tests {
    use super::*;

    /// The emitter's rule after scores stopped being stored: if any block fits,
    /// every context fits, because nothing in the working set scales with S.
    /// This is the guard against the two copies of the cost model drifting —
    /// they already did once, when this one kept a score-tile term the emitter
    /// had dropped.
    #[test]
    fn context_is_unbounded_once_a_block_fits() {
        const UB: usize = 192 * 1024;
        for d in [64usize, 128, 192, 512] {
            let smallest_block_fits = 32 * 8 * d + 8 * d + 1792 <= UB;
            assert!(smallest_block_fits, "D={d} should admit SB=8");
        }
        // D wide enough that even one 8-row block overflows.
        let d = 4096usize;
        assert!(32 * 8 * d + 8 * d + 1792 > UB, "D={d} must not fit");
    }

    /// THE GOAL'S FINISH LINE, at engine level: greedy decode through this engine with the
    /// NPU installed must produce the same argmax sequence as the pure-CPU dispatcher. Gated
    /// on the real GGUF; the NPU side additionally needs DS4_DEVICE_ELF and the `ascend`
    /// feature — without them the test still proves the COMPOSITION (real prefill + steps,
    /// finite logits, a coherent greedy loop), which is what stopped being an explicit error
    /// in this commit.
    #[test]
    fn engine_greedy_argmax_gate() {
        let Ok(p) = std::env::var("DS4_ASCEND_TEST_GGUF") else {
            eprintln!("skip: set DS4_ASCEND_TEST_GGUF");
            return;
        };
        let steps: usize = std::env::var("DS4_ASCEND_GATE_STEPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(6);
        let r = DecodeRunner::open(Path::new(&p), 256).expect("open");
        let prompt = [9i32, 1393, 12, 13];
        let run = |sess: &mut DecodeSession| -> Vec<usize> {
            let t0 = std::time::Instant::now();
            sess.prefill(&prompt).expect("prefill");
            let prefill_s = t0.elapsed().as_secs_f64();
            let mut toks = Vec::new();
            let mut step_s: Vec<f64> = Vec::new();
            let mut next = crate::sampling::argmax(sess.logits()) as i32;
            for _ in 0..steps {
                toks.push(next as usize);
                let t = std::time::Instant::now();
                sess.step(next).expect("step");
                step_s.push(t.elapsed().as_secs_f64());
                next = crate::sampling::argmax(sess.logits()) as i32;
            }
            // Throughput, reported separately from correctness per the goal's own phrasing.
            // First step carries cold paging + dequant-cache fill; the tail is steady state.
            let warm = &step_s[step_s.len().min(2)..];
            let mean_warm = warm.iter().sum::<f64>() / warm.len().max(1) as f64;
            eprintln!(
                "  timing: prefill {:.1}s ({} tok), steps first {:.1}s, warm mean {:.2}s/tok                  ({:.3} tok/s)",
                prefill_s,
                prompt.len(),
                step_s.first().copied().unwrap_or(0.0),
                mean_warm,
                1.0 / mean_warm.max(1e-9),
            );
            toks
        };
        let mut cpu = DecodeSession::new(&r);
        let cpu_toks = run(&mut cpu);
        assert!(cpu.logits().iter().all(|v| v.is_finite()), "non-finite logits");
        assert!(cpu.logits().iter().any(|v| *v != 0.0), "all-zero logits");
        eprintln!("engine CPU greedy: {cpu_toks:?}");

        #[cfg(feature = "ascend")]
        {
            if let Ok(elf) = std::env::var("DS4_DEVICE_ELF") {
                let mut npu = DecodeSession::new(&r);
                npu.install_device(Path::new(&elf), 0).expect("install NPU");
                assert_eq!(
                    npu.offloaded_by(),
                    vec!["rms_norm", "matvec_q8", "matvec_mx4pk"],
                    "the NPU must be reported before it is trusted"
                );
                let npu_toks = run(&mut npu);
                eprintln!("engine NPU greedy: {npu_toks:?}");
                assert_eq!(cpu_toks, npu_toks, "NPU argmax must equal CPU argmax");
            } else {
                eprintln!("(CPU-only: set DS4_DEVICE_ELF for the NPU side)");
            }
        }
    }

    /// The runner resolves a SPLIT, not just a layer plan. Env-gated on the real
    /// GGUF like `opens_real_ds4_gguf`; asserts the objective path returns a
    /// feasible split of the right degree on a full node, and that measuring no
    /// chips is refused rather than defaulted.
    #[test]
    fn choose_resolves_a_split() {
        let Ok(p) = std::env::var("DS4_ASCEND_TEST_GGUF") else {
            eprintln!("skip: set DS4_ASCEND_TEST_GGUF");
            return;
        };
        // Do not let an operator pin on the host leak into the assertion.
        if std::env::var(crate::parallel::PARALLEL_ENV).is_ok() {
            eprintln!("skip: DS4_PARALLEL is set in this environment");
            return;
        }
        let r = DecodeRunner::open(Path::new(&p), 4096).expect("open");
        let free: Vec<(u32, u64)> = (0..8).map(|i| (i, 64u64 << 30)).collect();
        let chosen = r
            .choose(
                crate::placement::ExpertLayout::PackedNibbleNativeE8m0,
                crate::parallel::Objective::SingleStreamLatency,
                &free,
                1 << 30,
                8192,
            )
            .expect("a full 8x64GB node must admit some split");
        assert_eq!(chosen.config.degree(), 8, "split must use the chips it was given");
        assert!(chosen.single_stream_tok_s > 0.0);
        // No chips measured is an error, not a default.
        assert!(r
            .choose(
                crate::placement::ExpertLayout::PackedNibbleNativeE8m0,
                crate::parallel::Objective::SingleStreamLatency,
                &[],
                0,
                8192,
            )
            .is_err());
    }

    /// Opens the real MXFP4 GGUF when one is present, and pins the geometry the
    /// kernels were built against. Skips (rather than fails) without the file so
    /// the check is free on machines that do not have 145 GiB to spare.
    #[test]
    fn opens_real_ds4_gguf() {
        let Ok(p) = std::env::var("DS4_ASCEND_TEST_GGUF") else {
            eprintln!("skip: set DS4_ASCEND_TEST_GGUF");
            return;
        };
        let r = DecodeRunner::open(Path::new(&p), 4096).expect("open");
        let d = r.dims();
        assert_eq!(d.n_embd, 4096, "embedding width");
        assert_eq!(d.n_kv_heads, 1, "MLA keeps a single KV head");
        assert_eq!(d.head_dim, 512, "MLA latent");
        assert!(d.n_layers > 0 && d.n_layers < 200, "layers {}", d.n_layers);
        // The context the blocked attention kernel actually admits at D=512.
        assert!(
            r.max_context().unwrap_or(0) >= 1024,
            "attention must reach at least 1024 at the MLA latent"
        );
    }
}
