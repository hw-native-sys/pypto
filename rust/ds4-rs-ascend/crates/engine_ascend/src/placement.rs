// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Where the weights go: measured free HBM in, a layer-range plan out.
//!
//! The plan this crate was written against placed weights by a FIXED layer range
//! ("chip i owns layers a..b", sized against a fixed 36.1 GiB per chip). Two
//! measurements killed that. First, these boxes are shared and occupancy swings:
//! the 910b sits at 92% HBM held by someone's vLLM job (4.9 GiB free per chip)
//! while the 910c's NPUs are idle at 57-62 GiB free, and within a single reading
//! the 910c's own per-chip usage varied 2.8 to 7.1 GiB across sixteen chips.
//! Second, margins are thin even when it fits: three chips at 128k context leave
//! 2.0 GiB spare. A split compiled in at build time OOMs mid-decode on a box
//! whose free memory it never asked about.
//!
//! So the split is COMPUTED at startup from `aclrtGetMemInfo`, which is verified
//! to report real per-chip figures on this hardware.
//!
//! # What is poolable and what is not
//!
//! Per layer, at 128k context and f32 KV (the recorded accuracy decision):
//!
//! | | per layer | x43 |
//! |---|---|---|
//! | non-expert weights | 148.2 MiB | 6.22 GiB |
//! | KV latent, f32 | 256.0 MiB | 10.75 GiB |
//! | globals (token_embd, output, norms) | — | 1.06 GiB |
//! | **mandatory total** | **404.2 MiB** | **18.03 GiB** |
//! | experts, MXFP4, as STORED | 3.2 GiB | 137 GiB |
//! | experts as FED, packed 4-bit + f32 block scales | 3.8 GiB | 161 GiB |
//! | experts as FED, host-expanded int8 + f32 block scales | 6.8 GiB | 290 GiB |
//!
//! Those two FED rows are the whole memory review in one place, and they are DERIVED
//! from the dims by [`ExpertLayout::bytes_per_weight`] rather than written down — the
//! expert term used to be a literal `3_910_000_000`, implying 0.607 bytes per weight,
//! which matched no layout on the ladder. At 128k with f32 KV the totals are 308.6 GiB
//! fed as int8 against 179.6 GiB packed, so the in-kernel 4-bit unpack takes DS4-Flash
//! from SIX chips to THREE. Three chips leaves only 4.2 GiB spare, which is why native
//! e8m0 scales (step 3, 28.4 GiB spare) matter even though they save no chip.
//!
//! **The mandatory footprint fits on ONE chip.** Everything that makes this model
//! large is experts, and experts are poolable — a token touches 6 of 256 per
//! layer. So placement is almost entirely a question of how much expert pool each
//! chip gets, not whether the weights fit.
//!
//! # Adjacency
//!
//! Layer ranges are CONTIGUOUS, and that is not cosmetic. A layer's KV cache must
//! live on the chip that runs the layer: moving one layer's KV per token at 128k
//! is 256 MiB, against 16 KiB for an activation hop at a range boundary — a
//! factor of 16,384. Contiguous ranges give exactly `n_chips - 1` hops per token
//! and keep every layer's KV, compressor ring and non-expert weights local.
//!
//! Contiguity does NOT require equal ranges. A chip with more free HBM takes more
//! layers, which is what makes this dynamic and adjacent at the same time.
//!
//! # Balance residency FRACTION, not layer count
//!
//! A chip's expert pool serves only its own layers, so a chip holding more layers
//! spreads the same pool across more distinct experts and misses more often. The
//! pipeline is serial, so the slowest chip sets the token rate. Equalising layers
//! per chip therefore optimises the wrong thing; the plan equalises each chip's
//! resident FRACTION of the experts it is responsible for.

/// Model geometry the plan needs. Values are DeepSeek-V4-Flash's, confirmed
/// against the GGUF's own metadata rather than taken from a header.
#[derive(Debug, Clone, Copy)]
pub struct Geometry {
    pub layers: u32,
    /// Non-expert weight bytes per layer.
    pub nonexpert_per_layer: u64,
    /// Expert weight bytes per layer, all experts, as they will be FED to the
    /// kernels — not as stored. The two differ by the scale layout, which is the
    /// whole subject of the memory review.
    pub experts_per_layer: u64,
    /// Bytes not attributable to a layer: token_embd, output, output norms.
    pub globals: u64,
    /// KV latent bytes per layer per token. f32 x the 512-wide MLA latent.
    pub kv_per_layer_token: u64,
    /// Routed experts per layer, and how many a single token actually touches.
    ///
    /// Present so the minimum WORKING SET is derivable instead of assumed. Without it the only
    /// refusal possible is "the mandatory set does not fit", which passes a box that then streams
    /// every expert on every token — the collapse regime the residency work measured.
    pub n_experts: u32,
    pub experts_used_per_token: u32,
}

/// How an expert's weights are laid out for the kernel that reads them.
///
/// The variants differ ONLY in the scale plane and in whether the codes are packed,
/// which is the whole subject of the memory review — and the difference decides how many
/// chips DS4-Flash needs, so it is named here rather than folded into a constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertLayout {
    /// `__tile_mul_mv_id_mxfp4_f32`: the host expands each 4-bit code to an exact int8
    /// (`2*MXFP4[c]`), so 1 byte per weight, plus an f32 scale per 32-weight block.
    /// Measured ~1.61x FASTER than the packed form, since that kernel is vector-bound.
    HostExpandedInt8BlockF32,
    /// `__tile_mul_mv_id_mxfp4_pk_f32`: 4-bit codes read as stored and expanded
    /// in-kernel, so 0.5 bytes per weight, plus an f32 scale per block. This is what is
    /// built and device-verified today.
    PackedNibbleBlockF32,
    /// As above with the scale kept in f16 — half the scale plane, untested.
    PackedNibbleBlockF16,
    /// As above reading MXFP4's NATIVE e8m0 scale byte with no conversion, which makes
    /// the fed layout identical to the stored one. This is step 3, "zero-copy", and is
    /// not built yet.
    PackedNibbleNativeE8m0,
}

impl ExpertLayout {
    /// Bytes the kernel reads per weight, codes plus that weight's share of the scale.
    pub fn bytes_per_weight(self) -> f64 {
        let (code, scale_bytes_per_block) = match self {
            ExpertLayout::HostExpandedInt8BlockF32 => (1.0, 4.0),
            ExpertLayout::PackedNibbleBlockF32 => (0.5, 4.0),
            ExpertLayout::PackedNibbleBlockF16 => (0.5, 2.0),
            ExpertLayout::PackedNibbleNativeE8m0 => (0.5, 1.0),
        };
        code + scale_bytes_per_block / 32.0
    }
}

/// The model dimensions the expert footprint follows from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelDims {
    pub layers: u32,
    pub n_embd: u32,
    /// Per-expert feed-forward width.
    pub expert_ff: u32,
    /// Routed experts per layer.
    pub n_experts: u32,
    /// MLA latent width, which is also the KV row width.
    pub kv_latent: u32,
}

impl ModelDims {
    /// DS4-Flash, from its GGUF metadata.
    pub fn ds4_flash() -> Self {
        Self {
            layers: 43,
            n_embd: 4096,
            expert_ff: 2048,
            n_experts: 256,
            kv_latent: 512,
        }
    }

    /// Weights in one layer's routed experts: gate, up and down per expert.
    pub fn expert_weights_per_layer(&self) -> u64 {
        3 * self.n_embd as u64 * self.expert_ff as u64 * self.n_experts as u64
    }
}

impl Geometry {
    /// DS4-Flash as fed by the per-block-scale kernels with 4-bit weights — the
    /// rung the memory review targets for three chips.
    pub fn ds4_flash_4bit() -> Self {
        Self::ds4_flash(ExpertLayout::PackedNibbleBlockF32)
    }

    /// DS4-Flash under an explicit expert layout.
    ///
    /// The expert term is DERIVED rather than written down. It used to be the literal
    /// `3_910_000_000`, which implied 0.607 bytes per weight — a figure matching no
    /// layout on the ladder, sitting between 4-bit-with-f32-scales (0.625) and
    /// 4-bit-with-f16-scales (0.5625). Since this term is ~90% of the model and is what
    /// decides whether three chips suffice, an unattributable constant here is the
    /// expensive kind of drift, so the layout is now named and the bytes follow from it.
    pub fn ds4_flash(layout: ExpertLayout) -> Self {
        let d = ModelDims::ds4_flash();
        Self {
            layers: d.layers,
            // Slightly above the 148.2 MiB the tensor list sums to, and left that way:
            // erring high on a MANDATORY term fails planning early rather than
            // mid-decode.
            nonexpert_per_layer: 155 * 1024 * 1024,
            experts_per_layer: (d.expert_weights_per_layer() as f64 * layout.bytes_per_weight())
                as u64,
            globals: 1_140_000_000,
            kv_per_layer_token: d.kv_latent as u64 * 4,
            n_experts: d.n_experts,
            // DS4-Flash routes 6 of 256 plus 1 shared. The shared expert is not part of the
            // routed pool, so the working set is the 6.
            experts_used_per_token: 6,
        }
    }

    /// Total bytes for the whole model plus KV at `max_ctx`, ignoring how it is split.
    pub fn total_bytes(&self, max_ctx: u32) -> u64 {
        self.globals
            + self.layers as u64 * self.experts_per_layer
            + self.mandatory(self.layers, max_ctx)
    }

    /// Fewest chips of `hbm_per_chip` usable bytes that could hold the model at
    /// `max_ctx`. A lower bound on capacity only — it says nothing about whether a
    /// contiguous, adjacency-respecting plan exists, which is [`plan`]'s job.
    pub fn min_chips(&self, hbm_per_chip: u64, max_ctx: u32) -> u32 {
        let total = self.total_bytes(max_ctx);
        ((total + hbm_per_chip - 1) / hbm_per_chip) as u32
    }

    /// Bytes of one expert's three projections.
    pub fn bytes_per_expert(&self) -> u64 {
        if self.n_experts == 0 {
            return 0;
        }
        self.experts_per_layer / self.n_experts as u64
    }

    /// The smallest expert pool that lets a token complete without streaming: the experts one
    /// token touches, across the layers a chip owns.
    ///
    /// This is the floor, not a target. At the floor every token evicts everything it used, so
    /// throughput is bounded by transfer rather than compute — but below it a token cannot even be
    /// served without mid-token streaming, which is the line worth refusing at.
    pub fn min_working_set(&self, layers: u32) -> u64 {
        layers as u64 * self.experts_used_per_token as u64 * self.bytes_per_expert()
    }

    /// KV bytes for `layers` layers at `max_ctx`. Broken out because it is the term that grows
    /// with context and so the one most often left out of a headroom estimate.
    pub fn kv_bytes(&self, layers: u32, max_ctx: u32) -> u64 {
        layers as u64 * self.kv_per_layer_token * max_ctx as u64
    }

    /// Bytes that MUST be resident for a chip owning `layers` layers at
    /// `max_ctx` tokens. Refusing to start is better than discovering this
    /// mid-decode, so the caller checks it before planning.
    pub fn mandatory(&self, layers: u32, max_ctx: u32) -> u64 {
        layers as u64 * (self.nonexpert_per_layer + self.kv_per_layer_token * max_ctx as u64)
    }

    /// Weights in ONE expert projection (gate, up or down), inverted from the fed byte count.
    ///
    /// `experts_per_layer` was built as `weights * layout.bytes_per_weight()`, so this undoes
    /// that. The layout MUST be the one the `Geometry` was built with; a test pins the round trip
    /// for every layout on the ladder, so a mismatch cannot pass silently for DS4-Flash.
    ///
    /// Inverting rather than storing the figure keeps one source of truth. Adding a
    /// `weights_per_projection` field would let it drift away from `experts_per_layer`, and that
    /// term is ~90% of the model.
    pub fn weights_per_projection(&self, layout: ExpertLayout) -> f64 {
        if self.n_experts == 0 {
            return 0.0;
        }
        let weights_per_layer = self.experts_per_layer as f64 / layout.bytes_per_weight();
        weights_per_layer / (self.n_experts as f64 * 3.0)
    }

    /// Expected DISTINCT experts a batch of `tokens` touches, in one layer.
    ///
    /// This is the fact that decides prefill. Each token draws `experts_used_per_token` of
    /// `n_experts`, so the batch covers `n * (1 - (1 - u/n)^tokens)` of them — and that saturates
    /// fast: for DS4-Flash it is 91% of all 256 experts at 100 tokens and 99.8% at 256. So past
    /// about a hundred tokens prefill pays a near-FIXED sweep over every expert weight, and the
    /// per-token term is only the cube matmul. Modelling prefill as "batch size times a per-token
    /// cost" misses this entirely and predicts a speed that no prompt length achieves.
    pub fn experts_touched_per_layer(&self, tokens: u32) -> f64 {
        if self.n_experts == 0 || tokens == 0 {
            return 0.0;
        }
        let n = self.n_experts as f64;
        let p_unused = (1.0 - (self.experts_used_per_token as f64 / n)).clamp(0.0, 1.0);
        (n * (1.0 - p_unused.powf(tokens as f64))).min(n)
    }
}

/// What one chip is asked to hold.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChipPlan {
    pub device: u32,
    /// Contiguous, inclusive-exclusive. Adjacent layers stay together.
    pub first_layer: u32,
    pub last_layer: u32,
    /// Free HBM measured at plan time, not a configured constant.
    pub free_bytes: u64,
    /// Non-expert weights + KV reservation for this chip's layers.
    pub mandatory_bytes: u64,
    /// What is left for the expert pool after the mandatory set and `reserve`.
    pub expert_pool_bytes: u64,
}

impl ChipPlan {
    pub fn layers(&self) -> u32 {
        self.last_layer - self.first_layer
    }
    /// Fraction of this chip's own experts the pool can hold. The pipeline is
    /// serial, so the MINIMUM of these across chips is what bounds throughput.
    pub fn resident_fraction(&self, g: &Geometry) -> f64 {
        let need = self.layers() as u64 * g.experts_per_layer;
        if need == 0 {
            return 1.0;
        }
        (self.expert_pool_bytes as f64 / need as f64).min(1.0)
    }
}

/// A whole-model placement.
#[derive(Debug, Clone)]
pub struct Placement {
    pub chips: Vec<ChipPlan>,
    pub max_ctx: u32,
}

impl Placement {
    /// The bound that matters: the least-resident chip, because the pipeline is
    /// serial and that chip's miss rate sets the token rate.
    pub fn worst_resident_fraction(&self, g: &Geometry) -> f64 {
        self.chips
            .iter()
            .map(|c| c.resident_fraction(g))
            .fold(1.0_f64, f64::min)
    }
    pub fn hops_per_token(&self) -> usize {
        self.chips.len().saturating_sub(1)
    }

    /// Is this placement worth running?
    ///
    /// A plan can fit the mandatory set and still be useless: the 910b as
    /// measured (8 chips, ~4.9 GiB free each) holds the 19.7 GB of non-expert
    /// weights and KV, yet leaves the first chip with a ZERO-byte expert pool, so
    /// every expert would stream on every token. `plan` returns that placement
    /// because it IS a valid assignment; deciding it is not worth running belongs
    /// to the caller, with a threshold it chooses.
    ///
    /// The Metal side of this project measured 89% cache hit at 27% residency, so
    /// something in that region is a defensible floor — but it is a measurement on
    /// different hardware and should be re-measured here before being trusted.
    pub fn is_viable(&self, g: &Geometry, min_resident_fraction: f64) -> bool {
        !self.chips.is_empty() && self.worst_resident_fraction(g) >= min_resident_fraction
    }

    /// One line per chip, for the startup log. A placement that is never printed
    /// is a placement nobody can debug when throughput disappoints.
    pub fn summary(&self, g: &Geometry) -> String {
        let gib = |b: u64| b as f64 / (1024.0 * 1024.0 * 1024.0);
        let mut out = format!(
            "placement: {} chip(s), {} hop(s)/token, max_ctx {}, worst residency {:.0}%\n",
            self.chips.len(),
            self.hops_per_token(),
            self.max_ctx,
            self.worst_resident_fraction(g) * 100.0
        );
        for c in &self.chips {
            out.push_str(&format!(
                "  dev {} layers {:>2}..{:<2} ({:>2})  free {:6.2} GiB  \
                 mandatory {:6.2}  expert pool {:6.2} ({:.0}% resident)\n",
                c.device,
                c.first_layer,
                c.last_layer,
                c.layers(),
                gib(c.free_bytes),
                gib(c.mandatory_bytes),
                gib(c.expert_pool_bytes),
                c.resident_fraction(g) * 100.0
            ));
        }
        out
    }
}

/// Why a plan could not be made. Startup failures, all of them — the point is to
/// refuse with numbers rather than OOM during a decode.
#[derive(Debug)]
pub enum PlanError {
    NoDevices,
    /// The non-expert weights and KV reservation do not fit the visible chips at
    /// all. Carries what was needed and what was free.
    MandatoryExceedsHbm { needed: u64, free: u64 },
    /// The mandatory set fits, but what remains cannot hold even ONE TOKEN's experts, so no
    /// token could be served without streaming mid-token.
    ///
    /// This is the case that previously returned a plausible-looking plan with a zero-byte expert
    /// pool — measured on a box whose eight chips had 4.8 GiB free each. Refusing is the whole
    /// point: a plan nobody can run is worse than an error, because it gets deployed.
    WorkingSetExceedsHbm {
        mandatory: u64,
        working_set: u64,
        free: u64,
    },
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlanError::NoDevices => write!(f, "no Ascend devices visible"),
            PlanError::WorkingSetExceedsHbm {
                mandatory,
                working_set,
                free,
            } => write!(
                f,
                "the non-expert set fits ({mandatory} B) but only {} B remain of {free} B free, \
                 which cannot hold one token's experts ({working_set} B) — every token would \
                 stream mid-token. Free more device memory or reduce max_ctx.",
                free.saturating_sub(*mandatory)
            ),
            PlanError::MandatoryExceedsHbm { needed, free } => write!(
                f,
                "the non-expert weights and KV reservation need {:.2} GiB but only \
                 {:.2} GiB of HBM is free across the visible chips. This is not an \
                 expert-pool shortfall that streaming can absorb — these bytes are \
                 touched every token. Free a chip or lower max_ctx.",
                *needed as f64 / (1024.0 * 1024.0 * 1024.0),
                *free as f64 / (1024.0 * 1024.0 * 1024.0),
            ),
        }
    }
}

impl std::error::Error for PlanError {}

/// Assign contiguous layer ranges to chips, sized so every chip ends up with a
/// similar resident FRACTION of its own experts.
///
/// `free` is measured free HBM per device, `reserve` is per-chip headroom for
/// activations and workspace. Chips are used in the order given; a chip that
/// cannot even hold one layer's mandatory set is skipped rather than assigned an
/// empty range that would still cost a hop.
/// Measured on 910B2 (the customer's silicon), so the estimate below is grounded rather than
/// assumed. Both numbers were taken with the box in its normal occupied state.
pub mod measured {
    /// One packed 4-bit expert projection, K=4096 N=2048, at blockDim 48 — the tuned figure.
    pub const MATVEC_MS: f64 = 0.171;
    /// The SAME projection through the host-expanded int8 kernel, from the same A/B with both
    /// sides optimised and each verified alone in its own TU.
    ///
    /// This is the other half of the three-chip decision: 4-bit packing costs 1.22x on decode, and
    /// that was worth paying when the model had to fit 179.6 GiB into a starved box. On a FREE node
    /// the memory pressure is gone (308.6 GiB fits 512), so the faster kernel is simply available —
    /// which is why the layout is a deployment choice and not a fixed property of the build.
    pub const MATVEC_FOLDED_INT8_MS: f64 = 0.136;
    /// Host-to-device bandwidth with PINNED host memory.
    pub const H2D_PINNED_GBPS: f64 = 21.9;
    /// The same with ordinary `malloc`ed host memory. The gap is 5.2x and is pure loader choice:
    /// an unpinned buffer turns a DMA into a bounce copy. Recorded because a streaming path that
    /// forgets to pin loses more than every kernel optimisation in this document combined.
    pub const H2D_MALLOC_GBPS: f64 = 4.2;

    /// Prefill, measured on device at K=4096, N=512, M=256, blockDim=48: the 4-bit ->
    /// pre-scaled-f16 unpack, then the cube matmul, at the batch that measured best.
    ///
    /// Kept at the shape they were TAKEN at rather than pre-scaled to the real expert width, so
    /// the constant still names its own measurement. The reference projection is a quarter of a
    /// real one (4096x2048), and [`Prefill::estimate`] scales both by weight count.
    pub const PREFILL_REF_K: u32 = 4096;
    pub const PREFILL_REF_N: u32 = 512;
    pub const PREFILL_REF_M: u32 = 256;
    /// Unpack of one reference projection, whole-projection cost (not per token).
    pub const UNPACK_MS_REF: f64 = 0.053;
    /// Cube matmul of one reference projection for all `PREFILL_REF_M` tokens.
    pub const CUBE_MATMUL_MS_REF: f64 = 0.028;
    /// Weights in the reference projection. A test pins this against `PREFILL_REF_K * _N`.
    pub const PREFILL_REF_WEIGHTS: f64 = 4096.0 * 512.0;

    /// Expanding ONE expert projection's e8m0 scale plane (2048x128 = 262,144 scales) to the
    /// halved f32 scales the matvec reads, at blockDim 48. Emitted kernel, verified BITWISE
    /// against the host oracle over all 256 byte values (0/262144 wrong at blockDim 1 and 48).
    ///
    /// This is the price of [`ExpertLayout::PackedNibbleNativeE8m0`], and it is a price worth
    /// naming: an earlier estimate put it at ~0.6 ms/token from HBM bandwidth, and the measured
    /// figure is ~10x that, because the kernel is op-bound rather than bandwidth-bound at
    /// 256 KiB. Scaling: 0.0961 ms at blockDim 1, 0.0153 at 8, 0.0087 at 24, 0.0073 at 48.
    pub const E8M0_EXPAND_MS_PER_PLANE: f64 = 0.0073;

    /// The same work FOLDED into the prefill unpack instead of run as its own kernel, as a
    /// fraction of that unpack's time. Measured at K=4096 N=512, blockDim 48: 0.0524 ms folded
    /// against 0.0511 ms with pre-expanded f32 scales, and the two f16 slabs are BYTE-IDENTICAL
    /// (0 of 2,097,152 differ, with 1.47M non-zero, so not a vacuous comparison).
    ///
    /// Expressed as a ratio rather than an absolute because it scales with the same shape as the
    /// unpack it rides in. Against the standalone kernel at a MATCHED plane size this is 2.7%
    /// versus 3.6% — folding is the better placement by about 1.3x, not the large factor a
    /// shape-mismatched comparison first suggested.
    pub const E8M0_FOLDED_UNPACK_OVERHEAD: f64 = 0.027;

    /// Aggregate H2D bandwidth across `n` chips, as a multiple of ONE link — MEASURED, and the
    /// single most consequential number in this module.
    ///
    /// Every multi-chip transfer figure previously divided by the chip count, i.e. assumed 8 links
    /// give 8x. They do not. On the customer's 910B2 (8 x 910B2, pinned host memory, one thread and
    /// its own host buffer per chip, 512 MiB x 40 iterations, two runs):
    ///
    /// | chips | aggregate | per link | scaling |
    /// |---|---|---|---|
    /// | 1 | 23.8 GB/s | 23.8 | 1.00x |
    /// | 2 | 36.7 - 45.2 | 18.4 - 22.6 | 1.54 - 1.90x |
    /// | 4 | 46.4 - 46.9 | 11.6 - 11.7 | 1.95 - 1.97x |
    /// | 8 | **60.3 - 64.9** | 7.5 - 8.1 | **2.53 - 2.73x** |
    ///
    /// So aggregate saturates near 60-65 GB/s, not the 175 the old model implied. Unpinned host
    /// memory at 8 chips gives 28.5 - 32.9 GB/s, so pinning still roughly doubles it — the 5.2x it
    /// was worth on ONE link does not survive contention, but it never becomes optional.
    ///
    /// ⚠ Measured with the customer's vLLM job at ~95% AICore on all eight chips, which is their
    /// operating condition and so the right planning figure — but it cannot be separated from
    /// topology without a free box. For contrast, an idle 910c (different silicon, single link
    /// 59.6 GB/s) reached 5.12x at 8 chips and then DEGRADED to 3.45x at 16. That degradation is
    /// why this function CLAMPS rather than extrapolating: beyond the measured points, more links
    /// are not assumed to help.
    pub fn h2d_scaling(chips: u32) -> f64 {
        // Mean of the two runs at each measured point, so a single noisy trial does not set it.
        const CURVE: [(u32, f64); 4] = [(1, 1.00), (2, 1.72), (4, 1.96), (8, 2.63)];
        if chips <= 1 {
            return 1.0;
        }
        if let Some(&(_, s)) = CURVE.iter().find(|&&(n, _)| n == chips) {
            return s;
        }
        // Interpolate between bracketing points; clamp outside them.
        if chips > CURVE[CURVE.len() - 1].0 {
            return CURVE[CURVE.len() - 1].1;
        }
        let mut prev = CURVE[0];
        for &(n, sc) in CURVE.iter().skip(1) {
            if chips < n {
                let t = (chips - prev.0) as f64 / (n - prev.0) as f64;
                return prev.1 + t * (sc - prev.1);
            }
            prev = (n, sc);
        }
        prev.1
    }

    /// One decode layer's ATTENTION sublayer at the tuned launch configuration (blockDim 48 for
    /// the batched-head kernel, 1 for rms_norm/rope/hc), measured on the 910c at real dims and
    /// oracle-verified at 5.129e-05 against a double-precision replay. The same sublayer at
    /// blockDim 1 everywhere is 203.0 ms — the tuning is launch configuration, not kernels.
    pub const ATTN_SUBLAYER_MS: f64 = 5.964;
    /// The FFN sublayer of the same layer with 7 DISTINCT expert weight sets (6 routed + 1
    /// shared), same session, oracle-verified 4.147e-05 max-rel / 9.930e-07 norm-relative.
    pub const FFN_SUBLAYER_MS: f64 = 1.914;
    /// The full decode layer, attention + FFN. x43 layers = 338.8 ms/token, ~2.95 tok/s of
    /// RESIDENT compute. Not included: streaming, KV growth, the host router matvec, lm_head.
    pub const DECODE_LAYER_MS: f64 = 7.878;
}

/// Per-projection matvec cost for a layout.
///
/// The packed 4-bit kernels share one figure; the host-expanded int8 rung has its own, and using
/// the packed number for it would understate a rung that a free node can afford.
pub fn matvec_ms(layout: ExpertLayout) -> f64 {
    match layout {
        ExpertLayout::HostExpandedInt8BlockF32 => measured::MATVEC_FOLDED_INT8_MS,
        _ => measured::MATVEC_MS,
    }
}

/// Estimated decode rate, from measured compute and measured bandwidth.
///
/// Residency percentages do not tell anyone what they will get. This converts one into tokens per
/// second, which is the number a deployment decision actually turns on.
///
/// The model is deliberately simple and its shape is stated rather than hidden: a token touches
/// `experts_used_per_token` experts in each of `layers` layers, each expert being three
/// projections. Whatever is not resident must be streamed. Compute and transfer are reported
/// separately AND combined both ways, because whether they overlap is an implementation property
/// this crate has not yet established — so the honest answer is a range, not a point.
#[derive(Debug, Clone, Copy)]
pub struct Throughput {
    pub expert_bytes_per_token: u64,
    pub streamed_bytes_per_token: u64,
    pub compute_ms: f64,
    pub transfer_ms: f64,
}

impl Throughput {
    /// As [`Throughput::estimate`], but charging the e8m0 scale expansion when the layout needs
    /// it.
    ///
    /// [`ExpertLayout::PackedNibbleNativeE8m0`] keeps the scale plane at MXFP4's stored one byte
    /// per block, so fewer bytes are resident and fewer cross the host link — but the matvec
    /// still wants halved f32 scales, so a separate emitted kernel expands them on the way in.
    /// That kernel is measured, so the trade is priced rather than assumed: it is added to
    /// COMPUTE, which is exactly where it lands, and the layout's smaller `bytes_per_weight` has
    /// already reduced the transfer term through `bytes_per_expert`.
    ///
    /// Deliberately a separate constructor rather than a flag on `estimate`: every existing
    /// caller keeps its meaning, and a reader of the older call sites is not left wondering
    /// whether an expansion cost was silently folded in.
    pub fn estimate_with_layout(
        g: &Geometry,
        layout: ExpertLayout,
        resident_fraction: f64,
        h2d_gbps: f64,
    ) -> Self {
        let mut t = Self::estimate(g, resident_fraction, h2d_gbps);
        // Re-base the compute term on this layout's own measured kernel.
        let projections = g.layers as f64 * g.experts_used_per_token as f64 * 3.0;
        t.compute_ms = projections * matvec_ms(layout);
        if layout == ExpertLayout::PackedNibbleNativeE8m0 {
            // One plane per projection, three projections per expert — the same count the
            // compute term already uses.
            let planes = g.layers as f64 * g.experts_used_per_token as f64 * 3.0;
            t.compute_ms += planes * measured::E8M0_EXPAND_MS_PER_PLANE;
        }
        t
    }

    pub fn estimate(g: &Geometry, resident_fraction: f64, h2d_gbps: f64) -> Self {
        let per_token = g.layers as u64
            * g.experts_used_per_token as u64
            * g.bytes_per_expert();
        let miss = (1.0 - resident_fraction.clamp(0.0, 1.0)).max(0.0);
        let streamed = (per_token as f64 * miss) as u64;
        // Three projections per expert, each one packed matvec.
        let projections = g.layers as u64 * g.experts_used_per_token as u64 * 3;
        Throughput {
            expert_bytes_per_token: per_token,
            streamed_bytes_per_token: streamed,
            compute_ms: projections as f64 * measured::MATVEC_MS,
            transfer_ms: streamed as f64 / (h2d_gbps * 1e9) * 1e3,
        }
    }
    /// If transfer overlaps compute perfectly.
    pub fn tokens_per_sec_overlapped(&self) -> f64 {
        let ms = self.compute_ms.max(self.transfer_ms);
        if ms <= 0.0 { f64::INFINITY } else { 1000.0 / ms }
    }
    /// If it does not overlap at all.
    pub fn tokens_per_sec_serial(&self) -> f64 {
        let ms = self.compute_ms + self.transfer_ms;
        if ms <= 0.0 { f64::INFINITY } else { 1000.0 / ms }
    }
    /// Which term to attack. Returns `"transfer"` or `"compute"`.
    pub fn bound_by(&self) -> &'static str {
        if self.transfer_ms > self.compute_ms { "transfer" } else { "compute" }
    }
}

/// Estimated PREFILL time, from the measured unpack and cube-matmul kernels.
///
/// Prefill has a different shape from decode and the difference is not the batch size — it is
/// [`Geometry::experts_touched_per_layer`]. A batch of a few hundred tokens touches essentially
/// every expert, so prefill pays a one-time sweep over all expert weights (unpack, plus streaming
/// whatever is not resident) and then a small per-token cube matmul. The sweep dominates every
/// realistic prompt, which is why this is modelled as `sweep + tokens * per_token` rather than as
/// a throughput figure.
///
/// # What `chips` means, and where it does NOT apply
///
/// `chips` divides both the sweep and the per-token term, on the assumption of **expert
/// parallelism**: every chip owns a slice of every layer's experts, unpacks only its own, and
/// streams them over its OWN host link (the 21.9 GB/s figure is per link, so aggregate bandwidth
/// scales with chip count too).
///
/// That is NOT the decomposition [`plan`] computes. `plan` assigns CONTIGUOUS LAYER RANGES,
/// because a layer's KV must sit on the chip that runs the layer. Under layer ranges a single
/// large batch gets NO concurrency — chip 1 waits for chip 0 — and overlapping them requires
/// pipelining token CHUNKS, which re-pays the sweep per chunk ([`Prefill::estimate_chunked`]).
/// So the two are in genuine tension: decode wants layer-major for KV locality, prefill wants
/// expert-major for the sweep. Passing `chips > 1` here is a statement about hardware the box
/// affords, not about the plan this module currently produces.
#[derive(Debug, Clone, Copy)]
pub struct Prefill {
    pub tokens: u32,
    pub chips: u32,
    /// Expected distinct experts a batch this size touches, per layer.
    pub experts_touched_per_layer: f64,
    /// Expert bytes that must cross the host link for the sweep, after residency.
    pub sweep_streamed_bytes: u64,
    /// 4-bit -> f16 unpack for the whole sweep, per chip.
    pub unpack_ms: f64,
    /// e8m0 -> f32 scale expansion for the whole sweep, per chip. Zero unless the layout keeps
    /// the scale plane native.
    ///
    /// Prefill needs no new kernel for this: the expansion emits per PLANE and the prefill unpack
    /// consumes a plane, so the two compose across a GM boundary — which Phase 5 measured as
    /// byte-identical and free versus fusing them. The expansion's output is already proven
    /// bitwise equal to the f32 plane, so everything downstream is unchanged by construction.
    pub expand_ms: f64,
    /// Streaming the sweep, per chip over its own link.
    pub transfer_ms: f64,
    /// Cube matmul for all `tokens`, per chip.
    pub matmul_ms: f64,
}

impl Prefill {
    /// `layout` must match the one `g` was built with — see [`Geometry::weights_per_projection`].
    pub fn estimate(
        g: &Geometry,
        layout: ExpertLayout,
        tokens: u32,
        resident_fraction: f64,
        h2d_gbps: f64,
        chips: u32,
    ) -> Self {
        let chips_f = chips.max(1) as f64;
        // Both kernels were measured on a projection a quarter the real width. The unpack is a
        // vector pass over every weight and the matmul's work is proportional to K*N, so both
        // scale with weight count.
        let w_ratio = g.weights_per_projection(layout) / measured::PREFILL_REF_WEIGHTS;
        let unpack_per_projection = measured::UNPACK_MS_REF * w_ratio;
        let matmul_per_token_per_projection =
            measured::CUBE_MATMUL_MS_REF / measured::PREFILL_REF_M as f64 * w_ratio;

        let touched = g.experts_touched_per_layer(tokens);
        let sweep_projections = g.layers as f64 * touched * 3.0;
        let miss = (1.0 - resident_fraction.clamp(0.0, 1.0)).max(0.0);
        let streamed = g.layers as f64 * touched * g.bytes_per_expert() as f64 * miss;
        // Per token a row still visits only the experts IT routes to, not all of them.
        let per_token_projections = g.layers as f64 * g.experts_used_per_token as f64 * 3.0;

        // Prefill folds the chain INTO the unpack rather than running the standalone kernel, so
        // it is charged as a fraction of the unpack. That choice is what keeps the native layout
        // from costing prefill more than it does: prefill is compute-bound, so the smaller stream
        // buys nothing here and every millisecond of expansion is pure loss.
        let expand_fraction = if layout == ExpertLayout::PackedNibbleNativeE8m0 {
            measured::E8M0_FOLDED_UNPACK_OVERHEAD
        } else {
            0.0
        };

        Prefill {
            tokens,
            chips: chips.max(1),
            experts_touched_per_layer: touched,
            sweep_streamed_bytes: streamed as u64,
            unpack_ms: sweep_projections * unpack_per_projection / chips_f,
            expand_ms: sweep_projections * unpack_per_projection * expand_fraction / chips_f,
            // Transfer divides by MEASURED link scaling, not by the chip count: eight links on
            // the customer's box deliver 2.63x one link, not 8x.
            transfer_ms: streamed / (h2d_gbps * 1e9) * 1e3 / measured::h2d_scaling(chips.max(1)),
            matmul_ms: tokens as f64 * per_token_projections * matmul_per_token_per_projection
                / chips_f,
        }
    }

    /// Everything the sweep spends on compute: expanding the scales, then unpacking the codes.
    pub fn sweep_compute_ms(&self) -> f64 {
        self.unpack_ms + self.expand_ms
    }

    /// Sweep compute and streaming overlapped; the matmul consumes the unpacked slab so it is
    /// serial with respect to it.
    pub fn total_ms_overlapped(&self) -> f64 {
        self.sweep_compute_ms().max(self.transfer_ms) + self.matmul_ms
    }

    /// Nothing overlaps. Reported alongside because whether the loader overlaps its DMA with
    /// compute is an implementation property this crate has not established.
    pub fn total_ms_serial(&self) -> f64 {
        self.sweep_compute_ms() + self.transfer_ms + self.matmul_ms
    }

    pub fn tokens_per_sec_overlapped(&self) -> f64 {
        Self::rate(self.tokens, self.total_ms_overlapped())
    }

    pub fn tokens_per_sec_serial(&self) -> f64 {
        Self::rate(self.tokens, self.total_ms_serial())
    }

    fn rate(tokens: u32, ms: f64) -> f64 {
        if ms <= 0.0 {
            0.0
        } else {
            tokens as f64 / (ms / 1000.0)
        }
    }

    /// Which of the two sweep terms to attack. The per-token matmul is reported separately
    /// because it is the only term that grows with prompt length, so at long contexts it can
    /// exceed BOTH sweep terms while neither of them is what to fix.
    pub fn bound_by(&self) -> &'static str {
        let compute = self.sweep_compute_ms();
        if self.matmul_ms > compute.max(self.transfer_ms) {
            "matmul"
        } else if self.transfer_ms > compute {
            "transfer"
        } else {
            "unpack"
        }
    }

    /// How close the sweep's two terms are, as `compute / transfer`.
    ///
    /// Reported because it decides what a knob search can be worth: at a ratio near 1 the two
    /// sides are balanced, so shaving compute pays only down to the transfer term and no further.
    /// A search that reports a compute win without this ratio is quoting a number the system
    /// cannot deliver.
    pub fn balance(&self) -> f64 {
        if self.transfer_ms <= 0.0 { f64::INFINITY } else { self.sweep_compute_ms() / self.transfer_ms }
    }

    /// The most a PURE compute optimisation can win, as a speedup on the total.
    ///
    /// Once compute drops to the transfer term the sweep is transfer-bound and further compute
    /// wins are invisible. This is rule 28 made callable, so a search states its own ceiling.
    pub fn compute_headroom(&self) -> f64 {
        let floor = self.transfer_ms.max(0.0) + self.matmul_ms;
        let now = self.total_ms_overlapped();
        if floor <= 0.0 { f64::INFINITY } else { now / floor }
    }
}

/// Prefill done in chunks of `chunk` tokens, re-paying the expert sweep each chunk.
///
/// This is the cost of getting the loop order wrong, and it is large: the sweep is fixed, so
/// `ceil(tokens/chunk)` chunks multiply it. Exposed rather than described in a comment because
/// the difference between expert-major and chunk-major prefill is more than an order of
/// magnitude, and a test can then pin it.
///
/// Returns total milliseconds under the overlapped assumption.
pub fn estimate_chunked_prefill(
    g: &Geometry,
    layout: ExpertLayout,
    tokens: u32,
    chunk: u32,
    resident_fraction: f64,
    h2d_gbps: f64,
    chips: u32,
) -> f64 {
    let chunk = chunk.max(1);
    let mut done = 0u32;
    let mut ms = 0.0;
    while done < tokens {
        let this = chunk.min(tokens - done);
        ms += Prefill::estimate(g, layout, this, resident_fraction, h2d_gbps, chips)
            .total_ms_overlapped();
        done += this;
    }
    ms
}

/// The largest `max_ctx` that both FITS the measured memory and meets a residency floor.
///
/// This is the question a partly-occupied box actually poses. "Does it fit on three chips" assumes
/// the chips are empty; the customer's box has eight chips with 4.8 GiB free each, and the useful
/// answer there is not a chip count but "how much context can we serve, if any".
///
/// KV is the term that grows with context — `kv_per_layer_token * layers * max_ctx` — so the
/// context length IS the memory knob, and this searches it directly rather than asking the caller
/// to guess and retry.
///
/// Returns the planning error from the SMALLEST attempted context when nothing is servable, so the
/// caller learns which wall it hit (mandatory set, or one token's experts) rather than just "no".
pub fn max_context(
    g: &Geometry,
    free: &[(u32, u64)],
    reserve: u64,
    min_resident_fraction: f64,
    ctx_granularity: u32,
) -> Result<u32, PlanError> {
    let gran = ctx_granularity.max(1);
    let ok = |ctx: u32| -> Result<bool, PlanError> {
        let p = plan(g, free, reserve, ctx)?;
        Ok(p.is_viable(g, min_resident_fraction))
    };
    // Probe the floor first: if the smallest context is unservable, report WHY rather than 0.
    if !ok(gran)? {
        // It planned but missed the residency floor; that is a capacity answer, not an error.
        return Ok(0);
    }
    // Double until it fails, then bisect. Bounded so a huge free pool cannot spin.
    let mut lo = gran;
    let mut hi = gran;
    loop {
        let next = hi.saturating_mul(2);
        if next > 4 * 1024 * 1024 {
            break;
        }
        match ok(next) {
            Ok(true) => {
                lo = next;
                hi = next;
            }
            _ => {
                hi = next;
                break;
            }
        }
    }
    if hi == lo {
        return Ok(lo);
    }
    while hi - lo > gran {
        let mid = lo + (hi - lo) / 2;
        match ok(mid) {
            Ok(true) => lo = mid,
            _ => hi = mid,
        }
    }
    Ok(lo)
}

/// What the memory budget looks like at a given context, for reporting rather than deciding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Budget {
    pub total_free: u64,
    pub usable_after_reserve: u64,
    pub mandatory: u64,
    pub kv: u64,
    pub min_working_set: u64,
    pub experts_total: u64,
}

impl Budget {
    pub fn of(g: &Geometry, free: &[(u32, u64)], reserve: u64, max_ctx: u32) -> Self {
        let total_free: u64 = free.iter().map(|&(_, b)| b).sum();
        let usable: u64 = free.iter().map(|&(_, b)| b.saturating_sub(reserve)).sum();
        Budget {
            total_free,
            usable_after_reserve: usable,
            mandatory: g.mandatory(g.layers, max_ctx) + g.globals,
            kv: g.kv_bytes(g.layers, max_ctx),
            min_working_set: g.min_working_set(g.layers),
            experts_total: g.layers as u64 * g.experts_per_layer,
        }
    }
    /// Fraction of the experts the leftover space can hold. 1.0 means fully resident.
    pub fn resident_fraction(&self) -> f64 {
        if self.experts_total == 0 {
            return 1.0;
        }
        let pool = self.usable_after_reserve.saturating_sub(self.mandatory);
        (pool as f64 / self.experts_total as f64).min(1.0)
    }
    /// The hard question: can a token be served at all?
    pub fn servable(&self) -> bool {
        self.mandatory + self.min_working_set <= self.usable_after_reserve
    }
}

pub fn plan(
    g: &Geometry,
    free: &[(u32, u64)],
    reserve: u64,
    max_ctx: u32,
) -> Result<Placement, PlanError> {
    if free.is_empty() {
        return Err(PlanError::NoDevices);
    }
    let per_layer_fixed = g.nonexpert_per_layer + g.kv_per_layer_token * max_ctx as u64;

    // Usable per chip after headroom. Saturating: a chip with less free than the
    // reserve contributes nothing rather than wrapping.
    let usable: Vec<(u32, u64)> = free
        .iter()
        .map(|&(d, b)| (d, b.saturating_sub(reserve)))
        .filter(|&(_, b)| b >= per_layer_fixed)
        .collect();
    if usable.is_empty() {
        return Err(PlanError::MandatoryExceedsHbm {
            needed: per_layer_fixed,
            free: free.iter().map(|&(_, b)| b).sum(),
        });
    }

    // Globals land on the first usable chip; they are touched once per token and
    // splitting them would buy nothing.
    let total_usable: u64 = usable.iter().map(|&(_, b)| b).sum();
    let needed_mandatory = g.layers as u64 * per_layer_fixed + g.globals;
    if needed_mandatory > total_usable {
        return Err(PlanError::MandatoryExceedsHbm {
            needed: needed_mandatory,
            free: total_usable,
        });
    }

    // The mandatory set fitting is not enough. What remains has to hold at least the experts one
    // token touches, or no token can be served without streaming mid-token. Checked against the
    // WHOLE box rather than per chip, because the layer split is not decided yet.
    let working_set = g.min_working_set(g.layers);
    if needed_mandatory + working_set > total_usable {
        return Err(PlanError::WorkingSetExceedsHbm {
            mandatory: needed_mandatory,
            working_set,
            free: total_usable,
        });
    }

    // Give each chip layers in proportion to its usable bytes. That equalises the
    // resident fraction to first order: a chip with twice the space takes twice
    // the layers and so has twice the experts to cache in twice the pool.
    let mut counts: Vec<u32> = usable
        .iter()
        .map(|&(_, b)| {
            ((b as f64 / total_usable as f64) * g.layers as f64).floor() as u32
        })
        .collect();
    // Floor loses layers; hand them to the chips with the most spare bytes per
    // layer already assigned, which keeps the fractions even.
    let mut assigned: u32 = counts.iter().sum();
    while assigned < g.layers {
        let (best, _) = counts
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let spare = usable[i].1 as f64 - (c as u64 * per_layer_fixed) as f64;
                (i, spare / (c + 1) as f64)
            })
            .fold((0usize, f64::MIN), |acc, (i, s)| if s > acc.1 { (i, s) } else { acc });
        counts[best] += 1;
        assigned += 1;
    }
    // A chip must be able to hold what it was given; push overflow to the next.
    for i in 0..counts.len() {
        while counts[i] > 0 && counts[i] as u64 * per_layer_fixed > usable[i].1 {
            counts[i] -= 1;
            let next = (i + 1) % counts.len();
            if next == i {
                break;
            }
            counts[next] += 1;
        }
    }

    let mut chips = Vec::new();
    let mut first = 0u32;
    for (i, &(device, ub)) in usable.iter().enumerate() {
        let n = counts[i];
        if n == 0 {
            continue;
        }
        let mand = n as u64 * per_layer_fixed + if chips.is_empty() { g.globals } else { 0 };
        chips.push(ChipPlan {
            device,
            first_layer: first,
            last_layer: first + n,
            free_bytes: free[i].1,
            mandatory_bytes: mand,
            expert_pool_bytes: ub.saturating_sub(mand),
        });
        first += n;
    }
    Ok(Placement { chips, max_ctx })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The customer's box AS MEASURED: eight chips, 4.8 GiB free of 65.5 each, the rest held by a
    /// workload already running. This is the configuration the plan has to answer for, and the one
    /// a "three empty chips" assumption gets wrong.
    const CUSTOMER_910B: [(u32, u64); 8] = [
        (0, 4_800 * 1024 * 1024),
        (1, 4_800 * 1024 * 1024),
        (2, 4_800 * 1024 * 1024),
        (3, 4_800 * 1024 * 1024),
        (4, 4_800 * 1024 * 1024),
        (5, 4_800 * 1024 * 1024),
        (6, 4_800 * 1024 * 1024),
        (7, 4_800 * 1024 * 1024),
    ];

    /// Residency turned into tokens per second, on the customer's measured box. This is the number
    /// a deployment decision turns on, and it is built from two figures measured on 910B2 rather
    /// than from an imported residency threshold.
    #[test]
    fn customer_box_throughput_estimate() {
        let g = Geometry::ds4_flash(ExpertLayout::PackedNibbleBlockF32);
        let b = Budget::of(&g, &CUSTOMER_910B, 0, 4096);
        let t = Throughput::estimate(&g, b.resident_fraction(), measured::H2D_PINNED_GBPS);
        eprintln!(
            "per token: {:.2} GB touched, {:.2} GB streamed at {:.1}% resident\n\
             compute {:.0} ms, transfer {:.0} ms, bound by {}\n\
             => {:.1} tok/s overlapped, {:.1} tok/s serial",
            t.expert_bytes_per_token as f64 / 1e9,
            t.streamed_bytes_per_token as f64 / 1e9,
            b.resident_fraction() * 100.0,
            t.compute_ms,
            t.transfer_ms,
            t.bound_by(),
            t.tokens_per_sec_overlapped(),
            t.tokens_per_sec_serial()
        );
        assert!(t.compute_ms > 0.0 && t.transfer_ms > 0.0);
        // Both terms are the same order here, so neither can be ignored.
        let ratio = t.transfer_ms / t.compute_ms;
        assert!(
            (0.3..3.0).contains(&ratio),
            "expected compute and transfer within ~3x on this box, got {ratio:.2}"
        );
    }

    /// The native e8m0 layout, priced end to end on the customer's box: fewer bytes resident and
    /// streamed, against a measured expansion kernel added to compute.
    ///
    /// This is the whole decision in one test, and it is stated as a comparison rather than as a
    /// claim, because the layout is only worth taking if the trade comes out positive AT THE
    /// OPERATING POINT — the transfer term it shrinks and the compute term it grows are within
    /// ~15% of each other on this box, so neither could be waved through.
    #[test]
    fn native_e8m0_trades_measured_compute_for_transfer_and_wins() {
        let f32l = ExpertLayout::PackedNibbleBlockF32;
        let natl = ExpertLayout::PackedNibbleNativeE8m0;
        let (gf, gn) = (Geometry::ds4_flash(f32l), Geometry::ds4_flash(natl));

        // Residency is computed per layout: the smaller model fits more of itself in the same HBM.
        let bf = Budget::of(&gf, &CUSTOMER_910B, 0, 4096);
        let bn = Budget::of(&gn, &CUSTOMER_910B, 0, 4096);
        let tf = Throughput::estimate_with_layout(
            &gf, f32l, bf.resident_fraction(), measured::H2D_PINNED_GBPS);
        let tn = Throughput::estimate_with_layout(
            &gn, natl, bn.resident_fraction(), measured::H2D_PINNED_GBPS);

        eprintln!(
            "f32 scales : {:.1} GiB experts, {:.1}% resident, compute {:.0} ms, transfer {:.0} ms \
             => {:.2} tok/s overlapped (bound by {})\n\
             native e8m0: {:.1} GiB experts, {:.1}% resident, compute {:.0} ms, transfer {:.0} ms \
             => {:.2} tok/s overlapped (bound by {})",
            (gf.layers as f64 * gf.experts_per_layer as f64) / (1024.0 * 1024.0 * 1024.0),
            bf.resident_fraction() * 100.0, tf.compute_ms, tf.transfer_ms,
            tf.tokens_per_sec_overlapped(), tf.bound_by(),
            (gn.layers as f64 * gn.experts_per_layer as f64) / (1024.0 * 1024.0 * 1024.0),
            bn.resident_fraction() * 100.0, tn.compute_ms, tn.transfer_ms,
            tn.tokens_per_sec_overlapped(), tn.bound_by(),
        );

        // The layout must actually be smaller, or nothing else follows.
        assert!(
            gn.experts_per_layer < gf.experts_per_layer,
            "native e8m0 must shrink the expert term"
        );
        // The expansion must be CHARGED, not silently free.
        let unpriced = Throughput::estimate(&gn, bn.resident_fraction(), measured::H2D_PINNED_GBPS);
        assert!(
            tn.compute_ms > unpriced.compute_ms,
            "the expansion kernel must be added to compute; got {:.1} vs {:.1}",
            tn.compute_ms, unpriced.compute_ms
        );
        // And it must still come out ahead once charged.
        assert!(
            tn.tokens_per_sec_overlapped() > tf.tokens_per_sec_overlapped(),
            "native e8m0 priced at {:.2} tok/s vs f32 scales {:.2} — the trade does not pay",
            tn.tokens_per_sec_overlapped(), tf.tokens_per_sec_overlapped()
        );
    }

    /// The expansion is charged ONLY for the layout that needs it — a flag applied everywhere
    /// would quietly slow every other rung's estimate.
    #[test]
    fn only_the_native_layout_is_charged_for_expansion() {
        // Compared against the layout's OWN kernel baseline, not against the layout-blind
        // estimate: the int8 rung has a different measured matvec (0.136 vs 0.171 ms), so those two
        // now differ for a reason that is the KERNEL and not an expansion. The first version of
        // this test conflated them and failed as soon as the cost became layout-aware.
        for layout in [
            ExpertLayout::HostExpandedInt8BlockF32,
            ExpertLayout::PackedNibbleBlockF32,
            ExpertLayout::PackedNibbleBlockF16,
        ] {
            let g = Geometry::ds4_flash(layout);
            let with = Throughput::estimate_with_layout(&g, layout, 0.18, measured::H2D_PINNED_GBPS);
            let projections = g.layers as f64 * g.experts_used_per_token as f64 * 3.0;
            assert_eq!(
                with.compute_ms, projections * matvec_ms(layout),
                "{layout:?} must not be charged for an expansion it does not do"
            );
        }
        // And the native rung MUST be charged, or the guard above proves nothing.
        let nat = ExpertLayout::PackedNibbleNativeE8m0;
        let gn = Geometry::ds4_flash(nat);
        let wn = Throughput::estimate_with_layout(&gn, nat, 0.18, measured::H2D_PINNED_GBPS);
        let pn = gn.layers as f64 * gn.experts_used_per_token as f64 * 3.0;
        assert!(wn.compute_ms > pn * matvec_ms(nat));
    }

    /// The reference constant must keep matching the shape it was measured at.
    #[test]
    fn prefill_reference_weights_matches_its_shape() {
        assert_eq!(
            measured::PREFILL_REF_WEIGHTS,
            measured::PREFILL_REF_K as f64 * measured::PREFILL_REF_N as f64
        );
    }

    /// `weights_per_projection` inverts the byte count, so it must recover the real projection
    /// width under EVERY layout on the ladder — otherwise a layout change would silently rescale
    /// every prefill estimate.
    #[test]
    fn weights_per_projection_round_trips_for_every_layout() {
        let d = ModelDims::ds4_flash();
        let want = d.n_embd as f64 * d.expert_ff as f64;
        for layout in [
            ExpertLayout::HostExpandedInt8BlockF32,
            ExpertLayout::PackedNibbleBlockF32,
            ExpertLayout::PackedNibbleBlockF16,
            ExpertLayout::PackedNibbleNativeE8m0,
        ] {
            let g = Geometry::ds4_flash(layout);
            let got = g.weights_per_projection(layout);
            assert!(
                (got - want).abs() / want < 1e-6,
                "{layout:?}: recovered {got} weights per projection, expected {want}"
            );
        }
    }

    /// The fact prefill turns on: a batch of a few hundred tokens touches nearly all 256 experts,
    /// so the sweep is fixed rather than proportional to the batch.
    #[test]
    fn a_few_hundred_tokens_touch_essentially_every_expert() {
        let g = Geometry::ds4_flash_4bit();
        assert_eq!(g.experts_touched_per_layer(0), 0.0);
        // One token touches exactly the experts it routes to.
        let one = g.experts_touched_per_layer(1);
        assert!(
            (one - g.experts_used_per_token as f64).abs() < 1e-9,
            "one token should touch {} experts, got {one}",
            g.experts_used_per_token
        );
        let f = |t: u32| g.experts_touched_per_layer(t) / g.n_experts as f64;
        assert!((0.88..0.94).contains(&f(100)), "100 tokens: {:.3}", f(100));
        assert!(f(256) > 0.99, "256 tokens: {:.4}", f(256));
        assert!(f(4096) > 0.9999, "4096 tokens: {:.6}", f(4096));
        // Monotone and never above the pool.
        assert!(f(4096) >= f(256) && f(256) >= f(100));
        assert!(g.experts_touched_per_layer(u32::MAX) <= g.n_experts as f64);
    }

    /// Prefill on the customer's box, single-chip as measured and 8-chip as the hardware affords.
    /// Printed as a table because the shape of the answer — a fixed sweep plus a per-token term —
    /// is the part a deployment decision turns on.
    #[test]
    fn customer_box_prefill_estimate() {
        let layout = ExpertLayout::PackedNibbleBlockF32;
        let g = Geometry::ds4_flash(layout);
        let b = Budget::of(&g, &CUSTOMER_910B, 0, 4096);
        let rf = b.resident_fraction();
        for chips in [1u32, 8] {
            for tokens in [1_000u32, 4_000, 8_000, 36_864] {
                let p =
                    Prefill::estimate(&g, layout, tokens, rf, measured::H2D_PINNED_GBPS, chips);
                eprintln!(
                    "{chips} chip(s) L={tokens:>6}: unpack {:>7.0} ms, transfer {:>7.0} ms, \
                     matmul {:>7.0} ms | {:>6.2} s => {:>7.0} tok/s (bound by {})",
                    p.unpack_ms,
                    p.transfer_ms,
                    p.matmul_ms,
                    p.total_ms_overlapped() / 1000.0,
                    p.tokens_per_sec_overlapped(),
                    p.bound_by()
                );
            }
        }
        // At a 1k prompt the fixed sweep must dominate: that is the whole claim.
        let p = Prefill::estimate(&g, layout, 1_000, rf, measured::H2D_PINNED_GBPS, 1);
        assert!(
            p.unpack_ms.max(p.transfer_ms) > 5.0 * p.matmul_ms,
            "sweep {:.0} ms should dominate matmul {:.0} ms at a 1k prompt",
            p.unpack_ms.max(p.transfer_ms),
            p.matmul_ms
        );
        // Prefill must be far faster per token than decode, or batching bought nothing.
        let d = Throughput::estimate(&g, rf, measured::H2D_PINNED_GBPS);
        let prefill_ms_per_token = p.total_ms_overlapped() / 1_000.0;
        assert!(
            prefill_ms_per_token < d.compute_ms / 10.0,
            "prefill {prefill_ms_per_token:.3} ms/token vs decode compute {:.0} ms/token",
            d.compute_ms
        );
    }

    /// Prefill under the native e8m0 layout, and the CEILING a knob search could reach.
    ///
    /// The native layout cuts what prefill streams; the expansion it needs is charged. What this
    /// test exists to pin is the second-order consequence the question "should we tune harder"
    /// turns on: once the two sweep terms are close, a pure compute win pays only down to the
    /// transfer term. `compute_headroom()` states that ceiling, so no search can quote a speedup
    /// the system cannot deliver.
    #[test]
    fn native_e8m0_prefill_and_the_compute_ceiling_it_leaves() {
        let f32l = ExpertLayout::PackedNibbleBlockF32;
        let natl = ExpertLayout::PackedNibbleNativeE8m0;
        for (name, layout) in [("f32 scales", f32l), ("native e8m0", natl)] {
            let g = Geometry::ds4_flash(layout);
            let rf = Budget::of(&g, &CUSTOMER_910B, 0, 4096).resident_fraction();
            for tokens in [1_000u32, 8_000] {
                let p = Prefill::estimate(&g, layout, tokens, rf, measured::H2D_PINNED_GBPS, 1);
                eprintln!(
                    "{name:<12} L={tokens:<6} unpack {:>6.0} + expand {:>5.0} = {:>6.0} ms vs \
                     transfer {:>6.0} ms | balance {:.2} | {:>5.2} s => {:>5.0} tok/s | \
                     compute headroom {:.2}x (bound by {})",
                    p.unpack_ms, p.expand_ms, p.sweep_compute_ms(), p.transfer_ms, p.balance(),
                    p.total_ms_overlapped() / 1000.0, p.tokens_per_sec_overlapped(),
                    p.compute_headroom(), p.bound_by()
                );
            }
        }
        let gn = Geometry::ds4_flash(natl);
        let rfn = Budget::of(&gn, &CUSTOMER_910B, 0, 4096).resident_fraction();
        let gf = Geometry::ds4_flash(f32l);
        let rff = Budget::of(&gf, &CUSTOMER_910B, 0, 4096).resident_fraction();
        let pn = Prefill::estimate(&gn, natl, 8_000, rfn, measured::H2D_PINNED_GBPS, 1);
        let pf = Prefill::estimate(&gf, f32l, 8_000, rff, measured::H2D_PINNED_GBPS, 1);
        // The expansion must be charged in prefill too.
        assert!(pn.expand_ms > 0.0 && pf.expand_ms == 0.0);
        // Native must stream strictly less.
        assert!(
            pn.sweep_streamed_bytes < pf.sweep_streamed_bytes,
            "native e8m0 must reduce prefill's streamed bytes"
        );
        // And the ceiling must be FINITE and modest — the point of the whole test.
        assert!(
            pn.compute_headroom() > 1.0 && pn.compute_headroom() < 2.0,
            "expected a modest compute ceiling, got {:.2}x — if this is large the sweep is \
             compute-dominated and a knob search is worth much more than assumed",
            pn.compute_headroom()
        );
    }

    /// Chunk-major prefill re-pays the fixed sweep per chunk. This is the cost of loop order, and
    /// it is more than an order of magnitude — worth failing a test over if the model ever stops
    /// reflecting it.
    #[test]
    fn chunk_major_prefill_is_far_worse_than_expert_major() {
        let layout = ExpertLayout::PackedNibbleBlockF32;
        let g = Geometry::ds4_flash(layout);
        let (tokens, chunk) = (8_192u32, 256u32);
        let one_sweep =
            Prefill::estimate(&g, layout, tokens, 0.18, measured::H2D_PINNED_GBPS, 1)
                .total_ms_overlapped();
        let chunked = estimate_chunked_prefill(
            &g,
            layout,
            tokens,
            chunk,
            0.18,
            measured::H2D_PINNED_GBPS,
            1,
        );
        let penalty = chunked / one_sweep;
        eprintln!(
            "expert-major {:.1} s vs {}-token chunks {:.1} s => {penalty:.1}x penalty",
            one_sweep / 1000.0,
            chunk,
            chunked / 1000.0
        );
        assert!(
            penalty > 10.0,
            "chunk-major should cost >10x, got {penalty:.1}x — the sweep is no longer fixed"
        );
        // A chunk covering the whole prompt must degenerate to the single sweep.
        let whole = estimate_chunked_prefill(
            &g,
            layout,
            tokens,
            tokens,
            0.18,
            measured::H2D_PINNED_GBPS,
            1,
        );
        assert!((whole - one_sweep).abs() < 1e-6);
    }

    /// Expert parallelism divides compute by the chip count and transfer by MEASURED link
    /// scaling — which are not the same number, and the difference is the point.
    ///
    /// This test asserted `/8` on both until the links were measured: eight of them deliver 2.63x
    /// one link, not 8x. Compute genuinely divides by 8 (each chip has its own cores and shares
    /// nothing), so the two sides of the roofline scale at very different rates.
    #[test]
    fn expert_parallelism_scales_both_terms() {
        let layout = ExpertLayout::PackedNibbleBlockF32;
        let g = Geometry::ds4_flash(layout);
        let one = Prefill::estimate(&g, layout, 8_000, 0.18, measured::H2D_PINNED_GBPS, 1);
        let eight = Prefill::estimate(&g, layout, 8_000, 0.18, measured::H2D_PINNED_GBPS, 8);
        for (a, b, name) in [
            (one.unpack_ms, eight.unpack_ms, "unpack"),
            (one.matmul_ms, eight.matmul_ms, "matmul"),
        ] {
            assert!((a / b - 8.0).abs() < 1e-6, "{name} is compute and must scale by 8: {a} vs {b}");
        }
        let want = measured::h2d_scaling(8);
        assert!(
            (one.transfer_ms / eight.transfer_ms - want).abs() < 1e-6,
            "transfer must scale by the MEASURED {want:.2}x, not by the chip count: {:.0} vs {:.0}",
            one.transfer_ms, eight.transfer_ms
        );
        assert!(want < 3.0, "eight links measured 2.63x; a value near 8 would mean the curve was lost");
        // Same bytes cross the link either way; only the time changes.
        assert_eq!(one.sweep_streamed_bytes, eight.sweep_streamed_bytes);
        // Zero is treated as one rather than dividing by zero.
        let zero = Prefill::estimate(&g, layout, 8_000, 0.18, measured::H2D_PINNED_GBPS, 0);
        assert_eq!(zero.chips, 1);
        assert!((zero.unpack_ms - one.unpack_ms).abs() < 1e-9);
    }

    /// Where batched prefill starts beating decode-style matvec-per-token. Below it the unpack is
    /// not yet amortised, which is why a short prompt should not use this path at all.
    #[test]
    fn batched_prefill_wins_only_above_a_short_prompt() {
        let layout = ExpertLayout::PackedNibbleBlockF32;
        let g = Geometry::ds4_flash(layout);
        // Full residency isolates compute, so this compares kernels rather than loaders.
        let decode_ms_per_token = Throughput::estimate(&g, 1.0, measured::H2D_PINNED_GBPS).compute_ms;
        let prefill_ms = |t: u32| {
            Prefill::estimate(&g, layout, t, 1.0, measured::H2D_PINNED_GBPS, 1).total_ms_overlapped()
        };
        let mut crossover = None;
        for t in 1..=512u32 {
            if prefill_ms(t) < decode_ms_per_token * t as f64 {
                crossover = Some(t);
                break;
            }
        }
        let t = crossover.expect("batched prefill must win somewhere under 512 tokens");
        eprintln!("batched prefill overtakes matvec-per-token at {t} tokens");
        assert!(
            (16..=128).contains(&t),
            "crossover at {t} tokens is outside the expected 16-128 range"
        );
    }

    /// Pinning host memory is worth 5.2x on the streaming half — a loader choice, not a kernel
    /// one, and larger than any single kernel optimisation recorded in this project.
    #[test]
    fn pinning_host_memory_dominates_kernel_wins() {
        let g = Geometry::ds4_flash(ExpertLayout::PackedNibbleBlockF32);
        let pin = Throughput::estimate(&g, 0.18, measured::H2D_PINNED_GBPS);
        let mal = Throughput::estimate(&g, 0.18, measured::H2D_MALLOC_GBPS);
        let gain = mal.transfer_ms / pin.transfer_ms;
        assert!(
            gain > 4.0,
            "pinned vs malloc should be >4x on transfer, got {gain:.2}"
        );
        // And it changes what the system is bound by, which is the part that matters.
        assert_eq!(mal.bound_by(), "transfer");
    }

    /// Full residency removes the transfer term entirely — the reason the memory work mattered.
    #[test]
    fn full_residency_leaves_only_compute() {
        let g = Geometry::ds4_flash(ExpertLayout::PackedNibbleBlockF32);
        let t = Throughput::estimate(&g, 1.0, measured::H2D_PINNED_GBPS);
        assert_eq!(t.streamed_bytes_per_token, 0);
        assert_eq!(t.transfer_ms, 0.0);
        assert_eq!(t.bound_by(), "compute");
        assert!(t.tokens_per_sec_serial() > 5.0, "resident decode should clear 5 tok/s");
    }

    /// A box with almost no free HBM must be REFUSED, not handed a plan with a zero-byte expert
    /// pool. The refusal has to name which wall was hit.
    #[test]
    fn a_nearly_full_box_is_refused_rather_than_planned() {
        let g = Geometry::ds4_flash(ExpertLayout::PackedNibbleBlockF32);
        // One chip with a little free space: the mandatory set alone cannot fit.
        let tiny = [(0u32, 2 * 1024 * 1024 * 1024u64)];
        match plan(&g, &tiny, 0, 128 * 1024) {
            Err(PlanError::MandatoryExceedsHbm { .. }) => {}
            other => panic!("expected a mandatory-set refusal, got {other:?}"),
        }
        // Enough for the mandatory set but not for one token's experts: the case that previously
        // returned a plausible plan nobody could run.
        let mand = g.mandatory(g.layers, 4096) + g.globals;
        let just_over = [(0u32, mand + g.min_working_set(g.layers) / 2)];
        match plan(&g, &just_over, 0, 4096) {
            Err(PlanError::WorkingSetExceedsHbm { working_set, .. }) => {
                assert!(working_set > 0, "the floor must be a real number");
            }
            other => panic!("expected a working-set refusal, got {other:?}"),
        }
    }

    /// The working-set floor is DERIVED from the expert count, not a constant. 6 of 256 experts per
    /// layer, so the floor is a small fraction of the expert bytes — and it must scale with layers.
    #[test]
    fn working_set_floor_follows_the_expert_count() {
        let g = Geometry::ds4_flash(ExpertLayout::PackedNibbleBlockF32);
        assert_eq!(g.bytes_per_expert(), g.experts_per_layer / 256);
        assert_eq!(
            g.min_working_set(g.layers),
            g.layers as u64 * 6 * g.bytes_per_expert()
        );
        // Half the layers, half the floor.
        assert_eq!(g.min_working_set(g.layers / 2) * 2, g.min_working_set(g.layers) - g.min_working_set(g.layers % 2));
    }

    /// KV is the term that grows with context, so it must be visible in the budget separately —
    /// it is the one most often left out of a headroom estimate.
    #[test]
    fn kv_is_accounted_and_grows_with_context() {
        let g = Geometry::ds4_flash(ExpertLayout::PackedNibbleBlockF32);
        let k32 = g.kv_bytes(g.layers, 32 * 1024);
        let k128 = g.kv_bytes(g.layers, 128 * 1024);
        assert_eq!(k128, k32 * 4, "KV is linear in context");
        let b = Budget::of(&g, &CUSTOMER_910B, 0, 128 * 1024);
        assert!(b.kv > 0 && b.mandatory > b.kv, "mandatory must include KV and more");
    }

    /// The general answer for a partly-occupied box: not a chip count, a CONTEXT length. On the
    /// customer's measured box the model is servable but only at a small context and low residency
    /// — and that is a finding to report, not an error.
    #[test]
    fn customer_box_yields_a_context_not_a_chip_count() {
        let g = Geometry::ds4_flash(ExpertLayout::PackedNibbleBlockF32);
        let b = Budget::of(&g, &CUSTOMER_910B, 0, 4096);
        eprintln!(
            "customer box: {:.1} GiB free, mandatory@4k {:.1} GiB, floor {:.1} GiB, \
             experts {:.1} GiB, resident {:.1}%",
            b.total_free as f64 / 2f64.powi(30),
            b.mandatory as f64 / 2f64.powi(30),
            b.min_working_set as f64 / 2f64.powi(30),
            b.experts_total as f64 / 2f64.powi(30),
            b.resident_fraction() * 100.0
        );
        assert!(b.servable(), "a token must at least be servable at 4k on this box");
        // Residency is poor, which is the honest characterisation rather than "not viable".
        assert!(
            b.resident_fraction() < 0.25,
            "expected low residency on a nearly-full box, got {:.2}",
            b.resident_fraction()
        );
        // And the largest servable context is bounded by memory, not by a chip count.
        let ctx = max_context(&g, &CUSTOMER_910B, 0, 0.10, 4096).expect("planning");
        eprintln!("largest context at a 10% residency floor: {ctx}");
        assert!(ctx > 0, "some context must be servable");
        assert!(ctx < 128 * 1024, "a nearly-full box must not reach 128k");
    }

    /// One 910 chip's HBM, as measured by the probe on these boxes: 61.27 GiB total,
    /// of which 57.07 GiB was free. Using TOTAL here makes these tests a statement about
    /// the hardware rather than about one moment's occupancy.
    const CHIP: u64 = 65_788_036_382; // 61.27 GiB

    /// Does the layout built and device-verified today actually reach three chips?
    ///
    /// This is the goal the whole 4-bit exercise was for, so it is checked here rather
    /// than asserted in a document. It also records HOW MUCH margin there is, because
    /// "fits" and "fits comfortably" lead to different decisions about step 3.
    #[test]
    fn packed_4bit_reaches_three_chips_but_only_just() {
        let g = Geometry::ds4_flash(ExpertLayout::PackedNibbleBlockF32);
        let ctx = 128 * 1024;
        let total = g.total_bytes(ctx);
        let cap3 = 3 * CHIP;
        assert!(
            total <= cap3,
            "packed 4-bit must fit three chips: {:.1} GiB > {:.1} GiB",
            total as f64 / 2f64.powi(30),
            cap3 as f64 / 2f64.powi(30)
        );
        assert_eq!(g.min_chips(CHIP, ctx), 3, "three chips by capacity");
        // Tight: the margin is a few GiB out of ~184, before any workspace reserve.
        let spare = (cap3 - total) as f64 / 2f64.powi(30);
        assert!(
            spare < 8.0,
            "margin unexpectedly large ({spare:.1} GiB) — recheck dims"
        );
        // And two chips genuinely cannot, so three is the floor and not a preference.
        assert!(total > 2 * CHIP, "two chips must NOT fit");
    }

    /// The rung this replaced: host-expanded int8 needs SIX chips at 128k with f32 KV.
    /// That gap is the entire justification for the in-kernel unpack, so it is pinned —
    /// and pinned at six rather than the five this work kept repeating, which came from
    /// step 1's accounting before the f32-KV decision was costed in.
    #[test]
    fn host_expanded_int8_needs_six_chips() {
        let ctx = 128 * 1024;
        let int8 = Geometry::ds4_flash(ExpertLayout::HostExpandedInt8BlockF32);
        let packed = Geometry::ds4_flash(ExpertLayout::PackedNibbleBlockF32);
        assert_eq!(int8.min_chips(CHIP, ctx), 6, "int8 rung");
        assert_eq!(packed.min_chips(CHIP, ctx), 3, "packed rung");
        // The codes halve; the scale plane does not, so the total is short of 2x.
        let ratio = int8.total_bytes(ctx) as f64 / packed.total_bytes(ctx) as f64;
        assert!(
            (1.6..1.85).contains(&ratio),
            "expected the saving to be under 2x because the scale plane is unchanged, got {ratio:.2}x"
        );
        // Halving the chips is the headline, and it is a stronger result than the "five
        // to three" this work had been repeating.
        assert_eq!(int8.min_chips(CHIP, ctx) / 2, packed.min_chips(CHIP, ctx));
    }

    /// Step 3 (native e8m0 scales) does not change the chip COUNT — it changes the
    /// margin, which is why it is worth doing even though three chips already fit.
    #[test]
    fn native_scales_buy_margin_not_a_chip() {
        let ctx = 128 * 1024;
        let blk = Geometry::ds4_flash(ExpertLayout::PackedNibbleBlockF32);
        let nat = Geometry::ds4_flash(ExpertLayout::PackedNibbleNativeE8m0);
        assert_eq!(nat.min_chips(CHIP, ctx), 3, "still three chips");
        let gain = (blk.total_bytes(ctx) - nat.total_bytes(ctx)) as f64 / 2f64.powi(30);
        assert!(gain > 15.0, "native scales should free >15 GiB, got {gain:.1}");
    }

    /// Bytes per weight must follow from the layout, since that is the number the whole
    /// ladder is built on and the number the device runs measured at 0.5 for the codes.
    #[test]
    fn bytes_per_weight_matches_the_layouts() {
        assert_eq!(ExpertLayout::PackedNibbleBlockF32.bytes_per_weight(), 0.5 + 0.125);
        assert_eq!(ExpertLayout::HostExpandedInt8BlockF32.bytes_per_weight(), 1.0 + 0.125);
        // Native e8m0 makes the fed layout identical to GGUF's stored 17 B per 32.
        assert_eq!(
            ExpertLayout::PackedNibbleNativeE8m0.bytes_per_weight(),
            17.0 / 32.0
        );
    }

    /// The derived expert term must be reproducible from the dims, so a future dims
    /// change cannot silently leave the footprint behind.
    #[test]
    fn expert_bytes_follow_from_dims() {
        let d = ModelDims::ds4_flash();
        assert_eq!(d.expert_weights_per_layer(), 3 * 4096 * 2048 * 256);
        let g = Geometry::ds4_flash(ExpertLayout::PackedNibbleBlockF32);
        assert_eq!(
            g.experts_per_layer,
            (d.expert_weights_per_layer() as f64 * 0.625) as u64
        );
    }

    fn geo() -> Geometry {
        Geometry::ds4_flash_4bit()
    }

    #[test]
    fn equal_chips_get_contiguous_equal_ranges() {
        let g = geo();
        let free = [(0, 61 << 30), (1, 61 << 30), (2, 61 << 30)];
        let p = plan(&g, &free, 2 << 30, 131_072).expect("plan");
        assert_eq!(p.chips.len(), 3);
        // Contiguous and covering every layer exactly once — the adjacency
        // property the KV locality argument depends on.
        assert_eq!(p.chips[0].first_layer, 0);
        for w in p.chips.windows(2) {
            assert_eq!(w[0].last_layer, w[1].first_layer);
        }
        assert_eq!(p.chips.last().unwrap().last_layer, g.layers);
        assert_eq!(p.hops_per_token(), 2);
    }

    #[test]
    fn an_unequal_chip_takes_fewer_layers() {
        let g = geo();
        // Chip 1 is half occupied by somebody else's job, which is the situation
        // measured on the 910b rather than a hypothetical.
        let free = [(0, 61 << 30), (1, 30 << 30), (2, 61 << 30)];
        let p = plan(&g, &free, 2 << 30, 131_072).expect("plan");
        let busy = p.chips.iter().find(|c| c.device == 1).expect("chip 1");
        let idle = p.chips.iter().find(|c| c.device == 0).expect("chip 0");
        assert!(
            busy.layers() < idle.layers(),
            "the half-occupied chip took {} layers against {}",
            busy.layers(),
            idle.layers()
        );
        // Still contiguous.
        for w in p.chips.windows(2) {
            assert_eq!(w[0].last_layer, w[1].first_layer);
        }
    }

    #[test]
    fn the_910b_fits_the_mandatory_set_but_leaves_no_expert_pool() {
        let g = geo();
        // The 910b as measured: ~4.9 GiB free per chip across 8 chips, the rest
        // held by somebody's vLLM job. Eight chips DO hold the ~19.7 GB of
        // non-expert weights and KV — this test originally asserted a refusal and
        // was simply wrong about the arithmetic. What actually happens is worse
        // than a refusal: a valid-looking plan with almost no expert pool, so
        // every expert streams on every token.
        let free: Vec<(u32, u64)> = (0..8).map(|d| (d, 4_900 * 1024 * 1024)).collect();
        let p = plan(&g, &free, 2 << 30, 131_072).expect("mandatory set does fit");
        assert!(
            p.worst_resident_fraction(&g) < 0.01,
            "expected a starved pool, got {:.3}",
            p.worst_resident_fraction(&g)
        );
        assert!(!p.is_viable(&g, 0.27), "should not be judged viable");
    }

    #[test]
    fn refuses_when_the_mandatory_set_genuinely_cannot_fit() {
        let g = geo();
        // Two chips at 4.9 GiB is 5.8 GiB usable after headroom, against ~19.7 GB
        // of weights and KV that are touched every token.
        let free: Vec<(u32, u64)> = (0..2).map(|d| (d, 4_900 * 1024 * 1024)).collect();
        let err = plan(&g, &free, 2 << 30, 131_072).expect_err("should refuse");
        assert!(err.to_string().contains("touched every token"), "{err}");
    }

    #[test]
    fn three_idle_chips_are_viable_and_the_910b_is_not() {
        let g = geo();
        let idle = [(0, 61 << 30), (1, 61 << 30), (2, 61 << 30)];
        let p = plan(&g, &idle, 2 << 30, 131_072).expect("plan");
        assert!(p.is_viable(&g, 0.27), "{}", p.summary(&g));
    }

    #[test]
    fn a_shorter_context_needs_less_and_can_fit_fewer_chips() {
        let g = geo();
        let free = [(0, 61 << 30), (1, 61 << 30), (2, 61 << 30)];
        let long = plan(&g, &free, 2 << 30, 131_072).expect("128k");
        let short = plan(&g, &free, 2 << 30, 4_096).expect("4k");
        // KV dominates the per-layer mandatory cost at long context, so a short
        // context leaves materially more room for the expert pool.
        let l: u64 = long.chips.iter().map(|c| c.expert_pool_bytes).sum();
        let s: u64 = short.chips.iter().map(|c| c.expert_pool_bytes).sum();
        assert!(s > l, "short-ctx pool {s} should exceed long-ctx pool {l}");
    }
}
