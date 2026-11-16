// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! The cube block planner agrees with the artefact ptoas already accepts.
//!
//! `plan_cube_blocks` is the first component of the K/N-blocked cube arm that
//! `mlir_to_cpp` does not yet have, and the arm is the only remaining lever on the
//! PRIMARY KPI that needs no new declared contract. This file holds the planner to
//! the plan the PTO column already runs, because a planner that satisfied the
//! capacity arithmetic while disagreeing with a validated artefact would be a
//! second opinion rather than a plan.
//!
//! The reference is `bench/deepseek_v41/pto/gemm_f16_lmhead_v41_16x5120x12928.pto`,
//! which ptoas 0.68 accepts: Kb=256 over 20 K-blocks, Nb=64 over 202 N-blocks, an
//! A block of 16x256 f16, a B block of 256x64 f16 and a 16x64 f32 accumulator.

use ascendc_emit::mlir_to_cpp::{blocked_cube_for_text, cube_caps, cube_dma_limits, plan_cube_blocks};

/// The validated shape, and the plan must be the artefact's.
#[test]
fn the_lmhead_plan_matches_the_stored_pto() {
    let p = plan_cube_blocks(16, 5120, 12928, 2, 2).expect("the lmhead must plan");
    assert_eq!(p.mpad, 16, "M=16 is already a whole fractal");
    assert_eq!(
        (p.kb, p.nb),
        (256, 64),
        "the planner disagrees with the stored .pto, which ptoas accepts at Kb=256 \
         Nb=64. One of the two is wrong and the artefact is the one with a device \
         behind it -- re-derive the planner, not the artefact."
    );
}

/// Every block of the chosen plan fits the capacities read from Ascend910B2.ini.
#[test]
fn every_block_of_a_chosen_plan_fits_the_measured_caps() {
    for (m, k, n, lb, rb) in [
        (16u32, 5120u32, 12928u32, 2u64, 2u64),
        (16, 5120, 16160, 2, 2),
        (16, 5120, 2304, 1, 1),
    ] {
        let p = plan_cube_blocks(m, k, n, lb, rb)
            .unwrap_or_else(|e| panic!("{m}x{k}x{n} must plan: {e}"));
        let a = p.mpad as u64 * p.kb as u64 * lb;
        let b = p.kb as u64 * p.nb as u64 * rb;
        let c = p.mpad as u64 * p.nb as u64 * cube_caps::ACC_BYTES;
        assert!(a <= cube_caps::L0A, "{m}x{k}x{n}: A block {a} > L0A");
        assert!(b <= cube_caps::L0B, "{m}x{k}x{n}: B block {b} > L0B");
        assert!(c <= cube_caps::L0C, "{m}x{k}x{n}: acc {c} > L0C");
        assert!(a + b <= cube_caps::L1, "{m}x{k}x{n}: A+B {} > L1", a + b);
        // The loop nest must be exact: a tail block would need a second body.
        assert_eq!(k % p.kb, 0, "{m}x{k}x{n}: Kb={} does not divide K", p.kb);
        assert_eq!(n % p.nb, 0, "{m}x{k}x{n}: Nb={} does not divide N", p.nb);
        assert_eq!(p.kb % cube_caps::FRACTAL, 0, "Kb off the fractal");
        assert_eq!(p.nb % cube_caps::FRACTAL, 0, "Nb off the fractal");
    }
}

/// M below the fractal is PADDED, not refused — the thing that makes M=1 reachable.
#[test]
fn m_below_the_fractal_is_padded_rather_than_refused() {
    let p = plan_cube_blocks(1, 5120, 2304, 1, 1).expect("M=1 must plan, padded");
    assert_eq!(
        p.mpad,
        cube_caps::FRACTAL,
        "M=1 must pad to the 16-row fractal. Refusing it is what made \
         gemm_i8_v41_1x5120x2304 unreachable, and padding the SOURCE instead \
         duplicates its 16-row sibling."
    );
}

/// A dimension with no fractal-multiple divisor is refused, with the reason.
#[test]
fn a_prime_dimension_is_refused_and_the_message_says_why() {
    // 12927 = 3 x 31 x 139, so no multiple of 16 divides it.
    let err = plan_cube_blocks(16, 5120, 12927, 2, 2)
        .expect_err("a dimension with no fractal divisor cannot be blocked exactly");
    assert!(
        err.contains("prime dimension has none") || err.contains("divides K and N"),
        "the refusal must name the divisibility requirement rather than the caps \
         alone, since the caps are satisfiable here and the divisor is not: {err}"
    );
}

/// The planner must NOT carry the two ptoas-specific limits.
#[test]
fn the_assembler_only_limits_are_not_imported() {
    // The 2^23 outer-stride bound would refuse a wide N. 5120 x 49152 is
    // Kb x N = 12.6M at Kb=256, which ptoas tolerates and a wider N does not --
    // but that bound belongs to ptoas's DMA descriptor, not to this path.
    let wide = plan_cube_blocks(16, 5120, 65536, 2, 2);
    assert!(
        wide.is_ok(),
        "a wide N was refused: the ptoas 24-bit outer-stride limit must not be \
         carried into the AscendC path, which never goes through ptoas. {wide:?}"
    );
    // And i8 must not be forced to Nb=256 by the BLayout workaround.
    let i8p = plan_cube_blocks(16, 5120, 2304, 1, 1).expect("i8 must plan");
    assert!(
        i8p.nb != 256 || 2304 % 256 == 0,
        "i8 Nb looks pinned to 256; that constant exists because ptoas picks the \
         L0A BLayout from the Right width, which is not a constraint here"
    );
}

// ─── The DMA and cube field widths, read from the box's own CANN headers ────
//
// Measured 2026-11-09 on opencompass-01 (910B2, reachable as `ssh 910b`), in
// /usr/local/Ascend/ascend-toolkit/latest/asc/include/basic_api/. These are
// FIELD WIDTHS in the hardware's parameter structs, not policy, so they bound
// any blocked cube arm whatever its block plan — which is why they are pinned
// here beside the plan rather than left in a comment.

/// `Nd2NzParams` is all-`uint16_t` on dav-2201, so an ND row stride is bounded.
///
/// `kernel_struct_data_copy.h` has two constructors under
/// `#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102))`:
/// the arch35 one widens `dValue` to `uint32_t` and the strides to `uint64_t`,
/// the else-branch — ours — keeps every field `uint16_t`. The blocked arm
/// strides A by K and B by N, so both must fit.
#[test]
fn the_nd2nz_row_stride_bound_is_the_uint16_field_and_both_lmheads_fit() {
    assert_eq!(cube_dma_limits::MAX_ND_ROW_STRIDE, u16::MAX as u32);
    // K, then the two N values the corpus has.
    for (k, n) in [(5120u32, 12928u32), (5120, 16160)] {
        assert!(
            k <= cube_dma_limits::MAX_ND_ROW_STRIDE && n <= cube_dma_limits::MAX_ND_ROW_STRIDE,
            "{k}x{n} does not fit the uint16 ND stride, so this shape would need \
             a different copy scheme than one Nd2Nz per block"
        );
    }
}

/// `LoadData2DParams::repeatTimes` is `uint8_t`, and that alone rules out the
/// unblocked pipeline for an lmhead.
///
/// `repeatTimes` carries the number of 16-element fractal blocks along K. The
/// chosen plan uses `Kb/16`; the existing unblocked `__tile_matmul_cube`
/// pipeline loads the whole K at once and would need `K/16`. For K=5120 that is
/// 320, which does not fit the field — an INDEPENDENT reason from the 252x L1
/// overflow, and the one that cannot be fixed by a bigger L1.
#[test]
fn the_load_repeat_field_rules_out_the_unblocked_pipeline_for_an_lmhead() {
    assert_eq!(cube_dma_limits::MAX_LOAD_REPEAT, u8::MAX as u32);
    let plan = plan_cube_blocks(16, 5120, 12928, 2, 2).expect("the lmhead plans");
    let blocked = plan.kb / cube_caps::FRACTAL;
    let unblocked = 5120 / cube_caps::FRACTAL;
    assert!(
        blocked <= cube_dma_limits::MAX_LOAD_REPEAT,
        "the chosen plan needs repeatTimes={blocked}, over the uint8 field"
    );
    assert!(
        unblocked > cube_dma_limits::MAX_LOAD_REPEAT,
        "K=5120 unblocked needs repeatTimes={unblocked}; if that now fits the field \
         this test's claim about the unblocked pipeline is stale and the refusal \
         message quoting it must change too"
    );
}

// ─── The whole-kernel recogniser ───────────────────────────────────────────
//
// The arm it gates is NOT WRITTEN YET. These two tests exist so that when it is,
// it is wired to the shapes that refuse and to nothing else: a recogniser that
// quietly matched a kernel the ordinary path already emits would replace working
// output with a new and unverified route.

/// The lmhead is recognised, with the artefact's plan.
#[test]
fn the_lmhead_is_recognised_as_a_whole_kernel_blocked_cube() {
    let src = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../bench/deepseek_v41/tile/gemm_f16_lmhead_v41_16x5120x12928.tile"
    ))
    .expect("the corpus artefact is readable");
    // 94208 is the per-buffer budget the emitter reports for this kernel.
    let bc = blocked_cube_for_text(&src, 94_208).expect("the four-op lmhead is recognised");
    assert_eq!((bc.m, bc.k, bc.n), (16, 5120, 12928));
    assert_eq!((bc.plan.mpad, bc.plan.kb, bc.plan.nb), (16, 256, 64));
    assert_eq!((bc.a_gm, bc.b_gm, bc.c_gm), (0, 1, 2));
}

/// A matmul the ordinary path already emits is NOT recognised.
///
/// `playground16/matmul.tile` is 1x64x64 f16 and fits UB with room to spare. It
/// emits today — as a scalar `TILE_DEGENERATE_MATMUL` loop, which is slow rather
/// than wrong — and this arm must leave it alone.
#[test]
fn a_matmul_that_already_fits_is_left_to_the_existing_path() {
    let src = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../bench/playground16/tile/matmul.tile"
    ))
    .expect("the corpus artefact is readable");
    assert!(
        blocked_cube_for_text(&src, 94_208).is_none(),
        "a 1x64x64 f16 matmul fits UB and must keep the path that already emits it"
    );
}

/// The budget gate is live: the SAME lmhead is not recognised when it fits.
///
/// The two tests above differ in both the artefact and the budget, so either one
/// alone could pass with the gate stuck. This holds the artefact fixed and moves
/// only the budget, which is the condition the arm is supposed to turn on.
#[test]
fn the_recogniser_turns_on_the_budget_and_not_the_shape() {
    let src = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../bench/deepseek_v41/tile/gemm_f16_lmhead_v41_16x5120x12928.tile"
    ))
    .expect("the corpus artefact is readable");
    assert!(
        blocked_cube_for_text(&src, 94_208).is_some(),
        "recognised at the real per-buffer budget"
    );
    assert!(
        blocked_cube_for_text(&src, u64::MAX).is_none(),
        "a budget that holds both operands must leave the kernel to the ordinary \
         path -- this arm exists for the shapes that refuse, and nothing else"
    );
}

/// M=1 routes to the vector GEMV, not to a cube padded to 16 rows.
///
/// The i8 `.tile` now carries the scale operand the 16x form passes. The
/// recogniser then reads M=1 and the generator must not hand it to the cube
/// (M is below the 16-row fractal); it reuses the AIV dequant and does a
/// contiguous-column f16 dot instead.
#[test]
fn the_m1_i8_gemm_emits_a_vector_gemv_instead_of_a_padded_cube() {
    let src = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../bench/deepseek_v41/tile/gemm_i8_v41_1x5120x2304.tile"
    ))
    .expect("the corpus artefact is readable");
    let bc = blocked_cube_for_text(&src, 94_208)
        .expect("the corrected 7-arg i8 call is recognised");
    assert_eq!((bc.m, bc.k, bc.n), (1, 5120, 2304));
    assert_eq!(
        bc.scale_gm,
        Some(3),
        "the per-column scale is the 4th GM argument, not an invented value"
    );
    let cpp = ascendc_emit::mlir_to_cpp::blocked_cube_cpp_for_text(&src, 94_208)
        .expect("the M=1 i8 GEMM emits");
    assert!(
        cpp.contains("bm_gemm_i8_1_5120_2304_mv(GM_ADDR"),
        "the M=1 contraction must be the vector GEMV entry"
    );
    assert!(
        !cpp.contains("AscendC::Mmad"),
        "M=1 must not reach the cube, whose fixed row size is 16"
    );
    assert!(cpp.contains("AscendC::Muls") && cpp.contains("AscendC::Add"));
}

// ─── The emitted schedule, checked where the ORACLE CANNOT SEE ─────────────
//
// Measured by the peer lane on 2026-11-09 and the reason these tests exist: at
// K = 5120 a SINGLE dropped k is ~2e-4 relative, which is BELOW f16's
// representation floor at |C| ~ 1920 (spacing 1.0, representation error
// ~2.6e-4). So the general-GEMM oracle at lmhead shape can detect a dropped
// K-BLOCK (~10%) and provably cannot detect an off-by-one in the k walk. That is
// a property of the output dtype at that magnitude, not something a tolerance
// change fixes.
//
// An off-by-one in the k walk is exactly this arm's most likely defect, so it
// needs a gate that is not the oracle. These assert the emitted SCHEDULE: the
// device run covers layout and dropped blocks, these cover the walk. Neither
// alone is sufficient and saying so is the point.

fn lmhead_cpp() -> String {
    let src = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../bench/deepseek_v41/tile/gemm_f16_lmhead_v41_16x5120x12928.tile"
    ))
    .expect("the corpus artefact is readable");
    ascendc_emit::mlir_to_cpp::blocked_cube_cpp_for_text(&src, 94_208)
        .expect("the lmhead emits a blocked cube class")
}

/// The K walk covers `[0, K)` exactly once, and the last block carries its TRUE
/// width rather than `KB`.
#[test]
fn the_k_walk_covers_the_contraction_exactly_once() {
    let cpp = lmhead_cpp();
    assert!(
        cpp.contains("for (int k0 = 0; k0 < K; k0 += KB)"),
        "the K loop must be a half-open walk in KB steps: \n{cpp}"
    );
    assert!(
        cpp.contains("const int kw = (K - k0 < KB) ? (K - k0) : KB;"),
        "the block width must be CLAMPED, or a K not divisible by KB reads past the end"
    );
    // The Mmad's k must be the clamped width, never the nominal block size --
    // `mp.k = KB` on a tail block would contract over data that was not copied.
    assert!(
        cpp.contains("mp.k = kw;") && !cpp.contains("mp.k = KB"),
        "Mmad must contract over the clamped width kw, not the nominal KB"
    );
    // Same for the fractal repeat count.
    assert!(
        cpp.contains("(uint8_t)(kw / F)") && !cpp.contains("(KB / F)"),
        "repeatTimes must derive from the clamped width"
    );
}

/// `cmatrixInitVal` is true on exactly the first K block, and `isBias` is its
/// complement.
///
/// `MmadParams::cmatrixInitVal` DEFAULTS TO TRUE ("the number in C matrix is 0",
/// `kernel_struct_mm.h`), so leaving it alone zeroes the accumulator on every
/// block and the result is the LAST K block instead of the sum. The pair has to
/// be complementary, and asserting both is what makes a half-fix visible.
#[test]
fn the_accumulator_is_initialised_on_the_first_k_block_only() {
    let cpp = lmhead_cpp();
    assert!(
        cpp.contains("mp.cmatrixInitVal = (k0 == 0);"),
        "the accumulator must be treated as zero only on the first K block"
    );
    assert!(
        cpp.contains("mp.isBias = (k0 != 0);"),
        "every later K block must multiply-AND-ADD into the accumulator"
    );
    assert!(
        cpp.contains("mp.cmatrixSource = false;"),
        "the C matrix is in L0C, not the bias table"
    );
}

/// The ND source strides are the FULL array widths, not the block widths.
///
/// The classic blocked-copy defect: `srcDValue` is the stride of the array being
/// read from, so A strides by K and B by N even though the blocks are kw and gw
/// wide. Using the block width walks a contiguous kw*gw region instead of a
/// sub-block, which compiles, runs, and returns a different matrix.
#[test]
fn the_nd_source_strides_are_the_array_widths_not_the_block_widths() {
    let cpp = lmhead_cpp();
    assert!(
        cpp.contains("pa.srcDValue = K;"),
        "A is a sub-block of an M x K array, so its ND row stride is K"
    );
    assert!(
        cpp.contains("pb.srcDValue = N;"),
        "B is a sub-block of a K x N array, so its ND row stride is N"
    );
    assert!(
        !cpp.contains("srcDValue = kw") && !cpp.contains("srcDValue = gw"),
        "a block width as the source stride reads a contiguous region, not a sub-block"
    );
    // And the B offset is the row-major address of (k0, n0+g).
    assert!(
        cpp.contains("bGM[(int64_t)k0 * N + n0 + g]"),
        "B's block origin must be the row-major address of (k0, n0+g), widened to \
         int64 -- 5120*12928 overflows int32 at the far end of this kernel"
    );
}

// ─── The scorable small shape ──────────────────────────────────────────────

/// The small fixture gives THREE K blocks and THREE N blocks, and is recognised.
///
/// Three rather than two on purpose: with two blocks, an off-by-one at the last
/// and one at the first are the same single case, and the K walk is where this
/// arm's likeliest defect lives. `tests/fixtures/blocked_cube_16x768x192.tile`
/// carries the rest of the reasoning, including why the lmhead cannot gate it.
#[test]
fn the_small_fixture_blocks_three_by_three_and_is_recognised() {
    let src = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/blocked_cube_16x768x192.tile"
    ))
    .expect("the fixture is readable");
    let bc = blocked_cube_for_text(&src, 94_208).expect("recognised: B is over the budget");
    assert_eq!((bc.m, bc.k, bc.n), (16, 768, 192));
    assert_eq!((bc.plan.mpad, bc.plan.kb, bc.plan.nb), (16, 256, 64));
    assert_eq!(bc.k / bc.plan.kb, 3, "three K blocks");
    assert_eq!(bc.n / bc.plan.nb, 3, "three N blocks");
    // EXACT MULTIPLES, stated because it bounds what a run on this shape proves:
    // there is no K tail and no N tail, so the `kw`/`gw` CLAMPING that the
    // schedule tests above assert structurally is NOT exercised at runtime here.
    // A tail shape is the next fixture, deliberately after this one scores --
    // stacking two unknowns onto one run makes a failure unattributable.
    assert_eq!(bc.k % bc.plan.kb, 0);
    assert_eq!(bc.n % bc.plan.nb, 0);
    let cpp = ascendc_emit::mlir_to_cpp::blocked_cube_cpp_for_text(&src, 94_208)
        .expect("and it emits");
    assert!(cpp.contains("constexpr int M = 16, K = 768, N = 192;"));
    // THE `__global__` ENTRY, which the generator owes, and NOT the `_do`
    // launcher, which the CALLER appends for every path out of
    // `generate_kernel`. This assertion named `_do` until 2026-11-09, when the
    // generator stopped emitting it: emitting both produced two `_do` symbols in
    // the wired output and would not have linked.
    assert!(
        cpp.contains("__global__ __aicore__ void bm_gemm_f16_16_768_192("),
        "the generator owes the __global__ entry the caller's _do launcher calls"
    );
    assert!(
        !cpp.contains("_do(uint32_t blockDim"),
        "the `_do` launcher is the caller's; emitting it here duplicates the symbol"
    );
}

// ─── The size-bisect ladder ────────────────────────────────────────────────
//
// Written after the 16x768x192 fixture ran on device and returned wrong values.
// Each rung adds exactly ONE mechanism to the rung below it, so the rung that
// first disagrees with the oracle names the mechanism. The ladder exists because
// of the mirror of runs 6-8 in `evidence/0024-the-cube-arm-runs`: three
// independent perturbations of a parameter space that fail identically show the
// defect is NOT in that space -- and tuning parameters before the smallest
// working case exists is the same error from the other side.
//
// WHICH RUNG CLEARS WHICH PARAMETER, because a PASS at the smallest rung
// licenses NOTHING about strides -- at 16x16x16 every stride candidate equals
// 16, so a wrong rule and a right rule AGREE there. That is no evidence, not
// weak evidence.
//
//   A 16x16x16    the Mmad, both loads and the store at one fractal; nothing strided
//   B 16x512x16   K != kw: `pa.srcDValue = K`, the `kw` clamp, the accumulation
//   C 16x16x128   N != gw: `pb.srcDValue = N`, the B block origin, the group walk,
//                 `fp.dstStride`
//   D 16x768x192  all of it interacting, 3 K blocks x 3 N blocks
//
// All three new rungs are below the recogniser's budget gate, so they are emitted with a
// forced small `per_buf_bytes`. The corpus path is untouched: `94_208` is the
// real per-buffer budget and these shapes are not recognised at it, which the
// last assertion in each case pins.
#[test]
fn the_size_bisect_ladder_adds_one_mechanism_per_rung() {
    // (fixture, K, N, expected K blocks, expected 16-column groups per N block,
    //  expected N blocks)
    let rungs = [
        ("blocked_cube_16x16x16", 16u32, 16u32, 1u32, 1u32, 1u32),
        ("blocked_cube_16x512x16", 512, 16, 2, 1, 1),
        ("blocked_cube_16x16x128", 16, 128, 1, 4, 2),
        ("blocked_cube_16x768x192", 768, 192, 3, 4, 3),
    ];
    for (name, k, n, kb_count, groups, nb_count) in rungs {
        let path = format!(
            "{}/tests/fixtures/{name}.tile",
            env!("CARGO_MANIFEST_DIR")
        );
        let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        // A tiny budget, because every rung but the last fits UB comfortably.
        let bc = blocked_cube_for_text(&src, 256).unwrap_or_else(|| panic!("{name} recognised"));
        assert_eq!((bc.k, bc.n), (k, n), "{name} shape");
        assert_eq!(bc.k.div_ceil(bc.plan.kb), kb_count, "{name} K blocks");
        assert_eq!(bc.plan.nb / 16, groups, "{name} 16-column groups per N block");
        assert_eq!(bc.n.div_ceil(bc.plan.nb), nb_count, "{name} N blocks");
        // The ladder is a TEST instrument. Only the last rung is over the real
        // per-buffer budget, so only it is on the corpus path.
        let on_corpus_path = blocked_cube_for_text(&src, 94_208).is_some();
        assert_eq!(
            on_corpus_path,
            name == "blocked_cube_16x768x192",
            "{name}: a rung below the real budget must not change what the corpus emits"
        );
    }
}

/// The widened schedule's `LoadData` repeat count still fits the `uint8_t` field.
///
/// The first working version ran one `Mmad` per 16-column group, so B's
/// `repeatTimes` was `kw / F` — at most 16, nowhere near the bound. The widened
/// form brings the WHOLE N block into L0B in one `LoadData`, which is
/// `(kw / F) * (nb / F)` fractals, and `LoadData2DParams::repeatTimes` is
/// `uint8_t`. At Kb=256 Nb=64 that is 16 * 4 = 64; the plan must never choose a
/// pair whose product exceeds 255, and nothing else in the emitter checks it.
#[test]
fn the_widened_b_load_fits_the_uint8_repeat_field() {
    let shapes = [
        (16u32, 5120u32, 12928u32),
        (16, 5120, 16160),
        (16, 768, 192),
        (16, 512, 16),
        (16, 16, 128),
        (16, 16, 16),
    ];
    for (m, k, n) in shapes {
        let p = plan_cube_blocks(m, k, n, 2, 2).unwrap_or_else(|e| panic!("{m}x{k}x{n}: {e}"));
        let repeats = (p.kb / cube_caps::FRACTAL) * (p.nb / cube_caps::FRACTAL);
        assert!(
            repeats <= cube_dma_limits::MAX_LOAD_REPEAT,
            "{m}x{k}x{n}: Kb={} Nb={} needs repeatTimes={repeats}, over the uint8 field. \
             The widened B load brings the whole N block into L0B at once, so the plan's \
             Kb and Nb are coupled through this field in a way the narrow schedule hid",
            p.kb,
            p.nb
        );
        // And the A load, which is unchanged by the widening.
        assert!((p.kb / cube_caps::FRACTAL) <= cube_dma_limits::MAX_LOAD_REPEAT);
    }
}
