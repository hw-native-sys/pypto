// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! # tile_codegen -- the vendor-neutral codegen skeleton
//!
//! The generic half of the Rust codegen path: the trait every target
//! implements, the registry they join, and the hardware parameters a target is
//! described by. Standard library only -- no LLVM, no MLIR bindings, no rustc
//! internals -- which is why it builds on stable Rust in seconds on any host.
//!
//! Everything target-specific plugs in through one trait ([`CodegenTarget`]) and
//! one registry ([`TargetRegistry`]). A target implements the trait in its own
//! crate and calls [`TargetRegistry::register`]; the arrow points inward only,
//! so nothing here knows which targets exist. `rspto_emit` registers PTO that
//! way.
//!
//! ## What a target has to answer
//!
//! [`CodegenTarget::emit`] is pure MLIR text to target source: no filesystem, no
//! environment, no toolchain. That is what makes it unit-testable, and what lets
//! two input paths be checked for byte-identical output. Anything a target needs
//! to know about the machine arrives in [`EmitOpts`], whose [`HardwareParams`]
//! carries the capacities and the instruction-encoding limits -- the fields a
//! tiling decision is actually bounded by.
//!
//! ## What is deliberately absent
//!
//! An earlier version of this crate carried a feature that wired in emitters for
//! other vendors' processors. This repository's licence does not cover those
//! targets, so neither the feature nor the emitters are here. The seam they used
//! is unchanged and is the one `rspto_emit` joins through.

pub mod bandit;
pub mod hazard;
pub mod pointwise;
pub mod registry;
pub mod schedule;
pub mod target;
pub mod targets;

pub use bandit::{ArmStat, BanditState, bandit_enabled};
pub use pointwise::{
    AxisClass, Collapsed, InstanceKey, Operand, OperandKind, PointwiseSchema, Promoted,
    PromotionKind, ScalarType, TilePlan, TypeCategory, collapse, elementwise_dtypes, plan_tiles,
    promote_types,
};
pub use registry::TargetRegistry;
pub use schedule::{K_UNROLL_ARMS, MAX_K_UNROLL, Schedule};
pub use target::{
    CodegenTarget, EmitOpts, EmitOut, HardwareParams, NanTest, PathRates, TargetMeta,
    TargetSemantics,
};
pub use targets::{DebugTarget, EmitterTarget};

// The `emitters` feature wired the non-Ascend `convert_mlir_to_*` backends in
// through `#[path]` includes. Those targets are outside what this repository's
// licence covers, so the feature, the includes and the module are gone. The PTO
// emitter is not wired here either: it lives in `rspto_emit`, which registers
// itself through `TargetRegistry::register` exactly as any other target would.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_is_populated_and_selectable() {
        let r = TargetRegistry::with_builtin();
        assert!(!r.is_empty(), "registry should have at least the debug target");
        assert!(r.select("debug").is_some(), "debug target must be registered");
        assert!(r.select("does-not-exist").is_none());
        assert!(r.names().contains(&"debug"));
    }

    #[test]
    fn debug_target_emits_source() {
        let r = TargetRegistry::with_builtin();
        let t = r.select("debug").expect("debug target");
        let out = t
            .emit("func.func @kernel() { return }", &EmitOpts::default())
            .expect("emit ok");
        assert!(out.source.contains("func.func"));
        assert!(out.source.contains("tile-rs debug target"));
        assert_eq!(out.ext, "mlir.txt");
    }

    #[test]
    fn empty_mlir_is_an_error() {
        let r = TargetRegistry::with_builtin();
        let t = r.select("debug").unwrap();
        assert!(t.emit("   \n  ", &EmitOpts::default()).is_err());
    }

    #[test]
    fn register_adds_a_custom_target() {
        struct Noop;
        impl CodegenTarget for Noop {
            fn name(&self) -> &'static str {
                "noop"
            }
            fn emit(&self, _m: &str, _o: &EmitOpts) -> Result<EmitOut, String> {
                Ok(EmitOut::default())
            }
        }
        let mut r = TargetRegistry::new();
        assert!(r.is_empty());
        r.register(Box::new(Noop));
        assert_eq!(r.len(), 1);
        assert!(r.select("noop").is_some());
    }

    #[test]
    fn a_nan_test_lowers_to_not_eq_where_ne_is_ordered() {
        // Ascend's `!=` returns FALSE for a NaN operand, exactly like `<`, so
        // `Compare(v, v, NE)` is a NaN test that never fires. The backend has to
        // emit `!(v == v)` instead, and this is where it learns that.
        assert_eq!(TargetSemantics::ascend_arch32().nan_test().unwrap(), NanTest::NotEq);
        // Everywhere else the ordinary spelling is correct.
        assert_eq!(TargetSemantics::default().nan_test().unwrap(), NanTest::Ne);
        // arch35 reworks the vector unit, so arch32's answer does not transfer
        // and the query refuses rather than guessing.
        assert!(TargetSemantics::ascend_arch35_unmeasured().nan_test().is_err());
    }

    #[test]
    fn bitwise_ops_count_in_sixteen_bit_lanes_on_ascend() {
        let a = TargetSemantics::ascend_arch32();
        // An int32 buffer of n elements needs 2n: passing n leaves HALF the
        // destination untouched, which is how this was found.
        assert_eq!(a.bitwise_count(1024, 4).unwrap(), 2048);
        // 16-bit dtypes are already one lane each.
        assert_eq!(a.bitwise_count(1024, 2).unwrap(), 1024);
        // A 1-byte dtype rounds up rather than dropping the odd element.
        assert_eq!(a.bitwise_count(3, 1).unwrap(), 2);
        assert_eq!(a.bitwise_count(4, 1).unwrap(), 2);
        // Elsewhere the count is simply the element count.
        assert_eq!(TargetSemantics::default().bitwise_count(1024, 4).unwrap(), 1024);
        // arch35 refuses: a half-written destination is silent.
        assert!(TargetSemantics::ascend_arch35_unmeasured().bitwise_count(1024, 4).is_err());
        // So does an Ascend vector unit nobody has probed at all.
        assert!(TargetSemantics::ascend_unmeasured().bitwise_count(1024, 4).is_err());
    }

    #[test]
    fn ascend_semantics_must_not_be_the_default_a_non_ascend_target_gets() {
        // The failure this guards is silent and one-directional. Reaching for
        // `Default` on an Ascend path compiles, passes every test here, and emits
        // `x != x` as a NaN test -- which on Ascend returns false for a NaN and
        // so never fires. Nothing in the generated C++ says why. The default is
        // the IEEE answer and is CORRECT for cuda/metal/vulkan/aie/nki, which is
        // exactly why it is dangerous to reach for by convenience.
        let non_ascend = TargetSemantics::default();
        assert_eq!(non_ascend.nan_test().unwrap(), NanTest::Ne);
        assert_eq!(non_ascend.bitwise_count(1024, 4).unwrap(), 1024);

        // Every Ascend answer differs from it in at least one of the two facts,
        // so no Ascend path can be served by the default even if a future one
        // forgets to ask which chip it is on.
        for ascend in [
            TargetSemantics::ascend_arch32(),
            TargetSemantics::ascend_arch35_unmeasured(),
            TargetSemantics::ascend_unmeasured(),
        ] {
            assert_ne!(ascend.nan_test(), non_ascend.nan_test());
            assert_ne!(ascend.bitwise_count(1024, 4), non_ascend.bitwise_count(1024, 4));
        }

        // And the measured one is the only one that answers at all.
        assert!(TargetSemantics::ascend_arch32().nan_test().is_ok());
        assert!(TargetSemantics::ascend_unmeasured().nan_test().is_err());
    }

    #[test]
    fn semantics_ride_on_emit_opts_beside_the_resource_bounds() {
        // A target reads what its instructions MEAN next to how much fits.
        let opts = EmitOpts {
            hw: HardwareParams::ascend_910b(),
            sem: TargetSemantics::ascend_arch32(),
        };
        assert_eq!(opts.sem.nan_test().unwrap(), NanTest::NotEq);
        assert_eq!(opts.hw.npu_arch, "DAV_2201");
        // and the default target is unaffected by either
        let plain = EmitOpts::default();
        assert_eq!(plain.sem.nan_test().unwrap(), NanTest::Ne);
        assert_eq!(plain.sem.bitwise_count(99, 4).unwrap(), 99);
    }

    #[test]
    fn an_unmeasured_arch_refuses_rather_than_borrowing_another_chips_numbers() {
        let a = HardwareParams::ascend_910b();
        assert_eq!(a.npu_arch, "DAV_2201");
        assert!(a.measured);

        // A 910B number is not a 950 number. Validating a 950 tiling against
        // arch32 capacities would approve exactly the tilings that fail there,
        // so every check refuses instead.
        let b = HardwareParams::ascend_950_unmeasured();
        assert_eq!(b.npu_arch, "DAV_3510");
        assert!(!b.measured);
        for e in [
            b.check_ub(1024).unwrap_err(),
            b.check_repeat(64, 4).unwrap_err(),
            b.check_dma_stride(8, "src").unwrap_err(),
            b.check_cube_tile(16, 16, 16, 2).unwrap_err(),
        ] {
            assert!(e.contains("MEASURED"), "{e}");
            assert!(e.contains("DAV_3510"), "{e}");
        }
        // and the refusal is not "everything is fine": a shape the 910B accepts
        // is still refused for the 950, because nobody has looked.
        assert!(a.check_ub(1024).is_ok());
    }

    #[test]
    fn unknown_path_rates_refuse_the_fill_ratio_question() {
        // A target whose rates nobody has filled in must not answer "false"
        // to "is the L0A feed faster?". False is a real claim -- it is the
        // 950's answer -- and returning it by default would hand a fill-ratio
        // planner the 950 answer for every unknown chip.
        let unknown = PathRates { known: false, ..PathRates::ascend_910b_documented() };
        let e = unknown.l0_feed_is_asymmetric().unwrap_err();
        assert!(e.contains("NOT KNOWN"), "{e}");

        // A non-Ascend target is a different case and is NOT a refusal: having
        // no L1->L0 feed at all is a fact about it, so the question answers.
        assert!(PathRates::none().known);
        assert!(!PathRates::none().l0_feed_is_asymmetric().unwrap());
        assert!(!HardwareParams::default().rates.l0_feed_is_asymmetric().unwrap());
    }

    #[test]
    fn equal_l0_capacities_are_not_evidence_the_950_behaves_like_the_910b() {
        // The trap this pins is a SILENT one, and it is the likely way the
        // refusal above gets switched off by someone acting in good faith.
        //
        // CANN 9.2.0's platform_config has L0A and L0B byte-identical across
        // the two families, and L1 identical too. So a probe that reads
        // CAPACITIES off a 950 part finds nothing changed and concludes the
        // 910B descriptor transfers.
        let a = HardwareParams::ascend_910b();
        let b = HardwareParams::ascend_950_unmeasured();
        assert_eq!(a.l0a_bytes, b.l0a_bytes, "L0A is the same on both");
        assert_eq!(a.l0b_bytes, b.l0b_bytes, "L0B is the same on both");

        // It does not transfer, and the reason is an axis no CAPACITY carries.
        // The same file gives l1_to_l0_a_rate 512 -> 256 while l1_to_l0_b_rate
        // stays 256: on 910B the L1->L0A feed is TWICE the L0B feed, on 950
        // they are equal. Any tile justified by L0 fill ratios -- "filling L0B
        // alone starves L0A" -- rests on that 2:1 asymmetry and loses its
        // premise, with no capacity moving to show it.
        assert_eq!(a.rates.l1_to_l0_a, 512);
        assert_eq!(a.rates.l1_to_l0_b, 256);
        assert_eq!(b.rates.l1_to_l0_a, 256);
        assert_eq!(b.rates.l1_to_l0_b, 256);
        assert_ne!(
            a.rates, b.rates,
            "equal capacities, DIFFERENT paths -- the whole point"
        );

        // and said as the decision rather than as two numbers: the premise of
        // every fill-ratio tile holds on one chip and not the other.
        assert!(a.rates.l0_feed_is_asymmetric().unwrap(), "910B feeds L0A 2x");
        assert!(!b.rates.l0_feed_is_asymmetric().unwrap(), "950 feeds equally");

        // The 950 also changes the SHAPE of that feed, not only its rate.
        assert!(!a.rates.l0a_layout_zn);
        assert!(b.rates.l0a_layout_zn);
        //
        // So capacity equality must never be the thing that sets `measured`.
        // While the rates are unrepresentable here, the only honest state for
        // this descriptor is refusal.
        assert!(!b.measured, "a capacity probe must not be able to set this");
        assert!(b.check_cube_tile(16, 16, 16, 2).is_err());

        // The capacities that DID move are carried, so the descriptor is not
        // quietly asserting 910B numbers it is known to contradict.
        assert_eq!(b.l0c_bytes, 262144, "950 doubles L0C");
        assert_eq!(a.l0c_bytes, 131072);
        assert_eq!(b.ub_size, 253952);
        assert_eq!(a.ub_size, 196608);

        // and both SKUs inherit the corrected numbers, not 910B's.
        for sku in [
            HardwareParams::ascend_950pr_unmeasured(),
            HardwareParams::ascend_950dt_unmeasured(),
        ] {
            assert_eq!(sku.l0c_bytes, 262144);
            assert_eq!(sku.ub_size, 253952);
            assert!(!sku.measured);
        }
    }

    #[test]
    fn pr_and_dt_share_an_isa_target_but_are_different_machines() {
        let pr = HardwareParams::ascend_950pr_unmeasured();
        let dt = HardwareParams::ascend_950dt_unmeasured();
        // Same compile identity — there is no separate DT flag.
        assert_eq!(pr.npu_arch, "DAV_3510");
        assert_eq!(dt.npu_arch, "DAV_3510");
        assert!(!pr.measured && !dt.measured);
        // Different silicon ids, so a ceiling measured on one cannot be
        // filed under the other.
        assert_eq!(pr.chip, "Ascend950PR_9589");
        assert_eq!(dt.chip, "Ascend950DT_9582");
        assert_ne!(pr.chip, dt.chip);
        // Capacity checks still refuse on both (nothing measured on either).
        assert!(pr.check_ub(1024).unwrap_err().contains("Ascend950PR_9589"));
        assert!(dt.check_ub(1024).unwrap_err().contains("Ascend950DT_9582"));
        // Eval-image OP gap is DT-only evidence: PR has the dyn-shape set,
        // DT does not — and that refusal does not depend on `measured`.
        assert!(pr.check_vendor_op("Cat").is_ok());
        let e = dt.check_vendor_op("ConcatD").unwrap_err();
        assert!(e.contains("no OP JSON"), "{e}");
        assert!(e.contains("Ascend950DT_9582"), "{e}");
        // 910B and the generic 950 SKU have no known OP gap.
        assert!(HardwareParams::ascend_910b().check_vendor_op("Cat").is_ok());
        assert!(HardwareParams::ascend_950_unmeasured().check_vendor_op("Cat").is_ok());
    }

    #[test]
    fn c7_the_repeat_field_is_eight_bits() {
        let hw = HardwareParams::ascend_910b();
        // 255 repeats x 64 lanes is the furthest one f32 instruction reaches.
        assert!(hw.check_repeat(16320, 4).is_ok());
        let e = hw.check_repeat(16321, 4).unwrap_err();
        assert!(e.contains("C7") && e.contains("16320"), "{e}");
        // A 2-byte dtype gets 128 lanes, so it reaches twice as far.
        assert!(hw.check_repeat(32640, 2).is_ok());
        assert!(hw.check_repeat(32641, 2).is_err());
        // What a tile search must not propose: 256x256 f32 is 1024 repeats.
        assert!(hw.check_repeat(256 * 256, 4).is_err());
        assert!(hw.check_repeat(64 * 64, 4).is_ok());
    }

    #[test]
    fn c8_the_dma_stride_field_is_sixteen_bits() {
        let hw = HardwareParams::ascend_910b();
        assert!(hw.check_dma_stride(32767, "src").is_ok());
        assert!(hw.check_dma_stride(32768, "src").unwrap_err().contains("C8"));
    }

    #[test]
    fn c9_a_cube_tile_must_fit_l0a_l0b_and_l0c_at_once() {
        let hw = HardwareParams::ascend_910b();
        // The shipped 910B tiling: L0A and L0B both full, L0C full.
        assert!(hw.check_cube_tile(128, 256, 128, 2).is_ok());
        // Fills L0B and starves L0A. It FITS -- C9 is a capacity bound, not a
        // performance one -- and it was the slowest of four measured.
        assert!(hw.check_cube_tile(64, 512, 64, 2).is_ok());
        // 512x512x128 runs out of L0A first: 512*128*2 = 131072B of a 65536B
        // buffer. The checker names the buffer that actually overflows.
        let a = hw.check_cube_tile(512, 512, 128, 2).unwrap_err();
        assert!(a.contains("C9") && a.contains("L0A"), "{a}");
        // A shape that fits both operand buffers and still cannot accumulate:
        // k=16 keeps L0A and L0B at 4096B each, while m*n*4 = 262144B is twice
        // what L0C holds.
        let c = hw.check_cube_tile(256, 256, 16, 2).unwrap_err();
        assert!(c.contains("C9") && c.contains("L0C"), "{c}");
    }

    #[test]
    fn an_unset_limit_never_rejects() {
        let none = HardwareParams::default();
        assert!(none.check_repeat(1 << 20, 4).is_ok());
        assert!(none.check_dma_stride(1 << 20, "src").is_ok());
        assert!(none.check_cube_tile(4096, 4096, 4096, 4).is_ok());
        assert!(none.check_ub(1 << 30).is_ok());
    }

    #[test]
    fn hardware_params_carry_ub_size() {
        // The one asymmetry (AscendC's ub_size) rides on EmitOpts uniformly.
        let opts = EmitOpts {
            hw: HardwareParams { ub_size: 192 * 1024, ..Default::default() },
            ..Default::default()
        };
        assert_eq!(opts.hw.ub_size, 192 * 1024);
    }

    #[test]
    fn emitter_target_adapter_wraps_a_plain_fn() {
        // This is exactly how a real emitter joins: a plain
        // `fn(&str) -> Result<String, String>` lifted into the trait and keyed
        // by the name the codegen-path environment variable selects.
        fn fake_convert(mlir: &str) -> Result<String, String> {
            if mlir.is_empty() {
                return Err("empty".into());
            }
            Ok(format!("// probe\n{mlir}"))
        }
        let mut r = TargetRegistry::new();
        r.register(Box::new(targets::EmitterTarget::new("probe", "txt", fake_convert)));
        let t = r.select("probe").expect("probe target registered");
        let out = t.emit("module {}", &EmitOpts::default()).unwrap();
        assert_eq!(out.ext, "txt");
        assert!(out.source.starts_with("// probe"));
        assert!(t.emit("", &EmitOpts::default()).is_err());
    }
}
