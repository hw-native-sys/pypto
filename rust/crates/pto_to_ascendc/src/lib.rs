// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! `pto-to-ascendc` -- emit AscendC C++ from a kernel, to compare representations.
//!
//! ```text
//! kernel.tile ──→ ascendc_emit ──→ kernel.cpp
//! ```
//!
//! The purpose is evaluation. A kernel exists here as tile-rs source, as
//! tile-dialect MLIR, as PTO and as AscendC, and the useful question is what each
//! representation makes easy to see. Having the AscendC beside the PTO is what
//! makes that question answerable rather than rhetorical.
//!
//! # On the name, and what it does not do yet
//!
//! `.pto` input is the direction this tool is named for, and it is **not**
//! implemented here. There are two routes to it and neither is in this tree yet:
//!
//! * the pinned `ptoas` assembler, which performs PTO to AscendC directly and is
//!   the reference answer; and
//! * lifting the PTO back to tile-rs and lowering that, which needs the lift and
//!   the Rust input mode.
//!
//! Having both is the point rather than a redundancy: the two outputs can be
//! diffed, and that diff is a stronger statement about either route than either
//! one alone. Until then `.pto` is recognised and refused with exit code 4.

use tile_codegen::registry::TargetRegistry;
use tile_codegen::target::{EmitOpts, HardwareParams, TargetSemantics};

/// Which Ascend SoC the kernel is being emitted for.
///
/// Not optional, and with no sensible blank default -- which is the one real
/// difference from the PTO tool. `EmitOpts::default()` carries `ub_size: 0` on
/// purpose: it describes a target with no unified buffer, where every budget
/// check correctly answers Ok. Handing that to this emitter turns a missing
/// hardware parameter into an overflow on every kernel -- measured on the
/// 16-kernel bench, 0 of 16 emitted, with `vec_add` "overflowing" at 1024 bytes.
/// So the default here is a real part rather than an empty one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Soc {
    /// 910B. 192 KiB unified buffer, measured.
    A910b,
    /// 950, capacities not yet measured -- the emitter refuses what it cannot bound.
    A950,
    /// 950PR, unmeasured.
    A950pr,
    /// 950DT, unmeasured.
    A950dt,
}

impl Soc {
    pub const ALL: [Soc; 4] = [Soc::A910b, Soc::A950, Soc::A950pr, Soc::A950dt];

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "910b" => Some(Soc::A910b),
            "950" => Some(Soc::A950),
            "950pr" => Some(Soc::A950pr),
            "950dt" => Some(Soc::A950dt),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Soc::A910b => "910b",
            Soc::A950 => "950",
            Soc::A950pr => "950pr",
            Soc::A950dt => "950dt",
        }
    }

    /// The hardware parameters and the instruction semantics this part implies,
    /// returned together so they cannot be mismatched: arch32 semantics belong
    /// with 910B, arch35 with the 950 family, and they are different
    /// instruction sets rather than two sizes of one.
    pub fn emit_opts(self) -> EmitOpts {
        let (hw, sem) = match self {
            Soc::A910b => (
                HardwareParams::ascend_910b(),
                TargetSemantics::ascend_arch32(),
            ),
            Soc::A950 => (
                HardwareParams::ascend_950_unmeasured(),
                TargetSemantics::ascend_arch35_unmeasured(),
            ),
            Soc::A950pr => (
                HardwareParams::ascend_950pr_unmeasured(),
                TargetSemantics::ascend_arch35_unmeasured(),
            ),
            Soc::A950dt => (
                HardwareParams::ascend_950dt_unmeasured(),
                TargetSemantics::ascend_arch35_unmeasured(),
            ),
        };
        EmitOpts { hw, sem }
    }
}

#[derive(Debug)]
pub enum Error {
    /// The emitter rejected the MLIR.
    Emit(String),
    Io(std::io::Error),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Emit(s) => write!(f, "{s}"),
            Error::Io(e) => write!(f, "{e}"),
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

/// MLIR text in, AscendC C++ out.
///
/// Goes through the registry rather than calling the emitter directly, so this
/// exercises the same `CodegenTarget` dispatch any other consumer would.
pub fn mlir_to_ascendc(mlir: &str, soc: Soc) -> Result<String, Error> {
    if mlir.trim().is_empty() {
        return Err(Error::Emit("the MLIR input is empty".into()));
    }
    let mut r = TargetRegistry::new();
    ascendc_emit::register(&mut r);
    let target = r
        .select("cpp")
        .ok_or_else(|| Error::Emit("the cpp target did not register".into()))?;
    let out = target
        .emit(mlir, &soc.emit_opts())
        .map_err(Error::Emit)?;
    Ok(out.source)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pinned here because it is the one place the reason for `--soc` is
    /// unambiguous: the blank default describes a part with no memory.
    #[test]
    fn the_blank_default_has_no_unified_buffer_and_every_soc_has_one() {
        assert_eq!(EmitOpts::default().hw.ub_size, 0);
        assert_eq!(Soc::A910b.emit_opts().hw.ub_size, 192 * 1024);
        for s in Soc::ALL {
            assert!(s.emit_opts().hw.ub_size > 0, "{} has no budget", s.name());
        }
    }

    #[test]
    fn every_soc_name_round_trips() {
        for s in Soc::ALL {
            assert_eq!(Soc::parse(s.name()), Some(s));
        }
        assert_eq!(Soc::parse("910c"), None);
    }

    #[test]
    fn empty_mlir_is_rejected_with_a_clear_message() {
        let e = mlir_to_ascendc("  \n ", Soc::A910b).unwrap_err();
        assert!(format!("{e}").contains("empty"), "{e}");
    }
}
