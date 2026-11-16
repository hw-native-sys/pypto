// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Which Ascend generation we are built for, and a guard against running on
//! the wrong one.
//!
//! Two boxes are in play and they are NOT the same silicon:
//!
//! | box  | `aclrtGetSocName` | generation |
//! |------|-------------------|------------|
//! | 910c | `Ascend910_9392`  | a3         |
//! | 910b | `Ascend910B2`     | a2         |
//!
//! # What actually differs — measured, not assumed
//!
//! For every kernel this project generates, `ptoas --pto-arch=a2` and
//! `--pto-arch=a3` emit **byte-identical AscendC**: checked on the MXFP4
//! matvec, the composed MXFP4 layer, and blocked attention at S=2048/D=512.
//! The pinned pto-isa even ships one shared `a2a3` header directory. `a5`
//! genuinely differs, which is the control that makes the comparison mean
//! something.
//!
//! So the gate is deliberately thin. It exists to (1) pick the right flag when
//! building and (2) REFUSE at startup if the binary meets silicon it was not
//! built for — not to maintain two code paths that are currently the same
//! bytes. If a future kernel does diverge, `a2 == a3` stops being true and the
//! test that pins it fails, which is the point.

/// Ascend generation, as `ptoas --pto-arch` names them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    A2,
    A3,
    A5,
}

impl Arch {
    /// The `--pto-arch` flag value.
    pub fn pto_flag(self) -> &'static str {
        match self {
            Arch::A2 => "a2",
            Arch::A3 => "a3",
            Arch::A5 => "a5",
        }
    }

    /// The `ccec --cce-aicore-arch` value.
    ///
    /// a2 and a3 are both `dav-c220`; only the vector variant is used, since
    /// every kernel we emit today is vector-only (no cube). VERIFIED for a3 on
    /// the 910c; a2 is inferred from the shared core name and is NOT yet
    /// confirmed on 910b silicon.
    pub fn cce_arch(self) -> &'static str {
        match self {
            Arch::A2 | Arch::A3 => "dav-c220-vec",
            Arch::A5 => "dav-c310-vec",
        }
    }

    /// True when the two generations emit the same AscendC for our kernels.
    /// Measured today for a2/a3; kept as a predicate so the claim is one place.
    pub fn emits_same_as(self, other: Arch) -> bool {
        matches!(
            (self, other),
            (Arch::A2, Arch::A2)
                | (Arch::A3, Arch::A3)
                | (Arch::A5, Arch::A5)
                | (Arch::A2, Arch::A3)
                | (Arch::A3, Arch::A2)
        )
    }
}

/// Map a runtime `aclrtGetSocName` string to a generation.
///
/// Matching is prefix-based on the family rather than exact, because the SoC
/// string carries a board suffix (`Ascend910_9392`) that varies per part.
pub fn arch_of_soc(soc: &str) -> Option<Arch> {
    let s = soc.trim();
    if s.starts_with("Ascend910B") {
        Some(Arch::A2)
    } else if s.starts_with("Ascend910_") || s.eq_ignore_ascii_case("Ascend910C") {
        Some(Arch::A3)
    } else if s.starts_with("Ascend910_95") || s.starts_with("Ascend950") {
        Some(Arch::A5)
    } else {
        None
    }
}

/// The generation this binary's kernels were built for.
///
/// `DS4_ASCEND_ARCH` overrides at build time; the default is a3, matching
/// `ptoas`' own default and the box every kernel has been verified on.
pub fn built_for() -> Arch {
    match option_env!("DS4_ASCEND_ARCH") {
        Some("a2") => Arch::A2,
        Some("a5") => Arch::A5,
        _ => Arch::A3,
    }
}

/// Refuse to run kernels on silicon they were not built for.
///
/// Returns `Ok` when the generations match, or when they are known to emit the
/// same code — a3-built kernels on a2 silicon is currently fine and saying so
/// avoids a pointless rebuild. Anything else is an error rather than a warning:
/// a wrong-generation kernel does not fail loudly on Ascend, it produces
/// wrong numbers, and this session already spent a day on one of those.
pub fn check(soc: &str) -> Result<Arch, String> {
    let found = arch_of_soc(soc).ok_or_else(|| {
        format!("engine_ascend: unrecognised SoC {soc:?} — refusing to guess an architecture")
    })?;
    let built = built_for();
    if found == built {
        return Ok(found);
    }
    if built.emits_same_as(found) {
        return Ok(found);
    }
    Err(format!(
        "engine_ascend: built for {} but this device is {} ({soc}). \
         These generations do not emit the same code; rebuild with \
         DS4_ASCEND_ARCH={}.",
        built.pto_flag(),
        found.pto_flag(),
        found.pto_flag()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two real boxes, by the string their driver actually returns.
    #[test]
    fn the_two_boxes_map_correctly() {
        assert_eq!(arch_of_soc("Ascend910_9392"), Some(Arch::A3), "910c");
        assert_eq!(arch_of_soc("Ascend910B2"), Some(Arch::A2), "910b");
        assert_eq!(arch_of_soc("Ascend910B1"), Some(Arch::A2));
    }

    /// MEASURED: ptoas emits byte-identical AscendC for a2 and a3 across the
    /// matvec, the composed layer and blocked attention at 2048x512. a5 does
    /// not, which is the control.
    #[test]
    fn a2_and_a3_are_interchangeable_but_a5_is_not() {
        assert!(Arch::A3.emits_same_as(Arch::A2));
        assert!(Arch::A2.emits_same_as(Arch::A3));
        assert!(!Arch::A3.emits_same_as(Arch::A5));
        assert!(!Arch::A5.emits_same_as(Arch::A2));
    }

    /// The point of the gate: an a3 build runs on the 910b without a rebuild,
    /// because the emitted code is the same bytes.
    #[test]
    fn a3_build_accepts_910b_silicon() {
        assert_eq!(check("Ascend910B2"), Ok(Arch::A2));
        assert_eq!(check("Ascend910_9392"), Ok(Arch::A3));
    }

    /// An unknown SoC is refused rather than guessed. Guessing an architecture
    /// on Ascend does not fail loudly — it produces wrong numbers.
    #[test]
    fn an_unknown_soc_is_refused_not_guessed() {
        let e = check("Ascend999Z").unwrap_err();
        assert!(e.contains("refusing to guess"), "{e}");
        assert!(arch_of_soc("").is_none());
    }

    #[test]
    fn build_flags_are_the_ones_the_toolchain_accepts() {
        // ptoas --pto-arch=<a2|a3|a5>, from its own --help
        for a in [Arch::A2, Arch::A3, Arch::A5] {
            assert!(["a2", "a3", "a5"].contains(&a.pto_flag()));
        }
        assert_eq!(Arch::A3.cce_arch(), "dav-c220-vec", "verified on 910c");
        assert_eq!(Arch::A2.cce_arch(), Arch::A3.cce_arch(), "same core");
    }
}
