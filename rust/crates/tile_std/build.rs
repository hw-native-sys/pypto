// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Detect the rustc nightly's commit date and emit `cfg` flags so the
//! `#![no_core]` surface can track compiler-internal renames while still
//! building on the pinned `nightly-2025-08-04`.
//!
//! One gate per compiler change. Each date is the first commit date at which the
//! new spelling is required, measured with
//! `rust/scripts/nightly_drift_check.py --bisect FROM TO --signature '<error text>'`,
//! and `rust/scripts/nightly_drift_check.py --only gates` checks the nightlies on both
//! sides of every gate. Add a gate the same way, then `#[cfg]` the two spellings
//! in `src/core.rs`.

use std::process::Command;

/// (cfg name, first commit date that needs the new spelling)
const GATES: &[(&str, (u32, u32, u32))] = &[
    // `#![rustc_coherence_is_core]` is rejected on modules; only the crate root
    // (lib.rs) may carry it. Bisected: 2025-08-13 accepts, 2025-08-14 rejects.
    ("rustc_coherence_root_only", (2025, 8, 14)),
    // Scalar float intrinsics (floorf32, ceilf32, roundf32, truncf32, fmaf32,
    // sqrtf32, expf32, logf32) became `safe` (the rounding ones also `const`).
    // Bisected: safety mismatches under the unsafe spelling start at 2025-09-22.
    ("rustc_float_intrinsics_safe", (2025, 9, 22)),
    // `fabsf32` and `copysignf32` became `safe const` separately. Measured: still
    // unsafe at 2025-09-23, safe at 2025-09-24.
    ("rustc_sign_intrinsics_safe", (2025, 9, 24)),
    // `#[rustc_do_not_implement_via_object]` renamed `#[rustc_dyn_incompatible_trait]`.
    // Bisected: the old name is unknown from 2026-01-21.
    ("rustc_dyn_incompatible_trait_attr", (2026, 1, 21)),
    // The `fabsf32` intrinsic was removed. It is unused here, so omitting its
    // declaration early is harmless; both sides of this date check clean.
    ("rustc_fabsf32_removed", (2026, 3, 15)),
    // `#[rustc_layout_scalar_valid_range_start]` removed (NonNull/NonZero use
    // pattern types). Still accepted on 2026-04-30; both sides of this date check
    // clean. Omitting it only loses the `Option<NonNull>` niche.
    ("rustc_layout_range_attr_removed", (2026, 5, 1)),
    // `#[lang = "drop_in_place"]` renamed `#[lang = "drop_glue"]`. Bisected: `drop_glue`
    // is unknown through 2026-05-06 and required from 2026-05-07.
    ("rustc_drop_glue_lang", (2026, 5, 7)),
    // `expf32`/`logf32` became generic `exp<T>`/`log<T>` (rust-lang/rust #162117,
    // merged 2026-09-01 15:38 UTC, inside the nightly with commit date 2026-09-01).
    // Unused here, so omitted rather than ported.
    ("rustc_float_math_generic", (2026, 9, 1)),
];

fn main() {
    let date = rustc_commit_date();
    for (name, since) in GATES {
        println!("cargo::rustc-check-cfg=cfg({name})");
        // An unknown commit date (stable rustc, or no `--version --verbose`) keeps
        // every legacy spelling, matching the pinned toolchain.
        if matches!(date, Some(d) if d >= *since) {
            println!("cargo::rustc-cfg={name}");
        }
    }
}

/// Parse `commit-date: YYYY-MM-DD` from `rustc --version --verbose`.
fn rustc_commit_date() -> Option<(u32, u32, u32)> {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let out = Command::new(&rustc)
        .args(["--version", "--verbose"])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let date = text
        .lines()
        .find_map(|l| l.strip_prefix("commit-date:"))?
        .trim();
    let mut it = date.split('-').filter_map(|s| s.trim().parse::<u32>().ok());
    Some((it.next()?, it.next()?, it.next()?))
}
