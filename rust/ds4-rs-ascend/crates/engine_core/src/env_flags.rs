// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! One place to decide what a DS4_* boolean env flag MEANS.
//!
//! This repo's dominant convention is value-aware — `DS4_X=0` disables — and the docs,
//! scripts and comments are written in that dialect throughout (`DS4_CHAIN=0`,
//! `DS4_KABS_CACHE=0`, `DS4_MLA_LATENT=0`, …). But a minority of sites used the
//! presence-only idiom (`std::env::var("X").is_ok()`), which reads *backwards* to anyone
//! who has internalised the convention: `DS4_X=0` turns the thing ON.
//!
//! That is not hypothetical. `DS4_SSD_STREAM` was read presence-only at seven behavioural
//! sites across two crates, so `DS4_SSD_STREAM=0` — the obvious way to say "don't stream,
//! I have the RAM" — put the engine into SSD-streaming mode and collapsed decode to
//! SSD-bound (~0.02 tok/s instead of ~22). `gguf.rs` even printed "set DS4_SSD_STREAM=1 to
//! stream from SSD", advertising a value semantics the code did not honour. The same shape
//! was fixed once already on the Metal prefill flag (commit c40535166).
//!
//! Use [`flag_enabled`] for any flag whose name implies a boolean and whose value changes
//! BEHAVIOUR. Presence-only is legitimate in exactly one case: a flag that means "the
//! caller made an explicit choice, whatever the value" (see `ds4_app_ffi`'s
//! `pertoken_explicit`) — express that with `std::env::var_os(..).is_some()` and say so in
//! a comment, so the reader knows it is deliberate rather than the old bug.

/// Resolve a tuning knob from the environment.
///
/// The environment is the only source, so an operator always has the last word
/// on every knob. A signed-policy layer used to sit behind it as a second
/// source of defaults; see the note in `lib.rs` for why it is gone.
pub fn var(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

/// Numeric knob with a fallback, from the same source.
///
/// An unparsable value falls back rather than panicking: a malformed
/// number must never be able to take the engine down.
pub fn var_parsed<T: std::str::FromStr>(name: &str, default: T) -> T {
    var(name)
        .and_then(|v| v.trim().parse::<T>().ok())
        .unwrap_or(default)
}

/// Is a boolean flag enabled?
///
/// - unset            → `default`
/// - `0` / `false` / `no` (any case, surrounding space ignored) → `false`
/// - set to empty (`env X= cmd`) → `true` (an explicit, if unusual, opt-in)
/// - anything else (`1`, `on`, `yes`, …) → `true`
///
/// Reads through [`var`], so a signed policy can move the default while an
/// explicit env var still overrides it.
pub fn flag_enabled(name: &str, default: bool) -> bool {
    match var(name).ok_or(()) {
        Err(_) => default,
        Ok(v) => {
            let v = v.trim();
            if v.is_empty() {
                return true;
            }
            !(v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("no"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{flag_enabled, var, var_parsed};

    /// The environment is the only source, so an operator always has the last
    /// word. The previous version of this module had a second source behind the
    /// environment -- a signed policy file -- and these tests pinned the
    /// precedence between them. That source is gone (see `lib.rs`), so testing
    /// a precedence rule here would be testing a rule that no longer exists.
    #[test]
    fn an_unset_variable_reads_as_none() {
        let name = "DS4_TEST_DEFINITELY_UNSET_KNOB";
        std::env::remove_var(name);
        assert_eq!(var(name), None);
        assert!(!flag_enabled(name, false));
        // and the default is honoured when unset, which is the whole point.
        assert!(flag_enabled(name, true));
    }

    #[test]
    fn a_set_variable_is_read_back() {
        let name = "DS4_TEST_SET_KNOB";
        std::env::set_var(name, "7");
        assert_eq!(var(name).as_deref(), Some("7"));
        assert_eq!(var_parsed::<u32>(name, 3u32), 7);
        std::env::remove_var(name);
    }

    /// An unparsable value falls back rather than panicking. A malformed knob
    /// must never be able to take the engine down, which is the one property
    /// here worth more than a type signature.
    #[test]
    fn an_unparsable_value_falls_back_instead_of_panicking() {
        let name = "DS4_TEST_BAD_NUMBER";
        std::env::set_var(name, "not-a-number");
        assert_eq!(var_parsed::<u32>(name, 11u32), 11);
        std::env::remove_var(name);
    }
}
