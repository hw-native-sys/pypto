// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Device capability configuration for the safety checker.
//!
//! A `DeviceSpec` pins the capacities and op-level constraints the safety
//! checks in `safety.rs` use. Loaded from simple flat key=value text files
//! shipped in `devices/`, selectable by `--device <path>` on the CLI.
//!
//! The format is deliberately boring (no TOML dep) — one key per line,
//! `#` starts a comment, blank lines ignored. Integers accept `_`
//! separators and the `K`/`M` suffix for KiB/MiB.

use crate::plan::{AddressSpace, ExtDType, PtoArch};
use std::path::Path;

/// The bundled default device specs ship as files in `devices/`. Callers
/// who want to override one field should copy the default and edit it.
#[derive(Debug, Clone)]
pub struct DeviceSpec {
    pub name: String,
    pub arch: PtoArch,
    pub ub_bytes: u64,
    pub mat_bytes: u64,
    pub left_bytes: u64,
    pub right_bytes: u64,
    pub acc_bytes: u64,
    pub scaling_bytes: u64,

    /// ptoas stride-descriptor limit (see project_pto_matmul_stride_limits).
    pub matmul_outer_max: u64,
    pub matmul_row_max: u64,

    /// Empirically accepted (acc, lhs, rhs) triples for pto.tmatmul.
    /// Source: project_pto_tmatmul_dtype_rules (CANN 8.5 on-device probes).
    pub tmatmul_dtypes: Vec<(ExtDType, ExtDType, ExtDType)>,
}

impl DeviceSpec {
    pub fn capacity_of(&self, space: AddressSpace) -> Option<u64> {
        match space {
            AddressSpace::Ub => Some(self.ub_bytes),
            AddressSpace::Mat => Some(self.mat_bytes),
            AddressSpace::Left => Some(self.left_bytes),
            AddressSpace::Right => Some(self.right_bytes),
            AddressSpace::Acc => Some(self.acc_bytes),
            AddressSpace::Scaling => Some(self.scaling_bytes),
            AddressSpace::Gm => None,
        }
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let p = path.as_ref();
        let src = std::fs::read_to_string(p).map_err(|e| format!("read {}: {e}", p.display()))?;
        Self::parse(&src, &p.display().to_string())
    }

    pub fn parse(src: &str, origin: &str) -> Result<Self, String> {
        let mut name = None;
        let mut arch = None;
        let mut ub = None;
        let mut mat = None;
        let mut left = None;
        let mut right = None;
        let mut acc = None;
        let mut scaling = None;
        let mut outer_max = None;
        let mut row_max = None;
        let mut tmatmul_dtypes: Vec<(ExtDType, ExtDType, ExtDType)> = Vec::new();

        for (lineno, raw) in src.lines().enumerate() {
            let line = strip_comment(raw).trim();
            if line.is_empty() {
                continue;
            }
            let (k, v) = match line.split_once('=') {
                Some((a, b)) => (a.trim(), b.trim()),
                None => {
                    return Err(format!(
                        "{origin}:{}: expected key=value, got `{line}`",
                        lineno + 1
                    ))
                }
            };
            match k {
                "name" => name = Some(v.trim_matches('"').to_string()),
                "arch" => {
                    arch = PtoArch::parse(v).or_else(|| {
                        eprintln!("{origin}:{}: warn: unknown arch `{v}`", lineno + 1);
                        None
                    })
                }
                "ub_bytes" => ub = Some(parse_bytes(v, origin, lineno + 1)?),
                "mat_bytes" => mat = Some(parse_bytes(v, origin, lineno + 1)?),
                "left_bytes" => left = Some(parse_bytes(v, origin, lineno + 1)?),
                "right_bytes" => right = Some(parse_bytes(v, origin, lineno + 1)?),
                "acc_bytes" => acc = Some(parse_bytes(v, origin, lineno + 1)?),
                "scaling_bytes" => scaling = Some(parse_bytes(v, origin, lineno + 1)?),
                "matmul_outer_max" => outer_max = Some(parse_bytes(v, origin, lineno + 1)?),
                "matmul_row_max" => row_max = Some(parse_bytes(v, origin, lineno + 1)?),
                "tmatmul_dtype" => {
                    tmatmul_dtypes.push(parse_tmatmul_triple(v, origin, lineno + 1)?)
                }
                other => return Err(format!("{origin}:{}: unknown key `{other}`", lineno + 1)),
            }
        }

        Ok(DeviceSpec {
            name: name.ok_or_else(|| format!("{origin}: missing `name`"))?,
            arch: arch.ok_or_else(|| format!("{origin}: missing/invalid `arch`"))?,
            ub_bytes: ub.ok_or_else(|| format!("{origin}: missing `ub_bytes`"))?,
            mat_bytes: mat.ok_or_else(|| format!("{origin}: missing `mat_bytes`"))?,
            left_bytes: left.ok_or_else(|| format!("{origin}: missing `left_bytes`"))?,
            right_bytes: right.ok_or_else(|| format!("{origin}: missing `right_bytes`"))?,
            acc_bytes: acc.ok_or_else(|| format!("{origin}: missing `acc_bytes`"))?,
            scaling_bytes: scaling.ok_or_else(|| format!("{origin}: missing `scaling_bytes`"))?,
            matmul_outer_max: outer_max
                .ok_or_else(|| format!("{origin}: missing `matmul_outer_max`"))?,
            matmul_row_max: row_max.ok_or_else(|| format!("{origin}: missing `matmul_row_max`"))?,
            tmatmul_dtypes,
        })
    }
}

fn strip_comment(s: &str) -> &str {
    match s.find('#') {
        Some(i) => &s[..i],
        None => s,
    }
}

fn parse_bytes(v: &str, origin: &str, lineno: usize) -> Result<u64, String> {
    let (digits, scale) = if let Some(rest) = v.strip_suffix('K') {
        (rest, 1024u64)
    } else if let Some(rest) = v.strip_suffix('M') {
        (rest, 1024 * 1024)
    } else {
        (v, 1)
    };
    let digits = digits.trim().replace('_', "");
    digits
        .parse::<u64>()
        .map(|n| n * scale)
        .map_err(|e| format!("{origin}:{lineno}: bad integer `{v}`: {e}"))
}

fn parse_tmatmul_triple(
    v: &str,
    origin: &str,
    lineno: usize,
) -> Result<(ExtDType, ExtDType, ExtDType), String> {
    // Format: "(acc, lhs, rhs)" — e.g. `(i32, i8, i8)`
    let inner = v.trim().trim_start_matches('(').trim_end_matches(')');
    let parts: Vec<_> = inner.split(',').map(|s| s.trim()).collect();
    if parts.len() != 3 {
        return Err(format!("{origin}:{lineno}: expected 3 dtypes, got `{v}`"));
    }
    let parse_one = |p: &str| {
        ExtDType::parse(p).ok_or_else(|| format!("{origin}:{lineno}: unknown dtype `{p}`"))
    };
    Ok((
        parse_one(parts[0])?,
        parse_one(parts[1])?,
        parse_one(parts[2])?,
    ))
}

/// Bundled default for Ascend 910B2 on CANN 8.5. Used when the CLI is
/// invoked without `--device` and the input's `pto.target_arch` is `a5`.
pub fn default_a5_910b2_cann85() -> DeviceSpec {
    DeviceSpec::parse(
        DEFAULT_A5_910B2_CANN85,
        "<embedded default a5_910b2_cann85>",
    )
    .expect("embedded default a5 spec must parse")
}

/// Baked-in copy of `devices/a5_910b2_cann85.spec` — kept in sync with
/// that file so the CLI works without a filesystem lookup.
const DEFAULT_A5_910B2_CANN85: &str = r#"
# Ascend 910B2, CANN 8.5 — default a5 target.
# Source: CANN 8.5 headers + on-device probes recorded in memorybank
# (project_pto_matmul_stride_limits, project_pto_tmatmul_dtype_rules).
name = "Ascend910B2 (CANN 8.5)"
arch = a5

ub_bytes      = 192K
mat_bytes     = 512K
left_bytes    = 64K
right_bytes   = 64K
acc_bytes     = 256K
scaling_bytes = 4K

# ptoas descriptor limits: OUTER < 2^24 and ROW (= physical N) < 2^16.
matmul_outer_max = 16_777_216
matmul_row_max   = 65_536

# Empirically accepted (acc, lhs, rhs) triples on CANN 8.5.
# Mixed f16+f32-acc IS supported; all-f16 acc is NOT.
tmatmul_dtype = (i32, i8, i8)
tmatmul_dtype = (f32, f16, f16)
tmatmul_dtype = (f32, bf16, bf16)
tmatmul_dtype = (f32, f32, f32)
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_spec_parses() {
        let s = default_a5_910b2_cann85();
        assert_eq!(s.arch, PtoArch::A5);
        assert_eq!(s.ub_bytes, 192 * 1024);
        assert_eq!(s.left_bytes, 64 * 1024);
        assert_eq!(s.matmul_row_max, 65_536);
        assert!(s
            .tmatmul_dtypes
            .contains(&(ExtDType::I32, ExtDType::I8, ExtDType::I8)));
        assert!(s
            .tmatmul_dtypes
            .contains(&(ExtDType::F32, ExtDType::F16, ExtDType::F16)));
    }

    #[test]
    fn bytes_suffixes() {
        assert_eq!(parse_bytes("192K", "x", 1).unwrap(), 192 * 1024);
        assert_eq!(parse_bytes("1M", "x", 1).unwrap(), 1024 * 1024);
        assert_eq!(parse_bytes("16_777_216", "x", 1).unwrap(), 16_777_216);
        assert_eq!(parse_bytes("42", "x", 1).unwrap(), 42);
    }

    #[test]
    fn rejects_unknown_key() {
        let bad = "name = \"x\"\narch = a5\nub_bytes = 1\nmat_bytes = 1\nleft_bytes = 1\nright_bytes = 1\nacc_bytes = 1\nscaling_bytes = 1\nmatmul_outer_max = 1\nmatmul_row_max = 1\nfoo_bar = 42\n";
        let err = DeviceSpec::parse(bad, "t").unwrap_err();
        assert!(err.contains("unknown key"), "got: {err}");
    }
}
