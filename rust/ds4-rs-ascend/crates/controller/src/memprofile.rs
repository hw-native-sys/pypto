// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Where the box's free HBM comes from.
//!
//! Two sources, in order of truth:
//!   1. `aclrtGetMemInfo` per device, under the `ascend` feature — the number the engine will
//!      actually allocate against. This is authoritative and is what `plan` uses on-box.
//!   2. Parsing `npu-smi info`, which needs no CANN linkage and works from any shell. This is
//!      the planning/what-if source: a user can size a deployment on a login node, or read a
//!      box the CLI was not built with the feature for.
//!
//! Both yield the same shape — per-chip `(id, free_bytes, total_bytes)` — so the planner does
//! not care which produced it, and `--mem-source` lets a user force one.

use anyhow::{bail, Context, Result};

/// One accelerator's memory, in bytes.
#[derive(Debug, Clone, Copy)]
pub struct ChipMem {
    /// Logical index in enumeration order — what the planner keys chips by.
    pub id: u32,
    pub free_bytes: u64,
    pub total_bytes: u64,
}

impl ChipMem {
    pub fn used_bytes(&self) -> u64 {
        self.total_bytes.saturating_sub(self.free_bytes)
    }
    pub fn busy_fraction(&self) -> f64 {
        if self.total_bytes == 0 {
            0.0
        } else {
            self.used_bytes() as f64 / self.total_bytes as f64
        }
    }
}

/// Which source to read the profile from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemSource {
    /// Prefer `aclrtGetMemInfo` (feature `ascend`), fall back to `npu-smi`.
    Auto,
    /// Force the `npu-smi info` parse.
    NpuSmi,
    /// Force the ACL query — errors if the feature is not compiled in.
    Acl,
}

/// Read the memory profile of every chip on this box.
pub fn read_profile(source: MemSource) -> Result<Vec<ChipMem>> {
    match source {
        MemSource::Acl => acl_profile().context("ACL memory query"),
        MemSource::NpuSmi => npu_smi_profile(),
        MemSource::Auto => match acl_profile() {
            Ok(v) if !v.is_empty() => Ok(v),
            _ => npu_smi_profile(),
        },
    }
}

/// The `aclrtGetMemInfo` path — the authoritative free-HBM number.
#[cfg(feature = "ascend")]
fn acl_profile() -> Result<Vec<ChipMem>> {
    Ok(engine_ascend::device::acl_memory_profile()?
        .into_iter()
        .map(|(id, free, total)| ChipMem { id, free_bytes: free, total_bytes: total })
        .collect())
}

#[cfg(not(feature = "ascend"))]
fn acl_profile() -> Result<Vec<ChipMem>> {
    bail!("this build has no ACL linkage (rebuild with --features ascend to use --mem-source acl)")
}

/// Run `npu-smi info` and parse its per-chip HBM usage.
fn npu_smi_profile() -> Result<Vec<ChipMem>> {
    let out = std::process::Command::new("npu-smi")
        .arg("info")
        .output()
        .context("running `npu-smi info` — is the Ascend driver installed and on PATH?")?;
    if !out.status.success() {
        bail!(
            "npu-smi info failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let chips = parse_npu_smi(&text);
    if chips.is_empty() {
        bail!("npu-smi info produced no HBM rows this parser recognised:\n{text}");
    }
    Ok(chips)
}

/// Parse the HBM `used / total` pairs out of `npu-smi info`.
///
/// The table prints two rows per chip; the HBM-Usage column (`used / total` MB) is on the
/// SECOND row, e.g. `60760/ 65536`. On-chip Memory-Usage prints as small `a / b` pairs (often
/// `0 / 0`), so an HBM row is distinguished by a total at least [`MIN_HBM_MB`] — no positional
/// column counting, which the various npu-smi layouts (910/910B2/910C) format differently.
///
/// Separated from the process spawn so the exact device text captured on each box can be a unit
/// test, which is the only way to keep this robust across driver versions without a device.
pub fn parse_npu_smi(text: &str) -> Vec<ChipMem> {
    /// A total this large is HBM, not the on-chip Memory-Usage column.
    const MIN_HBM_MB: u64 = 4096;
    let mut chips = Vec::new();
    for line in text.lines() {
        // Consider only the numeric HBM pair with the largest total on this line, so a row
        // carrying both `0 / 0` (memory) and `60760 / 65536` (HBM) yields the HBM one.
        let mut best: Option<(u64, u64)> = None;
        let bytes = line.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i].is_ascii_digit() {
                let start = i;
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                }
                let used: u64 = line[start..i].parse().unwrap_or(0);
                // Skip spaces then require a '/'.
                let mut j = i;
                while j < bytes.len() && bytes[j] == b' ' {
                    j += 1;
                }
                if j < bytes.len() && bytes[j] == b'/' {
                    j += 1;
                    while j < bytes.len() && bytes[j] == b' ' {
                        j += 1;
                    }
                    let tstart = j;
                    while j < bytes.len() && bytes[j].is_ascii_digit() {
                        j += 1;
                    }
                    if j > tstart {
                        let total: u64 = line[tstart..j].parse().unwrap_or(0);
                        if total >= MIN_HBM_MB
                            && used <= total
                            && best.map_or(true, |(_, bt)| total > bt)
                        {
                            best = Some((used, total));
                        }
                    }
                    i = j;
                    continue;
                }
            }
            i += 1;
        }
        if let Some((used, total)) = best {
            let mb = 1024u64 * 1024;
            chips.push(ChipMem {
                id: chips.len() as u32,
                free_bytes: (total - used) * mb,
                total_bytes: total * mb,
            });
        }
    }
    chips
}

#[cfg(test)]
mod tests {
    use super::*;

    // Captured from the customer 910B2 (8x64 GiB) under its vLLM job.
    const B2: &str = "\
+------------------------------------------------------------------------------+
| NPU   Name       | Health | Power(W)   AICore(%)  Memory  HBM-Usage(MB)      |
| 0     910B2      | OK     | 95.3       51         0 / 0                       |
| 0                | busid  | 0          0 / 0      60760/ 65536                |
| 1     910B2      | OK     | 95.1       50         0 / 0                       |
| 0                | busid  | 0          0 / 0      60758/ 65536                |
";
    // Captured from an idle 910C.
    const C: &str = "\
| 0     Ascend910  | OK     | 172.2      34         0 / 0                       |
| 0     0          | busid  | 0          0 / 0      3223 / 65536                |
";

    #[test]
    fn parses_910b2_busy_box() {
        let v = parse_npu_smi(B2);
        assert_eq!(v.len(), 2, "two chips");
        assert_eq!(v[0].total_bytes, 65536 * 1024 * 1024);
        assert_eq!(v[0].free_bytes, (65536 - 60760) * 1024 * 1024);
        assert!(v[0].busy_fraction() > 0.9, "customer box is nearly full");
    }

    #[test]
    fn parses_910c_idle_box() {
        let v = parse_npu_smi(C);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].free_bytes, (65536 - 3223) * 1024 * 1024);
    }

    #[test]
    fn ignores_the_memory_usage_pair_on_the_same_row() {
        // A row with both `0 / 0` and the real HBM pair must yield the HBM one.
        let v = parse_npu_smi("| 0 | busid | 0  0 / 0  60760/ 65536 |");
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].total_bytes, 65536 * 1024 * 1024);
    }

    #[test]
    fn no_hbm_rows_is_empty_not_a_panic() {
        assert!(parse_npu_smi("garbage\n| header only |").is_empty());
    }
}
