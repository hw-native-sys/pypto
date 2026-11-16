// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! The hardware+toolchain fingerprint (decision 2): `{soc_arch, soc_name, cann_version,
//! ptoas_version}`. ELFs need a full match to load; configs match on `soc_arch` alone.
//!
//! Gathered best-effort so the CLI is useful off-box: the SoC comes from the memory profile's
//! chip name when available, arch from `engine_ascend::soc`, and the toolchain versions from the
//! environment (the CANN path, a pinned ptoas), each overridable so a planning run on a login
//! node can still name the target it is sizing for.

use crate::registry::sha256_hex;

#[derive(Debug, Clone)]
pub struct Fingerprint {
    pub soc_arch: String,
    pub soc_name: String,
    pub cann_version: String,
    pub ptoas_version: String,
}

impl Fingerprint {
    /// The full-match hash — what an ELF is keyed by.
    pub fn full(&self) -> String {
        let joined = format!(
            "{}|{}|{}|{}",
            self.soc_arch, self.soc_name, self.cann_version, self.ptoas_version
        );
        sha256_hex(joined.as_bytes())[..16].to_string()
    }

    /// Human note stored beside the hash, e.g. `Ascend910B2 (a3) / CANN 9.0.0 / ptoas 0.54`.
    pub fn note(&self) -> String {
        format!(
            "{} ({}) / CANN {} / ptoas {}",
            self.soc_name, self.soc_arch, self.cann_version, self.ptoas_version
        )
    }
}

/// CANN version from `ASCEND_HOME_PATH`/`DS4_CANN_VERSION` or the conventional path.
fn cann_version() -> String {
    if let Ok(v) = std::env::var("DS4_CANN_VERSION") {
        return v;
    }
    for var in ["ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"] {
        if let Ok(p) = std::env::var(var) {
            // .../cann-9.0.0/... -> 9.0.0
            if let Some(seg) = p.split('/').find(|s| s.starts_with("cann-")) {
                return seg.trim_start_matches("cann-").to_string();
            }
        }
    }
    "unknown".to_string()
}

fn ptoas_version() -> String {
    std::env::var("DS4_PTOAS_VERSION").unwrap_or_else(|_| "0.54".to_string())
}

/// Derive the fingerprint. `soc_name` is passed in from the memory profile (npu-smi chip name)
/// when known, or `None` to read `DS4_SOC`/`ACLRS_SOC_VERSION`.
pub fn derive(soc_name: Option<&str>) -> Fingerprint {
    let name = soc_name
        .map(String::from)
        .or_else(|| std::env::var("DS4_SOC").ok())
        .or_else(|| std::env::var("ACLRS_SOC_VERSION").ok())
        .unwrap_or_else(|| "unknown".to_string());
    let arch = engine_ascend::soc::arch_of_soc(&name)
        .map(|a| format!("{a:?}").to_lowercase())
        .unwrap_or_else(|| "unknown".to_string());
    Fingerprint {
        soc_arch: arch,
        soc_name: name,
        cann_version: cann_version(),
        ptoas_version: ptoas_version(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_hash_is_stable_and_arch_is_a_prefix_view() {
        std::env::set_var("DS4_CANN_VERSION", "9.0.0");
        std::env::set_var("DS4_PTOAS_VERSION", "0.54");
        let fp = derive(Some("Ascend910B2"));
        assert_eq!(fp.cann_version, "9.0.0");
        assert_eq!(fp.full().len(), 16);
        // Same inputs -> same hash; a CANN change -> different hash (ELF-invalidating).
        let a = derive(Some("Ascend910B2")).full();
        std::env::set_var("DS4_CANN_VERSION", "9.1.0");
        let b = derive(Some("Ascend910B2")).full();
        assert_ne!(a, b, "a CANN bump must change the ELF fingerprint");
        std::env::remove_var("DS4_CANN_VERSION");
        std::env::remove_var("DS4_PTOAS_VERSION");
    }

    #[test]
    fn note_is_human_readable() {
        let fp = derive(Some("Ascend910B2"));
        assert!(fp.note().contains("Ascend910B2"));
    }
}
