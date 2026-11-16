// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! The binary's name is written down in more than one place, so it is asserted
//! rather than hoped over.
//!
//! This test exists because renaming the binary once broke the thing that
//! consumes it, silently: nothing in the crate's own suite built the published
//! target, so the manifest and the consumer disagreed and both looked fine. The
//! shape of that failure is general, which is why the names here are READ from
//! the files rather than repeated in the assertion.

use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    // crates/ptoas_rs -> crates -> rust -> repository root
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .expect("the repository root must be reachable")
}

fn manifest() -> String {
    std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml"))
        .expect("this crate's manifest")
}

/// Every `[[bin]] name` this crate declares.
fn declared_bins() -> Vec<String> {
    manifest()
        .lines()
        .filter_map(|l| {
            let l = l.trim();
            l.strip_prefix("name = \"")
                .and_then(|r| r.strip_suffix('"'))
                .map(str::to_string)
        })
        // the first `name =` is the package, the rest are the bins
        .skip(1)
        .collect()
}

#[test]
fn the_lift_binary_is_declared_under_the_name_the_crate_is_called() {
    let bins = declared_bins();
    assert!(
        bins.iter().any(|b| b == "ptoas-rs"),
        "no `ptoas-rs` binary is declared; found {bins:?}"
    );
}

#[test]
fn the_readme_names_every_binary_this_crate_declares() {
    // The README is the installation and usage surface. A binary it does not
    // mention is one nobody will know to run; a binary it mentions that does
    // not exist is worse.
    let readme = std::fs::read_to_string(repo_root().join("rust/README.md"))
        .expect("rust/README.md");
    for bin in declared_bins() {
        assert!(
            readme.contains(&bin),
            "the README does not mention the `{bin}` binary"
        );
    }
}

#[test]
fn the_installer_installs_every_binary_this_crate_declares() {
    // The installer is what the README's one-line curl runs. A binary it skips
    // is declared, documented, and absent after installation.
    let sh = std::fs::read_to_string(repo_root().join("rust/install.sh"))
        .expect("rust/install.sh");
    for bin in declared_bins() {
        assert!(
            sh.contains(&bin),
            "install.sh does not install the `{bin}` binary"
        );
    }
}
