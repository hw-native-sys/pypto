// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! What this run could and could not check.
//!
//! Several gates here need tools that a GitHub-hosted runner does not have: the
//! pinned nightly for the kernel-side DSL, Miri on that nightly, and the
//! `ptoas` assembler. They SKIP rather than fail, which is right -- a tree
//! where nothing is wrong should not turn CI red -- but it leaves a green run
//! that checked eight fewer things than it appears to, and the skip notices are
//! swallowed because cargo captures a passing test's stderr.
//!
//! So this prints a summary, unconditionally. Named `zz_` so it sorts last in
//! the test-binary order and the summary lands near the end of a log rather
//! than scrolled off the top.
//!
//! It does NOT fail on a missing tool. Set `PYPTO_REQUIRE_FULL_GATES=1` to make
//! it fail instead -- for a runner that is SUPPOSED to have everything, where a
//! silently reduced run is the thing worth catching.

mod shared;

#[test]
fn report_which_optional_gates_this_run_could_execute() {
    let tc = shared::kernel_toolchain();
    let rows = [
        ("kernel toolchain", tc.as_str(), shared::have_kernel_toolchain()),
        ("miri", tc.as_str(), shared::have_miri()),
        ("ptoas assembler", "$PATH or $PTOAS", shared::have_ptoas()),
    ];
    eprintln!("\n--- optional gate prerequisites ---");
    for (what, where_, present) in &rows {
        eprintln!(
            "  {:<18} {:<24} {}",
            what,
            where_,
            if *present { "present" } else { "ABSENT -- its gates skipped" }
        );
    }
    let missing: Vec<&str> = rows
        .iter()
        .filter(|(_, _, p)| !*p)
        .map(|(w, _, _)| *w)
        .collect();
    if missing.is_empty() {
        eprintln!("  all optional gates ran.");
    } else {
        eprintln!(
            "  {} prerequisite(s) absent, so some gates proved NOTHING: {}",
            missing.len(),
            missing.join(", ")
        );
        eprintln!(
            "  This is expected on a hosted runner. Set PYPTO_REQUIRE_FULL_GATES=1 on a \
             machine that should have them."
        );
    }
    eprintln!("-----------------------------------\n");

    if std::env::var("PYPTO_REQUIRE_FULL_GATES").is_ok() {
        assert!(
            missing.is_empty(),
            "PYPTO_REQUIRE_FULL_GATES is set, so a reduced run is a failure. Absent: {}. \
             Install the pinned nightly (`rustup toolchain install {tc}`), miri on it \
             (`rustup +{tc} component add miri`), and the ptoas assembler.",
            missing.join(", ")
        );
    }
}
