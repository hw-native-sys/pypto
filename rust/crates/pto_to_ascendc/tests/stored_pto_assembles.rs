// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Every `.pto` the corpus stores must assemble.
//!
//! The corpus already had a byte-identity gate over its PTO: re-run the
//! emitter, compare against the stored file. That gate is sound, it was green,
//! and three of the stored artefacts did not parse. It cannot see them,
//! because it compares the emitter's output to the file the emitter wrote --
//! when an emitter produces something invalid, both sides hold the SAME invalid
//! text and the comparison succeeds. **Reproducibility is not validity**, and
//! a second gate is needed to say so.
//!
//! This is a cargo integration test rather than a lint script on purpose. The
//! assembler reaches the corpus through `pto-to-ascendc`, whose path cargo
//! supplies here; the same check written as a `pre-commit` hook would need a
//! release build that the hook job does not do, and a gate that cannot run is
//! worse than no gate because it reports a pass.
//!
//! # What this gate found, and what it cost to close
//!
//! `KNOWN_INVALID` is now EMPTY: every stored artefact assembles. It got there
//! by fixing five emitter defects, and the list is kept (rather than deleted
//! with the defects) because it is the mechanism that retires an entry. When a
//! defect is recorded here the test asserts it STILL fails, so a fix flips this
//! file and tells whoever fixed it to check the artefact in and drop the entry.
//! An entry with no test behind it outlives the defect it was written for.
//!
//! The three recorded artefacts named two defects, and both were the FIRST
//! blocker rather than the whole of it. Each fix uncovered the next:
//!
//! `pto.tci`
//!   1. `{descending = false}` sat inside the `ins(...)` operand list, where
//!      MLIR's parser wants `:`. Moved onto the op, after `outs(...)`.
//!   2. the index tile was `ui32` while the start scalar is
//!      `arith.constant : i32`, and the op's verifier requires them to be the
//!      same type. Now `i32`.
//!
//! the topk composition (behind the same two `tci` artefacts)
//!   3. `pto.tgather` with a mask pattern requires an `axis`; `row` is the only
//!      direction the verifier accepts for this 2:1 gather.
//!   4. the `1×k` result tile with `k=6` is 24 bytes and cannot be allocated at
//!      all: a row must be 32-byte aligned. The tile is now padded with `v_col`
//!      carrying the logical width.
//!   5. `pto.tmov` refuses `1×cols → 1×k`; ptoas does not truncate a vector
//!      tile. The head extract is `pto.textract` at (0, 0).
//!
//! `attention_gqa`
//!   6. the arm reused the whole-operand partition view for a per-head use, so
//!      one SSA value carried `<32x64xf32>` and `<16x64xf32>`. The per-head
//!      view is built at `s×d` unless the tile's own view already matches.
//!   7. with that view fixed, the store moved the whole `s*heads_q × d` output
//!      view out of an `s×d` tile. The result is now registered as covering
//!      only its own extent, so the destination view is `s×d`.
//!
//! None of those settled what the emitted kernels COMPUTE. Fixes 3-5 are forced
//! by the assembler, not chosen, and the topk arm's `P1010` polarity is still
//! unresolved -- see the doc comment on `translate_topk`. This gate is about
//! whether the text parses, and nothing here widens it into a correctness claim.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Stored artefacts that do NOT assemble, each with the defect it demonstrates
/// and the assembler text that proves it is still THAT defect.
///
/// Empty: every stored artefact assembles. Keep the list and the tests below --
/// they are how the next defect gets recorded without being mistaken for a
/// fixed one.
const KNOWN_INVALID: &[(&str, &str, &str)] = &[];

fn bench_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../bench")
        .canonicalize()
        .expect("the bench corpus must be reachable from this crate")
}

fn stored() -> Vec<PathBuf> {
    let root = bench_root();
    let mut out = Vec::new();
    for corpus in std::fs::read_dir(&root).expect("rust/bench is readable").flatten() {
        let pto = corpus.path().join("pto");
        if !pto.is_dir() {
            continue;
        }
        for sub in [pto.clone(), pto.join("fixed")] {
            if !sub.is_dir() {
                continue;
            }
            for e in std::fs::read_dir(&sub).expect("a pto dir is readable").flatten() {
                let p = e.path();
                if p.extension().is_some_and(|x| x == "pto") {
                    out.push(p);
                }
            }
        }
    }
    out.sort();
    out
}

fn assembles(src: &Path) -> (bool, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_pto-to-ascendc"))
        .arg(src)
        .arg("-o")
        .arg("/dev/null")
        .output()
        .expect("pto-to-ascendc runs");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (out.status.success(), text)
}

fn relative(p: &Path) -> String {
    let root = bench_root();
    p.strip_prefix(&root).unwrap_or(p).to_string_lossy().replace('\\', "/")
}

#[test]
fn the_corpus_stores_pto_to_check() {
    // Guard against this file passing because it found nothing. A gate over an
    // empty list satisfies every assertion in it.
    let s = stored();
    assert!(
        s.len() >= 35,
        "only {} stored .pto found; the walker or the corpus broke",
        s.len()
    );
}

#[test]
fn every_stored_pto_assembles_or_is_a_recorded_defect() {
    let mut unexpected_fail = Vec::new();
    let mut unexpected_pass = Vec::new();
    let mut expected_fail = 0;

    for path in stored() {
        let rel = relative(&path);
        let known = KNOWN_INVALID.iter().find(|(p, _, _)| *p == rel);
        let (ok, text) = assembles(&path);

        match (ok, known) {
            (true, Some((_, defect, _))) => unexpected_pass.push(format!(
                "{rel}: listed as invalid ({defect}) but it now assembles -- \
                 the emitter defect looks fixed, so drop it from KNOWN_INVALID"
            )),
            (false, None) => {
                let first = text
                    .lines()
                    .find(|l| l.contains("error:"))
                    .unwrap_or_else(|| text.lines().last().unwrap_or(""));
                unexpected_fail.push(format!("{rel}: does not assemble and is not recorded: {first}"));
            }
            (false, Some(_)) => expected_fail += 1,
            (true, None) => {}
        }
    }

    assert!(unexpected_pass.is_empty(), "{}", unexpected_pass.join("\n"));
    assert!(
        unexpected_fail.is_empty(),
        "A stored .pto does not assemble. The byte-identity gate cannot catch this: \
         it compares the emitter's output to the file the emitter wrote, so an invalid \
         emission matches an invalid artefact.\n{}",
        unexpected_fail.join("\n")
    );
    assert_eq!(
        expected_fail,
        KNOWN_INVALID.len(),
        "every KNOWN_INVALID entry must be reached; {} of {} were",
        expected_fail,
        KNOWN_INVALID.len()
    );
}

#[test]
fn the_recorded_defects_still_fail_for_the_recorded_reason() {
    // Not merely "still fails". A refusal that changed cause is a different
    // defect wearing the same entry, and the failure that matters is one of
    // these quietly becoming a DIFFERENT problem while the list still claims to
    // describe it. `needle` is data, not a branch on the path, so recording a
    // new defect cannot accidentally assert the previous one's message.
    let root = bench_root();
    for (rel, _, needle) in KNOWN_INVALID {
        let path = root.join(rel);
        assert!(path.is_file(), "{rel}: recorded as invalid but the file is gone");
        let (ok, text) = assembles(&path);
        assert!(!ok, "{rel}: now assembles; drop it from KNOWN_INVALID");
        assert!(
            text.contains(needle),
            "{rel}: still fails, but not with `{needle}`. The cause changed, so the \
             recorded defect no longer describes it:\n{text}"
        );
    }

    // And the list being empty is itself asserted, so that "no recorded
    // defects" is a result this gate states rather than a list it forgot.
    assert!(
        KNOWN_INVALID.is_empty(),
        "{} artefact(s) are still recorded as invalid; the corpus is not yet \
         fully assembling",
        KNOWN_INVALID.len()
    );
}
