// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! The one place a proof and this codebase touch a shared DISCIPLINE.
//!
//! `proofs/lambda_tile/coq/audit/typestate_pnd_rdy_mirror.rs` is a hand-written
//! Rust file whose header claims:
//!
//! > "Fidelity mirror of tile_std::buf's Pnd/Rdy typestate ... **This mirrors
//! > buf.rs exactly**: DmaPending <-> TBuf Pnd (only method: sync);
//! > UbView <-> TBuf Rdy (only data bridge: raw); sync(self) <-> T_Sync :
//! > Pnd -> Rdy (+ __tile_pipe_barrier); raw(&self) <-> the read/vec-op
//! > surface (Rdy-only)"
//!
//! That is a claim about `crates/tile_std/src/buf.rs`, and when this test was
//! written **nothing checked it** -- the mirror is referenced by one markdown
//! file and compiled by nothing. A hand-written model agreeing with its
//! reference demonstrates the author's understanding, not the file; this crate
//! has paid for that distinction before.
//!
//! The claim is worth checking rather than deleting, because it is the only
//! mechanised-proof property that is actually realised in shipping Rust: the
//! calculus's `T_Read` requires a `Rdy` buffer and `T_Sync` is the only way to
//! get one, and in `buf.rs` that comes out as move-typestate -- reading before
//! the barrier is a TYPE error, not a runtime check. If `buf.rs` drifts, the
//! proof still holds about the calculus and stops saying anything about the
//! code.
//!
//! Lives in this crate rather than in `tile_std` on purpose: `tile_std` is
//! mapped to a PUBLIC deploy target, and the upstream proof work deliberately
//! broke the link between itself and this code's repository. This crate's
//! targets are private.

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn read(rel: &str) -> String {
    let p = repo_root().join(rel);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()))
}

const MIRROR: &str = "proofs/lambda_tile/coq/audit/typestate_pnd_rdy_mirror.rs";
const BUF: &str = "crates/tile_std/src/buf.rs";

/// The two typestates exist in the real crate with the signatures the mirror
/// assumes.
/// DEFERRED, NOT RETIRED -- every test in this file compares against the
/// kernel-side DSL's typestates, and that crate is not in this tree yet.
#[test]
fn the_two_typestates_have_the_mirrored_signatures() {
    let buf = read(BUF);
    for want in [
        "pub struct UbView<'a, const CAP: usize, T>",
        "pub struct DmaPending<'a, const CAP: usize, T>",
    ] {
        assert!(
            buf.contains(want),
            "`{BUF}` no longer declares `{want}`. The mirror in {MIRROR} claims to mirror \
             it exactly; if the real type changed, the mirror is describing a type that is \
             gone and the proof's connection to this code is broken"
        );
    }
}

/// `sync` CONSUMES its receiver. This is the whole discipline.
///
/// `T_Sync : TBuf Pnd t R C -> TBuf Rdy t R C` is a linear transition: the
/// pending buffer is gone afterwards. In Rust that is `fn sync(self)`. Change
/// it to `&self` or `&mut self` and a pending buffer survives its own barrier,
/// so a second read could observe the pre-barrier state -- which is the
/// `Hz_read_pnd` hazard the calculus rules out, reintroduced in the code the
/// calculus was written about.
/// DEFERRED, NOT RETIRED -- every test in this file compares against the
/// kernel-side DSL's typestates, and that crate is not in this tree yet.
#[test]
fn sync_consumes_the_pending_buffer() {
    let buf = read(BUF);
    assert!(
        buf.contains("pub fn sync(self) -> UbView<'a, CAP, T>"),
        "`sync` no longer has the signature `sync(self) -> UbView<'a, CAP, T>`. By value is \
         the linearity; by reference would let a `DmaPending` outlive its own barrier"
    );
    assert!(
        !buf.contains("pub fn sync(&self)") && !buf.contains("pub fn sync(&mut self)"),
        "`sync` has a by-reference form, which breaks the Pnd -> Rdy transition's linearity"
    );
}

/// `DmaPending` has exactly ONE method, and the mirror says so.
///
/// "the ONLY method" is load-bearing: the point of the pending typestate is
/// that nothing can be done with a buffer in flight EXCEPT wait for it. Any
/// second method is a way to touch a Pnd buffer, which is what `Hz_read_pnd`
/// and `Hz_free_pnd` are.
/// DEFERRED, NOT RETIRED -- every test in this file compares against the
/// kernel-side DSL's typestates, and that crate is not in this tree yet.
#[test]
fn a_pending_buffer_admits_only_sync() {
    let buf = read(BUF);
    let i = buf
        .find("impl<'a, const CAP: usize, T> DmaPending<'a, CAP, T>")
        .unwrap_or_else(|| panic!("no inherent impl for DmaPending in {BUF}"));
    // To the end of that impl block, by brace balance.
    let rest = &buf[i..];
    let open = rest.find('{').expect("impl block opens");
    let mut depth = 0usize;
    let mut end = open;
    for (k, c) in rest[open..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end = open + k;
                    break;
                }
            }
            _ => {}
        }
    }
    let block = &rest[open..=end];
    let methods: Vec<&str> = block
        .lines()
        .filter(|l| l.trim_start().starts_with("pub fn "))
        .collect();
    assert_eq!(
        methods.len(),
        1,
        "`DmaPending` now has {} public methods, not the one the mirror calls \"the ONLY \
         method\": {:?}. Every additional one is a way to touch a buffer that is still in \
         flight, which is the `read_pnd` / `free_pnd` hazard family",
        methods.len(),
        methods
    );
    assert!(
        methods[0].contains("fn sync"),
        "`DmaPending`'s single method is no longer `sync`: {:?}",
        methods[0]
    );
}

/// The mirror still claims what this test checks.
///
/// Without this, the mirror could be rewritten to claim something weaker and
/// these assertions would keep passing while testing a claim nobody makes --
/// the test would have outlived its own premise.
/// DEFERRED for a DIFFERENT reason than the rest of this file, which is why
/// the other three tests here now run and this one does not.
///
/// They compare against the kernel-side DSL, which is in this tree. This one
/// reads the OTHER side of the comparison -- the Coq development's Rust
/// mirror at `proofs/lambda_tile/coq/audit/` -- and the proofs tree is not
/// part of this contribution. Carrying the single mirror file would make the
/// test green while orphaning it from the development that gives it meaning,
/// so the premise check waits for the proofs tree rather than for the DSL.
///
/// The blanket reason this used to carry ("needs tile_std") was wrong, and
/// wrong in the direction that matters: it pointed at a dependency that had
/// since arrived, so the test looked unblocked when it was not.
#[ignore = "reads the Coq development's mirror under proofs/, which is not part of this tree"]
#[test]
fn the_mirror_still_makes_the_claim_being_checked() {
    let m = read(MIRROR);
    for claim in [
        "Fidelity mirror of tile_std::buf's Pnd/Rdy typestate",
        "This mirrors buf.rs exactly",
        "the ONLY method",
    ] {
        assert!(
            m.contains(claim),
            "the mirror no longer says `{claim}`; if its claim changed, this test is \
             checking something it does not assert"
        );
    }
    // And it is still not built by anything, which is WHY this test exists. If
    // it ever gains a build, prefer that and delete this file.
    let manifest_hits = read("proofs/README.md");
    assert!(
        !manifest_hits.contains("compiles the mirror"),
        "the mirror is now compiled somewhere -- prefer a real build over this text check"
    );
}
