// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! `pto.barrier`: why it is droppable, and the two cases where it is not.
//!
//! The lift refused every artifact carrying a barrier -- 814 instances, the
//! single most common op in the corpus -- on the grounds that
//! `__tile_pipe_barrier` sits at the `tile_std` crate root rather than in the
//! `tile_std::tile` glob, and that "whether a tile-API kernel needs an
//! explicit barrier at all is UNVERIFIED, so emitting nothing would be a guess
//! that happens to compile".
//!
//! Both halves of that are now settled, and the second one the other way
//! round: emitting a barrier CALL would be the guess. `mlir_to_pto` deletes
//! those calls, so one could never survive a recompile back to PTO.
//!
//! The tests below pin the premises rather than restate them, because each is
//! a fact about another crate's source or about checked-in artifacts, and
//! either can move without this crate noticing.
//!
//! What is deliberately NOT claimed: that `ptoas --enable-insert-sync`
//! reconstructs every barrier. `mlir_to_pto.rs` says the opposite for one
//! pattern, and `ptoas` is not on this machine. The rule needs no such claim --
//! it asks only whether an ordering survives into the generated Rust.

use ptoas_rs::convert_pto_to_rust;

/// Fixtures live beside this test rather than at repository-root-relative
/// paths. The ported version read four separate trees of the repository these
/// files came from -- a test that asserts premises about someone else's layout
/// fails for reasons that have nothing to do with the premises.
fn read(rel: &str) -> String {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/barrier")
        .join(rel);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()))
}

/// The forward emitter, which is in this workspace.
fn emitter_source() -> String {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../rspto_emit/src/mlir_to_pto.rs");
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()))
}

// ─── The premises ───────────────────────────────────────────────────────────

/// PREMISE 1: the forward emitter DELETES `__tile_pipe_barrier` lines.
///
/// This is what makes a barrier inexpressible rather than merely awkward to
/// import. If `mlir_to_pto` grows an arm that emits `pto.barrier` for the
/// call, the lift could preserve barriers and should -- this test is how that
/// becomes visible instead of silently making the drop lossy.
#[test]
fn the_forward_emitter_still_deletes_barrier_calls() {
    let src = emitter_source();
    // The skip is a `continue` guard listing the lines that produce no PTO.
    let guard = src
        .find("|| line.contains(\"__tile_pipe_barrier\")")
        .map(|i| &src[i..(i + 400).min(src.len())]);
    assert!(
        guard.is_some(),
        "mlir_to_pto no longer skips `__tile_pipe_barrier` lines; if it now EMITS \
         a barrier for them, the Rust lift can preserve barriers and this crate's \
         decision to drop them needs revisiting"
    );
    assert!(
        guard.unwrap().contains("continue"),
        "the `__tile_pipe_barrier` line is matched but no longer `continue`d: {:?}",
        guard.unwrap()
    );
}

/// PREMISE 2: a barrier-free PTO file still yields AscendC WITH barriers.
///
/// The end-to-end evidence that the sync is manufactured below this level,
/// read from two checked-in artifacts rather than from a comment: the tile-API
/// Rust kernel makes no barrier call, its PTO has no `pto.barrier`, and the
/// `ptoas` output beside it has several `pipe_barrier`.
#[test]
fn ptoas_inserts_sync_into_a_barrier_free_module() {
    let rust_kernel = read("tile_softmax_kernel.rs");
    assert_eq!(
        rust_kernel.matches("pipe_barrier").count(),
        0,
        "the example kernel now calls a barrier; it is no longer evidence that \
         a barrier-FREE kernel gets sync inserted"
    );

    let pto = read("tile_softmax.pto");
    assert_eq!(
        pto.matches("pto.barrier").count(),
        0,
        "the example's PTO now carries barriers; same loss of evidence"
    );

    let cpp = read("tile_softmax.pto.cpp");
    let inserted = cpp.matches("pipe_barrier(").count();
    assert!(
        inserted > 0,
        "the ptoas output for a barrier-free module has no `pipe_barrier`, which \
         would mean the sync is NOT manufactured downstream"
    );
    // 2026-11-01: 1 PIPE_ALL + 4 PIPE_V.
    assert!(
        inserted >= 5,
        "only {inserted} barriers inserted (was 5 on 2026-11-01)"
    );
}

// ─── The consequence ────────────────────────────────────────────────────────

/// The four fully-modelled barriered artifacts lift, barriers and all.
#[test]
fn a_barriered_artifact_lifts_and_emits_no_barrier_call() {
    for rel in [
        "barriered_e.pto",
        "barriered_f.pto",
        "barriered_g.pto",
        "barriered_k.pto",
    ] {
        let src = read(rel);
        assert!(
            src.contains("pto.barrier"),
            "{rel} has no barrier, so it cannot test the barrier decision"
        );
        let rust = convert_pto_to_rust(&src).unwrap_or_else(|e| panic!("{rel}: {e}"));
        assert!(
            !rust.contains("barrier"),
            "{rel}: the lift emitted a barrier call, which `mlir_to_pto` would \
             delete on the way back:\n{rust}"
        );
    }
}

// ─── The two blockers, each proven able to fire ─────────────────────────────
//
// ONE blocker, since 2026-11-09: the in-place clause was retired as false and
// its test now pins the relaxation instead. The survivor does fire on the
// corpus -- bm_rmsnorm_8_5120 refuses on it today, having previously refused on
// the in-place clause ahead of it -- and is also perturbed below, because a
// corpus instance can disappear with one generator change.

/// An in-place tile write does NOT block the drop, and once did.
///
/// RETIRED PREMISE, 2026-11-09. The clause said `ins(%x, ...) outs(%x)` is two
/// PTO ops sharing storage that "does not appear in the lifted Rust at all", so
/// a barrier ordering it has nothing to be carried by. It is the opposite:
/// `pto.tadd ins(%a, %b) outs(%a)` lifts to `let a = tile_add_f32(a, b);`, whose
/// initializer READS THE PREVIOUS BINDING. Rust `let` shadowing is positional
/// and the emitter walks the ops in PTO order into one flat statement list, so
/// the dependency is in the value graph -- the strongest case for dropping the
/// barrier rather than an exception to it.
///
/// This test now pins the relaxation, because the same false premise had been
/// written down in three places (here, the multi-write refusal, and the
/// `the_last_tier` tier list) and each one had to be retired separately.
///
/// What is NOT claimed: that a re-emission allocates the same buffers.
/// `mlir_to_pto` allocates per value, so this chain would ask for one tile per
/// accumulate where the original reused one. That is a capacity property of the
/// `.tile` -> `.pto` arm, not an ordering the Rust fails to carry.
#[test]
fn an_in_place_write_no_longer_blocks_the_barrier_drop() {
    let src = read("barriered_e.pto");
    assert!(convert_pto_to_rust(&src).is_ok(), "baseline must lift");

    // The GAIN multiply, which sits AFTER the fused chain. Perturbing the
    // chain's own opening square instead (`outs(%pto8)` -> `outs(%pto2)`) is
    // what this test used to do, and it no longer isolates the question: it
    // breaks the rms-norm fusion pattern, so the chain emits op by op and the
    // artifact refuses on `pto.tsqrt` having no `tile_sqrt_*` wrapper. The old
    // perturbation only looked clean because the in-place blocker fired ahead of
    // that. Measured 2026-11-09.
    let gain = src
        .lines()
        .find(|l| l.contains("pto.tmul ins(%pto16, %pto5"))
        .expect("the artifact has the gain multiply")
        .to_string();
    let patched = src.replace(&gain, &gain.replace("outs(%pto17 :", "outs(%pto16 :"));
    assert_ne!(patched, src, "the perturbation did not apply");

    let rs = convert_pto_to_rust(&patched)
        .expect("an in-place write is no longer a barrier blocker");
    // The dependency the retired clause claimed was lost: the new binding's
    // initializer names the old one.
    assert!(
        rs.contains("let t16 = tile_mul_f32::<1, 4096>(t16, t5);"),
        "the in-place write must lift to a self-reading shadowed binding, which \
         is what carries the ordering: \n{rs}"
    );
}

/// A `pto.tload` after a `pto.tstore` blocks the drop.
///
/// That ordering goes through raw pointers, which the lifted Rust's value
/// graph does not model, so dropping the barrier between them would drop the
/// only thing expressing it.
#[test]
fn a_load_after_a_store_blocks_the_barrier_drop() {
    let src = read("barriered_e.pto");
    assert!(convert_pto_to_rust(&src).is_ok(), "baseline must lift");

    // Re-load the output view after storing to it. `%pto19` is the output
    // partition view and `%pto9` a spare 1x2048 tile, so the shapes differ --
    // which does not matter here, because the refusal must happen before any
    // shape check on the load.
    let reload = "    pto.tload ins(%pto19 : !pto.partition_tensor_view<1x4096xf32>) \
                  outs(%pto15 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=4096, \
                  v_row=1, v_col=4096, blayout=row_major, slayout=none_box, fractal=512, \
                  pad=0>)\n    return";
    let patched = src.replace("    return", reload);
    assert_ne!(patched, src, "the perturbation did not apply");

    let err = convert_pto_to_rust(&patched)
        .expect_err("a load after a store must block the barrier drop");
    assert!(
        err.contains("pto.tload") && err.contains("pto.tstore"),
        "the refusal does not name the store/load pair: {err}"
    );
}

/// Two stores block the drop as well -- the store-to-store ordering is the
/// same raw-pointer hazard as store-then-load.
#[test]
fn a_second_store_blocks_the_barrier_drop() {
    let src = read("barriered_e.pto");
    let store = src
        .lines()
        .find(|l| l.contains("pto.tstore"))
        .expect("the artifact has a store")
        .to_string();
    let patched = src.replace(&store, &format!("{store}\n{store}"));
    assert_ne!(patched, src, "the perturbation did not apply");

    let err =
        convert_pto_to_rust(&patched).expect_err("a second store must block the barrier drop");
    assert!(
        err.contains("pto.tstore"),
        "the refusal does not name the second store: {err}"
    );
}

/// A blocker INSIDE a region is found, and reported ahead of the region.
///
/// This is the test that gives `barrier_blocker`'s recursion into `scf.for` and
/// `scf.if` something to prove. The regions always refuse, so the recursion
/// looks like dead code -- a mutant that disabled it killed nothing. But the two
/// orders are distinguishable: the blocker is computed over the whole function
/// BEFORE any op is emitted, so a function with top-level barriers and a blocker
/// inside a loop refuses with the BLOCKER's message if the recursion runs, and
/// with the `scf.for` message if it does not.
///
/// The discriminator was an in-place write until 2026-11-09, when that clause
/// was retired as false (see `an_in_place_write_no_longer_blocks_the_barrier_drop`).
/// It is now the surviving clause: a `pto.tload` from a view a `pto.tstore` in
/// the same region already wrote, which goes through raw pointers and is a real
/// ordering the Rust cannot carry.
#[test]
fn a_blocker_inside_a_region_is_reported_before_the_region_itself() {
    let src = read("barriered_e.pto");
    // The artifact's one store, which sits after the chain and after all eight
    // barriers. Wrapping an op that precedes them would not discriminate: the
    // emit loop would reach the region first and refuse on it either way.
    let store = src
        .lines()
        .find(|l| l.contains("pto.tstore ins(%pto17"))
        .expect("the artifact has one store")
        .to_string();
    // The same store followed by a LOAD from the view it just wrote, both
    // wrapped in a loop. The function keeps its eight top-level barriers, all
    // of them ahead of the region.
    let load = src
        .lines()
        .find(|l| l.contains("pto.tload ins(%pto1 "))
        .expect("the artifact loads through a partition view")
        .to_string();
    let reload = load.replace("ins(%pto1 ", "ins(%pto19 ");
    assert_ne!(reload, load, "the load perturbation did not apply");
    let wrapped = format!(
        "    scf.for %r = %c0 to %c1 step %c1 {{\n    {}\n    {}\n    }}",
        store.trim(),
        reload.trim(),
    );
    let patched = src.replace(&store, &wrapped);
    assert_ne!(patched, src, "the perturbation did not apply");

    let err = convert_pto_to_rust(&patched).expect_err("neither form of this lifts");
    assert!(
        err.contains("%pto19") && err.contains("pto.tstore"),
        "the store-then-load inside the `scf.for` was not found -- the blocker did \
         not recurse into the region, and refused on the region instead: {err}"
    );
}

/// The decision is made per FUNCTION, and a region cannot hide a blocker.
///
/// `barrier_blocker` recurses into `scf.for` and `scf.if` bodies. Both region
/// arms refuse before emitting a body today, so this is the test that keeps
/// the recursion honest if either ever starts emitting one -- and it is why
/// the op emitter's own `Barrier` arm is documented as unreachable rather than
/// deleted.
#[test]
fn barrier_in_a_region_is_still_decided_by_the_function() {
    let src = read("barriered_e.pto");
    // Wrap the whole chain's opening square in a loop, keeping it in place by
    // reusing a constant the artifact already defines for the bounds.
    let sq = src
        .lines()
        .find(|l| l.contains("pto.tmul ins(%pto2, %pto2"))
        .expect("the artifact has the opening square")
        .to_string();
    let wrapped = format!(
        "    scf.for %r = %c0 to %c1 step %c1 {{\n    {}\n    }}",
        sq.trim()
    );
    let patched = src.replace(&sq, &wrapped);
    assert_ne!(patched, src, "the perturbation did not apply");

    // The loop itself is refused, which is the point: the refusal names the
    // region rather than a barrier, so the barrier decision did not
    // accidentally certify a function whose body it never looked at.
    let err = convert_pto_to_rust(&patched).expect_err("an scf.for body is not lifted");
    assert!(
        err.contains("scf.for"),
        "expected the region refusal, got: {err}"
    );
}
