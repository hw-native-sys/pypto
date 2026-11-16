// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! The crate has TWO lift paths, and their names say the opposite of what they do.
//!
//! `convert_pto_to_rust` (the default, and `ptoas-rs` with no flag) emits
//! calls to `tile_std`'s SAFE WRAPPERS -- `tile_load_f32::<32, 32>(p)` -- so
//! every shape is a const generic and rustc proves it. That is the claim in the
//! generated file's own header and it is checked, artifact by artifact, in
//! `tests/lifted_kernels_compile.rs`.
//!
//! `convert_pto_to_tile_safe` (`--checked`) was documented as "the shape- and
//! linearity-checked tile-DSL form" and as lifting "with statically checked
//! shapes". Measured 2026-11-01: it emits **zero** `Tile<R, C, T>` over the
//! whole corpus. It writes `extern "C"` intrinsics over `u32` buffer handles
//! with the extents as runtime arguments, `__tile_load_f32(arg0, 32u32, 32u32)`,
//! so rustc proves nothing about any shape in it. The strictness it does have
//! is real but it is the EMITTER's, applied while lifting: it accepts 5 of the
//! 45 distinct artifacts against the default's 13, and the 5 are a strict
//! subset, so "refuses strictly more" is true.
//!
//! The second measurement is why no compile gate covers it. Every kernel it
//! emits carries `#[tile_std::tile_kernel]`, which expands to the codegen
//! marker `#[tile::kernel]`, so stock rustc stops at
//! `E0433: unresolved module `tile``. That is the FORM's property, not this
//! emitter's defect, and the control is in this file: a hand-written kernel
//! carrying the same attribute fails identically, and so does the repo's own
//! `examples/tile_softmax/kernels` (measured with `cargo check` on the pinned
//! nightly, same `E0433` at its `#[tile_std::tile_kernel]`).
//!
//! Without that control the obvious "fix" is to delete the attribute, which
//! would make the output compile and stop it being a tile-rs kernel.

mod shared;

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

/// Every distinct artifact, as text.
fn distinct_artifacts() -> Vec<(String, String)> {
    // `corpus_root()`, not `repo_root()`: the bench column lives outside the
    // cargo workspace, and a walk rooted at `rust/` finds 27 artifacts that are
    // almost all this crate's own fixtures.
    let root = shared::corpus_root();
    let mut files = Vec::new();
    shared::collect_pto_files(&root, &mut files);
    let mut by: std::collections::BTreeMap<Vec<u8>, std::path::PathBuf> = Default::default();
    for f in files {
        if let Ok(b) = std::fs::read(&f) {
            by.entry(b).or_insert(f);
        }
    }
    // 60 distinct artifacts at the revision this was written against. The
    // floor is a guard against the walk silently finding nothing, not a
    // measurement, so it sits well below the real count -- but above the 27
    // that a `rust/`-rooted walk returns, so that mistake cannot pass.
    assert!(
        by.len() > 40,
        "only {} distinct artifacts found -- is the walk rooted at `rust/` \
         instead of the repository root?",
        by.len()
    );
    by.into_iter()
        .map(|(bytes, path)| {
            (
                path.strip_prefix(&root)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .to_string(),
                String::from_utf8_lossy(&bytes).to_string(),
            )
        })
        .collect()
}

/// THE CORRECTION: the checked path's shapes are not in the type system.
///
/// Counted over the corpus rather than asserted on one fixture, because a
/// single hand-picked artifact is how a claim like the old one survives.
/// DEFERRED, NOT RETIRED -- it reads the kernel-side DSL (or its macro crate),
/// which is not in this tree yet. Restore it with the Rust input mode: the
/// point it pins is that the two lift paths differ in what rustc can check, and
/// that is only decidable with the DSL present.
#[test]
fn the_checked_path_emits_no_const_generics_and_the_default_emits_many() {
    let mut checked_tiles = 0usize;
    let mut checked_runtime_extents = 0usize;
    let mut default_generics = 0usize;
    let mut checked_accepted = 0usize;
    let mut default_accepted = 0usize;

    for (name, src) in distinct_artifacts() {
        if let Ok(out) = ptoas_rs::convert_pto_to_rust(&src) {
            default_accepted += 1;
            default_generics += out.matches("::<").count();
            assert!(
                !out.contains("#[tile_std::tile_kernel]"),
                "{name}: the DEFAULT path has started emitting the codegen-marker attribute, \
                 which would take its output out of reach of the compile gate"
            );
        }
        if let Ok(out) = ptoas_rs::convert_pto_to_tile_safe(&src) {
            checked_accepted += 1;
            checked_tiles += out.matches("Tile<").count();
            checked_runtime_extents += out.matches("u32,").count();
        }
    }

    assert_eq!(
        checked_tiles, 0,
        "the checked path now emits {checked_tiles} `Tile<` value(s). If it has moved to the \
         const-generic wrapper API, its docs' original claim has become true and the \
         correction in `convert_pto_to_tile_safe` should be revisited -- and a compile gate \
         may now be possible for it"
    );
    assert!(
        checked_runtime_extents > 0,
        "the checked path no longer passes extents as runtime `u32` arguments; re-measure \
         what form it emits before trusting either doc"
    );
    assert!(
        default_generics > 100,
        "only {default_generics} const-generic call sites in the default path's output over \
         {default_accepted} artifacts; the claim that rustc proves its shapes rests on these"
    );
    eprintln!(
        "default: {default_accepted} accepted, {default_generics} const-generic sites; \
         checked: {checked_accepted} accepted, {checked_tiles} Tile<>, \
         {checked_runtime_extents} runtime extents"
    );
}

/// "Refuses strictly more" -- true, and worth pinning as the SUBSET relation
/// rather than as two counts.
///
/// Two counts can both move and still satisfy `5 < 13` while the sets have
/// drifted apart, which would mean `--checked` accepts something the default
/// refuses and the word "strictly" has become wrong.
/// DEFERRED, NOT RETIRED -- it reads the kernel-side DSL (or its macro crate),
/// which is not in this tree yet. Restore it with the Rust input mode: the
/// point it pins is that the two lift paths differ in what rustc can check, and
/// that is only decidable with the DSL present.
#[test]
fn the_checked_path_accepts_a_strict_subset_of_the_default() {
    let mut only_checked = Vec::new();
    let (mut both, mut only_default) = (0usize, 0usize);
    for (name, src) in distinct_artifacts() {
        let d = ptoas_rs::convert_pto_to_rust(&src).is_ok();
        let c = ptoas_rs::convert_pto_to_tile_safe(&src).is_ok();
        match (d, c) {
            (true, true) => both += 1,
            (true, false) => only_default += 1,
            (false, true) => only_checked.push(name),
            (false, false) => {}
        }
    }
    assert!(
        only_checked.is_empty(),
        "`--checked` accepts {} artifact(s) the default refuses, so it does not refuse \
         STRICTLY more and its help text is wrong: {:?}",
        only_checked.len(),
        only_checked
    );
    assert!(
        both > 0 && only_default > 0,
        "both={both}, only_default={only_default} -- the two paths have stopped being a \
         subset and a superset, so there is nothing for 'strictly more' to mean"
    );
}

/// THE CONTROL: the attribute, not the emitter, is what stock rustc cannot take.
///
/// Hand-written rather than lifted, so that the thing under test is the FORM.
/// If this ever compiles, the codegen marker has become resolvable by stock
/// rustc and the checked path becomes gateable -- which is a change worth
/// noticing, not a test to delete.
/// DEFERRED, NOT RETIRED -- it reads the kernel-side DSL (or its macro crate),
/// which is not in this tree yet. Restore it with the Rust input mode: the
/// point it pins is that the two lift paths differ in what rustc can check, and
/// that is only decidable with the DSL present.
#[test]
fn the_codegen_marker_attribute_is_what_stock_rustc_rejects() {
    // The macro's expansion, read from its source: this is the attribute that
    // needs the custom backend.
    let macros = read("crates/tile_std_macros/src/lib.rs");
    assert!(
        macros.contains("parse_quote!(#[tile::kernel])"),
        "`tile_kernel` no longer expands to the `#[tile::kernel]` codegen marker; if the \
         expansion changed, re-measure whether its output can be type-checked here"
    );

    // The repo's own hand-written kernels carry the same attribute, which is
    // why this is the form's property and not the lift's.
    //
    // COUNTED, not `contains`: that file declares two kernels and carries the
    // attribute twice, so a `contains` is satisfied by whichever one a partial
    // migration has not touched yet -- and a partial migration is the drift
    // this is here to notice. (Measured when written: 2.)
    let hand_written = read("examples/tile_softmax/kernels/src/lib.rs");
    let marked = hand_written.matches("#[tile_std::tile_kernel]").count();
    let kernels = hand_written.matches("pub fn ").count();
    assert!(
        marked >= 2 && marked >= kernels.min(2),
        "the repo's hand-written tile kernels carry the attribute {marked} time(s) over \
         {kernels} `pub fn`; if they are moving off the form the checked lift emits, the \
         two have diverged and this file's argument needs redoing"
    );

    // And every artifact the checked path accepts gets it.
    let mut seen = 0usize;
    for (name, src) in distinct_artifacts() {
        if let Ok(out) = ptoas_rs::convert_pto_to_tile_safe(&src) {
            seen += 1;
            assert!(
                out.contains("#[tile_std::tile_kernel]"),
                "{name}: the checked path emitted a kernel WITHOUT the attribute. If that is \
                 deliberate the output is no longer a tile-rs kernel; if it is an omission \
                 the file will not be routed to the backend"
            );
        }
    }
    assert!(seen > 0, "the checked path accepted nothing; this test checked nothing");
}
