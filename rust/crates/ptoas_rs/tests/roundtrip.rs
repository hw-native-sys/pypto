// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! S3: round-trip verifier coverage.
//!
//! For each representative PTO fixture, assert that
//! parse → emit_pto → parse preserves the AST. This catches silent
//! information loss in either direction (parser dropping a field, emitter
//! forgetting one).

use ptoas_rs::{ast::PtoOp, op_model_coverage, op_name, verify_roundtrip, MODELLED_OPS};

mod shared;
use shared::{collect_pto_files, module_with, op_case, Case};

fn assert_roundtrip(src: &str) {
    if let Err(m) = verify_roundtrip(src) {
        if let Some(why) = &m.failure {
            panic!("round-trip did not complete: {why}");
        }
        let (before, after, reemitted) = (&m.before, &m.after, &m.reemitted);
        panic!(
            "AST changed across round-trip\n\
             --- re-emitted PTO ---\n{reemitted}\n\
             --- before ---\n{before:#?}\n\
             --- after ---\n{after:#?}"
        );
    }
}

/// The canonical softmax fixture. Exercises every op variant S1 supports
/// and the row-reduce scratch-operand handling.
const SOFTMAX_PTO: &str = include_str!("fixtures/softmax.pto");

#[test]
fn softmax_roundtrips() {
    assert_roundtrip(SOFTMAX_PTO);
}

/// Plain pointwise chain: load, exp, store.
const POINTWISE: &str = r#"
module {
  func.func @k(%a: !pto.ptr<f32>, %b: !pto.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %v0 = pto.make_tensor_view %a, shape = [%c1, %c32], strides = [%c32, %c1] : !pto.tensor_view<?x?xf32>
    %p0 = pto.partition_view %v0, offsets = [%c0, %c0], sizes = [%c1, %c32] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<1x32xf32>
    %t0 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32>
    %t1 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32>
    pto.tload ins(%p0 : !pto.partition_tensor_view<1x32xf32>) outs(%t0 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32>)
    pto.texp ins(%t0 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32>) outs(%t1 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32>)
    %v1 = pto.make_tensor_view %b, shape = [%c1, %c32], strides = [%c32, %c1] : !pto.tensor_view<?x?xf32>
    %p1 = pto.partition_view %v1, offsets = [%c0, %c0], sizes = [%c1, %c32] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<1x32xf32>
    pto.tstore ins(%t1 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32>) outs(%p1 : !pto.partition_tensor_view<1x32xf32>)
    return
  }
}
"#;

#[test]
fn pointwise_chain_roundtrips() {
    assert_roundtrip(POINTWISE);
}

/// Matmul exercises the shape-agreeing `<M,K,N>` triple.
const MATMUL: &str = r#"
module {
  func.func @k(%a: !pto.ptr<f32>, %b: !pto.ptr<f32>, %c: !pto.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %va = pto.make_tensor_view %a, shape = [%c16, %c32], strides = [%c32, %c1] : !pto.tensor_view<?x?xf32>
    %pa = pto.partition_view %va, offsets = [%c0, %c0], sizes = [%c16, %c32] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<16x32xf32>
    %vb = pto.make_tensor_view %b, shape = [%c32, %c64], strides = [%c64, %c1] : !pto.tensor_view<?x?xf32>
    %pb = pto.partition_view %vb, offsets = [%c0, %c0], sizes = [%c32, %c64] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x64xf32>
    %ta = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32>
    %tb = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=64>
    %tc = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64>
    pto.tload ins(%pa : !pto.partition_tensor_view<16x32xf32>) outs(%ta : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32>)
    pto.tload ins(%pb : !pto.partition_tensor_view<32x64xf32>) outs(%tb : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=64>)
    pto.tmatmul ins(%ta, %tb : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=64>) outs(%tc : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64>)
    %vc = pto.make_tensor_view %c, shape = [%c16, %c64], strides = [%c64, %c1] : !pto.tensor_view<?x?xf32>
    %pc = pto.partition_view %vc, offsets = [%c0, %c0], sizes = [%c16, %c64] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<16x64xf32>
    pto.tstore ins(%tc : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64>) outs(%pc : !pto.partition_tensor_view<16x64xf32>)
    return
  }
}
"#;

#[test]
fn matmul_roundtrips() {
    assert_roundtrip(MATMUL);
}

/// Every real `.pto` artifact produced by `mlir_to_pto` in the repo.
///
/// This test has been wrong twice, in opposite directions, and the shape it has
/// now is the reconciliation of both corrections.
///
/// **First it over-reported.** It asserted that every artifact round-trips and
/// said 41 of 49 passed. That number was the complement of the truth: the
/// parser's dispatch fallthrough was `Ok(None)`, so an unmodelled op was
/// DROPPED, and a dropped op is absent from BOTH sides of a parse/emit/parse
/// comparison, so it cannot make the comparison differ. Census of this same
/// corpus, 2026-10-31: 1895 of 6158 op lines dropped, and **33 of those 41
/// "passing" artifacts contained at least one drop**. Exactly 8 files were
/// fully modelled.
///
/// **Then it under-reported.** The fix was a `MODELLED_OPS` gate in the parser:
/// refuse an unrecognised op by name. That made the drop loud, but it also
/// refused the ops the round trip could have carried losslessly, and the round
/// trip is the instrument that proves the reader -- so a file with one
/// unmodelled line stopped being read at all. `pto-rs-viewpoint-rebase` had
/// found the same drop independently and chosen the other remedy:
/// [`PtoOp::Unmodelled`], which keeps the line verbatim. That is the better
/// half, and it is what the parser does now.
///
/// Which leaves the hazard that remedy creates, and it is the hazard this crate
/// keeps paying for: **with every line carried, every file round-trips, so
/// green no longer means the reader understood anything.** A file of pure
/// garbage passes by byte passthrough. So the sweep asserts four things, and no
/// single mechanism satisfies all four:
///
/// 1. EVERY artifact round-trips. Universal now, and a real claim -- a parser
///    or emitter that mangles the text it does not understand fails here.
/// 2. The modelled/unmodelled partition computed from the TEXT agrees with the
///    one computed from the AST by [`op_model_coverage`]. This is what makes (1)
///    non-vacuous: passthrough is visible as `unmodelled > 0`, so "nothing was
///    interpreted" and "everything round-tripped" can no longer be the same
///    green. Two independent readings, kept independent on purpose -- the same
///    pairing caught a flat AST walk that did not descend into `scf.for` while
///    its text counterpart did.
/// 3. A file with an unmodelled op MUST be refused by the LIFT, by name. The
///    refusal moved from the parser to the emitter; this asserts it arrived.
/// 4. Every refusal must quote a token that is ACTUALLY IN THE FILE. Three
///    diagnostics in this crate did the opposite before 2026-10-31: a comment
///    reading `// pool: broadcast Q` was reported as an unrecognised *type*
///    `broadcast Q`, a rank-5 view was reported as a bad *dtype* `i8`, and a
///    dropped `pto.subview` surfaced later as `no shape for SSA %s`, blaming
///    the tstore that consumed it. A refusal that names nothing in its input is
///    a misattribution, and misattribution cost more time this session than
///    every genuine parse gap put together.
/// Artifacts this crate's lift does NOT read, with the reason for each.
///
/// Recorded rather than excluded by narrowing the sweep. These two are the
/// repository's own vPTO vector-function fixtures, and the lift failing on them
/// is a real gap in what it models -- not a property of the fixtures. A sweep
/// scoped to avoid them would report a denominator made of this crate's own
/// test inputs and call it coverage.
///
/// The set is asserted to be EXACTLY this, so both directions are visible: a
/// new unreadable artifact fails, and so does one of these starting to work.
/// The second is the event worth catching, because that is when the list is
/// stale and somebody should delete a line from it.
// FIXED and removed from this list: `bench/deepseek_v41/pto/topk_v41_1x384_k6.pto`
// used to fail here with "tile_buf missing/unknown dtype". The cause was
// `dtype=ui32` -- the PTO this project's OWN emitter writes for a top-k index
// output, which `pto.tsort32` requires, and which this crate's `DType` could
// not parse. `DType::U32` now carries it round-trip and the lift refuses it by
// name. The estimate that stopped it being fixed when it was found ("`DType` is
// matched at 99 sites across nine modules") counted every `DType::` mention;
// the variant is exhaustively matched at FIVE sites, all in `src/ast.rs`.
//
// Worth keeping as the reason this list is checked both ways: with the parse
// failure gone, the artifact refuses for `pto.tci`, an unmodelled op -- which
// is the real blocker and was invisible underneath a dtype error.
const KNOWN_UNMODELLED: &[(&str, &str)] = &[
    (
        "tests/ut/backend/vpto_vf_validation/helper.pto",
        "carries the function attribute `pto.aiv`; this crate models only \
         `pto.kernel_kind`, and carrying the file on without the attribute \
         would make a re-emission that dropped it look lossless",
    ),
    (
        "tests/ut/backend/vpto_vf_validation/caller.pto",
        "a cross-function vPTO call: the text walk sees operation lines the AST \
         walk does not, so the two readings disagree about what is modelled",
    ),
];

#[test]



fn all_repo_pto_artifacts_roundtrip() {
    // The repository root, not the Rust workspace: the kernel bench's PTO
    // column lives outside `rust/`, and a sweep that cannot see it would
    // report a denominator made entirely of this crate's own fixtures.
    let repo_root = shared::corpus_root();
    let mut files = Vec::new();
    collect_pto_files(&repo_root, &mut files);
    assert!(
        !files.is_empty(),
        "no .pto artifacts found under {}",
        repo_root.display()
    );

    let mut modelled = Vec::new();
    let mut lifted = Vec::new();
    let mut problems: Vec<String> = Vec::new();

    for path in &files {
        let src = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        // Counted over the text the LIFT parses. `convert_pto_to_rust` expands a
        // constant-bound chunk loop before parsing, so an `arith.muli` inside one is not a
        // gap: it never reaches the AST. Counting it here made the two readings disagree
        // about files the lift legitimately transforms.
        let lifted_src = ptoas_rs::unroll::unroll_constant_loops(&src);
        let gaps = unmodelled_ops(&lifted_src);
        let rel = path.display().to_string();

        // (1) Universal: every artifact round-trips.
        if let Err(m) = verify_roundtrip(&src) {
            let detail = match &m.failure {
                Some(why) => why.clone(),
                None => format!(
                    "AST diverged\n--- re-emitted ---\n{}\n--- before ---\n{:#?}\
                     \n--- after ---\n{:#?}",
                    m.reemitted, m.before, m.after
                ),
            };
            problems.push(format!("{rel}\n  must round-trip:\n  {detail}"));
            continue;
        }

        // (2) Text and AST must agree on WHETHER anything was not understood.
        match op_model_coverage(&lifted_src) {
            Err(e) => problems.push(format!("{rel}\n  op_model_coverage failed: {e}")),
            Ok((m_ops, u_ops)) => {
                if gaps.is_empty() != (u_ops == 0) {
                    problems.push(format!(
                        "{rel}\n  the two readings disagree: the text walk found {} \
                         unmodelled op line(s){}, the AST walk found {u_ops} of {} ops \
                         unmodelled",
                        gaps.len(),
                        gaps.first()
                            .map(|g| format!(" (first: {g})"))
                            .unwrap_or_default(),
                        m_ops + u_ops,
                    ));
                }
                if m_ops == 0 {
                    problems.push(format!(
                        "{rel}\n  0 ops modelled: this file round-tripped by byte \
                         passthrough, which proves nothing about the reader"
                    ));
                }
            }
        }

        if gaps.is_empty() {
            modelled.push(rel.clone());
        }

        // (3) and (4): the lift's verdict, and whether its refusal names the file.
        match ptoas_rs::convert_pto_to_rust(&src) {
            Ok(_) => {
                if gaps.is_empty() {
                    lifted.push(rel.clone());
                } else {
                    problems.push(format!(
                        "{rel}\n  the LIFT accepted a file with {} unmodelled op \
                         line(s), first: {}\n  an op with no AST meaning has no \
                         tile_std lowering either",
                        gaps.len(),
                        gaps[0]
                    ));
                }
            }
            Err(why) => {
                let quoted = backtick_spans(&why);
                if quoted.is_empty() {
                    problems.push(format!("{rel}\n  refusal quotes nothing: {why}"));
                } else if !quoted.iter().any(|q| src.contains(q)) {
                    problems.push(format!(
                        "{rel}\n  refusal names nothing in the file -- quoted {quoted:?}\n  \
                         {why}"
                    ));
                }
            }
        }
    }

    // Floor, not a snapshot: modelling an op moves a file from the refused half
    // to the fully-modelled half, so this can only rise. It is asserted so that
    // a regression in the parser cannot shrink coverage while the sweep stays
    // green on the other properties -- which, now that every file round-trips,
    // is the only thing standing between coverage and byte passthrough.
    const FULLY_MODELLED_FLOOR: usize = 23;

    eprintln!(
        "pto corpus: {} artifact(s); all round-trip; {} fully modelled, {} of those \
         lift to Rust; {} hold an op this crate does not model",
        files.len(),
        modelled.len(),
        lifted.len(),
        files.len() - modelled.len()
    );

    // Partition the problems: known gaps are reported, unknown ones fail.
    let (known, unknown): (Vec<&String>, Vec<&String>) = problems.iter().partition(|p| {
        KNOWN_UNMODELLED
            .iter()
            .any(|(path, _)| p.contains(path))
    });
    assert!(
        unknown.is_empty(),
        "{} artifact(s) misbehaved for reasons not recorded in KNOWN_UNMODELLED:\n{}",
        unknown.len(),
        unknown
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    );
    // And the other direction: a recorded gap that no longer reproduces means
    // the list is stale, which is a result worth failing on rather than a
    // quiet improvement nobody notices.
    for (path, why) in KNOWN_UNMODELLED {
        let still_fails = known.iter().any(|p| p.contains(path));
        let present = files.iter().any(|f| f.to_string_lossy().contains(path));
        if present {
            assert!(
                still_fails,
                "{path} now round-trips. Delete its KNOWN_UNMODELLED entry in \
                 this change -- the recorded reason was: {why}"
            );
        }
    }
    assert!(
        modelled.len() >= FULLY_MODELLED_FLOOR,
        "fully-modelled coverage fell to {} of {} (floor {FULLY_MODELLED_FLOOR}); \
         modelled files were:\n{}",
        modelled.len(),
        files.len(),
        modelled.join("\n")
    );
}

/// Op lines whose op name is not in `MODELLED_OPS`, computed from the text
/// alone. Deliberately independent of `parse_module`, so that the parser and
/// this walk can disagree and be caught doing it.
fn unmodelled_ops(src: &str) -> Vec<String> {
    let mut out = Vec::new();
    for (n, raw) in src.lines().enumerate() {
        let line = match raw.find("//") {
            Some(i) => raw[..i].trim(),
            None => raw.trim(),
        };
        if line.is_empty()
            || line.starts_with('}')
            || line.starts_with('{')
            || line.starts_with("module")
            || line.starts_with("func.func")
            || line.starts_with("return")
        {
            continue;
        }
        let op = op_name(line);
        if !MODELLED_OPS.contains(&op) {
            out.push(format!("line {} `{}`", n + 1, op));
        }
    }
    out
}

/// The backtick-quoted spans of a diagnostic -- what it claims to have seen.
fn backtick_spans(msg: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut rest = msg;
    while let Some(a) = rest.find('`') {
        rest = &rest[a + 1..];
        match rest.find('`') {
            Some(b) => {
                if b > 0 {
                    out.push(&rest[..b]);
                }
                rest = &rest[b + 1..];
            }
            None => break,
        }
    }
    out
}


// ─── Per-op round-trip coverage ─────────────────────────────────────────────

/// One minimal module per modelled op, round-tripped.
///
/// This exists because the corpus sweep above cannot do it. Measured
/// 2026-10-31: of the 49 artifacts, the 8 fully-modelled ones exercise only
/// **12 of the 25 modelled ops** -- every pointwise binary, all four
/// tile-by-scalar forms, `tneg`, `tabs` and `tmov` appear in no round-tripping
/// artifact at all, and 4 of those 8 "artifacts" are this crate's own
/// fixtures, so only four are really emitter output.
///
/// That was not a theoretical gap. Two mutants proved the sweep blind to it:
/// making the reverse emitter write `pto.tmul` for a `pto.tmuls`, and deleting
/// `pto.tmov` from `MODELLED_OPS`, both left the sweep GREEN. Each fails here.
///
/// `MISSING_CASE` below is the drift guard: adding a mnemonic to
/// `MODELLED_OPS` without adding a case makes this test fail, so coverage
/// cannot be widened in name only.
#[test]
fn every_modelled_op_round_trips() {
    let mut uncovered = Vec::new();
    for op in MODELLED_OPS {
        match op_case(op) {
            Case::InPreamble => {}
            Case::Line(l) => {
                let src = module_with(&l);
                assert_roundtrip(&src);
                // The list and the dispatch are two things, and the list used
                // to be checked BEFORE the dispatch ran -- so a mnemonic listed
                // here whose arm stopped claiming the line was certified by the
                // very list that named it. Now that an unclaimed line becomes
                // `PtoOp::Unmodelled` instead of an error, that drift would be
                // SILENT: the module would still round-trip, verbatim. This is
                // the assertion that closes it.
                let m = ptoas_rs::parser::parse_module(&src)
                    .unwrap_or_else(|e| panic!("`{op}` case did not parse: {e}"));
                let carried: Vec<&String> = m
                    .funcs
                    .iter()
                    .flat_map(|f| f.ops.iter())
                    .filter_map(|o| match o {
                        PtoOp::Unmodelled { op, .. } => Some(op),
                        _ => None,
                    })
                    .collect();
                assert!(
                    carried.is_empty(),
                    "`{op}` is in MODELLED_OPS but the dispatch did not claim it: \
                     parsed as Unmodelled {carried:?}"
                );
            }
            Case::Missing => uncovered.push(*op),
        }
    }
    assert!(
        uncovered.is_empty(),
        "these ops are in MODELLED_OPS with no round-trip case: {uncovered:?}"
    );

    // NARROWING the list is invisible to the loop above, which iterates it:
    // deleting an entry deletes its own case along with it. Measured
    // 2026-10-31 -- deleting `pto.tmov` left all five tests in this file
    // green, and the corpus floor did not see it either, because none of the
    // 8 round-tripping artifacts uses `tmov`. This count is the only
    // assertion that can fail on a narrowing, so it is not redundant with
    // `uncovered`; raise it when ops are added.
    const MODELLED_OPS_FLOOR: usize = 42;
    assert!(
        MODELLED_OPS.len() >= MODELLED_OPS_FLOOR,
        "MODELLED_OPS narrowed to {} (floor {MODELLED_OPS_FLOOR}): {MODELLED_OPS:?}",
        MODELLED_OPS.len()
    );
}
