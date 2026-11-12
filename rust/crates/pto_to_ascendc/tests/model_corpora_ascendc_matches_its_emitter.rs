// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! The AscendC column of the MODEL corpora, re-derived rather than trusted.
//!
//! WHY THIS EXISTS, measured 2026-11-08. The sibling gate
//! `bench_ascendc_matches_its_emitter.rs` opens with "Every `.cpp` in the bench
//! must be reproducible from its `.tile`", and its `bench_root()` returns
//! `../../bench/playground16`. It checks ONE corpus of five. The docstring states
//! a universal and the code tests an existential, so the other four columns drifted
//! with nothing comparing them -- and they had. A sweep found THIRTEEN stale
//! artefacts, in two classes:
//!
//!   * ARITY (3): glm_5_3's proj_kv_a_mqa, proj_q_a and proj_shexp_down each had a
//!     106-line `.cpp` taking THREE GM buffers, while their own `.tile` declares
//!     five and calls `__tile_matvec_w8a8_f32`. Those are pre-int8 artefacts: they
//!     read the int8 weight plane as f32 and take no scale and no offset. Each
//!     carried a recorded device PASS, so the PASS was obtained on a kernel
//!     computing DIFFERENT ARITHMETIC from the one the corpus declares. The fault
//!     surfaced as a LINK error -- `no matching function for call to
//!     'glm53_proj_kv_a_mqa_do'` -- when the w8a8 oracle's 5-argument call met the
//!     3-argument artefact. Note the asymmetry that made it invisible: the `.pto`
//!     column for the same three kernels was byte-identical and carried `si8`,
//!     because the PTO gate had been covering it all along.
//!
//!   * BUFFER (10): same arity, different `pipe.InitBuffer` sizes -- the emitter's
//!     allocation plan moved and the artefact did not. Both directions occur, and
//!     they are not equally harmless. glm_5_3 and kimi_k3's kda_delta had a
//!     committed buffer of 512 bytes where the emitter now computes 1024, so the
//!     device ran a kernel the current plan calls UNDER-ALLOCATED;
//!     routed_gate_up_glm53_8x2048 had 31360 where it now computes 16384, which is
//!     merely wasteful. The second is this session's own chunk-width change
//!     reaching the AscendC column -- landed in `pto/**` and not here, because
//!     nothing asked.
//!
//! One test per corpus, not one test over all four, for the same reason the PTO
//! gates are four files: the counts ARE the claim, and a number covering four
//! corpora says nothing about any of them when it moves.
//!
//! Three counts are pinned per corpus rather than one. `identical` alone cannot
//! distinguish "an artefact was repaired" from "an artefact was deleted", and the
//! THIRD count is the direction neither this gate nor its sibling ran before:
//! a kernel the emitter accepts whose `.cpp` is simply absent. The sweep found
//! SEVENTEEN of those across these four corpora -- 7 of them kimi_k3's KDA
//! projections, whose GLM 5.3 twins were all recorded and device-verified -- and all
//! seventeen are now landed, so the pinned value is 0 and stays 0. That is the
//! corpus's own rule, asserted as a hard failure in the playground16 sibling: "a
//! kernel that emits and is not recorded makes the corpus totals wrong". It is a
//! COUNT here rather than that assertion so the number cannot move silently in
//! either direction -- a kernel newly accepted by the emitter raises it, and the
//! message says to land the artefact rather than leaving the corpus mute about it.
//!
//! Three of the seventeen were worse than mute: deepseek_v41's two gemm_f16_lmhead
//! rows and glm_5_3's dense_gate_up_silu recorded an `ascendc_refused` UB-overflow
//! message for a bound the emitter no longer hits. A stale REFUSAL is the same
//! defect class as a stale artefact, pointing the other way: it understates the
//! emitter and makes the corpus's refusal spread wrong. Nothing re-ran those claims
//! either, for the same reason.
//!
//! A cargo integration test rather than a script, so it cannot skip:
//! `CARGO_BIN_EXE_pto-to-ascendc` is always present, and a skip that returns
//! success is how a gate stops being one.
//!
//! A FOURTH count was added 2026-11-08, and it is the same defect one hop
//! further out. Each corpus manifest records an `ascendc_sha256_16` per kernel,
//! and NOTHING compared it to the artefact beside it. The field was wrong
//! wherever it mattered most: on glm_5_3 it was STALE for all six artefacts the
//! same-day repair had rewritten and ABSENT for all nine the KDA work had added,
//! and every one of those fifteen rows carries a device PASS. kimi_k3's
//! kda_delta was stale too. That matters because `ascendc_sha256_16` is what a
//! DEVICE gate has to trust: the device host keeps its own copy of the corpus,
//! and attributing a device run to repo bytes means comparing the two by hash.
//! The campaign's own records were unharmed -- their full sha256 matches these
//! files and all sixteen were banked after the repair -- so this was a field
//! rotting quietly, not a wrong result. Pinning it here is what stops it
//! rotting again, and it is the precondition for the device-side gate rather
//! than a second opinion about the same bytes.

use std::path::{Path, PathBuf};
use std::process::Command;

fn corpus(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../bench")
        .join(name)
        .canonicalize()
        .unwrap_or_else(|e| panic!("the {name} corpus must be reachable: {e}"))
}

fn tiles(root: &Path) -> Vec<PathBuf> {
    let dir = root.join("tile");
    let mut v: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "tile"))
        .collect();
    v.sort();
    v
}

fn tmp(tag: &str, name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("ascendc_{tag}_{}_{name}", std::process::id()));
    p
}

/// Every `"ascendc_sha256_16": "<16 hex>"` in a corpus manifest, keyed by the
/// kernel stem that precedes it. Parsed by hand rather than with a JSON crate:
/// this crate has no JSON dependency, and adding one to a gate so it can read a
/// 16-character field is a worse trade than reading the two lines directly.
///
/// INDENTATION, not `trim()`, decides what a stem is. The manifests are written
/// with `indent=2`, so a kernel entry opens at exactly four spaces and its
/// fields sit at six. The first version of this trimmed each line and took any
/// `"<name>": {` as a stem, which made the nested `"pto_chunked": {` object a
/// kernel and filed the hash below it under that name -- so the gate reported
/// glm_5_3/dense_gate_up_silu as having no recorded hash when the manifest
/// records one two lines further down. A gate that misparses in the direction of
/// failing loudly is still a gate that lies about which row is wrong.
fn manifest_hashes(root: &Path) -> std::collections::BTreeMap<String, String> {
    manifest_field(root, "ascendc_sha256_16")
}

/// Every `"<key>": "<string value>"` in a corpus manifest, keyed by the kernel
/// stem that precedes it. `manifest_hashes` is this with the AscendC hash field;
/// `every_recorded_manifest_hash_names_the_file_it_hashes` needs the path fields
/// and the other two columns' hashes as well, and they are all the same shape.
fn manifest_field(root: &Path, key: &str) -> std::collections::BTreeMap<String, String> {
    let text = std::fs::read_to_string(root.join("manifest.json"))
        .unwrap_or_else(|e| panic!("{}: {e}", root.join("manifest.json").display()));
    let mut out = std::collections::BTreeMap::new();
    let mut stem = String::new();
    // INDENTATION IS THE SCHEMA here, and it has to be read at three depths rather
    // than two. The manifests are written with `indent=2`, so the `"kernels"` object
    // opens at two spaces, a kernel entry at four, and its own fields at six.
    //
    // Two spaces: only fields of the `kernels` object are kernels. `"pto"` occurs at
    // depth 4 as a field of OTHER top-level objects (seven times across the four
    // manifests, e.g. a prose description of the column), and attributing those to
    // whichever depth-4 object name came last is how a parser invents a kernel.
    //
    // Six spaces exactly, not `trim_start()`: a kernel may contain a nested object
    // (`"pto_chunked": {`) whose fields sit at eight, and `"pto"` occurs there once.
    // The AscendC-only predecessor of this function used `trim_start()` and was
    // merely lucky -- `ascendc_sha256_16` happens to occur at depth 6 and nowhere
    // else, while every PATH key occurs at all three depths.
    let mut in_kernels = false;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("  \"") {
            in_kernels = rest.starts_with("kernels\": {");
            continue;
        }
        if !in_kernels {
            continue;
        }
        if let Some(rest) = line.strip_prefix("    \"") {
            if let Some(name) = rest.strip_suffix("\": {") {
                stem = name.to_string();
                continue;
            }
        }
        if let Some(rest) = line.strip_prefix(&format!("      \"{key}\": \"")) {
            if let Some(value) = rest.split('"').next() {
                assert!(
                    !stem.is_empty(),
                    "{}: a `{key}` at kernel-field depth before any kernel entry",
                    root.display()
                );
                out.insert(stem.clone(), value.to_string());
            }
        }
    }
    out
}

fn sha256_16(bytes: &[u8]) -> String {
    // A 200-line dependency-free SHA-256 would be the wrong trade here: the
    // artefact bytes are already compared directly above, so this gate only has
    // to agree with whatever the generator wrote. `sha256sum` is what the
    // manifests were produced with and what the device-side gate reads.
    use std::io::Write;
    let mut child = Command::new("shasum")
        .args(["-a", "256"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .expect("shasum runs");
    child
        .stdin
        .as_mut()
        .expect("shasum stdin")
        .write_all(bytes)
        .expect("write to shasum");
    let out = child.wait_with_output().expect("shasum finishes");
    assert!(out.status.success(), "shasum failed");
    String::from_utf8_lossy(&out.stdout)
        .split_whitespace()
        .next()
        .expect("shasum prints a digest")
        .chars()
        .take(16)
        .collect()
}

/// (identical, refused, emits-but-unrecorded) for one corpus, asserting
/// byte-identity for every artefact that IS recorded, and that the manifest's
/// recorded hash is the hash of the artefact beside it.
fn walk(name: &str) -> (usize, usize, usize) {
    let root = corpus(name);
    let recorded_hashes = manifest_hashes(&root);
    let (mut identical, mut refused, mut unrecorded) = (0, 0, 0);

    for src in tiles(&root) {
        let stem = src.file_stem().unwrap().to_string_lossy().to_string();
        let stored = root.join(format!("ascendc/{stem}.cpp"));
        let out = tmp(name, &format!("{stem}.cpp"));

        let run = Command::new(env!("CARGO_BIN_EXE_pto-to-ascendc"))
            .arg(&src)
            .arg("-o")
            .arg(&out)
            .output()
            .expect("pto-to-ascendc runs");

        if !run.status.success() {
            assert!(
                !stored.exists(),
                "{name}/{stem}: the emitter refuses it, but a .cpp is checked in. \
                 Either the artefact predates a tightened check, or the check \
                 regressed -- and a refused kernel with an artefact beside it is \
                 exactly the shape that lets a device PASS outlive its source.\n{}",
                String::from_utf8_lossy(&run.stderr)
            );
            refused += 1;
            continue;
        }

        if !stored.exists() {
            // Emits and is not recorded. Counted, not failed: see the module note.
            unrecorded += 1;
            let _ = std::fs::remove_file(&out);
            continue;
        }

        let bytes = std::fs::read(&stored).unwrap();
        assert_eq!(
            std::fs::read(&out).unwrap(),
            bytes,
            "{name}/{stem}: the checked-in AscendC differs from what this emitter \
             produces. Regenerate it in the same change that altered the emitter -- \
             and if a device run is recorded against the old bytes, that run no \
             longer attributes to this artefact and must be re-run."
        );
        // The manifest's own record of these bytes. A device run is attributed to
        // this artefact by comparing the device's copy against THIS field, so a
        // stale or missing value makes the attribution unprovable rather than
        // merely untidy.
        let want = recorded_hashes.get(&stem).unwrap_or_else(|| {
            panic!(
                "{name}/{stem}: a recorded artefact with no `ascendc_sha256_16` in \
                 {name}/manifest.json. Every checked-in .cpp needs one: it is what a \
                 device-side gate compares the device's copy against, and without it a \
                 device PASS cannot be attributed to these bytes."
            )
        });
        assert_eq!(
            &sha256_16(&bytes),
            want,
            "{name}/{stem}: manifest.json records ascendc_sha256_16 {want}, which is \
             not the hash of the artefact beside it. Update the field in the same \
             change that regenerates the artefact."
        );
        identical += 1;
        let _ = std::fs::remove_file(&out);
    }
    (identical, refused, unrecorded)
}

#[test]
fn deepseek_v41_ascendc_is_reproducible() {
    // 10 refusals, and they are NOT one bound hit ten times: two blocked argmaxes
    // (the scan carries one running index), four missing arms
    // (__tile_attention_decode_f32, __tile_matmul_i8_acc_i32_dequant_f16 twice,
    // __tile_mul_mv_id_mxfp4_pk_f32, __tile_quantize_f32_i8, __tile_topk_f32) and
    // two UB overflows (the embedding's gather source, and rmsnorm_8x5120 needing
    // 163840 bytes).
    // 12 refusals since 2026-11-08, and the two that joined were already refused
    // ONCE before: both gemm_f16_lmhead variants. Their refusal was retracted as
    // stale that same day on the grounds that `pto-to-ascendc` exits 0, and the
    // retraction was wrong -- the retracted text named "buffer 0 needs 81920
    // elements (163840 bytes)" and the artefact it licensed still did exactly
    // that, faulting 507035 at `aclrtSynchronizeStream` on an Ascend910B2.
    //
    // ONE TOKEN caused it. `mlir_to_cpp` stands its chunker down for a body that
    // accumulates across the width, by a case-sensitive substring search over the
    // emitted text; the needle was `tile_matmul` and the matmul arm writes
    // `TILE_DEGENERATE_MATMUL`, so it matched nothing. Measured: 0 corpus
    // artefacts contain `tile_matmul`, 5 contain `TILE_DEGENERATE_MATMUL`. The
    // chunker then chunked a matmul's flat element count while the arm went on
    // indexing the unchunked extents, and the capacity check passed because it
    // sized the chunk. With the needle matching, the body reaches that check
    // unchunked and is refused with its own computed bound.
    //
    // So this count moving from (7, 10, 0) is the emitter telling the truth about
    // two kernels it cannot express. `mlir_to_pto` CAN express them -- a real
    // blocked matmul, `scf.for %n_i` over 202 N-blocks of 64 and `scf.for %k_i`
    // over 20 K-blocks of 256 with `pto.tmatmul.acc` -- so the kernels are not
    // unexpressible, this emitter simply has no blocked arm. Giving it one is the
    // way these two come back.
    //
    // 6 identical from 2026-11-08: rmsnorm_v41_8x5120 joined with the ROW-BLOCKED
    // arm. It is 8 x 5120 -- 163840 bytes per buffer against 47104 as one tile, and
    // 20480 per row -- so the kernel's own body now runs once per row with every GM
    // tensor rebased by r*cols and the row-count scalar pinned to 1. Pinning it to 1
    // is what makes this more than a fit: the emitted row-norm has a vector fast path
    // for a single row and a per-row scalar GetValue/SetValue fallback otherwise, so
    // the arm also moves the body onto the path already device-verified at this exact
    // width.
    //
    // THE ARM'S FIRST VERSION TOOK THE TWO GEMM ROWS, which are 16 x 5120 and so
    // "fit once per row" -- re-committing from the other direction the defect the
    // paragraph above reinstated the refusals for. It is now gated on row
    // INDEPENDENCE, keyed on `TILE_DEGENERATE_MATMUL` and `__mm_`: text those arms
    // actually emit, present in both gemm bodies and absent from the row norm's. A
    // per-row reduction is independent and still qualifies, which is why `ReduceSum`
    // is deliberately not a needle there.
    //
    // 2026-11-08, TOP-K: `mlir_to_cpp` gains `__tile_topk_f32` (and
    // `__tile_sigmoid_f32` beside it, which only the K3 router needs). Top-k is k
    // selection passes of `ReduceMax` plus a scalar first-hit scan, masking each
    // taken position out of a WORKING COPY -- the source survives, because the
    // sigmoid variant's `.tile` passes it twice. The k values leave through the
    // kernel's own store; the k indices through the GM pointer the intrinsic takes
    // as its third operand, as int32, the same decision as the argmax index.
    //
    // The arms needed a line in the BUFFER COUNT as well, and the first build did
    // not have one: `router_sigmoid_topk_k3_1x896_k16` named buf1 through buf9 and
    // declared only buf0, because the arms allocate from the emitter's own counter
    // while the TBufs come from the analysis of the MLIR. That C++ would not have
    // compiled, and nothing in the build would have said so -- it was caught by
    // reading the artefact.
    //
    // 8 identical from 2026-11-08: the EMBEDDING joins, and not by an emitter arm.
    // `__tile_embedding_f32` declares its weight table as a tile, so no arm can
    // satisfy it at vocabulary scale -- deepseek_v41's is 2.65 GB of f32 against a
    // 190 KB unified buffer. The generator now emits the operation as what it is at
    // decode shape, one row copy from GM: read the index, multiply by the row
    // width, `__tile_load_at_f32` at that element offset. The refusal it retires
    // was correct about the bound and wrong about the lowering, which is why it is
    // kept beside the row rather than deleted.
    // 10 identical from 2026-11-09, and the two that joined are the lmheads.
    // `mlir_to_cpp` gains a BLOCKED CUBE ARM: the whole-kernel
    // `load A; load B; matmul; store` shape, recognised when an operand exceeds
    // the whole addressable UB, lowered to a K/N-blocked nest that stages
    // GM -> L1 -> L0A/L0B a block at a time and leaves L0C through FixPipe. The
    // refusal it retires was correct about the resource and wrong about there
    // being no lowering, which is why it is kept beside each row under
    // `ascendc_refusal_retired_2026_11_09_blocked_cube` rather than deleted.
    //
    // Device-verified on the 910b at ORACLE 48 before these bytes were checked
    // in, on BOTH toolchains and on the artefact the TOOL emits rather than one
    // from a standalone entry point: 16x5120x12928 -> 206848 elements wrong=0,
    // 16x5120x16160 -> 258560 wrong=0, max_abs 0.000e+00 on each. What `wrong=0`
    // does and does not prove there -- the oracle's fill has period 17 in N, so
    // it is 272 independent checks repeated -- is in the campaign's
    // `evidence/0025-the-cube-arm-is-correct`.
    // 11 identical from 2026-11-09, and the one that joined is quant_i8_v41_8x5120.
    // `mlir_to_cpp` gains TWO arms, because this kernel needed two and was filed
    // under one: `__tile_quantize_f32_i8` and `__tile_store_i8`. The store drew no
    // refusal of its own -- the quantize refusal fires first and the store is never
    // reached -- so the asymmetry census did not list it as missing. One blocker
    // hiding another.
    //
    // The arithmetic is READ, not inferred: tile_std/src/tile.rs:1014 declares
    // "round(x / scale), clamped to [-128, 127]" and mlir_to_pto.rs:874 corroborates
    // with a broadcast divide, so the arm divides by a Duplicate'd tile rather than
    // multiplying by a reciprocal. The route is f32 -> f16 (CAST_NONE) -> s8
    // (CAST_RINT), because the part has NO f32 -> s8 convert: a direct cast is
    // accepted statically, garbage on device, and can raise 507035.
    //
    // It also needed a ONE-ELEMENT exception to the partial-block load guard, for
    // the 1 x 1 scale tile. That guard stays; at exactly one element its premise
    // fails, because the pad's measured block-wide write lands in padding a
    // 1 x 1 tile does not have. Scoped to one element, not "small".
    //
    // NOT DEVICE-VERIFIED YET, unlike the two lmheads above, and the record says so
    // rather than implying it: ORACLE 51 is calibrated five ways and will NAME which
    // convention the hardware implements, but this kernel has not run. It is checked
    // in because it EMITS and an emitting kernel with no artefact fails this very
    // gate -- the artefact's presence is a lowering claim, not a correctness one.
    // (14, 3, 0) from 2026-11-10: both argmaxes became emitted .cpp with the
    // generated per-row i32 arm. The three remaining refusals are mx4pk and the
    // two i8 GEMMs.
    // (15, 2, 0) from 2026-11-10: mx4pk became the 15th emitted .cpp with the
    // scalar packed-MXFP4 arm. The two remaining refusals are the i8 GEMMs.
    // (15, 1, 1) from 2026-11-11: `gemm_i8_v41_16x5120x2304` emits now -- the
    // i8 GEMM is two kernels, an AIV cast/scale kernel and the verified f16
    // blocked cube -- but no .cpp had been recorded for it yet, so it was
    // UNRECORDED, and the 1x i8 GEMM still refused (its 6-arg call carried no
    // scale operand).
    // (16, 0, 1) from 2026-11-11: the 1x `.tile` was corrected to the 7-arg call
    // and `mlir_to_cpp` routes M=1 through a vector GEMV after the same AIV
    // dequant, so the corpus has no AscendC refusal left. The 16x .cpp is now
    // stored (identical); the 1x emits but has no recorded .cpp, which is the
    // one unrecorded kernel.
    assert_eq!(walk("deepseek_v41"), (16, 0, 1));
}

#[test]
fn glm_5_3_ascendc_is_reproducible() {
    // 19 identical INCLUDING the three repaired on 2026-11-08 -- proj_kv_a_mqa,
    // proj_q_a and proj_shexp_down went from a 3-buffer f32 dot to the 5-buffer
    // w8a8 form their own .tile declares -- plus kda_delta and routed_gate_up,
    // whose buffer plans had drifted.
    // 23 identical from 2026-11-08: lm_head at 154880, proj_kv_b at 28672 and
    // proj_q_b at 16384 joined when the BLOCKED-REDUCTION arm landed. All three are
    // the same body -- a width-wide `Mul` summed once along one row -- and all three
    // had refused on the per-buffer UB bound. The arm runs the chain per chunk into
    // a tile accumulator and row-sums it once after the loop, which is the chain
    // `mlir_to_pto` already emitted for these kernels.
    // 4 refusals: two UB overflows (argmax at 619520 bytes and the embedding's
    // gather source) and two missing arms (__tile_attention_decode_f32,
    // __tile_topk_f32).
    // 0 unrecorded.
    //
    // 2026-11-08, the CHUNKED ARGMAX arm: the vocabulary argmax joins. Two things had
    // to change together, and the second is the one worth remembering. The arm itself
    // is the fourth rescue -- an argmax carries a (max, index) PAIR whose index means
    // nothing without its chunk base, so neither chunking nor a tile accumulator
    // helps, and the partials reduce pairwise on the scalar unit. The other change is
    // that this kernel's `.tile` declared `__tile_store_f32` for an INDEX, which
    // forced the arm to write `TILE_SUBSTITUTION: replacement=f32-index-storage` --
    // a marker about a choice the emitter never made. The generator now declares
    // `__tile_store_i32`, mlir_to_cpp has that store (mlir_to_pto always did), and
    // the index buffer is typed int32_t, so the store arm's dtype check enforces the
    // pair. A marker that describes the source's own declaration is not a
    // substitution, and it was capping the kernel's coverage row at `tiled`.
    //
    // 2026-11-08, TOP-K: `mlir_to_cpp` gains `__tile_topk_f32` (and
    // `__tile_sigmoid_f32` beside it, which only the K3 router needs). Top-k is k
    // selection passes of `ReduceMax` plus a scalar first-hit scan, masking each
    // taken position out of a WORKING COPY -- the source survives, because the
    // sigmoid variant's `.tile` passes it twice. The k values leave through the
    // kernel's own store; the k indices through the GM pointer the intrinsic takes
    // as its third operand, as int32, the same decision as the argmax index.
    //
    // The arms needed a line in the BUFFER COUNT as well, and the first build did
    // not have one: `router_sigmoid_topk_k3_1x896_k16` named buf1 through buf9 and
    // declared only buf0, because the arms allocate from the emitter's own counter
    // while the TBufs come from the analysis of the MLIR. That C++ would not have
    // compiled, and nothing in the build would have said so -- it was caught by
    // reading the artefact.
    //
    // 26 identical from 2026-11-08: the EMBEDDING joins, and not by an emitter arm.
    // `__tile_embedding_f32` declares its weight table as a tile, so no arm can
    // satisfy it at vocabulary scale -- glm_5_3's is 2.54 GB of f32 against a
    // 190 KB unified buffer. The generator now emits the operation as what it is at
    // decode shape, one row copy from GM: read the index, multiply by the row
    // width, `__tile_load_at_f32` at that element offset. The refusal it retires
    // was correct about the bound and wrong about the lowering, which is why it is
    // kept beside the row rather than deleted.
    // (27, 0, 0) from 2026-11-10: the decode attention became the 27th emitted
    // .cpp when the 1-query blocked arm landed. GLM has no AscendC refusals left.
    assert_eq!(walk("glm_5_3"), (27, 0, 0));
}

#[test]
fn kimi_k3_ascendc_is_reproducible() {
    // 14 identical, including kda_delta_k3_1x128 repaired on 2026-11-08 (its
    // committed 512-byte buffer against the 1024 the emitter now computes).
    // 24 identical from 2026-11-08: lm_head at 163840, proj_kv_b at 24576 and
    // proj_q_b at 18432 joined with the blocked-reduction arm, the same three shapes
    // as GLM 5.3's.
    // 4 refusals: two UB overflows (argmax at 655360 bytes and the embedding's
    // gather source) and two missing arms (__tile_attention_decode_f32,
    // __tile_sigmoid_f32).
    // 0 unrecorded.
    //
    // 2026-11-08, the CHUNKED ARGMAX arm: the vocabulary argmax joins. Two things had
    // to change together, and the second is the one worth remembering. The arm itself
    // is the fourth rescue -- an argmax carries a (max, index) PAIR whose index means
    // nothing without its chunk base, so neither chunking nor a tile accumulator
    // helps, and the partials reduce pairwise on the scalar unit. The other change is
    // that this kernel's `.tile` declared `__tile_store_f32` for an INDEX, which
    // forced the arm to write `TILE_SUBSTITUTION: replacement=f32-index-storage` --
    // a marker about a choice the emitter never made. The generator now declares
    // `__tile_store_i32`, mlir_to_cpp has that store (mlir_to_pto always did), and
    // the index buffer is typed int32_t, so the store arm's dtype check enforces the
    // pair. A marker that describes the source's own declaration is not a
    // substitution, and it was capping the kernel's coverage row at `tiled`.
    //
    // 2026-11-08, TOP-K: `mlir_to_cpp` gains `__tile_topk_f32` (and
    // `__tile_sigmoid_f32` beside it, which only the K3 router needs). Top-k is k
    // selection passes of `ReduceMax` plus a scalar first-hit scan, masking each
    // taken position out of a WORKING COPY -- the source survives, because the
    // sigmoid variant's `.tile` passes it twice. The k values leave through the
    // kernel's own store; the k indices through the GM pointer the intrinsic takes
    // as its third operand, as int32, the same decision as the argmax index.
    //
    // The arms needed a line in the BUFFER COUNT as well, and the first build did
    // not have one: `router_sigmoid_topk_k3_1x896_k16` named buf1 through buf9 and
    // declared only buf0, because the arms allocate from the emitter's own counter
    // while the TBufs come from the analysis of the MLIR. That C++ would not have
    // compiled, and nothing in the build would have said so -- it was caught by
    // reading the artefact.
    //
    // 27 identical from 2026-11-08: the EMBEDDING joins, and not by an emitter arm.
    // `__tile_embedding_f32` declares its weight table as a tile, so no arm can
    // satisfy it at vocabulary scale -- kimi_k3's is 4.70 GB of f32 against a
    // 190 KB unified buffer. The generator now emits the operation as what it is at
    // decode shape, one row copy from GM: read the index, multiply by the row
    // width, `__tile_load_at_f32` at that element offset. The refusal it retires
    // was correct about the bound and wrong about the lowering, which is why it is
    // kept beside the row rather than deleted.
    // (28, 0, 0) from 2026-11-10: the decode attention became the 28th emitted
    // .cpp with the runtime-S blocked arm. Kimi has no AscendC refusals left.
    assert_eq!(walk("kimi_k3"), (28, 0, 0));
}

#[test]
fn engine_ascendc_is_reproducible() {
    // The engine corpus is NOT scored by the migration campaign, and it drifted the
    // most: 7 of the 13 stale artefacts were here, every one a q8mv or w8a8
    // projection whose buffer plan had moved. Gated anyway -- an unscored corpus
    // whose artefacts carry device PASSes is the same defect, just unmeasured.
    // 0 refusals: every tile here emits.
    // 7 unrecorded: the six attention-stage probes and w8a8_probe_1x1536_k4096.
    assert_eq!(walk("engine"), (25, 0, 0));
}

/// A chunked body carries its accumulator across the chunk boundary.
///
/// WHY THIS EXISTS, measured 2026-11-08. `mlir_to_cpp` stands its chunker down for
/// a body that accumulates across the width by a case-sensitive substring search
/// over the emitted text. One needle, `tile_matmul`, matched NOTHING -- the matmul
/// arm writes `TILE_DEGENERATE_MATMUL` -- so the two `gemm_f16_lmhead` artefacts
/// were chunked on their flat element count while the arm went on indexing the
/// unchunked logical extents. Each then held a `DataCopy` of 163840 bytes into a
/// 51712-byte buffer and a `GetValue` at element 66191359 of a 25856-element one,
/// and the UB capacity check passed because it sized the chunk. On an Ascend910B2
/// that faults 507035 at `aclrtSynchronizeStream`.
///
/// WHY IT WAS REWRITTEN THE SAME DAY, and this is the part worth reading. The first
/// version of this gate opened by saying it "asserts the INVARIANT on the artefact
/// rather than the spelling of the needle, because a needle is exactly what went
/// silently dead" -- and then asked "is this body chunked?" by searching for the
/// single literal `// chunked:`. There were two needles in it, and it hardened one.
///
/// Hours later `mlir_to_cpp` grew a blocked-reduction arm whose chunk loop is
/// introduced by `// blocked reduction: <W> elements as <NC> x <CH>, tail <T>`.
/// Measured: the emitter writes exactly two chunk-window comments (mlir_to_cpp.rs
/// at the `// chunked:` and `// blocked reduction:` writelns), and the corpus holds
/// 4 artefacts with the first and 6 with the second. All six were invisible here.
///
/// Worse for the old statement than a blind spot: those six REFUTE it. They are
/// chunked AND they accumulate, and they are right -- all six are device-verified
/// on an Ascend910B2 at max_rel between 1.8e-08 and 8.2e-08. So "a body that
/// accumulates must never also be chunked" was never the invariant; it was true of
/// the one broken arm that existed when it was written.
///
/// The real invariant is about WHERE the accumulation happens:
///
///   * nothing that accumulates across the width may sit INSIDE the chunk loop,
///     because only one chunk is resident and such an op reads the full extent;
///   * an accumulation AFTER the loop is correct, and is how the blocked-reduction
///     arm works -- a tile accumulator zeroed before the loop, `Add(acc, acc, x)`
///     per chunk, one `ReduceSum` over it afterwards.
///
/// Both tests are keyed on GENERATED CODE rather than on comments: `constexpr int
/// __CH` and `for (int __c` are what the loop cannot exist without, while a comment
/// can be renamed by any later arm. Measured: `__CH` appears in exactly the 10
/// artefacts that carry one of the two comment spellings and in no others, so the
/// equality asserted below is the needle-death detector -- a third arm with a third
/// comment makes the two counts disagree and names the artefact.
///
/// One more correction. The old version skipped any artefact containing
/// `tile_matvec_w8a8_gm`, described as the legitimately chunked matvec. It was dead
/// on arrival: at the commit that introduced it, no chunked artefact contained
/// `tile_matvec` at all, and none does now (the five artefacts that call it carry
/// no `__CH`). An untested exemption inside a gate written to catch an untested
/// needle. It is gone; if a chunked w8a8 matvec ever lands, this gate should make
/// someone look at it rather than wave it through.
///
/// Flat degenerate matmuls are fine and are not what this checks: three corpus
/// artefacts carry the marker with no chunking, and one of them,
/// `engine/attn_sink_engine_64heads`, is device-verified. A scalar `GetValue` loop
/// on the vector core is slow, not wrong. The composition is the defect.
#[test]
fn a_chunked_body_carries_its_accumulator() {
    // Markers the emitter writes for an op that accumulates across the width.
    // `tile_matvec` is lowercase on purpose: that is the spelling its arm emits.
    const ACCUMULATES: [&str; 4] = [
        "TILE_DEGENERATE_MATMUL",
        "ReduceSum",
        "ReduceMax",
        "tile_matvec",
    ];
    // The chunk-window comments the emitter can write. Pinned so a NEW arm with a
    // new comment is caught by the equality below rather than silently skipped.
    const WINDOW_COMMENTS: [&str; 2] = ["// chunked:", "// blocked reduction:"];

    let mut inspected = 0usize;
    let mut chunked = 0usize;
    let mut commented = 0usize;

    for name in ["deepseek_v41", "glm_5_3", "kimi_k3", "engine"] {
        let dir = corpus(name).join("ascendc");
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        let mut paths: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("cpp"))
            .collect();
        paths.sort();
        for path in paths {
            let text = std::fs::read_to_string(&path).unwrap();
            inspected += 1;
            if WINDOW_COMMENTS.iter().any(|c| text.contains(c)) {
                commented += 1;
            }
            // STRUCTURE, not a comment: the chunk loop needs both of these.
            if !text.contains("constexpr int __CH") {
                continue;
            }
            chunked += 1;
            let lines: Vec<&str> = text.lines().collect();
            let open = lines
                .iter()
                .position(|l| l.contains("for (int __c = 0;"))
                .unwrap_or_else(|| {
                    panic!(
                        "{}: declares `constexpr int __CH` and has no `for (int __c = 0;` \
                         loop. The chunk width exists with no loop over it, or the loop \
                         header was respelled -- either way this gate can no longer find \
                         the span it has to check, and that must fail rather than pass.",
                        path.display()
                    )
                });
            let indent = lines[open].len() - lines[open].trim_start().len();
            // The matching close: the first line that is a lone `}` at the loop's own
            // indent. Asserted rather than defaulted, because a span this gate cannot
            // delimit is a span it cannot police.
            let close = lines
                .iter()
                .enumerate()
                .skip(open + 1)
                .find(|(_, l)| {
                    l.trim() == "}" && (l.len() - l.trim_start().len()) == indent
                })
                .map(|(k, _)| k)
                .unwrap_or_else(|| {
                    panic!(
                        "{}: the chunk loop opening at line {} has no closing `}}` at its \
                         own indent, so its body cannot be delimited.",
                        path.display(),
                        open + 1
                    )
                });

            for marker in ACCUMULATES {
                if let Some(hit) = lines[open..=close].iter().position(|l| l.contains(marker)) {
                    panic!(
                        "{}: `{marker}` accumulates across the width and sits INSIDE the \
                         chunk loop (line {}). Only one chunk is resident there, while such \
                         an op reads the full logical extent -- the composition is out of \
                         bounds by construction and faults on device. Either hoist the \
                         accumulation out of the loop into a carried accumulator, as the \
                         blocked-reduction arm does, or refuse the kernel in mlir_to_cpp.",
                        path.display(),
                        open + hit + 1
                    );
                }
                if lines[close + 1..].iter().any(|l| l.contains(marker)) {
                    // Correct shape -- but only if something carried the partial sums
                    // across the boundary. A reduce after an unaccumulated loop reads
                    // whichever chunk happened to be left resident.
                    assert!(
                        lines[..open]
                            .iter()
                            .any(|l| l.contains("Duplicate") && l.contains("(float)0")),
                        "{}: `{marker}` runs AFTER the chunk loop and no accumulator is \
                         zeroed before it (`Duplicate(<buf>, (float)0, ...)`). Reducing \
                         after a loop that did not accumulate reads only the last chunk.",
                        path.display()
                    );
                }
            }
        }
    }

    assert_eq!(
        chunked, commented,
        "{chunked} artefact(s) declare `constexpr int __CH` but {commented} carry one of \
         the chunk-window comments {WINDOW_COMMENTS:?}. A mismatch means mlir_to_cpp grew \
         a chunking arm whose window comment is not in that list -- which is how the \
         first version of this gate went blind to six artefacts. Add the new spelling, \
         and check whether anything else keys on the old one."
    );
    assert!(
        chunked >= 10,
        "expected at least 10 chunked artefacts, saw {chunked} -- a gate that finds no \
         chunked body polices nothing"
    );
    assert!(
        inspected >= 60,
        "expected to inspect at least 60 stored artefacts, saw {inspected}"
    );
}

/// Every recorded per-column hash is the hash of the file its own row names.
///
/// WHY, measured 2026-11-08. The module note above records adding the
/// `ascendc_sha256_16` cross-check and says plainly that NOTHING had compared the
/// field to the artefact beside it. It then checked ONE column. The corpora record
/// a hash for three -- `pto_sha256_16`, `ascendc_sha256_16`, `tile_rs_sha256_16` --
/// and `walk` above reads only the third of those spellings.
///
/// The ungated two had rotted, and the split is the measurement:
///
/// | field | recorded | wrong before this gate |
/// |---|---|---|
/// | `ascendc_sha256_16` (gated since 2026-11-08) | 71 | 0 |
/// | `pto_sha256_16` | 43 | 4 |
/// | `tile_rs_sha256_16` | 5 | 3 |
///
/// Seven of the forty-eight ungated hashes named bytes that are not in the tree,
/// while none of the seventy-one gated ones did.
///
/// Neither class is a wrong RESULT, and saying so matters as much as finding them:
/// in all seven cases the checked-in file is byte-identical to what the current
/// emitter produces (`glm53_corpus_matches_its_emitter` and `ptoas-rs` both
/// re-derive it), so the artefacts are right and the manifest's record of them was
/// wrong. The two provenances differ:
///
///   * The four `pto_sha256_16` rows were CORRECT until `3c87883b5`, whose message
///     is "land the regenerated PTO artefacts, and derive every manifest from
///     them". That commit rewrote the four `.pto` files and left their hashes at
///     the pre-regeneration values -- so the claim in its own subject line was
///     false for those rows, and eight later commits touched the same manifest
///     without noticing.
///   * The three `tile_rs_sha256_16` rows have been wrong since the initial commit
///     that introduced them. They were never derived from anything in the tree.
///
/// The reason a column-complete gate is worth more than a third opinion about the
/// same bytes: the regeneration gates (`*_corpus_matches_its_emitter`, `walk`
/// above) compare TREE against EMITTER and never read the manifest, so a hash can
/// rot without any of them moving. This gate is the only thing that reads the
/// manifest's own claim, and `pto_sha256_16` is what the campaign's device-artefact
/// gate compares a device host's copy against for the PTO column, exactly as
/// `ascendc_sha256_16` serves the AscendC one.
///
/// Every failure is collected rather than panicking on the first, because the
/// defect class is a FIELD rotting across rows and the useful output is which rows.
#[test]
fn every_recorded_manifest_hash_names_the_file_it_hashes() {
    // (path field, hash field) for the three emitted columns. The path comes out of
    // the manifest rather than being rebuilt from a `<dir>/<stem>.<ext>` convention:
    // a row that names a DIFFERENT file is exactly the drift being looked for, and a
    // convention would silently check the file the row should have named.
    const COLUMNS: [(&str, &str); 3] = [
        ("pto", "pto_sha256_16"),
        ("ascendc", "ascendc_sha256_16"),
        ("tile_rs", "tile_rs_sha256_16"),
    ];
    let mut checked = 0usize;
    let mut wrong: Vec<String> = Vec::new();

    for name in ["deepseek_v41", "glm_5_3", "kimi_k3", "engine"] {
        let root = corpus(name);
        for (path_key, hash_key) in COLUMNS {
            let paths = manifest_field(&root, path_key);
            for (stem, hash) in manifest_field(&root, hash_key) {
                let Some(rel) = paths.get(&stem) else {
                    wrong.push(format!(
                        "{name}/{stem}: records {hash_key} and no `{path_key}` path, \
                         so the hash names nothing"
                    ));
                    continue;
                };
                let file = root.join(rel);
                let Ok(bytes) = std::fs::read(&file) else {
                    wrong.push(format!(
                        "{name}/{stem}: {hash_key} is recorded for {rel}, which is not \
                         in the tree -- a hash that cannot be re-derived attributes nothing"
                    ));
                    continue;
                };
                checked += 1;
                let got = sha256_16(&bytes);
                if got != hash {
                    wrong.push(format!(
                        "{name}/{stem}: {hash_key} records {hash}, {rel} hashes {got}"
                    ));
                }
            }
        }
    }

    assert!(
        wrong.is_empty(),
        "{} manifest hash row(s) do not hash the file they name. Update the field in \
         the same change that rewrites the artefact; if instead the ARTEFACT is the \
         stale one, regenerate it and re-run any device claim recorded against the \
         old bytes.\n  {}",
        wrong.len(),
        wrong.join("\n  ")
    );
    assert!(
        checked >= 110,
        "expected at least 110 (kernel, column) hash rows across the four corpora, \
         saw {checked} -- a gate that reads no field reports a pass"
    );
}

/// Every needle in a stand-down list names text some arm actually emits.
///
/// WHY THIS EXISTS, and the honest version of why. The comment beside the
/// chunker's stand-down list in `ascendc_emit/src/mlir_to_cpp.rs` says that
/// `argmax` matches no corpus artefact "for a benign reason: every argmax kernel
/// refuses on the UB budget before an artefact exists, so that needle is untested
/// here rather than wrong", and then states that
/// `every_needle_names_text_an_arm_emits` pins the distinction. That test did not
/// exist. The comment was written in the same change that root-caused a needle
/// matching nothing, and it promised a gate instead of writing one -- which is the
/// defect it was fixing, one level up. This is that gate.
///
/// The distinction being pinned is between two needles that both match zero
/// artefacts:
///
///   * `tile_matmul` matched zero because the arm writes `TILE_DEGENERATE_MATMUL`.
///     The needle was WRONG, and the guard it belonged to silently did nothing.
///   * `argmax` matches zero because no argmax kernel ever reaches an artefact --
///     all four in the corpora are refused on the UB budget. The needle is
///     UNTESTED, not wrong, and deleting it would be a regression.
///
/// A count alone cannot tell those apart, so a zero-match needle has to be
/// declared, with the claim that makes it benign asserted separately. For `argmax`
/// that claim is "no argmax kernel has a stored artefact", checked below against
/// the `.tile` sources rather than taken on trust.
///
/// The lists are PARSED OUT OF THE EMITTER rather than restated here. A copy of a
/// needle list in a test is one more thing that can go stale while reporting a
/// pass, and this file already carries two findings of exactly that shape.
#[test]
fn every_needle_names_text_an_arm_emits() {
    // Zero-match needles that are untested rather than wrong, with the reason.
    const UNTESTED: [(&str, &str); 1] = [(
        "argmax",
        "every argmax kernel is refused on the UB budget before an artefact exists, \
         asserted below against the .tile sources",
    )];

    let emitter = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../ascendc_emit/src/mlir_to_cpp.rs")
        .canonicalize()
        .expect("the AscendC emitter source must be reachable from this crate");
    let src = std::fs::read_to_string(&emitter).unwrap();

    // The two stand-down lists: the generic chunker's, and the blocked-reduction
    // arm's. `ReduceSum` is deliberately absent from the second -- that arm exists
    // precisely to carry a row sum across the boundary -- so they are checked as
    // separate lists rather than merged.
    let lists = [
        ("the chunker's stand-down list", "let reduces = ["),
        ("the blocked-reduction arm's refusal list", "for bad in ["),
    ];

    // Stored artefacts, read once.
    let mut artefacts: Vec<String> = Vec::new();
    for name in ["deepseek_v41", "glm_5_3", "kimi_k3", "engine"] {
        let dir = corpus(name).join("ascendc");
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("cpp") {
                artefacts.push(std::fs::read_to_string(&path).unwrap());
            }
        }
    }
    assert!(
        artefacts.len() >= 60,
        "expected at least 60 stored artefacts to census needles against, saw {}",
        artefacts.len()
    );

    let mut live: Vec<&str> = Vec::new();
    for (what, anchor) in lists {
        let at = src.find(anchor).unwrap_or_else(|| {
            panic!(
                "{}: `{anchor}` is not in {}. The list was renamed or moved, and a gate \
                 that cannot find it must fail rather than report a pass.",
                what,
                emitter.display()
            )
        });
        let open = at + src[at..].find('[').expect("the anchor contains a `[`");
        let close = open
            + src[open..]
                .find(']')
                .unwrap_or_else(|| panic!("{what}: unterminated needle list"));
        let needles: Vec<&str> = src[open + 1..close]
            .split(',')
            .filter_map(|s| s.trim().strip_prefix('"')?.strip_suffix('"'))
            .collect();
        assert!(
            needles.len() >= 4,
            "{what}: parsed {} needles out of the emitter, which is too few to be the \
             real list -- the parse is broken, not the list: {needles:?}",
            needles.len()
        );

        for needle in &needles {
            let hits = artefacts.iter().filter(|t| t.contains(needle)).count();
            if hits > 0 {
                // LIVE, and that is the end of it. A needle that matches emitted
                // text needs no declaration, and a stale UNTESTED entry beside it
                // is NOT an error -- the entry simply stops being load-bearing,
                // and the claim that justified it stops being asserted below.
                //
                // That tolerance was added 2026-11-08 for a reason worth keeping.
                // The first version treated an UNTESTED entry as a promise that
                // the needle matches NOTHING, so the moment `argmax` acquired an
                // artefact -- a capability landing, which is the good case -- this
                // gate turned red in a direction nobody could act on without
                // reverting the capability. A gate that blocks the thing it exists
                // to measure is worse than no gate. The asymmetry is deliberate:
                // going LIVE is free, going dark needs a declaration.
                live.push(*needle);
                continue;
            }
            let declared = UNTESTED.iter().find(|(n, _)| n == needle);
            assert!(
                declared.is_some(),
                "{what}: needle `{needle}` matches 0 of {} stored artefacts and is not \
                 declared untested. Either the arm writes different text -- which is how \
                 `tile_matmul` silently disabled this guard and let two artefacts chunk an \
                 accumulating body -- or no kernel reaches that arm, in which case add it \
                 to UNTESTED with the claim that makes it benign.",
                artefacts.len()
            );
        }
    }

    // The claim that makes `argmax`'s zero benign, asserted rather than trusted --
    // and asserted ONLY WHILE THE ZERO HOLDS. Once some artefact carries the text,
    // the needle is live by measurement and "no argmax kernel reaches an artefact"
    // is no longer the thing keeping it honest, so re-asserting it would be a gate
    // demanding that a capability stay unimplemented.
    if live.contains(&"argmax") {
        return;
    }
    let mut argmax_tiles = 0usize;
    for name in ["deepseek_v41", "glm_5_3", "kimi_k3", "engine"] {
        let root = corpus(name);
        for src_tile in tiles(&root) {
            let text = std::fs::read_to_string(&src_tile).unwrap();
            if !text.contains("__tile_argmax") && !text.contains("__tile_argmin") {
                continue;
            }
            argmax_tiles += 1;
            let stem = src_tile.file_stem().unwrap().to_string_lossy().to_string();
            let stored = root.join(format!("ascendc/{stem}.cpp"));
            assert!(
                !stored.exists(),
                "{name}/{stem}: an argmax kernel WITH a stored artefact. The `argmax` \
                 needle in the chunker's stand-down list is declared untested on the \
                 grounds that no argmax kernel reaches an artefact; that is no longer \
                 true, so the needle is now live and must be shown to match the text \
                 this artefact contains."
            );
        }
    }
    assert!(
        argmax_tiles >= 4,
        "expected at least 4 argmax .tile sources across the corpora, saw {argmax_tiles} \
         -- if they were removed, the UNTESTED entry for `argmax` has no subject and the \
         needle should be re-examined rather than left declared"
    );
}
