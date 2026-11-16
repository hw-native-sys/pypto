// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! No stored artefact carries a `DataCopy` of a PARTIAL 32-byte block, either way.
//!
//! WHY, measured on device 2026-11-09. A decode embedding read its token id out of
//! global memory through a `1 x 1` tile, which `mlir_to_cpp` emitted as
//! `AscendC::DataCopy(buf0, gm1Tensor, (1) * (1))` -- four bytes. On an
//! Ascend910B2 (`dav-2201`, CANN 8.5.2) that does not fault, does not refuse, and
//! does not deliver the element.
//!
//! # BOTH DIRECTIONS, and that is also not mine
//!
//! This gate judged LOADS only, and its dtype came from the destination's
//! `LocalTensor<T>`. A store names a `GlobalTensor` there and carries its tile on
//! the other side, so a constant-extent store failed as "could not be judged" --
//! loudly, which is the safe direction, but with a message that sent the reader
//! hunting a missing declaration when the truth was a store. Measured by
//! injecting one, 2026-11-09.
//!
//! It judges both now, because the MECHANISM is one. A peer session measured the
//! fourth cell: a plain `DataCopy` STORE of 4 bytes delivers nothing, exactly as a
//! plain 4-byte load does. So plain `DataCopy` carries a block count and rounds
//! its length down to a multiple of 32 in either direction, while `DataCopyPad`
//! carries a byte length and honours it exactly on a store -- and on a load fills
//! the whole destination block with element 0, which is why the load-side pad was
//! withdrawn and the store-side pad has been sound for months.
//!
//! Store coverage here is LATENT: all 211 store sites in the corpus have runtime
//! extents and none is constant, so this half of the gate has zero instances to
//! judge today. It exists so that the first constant-extent store is judged rather
//! than reported as an unreadable dtype.
//!
//! # A MULTIPLE, not a minimum -- and that correction is not mine
//!
//! This file first said "at least one block" and held every extent against an
//! 8-f32 floor. That is insufficient, and a peer session measured why: the DMA
//! length bisheng encodes is the requested byte count rounded DOWN to a multiple
//! of 32. Extracting `.aicore_binary` across 19 extents from one generator
//! differing only in the copy-extent literal, compile-only:
//!
//! ```text
//! req=4 .. req=28   -> byte-identical device binaries   (md5 cf4a419a4ddd)
//! req=32, 36, 60    -> byte-identical TO EACH OTHER     (md5 d1337d5d5427)
//! req=64            -> distinct                         (md5 861bc5cf91a2)
//! req=96, 128       -> distinct from each other
//! ```
//!
//! `floor(req / 32) * 32` reproduces those groupings exactly. So a 36-byte load is
//! not sub-block, would have PASSED the old `bytes < 32` predicate, and still
//! silently loses four bytes. The predicate is `bytes % 32 != 0`.
//!
//! One caveat, raised here and independently by chip-agent, and since RESOLVED:
//! the "encoded length 0" reading came from a field the probe calls `field629`,
//! and read as a u32 that field COLLIDES between `req=4..28` and `req=256` while
//! their md5s differ, so it is not a length. Read as a SINGLE BYTE it is
//! `floor(bytes / 32) * 32` exactly -- 0x00, 0x20, 0x40, 0x60, 0x80 -- which does
//! reproduce. The receipt now leads with the byte-identity groupings and the
//! constant 3576-byte `.aicore_binary` size across extents 1..63 as the
//! load-bearing facts, and labels the length reading as an interpretation. The
//! rule here rests on the groupings either way.
//!
//! That also relocates the finding: **B30 is a toolchain row, not a silicon one.**
//! The instruction is emitted -- same section size, three bytes different -- with a
//! truncated length, so the DMA engine was never asked to move anything. "Present
//! in neither buffer" was a length-0 transfer.
//!
//! # The emitter REFUSES, and a pad repair was tried and withdrawn
//!
//! This section said the opposite for part of 2026-11-09 and the correction is
//! recorded rather than overwritten, because the retracted version is the kind of
//! claim that gets re-landed.
//!
//! The load-side `DataCopyPad` was measured and DOES deliver element 0 at four
//! bytes. On that evidence the refusal became a pad and this file said so. A
//! wider probe then showed cells 1, 2, 4 and 7 all reading element 0's value
//! where a poison prologue had put something else, 12 of 12, and the pad was
//! withdrawn.
//!
//! THAT SECOND READING WAS ALSO TOO STRONG, and this paragraph stated it in the
//! worst way -- "fills the WHOLE 32-byte destination block". Every arm behind it
//! was at FOUR BYTES, where element 0 is the only requested element and "honours
//! the length" and "fills the block" are the same bytes. Separated by reading a
//! cell INSIDE a larger extent: at 8 B cell 1 reads `gm1[1]`, at 16 B cell 3
//! reads `gm1[3]`, both correct, 6 of 6; and at a 64-element destination with a
//! 36 B pad, cell 1 reads `gm1[1]` correctly, 3 of 3. So the pad honours its byte
//! length and fills only the TAIL beyond the requested region.
//!
//! Three readings of one op in a day, each from a probe that could not see what
//! the next one measured. The durable lesson is the probe DESIGN, not the
//! conclusion: vary only the quantity under test, and read a cell inside the
//! extent rather than at its edge.
//!
//! The refusal nevertheless STANDS, for a reason that is not the pad's
//! correctness: the repair has no beneficiary. Zero constant-extent partial-block
//! sites remain in the corpus, the smallest constant extent is exactly 32 bytes,
//! and no kernel is refused on this ground -- the one that ever hit it was fixed
//! at the SOURCE to load a whole block. A refusal that blocks nothing costs
//! nothing, while the pad is licensed only by a precondition about this emitter
//! -- whole-block destinations, and no op reading the rounded count -- that
//! nothing enforced. It is now a test,
//! `ascendc_emit/tests/the_pad_precondition_is_enforced_not_audited.rs`, which is
//! what would make re-landing defensible rather than brave.
//!
//! The lesson for this file is the one its own history keeps teaching: the first
//! pad probe read ONE CELL, and a measurement that cannot see the thing being
//! claimed is not evidence for it. This file therefore remains the artefact-layer
//! backstop against a partial-block load reaching the corpus from any path, and
//! does NOT assume the emitter has a repair.
//!
//! # `DataCopyPad` IS OUT OF SCOPE, and by how much
//!
//! This gate matches `AscendC::DataCopy(` and therefore never sees a
//! `DataCopyPad(` site. Measured over the stored artefacts: 375 plain sites and
//! **105 pad sites**, so a fifth of the copies in the corpus are not judged here.
//! That is correct and not an omission -- the pad carries a BYTE length and
//! honours it, so a partial-block pad is legal where a partial-block plain copy
//! is not -- but it has to be stated, because a reader counting copy sites would
//! otherwise conclude that those 105 were judged and passed.
//!
//! The 2x2 that makes the exemption safe was closed on 2026-11-09, at both the
//! behaviour and instruction layers, and at both dtypes wherever dtype could have
//! mattered:
//!
//! ```text
//!                   LOAD                              STORE
//! DataCopy      rounds length down to 32 B        rounds length down to 32 B
//!               (4 B delivers nothing)            (4 B delivers nothing)
//! DataCopyPad   honours the length, FILLS         honours the length exactly
//!               the tail of the final block       (f32 and int8 identical)
//! ```
//!
//! Instruction-layer corroboration: a plain copy differs in three bytes with the
//! length in the middle, in both directions; a pad differs in one byte carrying
//! the exact count. So "plain carries a block count, pad carries a byte length"
//! is read off the encoding and not only inferred from behaviour.
//!
//! The one cell with a caveat is the pad LOAD's tail fill, which is why the
//! sub-block load stays refused rather than padded -- see the paragraph above on
//! the repair having no beneficiary, and
//! `ascendc_emit/tests/the_pad_precondition_is_enforced_not_audited.rs` for the
//! invariant a re-land would need.
//!
//! # What it judges, and the honest limit
//!
//! CONSTANT extents only, and the figures are worth stating because an earlier
//! version of this note got them wrong. Measured 2026-11-09 over the stored
//! artefacts, split by direction:
//!
//! ```text
//! GM->UB loads    124 constant,  145 runtime
//! UB->GM stores     0 constant,  211 runtime
//! ```
//!
//! AND THIS GATE IS NOW NARROWER THAN THE EMITTER, which is worth stating because
//! a backstop that silently covers less than the thing it backs is how the element
//! version of this file hid. `mlir_to_cpp` resolves a load's extent from the MLIR
//! CONSTANTS rather than from the stringified product, so it decides sites that a
//! C++-text scan like this one reads as `(__v0) * (__v1)` and skips. The 124 below
//! is a text-level count and is a LOWER BOUND on what the emitter judges. That is
//! the right direction for a backstop -- everything this file can see, it checks --
//! but it means a green run here is not evidence that the emitter's own rule was
//! applied everywhere, only that no partial block survived into a constant-extent
//! site.
//!
//! This note previously claimed "124 constant and 248 runtime", which matches
//! neither the runtime loads (145) nor runtime loads and stores together (356).
//! A peer session flagged the discrepancy and the recount is theirs; the wrong
//! figure was mine. A runtime extent cannot be judged from the text --
//! `(__v0) * (__v1)` is whatever the scalars hold -- so those 356 are outside this
//! gate by construction, and the emitter, which knows the extent symbolically,
//! deliberately leaves the runtime load sites alone: a runtime `% 32` branch would
//! rewrite most artefacts and detach every device PASS recorded against those
//! bytes.
//!
//! The reassuring half: the smallest constant extent in the corpus is **32 bytes**,
//! exactly one block, and every constant extent is a whole multiple of it. No
//! banked kernel was ever at risk from this.
//!
//! # It judges BYTES, and an earlier version of it judged ELEMENTS
//!
//! Corrected 2026-11-09. The first version held every constant extent against a
//! const named `BLOCK_F32 = 8` -- eight f32, one block -- and applied it to every
//! `DataCopy` in the corpus regardless of what the destination tile holds. For an
//! f32 tile those are the same rule. For anything narrower they are not: eight
//! `half` is 16 bytes and eight `int8_t` is 8 bytes, both partial blocks, and both
//! would have PASSED a gate written in elements.
//!
//! AND THE BYTE RULE IS NOW MEASURED, not merely the conservative reading. At f32
//! the two candidate rules are indistinguishable, because 8 f32 IS 32 bytes.
//! ORACLE 46 ran it at int8, where they diverge:
//!
//! ```text
//! 64 B -> 64 arrived,  0 poison   PASS 3/3
//! 32 B -> 32 arrived,  0 poison   PASS 3/3
//! 31 B ->  0 arrived, 31 poison   FAIL 3/3
//! 16 B ->  0 arrived, 16 poison   FAIL 3/3
//!  8 B ->  0 arrived,  8 poison   FAIL 3/3
//! ```
//!
//! An 8-ELEMENT floor would have passed the 8 B and 16 B arms, so the element
//! reading is refuted and `BLOCK_BYTES` is the part's rule rather than a safe
//! guess. 31 B delivering 0 of 31 confirms the MULTIPLE rule at a narrow dtype as
//! well, and `poison_bytes` equalling the whole extent in every failing arm is the
//! f32 length-0 signature: the transfer does not happen at all.
//!
//! Of the 124 constant-extent destinations, 122 are `float` and 2 are `half`, and
//! ranked in bytes the smallest three are the 32-byte embedding index loads while
//! both `half` sites are 128 B and 8192 B. The defect was LATENT with zero
//! instances. An unresolvable destination dtype FAILS rather than being skipped,
//! because skipping is how the element version hid: it never had to know a dtype,
//! so it never noticed it was judging two `half` sites by an f32 rule.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// One DMA block, in bytes. The part's, not a tuning knob: 32 bytes is what the
/// unit moves, and bisheng floors the encoded length to a multiple of it, which is
/// why the rule is a MULTIPLE expressed in bytes with the element count derived
/// from the destination tile's dtype rather than assumed.
const BLOCK_BYTES: u64 = 32;

/// Bytes per element of the AscendC spellings that appear as `LocalTensor<T>` in
/// this corpus. An absent entry is a FAILURE, not a skip -- see the module note.
fn elem_bytes(dtype: &str) -> Option<u64> {
    Some(match dtype {
        "int8_t" | "uint8_t" => 1,
        "half" | "bfloat16_t" | "int16_t" | "uint16_t" => 2,
        "float" | "int32_t" | "uint32_t" => 4,
        "int64_t" | "uint64_t" | "double" => 8,
        _ => return None,
    })
}

fn bench() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../bench")
        .canonicalize()
        .expect("the bench must be reachable")
}

/// `name -> dtype` for every `AscendC::LocalTensor<T> name = ...` in one artefact.
/// The emitter declares each tile buffer exactly once, at its `Get<T>()`.
fn declared_tile_types(text: &str) -> HashMap<String, String> {
    let mut out = HashMap::new();
    for line in text.lines() {
        let Some(rest) = line.split_once("LocalTensor<") else {
            continue;
        };
        let Some((dtype, after)) = rest.1.split_once('>') else {
            continue;
        };
        let name = after.trim().split(|c: char| !c.is_alphanumeric() && c != '_').find(|s| !s.is_empty());
        if let Some(name) = name {
            out.insert(name.to_string(), dtype.trim().to_string());
        }
    }
    out
}

/// `(extent_text, dst_name, elements)` for every `DataCopy` whose extent is two
/// integer literals multiplied, which is the shape the emitter writes for a
/// statically known tile: `(1) * (8)`, `(16) * (512)`.
fn constant_extents(text: &str) -> Vec<(String, String, String, u64)> {
    let mut out: Vec<(String, String, String, u64)> = Vec::new();
    for line in text.lines() {
        let Some(rest) = line.split_once("AscendC::DataCopy(") else {
            continue;
        };
        // The destination is the first argument, up to `,` or a `[` subscript.
        let dst: String = rest
            .1
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        let Some(args) = rest.1.rsplit_once(");") else { continue };
        let mut parts = args.0.rsplitn(2, ',');
        let Some(extent) = parts.next() else { continue };
        // The SECOND argument, for a store: its destination is a `GlobalTensor`
        // and carries no dtype a text scan can read, so the tile on the other
        // side is what names the width.
        let src: String = parts
            .next()
            .unwrap_or("")
            .rsplit(',')
            .next()
            .unwrap_or("")
            .trim()
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        let extent = extent.trim();
        // `(A) * (B)` with A and B integer literals, and nothing else.
        let Some((a, b)) = extent.split_once(" * ") else {
            continue;
        };
        let strip = |s: &str| {
            s.trim()
                .trim_start_matches('(')
                .trim_end_matches(')')
                .trim()
                .parse::<u64>()
                .ok()
        };
        if let (Some(a), Some(b)) = (strip(a), strip(b)) {
            out.push((extent.to_string(), dst, src, a * b));
        }
    }
    out
}

#[test]
fn every_constant_datacopy_extent_is_a_whole_number_of_blocks() {
    let mut artefacts = 0usize;
    let mut judged = 0usize;
    let mut smallest_bytes = u64::MAX;
    let mut narrow_sites = 0usize;
    let mut offenders: Vec<String> = Vec::new();
    let mut unresolved: Vec<String> = Vec::new();

    for model in ["deepseek_v41", "glm_5_3", "kimi_k3", "engine", "playground16"] {
        let dir = bench().join(model).join("ascendc");
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        let mut paths: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
        paths.sort();
        for path in paths {
            if path.extension().and_then(|e| e.to_str()) != Some("cpp") {
                continue;
            }
            let text = std::fs::read_to_string(&path).unwrap();
            artefacts += 1;
            let types = declared_tile_types(&text);
            for (extent, dst, src, elems) in constant_extents(&text) {
                // An unknown destination or dtype is an UNJUDGED site. The element
                // version of this gate had no dtype at all and so could not tell
                // the difference; collect them and fail below.
                // A LOAD names its destination tile; a STORE names a
                // `GlobalTensor` there and its tile is the source. Plain
                // `DataCopy` rounds its length DOWN to a multiple of 32 in BOTH
                // directions -- measured 2026-11-09, a 4-byte plain store
                // delivers nothing, exactly as a 4-byte plain load does -- so the
                // same rule applies and the only difference is which operand
                // carries the width.
                let resolved = types.get(&dst).or_else(|| types.get(&src));
                let Some(dtype) = resolved else {
                    unresolved.push(format!(
                        "{}: DataCopy into `{dst}` ({extent}), which has no                          `LocalTensor<T>` declaration in the file",
                        path.display()
                    ));
                    continue;
                };
                let Some(w) = elem_bytes(dtype) else {
                    unresolved.push(format!(
                        "{}: DataCopy into `{dst}`: unknown dtype `{dtype}`. Add its                          width to `elem_bytes` -- an unmeasured dtype must not pass                          by default",
                        path.display()
                    ));
                    continue;
                };
                judged += 1;
                if w < 4 {
                    narrow_sites += 1;
                }
                let bytes = elems * w;
                smallest_bytes = smallest_bytes.min(bytes);
                // A MULTIPLE, not a minimum: `floor(bytes / 32) * 32` is what
                // the encoded length becomes, so 36 bytes delivers 32 and
                // loses four. `bytes < BLOCK_BYTES` passed a 36-byte load.
                if bytes % BLOCK_BYTES != 0 {
                    offenders.push(format!(
                        "{}: DataCopy into {dst} of {extent} = {elems} x {dtype} = {bytes} \
                         bytes, which is {} byte(s) past a whole \
                         {BLOCK_BYTES}-byte block",
                        path.display(),
                        bytes % BLOCK_BYTES
                    ));
                }
            }
        }
    }

    assert!(
        unresolved.is_empty(),
        "{} constant-extent DataCopy site(s) could not be judged, which is not the          same as passing:\n  {}",
        unresolved.len(),
        unresolved.join("\n  ")
    );
    assert!(
        offenders.is_empty(),
        "{} partial-block DataCopy load(s). bisheng floors the encoded DMA \
         length to a multiple of 32 bytes, so a partial block silently delivers \
         less than it asked for and a four-byte load delivers NOTHING -- measured \
         on an Ascend910B2, where the index load left a poison 2.0 intact. \
         `mlir_to_cpp` REFUSES such a load; a `DataCopyPad` branch was tried and \
         withdrawn because the pad fills the whole destination block with element \
         zero. So an artefact reaching here was written by hand or came through a \
         path that bypassed that refusal -- widen the load to a whole block at \
         the SOURCE, or refuse the kernel.\n  {}",
        offenders.len(),
        offenders.join("\n  ")
    );
    // Floors, so this cannot pass by finding nothing. Measured 2026-11-09: 101
    // artefacts, 124 constant-extent sites, 122 `float` and 2 `half`.
    assert!(
        artefacts >= 95,
        "expected at least 95 stored artefacts, saw {artefacts} -- a gate over an \
         empty list satisfies every assertion in it"
    );
    assert!(
        judged >= 110,
        "expected at least 110 constant-extent DataCopy sites, saw {judged}. If the \
         emitter moved to runtime extents everywhere, this gate stopped covering \
         anything and the floor has to move into mlir_to_cpp."
    );
    // The control for the correction itself. Without a single non-f32 destination,
    // the dtype-aware path above never runs on anything the element version would
    // have got wrong, and this file would be claiming a coverage it had not
    // exercised. Both of these are in `matmul.cpp`, at 128 B and 8192 B.
    assert!(
        narrow_sites >= 2,
        "no constant-extent DataCopy destination narrower than f32 is left in the \
         corpus ({narrow_sites} found). The byte rule above is then untested against \
         the case it was written for -- find a narrow-dtype site or say here that \
         the corpus no longer has one."
    );
    assert_eq!(
        smallest_bytes, BLOCK_BYTES,
        "the smallest constant DataCopy load in the corpus is {smallest_bytes} \
         bytes, not {BLOCK_BYTES}. Below a block is the defect above; ABOVE it means \
         all three embedding index loads went away, and this gate's own subject with \
         them -- check why before raising the number."
    );
}
