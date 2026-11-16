// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! The precondition a `DataCopyPad` load would rest on, checked rather than audited.
//!
//! A sub-block GM load is REFUSED by `mlir_to_cpp`, not padded, and that is the
//! current decision. The pad was measured, landed, withdrawn and then licensed
//! again on 2026-11-09, and this file exists because of the last step: the licence
//! is a PRECONDITION about this emitter, it was established by a one-time static
//! audit, and nothing enforced it.
//!
//! `DataCopyPad` honours its byte length on a load and additionally fills the TAIL
//! of the final partial block. That is safe only while two things hold:
//!
//!   1. every tile-load destination is a whole number of 32-byte blocks, so the
//!      filled cells are the tile's own padding and not another value; and
//!   2. no consumer reads the ROUNDED element count -- ops are emitted with the
//!      logical extent `(rows) * (cols)`, so the padding is allocated and never
//!      computed over.
//!
//! Both were verified by hand and both are properties of the emitter TODAY. An arm
//! added next month that passes a rounded count to an op would break (2) silently,
//! and the pad's failure mode is a wrong answer rather than a refusal. So the
//! audit is here as a test.
//!
//! # Why the refusal stays anyway
//!
//! Because the repair has NO BENEFICIARY. Measured the same day over the 105
//! stored artefacts: zero constant-extent partial-block sites, the smallest
//! constant extent is exactly 32 bytes, and zero kernels are refused on this
//! ground -- the one kernel that ever hit it, a decode embedding reading its token
//! id through a 1x1 tile, was fixed at the SOURCE to load a whole block. A
//! refusal that blocks nothing costs nothing, while a pad whose precondition
//! lapses is silent. That asymmetry, not the pad's correctness, is the reason.
//!
//! This file is what would make re-landing defensible rather than brave: with the
//! invariant enforced, the pad rests on something a test fails when it breaks.
//! The form is in 5e13ccf40's diff and the predicate should stay `% 32 != 0`.
//! Limits any re-land must state: f32 only, one SKU, one CANN, and the load at a
//! narrow dtype unmeasured.

use std::path::Path;

fn source() -> String {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/mlir_to_cpp.rs");
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

/// Every spelling of the block round-up this file has seen. `+ 7) / 8 * 8` and
/// `+ 7) / 8) * 8` are the SAME arithmetic with the division parenthesised
/// differently, and the second was found by this gate firing on it rather than by
/// reading -- which is the behaviour wanted: a new spelling is a new shape until
/// someone confirms it is whole-block and adds it here.
const ROUNDUP: [&str; 3] = ["+ 7) / 8 * 8", "+7)/8*8", "+ 7) / 8) * 8"];

fn has_roundup(s: &str) -> bool {
    ROUNDUP.iter().any(|r| s.contains(r))
}

/// The ops a filled tail could reach if an allocation count were handed to one.
const OPS: [&str; 15] = [
    "DataCopy", "DataCopyPad", "Muls", "Adds", "Add", "Mul", "Sub", "Div", "Exp",
    "Cast", "Duplicate", "ReduceSum", "ReduceMax", "Max", "Min",
];

/// (2): the rounded count is an ALLOCATION size and never an op's extent.
#[test]
fn no_op_call_is_emitted_with_a_block_rounded_element_count() {
    let t = source();
    // The round-up idiom, in the spellings the file uses.
    let occurrences: usize = ROUNDUP.iter().map(|r| t.matches(r).count()).sum();
    // A floor, so this cannot pass because the idiom disappeared and the check
    // now covers nothing. 76 measured 2026-11-09.
    assert!(
        occurrences >= 60,
        "only {occurrences} block-rounded expressions left (76 measured). If the \
         emitter stopped rounding allocations up, the pad's precondition changed \
         shape and this file's reasoning needs redoing -- it does NOT simply pass."
    );

    let mut offenders = Vec::new();
    for op in OPS {
        let needle = format!("AscendC::{op}(");
        for (i, _) in t.match_indices(&needle) {
            // The call's own argument list, bounded so a later allocation on a
            // following line cannot be attributed to this call.
            let tail = &t[i..(i + 240).min(t.len())];
            let args = tail.split(");").next().unwrap_or(tail);
            if has_roundup(args) {
                offenders.push(format!("{op}: {}", args.replace('\n', " ")));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "{} op call(s) take a BLOCK-ROUNDED element count. That is the precondition \
         a `DataCopyPad` load rests on: the pad fills the tail of the final partial \
         block, which is safe only while the padding is allocated and never \
         computed over. One op reading the rounded count turns that tail into a \
         value the kernel uses.\n  {}",
        offenders.len(),
        offenders.join("\n  ")
    );
}

/// (1): every tile-buffer allocation is a whole number of blocks, by shape.
#[test]
fn every_buffer_allocation_has_a_known_whole_block_shape() {
    let t = source();
    let mut sites = 0usize;
    let mut unknown = Vec::new();
    for (i, _) in t.match_indices("buf_sizes.insert(") {
        sites += 1;
        let tail = &t[i..(i + 220).min(t.len())];
        let call = tail.split(");").next().unwrap_or(tail);
        let flat: String = call.split_whitespace().collect::<Vec<_>>().join(" ");
        let known = has_roundup(&flat)
            || flat.contains("nb(")
            || flat.contains("align")
            || flat.contains("\"8\".to_string()")
            || flat.contains("work_elems")
            || flat.contains("work_size")
            || flat.contains("256");
        if !known {
            unknown.push(flat.chars().take(110).collect::<String>());
        }
    }
    assert!(
        sites >= 100,
        "only {sites} allocation sites found (106 measured 2026-11-09); the \
         census this file pins has moved and the classification below is stale"
    );
    assert!(
        unknown.is_empty(),
        "{} buffer allocation(s) use a shape this file does not recognise as a \
         whole number of 32-byte blocks. The pad's precondition is that a filled \
         tail lands in the tile's OWN padding; an allocation sized to the logical \
         extent has no padding, so the fill would land in the next buffer. Either \
         the shape is whole-block and belongs in the list here, or it is not and \
         the pad must stay refused.\n  {}",
        unknown.len(),
        unknown.join("\n  ")
    );
}
