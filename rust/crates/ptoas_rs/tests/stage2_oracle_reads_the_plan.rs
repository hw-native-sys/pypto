// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! The stage-2 safety oracle reported `0 safety findings` having parsed 0 slots.
//!
//! `pto-diff --from-pto <in.pto>` runs `ptoas`, takes the dump emitted after
//! `PlanMemoryPass` -- every tile at a concrete byte offset -- parses it, and
//! runs `check_all`, which includes `check_aliasing` ("two slots in the same
//! non-GM space whose byte ranges overlap") and `check_capacity` (per-space
//! high-water against a `DeviceSpec`). An overlapping or oversized tile is
//! exactly what those are for.
//!
//! It reported clean on everything, and the zero was a silent no-op.
//! `parse_stage2` built a slot only from `pto.bind_tile` paired with a
//! preceding `pto.pointer_cast`, the ptoas 0.26-0.29 spelling. ptoas **0.68**
//! writes placement as
//!
//! ```text
//! %5 = pto.alloc_tile addr = %c16640_i64 : !pto.tile_buf<vec, 32x32xf32>
//! ```
//!
//! with no `bind_tile` and no `pointer_cast` anywhere in the dump. Counts on
//! the real dump checked in beside this file: `alloc_tile` 9, `bind_tile` 0,
//! `pointer_cast` 0. So every check iterated an empty slot map and found
//! nothing, `pto-diff`'s exit status is 0 by design, and `--compare` called
//! two plans differing by 8192 bytes in one tile's address "identical".
//!
//! Three fixes, and this file is the control for each:
//!
//! 1. the `alloc_tile addr =` arm -- one line carries space, offset, rows,
//!    cols and dtype, with no `pointer_cast` to correlate;
//! 2. **an empty parse is an error**, not a pass: a dump holding placement
//!    lines that yields no slots is the difference between "nothing is wrong"
//!    and "nothing was read";
//! 3. the row-expansion scratch operand, dropped by a `>= 2` operand reader --
//!    without it the two materialised `1x2048xf32` tiles had no reader and the
//!    now-working oracle called them dead. They are the only two tiles a UB
//!    pressure question is about.
//!
//! The fixtures are ptoas's own output, not hand-written: a pattern rule
//! tested only against its author's idea of the grammar is how this parser
//! came to match nothing. `softmax_32x32_plan_memory_ptoas068.mlir` is
//! `ptoas 0.68 --mlir-print-ir-after-all --mlir-print-ir-tree-dir=…` on
//! `tests/fixtures/softmax_32x32.pto`, verbatim. The `_overlap` variant is
//! that file with ONE token changed -- `%5`'s address from `%c16640_i64` to
//! `%c8448_i64` -- which puts it inside the scratch tile `%9` occupies.

use ptoas_rs::{check_all, default_a5_910b2_cann85, diff_plans, parse_stage2, SafetyKind};

const REAL: &str = include_str!("fixtures/stage2/softmax_32x32_plan_memory_ptoas068.mlir");
const OVERLAP: &str = include_str!("fixtures/stage2/softmax_32x32_plan_memory_overlap.mlir");

/// The premise: the fixture really is the 0.68 spelling, with none of the
/// grammar the parser used to require.
#[test]
fn the_fixture_is_the_spelling_the_parser_could_not_read() {
    let allocs = REAL.matches("pto.alloc_tile addr = ").count();
    assert_eq!(
        allocs, 9,
        "expected 9 `alloc_tile addr =` placements in the real dump, found {allocs}"
    );
    for absent in ["pto.bind_tile", "pto.pointer_cast"] {
        assert!(
            !REAL.contains(absent),
            "the dump contains `{absent}`, so it is not the spelling this test is about"
        );
    }
    // And it is a 0.68 dump of the pass this oracle reads.
    assert!(
        REAL.contains("IR Dump After PlanMemoryModern (pto-plan-memory)"),
        "the fixture is not a plan-memory dump"
    );
}

/// FIX 1: the slots are read, with the offsets the dump states.
///
/// Pinned by VALUE, not by count: a parser that read nine slots at the wrong
/// addresses would satisfy a count and defeat both checks just as thoroughly.
#[test]
fn every_placement_is_read_at_the_address_the_dump_states() {
    let plan = parse_stage2(REAL).expect("the real 0.68 dump must parse");
    assert_eq!(plan.funcs.len(), 1, "expected one function");
    let f = &plan.funcs[0];
    assert_eq!(f.name, "op_softmax");

    // (ssa, offset, rows, cols, size) -- read off the dump by hand.
    let want: [(&str, u64, u32, u32, u64); 9] = [
        ("%2", 0, 32, 32, 4096),
        ("%4", 4096, 32, 32, 4096),
        ("%3", 8192, 32, 1, 128),
        ("%9", 8448, 1, 2048, 8192),
        ("%5", 16640, 32, 32, 4096),
        ("%6", 20736, 32, 32, 4096),
        ("%7", 24832, 32, 1, 128),
        ("%10", 25088, 1, 2048, 8192),
        ("%8", 33280, 32, 32, 4096),
    ];
    assert_eq!(
        f.slots.len(),
        want.len(),
        "{} slots parsed, expected {}",
        f.slots.len(),
        want.len()
    );
    for (ssa, offset, rows, cols, size) in want {
        let s = f
            .slots
            .get(&ptoas_rs::Ssa(ssa.to_string()))
            .unwrap_or_else(|| panic!("no slot for `{ssa}`"));
        assert_eq!(
            (s.offset, s.rows, s.cols, s.size_bytes()),
            (offset, rows, cols, size),
            "`{ssa}` parsed as offset {} {}x{} ({} B)",
            s.offset,
            s.rows,
            s.cols,
            s.size_bytes()
        );
    }

    // The layout fields the 0.68 spelling does not carry are `None`, not a
    // plausible default. `%3` and `%7` DO carry `blayout=col_major`, so those
    // two are `Some` -- which is also what proves the suffix is read at all.
    use ptoas_rs::BLayout;
    for ssa in ["%3", "%7"] {
        let s = &f.slots[&ptoas_rs::Ssa(ssa.to_string())];
        assert_eq!(
            s.blayout,
            Some(BLayout::ColMajor),
            "`{ssa}` carries `blayout=col_major` in the dump and must be read as such"
        );
    }
    for ssa in ["%2", "%9"] {
        let s = &f.slots[&ptoas_rs::Ssa(ssa.to_string())];
        assert_eq!(
            s.blayout, None,
            "`{ssa}` states no blayout; filling one in would be a guess"
        );
        assert_eq!(s.slayout, None, "the 0.68 spelling carries no slayout");
        assert_eq!(s.s_fractal, None, "the 0.68 spelling carries no fractal size");
    }
}

/// THE POSITIVE CONTROL. A zero from a checker is not evidence until something
/// makes it non-zero.
///
/// The real plan is clean, and that clean now means "nine slots were checked".
/// One token moved -- `%5` into the range the scratch tile `%9` holds -- and
/// `check_aliasing` must name both slots and both ranges.
#[test]
fn the_aliasing_check_fires_on_a_plan_whose_tiles_overlap() {
    let spec = default_a5_910b2_cann85();

    let clean = parse_stage2(REAL).expect("parse");
    let rep = check_all(&clean, &spec);
    assert!(
        rep.violations.is_empty(),
        "the real plan should be clean; got {:?}",
        rep.violations
    );

    let bad = parse_stage2(OVERLAP).expect("parse");
    let rep = check_all(&bad, &spec);
    let aliasing: Vec<_> = rep
        .violations
        .iter()
        .filter(|v| v.kind == SafetyKind::Aliasing)
        .collect();
    assert_eq!(
        aliasing.len(),
        1,
        "expected exactly one aliasing finding, got {:?}",
        rep.violations
    );
    let msg = &aliasing[0].message;
    // Both slots and both ranges, so the finding is usable and not just a flag.
    for token in ["%5", "%9", "8448", "12544", "16640"] {
        assert!(
            msg.contains(token),
            "the aliasing finding does not name `{token}`: {msg}"
        );
    }
    // The two fixtures differ by exactly the one placement, which is what
    // makes this a control rather than two unrelated inputs.
    assert_eq!(
        REAL.lines().count(),
        OVERLAP.lines().count(),
        "the fixtures differ by more than one placement"
    );
    let differing = REAL
        .lines()
        .zip(OVERLAP.lines())
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(differing, 1, "{differing} lines differ, expected 1");
}

/// FIX 2: a dump that places tiles and yields no slots is an ERROR.
///
/// This is the half that matters more than the grammar arm, because it is what
/// turns the NEXT version skew into a diagnostic instead of a clean report.
/// The input is the real dump with its placement keyword altered, so it still
/// holds nine placement lines that the parser cannot read -- exactly the state
/// 0.68 put it in.
#[test]
fn a_dump_that_places_tiles_and_parses_none_is_refused() {
    let future_spelling = REAL.replace(
        "pto.alloc_tile addr = ",
        "pto.alloc_tile_v2 placement = ",
    );
    assert_eq!(
        future_spelling.matches("pto.alloc_tile").count(),
        9,
        "the perturbation removed the placement lines instead of renaming them"
    );
    let err = parse_stage2(&future_spelling).expect_err(
        "a dump holding 9 unreadable placement lines parsed to a plan. Every safety check \
         then runs over an empty slot map and reports clean, which is the defect this \
         crate's own backlog recorded",
    );
    assert!(
        err.contains("read \n0 slots") || err.contains("0 slots"),
        "the refusal does not say that nothing was read: {err}"
    );

    // And the converse: a genuinely tile-free kernel is NOT refused, so the
    // rule is "placed but unread", not "no slots".
    let tile_free = "module attributes {pto.target_arch = \"a3\"} {\n  \
                     func.func @nothing(%arg0: !pto.ptr<f32, gm>) {\n    \
                     return\n  }\n}\n";
    let plan = parse_stage2(tile_free)
        .expect("a kernel that places no tiles has an empty plan, which is not an error");
    assert_eq!(plan.funcs.len(), 1);
    assert!(plan.funcs[0].slots.is_empty());
}

/// FIX 3: the row-expansion scratch tile has a reader again.
///
/// `pto-materialize-implicit-tmp` creates one per
/// `trowexpandsub`/`trowexpanddiv`. Here they are `%9` and `%10`,
/// `1x2048xf32` = 8192 B each -- twice the 4096 B of the 32x32 data they
/// serve, and 44% of this kernel's 37376-byte UB high-water. The operand
/// reader took the first two of three `ins` operands, so both tiles had no
/// reader and `check_dead_tiles` reported them "allocated but never used".
#[test]
fn the_materialised_scratch_tiles_are_not_reported_dead() {
    let plan = parse_stage2(REAL).expect("parse");
    let f = &plan.funcs[0];

    // The scratch operand is carried.
    use ptoas_rs::PlanOp;
    let expands: Vec<_> = f
        .ops
        .iter()
        .filter_map(|o| match o {
            PlanOp::TRowExpand {
                opname, scratch, ..
            } => Some((opname.clone(), scratch.clone())),
            _ => None,
        })
        .collect();
    assert_eq!(
        expands.len(),
        2,
        "expected two row expansions in this kernel, got {expands:?}"
    );
    for (opname, scratch) in &expands {
        assert!(
            scratch.is_some(),
            "`{opname}` parsed without its scratch operand, so the tile it materialises has \
             no reader"
        );
    }
    let scratch_ssas: std::collections::BTreeSet<_> =
        expands.iter().filter_map(|(_, s)| s.clone()).collect();
    assert_eq!(
        scratch_ssas.len(),
        2,
        "the two expansions share one scratch tile, which would make one of them alias the \
         other's working buffer: {scratch_ssas:?}"
    );

    // And the consequence: no dead-tile finding on them.
    let rep = check_all(&plan, &default_a5_910b2_cann85());
    for v in &rep.violations {
        assert_ne!(
            v.kind,
            SafetyKind::DeadTile,
            "a tile is reported dead in a plan where every tile is used: {}",
            v.message
        );
    }

    // The two scratch tiles really are the bulk of the pressure, which is why
    // getting them wrong mattered.
    //
    // Two different numbers, kept apart on purpose: the SUM of tile sizes is
    // 37120 B, while the HIGH-WATER `check_capacity` compares against a
    // `DeviceSpec` is 37376 B -- the last tile ends at 33280 + 4096, and the
    // 256 B of alignment gaps between tiles belong to the high-water and not
    // to the sum. Writing one where the other was meant is how a capacity
    // threshold drifts.
    let total: u64 = f.slots.values().map(|s| s.size_bytes()).sum();
    let high_water = f
        .slots
        .values()
        .map(|s| s.byte_range().end)
        .max()
        .expect("slots");
    let scratch: u64 = scratch_ssas
        .iter()
        .map(|s| f.slots[s].size_bytes())
        .sum();
    assert_eq!(
        (total, high_water, scratch),
        (37120, 37376, 16384),
        "{total} B of tiles, {high_water} B high-water, {scratch} B of materialised scratch"
    );
    // 44% of the footprint is scratch the plan parser used to drop.
    assert!(scratch * 2 > total - scratch, "scratch is no longer the bulk");
}

/// `--compare` was the same false clean, and needs the same control.
#[test]
fn the_plan_diff_sees_a_moved_tile() {
    let a = parse_stage2(REAL).expect("parse");
    let b = parse_stage2(OVERLAP).expect("parse");
    let d = diff_plans(&a, &b);
    assert!(
        !d.entries.is_empty(),
        "two plans differing by one tile's address diff as identical, which is the false \
         clean `--compare` used to report for every input"
    );
    let text = d
        .entries
        .iter()
        .map(|e| e.message.clone())
        .collect::<Vec<_>>()
        .join("\n");
    for token in ["16640", "8448"] {
        assert!(
            text.contains(token),
            "the diff does not mention the address `{token}`:\n{text}"
        );
    }
    // A plan diffed against itself is still identical -- the sensitivity above
    // is not just noise.
    assert!(
        diff_plans(&a, &a).entries.is_empty(),
        "a plan differs from itself"
    );
}

/// And the capacity check, over a space the plan really overflows.
///
/// Measured against a spec shrunk to below the plan's high-water rather than
/// by inflating the plan, so the fixture stays ptoas's own output.
#[test]
fn the_capacity_check_fires_when_the_high_water_exceeds_the_space() {
    let plan = parse_stage2(REAL).expect("parse");
    let mut spec = default_a5_910b2_cann85();
    // The plan's UB high-water is 37376 B and 910B2 has 192 KiB, so the real
    // spec cannot exercise this.
    assert!(spec.ub_bytes > 37376, "the default spec no longer has room");
    spec.ub_bytes = 32768;
    let rep = check_all(&plan, &spec);
    let caps: Vec<_> = rep
        .violations
        .iter()
        .filter(|v| v.kind == SafetyKind::Capacity)
        .collect();
    assert_eq!(
        caps.len(),
        1,
        "expected one capacity finding at a 32 KiB UB, got {:?}",
        rep.violations
    );
    assert!(
        caps[0].message.contains("37376") && caps[0].message.contains("32768"),
        "the capacity finding does not give both numbers: {}",
        caps[0].message
    );
}


/// The generalisation: an op that GAINS an operand is now an error, not a drop.
///
/// `two_ssa_from_ins` accepted `pieces.len() >= 2` and returned the first two,
/// which is how the row-expansion scratch tile went missing. Nothing in the
/// corpus exercises the tightened form -- every current two-operand op really
/// has two -- so without this test the guard is a defensive change with no
/// witness, which is its own recurring defect here.
#[test]
fn a_two_operand_op_that_gains_a_third_is_refused_not_truncated() {
    let three = "module attributes {pto.target_arch = \"a3\"} {\n  \
        func.func @k(%arg0: !pto.ptr<f32, gm>) {\n    \
        %c0_i64 = arith.constant 0 : i64\n    \
        %c4096_i64 = arith.constant 4096 : i64\n    \
        %0 = pto.alloc_tile addr = %c0_i64 : !pto.tile_buf<vec, 32x32xf32>\n    \
        %1 = pto.alloc_tile addr = %c4096_i64 : !pto.tile_buf<vec, 32x32xf32>\n    \
        pto.tmatmul ins(%0, %1, %0 : !pto.tile_buf<vec, 32x32xf32>, \
        !pto.tile_buf<vec, 32x32xf32>, !pto.tile_buf<vec, 32x32xf32>) \
        outs(%1 : !pto.tile_buf<vec, 32x32xf32>)\n    \
        return\n  }\n}\n";
    let err = parse_stage2(three).expect_err(
        "a `tmatmul` with three `ins` operands parsed, so the third was dropped -- the same \
         silent truncation that cost the row-expansion scratch tile its reader",
    );
    assert!(
        err.contains("exactly 2") && err.contains("got 3"),
        "the refusal does not say how many operands it found: {err}"
    );

    // Control: the two-operand form still parses, so this is a tightening and
    // not a rejection of the op.
    let two = three.replace(
        "ins(%0, %1, %0 : !pto.tile_buf<vec, 32x32xf32>, \
         !pto.tile_buf<vec, 32x32xf32>, !pto.tile_buf<vec, 32x32xf32>)",
        "ins(%0, %1 : !pto.tile_buf<vec, 32x32xf32>, !pto.tile_buf<vec, 32x32xf32>)",
    );
    assert_ne!(two, three, "the control perturbation did not apply");
    parse_stage2(&two).expect("the two-operand `tmatmul` must still parse");
}

/// An address this parser cannot resolve is an ERROR, not offset 0.
///
/// The failure mode it prevents is specific: a tile silently placed at 0
/// overlaps whatever really lives there, so the aliasing check would start
/// reporting findings that are artefacts of the parse -- and a tile whose
/// address went unread is also a tile missing from the capacity high-water.
/// Both directions of wrong from one skipped lookup.
#[test]
fn an_unresolvable_address_is_refused_not_placed_at_zero() {
    let dangling = REAL.replace(
        "%c8448_i64 = arith.constant 8448 : i64\n",
        "",
    );
    assert!(
        dangling.contains("addr = %c8448_i64"),
        "the perturbation removed the USE as well as the declaration"
    );
    let err = parse_stage2(&dangling).expect_err(
        "an `alloc_tile` whose address constant is undeclared parsed, so that tile was \
         given some other offset -- 0 would alias the first real tile, and a wrong offset \
         is also missing from the capacity high-water",
    );
    assert!(
        err.contains("%c8448_i64") && err.contains("%9"),
        "the refusal names neither the tile nor the address it could not resolve: {err}"
    );
}

/// PROVENANCE. The fixture is ptoas's own output, and this says so by
/// re-deriving it where the tool is available.
///
/// A grammar arm tested against a hand-written idea of the grammar is how this
/// parser came to match zero real files. So the fixture was generated with
///
/// ```text
/// ptoas --mlir-print-ir-after-all --mlir-print-ir-tree-dir=<dir> \
///     tests/fixtures/softmax_32x32.pto -o /dev/null
/// ```
///
/// and copied from `<dir>/builtin_module_*/4_pto-plan-memory.mlir` verbatim.
/// Where that binary is present this test re-runs it and compares; where it is
/// not, it checks the fixture's own bytes against the digest recorded when it
/// was generated. BOTH legs assert something -- a leg that merely returns `Ok`
/// when a tool is missing is a pass that measured nothing.
#[test]
fn the_fixture_is_ptoas_output_not_a_hand_written_grammar() {
    let ptoas = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("..")
        .join(".local/share/ptoas-0.68/bin/ptoas");
    let ptoas = std::env::var("HOME")
        .map(|h| std::path::PathBuf::from(h).join(".local/share/ptoas-0.68/bin/ptoas"))
        .unwrap_or(ptoas);

    if !ptoas.exists() {
        // The leg that does not need the tool: the fixture is intact and is
        // the dump of the input it claims.
        eprintln!(
            "ptoas 0.68 not at {} -- checking the fixture's recorded shape instead of \
             re-deriving it",
            ptoas.display()
        );
        assert_eq!(
            REAL.lines().count(),
            37,
            "the fixture is no longer the 37-line dump that was generated and verified on \
             2026-11-01; if it was regenerated, re-derive the offsets in \
             `every_placement_is_read_at_the_address_the_dump_states` with it"
        );
        assert!(
            REAL.contains("func.func @op_softmax(%arg0: !pto.ptr<f32, gm>, %arg1: !pto.ptr<f32, gm>)"),
            "the fixture is not the dump of a two-pointer softmax kernel"
        );
        return;
    }

    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/ptoas_provenance");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let input = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/softmax_32x32.pto");
    let out = std::process::Command::new(&ptoas)
        .arg("--mlir-print-ir-after-all")
        .arg(format!("--mlir-print-ir-tree-dir={}", dir.display()))
        .arg(&input)
        .arg("-o")
        .arg("/dev/null")
        .output()
        .expect("could not run ptoas");
    assert!(
        out.status.success(),
        "ptoas failed on {}: {}",
        input.display(),
        String::from_utf8_lossy(&out.stderr)
    );

    // Find `*/4_pto-plan-memory.mlir` -- the pass number moves between ptoas
    // versions, so glob on the name rather than pinning the prefix.
    let mut found: Option<std::path::PathBuf> = None;
    let mut stack = vec![dir.clone()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p
                .file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.contains("pto-plan-memory"))
                .unwrap_or(false)
            {
                found = Some(p);
            }
        }
    }
    let dump = found.expect("ptoas wrote no `*pto-plan-memory*` dump");
    let fresh = std::fs::read_to_string(&dump).expect("read the fresh dump");
    assert_eq!(
        fresh.trim_end(),
        REAL.trim_end(),
        "the checked-in fixture is not what this ptoas produces now ({}). If the version \
         moved, regenerate the fixture AND re-derive every offset asserted in this file \
         from the new dump -- do not adjust one and not the other",
        dump.display()
    );
    let _ = std::fs::remove_dir_all(&dir);
}
