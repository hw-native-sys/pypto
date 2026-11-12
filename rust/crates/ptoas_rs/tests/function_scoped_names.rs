// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! SSA names are scoped to a function, and emitted PTO reuses them.
//!
//! `PtoModule` kept one `shapes` map and one `consts` map for the whole module
//! until 2026-11-01, keyed on the bare name, so the last function's binding
//! won. Corpus census that day: 10 of 49 artifacts hold more than one
//! `func.func`; **3 of them bind one SSA name to two different tile shapes (36
//! names in all)** and one binds two different constants.
//!
//! `examples/bench_softmax_tile/artifacts/tile_softmax_bench_kernels.acl.pto`
//! is the live case -- six kernels, each declaring `%pto2` at a different shape
//! (1x1024, 1x4096, 1x8192, 4x256, 16x256, 16x512). The reverse emitter wrote
//! `16x512` into all six, corrupting five. It is one of the eight artifacts the
//! corpus sweep reports as round-tripping, and the sweep could not see it: the
//! collided map is built the same way on both passes, so the ASTs agreed.

use ptoas_rs::{ast::Ssa, parser::parse_module, reemit_pto};

/// Two functions, one `%t` each, different shapes.
const TWO_FUNCS: &str = r#"
module {
  func.func @narrow(%a: !pto.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %v = pto.make_tensor_view %a, shape = [%c1, %c32], strides = [%c32, %c1] : !pto.tensor_view<?x?xf32>
    %p = pto.partition_view %v, offsets = [%c0, %c0], sizes = [%c1, %c32] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<1x32xf32>
    %t = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32>
    pto.tload ins(%p : !pto.partition_tensor_view<1x32xf32>) outs(%t : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32>)
    return
  }
  func.func @tall(%a: !pto.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %v = pto.make_tensor_view %a, shape = [%c4, %c8], strides = [%c8, %c1] : !pto.tensor_view<?x?xf32>
    %p = pto.partition_view %v, offsets = [%c0, %c0], sizes = [%c4, %c8] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<4x8xf32>
    %t = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=4, cols=8>
    pto.tload ins(%p : !pto.partition_tensor_view<4x8xf32>) outs(%t : !pto.tile_buf<loc=vec, dtype=f32, rows=4, cols=8>)
    return
  }
}
"#;

#[test]
fn the_same_ssa_name_at_two_shapes_stays_two_shapes() {
    let m = parse_module(TWO_FUNCS).expect("parse");
    assert_eq!(m.funcs.len(), 2);
    let t = Ssa("%t".to_string());
    let a = m.funcs[0].shapes.get(&t).copied().expect("%t in @narrow");
    let b = m.funcs[1].shapes.get(&t).copied().expect("%t in @tall");
    assert_eq!((a.rows, a.cols), (1, 32), "@narrow's %t");
    assert_eq!((b.rows, b.cols), (4, 8), "@tall's %t");

    // Checked PER FUNCTION BLOCK, not over the whole file. `alloc_tile`
    // re-emits from the op's own `ty` field rather than the shape map, so a
    // whole-file `contains` is satisfied by the alloc line alone and passes
    // even when every `tload`/`tstore` type is wrong. A mutant reverting the
    // union to last-wins slipped through exactly that way.
    let out = reemit_pto(TWO_FUNCS).expect("re-emit");
    let blocks = func_blocks(&out);
    assert_eq!(blocks.len(), 2, "expected two function blocks\n{out}");
    assert!(
        blocks[0].contains("rows=1, cols=32") && !blocks[0].contains("rows=4, cols=8"),
        "@narrow carries @tall's shape\n{}",
        blocks[0]
    );
    assert!(
        blocks[1].contains("rows=4, cols=8") && !blocks[1].contains("rows=1, cols=32"),
        "@tall carries @narrow's shape\n{}",
        blocks[1]
    );
}

/// The re-emitted text split into one string per `func.func`.
fn func_blocks(out: &str) -> Vec<String> {
    let mut blocks: Vec<String> = Vec::new();
    for line in out.lines() {
        if line.trim_start().starts_with("func.func ") {
            blocks.push(String::new());
        }
        if let Some(b) = blocks.last_mut() {
            b.push_str(line);
            b.push('\n');
        }
    }
    blocks
}

/// The module-wide union withholds a name it cannot answer for, so a consumer
/// reaching for it MISSES rather than receiving the last function's binding.
#[test]
fn the_union_withholds_an_ambiguous_name() {
    let m = parse_module(TWO_FUNCS).expect("parse");
    let t = Ssa("%t".to_string());
    assert!(
        m.ambiguous.contains(&t),
        "`%t` binds two shapes and must be recorded as ambiguous"
    );
    assert!(
        !m.shapes.contains_key(&t),
        "the union answered for an ambiguous name; a consumer would get one \
         function's shape while emitting another's"
    );
    // An unambiguous name is still in the union.
    assert!(
        m.shapes.contains_key(&Ssa("%p".to_string())) || m.ambiguous.contains(&Ssa("%p".to_string())),
        "%p should be either answered or withheld, not absent for another reason"
    );
}

/// The real artifact: six kernels, six shapes for `%pto2`, all six preserved.
#[test]
fn the_six_kernel_bench_artifact_keeps_all_six_shapes() {
    // The fixture lives beside this test: the ported version read an artifact
    // from a tree this workspace does not carry.
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/bench/six_kernels.pto");
    let src = std::fs::read_to_string(&path).expect("the bench artifact");
    let m = parse_module(&src).expect("parse");
    assert_eq!(m.funcs.len(), 6, "the artifact should hold six kernels");

    let pto2 = Ssa("%pto2".to_string());
    let mut seen: Vec<(u32, u32)> = m
        .funcs
        .iter()
        .map(|f| {
            let sh = f.shapes.get(&pto2).unwrap_or_else(|| {
                panic!("{} has no shape for %pto2", f.name);
            });
            (sh.rows, sh.cols)
        })
        .collect();
    seen.sort();
    let mut want = vec![
        (1, 1024),
        (1, 4096),
        (1, 8192),
        (4, 256),
        (16, 256),
        (16, 512),
    ];
    want.sort();
    assert_eq!(
        seen, want,
        "the six kernels' `%pto2` shapes collapsed onto one another"
    );

    // And each kernel's own block must carry its own shape and no other's.
    let out = reemit_pto(&src).expect("re-emit");
    let blocks = func_blocks(&out);
    assert_eq!(blocks.len(), 6, "expected six function blocks");
    for (i, f) in m.funcs.iter().enumerate() {
        let sh = f.shapes[&pto2];
        let mine = format!("rows={}, cols={}", sh.rows, sh.cols);
        assert!(
            blocks[i].contains(&mine),
            "{} lost its own {mine}\n{}",
            f.name,
            blocks[i]
        );
        for (r, c) in &want {
            if (*r, *c) == (sh.rows, sh.cols) {
                continue;
            }
            let theirs = format!("rows={r}, cols={c}");
            assert!(
                !blocks[i].contains(&theirs),
                "{} carries another kernel's {theirs}\n{}",
                f.name,
                blocks[i]
            );
        }
    }
}
