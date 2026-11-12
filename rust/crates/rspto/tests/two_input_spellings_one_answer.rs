// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! The claim the documentation makes, as a test: the input spellings cannot
//! disagree, and the extension alone decides the mode.
//!
//! `.tile` and `.mlir` are two names for tile-dialect MLIR. The documentation
//! says a file produced by the codegen backend therefore needs no renaming
//! before use. That is only true while both spellings reach the same emitter,
//! and a comment cannot hold that true -- so it is asserted here on real
//! emitter output rather than on the mode enum, which would pass even if the
//! two modes were wired to different emitters.

use std::process::Command;

/// A kernel in the `llvm.func` + `hacc.entry` + `__tile_*` form the emitter
/// consumes. Deliberately self-contained: a fixture copied out of a bench corpus
/// would make this test fail for reasons belonging to that corpus.
const KERNEL: &str = r#"
module {
  llvm.func @vec_add(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    %c32 = llvm.mlir.constant(32 : i32) : i32
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %t0 = llvm.call @__tile_load_f32(%arg0, %c32, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @__tile_load_f32(%arg1, %c32, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t2 = llvm.call @__tile_add_f32(%c0, %t0, %t1, %c32, %c32) : (i32, i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg2, %t2, %c32, %c32) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;

fn rspto() -> Command {
    Command::new(env!("CARGO_BIN_EXE_rspto"))
}

fn scratch(name: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("rspto_it_{}_{}", std::process::id(), name));
    p
}

#[test]
fn the_two_mlir_spellings_emit_byte_identical_pto() {
    let as_tile = scratch("a.tile");
    let as_mlir = scratch("a.mlir");
    std::fs::write(&as_tile, KERNEL).unwrap();
    std::fs::write(&as_mlir, KERNEL).unwrap();

    let out_tile = scratch("a_from_tile.pto");
    let out_mlir = scratch("a_from_mlir.pto");

    for (src, dst) in [(&as_tile, &out_tile), (&as_mlir, &out_mlir)] {
        let st = rspto().arg(src).arg("-o").arg(dst).status().unwrap();
        assert!(st.success(), "rspto refused {}", src.display());
    }

    let a = std::fs::read(&out_tile).unwrap();
    let b = std::fs::read(&out_mlir).unwrap();
    assert_eq!(
        a, b,
        "the .tile and .mlir spellings produced different PTO, so they are not \
         reaching the same emitter"
    );
    assert!(!a.is_empty(), "the emitter wrote nothing");

    for p in [as_tile, as_mlir, out_tile, out_mlir] {
        let _ = std::fs::remove_file(p);
    }
}

#[test]
fn the_emitted_pto_is_pto_and_not_the_input_echoed_back() {
    // Byte-identity between two spellings would also hold if both simply copied
    // the input, so identity alone is not enough: assert the output is PTO.
    let src = scratch("b.tile");
    let dst = scratch("b.pto");
    std::fs::write(&src, KERNEL).unwrap();
    assert!(rspto().arg(&src).arg("-o").arg(&dst).status().unwrap().success());

    let pto = std::fs::read_to_string(&dst).unwrap();
    assert!(pto.contains("func.func @vec_add("), "not PTO:\n{pto}");
    assert!(pto.contains("!pto.ptr<f32>"), "no PTO types:\n{pto}");
    assert!(pto.contains("pto.tadd"), "the add did not lower:\n{pto}");
    assert!(!pto.contains("llvm.func"), "the input was echoed:\n{pto}");

    let _ = std::fs::remove_file(src);
    let _ = std::fs::remove_file(dst);
}

#[test]
fn emit_tile_stops_at_the_mlir() {
    let src = scratch("c.tile");
    let dst = scratch("c_out.tile");
    std::fs::write(&src, KERNEL).unwrap();
    assert!(rspto()
        .arg(&src)
        .arg("--emit")
        .arg("tile")
        .arg("-o")
        .arg(&dst)
        .status()
        .unwrap()
        .success());
    let text = std::fs::read_to_string(&dst).unwrap();
    assert!(text.contains("llvm.func @vec_add"), "not the MLIR:\n{text}");
    assert!(!text.contains("pto.tadd"), "it emitted PTO, not MLIR");

    let _ = std::fs::remove_file(src);
    let _ = std::fs::remove_file(dst);
}

#[test]
fn rust_input_is_recognised_and_refused_with_the_toolchain_code() {
    // Not exit 1 and not exit 2: the mode is understood, the half that serves it
    // is absent. A user who gets "unknown extension" here would go looking for a
    // typo instead of for the missing toolchain.
    let src = scratch("d.rs");
    std::fs::write(&src, "// a tile-rs kernel\n").unwrap();
    let st = rspto().arg(&src).status().unwrap();
    assert_eq!(st.code(), Some(4), "expected the toolchain-unavailable code");
    let _ = std::fs::remove_file(src);
}

#[test]
fn an_unknown_extension_is_a_usage_error() {
    let src = scratch("e.txt");
    std::fs::write(&src, KERNEL).unwrap();
    let st = rspto().arg(&src).status().unwrap();
    assert_eq!(st.code(), Some(2), "expected the usage code");
    let _ = std::fs::remove_file(src);
}

#[test]
fn pto_is_the_only_registered_target() {
    // The licence gate is a lint over source text; this is the same claim
    // checked through behaviour, which is what a user would actually hit.
    //
    // Asserting the exact set rather than probing a list of absent names. Two
    // reasons, and the second is why the first version of this test failed the
    // licence lint: a probe list only catches the targets someone thought to
    // write down, and writing those names here would itself put other vendors'
    // target names into this tree.
    use tile_codegen::registry::TargetRegistry;
    let mut r = TargetRegistry::new();
    rspto_emit::register(&mut r);
    assert_eq!(
        r.names(),
        vec!["pto"],
        "a tree that carries one emitter must register exactly one target"
    );
}
