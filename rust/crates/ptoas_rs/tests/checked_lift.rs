// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Regressions for accepted source being silently changed by the checked lift.
use ptoas_rs::convert_pto_to_tile_safe as lift;

const SOFTMAX: &str = include_str!("fixtures/softmax.pto");

#[test]
fn hardware_sized_smoke_fixtures_preserve_shapes_and_operations() {
    let softmax = lift(include_str!("fixtures/softmax_32x32.pto")).unwrap();
    assert!(softmax.contains("__tile_softmax_f32(0,"));
    assert!(softmax.contains("32u32, 32u32"));
    let matmul = lift(include_str!("fixtures/matmul_32x64x32.pto")).unwrap();
    assert!(matmul.contains("__tile_matmul_f32(0,"));
    assert!(matmul.contains("32u32, 64u32, 32u32"));
}

fn copy_kernel(name: &str, cols: u32) -> String {
    format!(
        r#"func.func @{name}(%a: !pto.ptr<f32>, %b: !pto.ptr<f32>) {{
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %rows = arith.constant 2 : index
  %cols = arith.constant {cols} : index
  %in = pto.make_tensor_view %a, shape = [%rows, %cols], strides = [%cols, %one] : !pto.tensor_view<?x?xf32>
  %iv = pto.partition_view %in, offsets = [%zero, %zero], sizes = [%rows, %cols] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<2x{cols}xf32>
  %tile = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=2, cols={cols}, v_row=2, v_col={cols}>
  pto.tload ins(%iv : !pto.partition_tensor_view<2x{cols}xf32>) outs(%tile : !pto.tile_buf<loc=vec, dtype=f32, rows=2, cols={cols}>)
  %out = pto.make_tensor_view %b, shape = [%rows, %cols], strides = [%cols, %one] : !pto.tensor_view<?x?xf32>
  %ov = pto.partition_view %out, offsets = [%zero, %zero], sizes = [%rows, %cols] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<2x{cols}xf32>
  pto.tstore ins(%tile : !pto.tile_buf<loc=vec, dtype=f32, rows=2, cols={cols}>) outs(%ov : !pto.partition_tensor_view<2x{cols}xf32>)
  return
}}
"#
    )
}

#[test]
fn clean_softmax_has_one_fused_op_and_explicit_pointer_boundary() {
    let rust = lift(SOFTMAX).unwrap();
    assert!(rust.contains("pub unsafe fn tile_softmax(arg0: *const f32, arg1: *mut f32)"));
    assert!(rust.contains("unsafe { __tile_load_f32(arg0, 1u32, 1024u32) }"));
    assert_eq!(rust.matches("__tile_softmax_f32(0,").count(), 1);
    assert!(
        !rust.contains("GmView"),
        "view macro would introduce block offsets"
    );
}

#[test]
fn functions_keep_constants_and_tile_shapes_local() {
    let source = format!(
        "module {{\n{}{} }}",
        copy_kernel("first", 4),
        copy_kernel("second", 8)
    );
    let rust = lift(&source).unwrap();
    assert!(rust.contains("__tile_load_f32(arg0, 2u32, 4u32)"));
    assert!(rust.contains("__tile_load_f32(arg0, 2u32, 8u32)"));
    let nonzero = source.replacen("%zero = arith.constant 0", "%zero = arith.constant 1", 1);
    assert!(lift(&nonzero).unwrap_err().contains("offsets"));
}

#[test]
fn parent_row_stride_cannot_be_lost() {
    let source = copy_kernel("strided", 4).replacen(
        "shape = [%rows, %cols], strides = [%cols, %one]",
        "shape = [%rows, 8], strides = [8, %one]",
        1,
    );
    assert!(lift(&source)
        .unwrap_err()
        .contains("complete parent tensor shape"));
}

#[test]
fn sizes_operands_must_agree_with_partition_type() {
    let source =
        copy_kernel("sizes", 4).replacen("sizes = [%rows, %cols]", "sizes = [%one, %cols]", 1);
    assert!(lift(&source).unwrap_err().contains("partition sizes/type"));
}

#[test]
fn pto_types_do_not_hide_unknown_calls() {
    let source = copy_kernel("effects", 4).replace(
        "  pto.tload",
        "  func.call @mutate(%a) : (!pto.ptr<f32>) -> ()\n  pto.tload",
    );
    assert!(lift(&source).unwrap_err().contains("func.call"));
}

#[test]
fn a_second_operation_cannot_hide_at_the_end_of_a_line() {
    let source = copy_kernel("trailing_effect", 4)
        .lines()
        .map(|line| {
            if line.trim_start().starts_with("pto.tload ") {
                format!("{line} func.call @mutate(%a) : (!pto.ptr<f32>) -> ()")
            } else {
                line.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert!(lift(&source)
        .unwrap_err()
        .contains("one operation per line"));
}

#[test]
fn empty_implementations_are_refused() {
    for source in ["", "module {\n}\n", "func.func @empty() {\nreturn\n}\n"] {
        assert!(lift(source).is_err(), "empty kernel accepted: {source}");
    }
}

#[test]
fn returns_and_code_after_return_are_not_discarded() {
    let source = copy_kernel("result", 4);
    assert!(lift(&source.replacen(") {", ") -> !pto.ptr<f32> {", 1)).is_err());
    assert!(lift(&source.replace("  return", "  return %b : !pto.ptr<f32>")).is_err());
    assert!(lift(&source.replace("  return", "  return\n  arith.constant 1 : index")).is_err());
}

#[test]
fn partial_active_tiles_are_refused() {
    let source = copy_kernel("partial", 4).replace("v_row=2, v_col=4", "v_row=1, v_col=4");
    assert!(lift(&source)
        .unwrap_err()
        .contains("partial or dynamic v_row"));
}

#[test]
fn ssa_punctuation_does_not_alias_different_inputs() {
    let source = r#"func.func @add(%a: !pto.ptr<f32>, %b: !pto.ptr<f32>, %c: !pto.ptr<f32>) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %four = arith.constant 4 : index
  %ta = pto.make_tensor_view %a, shape = [%one, %four], strides = [%four, %one] : !pto.tensor_view<?x?xf32>
  %tb = pto.make_tensor_view %b, shape = [%one, %four], strides = [%four, %one] : !pto.tensor_view<?x?xf32>
  %tc = pto.make_tensor_view %c, shape = [%one, %four], strides = [%four, %one] : !pto.tensor_view<?x?xf32>
  %pa = pto.partition_view %ta, offsets = [%zero, %zero], sizes = [%one, %four] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<1x4xf32>
  %pb = pto.partition_view %tb, offsets = [%zero, %zero], sizes = [%one, %four] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<1x4xf32>
  %pc = pto.partition_view %tc, offsets = [%zero, %zero], sizes = [%one, %four] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<1x4xf32>
  %a.0 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=4>
  %a_0 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=4>
  %sum = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=4>
  pto.tload ins(%pa : !pto.partition_tensor_view<1x4xf32>) outs(%a.0 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=4>)
  pto.tload ins(%pb : !pto.partition_tensor_view<1x4xf32>) outs(%a_0 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=4>)
  pto.tadd ins(%a.0, %a_0 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=4>, !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=4>) outs(%sum : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=4>)
  pto.tstore ins(%sum : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=4>) outs(%pc : !pto.partition_tensor_view<1x4xf32>)
  return
}
"#;
    let rust = lift(source).unwrap();
    assert!(
        rust.contains("__tile_add_f32(0, v_a_2e0, v_a_5f0,"),
        "distinct inputs were aliased:\n{rust}"
    );
}
