// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Parse a `.acl.pto` file (stage-1 PTO MLIR) into a [`Stage1Plan`].
//!
//! Recognises only the address-shape skeleton needed for the stage-1 OOB
//! oracle:
//!   - `%cN = arith.constant N : index`
//!   - `%tv = pto.make_tensor_view %arg, shape = [%cR, %cC], strides = [...] : ...`
//!   - `%pv = pto.partition_view %tv, offsets = [%cRO, %cCO], sizes = [%cN, %cM] : ... -> ...`
//!   - `func.func @name(...) { ... }`
//!
//! Every other op (alloc_tile, tload, tstore, tmatmul, vector intrinsics) is
//! silently skipped — they don't carry parent-vs-partition shape info.

use crate::stage1_plan::*;
use std::collections::BTreeMap;

pub fn parse_stage1(src: &str) -> Result<Stage1Plan, String> {
    let mut funcs: Vec<Stage1Func> = Vec::new();
    let mut lines = src.lines().peekable();

    while let Some(raw) = lines.next() {
        let line = raw.trim();
        if let Some(name) = try_parse_func_header(line) {
            let f = parse_func_body(name, &mut lines)?;
            funcs.push(f);
        }
    }
    Ok(Stage1Plan { funcs })
}

fn try_parse_func_header(line: &str) -> Option<String> {
    // `func.func @name(%arg0: ...) { ... }`
    let rest = line.strip_prefix("func.func ")?;
    let at = rest.find('@')?;
    let open = rest[at..].find('(')?;
    Some(rest[at + 1..at + open].trim().to_string())
}

fn parse_func_body<'a, I: Iterator<Item = &'a str>>(
    name: String,
    lines: &mut std::iter::Peekable<I>,
) -> Result<Stage1Func, String> {
    let mut consts: BTreeMap<String, u64> = BTreeMap::new();
    let mut tvs: BTreeMap<String, TensorView> = BTreeMap::new();
    let mut parts: Vec<PartitionView> = Vec::new();
    let mut depth: i32 = 1; // header line ended in `{`

    while let Some(raw) = lines.next() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with("//") {
            continue;
        }

        // crude brace tracking — when depth returns to 0 we exit the func.
        depth += line.matches('{').count() as i32;
        depth -= line.matches('}').count() as i32;
        if depth <= 0 {
            break;
        }

        if let Some((ssa, val)) = try_parse_arith_const_index(line) {
            consts.insert(ssa, val);
            continue;
        }
        if let Some(tv) = try_parse_make_tensor_view(line, &consts)? {
            tvs.insert(tv.ssa.clone(), tv);
            continue;
        }
        if let Some(pv) = try_parse_partition_view(line, &consts)? {
            parts.push(pv);
            continue;
        }
        // anything else: skip silently
    }

    Ok(Stage1Func {
        name,
        tensor_views: tvs,
        partitions: parts,
    })
}

/// `%c1024 = arith.constant 1024 : index`
fn try_parse_arith_const_index(line: &str) -> Option<(String, u64)> {
    let eq = line.find(" = arith.constant ")?;
    let lhs = line[..eq].trim().to_string();
    if !lhs.starts_with('%') {
        return None;
    }
    let rhs = &line[eq + " = arith.constant ".len()..];
    // require `: index` (we only care about index constants for shape ops)
    let colon = rhs.rfind(" : ")?;
    if rhs[colon + 3..].trim() != "index" {
        return None;
    }
    let val_str = rhs[..colon].trim();
    let val: u64 = val_str.parse().ok()?;
    Some((lhs, val))
}

/// `%pto0 = pto.make_tensor_view %arg601, shape = [%c1, %c1024], strides = [%c1024, %c1] : !pto.tensor_view<?x?xf32>`
fn try_parse_make_tensor_view(
    line: &str,
    consts: &BTreeMap<String, u64>,
) -> Result<Option<TensorView>, String> {
    let Some(eq) = line.find(" = pto.make_tensor_view ") else {
        return Ok(None);
    };
    let ssa = line[..eq].trim().to_string();
    let rest = &line[eq + " = pto.make_tensor_view ".len()..];

    // ptr arg ends at first comma
    let ptr_end = rest
        .find(',')
        .ok_or("make_tensor_view: missing ptr comma")?;
    let gm_arg = rest[..ptr_end].trim().to_string();

    // shape = [...]
    let shape_in =
        extract_bracket_list(rest, "shape").ok_or("make_tensor_view: missing shape = [...]")?;
    let shape_ssas = split_csv_ssas(&shape_in);
    // Real corpus has 3-D and higher views (e.g. embedding output
    // `[B, S, H]`). Only the 2-D case is OOB-checkable today; skip
    // anything else rather than failing the parse.
    if shape_ssas.len() != 2 {
        return Ok(None);
    }
    let rows = resolve_const(&shape_ssas[0], consts).ok_or_else(|| {
        format!(
            "make_tensor_view {ssa}: shape rows {} not a known const",
            shape_ssas[0]
        )
    })?;
    let cols = resolve_const(&shape_ssas[1], consts).ok_or_else(|| {
        format!(
            "make_tensor_view {ssa}: shape cols {} not a known const",
            shape_ssas[1]
        )
    })?;

    // dtype from the trailing `!pto.tensor_view<?x?xTY>` type. The corpus
    // prints either ` : !pto.tensor_view<...>` (space-colon-space) or
    // `}: !pto.tensor_view<...>` immediately after a `{...}` attr dict —
    // accept both.
    let ty = trailing_type(rest, "!pto.tensor_view<")
        .ok_or("make_tensor_view: missing trailing `!pto.tensor_view<...>`")?;
    let dtype = parse_tv_dtype(&ty)
        .ok_or_else(|| format!("make_tensor_view {ssa}: cannot parse dtype from `{ty}`"))?;

    Ok(Some(TensorView {
        ssa,
        gm_arg,
        rows,
        cols,
        dtype,
    }))
}

/// `%pto1 = pto.partition_view %pto0, offsets = [%c0, %c0], sizes = [%c1, %c1024] : <in> -> <out>`
fn try_parse_partition_view(
    line: &str,
    consts: &BTreeMap<String, u64>,
) -> Result<Option<PartitionView>, String> {
    let Some(eq) = line.find(" = pto.partition_view ") else {
        return Ok(None);
    };
    let ssa = line[..eq].trim().to_string();
    let rest = &line[eq + " = pto.partition_view ".len()..];

    let parent_end = rest
        .find(',')
        .ok_or("partition_view: missing parent comma")?;
    let parent = rest[..parent_end].trim().to_string();

    let off_in =
        extract_bracket_list(rest, "offsets").ok_or("partition_view: missing offsets = [...]")?;
    let sz_in =
        extract_bracket_list(rest, "sizes").ok_or("partition_view: missing sizes = [...]")?;
    let off_ssas = split_csv_ssas(&off_in);
    let sz_ssas = split_csv_ssas(&sz_in);
    // Higher-rank partition_view (the real corpus has 3-D embedding
    // and 4-D attention slices) — skip rather than fail. The OOB
    // check only knows about 2-D parents anyway.
    if off_ssas.len() != 2 || sz_ssas.len() != 2 {
        return Ok(None);
    }
    // Offsets may be runtime SSAs (loop induction, block index): record None
    // and let the OOB check skip them.
    let row_offset = resolve_const(&off_ssas[0], consts);
    let col_offset = resolve_const(&off_ssas[1], consts);
    let size_rows = resolve_const(&sz_ssas[0], consts).ok_or_else(|| {
        format!(
            "partition_view {ssa}: size rows {} not a known const",
            sz_ssas[0]
        )
    })?;
    let size_cols = resolve_const(&sz_ssas[1], consts).ok_or_else(|| {
        format!(
            "partition_view {ssa}: size cols {} not a known const",
            sz_ssas[1]
        )
    })?;

    // dtype from the output partition_tensor_view<RxCxTY>
    let arrow = rest.rfind("->").ok_or("partition_view: missing `->`")?;
    let out_ty = rest[arrow + 2..].trim();
    let dtype = parse_pv_dtype(out_ty)
        .ok_or_else(|| format!("partition_view {ssa}: cannot parse dtype from `{out_ty}`"))?;

    Ok(Some(PartitionView {
        ssa,
        parent,
        row_offset,
        col_offset,
        size_rows,
        size_cols,
        dtype,
    }))
}

// ─── helpers ────────────────────────────────────────────────────────────────

/// Find `key = [ ... ]` in `s` and return the bracket interior (no brackets).
fn extract_bracket_list(s: &str, key: &str) -> Option<String> {
    let needle = format!("{} = [", key);
    let i = s.find(&needle)?;
    let after = &s[i + needle.len()..];
    let close = after.find(']')?;
    Some(after[..close].to_string())
}

fn split_csv_ssas(s: &str) -> Vec<String> {
    s.split(',')
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect()
}

fn resolve_const(ssa: &str, consts: &BTreeMap<String, u64>) -> Option<u64> {
    consts.get(ssa).copied()
}

/// Locate `prefix...>` in the latter half of `s` and return the slice
/// from `prefix` through the matching `>`. Used to extract the trailing
/// MLIR type printed at the end of an op line. Tolerates both
/// `... : !pto.tensor_view<...>` and `...{attr}: !pto.tensor_view<...>`.
fn trailing_type(s: &str, prefix: &str) -> Option<String> {
    let i = s.rfind(prefix)?;
    let after = &s[i..];
    let close = after.find('>')?;
    Some(after[..=close].to_string())
}

/// Extract `TY` from `!pto.tensor_view<?x?xTY>`.
fn parse_tv_dtype(ty: &str) -> Option<String> {
    let inner = ty.strip_prefix("!pto.tensor_view<")?.strip_suffix('>')?;
    Some(inner.rsplit('x').next()?.to_string())
}

/// Extract `TY` from `!pto.partition_tensor_view<RxCxTY>`.
fn parse_pv_dtype(ty: &str) -> Option<String> {
    let inner = ty
        .strip_prefix("!pto.partition_tensor_view<")?
        .strip_suffix('>')?;
    Some(inner.rsplit('x').next()?.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOFTMAX_GOOD: &str = r#"
module {
  func.func @tile_softmax(%arg601: !pto.ptr<f32>, %arg602: !pto.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %pto0 = pto.make_tensor_view %arg601, shape = [%c1, %c1024], strides = [%c1024, %c1] : !pto.tensor_view<?x?xf32>
    %pto1 = pto.partition_view %pto0, offsets = [%c0, %c0], sizes = [%c1, %c1024] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<1x1024xf32>
    return
  }
}
"#;

    #[test]
    fn parses_make_and_partition_view() {
        let plan = parse_stage1(SOFTMAX_GOOD).expect("parse ok");
        assert_eq!(plan.funcs.len(), 1);
        let f = &plan.funcs[0];
        assert_eq!(f.name, "tile_softmax");
        assert_eq!(f.tensor_views.len(), 1);
        assert_eq!(f.partitions.len(), 1);
        let tv = f.tensor_views.get("%pto0").unwrap();
        assert_eq!(tv.rows, 1);
        assert_eq!(tv.cols, 1024);
        assert_eq!(tv.dtype, "f32");
        let pv = &f.partitions[0];
        assert_eq!(pv.parent, "%pto0");
        assert_eq!(pv.row_offset, Some(0));
        assert_eq!(pv.col_offset, Some(0));
        assert_eq!(pv.size_rows, 1);
        assert_eq!(pv.size_cols, 1024);
    }

    #[test]
    fn parses_oob_fixture() {
        // Parent is 1×1024 but partition reads with offset [2, 166] +
        // size [1, 1024] — both row and col are OOB. The parser should
        // accept it (the *check* is what flags the violation).
        let bad = r#"
module {
  func.func @bad(%arg0: !pto.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c166 = arith.constant 166 : index
    %c1024 = arith.constant 1024 : index
    %pto0 = pto.make_tensor_view %arg0, shape = [%c1, %c1024], strides = [%c1024, %c1] : !pto.tensor_view<?x?xf32>
    %pto1 = pto.partition_view %pto0, offsets = [%c2, %c166], sizes = [%c1, %c1024] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<1x1024xf32>
    return
  }
}
"#;
        let plan = parse_stage1(bad).expect("parse ok");
        let pv = &plan.funcs[0].partitions[0];
        assert_eq!(pv.row_offset, Some(2));
        assert_eq!(pv.col_offset, Some(166));
    }

    #[test]
    fn skips_unrecognised_ops() {
        // Lines that don't match are silently dropped — alloc_tile, tload, etc.
        let plan = parse_stage1(SOFTMAX_GOOD).unwrap();
        assert_eq!(plan.funcs[0].partitions.len(), 1);
    }

    /// Real corpus on 910c (`/tmp/mgather_skip_ptoas/.../main_incore_0.pto`)
    /// prints `make_tensor_view` with a `{layout = #pto.layout<nd>}: !pto…`
    /// attr dict immediately before the type colon, and includes 3-D
    /// embedding views. Regression for both: attr-dict tolerance + skip
    /// (rather than fail) on rank≠2 shapes.
    #[test]
    fn parses_real_corpus_shape() {
        let real = r#"
module attributes {pto.target_arch = "a5"} {
  func.func @main_incore_0(%arg0: !pto.ptr<bf16>, %arg1: !pto.ptr<i32>, %arg2: !pto.ptr<bf16>) {
    %c1024_index = arith.constant 1024 : index
    %c128_index = arith.constant 128 : index
    %c1_index = arith.constant 1 : index
    %c2_index = arith.constant 2 : index
    %c8_index = arith.constant 8 : index
    %c0_index = arith.constant 0 : index
    %embed_view = pto.make_tensor_view %arg0, shape = [%c1024_index, %c128_index], strides = [%c128_index, %c1_index] {layout = #pto.layout<nd>}: !pto.tensor_view<?x?xbf16>
    %ids_view   = pto.make_tensor_view %arg1, shape = [%c2_index, %c8_index], strides = [%c8_index, %c1_index] {layout = #pto.layout<nd>}: !pto.tensor_view<?x?xi32>
    %out_view   = pto.make_tensor_view %arg2, shape = [%c2_index, %c8_index, %c128_index], strides = [%c1024_index, %c128_index, %c1_index] {layout = #pto.layout<nd>}: !pto.tensor_view<?x?x?xbf16>
    %ids_pview  = pto.partition_view %ids_view, offsets = [%c0_index, %c0_index], sizes = [%c2_index, %c8_index] : !pto.tensor_view<?x?xi32> -> !pto.partition_tensor_view<2x8xi32>
    return
  }
}
"#;
        let plan = parse_stage1(real).expect("parse real corpus");
        let f = &plan.funcs[0];
        // 2 of 3 views land (the 3-D output is dropped); 1 partition lands.
        assert_eq!(
            f.tensor_views.len(),
            2,
            "expected 2 2-D views (3-D out dropped), got {:?}",
            f.tensor_views.keys().collect::<Vec<_>>()
        );
        assert_eq!(f.partitions.len(), 1);
        assert_eq!(f.partitions[0].parent, "%ids_view");
        assert_eq!(f.partitions[0].size_rows, 2);
        assert_eq!(f.partitions[0].size_cols, 8);
    }
}
