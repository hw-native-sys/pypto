// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Shared MLIR text-parsing helpers consumed by the `mlir_to_cpp` (AscendC
//! C++) and `mlir_to_msl` (Apple Metal Shading Language) emitters, as well as
//! any future MLIR back end that needs to walk the same `@__tile_*`
//! intrinsic shape produced by `rustc_codegen_tile`.
//!
//! ## Why a separate file?
//!
//! Both `mlir_to_cpp.rs` and `mlir_to_msl.rs` are re-included via
//! `#[path = "..."] mod ...;` from lightweight test crates that have zero
//! third-party dependencies. So this module is **strictly std-only** — no
//! MLIR, LLVM, melior or rustc deps. Any helper that needs richer types
//! belongs in the consuming emitter, not here.
//!
//! ## Scope at extraction time (B1)
//!
//! - `MlirModule`, `MlirFunc`, `MlirGlobal`, `FuncArg` value types
//! - `parse_module` and its parsing primitives
//! - Brace/paren counters used to track block depth
//!
//! `KernelAnalysis` (the cpp-tile pipeline metadata bag) and
//! `determine_element_type` (which depends on it) stay in `mlir_to_cpp.rs`
//! — they are not reusable by the MSL emitter.
//!
//! Per-consumer dead code is expected: when this file is `#[path]`-included
//! from a test crate that exercises only one emitter (e.g.
//! `mlir_to_msl_tests`), helpers used only by the other emitter look
//! unreferenced. The crate-level `allow(dead_code)` keeps those builds clean.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

/// Count `{` characters in `s`. Used by parsers to track block depth.
pub fn count_open_braces(s: &str) -> usize {
    s.chars().filter(|&c| c == '{').count()
}

/// Count `}` characters in `s`. Used by parsers to track block depth.
pub fn count_close_braces(s: &str) -> usize {
    s.chars().filter(|&c| c == '}').count()
}

/// Count `(` characters in `s`. Used by parsers to track region nesting.
pub fn count_open_parens(s: &str) -> usize {
    s.chars().filter(|&c| c == '(').count()
}

/// Count `)` characters in `s`. Used by parsers to track region nesting.
pub fn count_close_parens(s: &str) -> usize {
    s.chars().filter(|&c| c == ')').count()
}
/// Map an MLIR scalar type string to a C++ type string.
pub fn mlir_scalar_type_to_cpp(ty: &str) -> String {
    match ty {
        "i8" => "int8_t".to_string(),
        "i16" => "int16_t".to_string(),
        "i32" => "int32_t".to_string(),
        "i64" => "int64_t".to_string(),
        "f32" => "float".to_string(),
        "f16" => "half".to_string(),
        _ => {
            // For array types, return element type
            if let Some(content) = ty
                .strip_prefix("!llvm.array<")
                .and_then(|s| s.strip_suffix('>'))
            {
                if let Some(x_pos) = content.find(" x ") {
                    return mlir_scalar_type_to_cpp(content[x_pos + 3..].trim());
                }
            }
            // tensor<NxT> format from dense attributes
            if let Some(content) = ty.strip_prefix("tensor<").and_then(|s| s.strip_suffix('>')) {
                if let Some(x_pos) = content.find('x') {
                    return mlir_scalar_type_to_cpp(content[x_pos + 1..].trim());
                }
            }
            "int32_t".to_string()
        }
    }
}

// ---------------------------------------------------------------------------
// MLIR text parsing
// ---------------------------------------------------------------------------

pub struct MlirFunc {
    pub name: String,
    pub args: Vec<FuncArg>,
    pub is_entry: bool,
    pub body_lines: Vec<String>,
    /// Per-tile shared-memory / UB budget in bytes, from a `tile.budget = <N>`
    /// function attribute (emitted by `#[tile_kernel(budget=..)]`). When present,
    /// backends emit a static capacity guard (F1) into the generated kernel so the
    /// budget guarantee reaches the DEPLOYED code, not just the typed source.
    pub budget: Option<u32>,
    /// K-loop unroll factor from a `tile.k_unroll = <N>` function attribute
    /// (emitted by `#[tile_kernel(schedule(k_unroll=..))]`). `None` = the
    /// backend's receipted default. Only honored where
    /// `emit_unrolled_k_accumulation` preserves sum order (same arithmetic).
    pub k_unroll: Option<u32>,
}

pub struct FuncArg {
    pub name: String, // e.g. "%arg0"
    pub ty: String,   // e.g. "!llvm.ptr<1>"
    pub is_gm: bool,  // true if ty contains "ptr<1>"
}

/// A global variable parsed from `llvm.mlir.global` declarations.
pub struct MlirGlobal {
    pub name: String,
    pub ty: String,                // e.g. "i32", "f32", "!llvm.array<5 x i32>"
    pub value: Option<String>,     // e.g. "42", "3.14159", "dense<[10, 20, 30]>"
    pub elem_type: Option<String>, // e.g. "i32" for arrays, None for scalars
    pub count: Option<usize>,      // array element count if applicable
}

pub struct MlirModule {
    pub functions: Vec<MlirFunc>,
    pub globals: Vec<MlirGlobal>,
}

/// Extract `tile.budget = <N>` (bytes) from a function-attribute line, in either
/// the pretty (`... attributes {hacc.entry, tile.budget = 4096 : i32} {`) or the
/// generic (`}) {hacc.entry, ..., tile.budget = 4096 : i32} : () -> ()`) format.
fn parse_budget_attr(line: &str) -> Option<u32> {
    let pos = line.find("tile.budget")?;
    let after_eq = line[pos + "tile.budget".len()..].split('=').nth(1)?;
    let digits: String = after_eq
        .trim()
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse::<u32>().ok()
}

/// Extract `tile.k_unroll = <N>` from a function-attribute line (pretty or
/// generic), same shape as [`parse_budget_attr`].
fn parse_k_unroll_attr(line: &str) -> Option<u32> {
    let pos = line.find("tile.k_unroll")?;
    let after_eq = line[pos + "tile.k_unroll".len()..].split('=').nth(1)?;
    let digits: String = after_eq
        .trim()
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse::<u32>().ok()
}

pub fn parse_module(mlir_text: &str) -> Result<MlirModule, String> {
    let mut functions = Vec::new();
    let mut globals = Vec::new();
    let lines: Vec<&str> = mlir_text.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i].trim();

        // Match global: pretty `llvm.mlir.global ... @name...` or generic
        // `"llvm.mlir.global"() <{..., sym_name = "name", ...}> ({ ... })`.
        if line.contains("llvm.mlir.global") {
            if let Some(g) = parse_global_decl(line) {
                globals.push(g);
            }
            i += 1;
        // Match function header in pretty format: llvm.func @name(args) attributes {...} {
        } else if line.contains("llvm.func @") {
            let name = extract_func_name(line)
                .ok_or_else(|| format!("Cannot parse function name from: {}", line))?;
            let args = extract_func_args(line);
            let is_entry = line.contains("hacc.entry");
            let budget = parse_budget_attr(line);
            let k_unroll = parse_k_unroll_attr(line);

            // Collect function body by tracking brace nesting
            let mut body_lines = Vec::new();
            let mut depth = count_open_braces(line) as i32 - count_close_braces(line) as i32;

            i += 1;
            while i < lines.len() && depth > 0 {
                let body_line = lines[i];
                depth += count_open_braces(body_line) as i32;
                depth -= count_close_braces(body_line) as i32;

                if depth > 0 {
                    // Don't include the final closing brace
                    body_lines.push(body_line.trim().to_string());
                }
                i += 1;
            }

            functions.push(MlirFunc {
                name,
                args,
                is_entry,
                body_lines,
                budget,
                k_unroll,
            });
        // Match function in MLIR generic format:
        //   "llvm.func"() <{sym_name = "name", function_type = ..., ...}> ({
        //   ^bb0(%arg0: type, ...):
        //     body...
        //   }) {hacc.entry, ...} : () -> ()
        } else if line.starts_with("\"llvm.func\"()") {
            let name = extract_generic_sym_name(line);
            let func_type_args = extract_generic_func_type_args(line);

            // Collect body and the closing line (which has discardable attrs)
            let mut body_lines = Vec::new();
            let mut is_entry = false;
            let mut budget: Option<u32> = None;
            let mut k_unroll: Option<u32> = None;
            let mut bb_args = Vec::new();

            // Track parenthesized region nesting: the region starts with ({
            // and ends with }) on a line.
            let mut depth = count_open_parens(line) as i32 - count_close_parens(line) as i32;

            i += 1;
            while i < lines.len() && depth > 0 {
                let body_line = lines[i].trim();
                depth += count_open_parens(body_line) as i32;
                depth -= count_close_parens(body_line) as i32;

                if depth <= 0 {
                    // This is the closing line: }) {hacc.entry, ...} : () -> ()
                    is_entry = body_line.contains("hacc.entry");
                    budget = parse_budget_attr(body_line);
                    k_unroll = parse_k_unroll_attr(body_line);
                } else if body_line.starts_with("^bb0(") || body_line.starts_with("^bb0:") {
                    // Extract args from ^bb0(%arg0: !llvm.ptr<1>, ...)
                    bb_args = extract_bb_args(body_line);
                } else {
                    body_lines.push(normalize_generic_to_pretty(body_line));
                }
                i += 1;
            }

            // Prefer ^bb0 args (have names), fall back to function_type args
            let args = if !bb_args.is_empty() {
                bb_args
            } else {
                func_type_args
            };

            if let Some(name) = name {
                functions.push(MlirFunc {
                    name,
                    args,
                    is_entry,
                    body_lines,
                    budget,
                    k_unroll,
                });
            }
        } else {
            i += 1;
        }
    }

    Ok(MlirModule { functions, globals })
}

/// Extract `sym_name = "..."` from generic format properties `<{..., sym_name = "foo", ...}>`.
pub fn extract_generic_sym_name(line: &str) -> Option<String> {
    let marker = "sym_name = \"";
    let start = line.find(marker)? + marker.len();
    let rest = &line[start..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// Extract function argument types from generic format `function_type = !llvm.func<void (ptr<1>, ptr<1>)>`.
/// Returns FuncArgs with synthetic names (%arg0, %arg1, ...).
pub fn extract_generic_func_type_args(line: &str) -> Vec<FuncArg> {
    let marker = "function_type = !llvm.func<";
    let start = match line.find(marker) {
        Some(p) => p + marker.len(),
        None => return Vec::new(),
    };
    let rest = &line[start..];
    // Find the matching > for the outer func type
    let mut depth = 1i32;
    let mut end = 0;
    for (j, ch) in rest.char_indices() {
        match ch {
            '<' => depth += 1,
            '>' => {
                depth -= 1;
                if depth == 0 {
                    end = j;
                    break;
                }
            }
            _ => {}
        }
    }
    let func_sig = &rest[..end]; // e.g. "void (ptr<1>, ptr<1>, ptr<1>)"

    // Find the argument types between ( and )
    let open = match func_sig.find('(') {
        Some(p) => p,
        None => return Vec::new(),
    };
    let close = func_sig.len().saturating_sub(1);
    if open + 1 > close {
        return Vec::new();
    }
    let arg_str = &func_sig[open + 1..close];
    if arg_str.trim().is_empty() {
        return Vec::new();
    }

    // Split by comma, handling nested <>
    let mut args = Vec::new();
    let mut current = String::new();
    let mut angle_depth = 0i32;
    let mut arg_idx = 0;
    for ch in arg_str.chars() {
        match ch {
            '<' => {
                angle_depth += 1;
                current.push(ch);
            }
            '>' => {
                angle_depth -= 1;
                current.push(ch);
            }
            ',' if angle_depth == 0 => {
                let ty = current.trim().to_string();
                if !ty.is_empty() {
                    let is_gm = ty.contains("ptr<1>");
                    args.push(FuncArg {
                        name: format!("%arg{}", arg_idx),
                        ty,
                        is_gm,
                    });
                    arg_idx += 1;
                }
                current.clear();
            }
            _ => current.push(ch),
        }
    }
    let ty = current.trim().to_string();
    if !ty.is_empty() {
        let is_gm = ty.contains("ptr<1>");
        args.push(FuncArg {
            name: format!("%arg{}", arg_idx),
            ty,
            is_gm,
        });
    }
    args
}

/// Extract arguments from a basic block header: ^bb0(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>):
pub fn extract_bb_args(line: &str) -> Vec<FuncArg> {
    let open = match line.find('(') {
        Some(p) => p,
        None => return Vec::new(),
    };
    // Find matching close paren
    let mut depth = 0;
    let mut close = open;
    for (j, ch) in line[open..].char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    close = open + j;
                    break;
                }
            }
            _ => {}
        }
    }
    // Unbalanced/truncated input: no matching ')' (close <= open) -> bail,
    // don't slice line[open+1..close] with start > end (panic).
    if close <= open {
        return Vec::new();
    }
    let arg_str = &line[open + 1..close];
    if arg_str.trim().is_empty() {
        return Vec::new();
    }

    // Split by comma, handling nested <>
    let mut args = Vec::new();
    let mut current = String::new();
    let mut angle_depth = 0i32;
    for ch in arg_str.chars() {
        match ch {
            '<' => {
                angle_depth += 1;
                current.push(ch);
            }
            '>' => {
                angle_depth -= 1;
                current.push(ch);
            }
            ',' if angle_depth == 0 => {
                if let Some(arg) = parse_single_bb_arg(&current) {
                    args.push(arg);
                }
                current.clear();
            }
            _ => current.push(ch),
        }
    }
    if let Some(arg) = parse_single_bb_arg(&current) {
        args.push(arg);
    }
    args
}

/// Parse a single BB arg like "%arg1704: !llvm.ptr<1>" into a FuncArg.
pub fn parse_single_bb_arg(s: &str) -> Option<FuncArg> {
    let s = s.trim();
    let colon = s.find(':')?;
    let name = s[..colon].trim().to_string();
    let ty = s[colon + 1..].trim().to_string();
    let is_gm = ty.contains("ptr<1>");
    Some(FuncArg { name, ty, is_gm })
}

/// Normalize a generic-format MLIR body line to pseudo-pretty-format.
///
/// Converts patterns like:
///   `%r = "llvm.call"(%a, %b) <{callee = @func, ...}> : (T1, T2) -> T3`
///   → `%r = llvm.call @func(%a, %b) : (T1, T2) -> T3`
///
///   `%r = "llvm.mlir.constant"() <{value = 42 : i64}> : () -> i32`
///   → `%r = llvm.mlir.constant(42 : i64) : i32`
///
///   `%r = "llvm.load"(%ptr) <{alignment = 4}> : (!llvm.ptr<1>) -> i32`
///   → `%r = llvm.load %ptr : !llvm.ptr<1> -> i32`
///
///   `"llvm.store"(%val, %ptr) <{alignment = 4}> : (i32, !llvm.ptr<1>) -> ()`
///   → `llvm.store %val, %ptr : i32, !llvm.ptr<1>`
///
///   `%r = "llvm.bitcast"(%val) : (i32) -> i32`
///   → `%r = llvm.bitcast %val : i32 to i32`
///
///   `%r = "llvm.alloca"(%size) <{elem_type = i8, ...}> : (i32) -> !llvm.ptr<1>`
///   → `%r = llvm.alloca %size x i8 : (i32) -> !llvm.ptr<1>`
///
///   `"llvm.br"(%a, %b)[^bb1] <{}> : (i32, i32)`
///   → `llvm.br ^bb1(%a, %b : i32, i32)`
///
///   `"llvm.cond_br"(%cond)[^bb1, ^bb2] <{...}> : (i1)`
///   → `llvm.cond_br %cond, ^bb1(...), ^bb2(...)`
///
///   `%r = "llvm.getelementptr"(%ptr, %idx) <{elem_type = f32, ...}> : (TYPE, TYPE) -> TYPE`
///   → `%r = llvm.getelementptr %ptr[%idx] : (TYPE, TYPE) -> TYPE, f32`
pub fn normalize_generic_to_pretty(line: &str) -> String {
    let trimmed = line.trim();

    // Handle "cf.br" and "cf.cond_br" in generic form before the "llvm." check
    if trimmed.contains("\"cf.br\"") || trimmed.contains("\"cf.cond_br\"") {
        return normalize_cf_branch(trimmed);
    }

    // Only process lines containing the generic op format ("op.name")
    if !trimmed.contains("\"llvm.") && !trimmed.contains("\"arith.") {
        return trimmed.to_string();
    }

    // Extract result prefix if present: %name =
    let (result_prefix, rest) = if let Some(eq_pos) = trimmed.find('=') {
        let before = trimmed[..eq_pos].trim();
        if before.starts_with('%') {
            (format!("{} = ", before), trimmed[eq_pos + 1..].trim())
        } else {
            (String::new(), trimmed)
        }
    } else {
        (String::new(), trimmed)
    };

    // --- llvm.call: "llvm.call"(args) <{callee = @func, ...}> : types ---
    if rest.starts_with("\"llvm.call\"") {
        let after_op = &rest["\"llvm.call\"".len()..];
        let operands = extract_paren_contents(after_op);
        let callee = extract_property(rest, "callee");
        let type_sig = extract_type_signature(rest);
        return format!(
            "{}llvm.call {}({}) : {}",
            result_prefix,
            callee.unwrap_or_default(),
            operands,
            type_sig
        );
    }

    // --- llvm.mlir.constant: "llvm.mlir.constant"() <{value = V : T}> : () -> RT ---
    if rest.starts_with("\"llvm.mlir.constant\"") {
        if let Some(value) = extract_property(rest, "value") {
            let result_type = extract_arrow_result_type(rest);
            return format!(
                "{}llvm.mlir.constant({}) : {}",
                result_prefix, value, result_type
            );
        }
    }

    // --- llvm.load: "llvm.load"(%ptr) <{...}> : (PT) -> RT ---
    if rest.starts_with("\"llvm.load\"") {
        let after_op = &rest["\"llvm.load\"".len()..];
        let operand = extract_paren_contents(after_op);
        let type_sig = extract_type_signature(rest);
        // Convert "(PT) -> RT" to "PT -> RT" for the pretty format parser
        let pretty_type = type_sig.trim_start_matches('(').replacen(") ->", " ->", 1);
        return format!("{}llvm.load {} : {}", result_prefix, operand, pretty_type);
    }

    // --- llvm.store: "llvm.store"(%val, %ptr) <{...}> : (VT, PT) -> () ---
    if rest.starts_with("\"llvm.store\"") {
        let after_op = &rest["\"llvm.store\"".len()..];
        let operands = extract_paren_contents(after_op);
        let type_sig = extract_type_signature(rest);
        // Convert operands "a, b" to "a, b" and type "(VT, PT) -> ()" to "VT, PT"
        let types = type_sig
            .split("->")
            .next()
            .unwrap_or("")
            .trim()
            .trim_start_matches('(')
            .trim_end_matches(')')
            .trim();
        return format!("{}llvm.store {} : {}", result_prefix, operands, types);
    }

    // --- llvm.bitcast: "llvm.bitcast"(%val) : (FT) -> TT ---
    if rest.starts_with("\"llvm.bitcast\"") {
        let after_op = &rest["\"llvm.bitcast\"".len()..];
        let operand = extract_paren_contents(after_op);
        let type_sig = extract_type_signature(rest);
        // Parse "(from_type) -> to_type"
        let parts: Vec<&str> = type_sig.splitn(2, "->").collect();
        let from_type = parts
            .first()
            .unwrap_or(&"")
            .trim()
            .trim_start_matches('(')
            .trim_end_matches(')')
            .trim();
        let to_type = parts.get(1).unwrap_or(&"").trim();
        return format!(
            "{}llvm.bitcast {} : {} to {}",
            result_prefix, operand, from_type, to_type
        );
    }

    // --- llvm.alloca: "llvm.alloca"(%size) <{elem_type = T, ...}> : types ---
    if rest.starts_with("\"llvm.alloca\"") {
        let after_op = &rest["\"llvm.alloca\"".len()..];
        let operand = extract_paren_contents(after_op);
        let elem_type = extract_property(rest, "elem_type").unwrap_or_else(|| "i8".to_string());
        let type_sig = extract_type_signature(rest);
        return format!(
            "{}llvm.alloca {} x {} : {}",
            result_prefix, operand, elem_type, type_sig
        );
    }

    // --- llvm.getelementptr: "llvm.getelementptr"(%ptr, %idx) <{elem_type = T, ...}> : types ---
    if rest.starts_with("\"llvm.getelementptr\"") {
        let after_op = &rest["\"llvm.getelementptr\"".len()..];
        let operands = extract_paren_contents(after_op);
        let parts: Vec<&str> = operands.splitn(2, ',').collect();
        let base = parts.first().unwrap_or(&"").trim();
        let indices = if parts.len() > 1 { parts[1].trim() } else { "" };
        let elem_type = extract_property(rest, "elem_type").unwrap_or_else(|| "i8".to_string());
        let type_sig = extract_type_signature(rest);
        return format!(
            "{}llvm.getelementptr {}[{}] : {}, {}",
            result_prefix, base, indices, type_sig, elem_type
        );
    }

    // --- llvm.extractvalue: "llvm.extractvalue"(%agg) <{position = array<i64: 0>}> : types ---
    if rest.starts_with("\"llvm.extractvalue\"") {
        let after_op = &rest["\"llvm.extractvalue\"".len()..];
        let operand = extract_paren_contents(after_op);
        // Extract position from <{position = array<i64: N>}>
        let pos = extract_property(rest, "position")
            .and_then(|p| {
                // Parse "array<i64: 0>" → "0"
                p.find(':')
                    .map(|c| p[c + 1..].trim_end_matches('>').trim().to_string())
            })
            .unwrap_or_else(|| "0".to_string());
        let type_sig = extract_type_signature(rest);
        return format!(
            "{}llvm.extractvalue {}[{}] : {}",
            result_prefix, operand, pos, type_sig
        );
    }

    // --- llvm.insertvalue: "llvm.insertvalue"(%val, %agg) <{position = ...}> : types ---
    if rest.starts_with("\"llvm.insertvalue\"") {
        let after_op = &rest["\"llvm.insertvalue\"".len()..];
        let operands = extract_paren_contents(after_op);
        let pos = extract_property(rest, "position")
            .and_then(|p| {
                p.find(':')
                    .map(|c| p[c + 1..].trim_end_matches('>').trim().to_string())
            })
            .unwrap_or_else(|| "0".to_string());
        let type_sig = extract_type_signature(rest);
        let parts: Vec<&str> = operands.splitn(2, ',').collect();
        let val = parts.first().unwrap_or(&"").trim();
        let agg = parts.get(1).unwrap_or(&"").trim();
        return format!(
            "{}llvm.insertvalue {} into {}[{}] : {}",
            result_prefix, val, agg, pos, type_sig
        );
    }

    // --- llvm.br: "llvm.br"(args)[^bb] <{}> : types ---
    if rest.starts_with("\"llvm.br\"") {
        let after_op = &rest["\"llvm.br\"".len()..];
        let operands = extract_paren_contents(after_op);
        // Extract successor block from [^bb1]
        let target = extract_bracket_contents(after_op);
        let type_sig = extract_type_signature(rest);
        if operands.is_empty() {
            return format!("{}llvm.br {}", result_prefix, target);
        }
        return format!(
            "{}llvm.br {}({} : {})",
            result_prefix,
            target,
            operands,
            type_sig.trim_start_matches('(').trim_end_matches(')')
        );
    }

    // --- llvm.cond_br: "llvm.cond_br"(%cond, args...)[^t, ^f] <{...}> : types ---
    if rest.starts_with("\"llvm.cond_br\"") {
        let after_op = &rest["\"llvm.cond_br\"".len()..];
        let operands = extract_paren_contents(after_op);
        let successors = extract_bracket_contents(after_op);
        let succ_parts: Vec<&str> = successors.split(',').collect();
        let true_block = succ_parts.first().unwrap_or(&"^bb0").trim();
        let false_block = succ_parts.get(1).unwrap_or(&"^bb0").trim();

        // First operand is condition, rest are block args
        let op_parts: Vec<&str> = operands.splitn(2, ',').collect();
        let cond = op_parts.first().unwrap_or(&"%0").trim();

        // Get operandSegmentSizes to split args between true/false blocks
        // For simplicity, pass all remaining args to the true block
        let remaining_args = if op_parts.len() > 1 {
            op_parts[1].trim()
        } else {
            ""
        };

        if remaining_args.is_empty() {
            return format!(
                "{}llvm.cond_br {}, {}, {}",
                result_prefix, cond, true_block, false_block
            );
        }
        return format!(
            "{}llvm.cond_br {}, {}({}), {}",
            result_prefix, cond, true_block, remaining_args, false_block
        );
    }

    // --- llvm.intr.*: "llvm.intr.X"(args) <{...}> : types ---
    if let Some(intr_start) = rest.find("\"llvm.intr.") {
        let after_quote = &rest[intr_start + 1..];
        let end_quote = after_quote.find('"').unwrap_or(after_quote.len());
        let intr_name = &after_quote[..end_quote]; // e.g. "llvm.intr.umul.with.overflow"
        let after_op_end = &rest[intr_start + end_quote + 2..]; // after closing "
        let operands = extract_paren_contents(after_op_end);
        let type_sig = extract_type_signature(rest);
        return format!("{}{} {} : {}", result_prefix, intr_name, operands, type_sig);
    }

    // --- Unary type cast ops: "llvm.trunc"(%val) <{...}> : (FROM) -> TO ---
    // Covers: trunc, zext, sext, fptoui, fptosi, uitofp, sitofp, fpext, fptrunc, ptrtoint, inttoptr
    {
        let unary_cast_ops = [
            "llvm.trunc",
            "llvm.zext",
            "llvm.sext",
            "llvm.fptoui",
            "llvm.fptosi",
            "llvm.uitofp",
            "llvm.sitofp",
            "llvm.fpext",
            "llvm.fptrunc",
            "llvm.ptrtoint",
            "llvm.inttoptr",
        ];
        for op in &unary_cast_ops {
            let quoted = format!("\"{}\"", op);
            if rest.starts_with(&quoted) {
                let after_op = &rest[quoted.len()..];
                let operand = extract_paren_contents(after_op);
                let type_sig = extract_type_signature(rest);
                // Parse "(FROM) -> TO" into "FROM to TO"
                let parts: Vec<&str> = type_sig.splitn(2, "->").collect();
                let from_type = parts
                    .first()
                    .unwrap_or(&"")
                    .trim()
                    .trim_start_matches('(')
                    .trim_end_matches(')')
                    .trim();
                let to_type = parts.get(1).unwrap_or(&"").trim();
                return format!(
                    "{}{} {} : {} to {}",
                    result_prefix, op, operand, from_type, to_type
                );
            }
        }
    }

    // --- llvm.fneg: "llvm.fneg"(%val) <{fastmathFlags = ...}> : (f32) -> f32 ---
    // → llvm.fneg %val : f32
    if rest.starts_with("\"llvm.fneg\"") {
        let after_op = &rest["\"llvm.fneg\"".len()..];
        let operand = extract_paren_contents(after_op);
        let result_type = extract_arrow_result_type(rest);
        return format!("{}llvm.fneg {} : {}", result_prefix, operand, result_type);
    }

    // --- arith.negf: "arith.negf"(%val) <{fastmath = ...}> : (f32) -> f32 ---
    // → arith.negf %val : f32
    if rest.starts_with("\"arith.negf\"") {
        let after_op = &rest["\"arith.negf\"".len()..];
        let operand = extract_paren_contents(after_op);
        let result_type = extract_arrow_result_type(rest);
        return format!("{}arith.negf {} : {}", result_prefix, operand, result_type);
    }

    // --- Binary ops: "llvm.add"(%a, %b) <{overflowFlags = ...}> : (T, T) -> T ---
    // Covers: add, sub, mul, udiv, sdiv, urem, srem, and, or, xor, shl, lshr, ashr
    {
        let binary_ops = [
            "llvm.add",
            "llvm.sub",
            "llvm.mul",
            "llvm.udiv",
            "llvm.sdiv",
            "llvm.urem",
            "llvm.srem",
            "llvm.and",
            "llvm.or",
            "llvm.xor",
            "llvm.shl",
            "llvm.lshr",
            "llvm.ashr",
            "llvm.fdiv",
            "llvm.frem",
        ];
        for op in &binary_ops {
            let quoted = format!("\"{}\"", op);
            if rest.starts_with(&quoted) {
                let after_op = &rest[quoted.len()..];
                let operands = extract_paren_contents(after_op);
                let result_type = extract_arrow_result_type(rest);
                return format!("{}{} {} : {}", result_prefix, op, operands, result_type);
            }
        }
    }

    // --- llvm.icmp: "llvm.icmp"(%a, %b) <{predicate = N : i64}> : (T, T) -> i1 ---
    // → llvm.icmp "pred_name" %a, %b : T
    if rest.starts_with("\"llvm.icmp\"") {
        let after_op = &rest["\"llvm.icmp\"".len()..];
        let operands = extract_paren_contents(after_op);
        // Extract predicate number from <{predicate = N : i64}>
        let pred_name = if let Some(pred_pos) = rest.find("predicate = ") {
            let after_pred = &rest[pred_pos + "predicate = ".len()..];
            let num_str: String = after_pred
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            match num_str.parse::<u32>().unwrap_or(0) {
                0 => "eq",
                1 => "ne",
                2 => "slt",
                3 => "sle",
                4 => "sgt",
                5 => "sge",
                6 => "ult",
                7 => "ule",
                8 => "ugt",
                9 => "uge",
                _ => "eq",
            }
        } else {
            "eq"
        };
        // Get the input type (first type in the signature)
        let input_type = extract_type_signature(rest)
            .trim_start_matches('(')
            .split(',')
            .next()
            .unwrap_or("i32")
            .trim()
            .to_string();
        return format!(
            "{}llvm.icmp \"{}\" {} : {}",
            result_prefix, pred_name, operands, input_type
        );
    }

    // --- llvm.select: "llvm.select"(%c, %a, %b) <{...}> : (i1, T, T) -> T ---
    if rest.starts_with("\"llvm.select\"") {
        let after_op = &rest["\"llvm.select\"".len()..];
        let operands = extract_paren_contents(after_op);
        let result_type = extract_arrow_result_type(rest);
        return format!(
            "{}llvm.select {} : {}",
            result_prefix, operands, result_type
        );
    }

    // --- arith.constant: "arith.constant"() <{value = V : T}> : () -> RT ---
    // Normalize to arith.constant(V : T) : RT (matching llvm.mlir.constant format)
    if rest.starts_with("\"arith.constant\"") {
        if let Some(value) = extract_property(rest, "value") {
            let result_type = extract_arrow_result_type(rest);
            return format!(
                "{}arith.constant({}) : {}",
                result_prefix, value, result_type
            );
        }
    }

    // --- arith binary ops: "arith.addi"(%a, %b) : (T, T) -> T ---
    // Integer: addi, subi, muli, divui, divsi, remui, remsi, andi, ori, xori, shli, shrui, shrsi
    // Float: addf, subf, mulf, divf, remf
    {
        let arith_binary_ops = [
            "arith.addi",
            "arith.subi",
            "arith.muli",
            "arith.divui",
            "arith.divsi",
            "arith.remui",
            "arith.remsi",
            "arith.andi",
            "arith.ori",
            "arith.xori",
            "arith.shli",
            "arith.shrui",
            "arith.shrsi",
            "arith.addf",
            "arith.subf",
            "arith.mulf",
            "arith.divf",
            "arith.remf",
        ];
        for op in &arith_binary_ops {
            let quoted = format!("\"{}\"", op);
            if rest.starts_with(&quoted) {
                let after_op = &rest[quoted.len()..];
                let operands = extract_paren_contents(after_op);
                let result_type = extract_arrow_result_type(rest);
                return format!("{}{} {} : {}", result_prefix, op, operands, result_type);
            }
        }
    }

    // --- arith.cmpi: "arith.cmpi"(%a, %b) <{predicate = N : i64}> : (T, T) -> i1 ---
    // → arith.cmpi pred_name, %a, %b : T
    if rest.starts_with("\"arith.cmpi\"") {
        let after_op = &rest["\"arith.cmpi\"".len()..];
        let operands = extract_paren_contents(after_op);
        let pred_name = if let Some(pred_pos) = rest.find("predicate = ") {
            let after_pred = &rest[pred_pos + "predicate = ".len()..];
            let num_str: String = after_pred
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            match num_str.parse::<u32>().unwrap_or(0) {
                0 => "eq",
                1 => "ne",
                2 => "slt",
                3 => "sle",
                4 => "sgt",
                5 => "sge",
                6 => "ult",
                7 => "ule",
                8 => "ugt",
                9 => "uge",
                _ => "eq",
            }
        } else {
            "eq"
        };
        let input_type = extract_type_signature(rest)
            .trim_start_matches('(')
            .split(',')
            .next()
            .unwrap_or("i32")
            .trim()
            .to_string();
        return format!(
            "{}arith.cmpi {}, {} : {}",
            result_prefix, pred_name, operands, input_type
        );
    }

    // --- arith.cmpf: "arith.cmpf"(%a, %b) <{predicate = N : i64}> : (T, T) -> i1 ---
    if rest.starts_with("\"arith.cmpf\"") {
        let after_op = &rest["\"arith.cmpf\"".len()..];
        let operands = extract_paren_contents(after_op);
        let pred_name = if let Some(pred_pos) = rest.find("predicate = ") {
            let after_pred = &rest[pred_pos + "predicate = ".len()..];
            let num_str: String = after_pred
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            match num_str.parse::<u32>().unwrap_or(0) {
                0 => "false",
                1 => "oeq",
                2 => "ogt",
                3 => "oge",
                4 => "olt",
                5 => "ole",
                6 => "one",
                7 => "ord",
                8 => "ueq",
                9 => "ugt",
                10 => "uge",
                11 => "ult",
                12 => "ule",
                13 => "une",
                14 => "uno",
                15 => "true",
                _ => "oeq",
            }
        } else {
            "oeq"
        };
        let input_type = extract_type_signature(rest)
            .trim_start_matches('(')
            .split(',')
            .next()
            .unwrap_or("f32")
            .trim()
            .to_string();
        return format!(
            "{}arith.cmpf {}, {} : {}",
            result_prefix, pred_name, operands, input_type
        );
    }

    // --- arith unary cast ops: "arith.extui"(%val) : (FROM) -> TO ---
    // Covers: extui, extsi, trunci, sitofp, uitofp, fptosi, fptoui, fpext, fptrunc, bitcast, index_cast
    {
        let arith_cast_ops = [
            "arith.extui",
            "arith.extsi",
            "arith.trunci",
            "arith.sitofp",
            "arith.uitofp",
            "arith.fptosi",
            "arith.fptoui",
            "arith.fpext",
            "arith.fptrunc",
            "arith.bitcast",
            "arith.index_cast",
        ];
        for op in &arith_cast_ops {
            let quoted = format!("\"{}\"", op);
            if rest.starts_with(&quoted) {
                let after_op = &rest[quoted.len()..];
                let operand = extract_paren_contents(after_op);
                let type_sig = extract_type_signature(rest);
                let parts: Vec<&str> = type_sig.splitn(2, "->").collect();
                let from_type = parts
                    .first()
                    .unwrap_or(&"")
                    .trim()
                    .trim_start_matches('(')
                    .trim_end_matches(')')
                    .trim();
                let to_type = parts.get(1).unwrap_or(&"").trim();
                return format!(
                    "{}{} {} : {} to {}",
                    result_prefix, op, operand, from_type, to_type
                );
            }
        }
    }

    // --- arith.select: "arith.select"(%c, %a, %b) : (i1, T, T) -> T ---
    if rest.starts_with("\"arith.select\"") {
        let after_op = &rest["\"arith.select\"".len()..];
        let operands = extract_paren_contents(after_op);
        let result_type = extract_arrow_result_type(rest);
        return format!(
            "{}arith.select {} : {}",
            result_prefix, operands, result_type
        );
    }

    // --- llvm.return: "llvm.return"(%val) : (T) -> () → llvm.return %val : T ---
    // Void: "llvm.return"() : () -> () → llvm.return
    if rest.starts_with("\"llvm.return\"") {
        let after_op = &rest["\"llvm.return\"".len()..];
        let operands = extract_paren_contents(after_op);
        if operands.is_empty() {
            return "llvm.return".to_string();
        }
        // Extract the result type from (T) -> ()
        let type_sig = extract_type_signature(rest);
        // type_sig is "(f32) -> ()" — extract just "f32"
        let ret_type = type_sig
            .split("->")
            .next()
            .unwrap_or("")
            .trim()
            .trim_start_matches('(')
            .trim_end_matches(')')
            .trim();
        return format!("llvm.return {} : {}", operands, ret_type);
    }

    // --- Fallback: strip quotes from op name ---
    // "llvm.X"(args) ... → llvm.X args ...
    if let Some(start) = rest.find('"') {
        let after = &rest[start + 1..];
        if let Some(end) = after.find('"') {
            let op_name = &after[..end];
            let remainder = &after[end + 1..];
            return format!("{}{}{}", result_prefix, op_name, remainder);
        }
    }

    trimmed.to_string()
}

/// Normalize generic-form `"cf.br"` and `"cf.cond_br"` to pretty form.
///
///   `"cf.br"()[^bb1] <{}> : () -> ()` → `cf.br ^bb1`
///   `"cf.br"(%a, %b)[^bb1] <{}> : (i32, i32) -> ()` → `cf.br ^bb1(%a : i32, %b : i32)`
///   `"cf.cond_br"(%c)[^bb1, ^bb2] <{...}> : (i1) -> ()` → `cf.cond_br %c, ^bb1, ^bb2`
///   `"cf.cond_br"(%c, %a, %b)[^bb1, ^bb2] <{...}> : (i1, i32, i32) -> ()` → with args
pub fn normalize_cf_branch(line: &str) -> String {
    if let Some(pos) = line.find("\"cf.cond_br\"") {
        let after = &line[pos + "\"cf.cond_br\"".len()..];
        // Extract operands (%c, %a, ...) from parentheses
        let operands = extract_paren_contents(after);
        let operand_list: Vec<&str> = if operands.is_empty() {
            Vec::new()
        } else {
            operands.split(',').map(|s| s.trim()).collect()
        };
        // Extract successors [^bb1, ^bb2] from brackets
        let bracket_open = after.find('[').unwrap_or(0);
        let bracket_close = after.find(']').unwrap_or(after.len());
        let successors_str = &after[bracket_open + 1..bracket_close];
        let successors: Vec<&str> = successors_str.split(',').map(|s| s.trim()).collect();
        let cond = operand_list.first().copied().unwrap_or("%unknown");
        let bb_true = successors.first().copied().unwrap_or("^bb0");
        let bb_false = successors.get(1).copied().unwrap_or("^bb0");
        return format!("cf.cond_br {}, {}, {}", cond, bb_true, bb_false);
    }
    if let Some(pos) = line.find("\"cf.br\"") {
        let after = &line[pos + "\"cf.br\"".len()..];
        let operands = extract_paren_contents(after);
        let operand_list: Vec<&str> = if operands.is_empty() {
            Vec::new()
        } else {
            operands.split(',').map(|s| s.trim()).collect()
        };
        // Extract successor [^bbN] from brackets
        let bracket_open = after.find('[').unwrap_or(0);
        let bracket_close = after.find(']').unwrap_or(after.len());
        let successor = after[bracket_open + 1..bracket_close].trim();
        // Extract type annotations to pair with operands
        let type_sig = extract_type_signature(after);
        let type_list: Vec<&str> = if type_sig.is_empty() {
            Vec::new()
        } else {
            let input_types = type_sig.split("->").next().unwrap_or("").trim();
            let inner = input_types.trim_start_matches('(').trim_end_matches(')');
            if inner.is_empty() {
                Vec::new()
            } else {
                inner.split(',').map(|s| s.trim()).collect()
            }
        };
        if operand_list.is_empty() {
            return format!("cf.br {}", successor);
        }
        // Build block args: ^bbN(%a : type, %b : type)
        let args: Vec<String> = operand_list
            .iter()
            .enumerate()
            .map(|(i, op)| {
                let ty = type_list.get(i).copied().unwrap_or("i64");
                format!("{} : {}", op, ty)
            })
            .collect();
        return format!("cf.br {}({})", successor, args.join(", "));
    }
    line.to_string()
}

/// Extract content between first pair of parentheses: "(a, b)" → "a, b"
pub fn extract_paren_contents(s: &str) -> String {
    let open = match s.find('(') {
        Some(p) => p,
        None => return String::new(),
    };
    let mut depth = 0;
    let mut close = open;
    for (j, ch) in s[open..].char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    close = open + j;
                    break;
                }
            }
            _ => {}
        }
    }
    if close <= open {
        return String::new();
    } // unbalanced/truncated guard
    s[open + 1..close].trim().to_string()
}

/// Extract content between first pair of square brackets: "[^bb1, ^bb2]" → "^bb1, ^bb2"
pub fn extract_bracket_contents(s: &str) -> String {
    let open = match s.find('[') {
        Some(p) => p,
        None => return String::new(),
    };
    let close = match s[open..].find(']') {
        Some(p) => open + p,
        None => return String::new(),
    };
    if close <= open {
        return String::new();
    } // unbalanced/truncated guard
    s[open + 1..close].trim().to_string()
}

/// Extract a named property from `<{..., name = VALUE, ...}>`.
pub fn extract_property(line: &str, name: &str) -> Option<String> {
    let pattern = format!("{} = ", name);
    let start = line.find(&pattern)? + pattern.len();
    let rest = &line[start..];
    // Value extends until the next comma, or closing }>, or end of line
    // Handle nested <> and () and {}
    let mut depth_angle = 0i32;
    let mut depth_brace = 0i32;
    let mut end = rest.len();
    for (j, ch) in rest.char_indices() {
        match ch {
            '<' => depth_angle += 1,
            '>' if depth_angle > 0 => depth_angle -= 1,
            '{' => depth_brace += 1,
            '}' if depth_brace > 0 => depth_brace -= 1,
            '}' if depth_brace == 0 => {
                end = j;
                break;
            }
            ',' if depth_angle == 0 && depth_brace == 0 => {
                end = j;
                break;
            }
            _ => {}
        }
    }
    let value = rest[..end].trim();
    if value.is_empty() {
        None
    } else {
        Some(value.to_string())
    }
}

/// Extract type signature after the last ` : ` (outside nested structures).
/// e.g., `... : (i32, !llvm.ptr<1>) -> i32` → `(i32, !llvm.ptr<1>) -> i32`
pub fn extract_type_signature(line: &str) -> String {
    // Find the last ` : ` that's outside of <{}>
    let bytes = line.as_bytes();
    let mut depth_angle = 0i32;
    let mut depth_brace = 0i32;
    let mut last_colon = None;
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'<' => depth_angle += 1,
            b'>' if depth_angle > 0 => depth_angle -= 1,
            b'{' => depth_brace += 1,
            b'}' if depth_brace > 0 => depth_brace -= 1,
            b':' if depth_angle == 0 && depth_brace == 0 => {
                // Check for " : " pattern
                if i > 0 && i + 1 < bytes.len() && bytes[i - 1] == b' ' && bytes[i + 1] == b' ' {
                    last_colon = Some(i);
                }
            }
            _ => {}
        }
        i += 1;
    }
    match last_colon {
        Some(pos) => line[pos + 2..].trim().to_string(),
        None => String::new(),
    }
}

/// Extract the result type after `->` in a type signature.
/// e.g., `() -> i32` → `i32`
pub fn extract_arrow_result_type(line: &str) -> String {
    if let Some(arrow) = line.rfind("-> ") {
        line[arrow + 3..].trim().to_string()
    } else {
        "i32".to_string()
    }
}

/// Parse a single `llvm.mlir.global` line into an MlirGlobal.
/// Pretty: `llvm.mlir.global constant @MAGIC {value = 42 : i32, ...}`
/// Generic: `"llvm.mlir.global"() <{global_type = ..., sym_name = "foo", ...}> ({...})`
pub fn parse_global_decl(line: &str) -> Option<MlirGlobal> {
    // Prefer generic `sym_name = "..."`; fall back to pretty `@name`.
    let name = if let Some(n) = extract_generic_sym_name(line) {
        n
    } else {
        let at_pos = line.find('@')?;
        let rest = &line[at_pos + 1..];
        let end = rest.find(|c: char| !c.is_alphanumeric() && c != '_' && c != '.')?;
        rest[..end].to_string()
    };
    if name.is_empty() {
        return None;
    }

    // Extract type from global_type attribute or from value attribute
    let mut ty = String::new();
    let mut value = None;
    let mut elem_type = None;
    let mut count = None;

    // Try to extract value from the line
    if let Some(val_pos) = line.find("value = ") {
        let val_rest = &line[val_pos + 8..];
        // Parse the value — could be "42 : i32" or "dense<[...]> : tensor<NxT>"
        if val_rest.starts_with("dense<") {
            // Dense array: dense<[v1, v2, ...]> : tensor<NxT>
            if let Some(bracket_end) = val_rest.find("]>") {
                let dense_content = &val_rest[6..bracket_end + 1]; // [v1, v2, ...]
                // Find the type after " : "
                let type_marker = &val_rest[bracket_end + 2..];
                if let Some(colon_pos) = type_marker.find(" : ") {
                    let type_str = type_marker[colon_pos + 3..].trim();
                    // Trim trailing }, etc.
                    let type_str = type_str
                        .trim_end_matches(|c: char| c == '}' || c == ',' || c.is_whitespace());
                    ty = type_str.to_string();

                    // Parse array element type and count from "!llvm.array<N x T>" or "tensor<NxT>"
                    if let Some(arr_content) = type_str
                        .strip_prefix("!llvm.array<")
                        .and_then(|s| s.strip_suffix('>'))
                    {
                        if let Some(x_pos) = arr_content.find(" x ") {
                            count = arr_content[..x_pos].trim().parse().ok();
                            elem_type = Some(arr_content[x_pos + 3..].trim().to_string());
                        }
                    } else if let Some(tensor_content) = type_str
                        .strip_prefix("tensor<")
                        .and_then(|s| s.strip_suffix('>'))
                    {
                        // tensor<NxT> format (no spaces around 'x')
                        if let Some(x_pos) = tensor_content.find('x') {
                            count = tensor_content[..x_pos].trim().parse().ok();
                            elem_type = Some(tensor_content[x_pos + 1..].trim().to_string());
                        }
                    }
                }
                value = Some(format!("dense<{}>", dense_content));
            }
        } else {
            // Scalar: "42 : i32" or "3.14159 : f32"
            if let Some(colon_pos) = val_rest.find(" : ") {
                let scalar_val = val_rest[..colon_pos].trim().to_string();
                let type_str = val_rest[colon_pos + 3..].trim();
                let type_str =
                    type_str.trim_end_matches(|c: char| c == '}' || c == ',' || c.is_whitespace());
                ty = type_str.to_string();
                value = Some(scalar_val);
            }
        }
    }

    // If we couldn't determine the type from value, try global_type attribute
    if ty.is_empty() {
        if let Some(gt_pos) = line.find("global_type = ") {
            ty = extract_balanced_type(&line[gt_pos + 14..]);
        }
    }

    // Struct-wrapped arrays (`!llvm.struct<packed (array<N x T>)>` or with a
    // leading ptr) carry no `value =` in generic form; derive count/elem so
    // emit_cpp_globals can emit a sized zero-init array for `&sym` addressof.
    if count.is_none() && (value.is_none() || ty.starts_with("!llvm.struct")) {
        if let Some((c, e)) = struct_array_layout(&ty) {
            count = Some(c);
            elem_type = Some(e);
        }
    }

    Some(MlirGlobal {
        name,
        ty,
        value,
        elem_type,
        count,
    })
}

/// Extract a type token starting at `s`, balancing `<`/`>` so commas inside
/// `!llvm.struct<packed (ptr<1>, array<16 x i8>)>` are not treated as terminators.
fn extract_balanced_type(s: &str) -> String {
    let mut depth = 0i32;
    for (i, c) in s.char_indices() {
        match c {
            '<' => depth += 1,
            '>' => {
                depth -= 1;
                if depth <= 0 {
                    return s[..=i].trim().to_string();
                }
            }
            ',' if depth <= 0 => {
                return s[..i].trim().to_string();
            }
            _ => {}
        }
    }
    s.trim().trim_end_matches('}').trim().to_string()
}

/// Byte count + element type for a (possibly struct-wrapped) global type.
/// Returns `(bytes_as_count, "i8")` when the payload is address-only.
fn struct_array_layout(ty: &str) -> Option<(usize, String)> {
    // Direct array: !llvm.array<N x T>
    if let Some(inner) = ty
        .strip_prefix("!llvm.array<")
        .and_then(|s| s.strip_suffix('>'))
    {
        if let Some(x) = inner.find(" x ") {
            let n: usize = inner[..x].trim().parse().ok()?;
            return Some((n, inner[x + 3..].trim().to_string()));
        }
        return None;
    }
    if !ty.starts_with("!llvm.struct") {
        return None;
    }
    // Find innermost `array<N x T>` and optional `ptr<...>` for layout size.
    let mut bytes = 0usize;
    let mut arr_count = 0usize;
    let mut elem = None;
    let mut rest = ty;
    while let Some(a) = rest.find("array<") {
        let after = &rest[a + 6..];
        let end = after.find('>')?;
        let inner = &after[..end];
        if let Some(x) = inner.find(" x ") {
            let n: usize = inner[..x].trim().parse().ok()?;
            let e = inner[x + 3..].trim().to_string();
            let ebytes = match e.as_str() {
                "i8" | "i1" => 1,
                "i16" | "f16" | "bf16" => 2,
                "i32" | "f32" => 4,
                "f64" | "i64" => 8,
                _ => 1,
            };
            bytes += n * ebytes;
            arr_count = n;
            if elem.is_none() {
                elem = Some(e);
            }
        }
        rest = &after[end + 1..];
    }
    let has_ptr = ty.contains("ptr<");
    if has_ptr {
        // Opaque (ptr, array) payload: expose as a byte blob for `&sym`.
        bytes += 8;
        if bytes == 0 {
            return None;
        }
        return Some((bytes, "i8".to_string()));
    }
    let elem = elem?;
    if arr_count == 0 {
        return None;
    }
    Some((arr_count, elem))
}

/// Keep the old API for backward compatibility (used by tests etc.)
pub fn parse_functions(mlir_text: &str) -> Result<Vec<MlirFunc>, String> {
    Ok(parse_module(mlir_text)?.functions)
}

// ---------------------------------------------------------------------------
// SSA resolution, shared by the pointer-walking emitters
// ---------------------------------------------------------------------------
//
// Eight emitter contexts -- aie, bang, gaudi, gpu, nki, pto, spirv, tpu -- carry
// the same maps under the same names and the same types (`const_map` and
// `ptr_aliases` in all eight, `gep_offsets` and `tile_shapes` in seven,
// `tile_vars` in six, `float_const_map` in four), and each had its own copy of
// the walks below: `resolve_ptr` eight times at 16 lines, `resolve_offset` seven
// times at 20. Two of those "divergences" were a local variable spelled `total`
// instead of `total_offset`.
//
// These take the maps rather than a context, and are wrapped by a one-line method
// on each context. That keeps every call site and every struct layout exactly as
// it was -- the alternative, a trait, costs more accessor boilerplate per target
// than the bodies it removes for the three-line accessors, and buys nothing the
// delegation does not.

/// Follow `ssa` back through pointer aliases to its origin.
///
/// Cycle-guarded with a `seen` set, which is not theoretical: an alias map built
/// from unverified MLIR can close a loop, and without the guard this is an
/// infinite loop inside the compiler rather than a diagnostic.
pub fn resolve_ptr(ptr_aliases: &HashMap<String, String>, ssa: &str) -> String {
    let mut current = ssa.to_string();
    let mut seen = HashSet::new();
    loop {
        if seen.contains(&current) {
            break;
        }
        seen.insert(current.clone());
        if let Some(origin) = ptr_aliases.get(&current) {
            current = origin.clone();
        } else {
            break;
        }
    }
    current
}

/// Sum the GEP offsets along the alias chain from `ssa` to its origin.
///
/// Saturating, so a malformed chain yields a clamped offset rather than wrapping
/// to a small one -- a wrapped offset would address valid memory at the wrong
/// place, which is the silent-wrongness case.
pub fn resolve_offset(
    gep_offsets: &HashMap<String, u32>,
    ptr_aliases: &HashMap<String, String>,
    ssa: &str,
) -> u32 {
    let mut current = ssa.to_string();
    let mut total_offset: u32 = 0;
    let mut seen = HashSet::new();
    loop {
        if seen.contains(&current) {
            break;
        }
        seen.insert(current.clone());
        if let Some(&off) = gep_offsets.get(&current) {
            total_offset = total_offset.saturating_add(off);
        }
        if let Some(origin) = ptr_aliases.get(&current) {
            current = origin.clone();
        } else {
            break;
        }
    }
    total_offset
}

/// An integer constant: the SSA value if it is a known constant, else parsed from
/// the operand text.
pub fn resolve_const(const_map: &HashMap<String, u32>, s: &str) -> u32 {
    if let Some(&n) = const_map.get(s.trim()) {
        return n;
    }
    parse_const_arg(s)
}

/// A float constant, falling back to the integer map so `0` reaches a target as
/// `0.0` rather than as something its parser rejects.
///
/// `mlir_to_linalg` deliberately does NOT use this: it needs `{:.6e}` because
/// linalg's own parser wants scientific notation, so it keeps a local override.
/// That is a real per-target difference, unlike the eight copies above.
pub fn resolve_float(
    float_const_map: &HashMap<String, String>,
    const_map: &HashMap<String, u32>,
    s: &str,
) -> String {
    let s = s.trim();
    if let Some(v) = float_const_map.get(s) {
        return v.clone();
    }
    if let Some(&n) = const_map.get(s) {
        return format!("{n}.0");
    }
    s.to_string()
}

/// Refuse a module that calls a `@__tile_*` intrinsic the target has no arm for.
///
/// One copy, because there were twelve. They were byte-identical once the target
/// name in the error string was normalised away, so a fix to the guard shape had
/// to be made twelve times and in practice was made twice: only `mlir_to_msl` and
/// `mlir_to_spirv` grew the post-classification property check that makes a
/// prefix allowlist safe, and the other ten kept the shape those two were fixed
/// from. Consolidating does not add the missing check — see the caveat below —
/// but it makes adding it an edit rather than a campaign.
///
/// `known` stays per-target, because it is the only thing that legitimately
/// differs: the allowlists run from 7 stems to 360 and describe what each target
/// has actually implemented.
///
/// **This gate matches by PREFIX and that is not sufficient on its own.** A stem
/// such as `__tile_cast_` admits every longer spelling, including ones no
/// dispatch arm names, so passing here means "not obviously unknown", never "has
/// an arm". A target that dispatches by exact match needs a second, structural
/// check after classification to catch the difference; without one, an
/// unimplemented intrinsic is accepted here and then silently dropped.
pub fn reject_unknown_intrinsics(
    mlir_text: &str,
    target: &str,
    known: &[&str],
) -> Result<(), String> {
    for line in mlir_text.lines() {
        if line.trim_start().starts_with("//") {
            continue;
        }
        let mut rest = line;
        while let Some(pos) = rest.find("@__tile_") {
            let after = &rest[pos + 1..];
            let name: String = after
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if !known.iter().any(|k| name.starts_with(k)) {
                return Err(format!(
                    "{target} has no arm for `{name}`. Emitting it would drop the \
                     operation silently: the dispatch skips calls it does not match, \
                     so the kernel would compile, run, and quietly omit this step."
                ));
            }
            rest = &after[name.len()..];
        }
    }
    Ok(())
}

pub fn extract_func_name(line: &str) -> Option<String> {
    let at_pos = line.find("@")?;
    let rest = &line[at_pos + 1..];
    let end = rest.find('(')?;
    Some(rest[..end].to_string())
}

pub fn extract_func_args(line: &str) -> Vec<FuncArg> {
    let mut args = Vec::new();

    // Find the argument list between the first ( and matching )
    let open = match line.find('(') {
        Some(p) => p,
        None => return args,
    };

    let mut depth = 0;
    let mut close = open;
    for (j, ch) in line[open..].char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    close = open + j;
                    break;
                }
            }
            _ => {}
        }
    }

    // Unbalanced / truncated input: no matching ')' was found (close still <= open).
    // Bail gracefully instead of slicing line[open+1..close] with start > end (panic).
    if close <= open {
        return args;
    }

    let arg_str = &line[open + 1..close];
    if arg_str.trim().is_empty() {
        return args;
    }

    // Split by comma, handling nested <> and ()
    let mut arg_parts = Vec::new();
    let mut current = String::new();
    let mut angle_depth = 0;
    let mut paren_depth = 0;
    for ch in arg_str.chars() {
        match ch {
            '<' => {
                angle_depth += 1;
                current.push(ch);
            }
            '>' => {
                angle_depth -= 1;
                current.push(ch);
            }
            '(' => {
                paren_depth += 1;
                current.push(ch);
            }
            ')' => {
                paren_depth -= 1;
                current.push(ch);
            }
            ',' if angle_depth == 0 && paren_depth == 0 => {
                arg_parts.push(current.trim().to_string());
                current.clear();
            }
            _ => current.push(ch),
        }
    }
    if !current.trim().is_empty() {
        arg_parts.push(current.trim().to_string());
    }

    for part in &arg_parts {
        // Format: %argN: TYPE
        let colon = match part.find(':') {
            Some(p) => p,
            None => continue,
        };
        let name = part[..colon].trim().to_string();
        let ty = part[colon + 1..].trim().to_string();
        let is_gm = ty.contains("ptr<1>");
        args.push(FuncArg { name, ty, is_gm });
    }

    args
}

// ---- additional helpers used by mlir_to_pto / mlir_to_msl (ds4-msl-pipeline) ----

pub fn extract_result_ssa(line: &str) -> Option<String> {
    let line = line.trim();
    if line.starts_with('%') {
        let end = line.find(" = ")?;
        Some(line[..end].trim().to_string())
    } else {
        None
    }
}

pub fn extract_call_args(line: &str) -> Option<Vec<String>> {
    let open = line.find('(')?;
    let args_section = &line[open + 1..];
    let close = args_section.find(") :")?;
    let inner = &args_section[..close];
    Some(inner.split(',').map(|s| s.trim().to_string()).collect())
}

pub fn parse_const_arg(s: &str) -> u32 {
    let s = s.trim();
    if let Ok(n) = s.parse::<u32>() {
        return n;
    }
    // `%c256` is MLIR's convention for a NAMED constant whose value is in the name, so
    // reading it is sound. A bare `%42` is SSA value number 42 and carries no value at all
    // -- reading it as an extent invents a plausible nonzero from a name, which is the
    // stand-in hazard: `validate_tile_shape`'s C0 cannot catch it because 42 is positive,
    // and the emitted kernel computes a wrong-sized corner of the right answer.
    // Unresolvable means 0.
    let digits_start = if let Some(rest) = s.strip_prefix("%c") {
        rest
    } else if s.starts_with('%') {
        ""
    } else {
        s
    };
    let digit_str: String = digits_start
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    if !digit_str.is_empty() {
        return digit_str.parse().unwrap_or(0);
    }
    0
}

pub fn is_builtin_helper(name: &str) -> bool {
    matches!(
        name,
        "get_block_idx"
            | "get_block_num"
            | "get_sub_block_idx"
            | "get_sub_block_num"
            | "__rust_eh_personality"
    ) || name.starts_with("__tile_")
        || name.starts_with("__rust_")
}

/// Guess a GM arg's element type from its NAME alone, as a last resort.
///
/// Lives here, with `infer_arg_dtype_from_body` below it, rather than in
/// `mlir_to_pto`: nine of the open emitters call one or both, and importing them
/// from the PTO file made each of those emitters depend on the CLOSED one. On main
/// that coupling is why `tile_codegen --features emitters` would not compile at
/// all; here it compiled and was simply wrong about what "open" meant. Neither
/// function is PTO-specific — `mlir_to_gpu` needs the body-scanning one for the
/// same reason PTO does, which is the evidence. `mlir_to_pto` re-exports both for
/// its own call sites. `mlir_to_msl` already took this route and says so at its
/// import: it "must not import from another backend".
pub fn infer_dtype_from_name(name: &str) -> &'static str {
    if name.contains("f16") || name.contains("half") {
        "f16"
    } else {
        "f32"
    }
}

/// Infer a GM arg's dtype by scanning the function body for the first
/// `__tile_load_*` / `__tile_store_*` call that uses it as a
/// pointer operand. Returns None if no such call is found; the caller
/// then falls back to `infer_dtype_from_name`.
///
/// This matters for f16 kernels whose Rust arg names don't contain
/// "f16" (e.g. `fn matmul(a: *const f16, b: *const f16, out: *mut f16)`):
/// body-scanning correctly sets `!pto.ptr<f16>` so ptoas generates
/// `__gm__ half*` parameters consistent with the `GlobalTensor<half, ...>`
/// views derived from them.
pub fn infer_arg_dtype_from_body(
    arg_name: &str,
    body_lines: &[String],
) -> Option<&'static str> {
    // Patterns that indicate this arg is used as a typed GM pointer.
    // The LLVM IR shape is: `llvm.call @__tile_load_fNN(%argK, ...)` or
    // `llvm.call @__tile_store_fNN(%argK, ...)`. We do a substring
    // check: if the line mentions both the arg name and a typed tile_load/
    // tile_store call, use the dtype from the callee name.
    for line in body_lines {
        if !line.contains(arg_name) {
            continue;
        }
        // Partition intrinsics are the OTHER way an arg reaches GM: a gemm
        // operand is partitioned, never loaded directly, so without this an
        // f16 gemm emitted `!pto.ptr<f32>` in the signature while
        // make_tensor_view used `tensor_view<?x?xf16>`. ptoas rejects that
        // outright: "use of value '%arg0' expects different type than prior
        // uses: '!pto.ptr<f16, gm>' vs '!pto.ptr<f32, gm>'".
        if line.contains("__tile_partition") {
            if line.contains("_f16") {
                return Some("f16");
            }
            if line.contains("_bf16") {
                return Some("bf16");
            }
            if line.contains("_f32") {
                return Some("f32");
            }
        }
        // Check store (GM arg is 1st pointer param, written).
        if line.contains("__tile_store_f16") {
            return Some("f16");
        }
        if line.contains("__tile_store_f32") {
            return Some("f32");
        }
        if line.contains("__tile_store_bf16") {
            return Some("bf16");
        }
        // Check load (GM arg is 1st pointer param, read).
        if line.contains("__tile_load_f16") {
            return Some("f16");
        }
        if line.contains("__tile_load_f32") {
            return Some("f32");
        }
        if line.contains("__tile_load_bf16") {
            return Some("bf16");
        }
        if line.contains("__tile_load_i8") {
            return Some("i8");
        }
        if line.contains("__tile_store_i8") {
            return Some("i8");
        }
        // Q8_0 routed matvec: __tile_mul_mv_id_q8_0_f32(src0s, src1, ids, dst, ...).
        // The real lowering loads src0s (arg position 0) as int8 weights (si8);
        // the other pointer args (x, scale, dst) are f32. ptoas type-checks the
        // func-signature ptr dtype against the make_tensor_view dtype, so the
        // weights arg MUST be declared si8 here.
        if line.contains("__tile_mul_mv_id_q8_0") || line.contains("__tile_mul_mv_id_iq2_xxs") {
            // Real lowering loads the weights arg (position 0) as int8 (si8);
            // the Q8_0 weights are native int8, the IQ2_XXS weights are the
            // host-precomputed int8 expansion. Other pointer args are f32.
            if let Some(args) = extract_call_args(line) {
                if args.first().map(|s| s.trim()) == Some(arg_name) {
                    return Some("si8");
                }
            }
            return Some("f32");
        }
        // int8 matmul has a scale arg at position 3. CANN 8.5 ptoas requires
        // the scale tile to be dtype=ui64 (TMovToFb needs uint64_t DstType),
        // so we emit `!pto.ptr<ui64>` here even though the Rust source declares
        // the pointer as `*const f32`. Host-side repacks f32 → u64 FB words
        // before launch (see pack_scale_f32_to_u64).
        if line.contains("__tile_matmul_i8_acc_i32_dequant_f16") {
            let args = match extract_call_args(line) {
                Some(a) => a,
                None => continue,
            };
            let scale_arg = args.get(3).map(|s| s.trim()).unwrap_or("");
            if scale_arg == arg_name {
                return Some("ui64");
            }
        }
        // add_rms_norm_rows carries FIVE GM pointers (x, residual, weight, out,
        // residual_out) that are ALL the intrinsic's I/O dtype, which is in the
        // callee name. Without this they default to f32 and ptoas rejects the
        // module for a bf16 kernel ("expects different type than prior uses").
        if let Some(rest) = line.split("__tile_add_rms_norm_rows_").nth(1) {
            let dt = rest.split('(').next().unwrap_or("").trim();
            return match dt {
                "bf16" => Some("bf16"),
                "f16" => Some("f16"),
                "f32" => Some("f32"),
                _ => None,
            };
        }
        // swiglu_quant_rows carries FOUR differently-typed GM pointers in one
        // call — (gate f16, up f16, y i8, scale f32) — so the dtype depends on
        // WHICH position the arg occupies, not merely on the callee name. A
        // name-only match here would type every pointer the same and ptoas
        // rejects the result ("expects different type than prior uses").
        if line.contains("__tile_swiglu_quant_rows") {
            let args = match extract_call_args(line) {
                Some(a) => a,
                None => continue,
            };
            let at = |i: usize| args.get(i).map(|s| s.trim()).unwrap_or("");
            if at(0) == arg_name || at(1) == arg_name {
                return Some("f16");
            }
            if at(2) == arg_name {
                return Some("i8");
            }
            if at(3) == arg_name {
                return Some("f32");
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Unit tests for the shared, std-only MLIR parser helpers. These run wherever
// this file is `#[path]`-included (tile_spec, codegen_tests, the mlir_to_*_tests
// crates) — NO LLVM toolchain needed — and pin the open emit surface's parser
// contract.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod parse_tests {
    use super::*;

    #[test]
    fn brace_and_paren_counts() {
        assert_eq!(count_open_braces("a { b { c"), 2);
        assert_eq!(count_close_braces("} } x }"), 3);
        assert_eq!(count_open_parens("f((x)"), 2);
        assert_eq!(count_close_parens("f((x))"), 2);
        assert_eq!(count_open_braces(""), 0);
    }

    #[test]
    fn scalar_type_to_cpp_primitives() {
        assert_eq!(mlir_scalar_type_to_cpp("i8"), "int8_t");
        assert_eq!(mlir_scalar_type_to_cpp("i16"), "int16_t");
        assert_eq!(mlir_scalar_type_to_cpp("i32"), "int32_t");
        assert_eq!(mlir_scalar_type_to_cpp("i64"), "int64_t");
        assert_eq!(mlir_scalar_type_to_cpp("f32"), "float");
        assert_eq!(mlir_scalar_type_to_cpp("f16"), "half");
    }

    #[test]
    fn scalar_type_to_cpp_arrays_and_tensors_and_default() {
        assert_eq!(mlir_scalar_type_to_cpp("!llvm.array<5 x f32>"), "float");
        assert_eq!(mlir_scalar_type_to_cpp("tensor<3xi16>"), "int16_t");
        // Unknown type falls back to int32_t.
        assert_eq!(mlir_scalar_type_to_cpp("something_else"), "int32_t");
    }

    #[test]
    fn extract_generic_sym_name_present_and_absent() {
        let line = "llvm.mlir.global constant @x() {sym_name = \"foo\", other = 1}";
        assert_eq!(extract_generic_sym_name(line).as_deref(), Some("foo"));
        assert_eq!(extract_generic_sym_name("no marker here"), None);
    }

    #[test]
    fn extract_generic_func_type_args_counts_gm_ptrs() {
        let line = "x {function_type = !llvm.func<void (ptr<1>, ptr<1>, i32)>} y";
        let args = extract_generic_func_type_args(line);
        assert_eq!(args.len(), 3);
        assert_eq!(args[0].name, "%arg0");
        assert!(args[0].is_gm); // ptr<1>
        assert!(args[1].is_gm);
        assert!(!args[2].is_gm); // i32
        // No marker -> empty.
        assert!(extract_generic_func_type_args("nope").is_empty());
        // Empty arg list -> empty.
        assert!(extract_generic_func_type_args("function_type = !llvm.func<void ()>").is_empty());
    }

    #[test]
    fn extract_func_name_and_args() {
        let line = "llvm.func @kernel(%arg0: !llvm.ptr<1>, %arg1: i32) attributes {}";
        assert_eq!(extract_func_name(line).as_deref(), Some("kernel"));
        let args = extract_func_args(line);
        assert_eq!(args.len(), 2);
        assert_eq!(args[0].name, "%arg0");
        assert!(args[0].is_gm);
        assert_eq!(args[1].ty, "i32");
        assert!(!args[1].is_gm);
        // No '@' -> None.
        assert_eq!(extract_func_name("plain line"), None);
        // No parens -> empty args.
        assert!(extract_func_args("@kernel no parens").is_empty());
    }

    #[test]
    fn extract_bb_args_named() {
        let args = extract_bb_args("^bb0(%a: !llvm.ptr<1>, %b: f32):");
        assert_eq!(args.len(), 2);
        assert_eq!(args[0].name, "%a");
        assert!(args[0].is_gm);
        assert_eq!(args[1].ty, "f32");
        assert!(extract_bb_args("^bb0:").is_empty());
    }

    #[test]
    fn extract_paren_and_bracket_contents() {
        assert_eq!(extract_paren_contents("call(a, (b), c) : x"), "a, (b), c");
        assert_eq!(extract_paren_contents("no parens"), "");
        assert_eq!(extract_bracket_contents("br [^bb1, ^bb2]"), "^bb1, ^bb2");
        assert_eq!(extract_bracket_contents("no brackets"), "");
    }

    #[test]
    fn extract_property_handles_nesting() {
        let line = "op <{value = 42 : i32, other = dense<[1, 2]>}>";
        assert_eq!(extract_property(line, "value").as_deref(), Some("42 : i32"));
        assert_eq!(extract_property(line, "missing"), None);
    }

    #[test]
    fn extract_type_signature_after_last_colon() {
        let line = "%r = llvm.call @f(%a) : (i32, !llvm.ptr<1>) -> i32";
        assert_eq!(extract_type_signature(line), "(i32, !llvm.ptr<1>) -> i32");
        assert_eq!(extract_type_signature("no colon here"), "");
    }

    #[test]
    fn extract_arrow_result_type_present_and_default() {
        assert_eq!(extract_arrow_result_type("(i32) -> f32"), "f32");
        // No arrow -> defaults to i32.
        assert_eq!(extract_arrow_result_type("(i32)"), "i32");
    }

    #[test]
    fn extract_result_ssa_and_call_args() {
        assert_eq!(
            extract_result_ssa("%t1 = llvm.call @f(%a) : (i32) -> i32").as_deref(),
            Some("%t1")
        );
        assert_eq!(extract_result_ssa("llvm.return"), None);
        let args = extract_call_args("llvm.call @f(%a, %b, %c) : (i32, i32, i32) -> i32").unwrap();
        assert_eq!(args, vec!["%a", "%b", "%c"]);
        assert_eq!(extract_call_args("no parens"), None);
    }

    #[test]
    fn parse_const_arg_decimal_and_named() {
        assert_eq!(parse_const_arg("1024"), 1024);
        assert_eq!(parse_const_arg("%c256"), 256);
        // `%42` is SSA value 42, not the number 42 -- see the invariant test below.
        assert_eq!(parse_const_arg("%42"), 0);
        assert_eq!(parse_const_arg("%arg0"), 0); // no leading digits
        assert_eq!(parse_const_arg("   7  "), 7);
    }

    /// INVARIANT (codegen rule 1): an extent that cannot be resolved reads as 0, never as a
    /// plausible nonzero invented from a name.
    ///
    /// `%c256` is MLIR's convention for a named constant carrying its value, so reading it is
    /// sound. Everything else that merely *contains* digits is not a value: `%42` is SSA
    /// number 42, `%arg3` is argument 3. Inventing an extent from those is the stand-in
    /// hazard -- a neighbouring fork seeded unresolved matmul extents with block constants
    /// and emitted a valid module computing one fixed corner of the output, its own budget
    /// guard passing because the stand-in tile genuinely fit. A zero is caught by C0; a
    /// plausible positive is not.
    ///
    /// A property test over FORMS rather than a list of cases, because the failure mode is a
    /// form nobody thought to enumerate.
    #[test]
    fn an_unresolvable_extent_reads_as_zero_and_never_as_an_invented_value() {
        // Sound: the value is genuinely in the token.
        assert_eq!(parse_const_arg("1024"), 1024);
        assert_eq!(parse_const_arg("%c256"), 256);
        assert_eq!(parse_const_arg("  7 "), 7);

        // Not values. Each carries digits that mean something else.
        for form in ["%42", "%0", "%arg3", "%kk", "%nn", "%12_tmp", "%v7"] {
            assert_eq!(
                parse_const_arg(form), 0,
                "{form} carries no extent; reading digits out of it invents a stand-in that \
                 C0 cannot catch because it is positive"
            );
        }
    }

    #[test]
    fn is_builtin_helper_matches() {
        assert!(is_builtin_helper("get_block_idx"));
        assert!(is_builtin_helper("get_sub_block_num"));
        assert!(is_builtin_helper("__tile_load_f32"));
        assert!(is_builtin_helper("__rust_alloc"));
        assert!(!is_builtin_helper("my_kernel"));
    }

    #[test]
    fn parse_global_decl_scalar_and_dense() {
        let scalar =
            parse_global_decl("llvm.mlir.global constant @MAGIC() {value = 42 : i32} : i32")
                .expect("scalar global");
        assert_eq!(scalar.name, "MAGIC");
        assert_eq!(scalar.value.as_deref(), Some("42"));

        let dense = parse_global_decl(
            "llvm.mlir.global constant @TABLE() {value = dense<[10, 20, 30]> : tensor<3xi32>}",
        )
        .expect("dense global");
        assert_eq!(dense.name, "TABLE");
        assert_eq!(dense.count, Some(3));
        // No '@' -> None.
        assert!(parse_global_decl("not a global").is_none());
    }

    #[test]
    fn parse_module_extracts_function_and_entry() {
        let mlir = r#"
module {
  llvm.func @softmax_1d(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %t0 = llvm.call @__tile_load_f32(%arg0, %c1, %c1) : (!llvm.ptr<1>, i32, i32) -> i32
    llvm.return
  }
}
"#;
        let m = parse_module(mlir).expect("parse ok");
        assert_eq!(m.functions.len(), 1);
        let f = &m.functions[0];
        assert_eq!(f.name, "softmax_1d");
        assert!(f.is_entry, "hacc.entry must mark the function as an entry");
        assert_eq!(f.args.len(), 2);
        assert!(f.args[0].is_gm);
        assert!(f.k_unroll.is_none(), "absent schedule must stay None");
    }

    #[test]
    fn parse_k_unroll_from_pretty_attributes() {
        let mlir = r#"
module {
  llvm.func @k(%arg0: !llvm.ptr<1>) attributes {hacc.entry, tile.k_unroll = 8 : i32} {
    llvm.return
  }
}
"#;
        let m = parse_module(mlir).expect("parse ok");
        assert_eq!(m.functions[0].k_unroll, Some(8));
    }

    #[test]
    fn parse_k_unroll_from_generic_closing_line() {
        let mlir = r#"
module {
  "llvm.func"() <{sym_name = "k", function_type = !llvm.func<void ()>}> ({
  ^bb0:
    llvm.return
  }) {hacc.entry, tile.k_unroll = 32 : i32} : () -> ()
}
"#;
        let m = parse_module(mlir).expect("parse ok");
        assert_eq!(m.functions.len(), 1);
        assert_eq!(m.functions[0].k_unroll, Some(32));
    }

    #[test]
    fn parse_k_unroll_attr_rejects_non_numeric() {
        assert_eq!(parse_k_unroll_attr("tile.k_unroll = abc"), None);
        assert_eq!(parse_k_unroll_attr("tile.k_unroll ="), None);
        assert_eq!(parse_k_unroll_attr("no attr"), None);
        assert_eq!(
            parse_k_unroll_attr("tile.k_unroll = 16 : i32"),
            Some(16)
        );
    }

    #[test]
    fn parse_module_extracts_global() {
        let mlir = r#"
module {
  llvm.mlir.global constant @TBL() {value = dense<[1, 2, 3, 4]> : tensor<4xi32>}
  llvm.func @k(%arg0: !llvm.ptr<1>) attributes {hacc.entry} {
    llvm.return
  }
}
"#;
        let m = parse_module(mlir).expect("parse ok");
        assert_eq!(m.functions.len(), 1);
        assert_eq!(m.globals.len(), 1);
        assert_eq!(m.globals[0].name, "TBL");
    }

    // ── normalize_generic_to_pretty: generic-print-form → pretty-form coverage ──
    // The generic MLIR print form (`"op.name"(args) <{attrs}> : types`) is rarely
    // emitted but must round-trip into the pretty form the rest of the parser
    // consumes. Each case below feeds one generic-form line per op family and
    // asserts the exact pretty-form normalization (OSS ratchet — pure string fn).
    #[test]
    fn test_normalize_generic_op_families() {
        let cases: &[(&str, &str)] = &[
            // call
            (
                "%0 = \"llvm.call\"(%a, %b) <{callee = @foo}> : (i32, i32) -> i32",
                "llvm.call @foo(%a, %b) : (i32, i32) -> i32",
            ),
            // constant
            (
                "%1 = \"llvm.mlir.constant\"() <{value = 7 : i32}> : () -> i32",
                "llvm.mlir.constant(7 : i32) : i32",
            ),
            // load
            (
                "%2 = \"llvm.load\"(%p) <{}> : (!llvm.ptr) -> f32",
                "llvm.load %p : !llvm.ptr -> f32",
            ),
            // store
            (
                "\"llvm.store\"(%v, %p) <{}> : (f32, !llvm.ptr) -> ()",
                "llvm.store %v, %p",
            ),
            // bitcast
            (
                "%3 = \"llvm.bitcast\"(%x) <{}> : (i32) -> f32",
                "llvm.bitcast %x : i32 to f32",
            ),
            // alloca
            (
                "%4 = \"llvm.alloca\"(%n) <{elem_type = f32}> : (i64) -> !llvm.ptr",
                "llvm.alloca %n x f32",
            ),
            // getelementptr
            (
                "%5 = \"llvm.getelementptr\"(%p, %i) <{elem_type = f32}> : (!llvm.ptr, i64) -> !llvm.ptr",
                "llvm.getelementptr %p[%i]",
            ),
            // extractvalue
            (
                "%6 = \"llvm.extractvalue\"(%agg) <{position = array<i64: 1>}> : (!llvm.struct) -> i32",
                "llvm.extractvalue %agg[1]",
            ),
            // insertvalue
            (
                "%7 = \"llvm.insertvalue\"(%v, %agg) <{position = array<i64: 0>}> : (i32, !llvm.struct) -> !llvm.struct",
                "llvm.insertvalue %v into %agg[0]",
            ),
            // unconditional br (no operands)
            ("\"llvm.br\"()[^bb1] <{}> : () -> ()", "llvm.br ^bb1"),
            // unary cast (one of the table)
            (
                "%8 = \"llvm.sext\"(%x) <{}> : (i16) -> i32",
                "llvm.sext %x : i16 to i32",
            ),
            // fneg
            (
                "%9 = \"llvm.fneg\"(%x) <{}> : (f32) -> f32",
                "llvm.fneg %x : f32",
            ),
            // arith.negf
            (
                "%10 = \"arith.negf\"(%x) <{}> : (f32) -> f32",
                "arith.negf %x : f32",
            ),
            // binary op (table)
            (
                "%11 = \"llvm.add\"(%a, %b) <{}> : (i32, i32) -> i32",
                "llvm.add %a, %b : i32",
            ),
            // icmp with predicate 4 (sgt)
            (
                "%12 = \"llvm.icmp\"(%a, %b) <{predicate = 4 : i64}> : (i32, i32) -> i1",
                "llvm.icmp \"sgt\" %a, %b : i32",
            ),
            // select
            (
                "%13 = \"llvm.select\"(%c, %a, %b) <{}> : (i1, i32, i32) -> i32",
                "llvm.select %c, %a, %b : i32",
            ),
            // arith.constant
            (
                "%14 = \"arith.constant\"() <{value = 3 : index}> : () -> index",
                "arith.constant(3 : index) : index",
            ),
            // arith binary (table)
            (
                "%15 = \"arith.addf\"(%a, %b) <{}> : (f32, f32) -> f32",
                "arith.addf %a, %b : f32",
            ),
            // return with value
            (
                "\"llvm.return\"(%v) <{}> : (i32) -> ()",
                "llvm.return %v : i32",
            ),
            // void return
            ("\"llvm.return\"() <{}> : () -> ()", "llvm.return"),
        ];
        for (input, expect) in cases {
            let got = normalize_generic_to_pretty(input);
            assert!(
                got.contains(expect),
                "normalize_generic_to_pretty({input:?})\n  got:    {got:?}\n  expect substring: {expect:?}"
            );
            // determinism
            assert_eq!(
                got,
                normalize_generic_to_pretty(input),
                "non-deterministic for {input:?}"
            );
        }
    }

    #[test]
    fn test_normalize_generic_passthrough_and_fallback() {
        // A pretty-form line (no quoted generic op) passes through unchanged.
        let pretty = "%0 = llvm.add %a, %b : i32";
        assert_eq!(normalize_generic_to_pretty(pretty), pretty);
        // Unknown quoted op hits the quote-stripping fallback.
        let unknown = "%0 = \"llvm.frobnicate\"(%a) : (i32) -> i32";
        let got = normalize_generic_to_pretty(unknown);
        assert!(
            got.contains("llvm.frobnicate"),
            "fallback should strip quotes: {got}"
        );
        assert!(
            !got.contains("\"llvm.frobnicate\""),
            "fallback should remove quotes: {got}"
        );
    }

    #[test]
    fn test_normalize_cf_branch_forms() {
        // Unconditional cf.br with no args.
        let br = "\"cf.br\"()[^bb1] <{}> : () -> ()";
        assert_eq!(normalize_cf_branch(br), "cf.br ^bb1");
        // cf.br with block args carries the operand:type pairs.
        let br_args = "\"cf.br\"(%a, %b)[^bb2] <{}> : (i32, i64) -> ()";
        let got = normalize_cf_branch(br_args);
        assert!(got.contains("cf.br ^bb2("), "missing block args: {got}");
        assert!(
            got.contains("%a : i32") && got.contains("%b : i64"),
            "missing typed args: {got}"
        );
        // cf.cond_br emits cond + two successors.
        let cbr = "\"cf.cond_br\"(%c)[^bbt, ^bbf] <{}> : (i1) -> ()";
        assert_eq!(normalize_cf_branch(cbr), "cf.cond_br %c, ^bbt, ^bbf");
        // routed through the dispatcher in normalize_generic_to_pretty too.
        assert_eq!(
            normalize_generic_to_pretty(cbr),
            "cf.cond_br %c, ^bbt, ^bbf"
        );
    }

    #[test]
    fn test_normalize_generic_control_flow_and_intr() {
        // llvm.br carrying block args → typed arg list.
        let br_args = "\"llvm.br\"(%a, %b)[^bb1] <{}> : (i32, i64) -> ()";
        let got = normalize_generic_to_pretty(br_args);
        assert!(got.contains("llvm.br ^bb1("), "br with args: {got}");
        assert!(
            got.contains("i32") && got.contains("i64"),
            "br arg types: {got}"
        );

        // llvm.cond_br with no extra args → cond + two successors.
        let cbr = "\"llvm.cond_br\"(%c)[^bbt, ^bbf] <{}> : (i1) -> ()";
        assert_eq!(
            normalize_generic_to_pretty(cbr),
            "llvm.cond_br %c, ^bbt, ^bbf"
        );

        // llvm.cond_br with block args after the condition.
        let cbr_args = "\"llvm.cond_br\"(%c, %x)[^bbt, ^bbf] <{}> : (i1, i32) -> ()";
        let got = normalize_generic_to_pretty(cbr_args);
        assert!(
            got.contains("llvm.cond_br %c, ^bbt("),
            "cond_br args: {got}"
        );

        // llvm.intr.* passthrough.
        let intr = "%0 = \"llvm.intr.smax\"(%a, %b) <{}> : (i32, i32) -> i32";
        let got = normalize_generic_to_pretty(intr);
        assert!(got.contains("llvm.intr.smax"), "intr: {got}");
    }

    #[test]
    fn test_normalize_generic_icmp_predicates() {
        // Every signed/unsigned integer compare predicate maps to its name.
        let cases: &[(u32, &str)] = &[
            (0, "eq"),
            (1, "ne"),
            (2, "slt"),
            (3, "sle"),
            (5, "sge"),
            (6, "ult"),
            (7, "ule"),
            (8, "ugt"),
            (9, "uge"),
        ];
        for (n, name) in cases {
            let line = format!(
                "%0 = \"llvm.icmp\"(%a, %b) <{{predicate = {n} : i64}}> : (i32, i32) -> i1"
            );
            let got = normalize_generic_to_pretty(&line);
            assert!(
                got.contains(&format!("llvm.icmp \"{name}\"")),
                "icmp pred {n} → {name}: {got}"
            );
        }
        // Out-of-range predicate falls back to "eq".
        let oob = "%0 = \"llvm.icmp\"(%a, %b) <{predicate = 99 : i64}> : (i32, i32) -> i1";
        assert!(
            normalize_generic_to_pretty(oob).contains("\"eq\""),
            "icmp oob → eq"
        );
    }

    #[test]
    fn test_normalize_generic_arith_cmp() {
        // arith.cmpi with a signed predicate.
        let cmpi = "%0 = \"arith.cmpi\"(%a, %b) <{predicate = 4 : i64}> : (i32, i32) -> i1";
        let got = normalize_generic_to_pretty(cmpi);
        assert!(got.contains("arith.cmpi sgt, %a, %b : i32"), "cmpi: {got}");

        // arith.cmpf with an ordered predicate (olt = 4).
        let cmpf = "%1 = \"arith.cmpf\"(%a, %b) <{predicate = 4 : i64}> : (f32, f32) -> i1";
        let got = normalize_generic_to_pretty(cmpf);
        assert!(got.contains("arith.cmpf olt, %a, %b : f32"), "cmpf: {got}");

        // arith.cmpf "true"/"false" sentinel predicates.
        let cmpf_true = "%2 = \"arith.cmpf\"(%a, %b) <{predicate = 15 : i64}> : (f32, f32) -> i1";
        assert!(
            normalize_generic_to_pretty(cmpf_true).contains("arith.cmpf true"),
            "cmpf pred 15 → true"
        );
    }

    #[test]
    fn test_normalize_generic_arith_binary_table() {
        // Each arith binary op in the table normalizes to its pretty infix form.
        let ops = [
            "arith.addi",
            "arith.subi",
            "arith.muli",
            "arith.divsi",
            "arith.remsi",
            "arith.andi",
            "arith.ori",
            "arith.xori",
            "arith.shli",
            "arith.shrsi",
            "arith.subf",
            "arith.mulf",
            "arith.divf",
        ];
        for op in ops {
            let line = format!("%0 = \"{op}\"(%a, %b) <{{}}> : (i32, i32) -> i32");
            let got = normalize_generic_to_pretty(&line);
            assert!(
                got.contains(&format!("{op} %a, %b")),
                "{op} normalization: {got}"
            );
        }
    }

    #[test]
    fn parse_functions_backward_compat_wrapper() {
        // `parse_functions` is the pub backward-compat shim over parse_module;
        // it returns just the function list. Drive a two-arg kernel and assert
        // the name + arg count round-trip.
        let mlir = r#"
module {
  llvm.func @kern(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    llvm.return
  }
}
"#;
        let funcs = parse_functions(mlir).expect("parse_functions should succeed");
        assert_eq!(funcs.len(), 1);
        assert_eq!(funcs[0].name, "kern");
        assert_eq!(funcs[0].args.len(), 2);
    }
}

// ---------------------------------------------------------------------------
// K-loop unroll (P6): a declared knob, not a hand-inlined constant
// ---------------------------------------------------------------------------

/// Default unroll factor for reduction (K) loops in matmul-family kernels.
///
/// Ported from a ranked receipt on Apple M5: upstream ml-explore/mlx PR #3843
/// measured **-12% kernel time at head_dim 128** from unroll-by-4 on the steel
/// attention K-loop, and the same pragma transplanted to the hottest decode
/// matvec promoted independently on the MLXFast leaderboard (submissions
/// `1077625`, `270ec98`, `a67cebb`, `89ad8a2`). Unrolling lets the compiler
/// interleave the next K-tile's loads with the running multiply-accumulate
/// chain instead of stalling on each load.
///
/// **The factor never changes results.** `emit_unrolled_k_accumulation`
/// accumulates into a single accumulator in strictly increasing `k`, and the
/// scalar tail continues that same order, so every factor produces the same
/// summation sequence — only the loop boundary moves. Nothing is reassociated,
/// which is what makes this safe under exact-output gates.
///
/// The optimum is target-specific (it trades I-cache and register pressure
/// against load/ALU overlap), so backends may override it; 4 is the receipted
/// default on Apple GPUs.
pub const DEFAULT_K_UNROLL: usize = 4;

/// Emit a fully-unrolled-by-`factor` reduction loop plus its scalar tail.
///
/// `term(k_expr)` renders one multiply-accumulate term given an index
/// expression. `acc`/`k` name the accumulator and induction variables, `bound`
/// is the loop limit expression, and `indent` prefixes every emitted line.
///
/// Emits, for factor 4:
/// ```text
/// int k = 0;
/// for (; k + 3 < K; k += 4) { acc += t(k); acc += t(k+1); ... }
/// for (; k < K; k++) { acc += t(k); }
/// ```
/// A factor of 0 or 1 degenerates to the plain scalar loop.
pub fn emit_unrolled_k_accumulation<F>(
    out: &mut Vec<String>,
    factor: usize,
    acc: &str,
    k: &str,
    bound: &str,
    decl: &str,
    indent: &str,
    term: F,
) where
    F: Fn(String) -> String,
{
    let factor = factor.max(1);
    out.push(format!("{}{} {} = 0;", indent, decl, k));
    if factor > 1 {
        out.push(format!(
            "{}for (; {} + {} < {}; {} += {}) {{",
            indent, k, factor - 1, bound, k, factor
        ));
        for u in 0..factor {
            let idx = if u == 0 { k.to_string() } else { format!("{} + {}", k, u) };
            out.push(format!("{}  {} += {};", indent, acc, term(idx)));
        }
        out.push(format!("{}}}", indent));
    }
    out.push(format!("{}for (; {} < {}; {}++) {{", indent, k, bound, k));
    out.push(format!("{}  {} += {};", indent, acc, term(k.to_string())));
    out.push(format!("{}}}", indent));
}

#[cfg(test)]
mod k_unroll_tests {
    use super::*;

    fn render(factor: usize) -> String {
        let mut v = Vec::new();
        emit_unrolled_k_accumulation(&mut v, factor, "acc", "k", "K", "int", "", |i| {
            format!("a[{}] * b[{}]", i, i)
        });
        v.join("\n")
    }

    /// The receipted default must reproduce the hand-written 4x shape the
    /// emitters used before this helper existed.
    #[test]
    fn factor_four_matches_the_hand_written_shape() {
        let s = render(DEFAULT_K_UNROLL);
        assert!(s.contains("for (; k + 3 < K; k += 4) {"), "{}", s);
        assert!(s.contains("acc += a[k] * b[k];"), "{}", s);
        assert!(s.contains("acc += a[k + 3] * b[k + 3];"), "{}", s);
        assert!(s.contains("for (; k < K; k++) {"), "{}", s);
    }

    /// Every factor must visit k in strictly increasing order with ONE
    /// accumulator — that invariant, not the factor, is what keeps the
    /// transform bit-exact.
    #[test]
    fn every_factor_preserves_accumulation_order() {
        for f in [1usize, 2, 4, 8] {
            let s = render(f);
            let offsets: Vec<i32> = s
                .lines()
                .filter(|l| l.contains("acc +="))
                .map(|l| {
                    let inside = &l[l.find("a[").unwrap() + 2..];
                    let idx = &inside[..inside.find(']').unwrap()];
                    idx.split('+').nth(1).map(|o| o.trim().parse().unwrap()).unwrap_or(0)
                })
                .collect();
            // main-body offsets ascend 0..f-1, then the tail re-emits offset 0
            let body = &offsets[..offsets.len() - 1];
            assert!(body.windows(2).all(|w| w[0] < w[1]),
                "factor {} must accumulate in increasing k: {:?}", f, offsets);
            assert_eq!(*offsets.last().unwrap(), 0, "tail must continue at k");
            assert_eq!(s.matches("acc +=").count(), if f > 1 { f + 1 } else { 1 },
                "factor {} term count", f);
        }
    }

    /// Factor 1 (and 0) must degenerate to a plain scalar loop with no
    /// unrolled body at all.
    #[test]
    fn factor_one_degenerates_to_scalar_loop() {
        for f in [0usize, 1] {
            let s = render(f);
            assert!(!s.contains("k += "), "factor {} must not emit a strided loop:\n{}", f, s);
            assert_eq!(s.matches("acc +=").count(), 1, "factor {}", f);
        }
    }
}

#[cfg(test)]
mod intrinsic_gate_tests {
    use super::reject_unknown_intrinsics;

    const KNOWN: &[&str] = &["__tile_load_", "__tile_store_", "__tile_add_"];

    #[test]
    fn a_known_stem_passes_and_an_unknown_one_names_the_target() {
        let ok = "%a = llvm.call @__tile_load_f32(%p) : (i32) -> i32";
        assert!(reject_unknown_intrinsics(ok, "mlir_to_demo", KNOWN).is_ok());

        let bad = "%a = llvm.call @__tile_rope_f32(%p) : (i32) -> i32";
        let e = reject_unknown_intrinsics(bad, "mlir_to_demo", KNOWN)
            .expect_err("an intrinsic outside the allowlist must refuse");
        assert!(e.contains("mlir_to_demo"), "the target must be named: {e}");
        assert!(e.contains("__tile_rope_f32"), "the intrinsic must be named: {e}");
    }

    /// A commented-out call is not a call. Pinned because the twelve copies this
    /// replaced all skipped comment lines, and dropping that would turn every
    /// worked example in a doc comment into a refusal.
    #[test]
    fn a_commented_call_is_not_a_call() {
        let src = "    // %a = llvm.call @__tile_rope_f32(%p) : (i32) -> i32";
        assert!(reject_unknown_intrinsics(src, "mlir_to_demo", KNOWN).is_ok());
    }

    /// The documented limitation, pinned as behaviour so that nobody reads a pass
    /// from this gate as "the target has an arm".
    ///
    /// `__tile_load_` admits `__tile_load_i8_strided_v4`, which no dispatch arm
    /// need name. The gate accepts it; a target that dispatches by exact match
    /// then falls through and drops the op. That is why `mlir_to_msl` and
    /// `mlir_to_spirv` carry `reject_unclassified_copy` after classification, and
    /// why the other ten are still exposed to it. If this test ever starts
    /// failing, the gate became exact-matching and the caveat on
    /// `reject_unknown_intrinsics` should be deleted along with this test.
    #[test]
    fn a_longer_spelling_under_a_known_stem_is_accepted_which_is_the_gap() {
        let src = "%a = llvm.call @__tile_load_i8_strided_v4(%p) : (i32) -> i32";
        assert!(
            reject_unknown_intrinsics(src, "mlir_to_demo", KNOWN).is_ok(),
            "prefix matching accepts this today; the check that catches it is a \
             separate post-classification one"
        );
    }
}
