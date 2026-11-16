// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Shared helpers for parsing the stderr output of
//! `ptoas --print-after-all --print-module-scope`.
//!
//! ptoas's `--print-after-all` dumps one `module attributes {…} { … }`
//! block per pass to stderr (no `// -----` / "IR Dump After X" banners —
//! those are LLVM-only). The dump we care about for safety checking is
//! the **last** one, emitted after `PlanMemoryPass` has assigned concrete
//! byte offsets to every tile.
//!
//! Both `pto-diff` (CLI) and `rustc_codegen_tile` (in-process safety
//! oracle) consume this output, so the extraction logic lives here.

/// Extract the last `module attributes {…} { … }` block from ptoas
/// stderr, returning it verbatim (including the outer `module` wrapper).
///
/// Returns `None` when no `module attributes` line appears at column 0.
///
/// Parsing strategy:
///   1. Find every line that starts with `module attributes` — note the
///      byte offset of each.
///   2. Start from the last such offset.
///   3. Walk brace depth past the attribute dict (`module attributes
///      {pto.target_arch = "a5"}`) — this is the FIRST `{…}` pair.
///   4. Continue walking brace depth over the module body — the SECOND
///      `{ … }` pair, which may be multi-line and nested.
///   5. Return the slice from the module's line-start to the body's
///      closing `}`, inclusive.
///
/// If the body never closes cleanly (truncated stderr), we return the
/// slice from the module start to EOF so callers can still try to parse.
pub fn extract_last_module(text: &str) -> Option<String> {
    let mut starts: Vec<usize> = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.starts_with("module attributes") {
            let offset: usize = text.lines().take(i).map(|l| l.len() + 1).sum();
            starts.push(offset);
        }
    }
    let start = *starts.last()?;
    let bytes = text.as_bytes();

    let hdr_open = text[start..].find('{')? + start;
    // Walk past the attribute dict `{…}`.
    let mut depth: i32 = 0;
    let mut i = hdr_open;
    while i < bytes.len() {
        match bytes[i] {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
            _ => {}
        }
        i += 1;
    }
    // Walk the module body.
    depth = 0;
    let mut seen_body_open = false;
    i += 1;
    while i < bytes.len() {
        match bytes[i] {
            b'{' => {
                depth += 1;
                seen_body_open = true;
            }
            b'}' => {
                depth -= 1;
                if seen_body_open && depth == 0 {
                    return Some(text[start..=i].to_string());
                }
            }
            _ => {}
        }
        i += 1;
    }
    Some(text[start..].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_module_roundtrip() {
        let s = "module attributes {pto.target_arch = \"a5\"} {\n\
                 func.func @k() { return }\n\
                 }\n";
        let got = extract_last_module(s).expect("should find module");
        assert!(got.starts_with("module attributes {pto.target_arch = \"a5\"}"));
        assert!(got.trim_end().ends_with('}'));
    }

    #[test]
    fn picks_last_of_many() {
        let s = "module attributes {pto.target_arch = \"a5\"} {\n\
                 func.func @k() { return }\n\
                 }\n\
                 some pass banner\n\
                 module attributes {pto.target_arch = \"a5\"} {\n\
                 func.func @k() {\n  %c = arith.constant 42 : i64\n  return\n}\n\
                 }\n";
        let got = extract_last_module(s).expect("should find last module");
        assert!(got.contains("arith.constant 42"), "got: {got}");
    }

    #[test]
    fn none_when_no_module() {
        assert!(extract_last_module("").is_none());
        assert!(extract_last_module("some stderr without a module block\n").is_none());
    }

    #[test]
    fn nested_braces_in_body() {
        // Tile config uses nested `{…}` — the walker must not stop at
        // the inner closer.
        let s = "module attributes {pto.target_arch = \"a5\"} {\n\
                 func.func @k() {\n  \
                 %0 = pto.pointer_cast(%c0_i64) {config = #pto.tile_buf_config<a=#pto.blayout<row_major>>} : memref<...>\n  \
                 return\n}\n}\n";
        let got = extract_last_module(s).expect("should extract");
        assert!(got.contains("tile_buf_config"));
        assert!(got.trim_end().ends_with('}'));
        // The extractor returns exactly up to the module's closing brace —
        // nothing after.
        let last = got.trim_end();
        assert_eq!(last.as_bytes()[last.len() - 1], b'}');
    }

    #[test]
    fn truncated_body_falls_back_to_rest() {
        let s = "module attributes {pto.target_arch = \"a5\"} {\n\
                 func.func @k() {\n  // truncated here\n";
        // No matching `}` for the body — we still return the slice.
        let got = extract_last_module(s).expect("should return partial");
        assert!(got.contains("truncated here"));
    }
}
