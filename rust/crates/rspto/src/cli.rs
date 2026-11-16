// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Argument parsing, hand-rolled.
//!
//! No clap, for the reason `tile_cli` gives: this repo ships three crates.io
//! dependencies in total, and a six-flag CLI is not worth multiplying that.

/// Exit codes, matching `tile_cli`'s so scripts can treat the two interchangeably.
pub const OK: i32 = 0;
pub const TRANSFORM_FAILED: i32 = 1;
pub const USAGE: i32 = 2;
/// The Rust → MLIR toolchain is not installed. Distinct from a failed transform:
/// nothing was wrong with the input.
pub const UNAVAILABLE: i32 = 4;

/// What to write out.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Emit {
    /// PTO-dialect MLIR assembly — the point of this tool.
    Pto,
    /// Stop at the tile-dialect MLIR, written as `.tile`. From MLIR input this
    /// is a copy under a known extension; it stays so that `--emit tile` means
    /// the same thing once the Rust input mode lands.
    Tile,
}

/// Which mode the input selected, decided by extension alone.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Input {
    /// `.rs` — tile-rs Rust. Needs the codegen backend and a nightly toolchain,
    /// neither of which is built into this release.
    Rust,
    /// `.tile` or `.mlir` — tile-dialect MLIR text. Needs nothing.
    Mlir,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Args {
    pub input: Option<String>,
    pub output: Option<String>,
    pub emit: Emit,
    pub check: bool,
    /// Fetch the codegen backend, so the Rust mode needs no other tool.
    pub help: bool,
    pub version: bool,
    pub verbose: bool,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            input: None,
            output: None,
            emit: Emit::Pto,
            check: false,
            help: false,
            version: false,
            verbose: false,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct UsageError(pub String);

pub fn parse<I: IntoIterator<Item = String>>(argv: I) -> Result<Args, UsageError> {
    let mut a = Args::default();
    let mut it = argv.into_iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-h" | "--help" => a.help = true,
            "-v" | "--version" => a.version = true,
            "-V" | "--verbose" => a.verbose = true,
            "--check" => a.check = true,
            "-o" | "--output" => {
                a.output = Some(
                    it.next()
                        .ok_or_else(|| UsageError("-o needs a path".into()))?,
                )
            }
            "--emit" => {
                let v = it
                    .next()
                    .ok_or_else(|| UsageError("--emit needs a form".into()))?;
                a.emit = match v.as_str() {
                    "pto" => Emit::Pto,
                    // `.mlir` stays accepted: it is the spelling the codegen backend writes.
                    "tile" | "mlir" => Emit::Tile,
                    other => {
                        return Err(UsageError(format!(
                            "--emit {other}: this tool emits `pto` or `mlir`"
                        )));
                    }
                };
            }
            other if other.starts_with('-') && other.len() > 1 => {
                return Err(UsageError(format!("unknown flag {other}")));
            }
            positional => {
                if a.input.is_some() {
                    return Err(UsageError(format!(
                        "two inputs given ({} and {positional}); this tool takes one",
                        a.input.as_deref().unwrap_or("")
                    )));
                }
                a.input = Some(positional.to_string());
            }
        }
    }
    Ok(a)
}

/// Decide the mode from the extension. Deliberately not content-sniffing: a wrong
/// guess here means either running a toolchain the user did not ask for, or feeding
/// Rust to an MLIR parser and reporting a parse error about their own source.
pub fn mode_of(path: &str) -> Result<Input, UsageError> {
    match path.rsplit_once('.').map(|(_, e)| e) {
        // `.tile` is this project's name for tile-dialect MLIR; `.mlir` is what
        // the codegen backend itself writes, so both reach the same mode and a
        // backend artefact needs no renaming before use.
        Some("tile") | Some("mlir") => Ok(Input::Mlir),
        Some("rs") => Ok(Input::Rust),
        Some(other) => Err(UsageError(format!(
            "{path}: unknown extension `.{other}` — give a .tile (tile-dialect MLIR) or a .rs (tile-rs kernel)"
        ))),
        None => Err(UsageError(format!(
            "{path}: no extension — give a .tile (tile-dialect MLIR) or a .rs (tile-rs kernel)"
        ))),
    }
}

pub const HELP: &str = "\
rspto — lower a tile-rs kernel to PTO assembly

USAGE
    rspto [OPTIONS] <input>

INPUT MODES (chosen by extension)
    <kernel>.tile    tile-dialect MLIR. Needs nothing at all.
    <kernel>.mlir    the same thing, under the spelling the codegen backend
                     writes. Accepted so a backend artefact needs no renaming.
    <kernel>.rs      a tile-rs Rust kernel. Needs the Rust → MLIR half, which is
                     NOT built into this release; refused with exit code 4.

OPTIONS
    -o, --output <file>   where to write; `-` is stdout.
                          Default: <input> with the emitted extension.
        --emit <form>     `pto` (default), or `tile` to stop at the MLIR.
        --check           report whether the Rust → MLIR half is usable, and
                          exit 0 if it is or 4 if it is not. Reads no input.
    -V, --verbose         narrate the steps on stderr.
    -v, --version         print the version.
    -h, --help            this.

EXIT CODES
    0 ok   1 transform failed   2 usage   4 toolchain unavailable

EXAMPLES
    rspto kernel.tile                   # kernel.pto
    rspto kernel.tile -o -              # PTO on stdout
    rspto kernel.mlir --emit tile -o k.tile
";
