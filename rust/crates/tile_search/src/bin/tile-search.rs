// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! The `tile-search` command. Thin: everything testable lives in the library.

use std::io::Write;
use std::process::ExitCode;

const OK: u8 = 0;
const NOTHING_ACCEPTED: u8 = 1;
const USAGE: u8 = 2;
const UNAVAILABLE: u8 = 4;

const HELP: &str = "\
tile-search — enumerate the tilings a kernel admits and rank the ones that work

USAGE
    tile-search [OPTIONS] <input>

INPUT MODES (chosen by extension)
    <kernel>.tile    tile-dialect MLIR. Needs nothing at all.
    <kernel>.mlir    the same thing, as the codegen backend spells it.
    <kernel>.rs      a tile-rs kernel. Needs the Rust -> MLIR half, which is
                     NOT built into this release; refused with exit code 4.

OPTIONS
    -o, --output <file>   where to write the chosen kernel; `-` is stdout.
                          Default: <input> with the emitted extension.
        --oracle <which>  which emitter decides a candidate: `pto` (default)
                          or `ascendc`. They do not agree -- the AscendC
                          emitter divides the Unified Buffer into fixed slots,
                          so its limit is per BUFFER where the PTO emitter's is
                          over the live working set. Searching against the
                          wrong one reports nothing to change for a kernel
                          that is refused downstream.
        --emit <form>     `tile` (default) writes the chosen tile-dialect MLIR.
                          `rs` lifts it back to tile-rs, so the output is a
                          kernel rather than a chosen number. The lift models
                          fewer ops than the emitter, so `rs` can fail for a
                          kernel `tile` succeeds on; the message says which.
        --dry-run         report the search and write nothing.
    -v, --version         print the version.
    -h, --help            this.

EXIT CODES
    0 ok   1 no candidate was accepted   2 usage   4 route not available

WHAT THIS DOES AND DOES NOT CLAIM
    Every candidate is decided by running the real emitter, so an accepted
    tiling is one the target genuinely takes. Nothing is run on hardware, so
    the ORDERING of accepted candidates is a declared cost model and not a
    measurement -- the report says which model, and lists every candidate it
    refused along with the bound each one hit.
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut input: Option<String> = None;
    let mut output: Option<String> = None;
    let mut emit_rs = false;
    let mut dry_run = false;
    let mut oracle = tile_search::Oracle::Pto;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print!("{HELP}");
                return ExitCode::from(OK);
            }
            "-v" | "--version" => {
                println!("tile-search {}", env!("CARGO_PKG_VERSION"));
                return ExitCode::from(OK);
            }
            "--dry-run" => dry_run = true,
            "--oracle" => {
                i += 1;
                match args.get(i).map(String::as_str) {
                    Some(v) => match tile_search::Oracle::parse(v) {
                        Some(o) => oracle = o,
                        None => {
                            return die(
                                USAGE,
                                &format!(
                                    "--oracle {v}: expected `pto` or `ascendc`"
                                ),
                            )
                        }
                    },
                    None => return die(USAGE, "--oracle needs a name"),
                }
            }
            "--emit" => {
                i += 1;
                match args.get(i).map(String::as_str) {
                    Some("tile") => {}
                    Some("rs") => emit_rs = true,
                    Some(other) => return die(USAGE, &format!("unknown form '{other}'; use tile or rs")),
                    None => return die(USAGE, "--emit needs a form"),
                }
            }
            "-o" | "--output" => {
                i += 1;
                match args.get(i) {
                    Some(v) => output = Some(v.clone()),
                    None => return die(USAGE, "-o needs a value"),
                }
            }
            other if other.starts_with('-') => {
                return die(USAGE, &format!("unknown flag {other}\n\ntile-search --help for usage"))
            }
            other => input = Some(other.to_string()),
        }
        i += 1;
    }

    let Some(input) = input else {
        return die(USAGE, "no input given\n\ntile-search --help for usage");
    };

    let mlir = match input.rsplit_once('.').map(|(_, e)| e) {
        Some("tile") | Some("mlir") => match std::fs::read_to_string(&input) {
            Ok(t) => t,
            Err(e) => return die(NOTHING_ACCEPTED, &format!("{input}: {e}")),
        },
        Some("rs") => {
            return die(
                UNAVAILABLE,
                &format!(
                    "{input}: tile-rs source input needs the Rust -> MLIR half, which is \
                     not built into this release.\n\nLower the kernel to tile-dialect MLIR \
                     first, then run `tile-search <kernel>.tile`."
                ),
            )
        }
        Some(other) => {
            return die(
                USAGE,
                &format!("{input}: unknown extension `.{other}` — give a .tile (tile-dialect MLIR)"),
            )
        }
        None => return die(USAGE, &format!("{input}: no extension — give a .tile")),
    };

    let report =
        tile_search::search_with(&mlir, tile_search::Cost::LargestAccepted, oracle);
    // The whole report goes to stderr, so `-o -` stays pipeable.
    eprint!("{}", report.render());

    let Some(chosen) = report.chosen_mlir.clone() else {
        return ExitCode::from(NOTHING_ACCEPTED);
    };
    if dry_run {
        return ExitCode::from(OK);
    }

    let (text, ext) = if emit_rs {
        match ptoas_rs::convert_pto_to_rust(&pto_of(&chosen)) {
            Ok(rs) => (rs, "rs"),
            Err(e) => {
                return die(
                    NOTHING_ACCEPTED,
                    &format!(
                        "the chosen tiling emits PTO, but the lift back to tile-rs refused \
                         it: {e}\n\nThe chosen tile-dialect MLIR is still available with \
                         `--emit tile`."
                    ),
                )
            }
        }
    } else {
        (chosen, "tile")
    };

    let dest = output.unwrap_or_else(|| match input.rsplit_once('.') {
        Some((stem, _)) => format!("{stem}.opt.{ext}"),
        None => format!("{input}.opt.{ext}"),
    });
    let wrote = if dest == "-" {
        std::io::stdout().write_all(text.as_bytes()).map(|_| ())
    } else {
        std::fs::write(&dest, &text)
    };
    if let Err(e) = wrote {
        return die(NOTHING_ACCEPTED, &format!("{dest}: {e}"));
    }
    if dest != "-" {
        eprintln!("tile-search: wrote {dest}");
    }
    ExitCode::from(OK)
}

/// Re-emit the chosen MLIR as PTO, which is what the lift reads.
fn pto_of(mlir: &str) -> String {
    use tile_codegen::registry::TargetRegistry;
    use tile_codegen::target::EmitOpts;
    let mut r = TargetRegistry::new();
    rspto_emit::register(&mut r);
    r.select("pto")
        .and_then(|t| t.emit(mlir, &EmitOpts::default()).ok())
        .map(|o| o.source)
        // The search already proved this emits, so a failure here would be a
        // different bug; returning the MLIR makes the lift's refusal say so.
        .unwrap_or_else(|| mlir.to_string())
}

fn die(code: u8, msg: &str) -> ExitCode {
    eprintln!("tile-search: {msg}");
    ExitCode::from(code)
}
