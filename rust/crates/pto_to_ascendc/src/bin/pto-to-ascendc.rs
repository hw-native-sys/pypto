// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! The `pto-to-ascendc` command. Thin: everything testable lives in the library.

use std::io::Write;
use std::process::{Command, ExitCode};

const OK: u8 = 0;
const TRANSFORM_FAILED: u8 = 1;
const USAGE: u8 = 2;
const UNAVAILABLE: u8 = 4;

const HELP: &str = "\
pto-to-ascendc — emit AscendC C++, for comparing a kernel's representations

USAGE
    pto-to-ascendc [OPTIONS] <input>

INPUT MODES (chosen by extension)
    <kernel>.tile    tile-dialect MLIR. Needs nothing at all.
    <kernel>.mlir    the same thing, under the spelling the codegen backend
                     writes. Accepted so a backend artefact needs no renaming.
    <kernel>.pto     PTO assembly. NOT implemented here; refused with exit 4.
                     Both routes to it — the pinned ptoas assembler, and lifting
                     the PTO back to tile-rs — are later changes.

OPTIONS
    -o, --output <file>   where to write; `-` is stdout.
                          Default: <input> with a .cpp extension.
        --soc <part>      which Ascend part to emit for: 910b (default), 950,
                          950pr, 950dt. Not optional in effect — the emitter
                          checks tiles against that part's unified-buffer
                          budget, and a blank target has none, so every kernel
                          would overflow.
    -v, --version         print the version.
    -h, --help            this.

EXIT CODES
    0 ok   1 transform failed   2 usage   4 route not available

EXAMPLES
    pto-to-ascendc kernel.tile              # kernel.cpp
    pto-to-ascendc kernel.tile -o -         # AscendC on stdout
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut input: Option<String> = None;
    let mut verbose = false;
    let mut output: Option<String> = None;
    let mut soc = pto_to_ascendc::Soc::A910b;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print!("{HELP}");
                return ExitCode::from(OK);
            }
            "-v" | "--version" => {
                println!("pto-to-ascendc {}", env!("CARGO_PKG_VERSION"));
                return ExitCode::from(OK);
            }
            "--soc" => {
                i += 1;
                match args.get(i).map(|s| pto_to_ascendc::Soc::parse(s)) {
                    Some(Some(s)) => soc = s,
                    Some(None) => {
                        return die(USAGE, &format!("unknown soc {:?}; use 910b, 950, 950pr or 950dt", args[i]))
                    }
                    None => return die(USAGE, "--soc needs a name"),
                }
            }
            "-o" | "--output" => {
                i += 1;
                match args.get(i) {
                    Some(v) => output = Some(v.clone()),
                    None => return die(USAGE, "-o needs a value"),
                }
            }
            "-V" | "--verbose" => verbose = true,
            other if other.starts_with('-') => {
                return die(USAGE, &format!("unknown flag {other}\n\npto-to-ascendc --help for usage"));
            }
            other => input = Some(other.to_string()),
        }
        i += 1;
    }

    let Some(input) = input else {
        return die(USAGE, "no input given\n\npto-to-ascendc --help for usage");
    };

    // Extension alone decides, deliberately: content sniffing makes a wrong
    // guess silent, while a wrong extension is a message the user can act on.
    let mlir = match input.rsplit_once('.').map(|(_, e)| e) {
        Some("tile") | Some("mlir") => match std::fs::read_to_string(&input) {
            Ok(t) => t,
            Err(e) => return die(TRANSFORM_FAILED, &format!("{input}: {e}")),
        },
        // The DIRECT route: hand the PTO to the assembler that owns that
        // translation. This is the honest division of labour -- `ptoas` is the
        // reference PTO -> AscendC implementation, and reimplementing it here
        // would produce a second answer that can disagree with the one the
        // device path actually uses.
        //
        // The other route (lift the PTO back to tile-rs, then lower that) is
        // still unavailable, and the lift is no longer what is missing: it
        // works. What is missing is the tile-rs -> MLIR half.
        Some("pto") => {
            let dest = output.clone().unwrap_or_else(|| match input.rsplit_once('.') {
                Some((stem, _)) => format!("{stem}.cpp"),
                None => format!("{input}.cpp"),
            });
            return assemble_with_ptoas(&input, &dest, verbose);
        }
        Some(other) => {
            return die(
                USAGE,
                &format!("{input}: unknown extension `.{other}` — give a .tile (tile-dialect MLIR)"),
            );
        }
        None => return die(USAGE, &format!("{input}: no extension — give a .tile")),
    };

    let cpp = match pto_to_ascendc::mlir_to_ascendc(&mlir, soc) {
        Ok(c) => c,
        Err(e) => return die(TRANSFORM_FAILED, &format!("{e}")),
    };

    let dest = output.unwrap_or_else(|| match input.rsplit_once('.') {
        Some((stem, _)) => format!("{stem}.cpp"),
        None => format!("{input}.cpp"),
    });
    let wrote = if dest == "-" {
        std::io::stdout().write_all(cpp.as_bytes()).map(|_| ())
    } else {
        std::fs::write(&dest, &cpp)
    };
    if let Err(e) = wrote {
        return die(TRANSFORM_FAILED, &format!("{dest}: {e}"));
    }
    if dest != "-" {
        eprintln!("pto-to-ascendc: wrote {dest}");
    }
    ExitCode::from(OK)
}

fn die(code: u8, msg: &str) -> ExitCode {
    eprintln!("pto-to-ascendc: {msg}");
    ExitCode::from(code)
}

/// The version of `ptoas` this repository pins, from `toolchain/versions.env`.
///
/// Checked and REPORTED, never enforced. A mismatch is worth saying out loud
/// because the emitted C++ is what reaches the device, but refusing to run
/// would make the tool useless on a machine with a working assembler of a
/// different version -- which is the common case while a pin is being moved.
const PINNED_PTOAS: &str = "v0.70";

/// Locate the assembler: `$PTOAS` first, then `ptoas` on `PATH`.
fn find_ptoas() -> Option<String> {
    if let Ok(p) = std::env::var("PTOAS") {
        // Checked, not trusted. Returning an unexamined `$PTOAS` meant a
        // misspelled path produced "could not run /x/ptoas: No such file"
        // instead of the message that says what to install and where the pin
        // lives -- the raw errno for the one case where the user has already
        // told us where to look.
        if !p.is_empty() && std::path::Path::new(&p).is_file() {
            return Some(p);
        }
        if !p.is_empty() {
            eprintln!("pto-to-ascendc: note: $PTOAS is set to `{p}`, which is not a file");
        }
    }
    let out = Command::new("sh").arg("-c").arg("command -v ptoas").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let path = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (!path.is_empty()).then_some(path)
}

fn ptoas_version(exe: &str) -> Option<String> {
    let out = Command::new(exe).arg("--version").output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    text.split_whitespace().last().map(str::to_string)
}

/// `.pto` -> `.cpp` through the assembler.
fn assemble_with_ptoas(input: &str, dest: &str, verbose: bool) -> ExitCode {
    let Some(exe) = find_ptoas() else {
        return die(
            UNAVAILABLE,
            &format!(
                "{input}: the PTO route needs the `ptoas` assembler, which is not on PATH.\n\n\
                 Install the pinned wheel ({PINNED_PTOAS}; see toolchain/versions.env), or \
                 point $PTOAS at the binary. From tile-dialect MLIR this tool needs \
                 nothing: pto-to-ascendc <kernel>.tile"
            ),
        );
    };
    if let Some(found) = ptoas_version(&exe) {
        // Reported, not enforced. Silence on a mismatch is the failure mode
        // worth avoiding: the C++ below is what reaches the device.
        let pinned = PINNED_PTOAS.trim_start_matches('v');
        if found.trim_start_matches('v') != pinned {
            eprintln!(
                "pto-to-ascendc: note: ptoas {found} in use, this repository pins \
                 {PINNED_PTOAS} — the emitted C++ may differ from the device path's"
            );
        } else if verbose {
            eprintln!("pto-to-ascendc: ptoas {found} ({exe})");
        }
    } else if verbose {
        eprintln!("pto-to-ascendc: ptoas at {exe} (version unreadable)");
    }

    // `--enable-insert-sync` is the invocation the generated PTO's own header
    // line names, so the two cannot drift apart silently.
    let out = match Command::new(&exe)
        .arg("--enable-insert-sync")
        .arg(input)
        .arg("-o")
        .arg(dest)
        .output()
    {
        Ok(o) => o,
        Err(e) => return die(UNAVAILABLE, &format!("could not run {exe}: {e}")),
    };
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        let msg = if err.trim().is_empty() {
            String::from_utf8_lossy(&out.stdout).to_string()
        } else {
            err.to_string()
        };
        return die(TRANSFORM_FAILED, &format!("ptoas refused {input}:\n{}", msg.trim_end()));
    }
    if dest != "-" {
        eprintln!("pto-to-ascendc: wrote {dest} (via ptoas)");
    }
    ExitCode::from(OK)
}
