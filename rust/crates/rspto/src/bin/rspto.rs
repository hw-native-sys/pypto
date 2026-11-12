// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! The `rspto` command. Thin: everything testable lives in the library.

use rspto::cli::{self, Emit, Input};
use std::io::Write;

fn main() -> std::process::ExitCode {
    let args = match cli::parse(std::env::args().skip(1)) {
        Ok(a) => a,
        Err(e) => {
            return die(
                cli::USAGE,
                &format!("{}\n\n{}", e.0, "rspto --help for usage"),
            );
        }
    };

    if args.help {
        print!("{}", cli::HELP);
        return code(cli::OK);
    }
    if args.version {
        println!("rspto {}", env!("CARGO_PKG_VERSION"));
        return code(cli::OK);
    }
    if args.check {
        // Honest about the half that is absent rather than silent about it: a
        // check that cannot fail would tell a user nothing.
        println!("Rust \u{2192} MLIR toolchain: not built into this release");
        println!("\n`rspto kernel.tile` works and needs nothing");
        return code(cli::UNAVAILABLE);
    }

    let Some(input) = args.input.clone() else {
        return die(cli::USAGE, "no input given\n\nrspto --help for usage");
    };
    let mode = match cli::mode_of(&input) {
        Ok(m) => m,
        Err(e) => return die(cli::USAGE, &e.0),
    };

    // --- read the MLIR ------------------------------------------------------
    let mlir = match mode {
        Input::Mlir => match std::fs::read_to_string(&input) {
            Ok(t) => t,
            Err(e) => return die(cli::TRANSFORM_FAILED, &format!("{input}: {e}")),
        },
        Input::Rust => {
            return die(
                cli::UNAVAILABLE,
                &format!(
                    "{input}: tile-rs source input needs the Rust \u{2192} MLIR half, \
                     which is not built into this release.\n\nLower the kernel to \
                     tile-dialect MLIR first, then run `rspto <kernel>.tile`."
                ),
            );
        }
    };

    // --- emit ---------------------------------------------------------------
    let (text, ext) = match args.emit {
        // Already MLIR on the way in, so this is a copy with a known extension.
        // It stays because it is what makes `--emit tile` mean the same thing in
        // both input modes once the other one lands.
        Emit::Tile => (mlir.clone(), "tile"),
        Emit::Pto => match rspto::mlir_to_pto(&mlir) {
            Ok(p) => (p, "pto"),
            Err(e) => return die(cli::TRANSFORM_FAILED, &format!("{e}")),
        },
    };

    // --- write --------------------------------------------------------------
    let dest = args
        .output
        .clone()
        .unwrap_or_else(|| replace_ext(&input, ext));
    let wrote = if dest == "-" {
        std::io::stdout().write_all(text.as_bytes()).map(|_| ())
    } else {
        std::fs::write(&dest, &text)
    };
    if let Err(e) = wrote {
        return die(cli::TRANSFORM_FAILED, &format!("{dest}: {e}"));
    }
    if dest != "-" {
        eprintln!("rspto: wrote {dest}");
    }
    code(cli::OK)
}

fn replace_ext(path: &str, ext: &str) -> String {
    match path.rsplit_once('.') {
        Some((stem, _)) => format!("{stem}.{ext}"),
        None => format!("{path}.{ext}"),
    }
}

fn die(c: i32, msg: &str) -> std::process::ExitCode {
    eprintln!("rspto: {msg}");
    code(c)
}

fn code(c: i32) -> std::process::ExitCode {
    std::process::ExitCode::from(c as u8)
}
