// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! The `ptoas-rs` command: step 5 of the seven-tool chain, lifting
//! PTO-dialect MLIR assembly back to tile-rs Rust.
//!
//! Thin on purpose: everything testable lives in the library
//! (`convert_pto_to_rust`, `convert_pto_to_rust_raw`, `convert_pto_to_tile_safe`),
//! all three exercised by this crate's own suite.
//!
//! Two honesty mechanisms, both of which exist because an exit code of 0 from
//! this tool has meant less than it looks:
//!
//! * **The coverage line.** A module the parser ACCEPTS while modelling none of
//!   its ops also returns `Ok`, with a near-empty translation -- garbage text
//!   returns `Ok` and emits zero functions. Since unmodelled ops are now carried
//!   verbatim rather than refused, that is the normal case for a file this crate
//!   does not understand, not an exotic one. `op_model_coverage` is the crate's
//!   own instrument for exactly that gap, so every run prints it.
//! * **Exit code 4.** Distinct from 0, for "succeeded but modelled nothing".
//!   Reported to a caller that reads the status and not the stderr.
//!
//! Both ported from `pto-rs-viewpoint-rebase`, along with the name: the public
//! release workflow builds `--bin ptoas-rs`.

use ptoas_rs::{
    convert_pto_to_rust, convert_pto_to_rust_raw, convert_pto_to_tile_safe, op_model_coverage,
};
use std::io::Write;
use std::process::ExitCode;

const HELP: &str = "\
ptoas-rs — lift PTO-dialect MLIR assembly to tile-rs Rust

Usage:
  ptoas-rs <input.pto> [-o <out.rs>] [--stdout] [--raw|--checked]
            [--safety] [--verify rustc|miri|both] [--emit rs|tile]

Options:
  -o, --output <PATH>  Output .rs file (default: input.pto -> input.rs)
      --stdout         Write output to stdout instead of a file
      --safety         Report the address-shape safety findings for the input
                       on the console: aliasing, capacity, op constraints,
                       matmul bounds, dead tiles and linear use. Advisory --
                       findings are printed, nothing is blocked.
      --verify <tier>  Check the lifted kernel and report what was proved.
                       `rustc` type-checks it against the kernel-side DSL
                       (shape agreement and tile-handle linearity). `miri`
                       RUNS it against a std CPU reference, with buffers sized
                       from the PTO, and reports undefined behaviour -- an
                       out-of-bounds load, a misaligned or dangling pointer, a
                       read of uninitialised memory. `both` does each in turn.
                       Needs nightly-2025-08-04 (and its miri component for
                       the second tier); override with
                       $PTOAS_RS_KERNEL_TOOLCHAIN. Reports unavailable rather
                       than clean when a checker could not run -- an empty
                       finding list and a checker that never ran are not the
                       same answer.
      --emit <form>    `rs` (default), or `tile` for the tile-dialect MLIR
                       (`llvm.func` + `hacc.entry` + `__tile_*`) written
                       straight from the AST -- no Rust toolchain needed.
                       `rspto` consumes that form, so `.pto -> .tile -> rspto`
                       closes the loop on its own output. Not combinable with
                       --verify, which checks the lifted Rust.
      --raw            Skip the softmax/rms-norm pattern fusions; only the
                       decomposed quantise chain still coalesces, because its
                       saturating casts have no per-op tile-API lowering
      --checked        Emit the INTRINSIC-level form instead of the default
                       wrapper lift. Refuses strictly more (5 of 45 corpus
                       artifacts against the default's 13, a strict subset):
                       casts, transposes, regions and non-origin views. Note
                       it is the DEFAULT output whose shapes rustc checks --
                       this form has no const generics, and carries
                       #[tile_std::tile_kernel], so it needs the tile-rs
                       rustc backend to compile at all.
  -h, --help           Show this help

Exit codes:
  0 = success
  1 = bad arguments
  2 = I/O error
  3 = conversion refused (parse or emit error from the library)
  4 = succeeded but modelled 0 of the input's ops, OR a requested route is
      unavailable on this machine. Either way: nothing was proved.
  5 = lifted, but a requested checker REJECTED the result

Input is PTO-dialect MLIR text: what `pto-diff --from-pto` and `ptoas` emit.
Every run prints a coverage line (modelled/unmodelled op count). Read it before
trusting an exit-0 result: an op this crate does not model is carried through a
PTO round trip verbatim, which keeps the round trip faithful and says nothing
about whether the lift understood the kernel.
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprint!("{HELP}");
        return ExitCode::from(1);
    }
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print!("{HELP}");
        return ExitCode::SUCCESS;
    }

    let mut input: Option<String> = None;
    let mut output: Option<String> = None;
    let mut to_stdout = false;
    let mut raw = false;
    let mut checked = false;
    let mut safety = false;
    let mut verify: Option<&'static str> = None;
    let mut emit_tile = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--output" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    eprintln!("ptoas-rs: {} needs a value", args[i - 1]);
                    return ExitCode::from(1);
                };
                output = Some(v.clone());
            }
            "--stdout" => to_stdout = true,
            "--safety" => safety = true,
            "--verify" => {
                i += 1;
                match args.get(i).map(String::as_str) {
                    Some("rustc") => verify = Some("rustc"),
                    Some("miri") => verify = Some("miri"),
                    Some("both") => verify = Some("both"),
                    Some(other) => {
                        eprintln!("ptoas-rs: unknown tier '{other}'; use rustc, miri or both");
                        return ExitCode::from(1);
                    }
                    None => {
                        eprintln!("ptoas-rs: --verify needs a tier");
                        return ExitCode::from(1);
                    }
                }
            }
            "--emit" => {
                i += 1;
                match args.get(i).map(String::as_str) {
                    Some("rs") => {}
                    Some("tile") => emit_tile = true,
                    Some(other) => {
                        eprintln!("ptoas-rs: unknown form '{other}'; use rs or tile");
                        return ExitCode::from(1);
                    }
                    None => {
                        eprintln!("ptoas-rs: --emit needs a form");
                        return ExitCode::from(1);
                    }
                }
            }
            "--raw" => raw = true,
            "--checked" => checked = true,
            // An unrecognised FLAG is a usage error, and this arm has to come
            // before the positional one. Without it `--check` -- a real flag,
            // but `rspto`'s -- became the input path, and the tool reported
            // "could not read --check: No such file or directory". A user
            // reading that goes looking for a missing file instead of for the
            // wrong tool, and a mistyped flag would be swallowed the same way.
            other if other.starts_with("--") => {
                eprintln!(
                    "ptoas-rs: unknown flag {other}\n\nptoas-rs --help for usage"
                );
                return ExitCode::from(2);
            }
            other if input.is_none() => input = Some(other.to_string()),
            other => {
                eprintln!("ptoas-rs: unexpected argument '{other}'\n\n{HELP}");
                return ExitCode::from(1);
            }
        }
        i += 1;
    }

    let Some(input) = input else {
        eprintln!("ptoas-rs: no input given\n\nptoas-rs --help for usage");
        return ExitCode::from(1);
    };
    if raw && checked {
        eprintln!("ptoas-rs: --raw and --checked select different emitters");
        return ExitCode::from(1);
    }
    // Verification is about the lifted RUST -- it compiles it against the
    // kernel-side DSL. There is nothing for rustc or Miri to say about a file
    // of MLIR, and quietly verifying the Rust while writing the MLIR would
    // report a proof about a file the user never received.
    if emit_tile && verify.is_some() {
        eprintln!(
            "ptoas-rs: --verify checks the lifted Rust, so it cannot be combined with \
             --emit tile.\n\nRun the lift twice if you want both: once with --verify, \
             once with --emit tile."
        );
        return ExitCode::from(2);
    }

    let pto_src = match std::fs::read_to_string(&input) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("ptoas-rs: could not read {input}: {e}");
            return ExitCode::from(2);
        }
    };

    let converted = if checked {
        convert_pto_to_tile_safe(&pto_src)
    } else if raw {
        convert_pto_to_rust_raw(&pto_src)
    } else {
        convert_pto_to_rust(&pto_src)
    };
    let rust = match converted {
        Ok(r) => r,
        Err(e) => {
            eprintln!("ptoas-rs: refused {input}: {e}");
            return ExitCode::from(3);
        }
    };

    // Coverage is reported even if the write below fails: the conversion is the
    // thing worth knowing about regardless of where its output ended up.
    let zero_coverage = match op_model_coverage(&pto_src) {
        Ok((modelled, unmodelled)) => {
            eprintln!("ptoas-rs: coverage {modelled} modelled op(s), {unmodelled} unmodelled");
            if modelled == 0 {
                eprintln!(
                    "ptoas-rs: WARNING -- 0 modelled ops. The input was accepted but \
                     nothing in it was recognised; this is not a conversion of your kernel."
                );
            }
            modelled == 0
        }
        Err(e) => {
            // `op_model_coverage` re-parses independently of the conversion; one
            // succeeding while the other fails would itself be worth knowing.
            eprintln!("ptoas-rs: coverage check could not run: {e}");
            false
        }
    };
    // --- the console reports, before the output is written ----------------
    // Printed to stderr so `--stdout` stays pipeable: a report mixed into the
    // Rust on stdout would make the lift's output unusable as input to anything.
    // `--emit tile` writes the tile-dialect MLIR straight from the AST. It does
    // NOT go via the lifted Rust: that route would need a prebuilt rustc
    // backend and an exact nightly to produce text the emitter writes directly,
    // which is why this used to refuse with exit 4 and no longer does.
    //
    // Replaces `rust` wholesale rather than writing a second file: the output
    // path and the `--stdout` plumbing below are shared, and a mode that wrote
    // both would have to invent a second destination.
    //
    // `rust` stays the lifted Rust, because that is what `--verify` checks.
    // `payload` is what gets WRITTEN. Collapsing the two would mean either
    // handing rustc a file of MLIR or losing the verification arm under
    // `--emit tile`, and the combination is refused above rather than silently
    // doing one of those.
    let payload = if emit_tile {
        match ptoas_rs::emitter_mlir::emit_source(&pto_src) {
            Ok(mlir) => mlir,
            Err(e) => {
                eprintln!("ptoas-rs: refused {input}: {e}");
                return ExitCode::from(3);
            }
        }
    } else {
        rust.clone()
    };

    if safety {
        match ptoas_rs::parse_stage2(&pto_src) {
            Ok(plan) => {
                let spec = ptoas_rs::default_a5_910b2_cann85();
                let report = ptoas_rs::check_all(&plan, &spec);
                if report.violations.is_empty() {
                    eprintln!("ptoas-rs: safety 0 findings");
                } else {
                    eprintln!("ptoas-rs: safety {} finding(s)", report.violations.len());
                    for v in &report.violations {
                        eprintln!("  [{}] {}: {}", v.kind.label(), v.func, v.message);
                    }
                }
            }
            Err(e) => {
                // Honest about the gap rather than silent: the stage-2 oracle
                // reads a memory-plan dump, and a plain .pto is not one.
                eprintln!(
                    "ptoas-rs: safety could not run -- {e}\n  \
                     The address-shape checks read a memory-plan dump \
                     (`ptoas --mlir-print-ir-after-all`), not a bare .pto."
                );
            }
        }
    }

    let mut rejected = false;
    let mut unprovable = false;
    if let Some(tier) = verify {
        let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let scratch = std::env::temp_dir().join(format!("ptoas_rs_verify_{}", std::process::id()));
        let tiers: &[ptoas_rs::verify_rustc::Tier] = match tier {
            "rustc" => &[ptoas_rs::verify_rustc::Tier::Check],
            "miri" => &[ptoas_rs::verify_rustc::Tier::Miri],
            _ => &[
                ptoas_rs::verify_rustc::Tier::Check,
                ptoas_rs::verify_rustc::Tier::Miri,
            ],
        };
        for t in tiers {
            let outcome = ptoas_rs::verify_rustc::verify(&rust, &pto_src, *t, &scratch, manifest_dir);
            eprintln!("ptoas-rs: {}", outcome.render());
            match outcome {
                ptoas_rs::verify_rustc::Outcome::Rejected { .. } => rejected = true,
                // Asking for verification and not getting it is not success.
                // Exit 0 here would let a caller reading only the status treat
                // "the checker never ran" as "the checker was happy", which is
                // the confusion this whole module exists to prevent.
                ptoas_rs::verify_rustc::Outcome::Unavailable { .. } => unprovable = true,
                ptoas_rs::verify_rustc::Outcome::Proved { .. } => {}
            }
        }
        let _ = std::fs::remove_dir_all(&scratch);
    }

    let status = if rejected {
        // A rejection outranks a coverage warning: the kernel was understood
        // well enough to compile, and the compiler said no.
        ExitCode::from(5)
    } else if zero_coverage || unprovable {
        ExitCode::from(4)
    } else {
        ExitCode::SUCCESS
    };

    if to_stdout {
        print!("{payload}");
        return status;
    }

    // The extension follows the FORM, not the tool. `--emit tile` writing a
    // `.rs` file would break the one contract all four tools share: the
    // extension decides how the next tool in the chain reads the file, so
    // `rspto` would refuse its own input.
    let ext = if emit_tile { "tile" } else { "rs" };
    let out_path = output.unwrap_or_else(|| match input.strip_suffix(".pto") {
        Some(stem) => format!("{stem}.{ext}"),
        None => format!("{input}.{ext}"),
    });
    match std::fs::File::create(&out_path).and_then(|mut f| f.write_all(payload.as_bytes())) {
        Ok(()) => {
            eprintln!("ptoas-rs: wrote {out_path}");
            status
        }
        Err(e) => {
            eprintln!("ptoas-rs: could not write {out_path}: {e}");
            ExitCode::from(2)
        }
    }
}
