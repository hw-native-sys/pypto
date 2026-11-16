// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! `pto-diff` — advisory safety + plan-diff oracle for ptoas output.
//!
//! Workflow:
//!   1. Run ptoas with its "print IR after each pass" option to produce one
//!      module dump per pass. The concrete invocation depends on the ptoas
//!      version:
//!        - ptoas >= 0.29 understands `--mlir-print-ir-after-all` and writes
//!          per-pass dumps to a directory via `--mlir-print-ir-tree-dir`
//!          (the dump file we want is `3_pto-plan-memory.mlir`).
//!        - ptoas 0.26 understood `--print-after-all --print-module-scope`
//!          and emitted all dumps to stderr as `IR Dump After …` blocks.
//!      `ptoas_stage2_dump` tries the 0.29 tree-dir path first and falls
//!      back to stderr scraping if the tree-dir output is empty.
//!   2. Extract the dump emitted immediately after the plan-memory pass
//!      (stage-2 IR — tiles are placed with concrete byte offsets).
//!   3. Parse it into a `Plan`; run every safety check against a
//!      `DeviceSpec` loaded from `--device` (defaults to 910B2/CANN 8.5).
//!   4. (Optional) If the user supplies `--compare <other-stage2.mlir>`,
//!      also diff the two plans and print any differences.
//!
//! Exit status: always 0 unless a CLI/IO error occurred. The point is to
//! *inform* — callers who want to fail CI can pipe through `grep -E
//! '^error:'` and `test -z`.

use std::path::PathBuf;
use std::process::{Command, ExitCode};

use ptoas_rs::{
    check_all, default_a5_910b2_cann85, diff_plans, extract_last_module, parse_stage1,
    parse_stage2, safety::check_all_stage1, DeviceSpec, Plan, PlanDiff, SafetyReport, Severity,
};

const USAGE: &str = "\
pto-diff — run advisory safety checks on ptoas stage-2 MLIR.

USAGE:
    pto-diff [OPTIONS] <stage2.mlir>
    pto-diff [OPTIONS] --from-pto <input.pto>
    pto-diff --stage1 <stage1.mlir>

OPTIONS:
    --device <spec>        Path to a DeviceSpec file (default: embedded
                           a5_910b2_cann85 for Ascend 910B2 / CANN 8.5).
    --ptoas <path>         ptoas binary (used with --from-pto).
                           Defaults to `ptoas` on PATH.
    --from-pto <file>      Run ptoas --print-after-all internally and parse
                           the post-PlanMemoryPass dump instead of reading
                           a pre-extracted stage-2 file.
    --stage1 <file>        Parse <file> as stage-1 PTO MLIR (the user-facing
                           IR before plan-memory) and run stage-1 safety
                           checks (OobPartition over partition_view bounds).
                           Mutually exclusive with the stage-2 paths above.
    --compare <stage2.mlir> Also run plan-diff against this plan and print
                           any differences.
    --fragment <file.pto>  Decide whether <file.pto> is inside the fragment a
                           refinement to the mechanised calculus could cover,
                           and name EVERY op that puts it outside. Reported for
                           the calculus as it stands and for each planned
                           extension, because the obligation the proofs leave
                           open (`sp_step_pres`, op-conservativity) is about
                           exactly this. See proofs/README.md. Standalone:
                           needs no device spec and no ptoas.
    -h, --help             Show this help.
";

struct Args {
    input: Option<PathBuf>,
    from_pto: Option<PathBuf>,
    stage1: Option<PathBuf>,
    device: Option<PathBuf>,
    ptoas: PathBuf,
    compare: Option<PathBuf>,
    fragment: Option<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        input: None,
        from_pto: None,
        stage1: None,
        device: None,
        ptoas: PathBuf::from("ptoas"),
        compare: None,
        fragment: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            "--device" => a.device = Some(PathBuf::from(it.next().ok_or("missing --device arg")?)),
            "--from-pto" => {
                a.from_pto = Some(PathBuf::from(it.next().ok_or("missing --from-pto arg")?))
            }
            "--stage1" => a.stage1 = Some(PathBuf::from(it.next().ok_or("missing --stage1 arg")?)),
            "--ptoas" => a.ptoas = PathBuf::from(it.next().ok_or("missing --ptoas arg")?),
            "--fragment" => {
                a.fragment = Some(PathBuf::from(it.next().ok_or("missing --fragment arg")?))
            }
            "--compare" => {
                a.compare = Some(PathBuf::from(it.next().ok_or("missing --compare arg")?))
            }
            other if other.starts_with('-') => return Err(format!("unknown flag `{other}`")),
            _ => {
                if a.input.is_some() {
                    return Err("multiple positional inputs".into());
                }
                a.input = Some(PathBuf::from(arg));
            }
        }
    }
    if a.fragment.is_some() {
        // Its own mode: it reads an artifact's op vocabulary and nothing else,
        // so pairing it with a stage-2 path would silently run one of them.
        if a.input.is_some() || a.from_pto.is_some() || a.stage1.is_some() || a.compare.is_some() {
            return Err("--fragment is its own mode; it takes no other input".into());
        }
    } else if a.stage1.is_some() {
        if a.input.is_some() || a.from_pto.is_some() || a.compare.is_some() {
            return Err("--stage1 is mutually exclusive with stage-2 inputs and --compare".into());
        }
    } else if a.input.is_none() && a.from_pto.is_none() {
        return Err("must supply a stage-2 file, --from-pto, --stage1, or --fragment".into());
    }
    Ok(a)
}

fn load_device(p: Option<&PathBuf>) -> Result<DeviceSpec, String> {
    match p {
        Some(path) => DeviceSpec::load(path),
        None => Ok(default_a5_910b2_cann85()),
    }
}

/// Invoke ptoas and extract the dump immediately after the plan-memory pass.
///
/// Tries ptoas 0.29's tree-dir interface first; falls back to ptoas 0.26's
/// stderr scraping if the tree-dir path does not produce dumps.
fn ptoas_stage2_dump(pto: &PathBuf, input: &PathBuf) -> Result<String, String> {
    match ptoas_stage2_dump_tree_dir(pto, input) {
        Ok(s) => Ok(s),
        Err(tree_err) => {
            // Older ptoas — flags and dump sink differ.
            match ptoas_stage2_dump_stderr(pto, input) {
                Ok(s) => Ok(s),
                Err(old_err) => Err(format!(
                    "ptoas dump extraction failed on both paths.\n  tree-dir: {tree_err}\n  stderr:  {old_err}"
                )),
            }
        }
    }
}

/// ptoas >= 0.29: write per-pass dumps to a fresh temp directory, then read
/// back the `*pto-plan-memory*.mlir` file.
fn ptoas_stage2_dump_tree_dir(pto: &PathBuf, input: &PathBuf) -> Result<String, String> {
    // Per-invocation tempdir so concurrent callers don't collide.
    let mut tree_dir = std::env::temp_dir();
    tree_dir.push(format!(
        "pto-diff-tree-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
    ));
    // If it exists from a previous aborted run, wipe it first.
    let _ = std::fs::remove_dir_all(&tree_dir);
    std::fs::create_dir_all(&tree_dir).map_err(|e| format!("mkdir {}: {e}", tree_dir.display()))?;

    // Discard stdout (pto0.29 prints generated C++ there when no -o is given)
    // but still check the exit status.
    let out = Command::new(pto)
        .arg("--mlir-print-ir-after-all")
        .arg(format!("--mlir-print-ir-tree-dir={}", tree_dir.display()))
        .arg(input)
        .output()
        .map_err(|e| format!("failed to run {}: {e}", pto.display()))?;
    if !out.status.success() {
        // Clean up the (now probably empty) tempdir before reporting.
        let _ = std::fs::remove_dir_all(&tree_dir);
        return Err(format!(
            "{} --mlir-print-ir-tree-dir exited {:?}: {}",
            pto.display(),
            out.status,
            String::from_utf8_lossy(&out.stderr),
        ));
    }

    // Walk one level deep and look for any file whose name contains
    // `pto-plan-memory`. ptoas puts dumps under `<tree_dir>/<module>/N_pass.mlir`.
    let chosen = find_plan_memory_dump(&tree_dir).ok_or_else(|| {
        format!(
            "no `*pto-plan-memory*.mlir` under {} after ptoas run",
            tree_dir.display(),
        )
    })?;
    let text =
        std::fs::read_to_string(&chosen).map_err(|e| format!("read {}: {e}", chosen.display()))?;
    // Best-effort cleanup — ignore errors.
    let _ = std::fs::remove_dir_all(&tree_dir);
    Ok(text)
}

fn find_plan_memory_dump(root: &std::path::Path) -> Option<PathBuf> {
    // Try the common `<root>/<module>/N_pto-plan-memory.mlir` layout first,
    // falling back to any file in the tree whose name matches.
    let mut best: Option<PathBuf> = None;
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let rd = match std::fs::read_dir(&dir) {
            Ok(rd) => rd,
            Err(_) => continue,
        };
        for entry in rd.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            let name = match path.file_name().and_then(|s| s.to_str()) {
                Some(s) => s,
                None => continue,
            };
            if name.ends_with(".mlir") && name.contains("pto-plan-memory") {
                // Pick the numerically-last one (ptoas prefixes `3_`, `4_` …).
                // Lexicographic ordering works because single-digit prefixes
                // are fine for any reasonable pass count (<10).
                match &best {
                    None => best = Some(path),
                    Some(cur) => {
                        if path.file_name() > cur.file_name() {
                            best = Some(path);
                        }
                    }
                }
            }
        }
    }
    best
}

/// ptoas 0.26: pass-dumps go to stderr; we extract the last `module {…}`
/// block, which (by pass ordering) is the post-`PlanMemoryPass` dump.
fn ptoas_stage2_dump_stderr(pto: &PathBuf, input: &PathBuf) -> Result<String, String> {
    let out = Command::new(pto)
        .arg("--print-after-all")
        .arg("--print-module-scope")
        .arg(input)
        .output()
        .map_err(|e| format!("failed to run {}: {e}", pto.display()))?;
    if !out.status.success() {
        return Err(format!(
            "{} exited with {:?}: {}",
            pto.display(),
            out.status,
            String::from_utf8_lossy(&out.stderr),
        ));
    }
    let text = String::from_utf8_lossy(&out.stderr).into_owned();
    extract_last_module(&text)
        .ok_or_else(|| "ptoas stderr contained no `module attributes {…}` block".to_string())
}

fn print_safety(path_label: &str, rep: &SafetyReport) {
    if rep.is_empty() {
        println!("{path_label}: 0 safety findings");
        return;
    }
    for v in &rep.violations {
        let tag = match v.severity {
            Severity::Error => "error",
            Severity::Warning => "warn",
        };
        println!(
            "{path_label}: {tag}: [{}] {}: {}",
            v.kind.label(),
            v.func,
            v.message
        );
    }
    let errs = rep
        .violations
        .iter()
        .filter(|v| v.severity == Severity::Error)
        .count();
    let warns = rep
        .violations
        .iter()
        .filter(|v| v.severity == Severity::Warning)
        .count();
    println!("{path_label}: {errs} error(s), {warns} warning(s)");
}

fn print_diff(diff: &PlanDiff) {
    if diff.is_empty() {
        println!("plan-diff: identical");
        return;
    }
    for e in &diff.entries {
        println!("plan-diff: {}: {}", e.where_, e.message);
    }
    println!("plan-diff: {} difference(s)", diff.entries.len());
}

/// Report fragment membership under each assumption, with every excluder.
///
/// Prints all three assumptions rather than only the current one, because the
/// question a reader has is not "is it in" but "what would it take" -- and the
/// answer is a different set of ops for each.
///
/// The two lower tiers are now EMPTY of members: the store constructor landed
/// on 2026-11-01 and the compute constructors with it, so every op that used to
/// be in `OpClass::Store` or `OpClass::Compute` is structural, and all three
/// tiers agree on every artifact in the corpus. They are still printed, and
/// their labels say what they mean rather than naming the extensions that have
/// already happened: the tiers are where the NEXT planned counterpart lands,
/// and a reader of a future artifact needs to be able to tell "the calculus
/// covers this" from "a planned extension would".
fn print_fragment(label: &str, src: &str) {
    use ptoas_rs::{ops_of, verdict_for, Assume, OpClass};
    let ops = ops_of(src);
    println!("{label}");
    println!("  ops: {} distinct", ops.len());
    for (name, assume) in [
        ("as the calculus stands", Assume::AsItStands),
        ("+ a planned store counterpart", Assume::WithStore),
        ("+ a planned store and compute counterpart", Assume::WithStoreAndCompute),
    ] {
        let v = verdict_for(src, assume);
        if v.in_fragment {
            println!("  {name}: IN FRAGMENT");
        } else {
            // Group the excluders by class: one line per class is what tells a
            // reader whether an extension or a separate model is needed.
            let mut store = Vec::new();
            let mut compute = Vec::new();
            let mut oof = Vec::new();
            let mut unknown = Vec::new();
            for e in &v.excluders {
                match e.class {
                    OpClass::Store => store.push(e.op.as_str()),
                    OpClass::Compute => compute.push(e.op.as_str()),
                    OpClass::OutOfFragment => oof.push(e.op.as_str()),
                    OpClass::Unknown => unknown.push(e.op.as_str()),
                    OpClass::Structural => {}
                }
            }
            println!("  {name}: outside ({} op(s))", v.excluders.len());
            if !store.is_empty() {
                println!("      store, no counterpart in the calculus: {}", store.join(" "));
            }
            if !compute.is_empty() {
                println!("      compute, one generic rule would cover: {}", compute.join(" "));
            }
            if !oof.is_empty() {
                println!("      needs its OWN model, not a bigger calculus: {}", oof.join(" "));
            }
            if !unknown.is_empty() {
                println!("      UNCLASSIFIED -- classify before claiming anything: {}", unknown.join(" "));
            }
        }
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;

    // Standalone and first: this reads only the artifact's op vocabulary, so it
    // must not require a device spec or a ptoas the caller may not have.
    if let Some(p) = args.fragment.as_ref() {
        let src = std::fs::read_to_string(p).map_err(|e| format!("read {}: {e}", p.display()))?;
        print_fragment(&p.display().to_string(), &src);
        return Ok(());
    }

    if let Some(s1_path) = args.stage1.as_ref() {
        let src = std::fs::read_to_string(s1_path)
            .map_err(|e| format!("read {}: {e}", s1_path.display()))?;
        let plan = parse_stage1(&src)?;
        let rep = check_all_stage1(&plan);
        print_safety(&s1_path.display().to_string(), &rep);
        return Ok(());
    }

    let spec = load_device(args.device.as_ref())?;

    let stage2 = if let Some(pto_in) = args.from_pto.as_ref() {
        ptoas_stage2_dump(&args.ptoas, pto_in)?
    } else {
        let p = args.input.as_ref().unwrap();
        std::fs::read_to_string(p).map_err(|e| format!("read {}: {e}", p.display()))?
    };

    let plan: Plan = parse_stage2(&stage2)?;
    let label = args
        .input
        .as_ref()
        .or(args.from_pto.as_ref())
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "<input>".into());

    let rep = check_all(&plan, &spec);
    print_safety(&label, &rep);

    if let Some(cmp) = args.compare.as_ref() {
        let src =
            std::fs::read_to_string(cmp).map_err(|e| format!("read {}: {e}", cmp.display()))?;
        let other = parse_stage2(&src)?;
        let diff = diff_plans(&plan, &other);
        print_diff(&diff);
    }

    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("pto-diff: {e}");
            eprintln!();
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
    }
}
