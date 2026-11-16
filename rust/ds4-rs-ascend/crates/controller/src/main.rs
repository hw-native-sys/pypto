// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! ds4-rs-ascend — the DeepSeek-V4-Flash-on-Ascend command controller.
//!
//! Pure Rust, no CLI framework (see `cli`), so it builds offline on the 910c/910b. It reads the
//! box's memory profile (`memprofile`), chooses a parallelism split for the user's mode through
//! the SAME chooser the engine uses (`plan`), finds or downloads the weights (`model`), and

mod cli;
mod fingerprint;
mod memprofile;
mod model;
mod netchan;
mod plan;
mod registry;
mod profile;
mod serve;

use anyhow::{bail, Result};
use cli::{Command, KernelsCmd};
use registry::{Provenance, Registry};

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = run(&argv) {
        eprintln!("ds4-rs-ascend: {e:#}");
        std::process::exit(1);
    }
}

fn run(argv: &[String]) -> Result<()> {
    let args = cli::parse(argv)?;
    match args.command {
        Command::Help { advanced } => {
            print!("{}", cli::help_text(advanced));
            Ok(())
        }
        Command::Version => {
            print!("{}", version_text());
            Ok(())
        }
        Command::Mem { source } => {
            let chips = memprofile::read_profile(source)?;
            print!("{}", render_mem(&chips));
            Ok(())
        }
        Command::Plan(opts) => {
            // Licence is REPORTED for plan, not required — capacity planning is free.
            let chips = memprofile::read_profile(opts.mem_source)?;
            let p = plan::build_plan(&opts, &chips)?;
            print!("{}", plan::render(&p));
            Ok(())
        }
        Command::Deploy(opts) => run_deploy(opts),
        Command::Profile(opts) => run_profile(opts),
        Command::Kernels(cmd) => run_kernels(cmd),
        Command::Update => run_update(),
        Command::Subscribe => run_subscribe(),
        Command::Share { id, yes } => run_share(&id, yes),
    }
}

fn run_kernels(cmd: KernelsCmd) -> Result<()> {
    let reg = Registry::open_default()?;
    match cmd {
        KernelsCmd::List => {
            let sets = reg.list()?;
            if sets.is_empty() {
                println!("no kernel sets yet. `ds4-rs-ascend update` for published sets, or `profile` to tune one.");
                return Ok(());
            }
            println!("{:<14} {:<9} {:<7} {:<8} {:>9}  {}", "id", "provenance", "active", "verified", "tok/s", "label");
            for s in sets {
                let tok = reg
                    .best_measurement(&s.id)?
                    .map(|t| format!("{t:.2}"))
                    .unwrap_or_else(|| "-".into());
                println!(
                    "{:<14} {:<9} {:<7} {:<8} {:>9}  {}{}",
                    s.id,
                    s.provenance.as_str(),
                    if s.active { "  *" } else { "" },
                    if s.verified { "yes" } else { "NO" },
                    tok,
                    s.label,
                    if s.bytes_present { "" } else { "  (evicted)" },
                );
            }
            Ok(())
        }
        KernelsCmd::Use(id) => {
            reg.set_active(&id)?;
            println!("active kernel set: {id}");
            Ok(())
        }
        KernelsCmd::Rm(id) => {
            reg.rm(&id)?;
            println!("removed {id}");
            Ok(())
        }
        KernelsCmd::Evict(id) => {
            reg.evict(&id)?;
            println!("evicted {id} (bytes freed; re-fetchable from the channel)");
            Ok(())
        }
        KernelsCmd::Verify(which) => {
            let sets = match which {
                Some(id) => vec![reg.get(&id)?.ok_or_else(|| anyhow::anyhow!("no set {id}"))?],
                None => reg.list()?,
            };
            let mut bad = 0;
            for s in &sets {
                match reg.verify(s) {
                    Ok(()) => println!("  {} OK", s.id),
                    Err(e) => {
                        bad += 1;
                        println!("  {} FAILED: {e}", s.id);
                    }
                }
            }
            if bad > 0 {
                bail!("{bad} set(s) failed verification");
            }
            println!("all {} set(s) verified", sets.len());
            Ok(())
        }
    }
}

fn run_update() -> Result<()> {
    let url = netchan::require_channel()?;
    let reg = Registry::open_default()?;
    let added = pull_index(&reg, &url, None)?;

        // Trusted build: silent when nothing is new, concise when there is.
        if added > 0 {
            println!("update: {added} new kernel set(s) available  (ds4-rs-ascend kernels use <id>)");
        }
    Ok(())
}

fn run_subscribe() -> Result<()> {
    let url = netchan::require_channel()?;
    let reg = Registry::open_default()?;
    let soc = fingerprint::derive(None).soc_arch;
    let added = pull_index(&reg, &url, Some(&soc))?;
    println!("subscribe: {added} community set(s) for soc_arch={soc} registered (verified, inactive).");
    println!("  they never auto-activate — choose one with:  ds4-rs-ascend kernels use <id>");
    Ok(())
}

/// Fetch + verify the signed index and register its sets as published (verified, inactive).
/// `soc_filter` restricts to one soc_arch (subscribe, D9); None takes all (update).
fn pull_index(reg: &Registry, url: &str, soc_filter: Option<&str>) -> Result<usize> {
    let index = netchan::fetch(&format!("{url}/index.json"))?;
    let sig = netchan::fetch(&format!("{url}/index.json.sig"))?;
    let sig_hex = String::from_utf8_lossy(&sig).trim().to_string();
    // Whole-index signature MUST verify against DS4_PUBLISHER_PUBKEY before anything is trusted (D4).
    netchan::verify_index(&index, &sig_hex)
        .map_err(|e| anyhow::anyhow!("published index rejected: {e}"))?;
    let sets = netchan::parse_index(&index)?;
    let mut n = 0;
    for set in sets {
        if let Some(f) = soc_filter {
            if set.soc_arch != f {
                continue;
            }
        }
        // Per-set signature is also checked, so a tampered single entry cannot slip through.
        if netchan::verify_set(&set.config_json, &set.sig).is_err() {
            eprintln!("  skipping {}: per-set signature did not verify", set.id);
            continue;
        }
        let label = match set.community_tok_s {
            Some(t) => format!("{} (community {t:.2} tok/s)", set.label),
            None => set.label.clone(),
        };
        reg.register(Provenance::Publisher, &label, &set.soc_arch, &set.config_json, Some(url), true)?;
        n += 1;
    }
    Ok(n)
}

fn run_share(id: &str, yes: bool) -> Result<()> {
    let reg = Registry::open_default()?;
    let set = reg.get(id)?.ok_or_else(|| anyhow::anyhow!("no set {id}"))?;
    // Preconditions (decision 11): profiled-only + a real local measurement.
    if set.provenance != Provenance::Profiled {
        bail!("share accepts only self-profiled sets; {id} is {}", set.provenance.as_str());
    }
    let tok = reg
        .best_measurement(id)?
        .ok_or_else(|| anyhow::anyhow!("{id} has no local measurement; a share must carry a real tok/s"))?;
    let install = reg.install_id()?;
    // The what-leaves confirmation IS the privacy policy at the point of action.
    println!("share → the upload will contain:");
    println!("  soc_class : {}", set.soc_arch);
    println!("  knobs     : {}", registry::canonical_config(&set.config_json));
    println!("  measured  : {tok:.2} tok/s");
    println!("  install-id: {install}  (NOT your email)");
    if !yes {
        bail!("re-run with --yes to upload, or omit to keep it local. Nothing was sent.");
    }
    let url = netchan::require_channel()?;
    // Flat submission payload (D7/D11): soc class, the config object verbatim, the measured
    // tok/s, and the RANDOM install-id — never the email.
    let soc_name = fingerprint::derive(None).soc_name;
    let body = format!(
        "{{\"soc_arch\":\"{}\",\"soc_name\":\"{}\",\"config\":{},\"tok_s\":{},\"install_id\":\"{}\",\"client_ver\":\"{}\"}}",
        set.soc_arch,
        soc_name,
        registry::canonical_config(&set.config_json),
        tok,
        install,
        env!("CARGO_PKG_VERSION"),
    );
    let resp = netchan::post_json(&format!("{url}/submit"), &body)?;
    println!("uploaded; the publisher acknowledged: {}", String::from_utf8_lossy(&resp).trim());
    println!("(the publisher re-signs before any redistribution; it is not yet trusted by other users.)");
    Ok(())
}


/// Report a W8A8 deployment package through the engine's own reader.
///
/// This is the engine READING the package: layers, dimensions, per-role dtypes and the byte
/// count, all derived from the tensors rather than a config file that may disagree with them.
/// Serving it needs the builder (package -> the structures `decode_step` consumes), which is
/// not written yet, so this reports and refuses rather than pretending to generate.
fn report_w8a8_package(dir: &std::path::Path) -> Result<()> {
    use engine_core::safetensors::Package;
    let pkg = Package::open(dir)?;
    println!("model: {}  (W8A8 deployment package)", dir.display());
    println!("tensors in index: {}", pkg.n_tensors());

    let mut layers = std::collections::BTreeSet::new();
    for name in pkg.names() {
        if let Some(rest) = name.strip_prefix("layers.") {
            if let Some((idx, _)) = rest.split_once('.') {
                if let Ok(n) = idx.parse::<usize>() {
                    layers.insert(n);
                }
            }
        }
    }
    let emb = pkg.tensor("embed.weight")?.shape;
    let wq_a = pkg.tensor("layers.0.attn.wq_a.weight")?.shape;
    let wq_b = pkg.tensor("layers.0.attn.wq_b.weight")?.shape;
    let wkv = pkg.tensor("layers.0.attn.wkv.weight")?.shape;
    let w1 = pkg.tensor("layers.0.ffn.experts.0.w1.weight")?.shape;
    let n_experts = pkg
        .names()
        .iter()
        .filter(|s| s.starts_with("layers.0.ffn.experts.") && s.ends_with(".w1.weight"))
        .count();
    println!(
        "  layers {}  vocab {}  d_model {}  q_lora {}  kv_lora {}  q_dim {}  expert_ffn {}  experts {}",
        layers.len(), emb[0], emb[1], wq_a[0], wkv[0], wq_b[0], w1[0], n_experts
    );
    println!(
        "  quantisation: W8A8 (int8 weights, per-output-channel scale/offset); \
         wo_a/wo_b/gate/embed are BF16, head is F32"
    );
    anyhow::bail!(
        "the package is READABLE but not yet SERVEABLE: the builder that maps its tensors onto \
         the structures `decode_step` consumes is not written. `kernels verify` and the projection \
         kernels are certified; what is missing is the wiring, not the arithmetic."
    );
}

fn run_deploy(opts: cli::DeployOpts) -> Result<()> {
    // 0. A W8A8 package directory is a different container from GGUF; recognise it before the
    //    GGUF locator, which would otherwise search for a .gguf that is not there.
    if let Some(d) = opts.model_dir.as_deref() {
        let p = std::path::Path::new(d);
        if p.join("quant_model_description.json").is_file() {
            return report_w8a8_package(p);
        }
    }

    // 1. Locate or fetch the weights.
    let spec = model::known(&opts.model)?;
    let weights =
        model::ensure_available(spec, opts.model_dir.as_deref(), opts.allow_download)?;
    println!("model: {}", weights.display());

    // 2. Profile the box and plan the split -- unless asked not to.
    if opts.no_plan {
        println!(
            "--no-plan: skipping the memory profile and the split. This serves \n\
             single-stream on device 0; the multi-chip plan is the throughput path and is \n\
             not wired into the engine, so nothing measured below depends on it."
        );
    } else {
        let chips = memprofile::read_profile(opts.mem_source)?;
        let p = plan::build_plan(&opts, &chips)?;
        print!("{}", plan::render(&p));
    }

    // 3. Serve: open the model and generate. Single-stream on device 0 (weights
    //    resident if they fit host RAM, else SSD-streamed); the planned split
    //    above is the multi-chip throughput path, not yet wired into the engine.
    serve::run(&weights, opts.prompt.as_deref(), opts.max_new)
}

fn run_profile(opts: cli::ProfileOpts) -> Result<()> {
    print!("{}", profile::describe(&opts));
    match profile::run(&opts) {
        Ok((msg, config_json)) => {
            println!("{msg}");
            // Register the tuned config as a `profiled` set, tagged with this box's fingerprint
            // (decisions 1/2/5), so it appears in `kernels list` and can be activated or shared.
            let soc = memprofile::read_profile(memprofile::MemSource::Auto)
                .ok()
                .and_then(|_| None::<String>); // chip name isn't in the profile; derive from env
            let fp = fingerprint::derive(soc.as_deref());
            match Registry::open_default() {
                Ok(reg) => {
                    let label = format!("profiled {} / {}", short_date(), fp.note());
                    let id = reg.register(
                        Provenance::Profiled,
                        &label,
                        &fp.soc_arch,
                        &config_json,
                        None,
                        true, // self-authored: trusted locally
                    )?;
                    let _ = reg.record_fingerprint(
                        &fp.full(),
                        &fp.soc_arch,
                        &fp.soc_name,
                        &fp.cann_version,
                        &fp.ptoas_version,
                        &fp.note(),
                    );
                    println!("registered as kernel set {id}  ({label})");
                    println!("  activate it with:  ds4-rs-ascend kernels use {id}");
                }
                Err(e) => eprintln!("ds4-rs-ascend: tuned config written but not registered: {e}"),
            }
            Ok(())
        }
        Err(e) => {
            // A refusal from the tuning step is not a crash — explain and exit
            // non-zero so a caller can tell the two apart.
            eprintln!("ds4-rs-ascend: {e}");
            std::process::exit(2);
        }
    }
}

/// UTC yyyy-mm-dd from the epoch, no chrono dependency — days since epoch back to a date.
fn short_date() -> String {
    let days = (std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
        / 86_400) as i64;
    // Civil-from-days (Howard Hinnant's algorithm), UTC.
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}

fn render_mem(chips: &[memprofile::ChipMem]) -> String {
    let mut s = String::from("Ascend memory profile\n=====================\n");
    let (mut free, mut total) = (0u64, 0u64);
    for c in chips {
        s.push_str(&format!(
            "  chip {:>2}:  {:6.1} GiB free / {:6.1} GiB  ({:.0}% used)\n",
            c.id,
            c.free_bytes as f64 / (1u64 << 30) as f64,
            c.total_bytes as f64 / (1u64 << 30) as f64,
            c.busy_fraction() * 100.0,
        ));
        free += c.free_bytes;
        total += c.total_bytes;
    }
    s.push_str(&format!(
        "  ------\n  total:    {:6.1} GiB free / {:6.1} GiB across {} chip(s)\n",
        free as f64 / (1u64 << 30) as f64,
        total as f64 / (1u64 << 30) as f64,
        chips.len(),
    ));
    s
}

fn version_text() -> String {
    format!("ds4-rs-ascend {}\n", env!("CARGO_PKG_VERSION"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mem_render_totals() {
        let chips = vec![
            memprofile::ChipMem { id: 0, free_bytes: 60 << 30, total_bytes: 64 << 30 },
            memprofile::ChipMem { id: 1, free_bytes: 4 << 30, total_bytes: 64 << 30 },
        ];
        let s = render_mem(&chips);
        assert!(s.contains("chip  0") && s.contains("chip  1"));
        assert!(s.contains("64.0 GiB free"), "total free line:\n{s}");
    }

    /// The version line identifies the build and nothing else. The words the
    /// guard refuses are asserted absent BY NAME: a re-added banner is exactly
    /// the regression that is invisible unless something looks for it.
    #[test]
    fn version_names_the_build() {
        let v = version_text();
        assert!(v.contains("ds4-rs-ascend"), "{v}");
        assert!(v.contains(env!("CARGO_PKG_VERSION")), "{v}");
        for word in ["licence", "license", "edition", "Pro"] {
            assert!(!v.contains(word), "the version line still mentions {word:?}: {v}");
        }
    }
}
