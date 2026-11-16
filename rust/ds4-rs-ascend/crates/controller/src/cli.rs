// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Hand-rolled argument parsing — no CLI framework, so the binary builds offline on the box.
//!
//! Two tiers of help by design: `ds4-rs-ascend help` shows the everyday options a deploying user
//! needs; `ds4-rs-ascend help --advanced` additionally lists the expert knobs (explicit split pins,
//! the profile/tuning step, memory-source override). Advanced flags still WORK without
//! `--advanced` — it only governs whether they are documented, so a support engineer can direct
//! a user to one without it being noise for everyone else.

use anyhow::{bail, Result};

/// Deployment intent — the one choice most users make.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Lowest latency for a single user. Maps to `SingleStreamLatency`.
    Latency,
    /// Most total tokens/sec across many users. Maps to `AggregateThroughput`.
    Throughput,
    /// A middle ground — good per-user latency while still serving several. Maps to a mixed
    /// EPxPP split under the latency objective (the measured `ep4 pp2`-class point).
    Balanced,
    /// The longest context the box can serve at all. Maps to `MaxContext`.
    MaxContext,
}

impl Mode {
    pub fn parse(s: &str) -> Result<Mode> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "latency" | "single" | "single-stream" => Mode::Latency,
            "throughput" | "aggregate" | "serving" => Mode::Throughput,
            "balanced" | "balance" | "mixed" => Mode::Balanced,
            "max-context" | "maxctx" | "context" => Mode::MaxContext,
            other => bail!(
                "unknown mode {other:?}; use one of: latency, throughput, balanced, max-context"
            ),
        })
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Mode::Latency => "latency",
            Mode::Throughput => "throughput",
            Mode::Balanced => "balanced",
            Mode::MaxContext => "max-context",
        }
    }
}

/// The parsed command line.
#[derive(Debug, Clone)]
pub struct Args {
    pub command: Command,
}

#[derive(Debug, Clone)]
pub enum Command {
    /// Print the plan (memory profile + chosen split) without launching.
    Plan(DeployOpts),
    /// Plan, then serve. (Serving itself lands with the engine; today this reports what it
    /// WOULD run and refuses rather than pretending to serve.)
    Deploy(DeployOpts),
    /// Just the memory profile of this box.
    Mem { source: crate::memprofile::MemSource },
    /// The kernel-tuning step.
    Profile(ProfileOpts),
    /// Help text. `advanced` toggles the expert section.
    Help { advanced: bool },
    /// Version and build provenance.
    Version,
    /// Local kernel-set management (free).
    Kernels(KernelsCmd),
    /// Pull the publisher's latest signed optimized sets.
    Update,
    /// Upload a self-profiled set to the publisher. Needs $DS4_PUBLISHER_URL.
    /// `yes` skips the confirmation.
    Share { id: String, yes: bool },
    /// Pull the community's ranked configs for this SoC. Needs $DS4_PUBLISHER_URL.
    Subscribe,
}

#[derive(Debug, Clone)]
pub enum KernelsCmd {
    /// List sets in the registry with provenance, verification, and best measured tok/s.
    List,
    /// Make a set the sticky default (verifies integrity first).
    Use(String),
    /// Destroy a self-profiled set.
    Rm(String),
    /// Free a published set's local bytes, keeping the re-fetchable stub.
    Evict(String),
    /// Verify a set's integrity (all sets if no id).
    Verify(Option<String>),
}


#[derive(Debug, Clone)]
pub struct DeployOpts {
    pub mode: Mode,
    pub model: String,
    /// Where to look for / download weights.
    pub model_dir: Option<String>,
    /// Download from Hugging Face if the model is not found locally.
    pub allow_download: bool,
    /// Context length to plan for (KV sizing).
    pub max_ctx: u32,
    /// Advanced: force an exact split, e.g. `ep4pp2`. Bypasses the mode ranking; refused if
    /// infeasible, never silently substituted.
    pub parallel_pin: Option<String>,
    /// Advanced: which memory source to read.
    pub mem_source: crate::memprofile::MemSource,
    /// Advanced: expert weight layout (packed 4-bit vs host-expanded int8).
    pub four_bit: bool,
    /// Serve: a one-shot prompt to answer (absent → interactive stdin loop).
    pub prompt: Option<String>,
    /// Serve: maximum new tokens to generate per prompt.
    pub max_new: usize,
    /// Serve WITHOUT planning the split.
    ///
    /// Planning reads `npu-smi`, so on a box with no Ascend driver `deploy` could not
    /// serve at all -- even though the single-stream path it falls into does not use the
    /// plan. This skips step 2 and says so; the plan is the multi-chip throughput path and
    /// is not wired into the engine anyway.
    pub no_plan: bool,
}

#[derive(Debug, Clone)]
pub struct ProfileOpts {
    pub model: String,
    pub mode: Mode,
    /// How long to spend tuning, in seconds (bounds the knob sweep).
    pub budget_s: u32,
    pub out: Option<String>,
}

impl Default for DeployOpts {
    fn default() -> Self {
        DeployOpts {
            mode: Mode::Balanced,
            model: "deepseek-v4-flash".to_string(),
            model_dir: None,
            allow_download: false,
            max_ctx: 8192,
            parallel_pin: None,
            mem_source: crate::memprofile::MemSource::Auto,
            four_bit: true,
            prompt: None,
            max_new: 256,
            no_plan: false,
        }
    }
}

/// One flag=value or flag-then-value, from an iterator that has already consumed the subcommand.
fn take_val<'a>(
    flag: &str,
    it: &mut std::iter::Peekable<impl Iterator<Item = String>>,
    inline: Option<&'a str>,
) -> Result<String> {
    if let Some(v) = inline {
        return Ok(v.to_string());
    }
    match it.next() {
        Some(v) if !v.starts_with('-') => Ok(v),
        _ => bail!("flag {flag} needs a value"),
    }
}

/// Split `--flag=value` into `(flag, Some(value))`, else `(flag, None)`.
fn split_eq(arg: &str) -> (String, Option<String>) {
    match arg.split_once('=') {
        Some((f, v)) => (f.to_string(), Some(v.to_string())),
        None => (arg.to_string(), None),
    }
}

pub fn parse(argv: &[String]) -> Result<Args> {
    let mut it = argv.iter().cloned().peekable();
    let sub = match it.next() {
        Some(s) => s,
        None => return Ok(Args { command: Command::Help { advanced: false } }),
    };
    let command = match sub.as_str() {
        "help" | "-h" | "--help" => {
            let advanced = it.any(|a| a == "--advanced" || a == "-a");
            Command::Help { advanced }
        }
        "version" | "-V" | "--version" => Command::Version,
        "mem" | "profile-mem" => {
            let mut source = crate::memprofile::MemSource::Auto;
            while let Some(arg) = it.next() {
                let (f, v) = split_eq(&arg);
                match f.as_str() {
                    "--mem-source" => {
                        source = parse_mem_source(&take_val("--mem-source", &mut it, v.as_deref())?)?
                    }
                    other => bail!("mem: unknown flag {other}"),
                }
            }
            Command::Mem { source }
        }
        "profile" => Command::Profile(parse_profile(&mut it)?),
        "plan" => Command::Plan(parse_deploy(&mut it)?),
        "deploy" | "run" => Command::Deploy(parse_deploy(&mut it)?),
        "kernels" | "kernel" => {
            let action = it.next().unwrap_or_else(|| "list".to_string());
            let mut arg = || -> Result<String> {
                it.next()
                    .filter(|a| !a.starts_with('-'))
                    .ok_or_else(|| anyhow::anyhow!("kernels: this action needs a set id"))
            };
            Command::Kernels(match action.as_str() {
                "list" | "ls" | "" => KernelsCmd::List,
                "use" | "activate" => KernelsCmd::Use(arg()?),
                "rm" | "remove" | "delete" => KernelsCmd::Rm(arg()?),
                "evict" => KernelsCmd::Evict(arg()?),
                "verify" => KernelsCmd::Verify(it.next().filter(|a| !a.starts_with('-'))),
                o => bail!("kernels: unknown action {o:?} (list|use|rm|evict|verify)"),
            })
        }
        "update" => Command::Update,
        "subscribe" => Command::Subscribe,
        "share" => {
            let (mut id, mut yes) = (None, false);
            while let Some(arg) = it.next() {
                match arg.as_str() {
                    "--yes" | "-y" => yes = true,
                    a if !a.starts_with('-') && id.is_none() => id = Some(a.to_string()),
                    o => bail!("share: unexpected {o}"),
                }
            }
            Command::Share {
                id: id.ok_or_else(|| anyhow::anyhow!("share needs a set id"))?,
                yes,
            }
        }
        other => bail!("unknown command {other:?}; run `ds4-rs-ascend help`"),
    };
    Ok(Args { command })
}

fn parse_mem_source(s: &str) -> Result<crate::memprofile::MemSource> {
    use crate::memprofile::MemSource;
    Ok(match s.to_ascii_lowercase().as_str() {
        "auto" => MemSource::Auto,
        "npu-smi" | "smi" => MemSource::NpuSmi,
        "acl" => MemSource::Acl,
        o => bail!("unknown mem-source {o:?} (auto | npu-smi | acl)"),
    })
}

fn parse_deploy(
    it: &mut std::iter::Peekable<impl Iterator<Item = String>>,
) -> Result<DeployOpts> {
    let mut o = DeployOpts::default();
    while let Some(arg) = it.next() {
        let (f, v) = split_eq(&arg);
        match f.as_str() {
            "--mode" => o.mode = Mode::parse(&take_val("--mode", it, v.as_deref())?)?,
            "--model" => o.model = take_val("--model", it, v.as_deref())?,
            "--model-dir" => o.model_dir = Some(take_val("--model-dir", it, v.as_deref())?),
            "--download" => o.allow_download = true,
            "--ctx" | "--max-ctx" => {
                o.max_ctx = take_val("--ctx", it, v.as_deref())?
                    .parse()
                    .map_err(|_| anyhow::anyhow!("--ctx needs an integer"))?
            }
            // Advanced (hidden from the basic help).
            "--parallel" => o.parallel_pin = Some(take_val("--parallel", it, v.as_deref())?),
            "--mem-source" => {
                o.mem_source = parse_mem_source(&take_val("--mem-source", it, v.as_deref())?)?
            }
            "--int8" => o.four_bit = false,
            "--4bit" => o.four_bit = true,
            "--no-plan" => o.no_plan = true,
            "--prompt" => o.prompt = Some(take_val("--prompt", it, v.as_deref())?),
            "--max" | "--max-new" => {
                o.max_new = take_val("--max", it, v.as_deref())?
                    .parse()
                    .map_err(|_| anyhow::anyhow!("--max needs an integer"))?
            }
            other => bail!("deploy: unknown flag {other} (see `ds4-rs-ascend help --advanced`)"),
        }
    }
    Ok(o)
}

fn parse_profile(
    it: &mut std::iter::Peekable<impl Iterator<Item = String>>,
) -> Result<ProfileOpts> {
    let mut o = ProfileOpts {
        model: "deepseek-v4-flash".to_string(),
        mode: Mode::Balanced,
        budget_s: 60,
        out: None,
    };
    while let Some(arg) = it.next() {
        let (f, v) = split_eq(&arg);
        match f.as_str() {
            "--model" => o.model = take_val("--model", it, v.as_deref())?,
            "--mode" => o.mode = Mode::parse(&take_val("--mode", it, v.as_deref())?)?,
            "--budget" | "--seconds" => {
                o.budget_s = take_val("--budget", it, v.as_deref())?
                    .parse()
                    .map_err(|_| anyhow::anyhow!("--budget needs an integer"))?
            }
            "--out" => o.out = Some(take_val("--out", it, v.as_deref())?),
            other => bail!("profile: unknown flag {other}"),
        }
    }
    Ok(o)
}

/// The two-tier help text.
pub fn help_text(advanced: bool) -> String {
    let mut s = String::new();
    s.push_str(
        "ds4-rs-ascend — deploy DeepSeek-V4-Flash on Huawei Ascend NPUs\n\
         \n\
         USAGE:\n\
         \x20 ds4-rs-ascend <command> [options]\n\
         \n\
         COMMANDS:\n\
         \x20 deploy     Plan for this box and serve the model\n\
         \x20 plan       Show the plan (memory profile + parallelism) without serving\n\
         \x20 mem        Show this box's per-chip free HBM\n\
         \x20 kernels    Manage local kernel sets: list | use <id> | verify\n\
         \x20 update     Fetch the publisher's latest optimized kernel sets\n\
         \x20 profile    Tune kernels for this box\n\
         \x20 version    Build provenance\n\
         \x20 help       This text  (help --advanced for expert options)\n\
         \n\
         COMMON OPTIONS (deploy / plan):\n\
         \x20 --mode <m>       latency | throughput | balanced | max-context   [balanced]\n\
         \x20                    latency     one user, lowest delay\n\
         \x20                    throughput  many users, most total tokens/sec\n\
         \x20                    balanced    good per-user latency while serving several\n\
         \x20                    max-context the longest prompt the box can hold\n\
         \x20 --model <name>   model to deploy                       [deepseek-v4-flash]\n\
         \x20 --model-dir <d>  where weights live / are downloaded to\n\
         \x20 --download       fetch from Hugging Face if not found locally\n\
         \x20 --ctx <n>        context length to plan for            [8192]\n\
         \n\
         EXAMPLES:\n\
         \x20 ds4-rs-ascend mem\n\
         \x20 ds4-rs-ascend plan --mode latency --ctx 32768\n\
         \x20 ds4-rs-ascend deploy --mode balanced --model-dir /data/models --download\n",
    );
    if advanced {
        s.push_str(
            "\n\
             ADVANCED OPTIONS (expert use):\n\
             \x20 --parallel <spec>   force an exact split, e.g. ep4pp2, tp1dp1ep8pp1.\n\
             \x20                       Bypasses --mode; REFUSED if it does not fit (never\n\
             \x20                       silently substituted). Same syntax as DS4_PARALLEL.\n\
             \x20 --mem-source <s>    auto | npu-smi | acl                    [auto]\n\
             \x20                       acl reads aclrtGetMemInfo (needs the ascend build);\n\
             \x20                       npu-smi parses `npu-smi info` with no CANN linkage.\n\
             \x20 --int8 / --4bit     expert weight layout: host-expanded int8 (faster,\n\
             \x20                       more HBM) vs packed 4-bit (fits smaller boxes) [4bit]\n\
             \n\
             PROFILE:\n\
             \x20 ds4-rs-ascend profile --model M --mode balanced --budget 120 --out tune.json\n\
             \x20   Sweeps tile-rs kernel knobs on THIS silicon and writes a tuning file the\n\
             \x20   deploy step then loads.\n\
             \x20   print what it would tune and stop.\n\
             \n\
             KERNEL REGISTRY (advanced):\n\
             \x20 kernels list              provenance, verification, best measured tok/s\n\
             \x20 kernels use <id>          set the sticky default (verifies integrity first)\n\
             \x20 kernels rm <id>           destroy a self-profiled set (profiled only)\n\
             \x20 kernels evict <id>        free a published set's bytes, keep the re-fetchable stub\n\
             \x20 kernels verify [<id>]     recompute hashes\n\
             \x20 subscribe                 pull the community's ranked configs (needs $DS4_PUBLISHER_URL)\n\
             \x20 share <id> [--yes]        upload a self-profiled set (needs $DS4_PUBLISHER_URL)\n\
             \x20                             prints exactly what leaves the box first\n\
             \n\
             ENVIRONMENT:\n\
             \x20 DS4_PARALLEL        same effect as --parallel\n\
             \x20 DS4_SSD_STREAM=1    stream weights from SSD when they exceed free HBM\n\
             \x20 DS4_PUBLISHER_URL       https signed-index endpoint for update/subscribe/share\n\
             \x20 HF_TOKEN            bearer token for gated Hugging Face model repos\n",
        );
    } else {
        s.push_str("\nRun `ds4-rs-ascend help --advanced` for expert options (explicit splits, kernel tuning).\n");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    fn v(a: &[&str]) -> Vec<String> {
        a.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn no_args_is_help() {
        assert!(matches!(parse(&[]).unwrap().command, Command::Help { advanced: false }));
    }

    #[test]
    fn mode_maps_and_ctx_parses() {
        let a = parse(&v(&["plan", "--mode", "latency", "--ctx", "32768"])).unwrap();
        match a.command {
            Command::Plan(o) => {
                assert_eq!(o.mode, Mode::Latency);
                assert_eq!(o.max_ctx, 32768);
            }
            _ => panic!("expected plan"),
        }
    }

    #[test]
    fn eq_form_and_space_form_agree() {
        let a = parse(&v(&["deploy", "--mode=throughput"])).unwrap();
        let b = parse(&v(&["deploy", "--mode", "throughput"])).unwrap();
        matches!((a.command, b.command), (Command::Deploy(_), Command::Deploy(_)));
    }

    #[test]
    fn advanced_flags_work_without_the_advanced_help() {
        let a = parse(&v(&["deploy", "--parallel", "ep4pp2", "--int8"])).unwrap();
        match a.command {
            Command::Deploy(o) => {
                assert_eq!(o.parallel_pin.as_deref(), Some("ep4pp2"));
                assert!(!o.four_bit);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn advanced_help_is_a_superset() {
        let basic = help_text(false);
        let adv = help_text(true);
        assert!(!basic.contains("--parallel"), "advanced flag leaked into basic help");
        assert!(adv.contains("--parallel") && adv.contains("profile"));
        assert!(basic.contains("--mode") && adv.contains("--mode"), "common opts in both");
    }

    #[test]
    fn unknown_flag_is_an_error_pointing_at_advanced_help() {
        let e = parse(&v(&["deploy", "--wat"])).unwrap_err().to_string();
        assert!(e.contains("--advanced"), "{e}");
    }

    /// A verb that is not in the command list is an error, not a silent
    /// something-else. Pinned on two spellings that a stray document could
    /// reintroduce, so the parser cannot grow a command by accident.
    #[test]
    fn unknown_verbs_are_refused() {
        assert!(parse(&v(&["license", "status"])).is_err());
        assert!(parse(&v(&["licence", "status"])).is_err());
    }
}
