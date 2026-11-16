// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Finding the weights, and fetching them from Hugging Face when they are not on disk.
//!
//! # Why the transfer shells out
//!
//! The controller is pure Rust with ZERO third-party dependencies so it builds offline on the
//! box. A pure-Rust HTTPS client (rustls + an HTTP crate) would add a dozen crates the box's
//! registry does not carry, breaking exactly the offline build this design protects. So the
//! ORCHESTRATION is Rust — locating the model, resolving the HF URLs, resumable/verified
//! transfer, laying the shards out — and the byte transfer itself runs through the system
//! `curl` or `wget`, the same tools the box already used to fetch this model (`fetch.sh`). This
//! is a server CLI, not the sandboxed macOS app, so a system downloader is the right dependency.

use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

/// A known model: its local directory names and its Hugging Face repo.
pub struct ModelSpec {
    pub canonical: &'static str,
    /// File names that, if present in a search dir, mean the model is available.
    pub weight_globs: &'static [&'static str],
    pub hf_repo: &'static str,
    pub hf_files: &'static [&'static str],
}

/// The models this controller knows how to place.
pub fn known(name: &str) -> Result<&'static ModelSpec> {
    // The single-file MXFP4 export and the split (00001/00002) both count as present.
    const DS4: ModelSpec = ModelSpec {
        canonical: "deepseek-v4-flash",
        weight_globs: &[
            "DeepSeek-V4-Flash-MXFP4-merged.gguf",
            "DeepSeek-V4-Flash-MXFP4-00001-of-00002.gguf",
            "antirez-DS4-MXFP4.gguf",
        ],
        hf_repo: "antirez/deepseek-v4-gguf",
        hf_files: &[
            "DeepSeek-V4-Flash-MXFP4-00001-of-00002.gguf",
            "DeepSeek-V4-Flash-MXFP4-00002-of-00002.gguf",
        ],
    };
    let n = name.to_ascii_lowercase();
    if n.contains("deepseek") || n == "ds4" || n == "ds4-flash" {
        Ok(&DS4)
    } else {
        bail!("unknown model {name:?}; this controller places deepseek-v4-flash")
    }
}

/// Search roots, in order.
///
/// An explicit `--model-dir` is EXCLUSIVE — if the user names a directory, that is where the
/// model must be, and searching elsewhere would surprise them (and make "not found there" mean
/// "found somewhere else"). Only when no directory is named does the search fall back to the
/// conventional roots.
fn search_dirs(explicit: Option<&str>) -> Vec<PathBuf> {
    if let Some(d) = explicit {
        return vec![PathBuf::from(d)];
    }
    let mut v = Vec::new();
    if let Ok(env) = std::env::var("DS4_MODEL_DIR") {
        v.push(PathBuf::from(env));
    }
    for p in ["/data/models", "/data0/models", "/data/yuyijun/models", "/data/yuyijun/ds4-mxfp4"]
    {
        v.push(PathBuf::from(p));
    }
    if let Ok(home) = std::env::var("HOME") {
        v.push(PathBuf::from(home).join("models"));
    }
    v
}

/// Find a usable weight file for `spec`, or `None`.
pub fn locate(spec: &ModelSpec, model_dir: Option<&str>) -> Option<PathBuf> {
    for dir in search_dirs(model_dir) {
        for g in spec.weight_globs {
            let p = dir.join(g);
            if p.is_file() {
                return Some(p);
            }
        }
    }
    None
}

/// Which system downloader is available.
fn downloader() -> Option<(&'static str, &'static [&'static str])> {
    for (bin, pre) in [
        ("curl", &["-fL", "--retry", "3", "-C", "-", "-o"] as &[&str]),
        ("wget", &["-c", "-O"]),
    ] {
        if std::process::Command::new(bin)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
        {
            return Some((bin, pre));
        }
    }
    None
}

/// Download `spec`'s files into `dest`, resumable. Returns the primary (first) file's path.
///
/// The URL is the HF `resolve/main` endpoint, the same one the box's `fetch.sh` used. An
/// `HF_TOKEN` in the environment is passed as a bearer header for gated repos.
pub fn download(spec: &ModelSpec, dest: &Path) -> Result<PathBuf> {
    std::fs::create_dir_all(dest)
        .with_context(|| format!("creating model dir {}", dest.display()))?;
    let (bin, pre) = downloader().ok_or_else(|| {
        anyhow::anyhow!(
            "no `curl` or `wget` on PATH to fetch from Hugging Face — install one, or place \
             the weights under --model-dir manually"
        )
    })?;
    let token = std::env::var("HF_TOKEN").ok();
    let mut first: Option<PathBuf> = None;
    for f in spec.hf_files {
        let out = dest.join(f);
        if out.is_file() {
            eprintln!("ds4-rs-ascend: {f} already present, skipping");
            if first.is_none() {
                first = Some(out);
            }
            continue;
        }
        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}?download=true",
            spec.hf_repo, f
        );
        eprintln!("ds4-rs-ascend: fetching {f} via {bin} …");
        let mut cmd = std::process::Command::new(bin);
        if let Some(t) = &token {
            // curl and wget both accept a header flag; use the portable long forms.
            match bin {
                "curl" => {
                    cmd.arg("-H").arg(format!("Authorization: Bearer {t}"));
                }
                "wget" => {
                    cmd.arg(format!("--header=Authorization: Bearer {t}"));
                }
                _ => {}
            }
        }
        cmd.args(pre).arg(&out).arg(&url);
        let status = cmd
            .status()
            .with_context(|| format!("spawning {bin} for {f}"))?;
        if !status.success() {
            bail!("{bin} failed downloading {f} (status {status}); partial file kept for resume");
        }
        if first.is_none() {
            first = Some(out);
        }
    }
    first.ok_or_else(|| anyhow::anyhow!("model has no files to download"))
}

/// Locate the model, downloading if permitted and absent.
pub fn ensure_available(
    spec: &ModelSpec,
    model_dir: Option<&str>,
    allow_download: bool,
) -> Result<PathBuf> {
    if let Some(p) = locate(spec, model_dir) {
        return Ok(p);
    }
    if !allow_download {
        bail!(
            "{} not found in any search dir. Pass --model-dir <path>, or --download to fetch \
             from Hugging Face ({}).",
            spec.canonical,
            spec.hf_repo
        );
    }
    let dest = model_dir
        .map(PathBuf::from)
        .or_else(|| std::env::var("DS4_MODEL_DIR").ok().map(PathBuf::from))
        .unwrap_or_else(|| {
            std::env::var("HOME")
                .map(|h| PathBuf::from(h).join("models"))
                .unwrap_or_else(|_| PathBuf::from("./models"))
        });
    download(spec, &dest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_resolves_ds4_aliases() {
        assert_eq!(known("DeepSeek-V4-Flash").unwrap().canonical, "deepseek-v4-flash");
        assert_eq!(known("ds4").unwrap().canonical, "deepseek-v4-flash");
        assert!(known("llama-70b").is_err());
    }

    #[test]
    fn locate_finds_a_planted_file(/* uses a temp dir, no network */) {
        let spec = known("ds4").unwrap();
        let dir = std::env::temp_dir().join(format!("ds4npu-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("DeepSeek-V4-Flash-MXFP4-merged.gguf");
        std::fs::write(&f, b"GGUF").unwrap();
        let found = locate(spec, Some(dir.to_str().unwrap())).expect("planted file");
        assert_eq!(found, f);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn absent_without_download_is_a_clear_error() {
        let spec = known("ds4").unwrap();
        let empty = std::env::temp_dir().join("ds4npu-empty-does-not-exist-xyz");
        let e = ensure_available(spec, Some(empty.to_str().unwrap()), false)
            .unwrap_err()
            .to_string();
        assert!(e.contains("--download"), "{e}");
    }
}
