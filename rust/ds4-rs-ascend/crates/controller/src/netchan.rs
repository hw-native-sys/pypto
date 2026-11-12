// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! The publisher channel for `update` / `subscribe` / `share`.
//!
//! Trust is an Ed25519-signed set index, the same discipline `ds4_update` uses for binary
//! updates: the index is refused unless its detached signature verifies against the publisher
//! PUBLISHING key baked in at build time, and only https URLs are accepted. Verify is real and
//! tested here; the byte transfer shells out to curl/wget (a server CLI, like the model
//! download), and every command refuses cleanly when no endpoint is configured or reachable —
//! there is no live the publisher server to pretend at.

use anyhow::{bail, Context, Result};
// `verify_strict` is an inherent method, so the `Verifier` trait is not needed.
use ed25519_dalek::{Signature, VerifyingKey};

/// Dev publishing key — the secret half lives in the repo test fixtures, distinct from any
/// release key, so a mis-built binary cannot accept real the publisher indexes or vice versa.
const DEV_PUBLISHER_PUBKEY: &str = "3b6a27bcceb6a42d62a3a8d02a6f0d73653215771de243a63ac048a18b59da29";

pub fn publisher_pubkey_hex() -> &'static str {
    option_env!("DS4_PUBLISHER_PUBKEY").unwrap_or(DEV_PUBLISHER_PUBKEY)
}

/// Base URL for the kernel index/repo. Refuses non-https, EXCEPT an explicit loopback for local
/// testing (mirrors ds4_server::update_cmd's carve-out): a manifest fetched in the clear over
/// the public internet could be swapped, but 127.0.0.1 is the developer's own server, and it is
/// how the end-to-end loop is exercised without deploying to production.
pub fn base_url() -> Result<String> {
    // NO FALLBACK. There was one, and it was three defects in a row:
    //
    //   * the baked default named a host as `<the> <publisher>.ai` -- two words
    //     with a SPACE between them, left by a blanket rename of an internal
    //     service name. It could never have resolved. The literal is not
    //     reproduced here on purpose: a test refuses that spelling anywhere in
    //     this tree, and a comment quoting it would be the one thing keeping
    //     the pattern alive;
    //   * it contradicted the design this module documents a few lines down,
    //     "only a build with neither refuses, rather than guessing". The
    //     fallback guessed, and `require_channel` then re-derived the same
    //     decision a second way;
    //   * worst, it broke the DIAGNOSTIC ORDER. `require_channel` checked for
    //     curl/wget first, so a box without either was told "no curl or wget on
    //     PATH to reach <that broken URL>" -- naming a transport problem and a
    //     nonexistent host, when the real state was that no endpoint had been
    //     configured at all.
    //
    // So the configured-ness decision lives here, once, where every caller of
    // `base_url` gets it.
    let url = std::env::var("DS4_PUBLISHER_URL")
        .ok()
        .or_else(|| option_env!("DS4_PUBLISHER_URL").map(str::to_string));
    let Some(url) = url else {
        bail!(
            "the publisher kernel channel is not configured on this build. Set \
             DS4_PUBLISHER_URL to the signed index endpoint to enable \
             update/subscribe/share; without it the command refuses rather than \
             guessing an endpoint."
        );
    };
    let loopback = url.starts_with("http://127.0.0.1") || url.starts_with("http://localhost");
    if !url.starts_with("https://") && !loopback {
        bail!("refusing a non-https publisher URL: {url}");
    }
    Ok(url)
}

/// One set as it appears in the signed index (the client-side view of a server IndexEntry).
#[derive(Debug, Clone)]
pub struct IndexSet {
    pub id: String,
    pub soc_arch: String,
    pub label: String,
    pub config_json: String,
    pub sig: String,
    pub community_tok_s: Option<f64>,
}

/// GET a URL into memory via curl/wget (transport only; verification is the caller's job).
pub fn fetch(url: &str) -> Result<Vec<u8>> {
    let (bin, args): (&str, Vec<String>) = if have("curl") {
        ("curl", vec!["-fsSL".into(), "-m".into(), "60".into(), url.into()])
    } else if have("wget") {
        ("wget", vec!["-qO-".into(), url.into()])
    } else {
        bail!("no curl or wget on PATH to reach {url}");
    };
    let out = std::process::Command::new(bin)
        .args(&args)
        .output()
        .with_context(|| format!("spawning {bin} for {url}"))?;
    if !out.status.success() {
        bail!("{bin} failed fetching {url} (status {})", out.status);
    }
    Ok(out.stdout)
}

/// POST a body to a URL via curl. Returns the response body.
pub fn post_json(url: &str, body: &str) -> Result<Vec<u8>> {
    if !have("curl") {
        bail!("share needs curl on PATH to POST to {url}");
    }
    let out = std::process::Command::new("curl")
        .args(["-fsSL", "-m", "60", "-X", "POST", "-H", "Content-Type: application/json", "--data-binary", body, url])
        .output()
        .with_context(|| format!("spawning curl POST to {url}"))?;
    if !out.status.success() {
        bail!("curl POST to {url} failed (status {})", out.status);
    }
    Ok(out.stdout)
}

fn have(bin: &str) -> bool {
    std::process::Command::new(bin)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Verify a per-set signature (over the canonical config) against the publisher's key.
/// This is what lets a subscribed set be trusted enough to register as `verified`.
pub fn verify_set(config_json: &str, sig_hex: &str) -> Result<()> {
    let canon = crate::registry::canonical_config(config_json);
    verify_index(canon.as_bytes(), sig_hex)
}

/// Parse the server's `{"sets":[{id,soc_arch,label,config,sig,community_tok_s,n_samples}]}` index.
/// Hand-parsed (no serde) to match the rest of the crate; the shape is fixed and server-emitted.
pub fn parse_index(bytes: &[u8]) -> Result<Vec<IndexSet>> {
    let s = std::str::from_utf8(bytes).context("index is not UTF-8")?;
    let mut sets = Vec::new();
    // Walk object-by-object at the top level of the "sets" array by brace matching.
    let arr_start = s.find("\"sets\"").and_then(|i| s[i..].find('[').map(|j| i + j + 1));
    let Some(mut pos) = arr_start else {
        return Ok(sets);
    };
    let bytes = s.as_bytes();
    while pos < bytes.len() {
        // find next '{' or ']'
        while pos < bytes.len() && bytes[pos] != b'{' && bytes[pos] != b']' {
            pos += 1;
        }
        if pos >= bytes.len() || bytes[pos] == b']' {
            break;
        }
        // brace-match one object
        let (obj, next) = match_object(s, pos)?;
        pos = next;
        let id = field_str(obj, "id").context("index set missing id")?;
        let soc_arch = field_str(obj, "soc_arch").unwrap_or_default();
        let label = field_str(obj, "label").unwrap_or_default();
        let config_json = field_object(obj, "config")
            .or_else(|| field_str(obj, "config"))
            .context("index set missing config")?;
        let sig = field_str(obj, "sig").context("index set missing sig")?;
        let community_tok_s = field_num(obj, "community_tok_s");
        sets.push(IndexSet { id, soc_arch, label, config_json, sig, community_tok_s });
    }
    Ok(sets)
}

fn match_object(s: &str, start: usize) -> Result<(&str, usize)> {
    let b = s.as_bytes();
    let (mut depth, mut in_str, mut esc) = (0usize, false, false);
    for i in start..b.len() {
        let c = b[i];
        if in_str {
            if esc {
                esc = false;
            } else if c == b'\\' {
                esc = true;
            } else if c == b'"' {
                in_str = false;
            }
            continue;
        }
        match c {
            b'"' => in_str = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Ok((&s[start..=i], i + 1));
                }
            }
            _ => {}
        }
    }
    bail!("unterminated object in index")
}

fn field_str(obj: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\"");
    let i = obj.find(&pat)? + pat.len();
    let after = obj[i..].trim_start_matches(|c: char| c == ':' || c.is_whitespace());
    if !after.starts_with('"') {
        return None;
    }
    let a = &after[1..];
    let mut out = String::new();
    let mut chars = a.chars();
    while let Some(c) = chars.next() {
        match c {
            '"' => return Some(out),
            '\\' => match chars.next()? {
                'n' => out.push('\n'),
                't' => out.push('\t'),
                'r' => out.push('\r'),
                o => out.push(o),
            },
            c => out.push(c),
        }
    }
    None
}

fn field_num(obj: &str, key: &str) -> Option<f64> {
    let pat = format!("\"{key}\"");
    let i = obj.find(&pat)? + pat.len();
    let after = obj[i..].trim_start_matches(|c: char| c == ':' || c.is_whitespace());
    let end = after
        .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == 'e' || c == 'E' || c == '+'))
        .unwrap_or(after.len());
    after[..end].parse().ok()
}

fn field_object(obj: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\"");
    let i = obj.find(&pat)? + pat.len();
    let after = obj[i..].trim_start_matches(|c: char| c == ':' || c.is_whitespace());
    if !after.starts_with('{') {
        return None;
    }
    // reuse the brace matcher against the tail
    match_object(after, 0).ok().map(|(o, _)| o.to_string())
}

fn hex_decode(s: &str) -> Result<Vec<u8>> {
    if s.len() % 2 != 0 {
        bail!("odd hex length");
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).context("bad hex"))
        .collect()
}

/// Verify a signed index blob (decision 4). Nothing in `raw` is trusted until this returns Ok.
pub fn verify_index(raw: &[u8], sig_hex: &str) -> Result<()> {
    let vk_bytes: [u8; 32] = hex_decode(publisher_pubkey_hex())?
        .try_into()
        .map_err(|_| anyhow::anyhow!("publisher verifying key must be 32 bytes"))?;
    let vk = VerifyingKey::from_bytes(&vk_bytes).context("bad publisher verifying key")?;
    let sig_bytes: [u8; 64] = hex_decode(sig_hex)?
        .try_into()
        .map_err(|_| anyhow::anyhow!("index signature must be 64 bytes"))?;
    vk.verify_strict(raw, &Signature::from_bytes(&sig_bytes))
        .map_err(|_| anyhow::anyhow!("index signature does not verify — refusing"))?;
    Ok(())
}

/// Is a system downloader present? (curl or wget)
pub fn have_downloader() -> bool {
    for bin in ["curl", "wget"] {
        if std::process::Command::new(bin)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
        {
            return true;
        }
    }
    false
}

/// The reachability/config preconditions common to update/subscribe/share, so each command
/// gives the SAME honest refusal rather than a transport error.
pub fn require_channel() -> Result<String> {
    // Configuration first, transport second. `base_url` refuses an
    // unconfigured build, so a box with no curl is never told its problem is
    // curl when it is also missing an endpoint. A release build can bake
    // DS4_PUBLISHER_URL at compile time; a runtime env var still overrides.
    let url = base_url()?;
    if !have_downloader() {
        bail!("no curl or wget on PATH to reach {url}");
    }
    Ok(url)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    #[test]
    fn a_good_signature_verifies_and_a_tampered_index_is_refused() {
        // Sign a blob with a throwaway key, verify against ITS public half.
        let sk = SigningKey::from_bytes(&[7u8; 32]);
        let vk_hex: String = sk.verifying_key().to_bytes().iter().map(|b| format!("{b:02x}")).collect();
        // Temporarily point the verifier at this test key via a direct check (publisher_pubkey_hex
        // is build-time; here we verify the primitive against the same key we signed with).
        let raw = br#"{"sets":[{"id":"ab12","soc_arch":"a3"}]}"#;
        let sig = sk.sign(raw);
        let sig_hex: String = sig.to_bytes().iter().map(|b| format!("{b:02x}")).collect();
        // Direct primitive check mirroring verify_index's body.
        let vk = VerifyingKey::from_bytes(&hex_decode(&vk_hex).unwrap().try_into().unwrap()).unwrap();
        assert!(vk
            .verify_strict(raw, &Signature::from_bytes(&sig_hex_to_arr(&sig_hex)))
            .is_ok());
        // Tamper one byte of the payload -> must fail.
        let mut bad = raw.to_vec();
        bad[2] ^= 0xFF;
        assert!(vk
            .verify_strict(&bad, &Signature::from_bytes(&sig_hex_to_arr(&sig_hex)))
            .is_err());
    }

    fn sig_hex_to_arr(h: &str) -> [u8; 64] {
        hex_decode(h).unwrap().try_into().unwrap()
    }

    #[test]
    fn an_unconfigured_build_refuses_rather_than_guessing_a_url() {
        // There used to be a baked fallback here, and its host was two words
        // with a space between them, left behind by a blanket rename. It could
        // never have resolved, and it meant `base_url` returned Ok for a build
        // with no endpoint at all -- the opposite of what this module
        // documents.
        //
        // Serialised with the other env-var tests by `--test-threads` is not
        // something this file can assume, so the variable is removed and
        // restored around the assertion rather than left set.
        let saved = std::env::var("DS4_PUBLISHER_URL").ok();
        std::env::remove_var("DS4_PUBLISHER_URL");
        if option_env!("DS4_PUBLISHER_URL").is_none() {
            let err = base_url().expect_err(
                "an unconfigured build must refuse. If this returns Ok, a fallback \
                 endpoint has come back -- check it is not a guess.",
            );
            let msg = format!("{err}");
            assert!(
                msg.contains("not configured"),
                "the refusal must say the channel is unconfigured, not blame transport \
                 or name a host: {msg}"
            );
            assert!(
                !msg.contains(' ') || !msg.contains("https://"),
                "the message appears to contain a URL; an unconfigured build has none \
                 to name: {msg}"
            );
        }
        if let Some(v) = saved {
            std::env::set_var("DS4_PUBLISHER_URL", v);
        }
    }

    #[test]
    fn non_https_url_is_refused() {
        std::env::set_var("DS4_PUBLISHER_URL", "http://evil.example/x");
        assert!(base_url().is_err());
        std::env::remove_var("DS4_PUBLISHER_URL");
    }

    #[test]
    fn unconfigured_channel_refuses_cleanly() {
        std::env::remove_var("DS4_PUBLISHER_URL");
        let e = require_channel().unwrap_err().to_string();
        assert!(e.contains("not configured") || e.contains("curl or wget"), "{e}");
    }

    #[test]
    fn loopback_url_is_allowed_for_local_testing() {
        std::env::set_var("DS4_PUBLISHER_URL", "http://127.0.0.1:8091/ds4/kernels");
        assert!(base_url().is_ok(), "loopback must be allowed");
        std::env::set_var("DS4_PUBLISHER_URL", "http://evil.example/x");
        assert!(base_url().is_err(), "non-loopback http must be refused");
        std::env::remove_var("DS4_PUBLISHER_URL");
    }

    #[test]
    fn parse_index_extracts_sets_with_object_config() {
        let j = br#"{"sets":[
          {"id":"ab12cd34ef56","soc_arch":"a3","label":"the publisher v1","config":{"knobs":{"matvec_block_dim":48}},"sig":"deadbeef","community_tok_s":6.7,"n_samples":3},
          {"id":"ff00ff00ff00","soc_arch":"a2","label":"other","config":{"k":1},"sig":"cafe","community_tok_s":null,"n_samples":0}
        ]}"#;
        let sets = parse_index(j).unwrap();
        assert_eq!(sets.len(), 2);
        assert_eq!(sets[0].id, "ab12cd34ef56");
        assert_eq!(sets[0].soc_arch, "a3");
        assert_eq!(sets[0].config_json, r#"{"knobs":{"matvec_block_dim":48}}"#);
        assert_eq!(sets[0].community_tok_s, Some(6.7));
        assert_eq!(sets[1].community_tok_s, None);
        assert_eq!(sets[1].config_json, r#"{"k":1}"#);
    }

    #[test]
    fn empty_index_parses_to_nothing() {
        assert!(parse_index(br#"{"sets":[]}"#).unwrap().is_empty());
    }
}
