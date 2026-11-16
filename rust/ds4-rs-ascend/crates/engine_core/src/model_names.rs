//! ONE PLACE FOR A MODEL'S NAMING.
//!
//! GLM 5.3 and Kimi K3 carry the same tensors under different names, and every driver that loads them needs to know both:
//!
//! ```text
//!                     GLM 5.3                     Kimi K3
//!   projections       kda_q/k/v/output            attn_q/k/v/output
//!   a_log             kda_a_log.weight            ssm_a          (no .weight)
//!   dt_bias           kda_dt_bias.weight          ssm_dt.bias
//!   beta              kda_beta.weight             ssm_beta.weight
//!   convs             kda_q_conv                  ssm_conv1d_q
//!   f gate            kda_f_a / kda_f_b           ssm_f_a / ssm_f_b
//!   OUTPUT gate       kda_g_a + kda_g_b (LOW-RANK)  ssm_g          (DIRECT)
//! ```
//!
//! This lived in the decode driver only, so the prefill could not serve Kimi at all, and the two copies of the naming could
//! drift. It is here so there is one copy.

use crate::glm5_stream::ShardedGguf;

/// Candidate tensor names for `base` in block `b`, in priority order.
///
/// **The candidates are per-base, not per-prefix.** A chain that tries every prefix for every base will eventually return a
/// real tensor from the wrong question -- `ssm_a` reached by a lookup for a `dt_bias` is 96 plausible numbers and no error.
pub fn candidates(b: usize, base: &str) -> Vec<String> {
    match base {
        "a_log" => vec![format!("blk.{b}.kda_a_log.weight"), format!("blk.{b}.ssm_a")],
        "dt_bias" => vec![format!("blk.{b}.kda_dt_bias.weight"), format!("blk.{b}.ssm_dt.bias")],
        "beta" => vec![format!("blk.{b}.kda_beta.weight"), format!("blk.{b}.ssm_beta.weight")],
        "g" => vec![format!("blk.{b}.kda_g.weight"), format!("blk.{b}.ssm_g.weight")],
        "q_conv" | "k_conv" | "v_conv" => {
            let short = &base[..1];
            vec![
                format!("blk.{b}.kda_{base}.weight"),
                format!("blk.{b}.ssm_conv1d_{short}.weight"),
            ]
        }
        other => vec![
            format!("blk.{b}.kda_{other}.weight"),
            format!("blk.{b}.attn_{other}.weight"),
            format!("blk.{b}.ssm_{other}.weight"),
        ],
    }
}

/// Dequantise `base` from block `b`, trying the names this model family uses. Returns the last error if none match, so the
/// message names the spellings that were tried rather than only the first.
pub fn dequant_any(g: &ShardedGguf, b: usize, base: &str) -> anyhow::Result<Vec<f32>> {
    let names = candidates(b, base);
    let mut last = None;
    for name in &names {
        match g.dequant_name(name) {
            Ok(v) => return Ok(v),
            Err(e) => {
                if std::env::var_os("SPELL_DEBUG").is_some() {
                    println!("  spelling {name}: {e}");
                }
                last = Some(e);
            }
        }
    }
    Err(last.unwrap_or_else(|| anyhow::anyhow!("no spelling of {base} in block {b}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// THE REGRESSION that cost a round: a resolver that tries every prefix for every base returned `ssm_a` when asked for a
    /// `dt_bias`, and Kimi read a 96-value a_log as a 12288-value dt bias without erroring.
    #[test]
    fn the_candidates_are_per_base_not_per_prefix() {
        for base in ["a_log", "dt_bias", "beta", "g"] {
            let c = candidates(3, base);
            assert!(c.iter().all(|n| n.starts_with("blk.3.")), "{base}: wrong block");
            // every candidate must be one of the names THIS base can have
            for name in &c {
                assert!(
                    name.contains("a_log") || name.contains("ssm_a") || name.contains("dt_bias")
                        || name.contains("ssm_dt") || name.contains("beta") || name.contains("g"),
                    "{base}: candidate {name} does not name {base}"
                );
            }
        }
        // the specific confusion: a dt_bias lookup must never offer ssm_a
        let dt = candidates(0, "dt_bias");
        assert!(!dt.iter().any(|n| n.ends_with("ssm_a")), "dt_bias offered ssm_a: {dt:?}");
        // and an a_log lookup must never offer ssm_dt.bias
        let al = candidates(0, "a_log");
        assert!(!al.iter().any(|n| n.contains("ssm_dt")), "a_log offered ssm_dt: {al:?}");
    }

    /// Both models' spellings are reachable for the shared projections, and the conv alias does not mangle itself.
    #[test]
    fn both_families_spellings_are_reachable() {
        for base in ["q", "k", "v", "output", "f_a", "f_b"] {
            let c = candidates(5, base);
            assert!(c.iter().any(|n| n.contains("kda_")), "{base}: no GLM spelling");
            assert!(c.iter().any(|n| n.contains("attn_") || n.contains("ssm_")), "{base}: no Kimi spelling");
        }
        assert!(candidates(1, "q_conv").contains(&"blk.1.ssm_conv1d_q.weight".to_string()));
        assert!(candidates(1, "v_conv").contains(&"blk.1.ssm_conv1d_v.weight".to_string()));
        // a chained `replace` once produced "ssm1d_q"; the alias must be exactly this
        assert!(!candidates(1, "q_conv").iter().any(|n| n.contains("ssm1d")), "mangled conv alias");
    }
}
