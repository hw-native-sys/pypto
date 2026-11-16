//! The MoE feed-forward, in the LIBRARY so both drivers call one implementation.
//!
//! It lived in an example until iteration 213, which is how the prefill driver came to have NO MoE at all while the
//! decode driver had one: two drivers, one author, and the shared piece was copied rather than shared. `moe_route` was
//! extracted for the same reason and it is the same lesson twice.

use crate::gguf::GgufFile;
use crate::glm5_stream::ShardedGguf;
use crate::kda_block::gated_rms_norm;

/// Read `expert_weights_scale` from the checkpoint, REFUSING rather than defaulting.
///
/// It was hardcoded 2.5 here until iteration 220, transcribed from a `grep` of the metadata. A transcription is a copy;
/// this is a read, and a checkpoint without the key is a loud error rather than a silently-1.0 expert sum.
pub fn expert_weights_scale(g: &GgufFile) -> anyhow::Result<f64> {
    for (k, v) in &g.meta {
        if k.ends_with("expert_weights_scale") {
            return match v {
                crate::gguf::MetaValue::F32(x) => Ok(*x as f64),
                crate::gguf::MetaValue::F64(x) => Ok(*x),
                other => anyhow::bail!("expert_weights_scale is {other:?}, not a float"),
            };
        }
    }
    anyhow::bail!("no expert_weights_scale in this checkpoint: the MoE sum would be weighted wrongly and silently")
}

/// The one thing `moe_ffn` needs from the checkpoint: a tensor by name, dequantised to f32.
///
/// It exists so the MoE can be tested without a 96 GB checkpoint. `moe_ffn` had NO unit test until this trait, because a
/// `&ShardedGguf` cannot be constructed in a test -- and a function that cannot be tested is a function whose assembly is
/// only ever exercised end to end, where a failure says "the token is wrong" rather than which line is wrong.
pub trait TensorSource {
    fn dequant(&self, name: &str) -> anyhow::Result<Vec<f32>>;
}

impl TensorSource for ShardedGguf {
    fn dequant(&self, name: &str) -> anyhow::Result<Vec<f32>> {
        self.dequant_name(name)
    }
}

pub fn moe_ffn(g: &ShardedGguf, scale: f64, b: usize, x: &[f64], d: usize, eps: f64) -> anyhow::Result<(Vec<f64>, usize)> {
    moe_ffn_with(g, scale, b, x, d, eps)
}

/// `moe_ffn` over any [`TensorSource`], which is what makes it testable.
pub fn moe_ffn_with<S: TensorSource + ?Sized>(
    src: &S,
    scale: f64,
    b: usize,
    x: &[f64],
    d: usize,
    eps: f64,
) -> anyhow::Result<(Vec<f64>, usize)> {
    // `scale` now comes from `expert_weights_scale(&GgufFile)` at the driver, so the 2.5 transcription is gone.
    let mv = |w: &[f32], xv: &[f64], k: usize, n: usize, off: usize| -> Vec<f64> {
        (0..n).map(|r| (0..k).map(|c| w[off + r * k + c] as f64 * xv[c]).sum()).collect()
    };
    let fnorm = src.dequant(&format!("blk.{b}.ffn_norm.weight"))?;
    let fin = gated_rms_norm(x, &fnorm.iter().map(|v| *v as f64).collect::<Vec<_>>(), &vec![1.0f64; d], eps);
    // router
    let rin = src.dequant(&format!("blk.{b}.ffn_gate_inp.weight"))?;
    let n_exp = rin.len() / d;
    let bias = src.dequant(&format!("blk.{b}.exp_probs_b.bias"))?;
    let mut scores: Vec<(f64, usize)> = (0..n_exp)
        .map(|e| {
            let s: f64 = (0..d).map(|c| rin[c * n_exp + e] as f64 * fin[c]).sum::<f64>() + bias[e] as f64;
            (s, e)
        })
        .collect();
    scores.sort_by(|a, b_| b_.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let used = 8.min(n_exp);
    // softmax over the selected
    let mx = scores[0].0;
    let exps: Vec<(f64, usize)> = scores[..used].iter().map(|(s, e)| ((s - mx).exp(), *e)).collect();
    let z: f64 = exps.iter().map(|(w, _)| w).sum();
    let gate_w = src.dequant(&format!("blk.{b}.ffn_gate_exps.weight"))?;
    let up_w = src.dequant(&format!("blk.{b}.ffn_up_exps.weight"))?;
    let dn_w = src.dequant(&format!("blk.{b}.ffn_down_exps.weight"))?;
    anyhow::ensure!(gate_w.len() == up_w.len() && gate_w.len() % n_exp == 0, "stacked expert planes disagree");
    let per = gate_w.len() / n_exp;           // values per expert
    let inter = per / d;                       // rows per expert
    anyhow::ensure!(inter * d == per, "expert plane {per} is not {inter} x {d}");
    let dn_per = dn_w.len() / n_exp;
    let mut acc = vec![0.0f64; d];
    // `glm5-next.expert_weights_scale = 2.5`. The checkpoint applies a scale to the expert contributions and my
    // first implementation IGNORED it, which is a wiring error of exactly the kind no amount of arithmetic
    // cross-checking would catch: every product is right, the sum is weighted differently from the model's.
    for (w, e) in if used == 0 { Vec::new() } else { exps } {
        let wgt = scale * w / z;
        let g_ = mv(&gate_w, &fin, d, inter, e * per);
        let u_ = mv(&up_w, &fin, d, inter, e * per);
        let act: Vec<f64> = g_.iter().zip(&u_).map(|(a, b_)| (a / (1.0 + (-a).exp())) * b_).collect();
        let dn = mv(&dn_w, &act, inter, d, e * dn_per);
        for i in 0..d { acc[i] += wgt * dn[i]; }
    }
    // the shared expert, always applied
    let sg = src.dequant(&format!("blk.{b}.ffn_gate_shexp.weight"))?;
    let su = src.dequant(&format!("blk.{b}.ffn_up_shexp.weight"))?;
    let sd = src.dequant(&format!("blk.{b}.ffn_down_shexp.weight"))?;
    let sinter = sg.len() / d;
    let sgv = mv(&sg, &fin, d, sinter, 0);
    let suv = mv(&su, &fin, d, sinter, 0);
    let sact: Vec<f64> = sgv.iter().zip(&suv).map(|(a, b_)| (a / (1.0 + (-a).exp())) * b_).collect();
    let sdn = mv(&sd, &sact, sinter, d, 0);
    for i in 0..d { acc[i] += sdn[i]; }
    Ok((acc, used))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{GgmlType, MetaValue};

    fn file_with(meta: Vec<(String, MetaValue)>) -> GgufFile {
        GgufFile {
            version: 3,
            n_tensors: 0,
            n_meta_kv: meta.len() as u64,
            meta,
            tensors: Vec::new(),
            tensor_table_offset: 0,
            tensor_data_offset: 0,
            file_size: 0,
        }
    }

    /// BUG (iteration 180): the expert scale was a transcribed 2.5. The reader must FIND it by suffix.
    #[test]
    fn the_expert_scale_is_read_by_suffix() {
        let g = file_with(vec![(
            "glm5-next.expert_weights_scale".to_string(),
            MetaValue::F32(2.5),
        )]);
        assert!((expert_weights_scale(&g).unwrap() - 2.5).abs() < 1e-6);
    }

    /// And a checkpoint WITHOUT the key must REFUSE rather than default, which was the actual defect: a missing
    /// scale weights the expert sum wrongly and silently.
    #[test]
    fn a_missing_expert_scale_is_refused() {
        let g = file_with(vec![(
            "glm5-next.expert_count".to_string(),
            MetaValue::U32(288),
        )]);
        let err = expert_weights_scale(&g).expect_err("a missing scale must not default");
        assert!(format!("{err}").contains("expert_weights_scale"), "got: {err}");
    }

    /// A FAKE CHECKPOINT: two experts, d = 2, inter = 2, built so the routing and the 2.5 scale are both checkable by
    /// hand. Until this trait, `moe_ffn` had no test at all -- a `&ShardedGguf` cannot be constructed in one.
    struct Fake {
        tensors: std::collections::HashMap<String, Vec<f32>>,
    }
    impl TensorSource for Fake {
        fn dequant(&self, name: &str) -> anyhow::Result<Vec<f32>> {
            self.tensors
                .get(name)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("fake has no {name}"))
        }
    }

    fn fake_two_experts() -> Fake {
        let mut t = std::collections::HashMap::new();
        let d = 2usize;
        let n_exp = 2usize;
        let inter = 2usize;
        t.insert("blk.0.ffn_norm.weight".to_string(), vec![1.0; d]);
        // router [d, n_exp]: expert 0 strongly preferred
        t.insert("blk.0.ffn_gate_inp.weight".to_string(), vec![1.0, 0.0, 1.0, 0.0]);
        t.insert("blk.0.exp_probs_b.bias".to_string(), vec![1.0, -1.0]);
        // two experts, each [inter, d] for gate/up and [d, inter] for down
        t.insert("blk.0.ffn_gate_exps.weight".to_string(), vec![1.0; n_exp * inter * d]);
        t.insert("blk.0.ffn_up_exps.weight".to_string(), vec![1.0; n_exp * inter * d]);
        t.insert("blk.0.ffn_down_exps.weight".to_string(), vec![0.5; n_exp * d * inter]);
        // the shared expert contributes a known constant
        t.insert("blk.0.ffn_gate_shexp.weight".to_string(), vec![1.0; inter * d]);
        t.insert("blk.0.ffn_up_shexp.weight".to_string(), vec![1.0; inter * d]);
        t.insert("blk.0.ffn_down_shexp.weight".to_string(), vec![0.0; d * inter]);
        Fake { tensors: t }
    }

    /// The assembly runs, returns `d` values, all finite, and reports how many experts it used.
    #[test]
    fn the_moe_assembly_runs_and_is_finite() {
        let f = fake_two_experts();
        let x = vec![1.0f64, 1.0];
        let (out, used) = moe_ffn_with(&f, 2.5, 0, &x, 2, 1e-5).expect("the fake should satisfy every read");
        assert_eq!(out.len(), 2, "the MoE returns one value per channel");
        assert!(out.iter().all(|v| v.is_finite()), "got {out:?}");
        assert!(used > 0, "no expert was used");
    }

    /// A MISSING TENSOR IS AN ERROR, not a silent zero: a fake with no `ffn_down_exps` must refuse.
    #[test]
    fn a_missing_tensor_is_refused_not_treated_as_zero() {
        let mut f = fake_two_experts();
        f.tensors.remove("blk.0.ffn_down_exps.weight");
        let x = vec![1.0f64, 1.0];
        let err = moe_ffn_with(&f, 2.5, 0, &x, 2, 1e-5).expect_err("a missing tensor must not become zeros");
        assert!(format!("{err}").contains("ffn_down_exps"), "got: {err}");
    }

    /// A non-float value is refused rather than cast.
    #[test]
    fn a_non_float_expert_scale_is_refused() {
        let g = file_with(vec![(
            "glm5-next.expert_weights_scale".to_string(),
            MetaValue::U32(2),
        )]);
        let err = expert_weights_scale(&g).expect_err("an integer scale must not be cast");
        assert!(format!("{err}").contains("not a float"), "got: {err}");
    }
}
