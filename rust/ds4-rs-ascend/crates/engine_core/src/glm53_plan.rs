//! Which attention — and which feed-forward — each GLM 5.3 / Kimi K3 block uses, read FROM THE FILE.
//!
//! The registry's note says GLM 5.3 is a hybrid of 34 delta-net blocks and 12 MLA blocks, and Kimi K3 of 69 and
//! 24. Those numbers came from somewhere; a forward pass cannot use them unless the source is the checkpoint,
//! because a schedule inferred from a note is a guess with a citation.
//!
//! The checkpoint states it TWICE and both are readable without the weights:
//!   * `blk.<n>.kda_*` tensors exist iff block `n` is a gated delta net;
//!   * `blk.<n>.attn_q_a` exists iff it is MLA.
//! So the schedule is a scan of the tensor table, which this module does.

use crate::gguf::GgufFile;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Attn {
    /// The gated delta-net / KDA path (`kda_q`, `kda_k`, `kda_v`, convs, gates).
    Kda,
    /// Multi-head latent attention (`attn_q_a`, `attn_kv_a`, ...).
    Mla,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ffn {
    /// An FFN with its own dense projections (`ffn_gate` / `ffn_up` / `ffn_down`).
    Dense,
    Moe,
    /// NO feed-forward projections of any kind: the block carries `ffn_norm` and nothing else. GLM 5.3's
    /// leading blocks are like this -- block 0 has `ffn_norm` and no `ffn_gate`, dense or expert -- so calling
    /// them "dense" was my label rather than the file's shape.
    None,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockPlan {
    pub attn: Attn,
    pub ffn: Ffn,
}

/// One entry per block, in index order.
pub fn schedule(g: &GgufFile) -> anyhow::Result<Vec<BlockPlan>> {
    schedule_of(&g.tensors)
}

/// The same scan over a TENSOR LIST, which is what a sharded checkpoint needs: a K3 shard is a disjoint
/// slice, so the schedule is only complete once every shard's table is merged. (Shard 1 of Kimi K3 carries
/// metadata only and has NO block tensors at all, which the single-file form reports as "not block-stacked".)
pub fn schedule_of(tensors: &[crate::gguf::TensorInfo]) -> anyhow::Result<Vec<BlockPlan>> {
    let g = GgufFile {
        version: 0,
        n_tensors: 0,
        n_meta_kv: 0,
        meta: Vec::new(),
        tensors: tensors.to_vec(),
        tensor_table_offset: 0,
        tensor_data_offset: 0,
        file_size: 0,
    };
    let g = &g;
    let n = g
        .tensors
        .iter()
        .filter_map(|t| {
            t.name
                .strip_prefix("blk.")
                .and_then(|r| r.split('.').next())
                .and_then(|b| b.parse::<usize>().ok())
        })
        .max()
        .ok_or_else(|| anyhow::anyhow!("no blk.<n>.* tensors: is this a block-stacked model?"))?
        + 1;

    let mut out = Vec::with_capacity(n);
    for b in 0..n {
        let has = |suffix: &str| g.tensors.iter().any(|t| t.name == format!("blk.{b}.{suffix}"));
        // A THIRD SPELLING. GLM 5.3 names the linear path `kda_q` and MLA `attn_q_a`; Kimi K3 names its linear path
        // `attn_q` -- which matches NEITHER of the first two and was reported as "the schedule cannot be read from names"
        // until the classifier learned it. `attn_q` cannot be confused with MLA, which always carries the `_a`.
        let attn = if has("kda_q.weight") || (has("attn_q.weight") && !has("attn_q_a.weight")) {
            Attn::Kda
        } else if has("attn_q_a.weight") {
            Attn::Mla
        } else if !tensors.iter().any(|t| t.name.starts_with(&format!("blk.{b}."))) {
            // A block with NO tensors is ABSENT, not malformed: a sharded checkpoint may be missing shards, and the
            // contiguous prefix before the first gap is what can actually be run. (Kimi K3 has 10 of 14 shards; blocks
            // 0-61 are complete and 62.. are absent, so the schedule stops here rather than refusing the whole model.)
            break;
        } else {
            anyhow::bail!("block {b} has neither kda_q nor attn_q_a: the schedule cannot be read from names");
        };
        // MoE iff the block carries an expert table; the leading dense blocks do not.
        // THE STACKED-EXPERT NAMES, read from the file rather than guessed: GLM 5.3 spells them
        // `blk.<n>.ffn_gate_exps.weight`, `ffn_up_exps`, `ffn_down_exps` (plus `ffn_*_shexp` for the shared
        // one and `exp_probs_b.bias` for the router). My first version looked for `ffn.experts.` and found
        // ZERO MoE blocks in a model whose registry note says 43 of them.
        let has_dense = g.tensors.iter().any(|t| t.name == format!("blk.{b}.ffn_gate.weight"));
        let ffn = if g.tensors.iter().any(|t| t.name == format!("blk.{b}.ffn_gate_exps.weight")) {
            Ffn::Moe
        } else if has_dense {
            Ffn::Dense
        } else {
            Ffn::None
        };
        out.push(BlockPlan { attn, ffn });
    }
    Ok(out)
}

/// A compact census, for reporting and for comparing two checkpoints.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Census {
    pub blocks: usize,
    pub kda: usize,
    pub mla: usize,
    pub moe: usize,
    pub dense: usize,
    pub no_ffn: usize,
}

impl Census {
    pub fn of(s: &[BlockPlan]) -> Census {
        let mut c = Census { blocks: s.len(), ..Default::default() };
        for b in s {
            match b.attn {
                Attn::Kda => c.kda += 1,
                Attn::Mla => c.mla += 1,
            }
            match b.ffn {
                Ffn::Moe => c.moe += 1,
                Ffn::Dense => c.dense += 1,
                Ffn::None => c.no_ffn += 1,
            }
        }
        c
    }
}


/// The schedule, choosing the SOURCE the checkpoint actually provides.
///
/// `head_count_kv` is a per-layer array where the model states its own schedule (Kimi K3: 93 values, 0 for
/// linear and 1 for MLA; GLM 5.3 does not carry the key at all). Where it exists it is authoritative and the
/// tensor names are not consulted. Where it does not, the names are, and each model has its own spelling --
/// GLM 5.3 uses `kda_q` for the delta net, Kimi K3 uses `attn_q`, and both use `attn_q_a` for MLA.
pub fn schedule_from(
    tensors: &[crate::gguf::TensorInfo],
    meta: &[(String, crate::gguf::MetaValue)],
) -> anyhow::Result<(Vec<BlockPlan>, &'static str)> {
    // 1. the per-layer array, if the model states one
    let kv = meta.iter().find(|(k, _)| k.ends_with("attention.head_count_kv")).map(|(_, v)| v);
    if let Some(crate::gguf::MetaValue::Array { values, .. }) = kv {
        let n = values.len();
        let mut out = Vec::with_capacity(n);
        for b in 0..n {
            let flag = format!("{:?}", values[b]);
            let attn = if flag.contains("I32(1)") || flag.contains("U32(1)") { Attn::Mla } else { Attn::Kda };
            let ffn = if tensors.iter().any(|t| t.name == format!("blk.{b}.ffn_gate_exps.weight")) {
                Ffn::Moe
            } else if tensors.iter().any(|t| t.name == format!("blk.{b}.ffn_gate.weight")) {
                Ffn::Dense
            } else {
                Ffn::None
            };
            out.push(BlockPlan { attn, ffn });
        }
        return Ok((out, "attention.head_count_kv (per-layer array)"));
    }
    // 2. otherwise the tensor names, which differ per model
    Ok((schedule_of(tensors)?, "tensor names"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{GgmlType, TensorInfo};

    fn t(name: &str) -> TensorInfo {
        TensorInfo { name: name.to_string(), dims: vec![1], ttype: GgmlType::F32, offset: 0 }
    }

    /// `schedule_from` prefers the PER-LAYER ARRAY when the checkpoint carries one, and REPORTS that source. This is the
    /// path Kimi K3 uses (`attention.head_count_kv`, 0 = linear, 1 = MLA) and it had NO test of its own -- it was only
    /// ever exercised by running against the real file.
    #[test]
    fn the_per_layer_array_wins_and_its_source_is_reported() {
        use crate::gguf::MetaValue;
        let tensors = vec![t("blk.0.attn_q.weight"), t("blk.1.attn_q.weight")];
        let kv = MetaValue::Array {
            elem_type: crate::gguf::GgufType::U32,
            len: 2,
            values: vec![MetaValue::U32(0), MetaValue::U32(1)],
        };
        let meta = vec![("kimi.attention.head_count_kv".to_string(), kv)];
        let (plan, source) = schedule_from(&tensors, &meta).expect("the array path should not need tensor names");
        assert_eq!(plan.len(), 2);
        assert_eq!(plan[0].attn, Attn::Kda, "0 means linear (KDA)");
        assert_eq!(plan[1].attn, Attn::Mla, "1 means MLA");
        assert!(source.contains("head_count_kv"), "the source must name the array, got {source:?}");
    }

    /// And without the array it falls back to the TENSOR NAMES, saying so -- the GLM path.
    #[test]
    fn without_the_array_it_falls_back_to_tensor_names() {
        let tensors = vec![t("blk.0.kda_q.weight")];
        let (plan, source) = schedule_from(&tensors, &[]).expect("the name path should work");
        assert_eq!(plan[0].attn, Attn::Kda);
        assert!(source.contains("tensor names"), "got {source:?}");
    }

    /// The scan is a function of the NAMES, so a synthetic file proves it without a checkpoint.
    #[test]
    fn the_schedule_is_read_from_tensor_names() {
        let g = GgufFile {
            version: 3,
            n_tensors: 0,
            n_meta_kv: 0,
            meta: vec![],
            tensors: vec![
                t("blk.0.kda_q.weight"),
                t("blk.0.ffn_gate_exps.weight"),
                t("blk.1.attn_q_a.weight"),
                t("blk.1.ffn_gate.weight"),
            ],
            tensor_table_offset: 0,
            tensor_data_offset: 0,
            file_size: 0,
        };
        let s = schedule(&g).unwrap();
        assert_eq!(s.len(), 2);
        assert_eq!(s[0], BlockPlan { attn: Attn::Kda, ffn: Ffn::Moe });
        assert_eq!(s[1], BlockPlan { attn: Attn::Mla, ffn: Ffn::Dense });
        assert_eq!(Census::of(&s), Census { blocks: 2, kda: 1, mla: 1, moe: 1, dense: 1, no_ffn: 0 });
    }

    /// A block with neither attention is a REFUSAL, not a default: guessing would silently produce a schedule
    /// that disagrees with the checkpoint.
    #[test]
    fn a_block_with_no_attention_is_refused() {
        let g = GgufFile {
            version: 3, n_tensors: 0, n_meta_kv: 0, meta: vec![],
            tensors: vec![t("blk.0.ffn.dense.w1.weight")],
            tensor_table_offset: 0, tensor_data_offset: 0, file_size: 0,
        };
        let err = schedule(&g).expect_err("no attention tensor");
        assert!(format!("{err}").contains("neither kda_q nor attn_q_a"), "got: {err}");
    }
}
