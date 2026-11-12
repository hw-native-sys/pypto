//! TWO positions through layer 0: the first case where attention's softmax is not trivially 1.
//!
//! With one position, `softmax([q.k + sink])` is 1 and every head emits v regardless of q. Two
//! positions make the scores matter, so this is the first run that can tell a working attention from a
//! broken one.
//!
//! The check is not a comparison against a reference (there is none on this box): it is that the SECOND
//! token's output DIFFERS when it attends over two rows than when it attends over itself alone. A kernel
//! that ignored the KV cache would give the same answer both times.
//!
//! usage: w8a8_two_tokens <package dir> <token a> <token b>

use engine_core::attn_dispatch::{
    decode_attn_prefix_with, decode_attn_suffix_with, AttnLayerWeights, AttnStepInputs,
    CpuAttentionDispatcher, KvCacheView, LayerParams,
};
use engine_core::moe::{hash_router_weights_from_probs, softplus_sqrt};
use engine_core::safetensors::Package;
use engine_core::w8a8_roles::{PackageViews, WeightSource};
use std::path::PathBuf;

fn matvec(w: &[f32], x: &[f32], n: usize) -> Vec<f32> {
    let k = x.len();
    (0..n).map(|i| (0..k).map(|j| w[i * k + j] * x[j]).sum()).collect()
}
fn rms_norm(x: &[f32], g: &[f32], eps: f32) -> Vec<f32> {
    let ss: f32 = x.iter().map(|v| v * v).sum();
    let inv = 1.0 / (ss / x.len() as f32 + eps).sqrt();
    x.iter().zip(g).map(|(v, g)| v * inv * g).collect()
}
fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }
fn expert_out(pkg: &Package, prefix: &str, h: &[f32], d_ffn: usize) -> anyhow::Result<Vec<f32>> {
    let w1 = pkg.dequant_tensor(&format!("{prefix}.w1.weight"))?;
    let w3 = pkg.dequant_tensor(&format!("{prefix}.w3.weight"))?;
    let w2 = pkg.dequant_tensor(&format!("{prefix}.w2.weight"))?;
    let g = matvec(&w1, h, d_ffn);
    let u = matvec(&w3, h, d_ffn);
    let act: Vec<f32> = g.iter().zip(&u).map(|(&a, &b)| silu(a) * b).collect();
    Ok(matvec(&w2, &act, h.len()))
}

fn main() -> anyhow::Result<()> {
    let dir = std::env::args().nth(1).map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("usage: w8a8_two_tokens <package dir> <token a> <token b>"))?;
    let ta: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(1000);
    let tb: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(2000);
    let pkg = Package::open(&dir)?;
    let views = PackageViews::new(&pkg);

    let d = 4096usize; let n_hc = 4usize; let heads = 64usize; let hd = 512usize; let lora_q = 1024usize;
    let d_ffn = 2048usize;

    let params = LayerParams {
        layer_idx: 0, d_embd: d as u32, n_hc: n_hc as u32, n_head: heads as u32, head_dim: hd as u32,
        n_rot: 64, n_lora_q: lora_q as u32, n_lora_kv: hd as u32,
        hc_sinkhorn_iter: 20, hc_eps: 1e-6, rms_eps: 1e-6,
        rope_orig_ctx: 65536, rope_freq_base: 10000.0, rope_freq_scale: 1.0 / 16.0,
        rope_ext_factor: 0.0, rope_attn_factor: 1.0, compress_ratio: 0, n_out_group: 8,
    };
    let view = views.layer(0)?.into_owned();
    let layer = AttnLayerWeights::from_layer_view(&views, &view, params, d_ffn as u32, f32::INFINITY, false)?;

    let emb = pkg.dequant_tensor("embed.weight").or_else(|_| pkg.f32_tensor("embed.weight"))?;
    let vocab = emb.len() / d;
    anyhow::ensure!(ta < vocab && tb < vocab, "a token id is past the vocabulary");

    let gamma = pkg.f32_tensor("layers.0.attn_norm.weight")?;
    let wq_a = pkg.dequant_tensor("layers.0.attn.wq_a.weight")?;
    let qn = pkg.f32_tensor("layers.0.attn.q_norm.weight")?;
    let wq_b = pkg.dequant_tensor("layers.0.attn.wq_b.weight")?;
    let wkv = pkg.dequant_tensor("layers.0.attn.wkv.weight")?;
    let p = "layers.0";

    let run = |kvc: &mut Vec<f32>, tok: usize, pos: u32, cap: u32| -> anyhow::Result<Vec<f32>> {
        let x = &emb[tok * d..(tok + 1) * d];
        let n = rms_norm(x, &gamma, 1e-6);
        let q_ln = rms_norm(&matvec(&wq_a, &n, lora_q), &qn, 1e-6);
        let q_heads = matvec(&wq_b, &q_ln, heads * hd);
        let kv_raw_row = matvec(&wkv, &n, hd);
        let mut hc = vec![0.0f32; n_hc * d];
        for h in 0..n_hc { hc[h * d..(h + 1) * d].copy_from_slice(x); }
        let zero = vec![0.0f32; d];
        let mut step = AttnStepInputs {
            cur_hc: &hc, kv_raw_row: &kv_raw_row, q_heads: &q_heads, pos,
            kv_view: KvCacheView { raw: kvc, raw_cap: cap, pos },
            n_raw: pos, raw_start: 0, routed_out: &zero,
            kv_comp_rows: None, n_comp: 0, comp_selected: None, n_selected: 0,
        };
        let prefix = decode_attn_prefix_with(&CpuAttentionDispatcher, &layer, &mut step);
        let fg = pkg.f32_tensor(&format!("{p}.ffn_norm.weight"))?;
        let hn = rms_norm(&prefix.normed, &fg, 1e-6);
        let mut routed = expert_out(&pkg, &format!("{p}.ffn.shared_experts"), &hn, d_ffn)?;
        let gate = pkg.f32_tensor(&format!("{p}.ffn.gate.weight"))
            .or_else(|_| pkg.dequant_tensor(&format!("{p}.ffn.gate.weight")))?;
        let probs = softplus_sqrt(&matvec(&gate, &hn, 256));
        let raw = pkg.raw(&format!("{p}.ffn.gate.tid2eid"))?.1;
        let ids: Vec<i32> = raw.chunks_exact(4).map(|c| i32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
        let picked: Vec<usize> = (0..6).map(|k| ids[tok * 6 + k] as usize).collect();
        let w = hash_router_weights_from_probs(&probs, &picked, 6);
        for (i, &e) in picked.iter().enumerate() {
            let y = expert_out(&pkg, &format!("{p}.ffn.experts.{e}"), &hn, d_ffn)?;
            for (a, b) in routed.iter_mut().zip(&y) { *a += w[i] * b; }
        }
        Ok(decode_attn_suffix_with(&CpuAttentionDispatcher, &layer, &prefix, &routed, pos))
    };

    // One-row cache: tb sees only itself.
    let mut one = vec![0.0f32; hd];
    let attn_b_alone = run(&mut one, tb, 0, 1)?;
    // Two-row cache: A at 0, then B at 1 over both rows.
    let mut two = vec![0.0f32; 2 * hd];
    let _ = run(&mut two, ta, 0, 2)?;
    let attn_b_two = run(&mut two, tb, 1, 2)?;

    let diff: f32 = attn_b_alone.iter().zip(&attn_b_two).map(|(a, b)| (a - b).abs()).sum();
    let scale: f32 = attn_b_alone.iter().map(|v| v.abs()).sum();
    println!("tb alone  rms {:.5}", (attn_b_alone.iter().map(|v| v*v).sum::<f32>()/attn_b_alone.len() as f32).sqrt());
    println!("tb two-row rms {:.5}", (attn_b_two.iter().map(|v| v*v).sum::<f32>()/attn_b_two.len() as f32).sqrt());
    println!("L1 difference {diff:.4} against scale {scale:.4} = {:.4}", diff / scale);
    anyhow::ensure!(attn_b_two.iter().all(|v| v.is_finite()), "the two-row run produced a non-finite value");
    anyhow::ensure!(diff / scale > 1e-3, "attending over two rows changed nothing, so the KV cache is not being read");
    println!("PASS: attention reads the KV cache -- tb's output changes when a second row is present");
    Ok(())
}
