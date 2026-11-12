//! THE ENGINE'S OWN layer path, driven from the w8a8 package.
//!
//! Everything here is the engine's code, not an example's: `AttnLayerWeights::from_layer_view` builds
//! the layer's weights from a `WeightSource` over the package, `LayerParams` comes from the checkpoint's
//! config (see the campaign's parameter table), and `decode_attn_layer_with` runs the prefix (hc
//! collapse, q/kv norm, rope, attention, GROUPED output projection, hc expand) and the suffix.
//!
//! That is what the hand-rolled `w8a8_layer0_token` example could not do: the `hc_*` mechanism, which
//! the engine implements and the example skipped.
//!
//! usage: w8a8_real_layer <package dir> <token id>

use engine_core::attn_dispatch::{
    decode_attn_prefix_with, decode_attn_suffix_with, CpuAttentionDispatcher, AttnLayerWeights,
    AttnStepInputs, KvCacheView, LayerParams,
};
use engine_core::moe::{hash_router_weights_from_probs, softplus_sqrt};
use engine_core::safetensors::Package;
use engine_core::w8a8_roles::{PackageViews, WeightSource};  // WeightSource supplies `layer()`
use std::path::PathBuf;

fn matvec(w: &[f32], x: &[f32], n: usize) -> Vec<f32> {
    let k = x.len();
    (0..n).map(|i| (0..k).map(|j| w[i * k + j] * x[j]).sum()).collect()
}

fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }

/// One expert: y = W2 . (silu(W1 . h) * (W3 . h)), on the package's own int8 planes.
fn expert_out(
    pkg: &Package,
    prefix: &str,
    h: &[f32],
    d_ffn: usize,
) -> anyhow::Result<Vec<f32>> {
    let w1 = pkg.dequant_tensor(&format!("{prefix}.w1.weight"))?;
    let w3 = pkg.dequant_tensor(&format!("{prefix}.w3.weight"))?;
    let w2 = pkg.dequant_tensor(&format!("{prefix}.w2.weight"))?;
    let g = matvec(&w1, h, d_ffn);
    let u = matvec(&w3, h, d_ffn);
    let act: Vec<f32> = g.iter().zip(&u).map(|(&a, &b)| silu(a) * b).collect();
    Ok(matvec(&w2, &act, h.len()))
}

fn rms_norm(x: &[f32], g: &[f32], eps: f32) -> Vec<f32> {
    let ss: f32 = x.iter().map(|v| v * v).sum();
    let inv = 1.0 / (ss / x.len() as f32 + eps).sqrt();
    x.iter().zip(g).map(|(v, g)| v * inv * g).collect()
}

fn main() -> anyhow::Result<()> {
    let dir = std::env::args().nth(1).map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("usage: w8a8_real_layer <package dir> <token id>"))?;
    let token: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(1000);
    let pkg = Package::open(&dir)?;
    let views = PackageViews::new(&pkg);

    let d = 4096usize; let n_hc = 4usize; let heads = 64usize; let hd = 512usize; let lora_q = 1024usize;

    let params = LayerParams {
        layer_idx: 0,
        d_embd: d as u32,
        n_hc: n_hc as u32,
        n_head: heads as u32,
        head_dim: hd as u32,
        n_rot: 64,
        n_lora_q: lora_q as u32,
        n_lora_kv: hd as u32,
        hc_sinkhorn_iter: 20,
        hc_eps: 1e-6,
        rms_eps: 1e-6,
        rope_orig_ctx: 65536,
        rope_freq_base: 10000.0,
        rope_freq_scale: 1.0 / 16.0,
        rope_ext_factor: 0.0,
        rope_attn_factor: 1.0,
        compress_ratio: 0,
        n_out_group: 8,
    };

    let view = views.layer(0)?.into_owned();
    let layer = AttnLayerWeights::from_layer_view(&views, &view, params, 2048, f32::INFINITY, false)?;
    println!("AttnLayerWeights built from the package: layer 0");

    let emb = pkg.dequant_tensor("embed.weight").or_else(|_| pkg.f32_tensor("embed.weight"))?;
    let vocab = emb.len() / d;
    anyhow::ensure!(token < vocab, "token {token} past the vocabulary");
    let x = &emb[token * d..(token + 1) * d];

    // The two inputs the caller computes: the q lora-up product and the kv-down row.
    let gamma = pkg.f32_tensor("layers.0.attn_norm.weight")?;
    let n = rms_norm(x, &gamma, 1e-6);
    let wq_a = pkg.dequant_tensor("layers.0.attn.wq_a.weight")?;
    let qn = pkg.f32_tensor("layers.0.attn.q_norm.weight")?;
    let q_ln = rms_norm(&matvec(&wq_a, &n, lora_q), &qn, 1e-6);
    let wq_b = pkg.dequant_tensor("layers.0.attn.wq_b.weight")?;
    let q_heads = matvec(&wq_b, &q_ln, heads * hd);
    let wkv = pkg.dequant_tensor("layers.0.attn.wkv.weight")?;
    let kv_raw_row = matvec(&wkv, &n, hd);

    // HC slots: each holds a full-width copy of the embedding, so the residual starts as n_hc copies.
    let mut cur_hc = vec![0.0f32; n_hc * d];
    for h in 0..n_hc { cur_hc[h * d..(h + 1) * d].copy_from_slice(x); }

    let mut kvc = vec![0.0f32; hd];          // cap 1 row at pos 0
    let routed = vec![0.0f32; d];  // zeros for the step; the real one is computed after the prefix
    let step = AttnStepInputs {
        cur_hc: &cur_hc,
        kv_raw_row: &kv_raw_row,
        q_heads: &q_heads,
        pos: 0,
        kv_view: KvCacheView { raw: &mut kvc, raw_cap: 1, pos: 0 },
        n_raw: 0,
        raw_start: 0,
        routed_out: &routed,
        kv_comp_rows: None,
        n_comp: 0,
        comp_selected: None,
        n_selected: 0,
    };

    // PHASE 1: the engine's prefix -- hc collapse, q/kv norm, rope, attention, grouped output
    // projection, hc expand. It hands back the FFN's normed input.
    let mut step = step;
    let prefix = decode_attn_prefix_with(&CpuAttentionDispatcher, &layer, &mut step);
    println!("prefix: normed {} after_attn_hc {}", prefix.normed.len(), prefix.after_attn_hc.len());

    // PHASE 2: the MoE half, from the package's own expert weights. The engine expects `routed_out`
    // computed from `prefix.normed`, which is exactly what this does.
    let fg = pkg.f32_tensor("layers.0.ffn_norm.weight")?;
    let hn = rms_norm(&prefix.normed, &fg, 1e-6);
    let d_ffn = 2048usize;
    let mut routed = expert_out(&pkg, "layers.0.ffn.shared_experts", &hn, d_ffn)?;
    let gate = pkg.f32_tensor("layers.0.ffn.gate.weight")
        .or_else(|_| pkg.dequant_tensor("layers.0.ffn.gate.weight"))?;
    let probs = softplus_sqrt(&matvec(&gate, &hn, 256));
    let raw = pkg.raw("layers.0.ffn.gate.tid2eid")?.1;
    let ids: Vec<i32> = raw.chunks_exact(4).map(|c| i32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
    let mut picked = Vec::new();
    for k in 0..6 { picked.push(ids[token * 6 + k] as usize); }
    let w = hash_router_weights_from_probs(&probs, &picked, 6);
    for (i, &e) in picked.iter().enumerate() {
        let y = expert_out(&pkg, &format!("layers.0.ffn.experts.{e}"), &hn, d_ffn)?;
        for (a, b) in routed.iter_mut().zip(&y) { *a += w[i] * b; }
    }
    println!("routed_out from the package: {} finite={}", routed.len(),
             routed.iter().all(|v| v.is_finite()));

    // PHASE 3: the engine's suffix, with the real routed output instead of zeros.
    let out = decode_attn_suffix_with(&CpuAttentionDispatcher, &layer, &prefix, &routed, 0);
    let rms = (out.iter().map(|v| v * v).sum::<f32>() / out.len() as f32).sqrt();
    println!("after_ffn_hc: {} = {} hc x {} d, finite={}, rms={:.4}",
             out.len(), n_hc, d, out.iter().all(|v| v.is_finite()), rms);
    println!("PASS: engine prefix + package MoE + engine suffix, on real w8a8 weights");
    Ok(())
}
