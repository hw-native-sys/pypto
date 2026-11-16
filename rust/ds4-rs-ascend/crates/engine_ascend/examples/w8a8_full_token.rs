//! ALL 43 LAYERS of DS4.1Flash through the ENGINE, then the head -> a token.
//!
//! Each layer is built from the package, run through the engine's own prefix/middle/suffix, and then
//! DROPPED before the next one -- so peak memory is one layer's f32 materialisation (~600 MB) rather
//! than 43 of them (~25 GB). That is the only reason this fits on the host at all.
//!
//! Still not a prefill: a single context position, so attention over one score is 1 and each MQA head
//! emits v itself.
//!
//! usage: w8a8_full_token <package dir> <token id> [max_layers]

use engine_ascend::attn_device::AscendAttentionDispatcher;
use engine_core::attn_dispatch::{
    decode_attn_prefix_with, decode_attn_suffix_with, AttnLayerWeights, AttnStepInputs,
    KvCacheView, LayerParams,
};
use engine_core::dispatch::CpuDispatcher;
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
        .ok_or_else(|| anyhow::anyhow!("usage: w8a8_full_token <package dir> <token id> [max_layers]"))?;
    let token: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(1000);
    let max_layers: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(43);
    let pkg = Package::open(&dir)?;
    // THE LEARNED HEAD COLLAPSE'S WEIGHTS. `hc_head_fn/base/scale` are in the package (24 non-layer hc tensors); the DS4 path
    // called `hc_mean_collapse`, a plain average, and the module that owns it says so in its own docstring. Loaded here, with
    // the mean as the fallback so a checkpoint without them still runs.
    let hc_head = (|| -> anyhow::Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        Ok((
            pkg.f32_tensor("hc_head_fn")?,
            pkg.f32_tensor("hc_head_base")?,
            pkg.f32_tensor("hc_head_scale")?,
        ))
    })();
    match &hc_head {
        Ok((f, b, sc)) => println!("head collapse: LEARNED (hc_head_fn {} values, base {}, scale {})", f.len(), b.len(), sc.len()),
        Err(e) => println!("head collapse: PLAIN MEAN -- the learned weights are not loadable ({e})"),
    }
    let views = PackageViews::new(&pkg);

    let d = 4096usize; let n_hc = 4usize; let heads = 64usize; let hd = 512usize;
    let lora_q = 1024usize; let d_ffn = 2048usize;

    let emb = pkg.dequant_tensor("embed.weight").or_else(|_| pkg.f32_tensor("embed.weight"))?;
    let vocab = emb.len() / d;
    anyhow::ensure!(token < vocab, "token {token} past the vocabulary");
    let x0 = &emb[token * d..(token + 1) * d];

    // THE SEAM, in use: the attention dispatcher holds the kernel dispatcher. With `CpuDispatcher`
    // inside it must reproduce the previous token exactly; swapping in `AscendDispatcher` is what
    // moves the routed methods onto the device.
    let attn = AscendAttentionDispatcher::new(CpuDispatcher);

    let mut cur_hc = vec![0.0f32; n_hc * d];
    for h in 0..n_hc { cur_hc[h * d..(h + 1) * d].copy_from_slice(x0); }

    for layer_idx in 0..max_layers {
        let params = LayerParams {
            layer_idx: layer_idx as u32,
            d_embd: d as u32, n_hc: n_hc as u32, n_head: heads as u32, head_dim: hd as u32,
            n_rot: 64, n_lora_q: lora_q as u32, n_lora_kv: hd as u32,
            hc_sinkhorn_iter: 20, hc_eps: 1e-6, rms_eps: 1e-6,
            rope_orig_ctx: 65536, rope_freq_base: 10000.0, rope_freq_scale: 1.0 / 16.0,
            rope_ext_factor: 0.0, rope_attn_factor: 1.0,
            compress_ratio: match layer_idx { 0 | 1 => 0, n if n % 2 == 0 => 4, _ => 128 },
            n_out_group: 8,
        };
        let view = views.layer(layer_idx as u32)?.into_owned();
        let layer = AttnLayerWeights::from_layer_view(&views, &view, params, d_ffn as u32, f32::INFINITY, false)?;

        // The layer's own attention norm, q/kv projections -- the caller's job in this API.
        let p = format!("layers.{layer_idx}");
        let gamma = pkg.f32_tensor(&format!("{p}.attn_norm.weight"))?;
        // THE COLLAPSE, not slot 0. The engine's prefix collapses all `n_hc` slots internally; the
        // q/kv projections the caller computes must see the SAME collapsed hidden state, or the two
        // halves of the attention block disagree about what the token is.
        let collapsed = match &hc_head {
            Ok((f, b, sc)) => engine_core::hyper_connection::head_collapse(
                &cur_hc.iter().map(|v| *v as f64).collect::<Vec<_>>(), f, b, sc, n_hc, d, 1e-6)
                .into_iter().map(|v| v as f32).collect(),
            Err(_) => engine_core::decode_step::hc_mean_collapse(&cur_hc, n_hc, d),
        };
        let n = rms_norm(&collapsed, &gamma, 1e-6);
        let wq_a = pkg.dequant_tensor(&format!("{p}.attn.wq_a.weight"))?;
        let qn = pkg.f32_tensor(&format!("{p}.attn.q_norm.weight"))?;
        let q_ln = rms_norm(&matvec(&wq_a, &n, lora_q), &qn, 1e-6);
        let wq_b = pkg.dequant_tensor(&format!("{p}.attn.wq_b.weight"))?;
        let q_heads = matvec(&wq_b, &q_ln, heads * hd);
        let wkv = pkg.dequant_tensor(&format!("{p}.attn.wkv.weight"))?;
        let kv_raw_row = matvec(&wkv, &n, hd);

        let mut kvc = vec![0.0f32; hd];
        let zero = vec![0.0f32; d];
        let mut step = AttnStepInputs {
            cur_hc: &cur_hc, kv_raw_row: &kv_raw_row, q_heads: &q_heads, pos: 0,
            kv_view: KvCacheView { raw: &mut kvc, raw_cap: 1, pos: 0 },
            n_raw: 0, raw_start: 0, routed_out: &zero,
            kv_comp_rows: None, n_comp: 0, comp_selected: None, n_selected: 0,
        };
        let prefix = decode_attn_prefix_with(&attn, &layer, &mut step);

        // MoE: hash routing for the first three layers, router scores after that.
        let fg = pkg.f32_tensor(&format!("{p}.ffn_norm.weight"))?;
        let hn = rms_norm(&prefix.normed, &fg, 1e-6);
        let mut routed = expert_out(&pkg, &format!("{p}.ffn.shared_experts"), &hn, d_ffn)?;
        let picked: Vec<usize> = if layer_idx < 3 {
            let raw = pkg.raw(&format!("{p}.ffn.gate.tid2eid"))?.1;
            let ids: Vec<i32> = raw.chunks_exact(4).map(|c| i32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
            (0..6).map(|k| ids[token * 6 + k] as usize).collect()
        } else {
            let gate = pkg.f32_tensor(&format!("{p}.ffn.gate.weight"))
                .or_else(|_| pkg.dequant_tensor(&format!("{p}.ffn.gate.weight")))?;
            let logits = matvec(&gate, &hn, 256);
            let probs = softplus_sqrt(&logits);
            let mut idx: Vec<usize> = (0..256).collect();
            idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
            idx[..6].to_vec()
        };
        let w = if layer_idx < 3 {
            let gate = pkg.f32_tensor(&format!("{p}.ffn.gate.weight"))
                .or_else(|_| pkg.dequant_tensor(&format!("{p}.ffn.gate.weight")))?;
            hash_router_weights_from_probs(&softplus_sqrt(&matvec(&gate, &hn, 256)), &picked, 6)
        } else {
            let gate = pkg.f32_tensor(&format!("{p}.ffn.gate.weight"))
                .or_else(|_| pkg.dequant_tensor(&format!("{p}.ffn.gate.weight")))?;
            let probs = softplus_sqrt(&matvec(&gate, &hn, 256));
            let s: f32 = picked.iter().map(|&e| probs[e]).sum();
            picked.iter().map(|&e| probs[e] / s.max(6.103515625e-5)).collect()
        };
        for (i, &e) in picked.iter().enumerate() {
            let y = expert_out(&pkg, &format!("{p}.ffn.experts.{e}"), &hn, d_ffn)?;
            for (a, b) in routed.iter_mut().zip(&y) { *a += w[i] * b; }
        }

        let out = decode_attn_suffix_with(&attn, &layer, &prefix, &routed, 0);
        anyhow::ensure!(out.len() == n_hc * d, "layer {layer_idx} returned {} floats", out.len());
        let r = (out.iter().map(|v| v * v).sum::<f32>() / out.len() as f32).sqrt();
        println!("layer {layer_idx:>2}: experts {picked:?} rms {r:.4} finite {}", out.iter().all(|v| v.is_finite()));
        cur_hc = out;
    }

    // Head: final norm then head.weight, on the first hc slot's collapse.
    let fgn = pkg.f32_tensor("norm.weight")?;
    let collapsed_final = match &hc_head {
        Ok((f, b, sc)) => engine_core::hyper_connection::head_collapse(
            &cur_hc.iter().map(|v| *v as f64).collect::<Vec<_>>(), f, b, sc, n_hc, d, 1e-6)
            .into_iter().map(|v| v as f32).collect(),
        Err(_) => engine_core::decode_step::hc_mean_collapse(&cur_hc, n_hc, d),
    };
    let hf = rms_norm(&collapsed_final, &fgn, 1e-6);
    let head = pkg.f32_tensor("head.weight")?;
    let logits = matvec(&head, &hf, vocab);
    let best = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap();
    println!("TOKEN from {max_layers} layers of the real model: {best} (vocab {vocab})");
    println!("PASS: the real layers ran through the engine and the head produced a token");
    Ok(())
}
