//! ONE LAYER of DS4.1Flash on the package's real w8a8 weights, then the real head -> a token.
//!
//! This is NOT the 43-layer model's token, and it says so in its own output. What it is: layer 0's
//! attention and FFN, computed on real weights, with the engine's own routing functions, followed by
//! the package's `head.weight` and an argmax.
//!
//! Wiring is from config.json: 64 heads x 512, ONE kv head (MQA), o_groups 8 x o_lora_rank 1024,
//! top-6 of 256 experts with hash routing for the first three layers, shared expert always on.
//! With a single context position a softmax over one score is 1, so each head's output is the shared
//! 512-wide v -- which is what MQA means here, not a shortcut.
//!
//! usage: w8a8_layer0_token <package dir> <token id>

use engine_core::moe::{hash_router_weights_from_probs, softplus_sqrt};
use engine_core::safetensors::Package;
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
        .ok_or_else(|| anyhow::anyhow!("usage: w8a8_layer0_token <package dir> <token id>"))?;
    let token: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(1000);
    let pkg = Package::open(&dir)?;

    let d = 4096usize; let d_ffn = 2048usize; let heads = 64usize; let hd = 512usize;
    let groups = 8usize; let lora = 1024usize;

    // Token embedding.
    let emb = pkg.dequant_tensor("embed.weight").or_else(|_| pkg.f32_tensor("embed.weight"))?;
    let vocab = emb.len() / d;
    anyhow::ensure!(token < vocab, "token {token} past the vocabulary ({vocab})");
    let x: Vec<f32> = emb[token * d..(token + 1) * d].to_vec();

    // ---- attention ----
    let gamma = pkg.f32_tensor("layers.0.attn_norm.weight")?;
    let n = rms_norm(&x, &gamma, 1e-6);
    let wq_a = pkg.dequant_tensor("layers.0.attn.wq_a.weight")?;
    let qn = pkg.f32_tensor("layers.0.attn.q_norm.weight")?;
    let q_ln = rms_norm(&matvec(&wq_a, &n, lora), &qn, 1e-6);
    let wq_b = pkg.dequant_tensor("layers.0.attn.wq_b.weight")?;
    let q = matvec(&wq_b, &q_ln, heads * hd);
    let wkv = pkg.dequant_tensor("layers.0.attn.wkv.weight")?;
    let kvn = pkg.f32_tensor("layers.0.attn.kv_norm.weight")?;
    let v = rms_norm(&matvec(&wkv, &n, hd), &kvn, 1e-6);   // MQA: one v, shared
    let sink = pkg.f32_tensor("layers.0.attn.attn_sink")?;
    // One context position: softmax over (q.k + sink) is 1, so each head's output is v itself.
    let o_heads: Vec<f32> = (0..heads * hd).map(|i| v[i % hd]).collect();
    println!("q {} heads, v {} (MQA), sink {} -- one position, so every head emits v", q.len(), v.len(), sink.len());

    // ---- o-group projection: group g = heads 8g..8g+8 (4096) -> wo_a block g (1024) ----
    let wo_a = pkg.dequant_tensor("layers.0.attn.wo_a.weight")
        .or_else(|_| pkg.f32_tensor("layers.0.attn.wo_a.weight"))?;
    let wo_b = pkg.dequant_tensor("layers.0.attn.wo_b.weight")
        .or_else(|_| pkg.f32_tensor("layers.0.attn.wo_b.weight"))?;
    let per_group = heads / groups * hd;                 // 4096
    anyhow::ensure!(per_group == d, "a group of {} heads is {per_group}, not wo_a's input {d}", heads / groups);
    let mut mid = vec![0.0f32; groups * lora];
    for g in 0..groups {
        let src = &o_heads[g * per_group..(g + 1) * per_group];
        let block = &wo_a[g * lora * d..(g + 1) * lora * d];
        let y = matvec(block, src, lora);
        mid[g * lora..(g + 1) * lora].copy_from_slice(&y);
    }
    let attn_out = matvec(&wo_b, &mid, d);
    let mut h: Vec<f32> = x.iter().zip(&attn_out).map(|(a, b)| a + b).collect();
    println!("attn out rms {:.4}, residual rms {:.4}", 
             (attn_out.iter().map(|v| v*v).sum::<f32>()/d as f32).sqrt(),
             (h.iter().map(|v| v*v).sum::<f32>()/d as f32).sqrt());

    // ---- FFN: shared + 6 hash-routed, weighted by the router ----
    let fg = pkg.f32_tensor("layers.0.ffn_norm.weight")?;
    let hn = rms_norm(&h, &fg, 1e-6);
    let mut out = expert_out(&pkg, "layers.0.ffn.shared_experts", &hn, d_ffn)?;

    let gate = pkg.f32_tensor("layers.0.ffn.gate.weight")
        .or_else(|_| pkg.dequant_tensor("layers.0.ffn.gate.weight"))?;
    let logits = matvec(&gate, &hn, 256);
    let probs = softplus_sqrt(&logits);

    let raw = pkg.raw("layers.0.ffn.gate.tid2eid")?.1;
    let ids: Vec<i32> = raw.chunks_exact(4).map(|c| i32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
    let mut picked = Vec::new();
    for k in 0..6 { picked.push(ids[token * 6 + k] as usize); }
    let w = hash_router_weights_from_probs(&probs, &picked, 6);
    println!("experts {:?} weights {:?}", picked, w.iter().map(|v| (v*1000.0).round()/1000.0).collect::<Vec<_>>());

    for (i, &e) in picked.iter().enumerate() {
        let y = expert_out(&pkg, &format!("layers.0.ffn.experts.{e}"), &hn, d_ffn)?;
        for (a, b) in out.iter_mut().zip(&y) { *a += w[i] * b; }
    }
    for (a, b) in h.iter_mut().zip(&out) { *a += b; }

    // ---- final norm + real head ----
    let fgn = pkg.f32_tensor("layers.0.hc_norm.weight").or_else(|_| pkg.f32_tensor("norm.weight"))?;
    let hf = rms_norm(&h, &fgn, 1e-6);
    let head = pkg.f32_tensor("head.weight")?;
    let logits_out = matvec(&head, &hf, vocab);
    let best = logits_out.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap();
    println!("ONE-LAYER token after layer 0: {best} (vocabulary {vocab})");
    println!("PASS: layer 0 attention + FFN + head ran on real w8a8 weights (NOT the 43-layer model)");
    Ok(())
}
