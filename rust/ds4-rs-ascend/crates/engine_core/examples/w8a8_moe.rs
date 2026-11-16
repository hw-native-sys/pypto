//! Run one layer's routed MoE over a real hidden state, from the W8A8 package.
//!
//! This is the executor the campaign has been walking toward: router -> top-k -> per-expert
//! SwiGLU, driven entirely by the package's own tensors, with each expert fetched lazily. It runs
//! on the HOST in f32 so the arithmetic path is established before the same three projections are
//! handed to the certified w8a8 kernel.
//!
//! usage: w8a8_moe <package dir> [layer] [top_k]

use engine_core::safetensors::Package;
use engine_core::w8a8_roles::PackageSource;

fn rms(v: &[f32]) -> f64 {
    if v.is_empty() { return 0.0 }
    (v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / v.len() as f64).sqrt()
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// y = W · x with W row-major [rows, cols].
fn matvec(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; rows];
    for i in 0..rows {
        let mut acc = 0.0f64;
        for k in 0..cols {
            acc += w[i * cols + k] as f64 * x[k] as f64;
        }
        y[i] = acc as f32;
    }
    y
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let dir = std::path::PathBuf::from(a.get(1).ok_or_else(|| anyhow::anyhow!("usage: w8a8_moe <pkg> [layer] [top_k]"))?);
    let layer: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    let top_k: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(6);
    let pkg = Package::open(&dir)?;
    let src = PackageSource::new(&pkg);

    let d_model = pkg.tensor("embed.weight")?.shape[1] as usize;
    let w1_shape = pkg.tensor(&format!("layers.{layer}.ffn.experts.0.w1.weight"))?.shape.clone();
    let (d_ffn, d_in) = (w1_shape[0] as usize, w1_shape[1] as usize);
    anyhow::ensure!(d_in == d_model, "expert reads {d_in}, d_model is {d_model}");

    // A hidden state with the right scale: the layer's own attn_norm gamma, which is O(1) here.
    let h = src.dequant(layer, "ffn_norm")?;
    anyhow::ensure!(h.len() == d_model, "ffn_norm has {} floats", h.len());

    // Router: ffn.gate.weight is [n_experts, d_model] and NOT w8a8 (BF16 in this package).
    let router = pkg.f32_tensor(&format!("layers.{layer}.ffn.gate.weight"))?;
    let n_experts = pkg
        .names()
        .iter()
        .filter(|s| s.starts_with(&format!("layers.{layer}.ffn.experts.")) && s.ends_with(".w1.weight"))
        .count();
    anyhow::ensure!(router.len() == n_experts * d_model, "router is {} floats for {n_experts} experts", router.len());
    let logits = matvec(&router, &h, n_experts, d_model);
    println!("layer {layer}: {n_experts} experts, d_model {d_model}, d_ffn {d_ffn}");
    println!("  router logits rms {:.4e}", rms(&logits));

    // Top-k by logit, then a softmax over just those.
    let mut idx: Vec<usize> = (0..n_experts).collect();
    idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    idx.truncate(top_k);
    let mx = idx.iter().map(|&i| logits[i]).fold(f32::MIN, f32::max);
    let exps: Vec<f32> = idx.iter().map(|&i| (logits[i] - mx).exp()).collect();
    let denom: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|e| e / denom).collect();
    println!("  top-{top_k} experts {:?}", idx);
    println!("  routing probs {:?}", probs.iter().map(|p| (p * 1000.0).round() / 1000.0).collect::<Vec<_>>());

    // Per expert: gate = w1·h, up = w3·h, act = silu(gate)*up, out += p * (w2·act).
    let mut out = vec![0.0f64; d_model];
    let t0 = std::time::Instant::now();
    for (j, &e) in idx.iter().enumerate() {
        let w1 = src.dequant_expert(layer, e, 1)?;
        let w3 = src.dequant_expert(layer, e, 3)?;
        let w2 = src.dequant_expert(layer, e, 2)?;
        anyhow::ensure!(w1.len() == d_ffn * d_model, "w1 {} floats", w1.len());
        anyhow::ensure!(w2.len() == d_model * d_ffn, "w2 {} floats", w2.len());
        let gate = matvec(&w1, &h, d_ffn, d_model);
        let up = matvec(&w3, &h, d_ffn, d_model);
        let act: Vec<f32> = gate.iter().zip(&up).map(|(g, u)| silu(*g) * u).collect();
        let down = matvec(&w2, &act, d_model, d_ffn);
        let p = probs[j] as f64;
        for i in 0..d_model {
            out[i] += p * down[i] as f64;
        }
        println!(
            "  expert {e:>3} p {:.3}  gate rms {:.3e}  act rms {:.3e}  down rms {:.3e}",
            probs[j], rms(&gate), rms(&act), rms(&down)
        );
    }
    let out: Vec<f32> = out.iter().map(|v| *v as f32).collect();
    println!(
        "\nMoE output: {} floats  rms {:.4e}  finite {}  in {:.1} s",
        out.len(),
        rms(&out),
        out.iter().all(|v| v.is_finite()),
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}
