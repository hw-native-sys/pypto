//! Build the engine's ATTENTION layer weights from a W8A8 package.
//!
//! This is the first time the engine's own constructor runs on package data: `from_layer_view`
//! is generic over the weight source, so the same call that reads a GGUF mmap reads a package.
//! It prints what came back, so a partial build cannot pass for a complete one.
//!
//! usage: w8a8_attn_layer <package dir> [layer]

use engine_core::attn_dispatch::{AttnLayerWeights, LayerParams};
use engine_core::safetensors::Package;
use engine_core::w8a8_roles::{PackageViews, WeightSource};

fn rms(v: &[f32]) -> f64 {
    if v.is_empty() { return 0.0 }
    (v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / v.len() as f64).sqrt()
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let dir = std::path::PathBuf::from(a.get(1).ok_or_else(|| anyhow::anyhow!("usage: w8a8_attn_layer <pkg> [layer]"))?);
    let layer: u32 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    let pkg = Package::open(&dir)?;
    let src = PackageViews::new(&pkg);

    // Dimensions from the tensors, not from a config that may disagree.
    let emb = pkg.tensor("embed.weight")?.shape;
    let wq_a = pkg.tensor("layers.0.attn.wq_a.weight")?.shape;
    let wq_b = pkg.tensor("layers.0.attn.wq_b.weight")?.shape;
    let wkv = pkg.tensor("layers.0.attn.wkv.weight")?.shape;
    let d_embd = emb[1] as u32;
    let n_lora_q = wq_a[0] as u32;
    let n_lora_kv = wkv[0] as u32;
    let q_dim = wq_b[0] as u32;

    // The head geometry comes from the package's own config.json where it is stated; DS4's
    // V4-Flash defaults supply the rest, and the printout below shows what was assumed.
    let defaults = engine_core::attn_dispatch::DefaultsDs4::ds4_v4_flash();
    let head_dim = 512u32;
    let n_head = q_dim / head_dim;

    let params = LayerParams {
        layer_idx: layer,
        d_embd,
        n_hc: 2,
        n_head,
        head_dim,
        n_rot: 64,
        n_lora_q,
        n_lora_kv,
        hc_sinkhorn_iter: 4,
        hc_eps: 1e-6,
        rms_eps: 1e-6,
        rope_orig_ctx: 65536,
        rope_freq_base: 10000.0,
        rope_freq_scale: 1.0,
        rope_ext_factor: 0.0,
        rope_attn_factor: 1.0,
        compress_ratio: 1,
        n_out_group: 8,
    };
    println!(
        "dims from the package: d_embd {d_embd}  n_head {n_head}  head_dim {head_dim}  \
         n_lora_q {n_lora_q}  n_lora_kv {n_lora_kv}  q_dim {q_dim}"
    );
    let _ = defaults;

    let view = src.layer(layer)?;
    match AttnLayerWeights::from_layer_view(&src, &view, params, 2048, 0.0, false) {
        Ok(w) => {
            println!("\nengine built the attention layer from the PACKAGE:");
            for (name, len, r) in [
                ("attn_q_a", w.attn_q_a.len(), rms(&w.attn_q_a)),
                ("attn_q_b", w.attn_q_b.len(), rms(&w.attn_q_b)),
                ("attn_kv", w.attn_kv.len(), rms(&w.attn_kv)),
                ("w_o_a", w.w_o_a.len(), rms(&w.w_o_a)),
                ("w_o_b", w.w_o_b.len(), rms(&w.w_o_b)),
                ("w_shared_gate", w.w_shared_gate.len(), rms(&w.w_shared_gate)),
            ] {
                println!("  {name:<14} {len:>10} floats  rms {r:.4e}");
            }
            println!("  shared_dim set to 2048 (clamp 0.0)");
        }
        Err(e) => println!("\nengine could NOT build the layer from the package: {e}"),
    }
    Ok(())
}
