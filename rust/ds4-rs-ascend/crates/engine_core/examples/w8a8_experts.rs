//! Why the routed experts cannot be stacked, and what lazy access costs instead.
//!
//! `LayerWeights` holds the routed experts as STACKED f32 planes: `[n_experts][d_ffn][d_model]`.
//! For this model that is 256 x 2048 x 4096 x 4 B = 8.6 GB PER PLANE PER LAYER, and there are three
//! planes and 43 layers -- 1.1 TB, against a ~128 GB resident budget. The package sidesteps this by
//! storing 256 separate int8 tensors per layer, and a token reads only the SELECTED experts.
//!
//! This prints both numbers so the design choice is a measurement rather than an opinion.
//!
//! usage: w8a8_experts <package dir> [layer] [top_k]

use engine_core::safetensors::Package;
use engine_core::w8a8_roles::PackageSource;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let dir = std::path::PathBuf::from(a.get(1).ok_or_else(|| anyhow::anyhow!("usage: w8a8_experts <pkg> [layer] [k]"))?);
    let layer: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    let top_k: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(6);
    let pkg = Package::open(&dir)?;
    let src = PackageSource::new(&pkg);

    let w1 = pkg.tensor(&format!("layers.{layer}.ffn.experts.0.w1.weight"))?;
    let n_experts = pkg
        .names()
        .iter()
        .filter(|s| s.starts_with(&format!("layers.{layer}.ffn.experts.")) && s.ends_with(".w1.weight"))
        .count();
    let (d_ffn, d_model) = (w1.shape[0] as u64, w1.shape[1] as u64);

    let stacked_bytes = 3 * n_experts as u64 * d_ffn * d_model * 4; // three planes, f32
    println!("layer {layer}: {n_experts} routed experts, {d_ffn} x {d_model}");
    println!(
        "  STACKED f32 (what LayerWeights holds): {:.1} GB for this layer alone, {:.1} TB for 43",
        stacked_bytes as f64 / 1e9,
        stacked_bytes as f64 * 43.0 / 1e12
    );
    println!(
        "  ON DISK as int8 per expert:           {:.3} GB for this layer",
        3.0 * n_experts as f64 * d_ffn as f64 * d_model as f64 / 1e9
    );

    // What a token actually reads: `top_k` experts, three planes.
    let t0 = std::time::Instant::now();
    let mut bytes = 0usize;
    let mut sum = 0.0f64;
    for e in 0..top_k {
        for plane in 1..=3u8 {
            let v = src.dequant_expert(layer, e, plane)?;
            bytes += v.len() * 4;
            sum += v.iter().map(|x| (*x as f64).abs()).sum::<f64>();
        }
    }
    let dt = t0.elapsed();
    println!(
        "  A TOKEN reads {top_k} experts x 3 planes = {:.3} GB of f32, in {:.2} s (rms-sum {:.3e})",
        bytes as f64 / 1e9,
        dt.as_secs_f64(),
        sum / bytes as f64
    );
    println!(
        "  ratio: stacked/lazy = {:.0}x compact. Lazy access is not an optimisation here; the\n  \
         stacked form cannot exist on this box at all.",
        stacked_bytes as f64 / bytes as f64
    );

    // The same token's experts as RAW BYTES + per-channel scale/offset: exactly the three
    // arguments the certified kernel takes. The host never builds the f32 plane.
    let t1 = std::time::Instant::now();
    let mut raw_bytes = 0usize;
    let mut scales = 0usize;
    for e in 0..top_k {
        for plane in 1..=3u8 {
            let (q, sc, off) = src.raw_expert(layer, e, plane)?;
            raw_bytes += q.len();
            scales += (sc.len() + off.len()) * 4;
        }
    }
    let dt1 = t1.elapsed();
    println!(
        "  RAW path: {:.3} GB of int8 + {:.1} MB of scale/offset, in {:.3} s  ({:.0}x faster than dequantising)",
        raw_bytes as f64 / 1e9,
        scales as f64 / 1e6,
        dt1.as_secs_f64(),
        dt.as_secs_f64() / dt1.as_secs_f64().max(1e-9)
    );
    Ok(())
}
