//! Load one layer's roles from a W8A8 package and report what they look like.
//!
//! Two questions, neither of which needs an oracle:
//!   * can EVERY role a layer needs be fetched and turned into f32 (w8a8, BF16, F32)?
//!   * are the resulting numbers physically sane -- weights of order 1e-2 and a
//!     projection of a unit-scale input that stays of order 1e-1, rather than collapsing?
//!
//! The second question exists because a package whose scales are uniformly too small
//! produces a model that RUNS and generates nonsense, which is this campaign's worst
//! failure mode. A magnitude check catches it before any token is generated.
//!
//! usage: w8a8_layer <package dir> [layer]

use engine_core::safetensors::Package;
use std::path::PathBuf;

fn rms(v: &[f32]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let s: f64 = v.iter().map(|x| (*x as f64) * (*x as f64)).sum();
    (s / v.len() as f64).sqrt()
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 2 {
        anyhow::bail!("usage: w8a8_layer <package dir> [layer]");
    }
    let dir = PathBuf::from(&a[1]);
    let layer = a.get(2).cloned().unwrap_or_else(|| "0".to_string());
    let layer_idx: usize = layer.parse().unwrap_or(0);
    let pkg = Package::open(&dir)?;
    println!("package: {}", dir.display());
    println!("layer:   {layer}\n");

    // (role, tensor name) -- roles the engine's decoder needs for one layer.
    let roles: [(&str, String); 12] = [
        ("attn_norm", format!("layers.{layer}.attn_norm.weight")),
        ("wq_a", format!("layers.{layer}.attn.wq_a.weight")),
        ("wkv", format!("layers.{layer}.attn.wkv.weight")),
        ("wq_b", format!("layers.{layer}.attn.wq_b.weight")),
        ("wo_a", format!("layers.{layer}.attn.wo_a.weight")),
        ("wo_b", format!("layers.{layer}.attn.wo_b.weight")),
        ("ffn_norm", format!("layers.{layer}.ffn_norm.weight")),
        ("ffn_gate", format!("layers.{layer}.ffn.gate.weight")),
        ("shared_w1", format!("layers.{layer}.ffn.shared_experts.w1.weight")),
        ("shared_w2", format!("layers.{layer}.ffn.shared_experts.w2.weight")),
        ("shared_w3", format!("layers.{layer}.ffn.shared_experts.w3.weight")),
        ("expert0_w1", format!("layers.{layer}.ffn.experts.0.w1.weight")),
    ];

    let mut wq_a: Option<(Vec<f32>, Vec<u64>)> = None;
    println!("{:<12} {:<10} {:<16} {:>12} {:>12}", "role", "dtype", "shape", "n", "rms");
    for (role, name) in &roles {
        let t = match pkg.tensor(name) {
            Ok(t) => t,
            Err(_) => {
                println!("{role:<12} MISSING    {name}");
                continue;
            }
        };
        let v = if t.dtype == engine_core::safetensors::Dtype::I8 {
            pkg.dequant_tensor(name)?
        } else {
            pkg.f32_tensor(name)?
        };
        println!(
            "{:<12} {:<10} {:<16} {:>12} {:>12.5e}",
            role,
            format!("{:?}", t.dtype),
            format!("{:?}", t.shape),
            v.len(),
            rms(&v)
        );
        if *role == "wq_a" {
            wq_a = Some((v, t.shape.clone()));
        }
    }

    // The same layer through the SOURCE the converted decoder loop will call: roles resolve to
    // owned f32 (I8 dequantised, BF16 widened), and the raw bytes stay available for the planes
    // the engine keeps quantised on the host.
    {
        use engine_core::w8a8_roles::PackageSource;
        let src = PackageSource::new(&pkg);
        println!("\nvia PackageSource:");
        for role in ["attn_norm", "attn_q_a", "attn_q_b", "attn_kv", "attn_output_a",
                     "ffn_gate_inp", "ffn_shared_w1"] {
            let role = if role == "ffn_shared_w1" { "ffn_gate_shexp" } else { role };
            match src.dequant(layer_idx, role) {
                Ok(v) => println!("  {role:<16} {} floats  rms {:.4e}", v.len(), rms(&v)),
                Err(e) => println!("  {role:<16} ERROR {e}"),
            }
        }
        match src.dequant_expert(layer_idx, 0, 1) {
            Ok(v) => println!("  expert w1        {} floats  rms {:.4e}", v.len(), rms(&v)),
            Err(e) => println!("  expert w1        ERROR {e}"),
        }
        match src.raw(layer_idx, "attn_q_a") {
            Ok(b) => println!("  raw attn_q_a     {} bytes (int8, dequantised on device)", b.len()),
            Err(e) => println!("  raw attn_q_a     ERROR {e}"),
        }
    }

    // A projection of a unit-scale input. This is the sanity check: |y| for a 4096-wide
    // row of N(0,1) inputs is ~sqrt(4096)*|w|, so if the weights are 27x too small the
    // layer's attention output lands 27x low and the residual stream stops moving.
    if let Some((w, shape)) = wq_a {
        let rows = shape[0] as usize;
        let cols: usize = shape.iter().skip(1).map(|d| *d as usize).product::<usize>().max(1);
        // deterministic, zero-mean, unit-variance input
        let x: Vec<f32> = (0..cols)
            .map(|k| (((k * 2654435761) % 1000) as f32 / 250.0) - 2.0)
            .collect();
        let mut y = vec![0.0f32; rows];
        for i in 0..rows {
            let mut acc = 0.0f64;
            for k in 0..cols {
                acc += w[i * cols + k] as f64 * x[k] as f64;
            }
            y[i] = acc as f32;
        }
        println!(
            "\nwq_a @ x : |x| rms {:.4} -> |y| rms {:.5e}   (rows {rows}, cols {cols})",
            rms(&x),
            rms(&y)
        );
    }
    Ok(())
}
