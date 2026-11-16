//! Layer 0's FFN half with the package's real w8a8 weights.
//!
//! Layer 0 is a HASH layer (`num_hash_layers = 3` in config.json), so its experts come from
//! `ffn.gate.tid2eid` [vocab, 6] indexed by the token id, not from a top-k over router scores.
//! The shared expert always runs (`n_shared_experts = 1`).
//!
//! Per expert: y = W2 . (silu(W1 . n) * (W3 . n)), with w1/w3 [d_ffn=2048, d=4096] and w2 [4096, 2048].
//! Every projection is I8 with the package's per-output-channel scale/offset, which is exactly the
//! layout the w8a8 kernels implement.
//!
//! usage: w8a8_layer0_ffn_real <package dir> <token id>

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

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

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
        .ok_or_else(|| anyhow::anyhow!("usage: w8a8_layer0_ffn_real <package dir> <token id>"))?;
    let token: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(1000);
    let pkg = Package::open(&dir)?;

    let d = 4096usize;
    let d_ffn = 2048usize;
    let mut x = vec![0.0f32; d];
    for (i, v) in x.iter_mut().enumerate() { *v = ((i % 17) as f32 - 8.0) * 0.01; }

    let gamma = pkg.f32_tensor("layers.0.ffn_norm.weight")?;
    let h = rms_norm(&x, &gamma, 1e-6);

    // Shared expert: always runs.
    let shared = expert_out(&pkg, "layers.0.ffn.shared_experts", &h, d_ffn)?;
    println!("shared expert out: {} finite={} rms={:.4}", shared.len(),
             shared.iter().all(|v| v.is_finite()),
             (shared.iter().map(|v| v * v).sum::<f32>() / shared.len() as f32).sqrt());

    // Hash routing: tid2eid[token] gives the six experts for this token.
    let table = pkg.tensor("layers.0.ffn.gate.tid2eid")?;
    let raw = pkg.raw("layers.0.ffn.gate.tid2eid")?.1;
    let ids: Vec<i32> = raw.chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let row = token * 6;
    anyhow::ensure!(row + 6 <= ids.len(), "token {token} past the routing table");
    let picked: Vec<usize> = ids[row..row + 6].iter().map(|&e| e as usize).collect();
    println!("tid2eid {:?} shape {:?} -> token {token} uses experts {:?}", table.shape, table.dtype, picked);

    let mut acc = vec![0.0f32; d];
    for &e in &picked {
        let y = expert_out(&pkg, &format!("layers.0.ffn.experts.{e}"), &h, d_ffn)?;
        anyhow::ensure!(y.iter().all(|v| v.is_finite()), "expert {e} produced a non-finite value");
        for (a, b) in acc.iter_mut().zip(&y) { *a += b; }
    }
    println!("{} routed experts summed: finite={}", picked.len(), acc.iter().all(|v| v.is_finite()));

    let mut out = shared;
    for (a, b) in out.iter_mut().zip(&acc) { *a += b; }
    println!("PASS: shared + {} routed experts ran on the package's own w8a8 weights; out rms={:.4}",
             picked.len(), (out.iter().map(|v| v * v).sum::<f32>() / out.len() as f32).sqrt());
    Ok(())
}
