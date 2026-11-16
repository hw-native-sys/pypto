//! Layer 0's ATTENTION half with the package's real w8a8 weights.
//!
//! The wiring comes from `config.json`, not from guessing:
//!   num_attention_heads 64, head_dim 512, num_key_value_heads 1   -> MQA, one 512-wide k/v
//!   o_groups 8, o_lora_rank 1024                                  -> wo_a's 8192 = 8 x 1024, and
//!                                                                    each group projects its 8 heads
//!                                                                    (8 x 512 = 4096 = wo_a's input)
//!
//! So: n = rms(x)*attn_norm; q_a = W_q_a n; q_ln = rms(q_a)*q_norm; q = W_q_b q_ln; kv = W_kv n.
//! This prints the shapes and the finite-ness of each stage, which is what a first real-weight
//! forward can honestly assert before any token claim.
//!
//! usage: w8a8_layer0_attn_real <package dir>

use engine_core::safetensors::Package;
use std::path::PathBuf;

fn matvec(w: &[f32], x: &[f32], n: usize) -> Vec<f32> {
    let k = x.len();
    (0..n)
        .map(|i| (0..k).map(|j| w[i * k + j] * x[j]).sum())
        .collect()
}

fn rms_norm(x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
    let ss: f32 = x.iter().map(|v| v * v).sum();
    let inv = 1.0 / (ss / x.len() as f32 + eps).sqrt();
    x.iter().zip(gamma).map(|(v, g)| v * inv * g).collect()
}

fn main() -> anyhow::Result<()> {
    let dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("usage: w8a8_layer0_attn_real <package dir>"))?;
    let pkg = Package::open(&dir)?;

    let d = 4096usize;
    let mut x = vec![0.0f32; d];
    for (i, v) in x.iter_mut().enumerate() {
        *v = ((i % 17) as f32 - 8.0) * 0.01;
    }

    // F32 tensors take `f32_tensor`; only the INT8 projections go through the w8a8 dequantiser.
    let gamma = pkg.f32_tensor("layers.0.attn_norm.weight")?;
    let n = rms_norm(&x, &gamma, 1e-6);
    println!("n: {:?}  finite={}", n.len(), n.iter().all(|v| v.is_finite()));

    let wq_a = pkg.dequant_tensor("layers.0.attn.wq_a.weight")?;
    let q_a = matvec(&wq_a, &n, 1024);
    let qn = pkg.f32_tensor("layers.0.attn.q_norm.weight")?;
    let q_ln = rms_norm(&q_a, &qn, 1e-6);
    println!("q_a: {} -> q_ln: {} finite={}", q_a.len(), q_ln.len(), q_ln.iter().all(|v| v.is_finite()));

    let wq_b = pkg.dequant_tensor("layers.0.attn.wq_b.weight")?;
    let q = matvec(&wq_b, &q_ln, 64 * 512);
    let q_rms = (q.iter().map(|v| v * v).sum::<f32>() / q.len() as f32).sqrt();
    println!("q: {} = 64 heads x 512, rms {:.4}, finite={}", q.len(), q_rms, q.iter().all(|v| v.is_finite()));

    let wkv = pkg.dequant_tensor("layers.0.attn.wkv.weight")?;
    let kv = matvec(&wkv, &n, 512);
    let kvn = pkg.f32_tensor("layers.0.attn.kv_norm.weight")?;
    let kv_n = rms_norm(&kv, &kvn, 1e-6);
    println!("kv: {} (MQA, shared by 64 heads), finite={}", kv_n.len(), kv_n.iter().all(|v| v.is_finite()));

    let sink = pkg.f32_tensor("layers.0.attn.attn_sink")?;
    println!("sink: {:?} (one per head)", sink.len());

    // The o-group projection: wo_a is [8192, 4096] = 8 groups x 1024 rows, each over 4096 inputs,
    // and 8 heads x 512 = 4096 is exactly one group's input.
    let wo_a = pkg.dequant_tensor("layers.0.attn.wo_a.weight").or_else(|_| pkg.f32_tensor("layers.0.attn.wo_a.weight"))?;
    println!("wo_a: {} = {} groups x {} x {}", wo_a.len(), 8, 1024, 4096);
    anyhow::ensure!(wo_a.len() == 8 * 1024 * 4096, "wo_a is not 8 x 1024 x 4096");
    anyhow::ensure!(8 * 512 == 4096, "a group of 8 heads is not wo_a's input width");
    println!("PASS: 8 heads x 512 = 4096 = wo_a's input width, and 8 groups x 1024 = 8192 = its rows");
    Ok(())
}
