//! The same three numbers an independent Q8_0 decoder prints, from the engine's own path.
//!
//! usage: dequant_crosscheck <gguf> <tensor>

use engine_core::glm5_stream::ShardedGguf;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let path = PathBuf::from(std::env::args().nth(1).ok_or_else(|| anyhow::anyhow!("usage: <gguf> <tensor>"))?);
    let name = std::env::args().nth(2).ok_or_else(|| anyhow::anyhow!("tensor name"))?;
    let g = ShardedGguf::open(&path)?;
    let v = g.dequant_name(&name)?;
    anyhow::ensure!(v.len() >= 4096, "tensor has only {} values", v.len());
    let head: Vec<f64> = v[..3].iter().map(|x| *x as f64).collect();
    let sum: f64 = v[..4096].iter().map(|x| *x as f64).sum();
    let rms = (v[..4096].iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / 4096.0).sqrt();
    println!("first 3: {:.9e} {:.9e} {:.9e}", head[0], head[1], head[2]);
    println!("sum of first 4096: {sum:.9e}");
    println!("rms of first 4096: {rms:.9e}");

    // ONE REAL PROJECTION: y[r] = sum_{c<4096} W[r, c] * x[c], with the same synthetic x the example uses.
    // This exercises the dequantiser and the matvec TOGETHER on the checkpoint's own numbers, which is the
    // closest thing to a projection cross-check available without a reference model.
    let d = 4096usize;
    anyhow::ensure!(v.len() % d == 0, "tensor is not a whole number of {d}-wide rows");
    let stride = v.len() / 4096;   // kda_v is [4096, 8192]: 8192 values per row
    let x: Vec<f64> = (0..d).map(|i| (((i % 11) as f64) - 5.0) * 0.01).collect();
    let rows = v.len() / stride;
    let y: Vec<f64> = (0..rows)
        .map(|r| (0..d).map(|c| v[r * stride + c] as f64 * x[c]).sum())
        .collect();
    let sum: f64 = y[..8192.min(y.len())].iter().sum();
    let rms = (y[..8192.min(y.len())].iter().map(|a| a * a).sum::<f64>() / 8192.min(y.len()) as f64).sqrt();
    println!("proj first 3: {:.9e} {:.9e} {:.9e}", y[0], y[1], y[2]);
    println!("proj sum of 8192: {sum:.9e}");
    println!("proj rms of 8192: {rms:.9e}");
    Ok(())
}
