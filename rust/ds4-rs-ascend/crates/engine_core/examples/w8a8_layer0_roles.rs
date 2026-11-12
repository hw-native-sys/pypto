//! Layer 0's real roles, from the package, with dtypes and shapes.
//!
//! The engine's CPU decode layer is a SIMPLIFIED model -- one dense `w_attn`, stacked expert planes --
//! while the package holds MLA projections and 256 separate expert tensors. Mapping one onto the other
//! is a modelling decision, so this prints the facts it should be made from rather than the names
//! remembered from an earlier round.
//!
//! usage: w8a8_layer0_roles <package dir>

use engine_core::safetensors::{Dtype, Package};
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("usage: w8a8_layer0_roles <package dir>"))?;
    let pkg = Package::open(&dir)?;

    let mut rows: Vec<(String, String, usize, Vec<usize>)> = Vec::new();
    for name in pkg.names() {
        if !name.starts_with("layers.0.") {
            continue;
        }
        if let Ok(meta) = pkg.tensor(name) {
            let dt = match meta.dtype {
                Dtype::I8 => "I8",
                Dtype::U8 => "U8",
                Dtype::F32 => "F32",
                Dtype::F16 => "F16",
                Dtype::BF16 => "BF16",
                Dtype::I32 => "I32",
                Dtype::Other => "?",
            };
            let nbytes = (meta.offsets.1.saturating_sub(meta.offsets.0)) as usize;
            let shape: Vec<usize> = meta.shape.iter().map(|&d| d as usize).collect();
            rows.push((name.to_string(), dt.to_string(), nbytes, shape));
        }
    }
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    println!("layer 0 tensors: {}", rows.len());
    for (name, dt, nbytes, shape) in &rows {
        let short = name.trim_start_matches("layers.0.");
        println!("  {short:<44} {dt:<4} {:>12} B  {shape:?}", nbytes);
    }
    let total: usize = rows.iter().map(|r| r.2).sum();
    println!("layer 0 total: {:.1} MB", total as f64 / 1e6);
    Ok(())
}
