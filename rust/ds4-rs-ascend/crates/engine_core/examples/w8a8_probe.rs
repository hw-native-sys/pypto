//! Report what a W8A8 package holds, in the engine's own role vocabulary.
//!
//! Run where the weights are: `cargo run --release --example w8a8_probe -- /data0/...-w8a8`.
//!
//! This exists to answer one question before any model builder is written: does the
//! package carry, in a form this engine can consume, the tensors each engine role needs?
//! It prints SHAPES and TYPES only -- no weights are dequantised, so it is cheap on a
//! 300 GB package.

use engine_core::safetensors::{Dtype, Package};
use std::collections::BTreeMap;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("usage: w8a8_probe <package dir>"))?;
    let pkg = Package::open(&dir)?;
    println!("package: {}", dir.display());
    println!("tensors in index: {}", pkg.n_tensors());

    // Group by the trailing component so a 40-layer file reports once per role.
    let mut roles: BTreeMap<String, (Dtype, Vec<u64>, usize)> = BTreeMap::new();
    let mut skipped = 0usize;
    for name in pkg.names() {
        let t = match pkg.tensor(name) {
            Ok(t) => t,
            Err(_) => {
                skipped += 1;
                continue;
            }
        };
        // "layers.3.attn.wq_a.weight_scale" -> "layers.*.attn.wq_a.weight_scale"
        let role = name
            .split('.')
            .map(|c| if c.chars().all(|d| d.is_ascii_digit()) && !c.is_empty() { "*" } else { c })
            .collect::<Vec<_>>()
            .join(".");
        roles
            .entry(role)
            .and_modify(|e| e.2 += 1)
            .or_insert((t.dtype, t.shape.clone(), 1));
    }
    println!("distinct roles: {}", roles.len());
    if skipped > 0 {
        println!("tensors the index claims but a shard does not have: {skipped}");
    }
    for (role, (dtype, shape, count)) in &roles {
        println!("  {role:<48} {dtype:?} {shape:?}  x{count}");
    }

    // The roles the engine's decoder actually needs, and whether the package has them.
    let need = [
        "layers.*.attn.wq_a.weight",
        "layers.*.attn.wq_b.weight",
        "layers.*.attn.wkv.weight",
        "layers.*.attn.wo_a.weight",
        "layers.*.attn.wo_b.weight",
        "layers.*.ffn.gate.weight",
        "layers.*.ffn.experts.*.w1.weight",
        "layers.*.ffn.experts.*.w2.weight",
        "layers.*.ffn.experts.*.w3.weight",
        "layers.*.ffn.shared_experts.w1.weight",
        "layers.*.ffn.shared_experts.w2.weight",
        "layers.*.ffn.shared_experts.w3.weight",
        "layers.*.ffn_norm.weight",
        "layers.*.attn_norm.weight",
        "embed.weight",
        "head.weight",
    ];
    println!("\nengine roles:");
    let mut missing = 0;
    for r in need {
        match roles.get(r) {
            Some((d, s, c)) => println!("  HAVE    {r:<44} {d:?} {s:?}  x{c}"),
            None => {
                missing += 1;
                println!("  MISSING {r}");
            }
        }
    }
    println!("\nmissing engine roles: {missing} of {}", need.len());
    Ok(())
}
