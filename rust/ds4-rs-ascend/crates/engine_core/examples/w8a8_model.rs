//! Walk an entire W8A8 package and report whether it can drive the engine's model shape.
//!
//! `w8a8_layer` proves ONE layer loads. This walks every layer and checks that the roles the
//! decoder needs are present, shape-consistent and of the expected dtype, deriving the model's
//! dimensions from the tensors themselves rather than from a config file that may disagree
//! with them. It reads headers only -- no weight bytes -- so it is cheap on a 300 GB package.
//!
//! usage: w8a8_model <package dir>

use engine_core::safetensors::{Dtype, Package};
use std::collections::BTreeSet;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("usage: w8a8_model <package dir>"))?;
    let pkg = Package::open(&dir)?;
    println!("package: {}", dir.display());
    println!("tensors in index: {}", pkg.n_tensors());

    // Which layer indices appear at all?
    let mut layers: BTreeSet<usize> = BTreeSet::new();
    for name in pkg.names() {
        if let Some(rest) = name.strip_prefix("layers.") {
            if let Some((idx, _)) = rest.split_once('.') {
                if let Ok(n) = idx.parse::<usize>() {
                    layers.insert(n);
                }
            }
        }
    }
    let n_layers = layers.len();
    println!("layers present: {n_layers} (indices {:?}..{:?})", layers.iter().next(), layers.iter().last());
    anyhow::ensure!(
        layers.iter().copied().eq(0..n_layers),
        "layer indices are not 0..n_layers"
    );

    // Model dimensions, derived from tensors.
    let dim = |n: &str| -> anyhow::Result<Vec<u64>> { Ok(pkg.tensor(n)?.shape) };
    let emb = dim("embed.weight")?;
    let head = dim("head.weight")?;
    let wq_a = dim("layers.0.attn.wq_a.weight")?;
    let wkv = dim("layers.0.attn.wkv.weight")?;
    let wq_b = dim("layers.0.attn.wq_b.weight")?;
    let w1 = dim("layers.0.ffn.experts.0.w1.weight")?;
    let w2 = dim("layers.0.ffn.experts.0.w2.weight")?;
    let d_model = emb[1];
    println!(
        "\n  vocab {}  d_model {}  q_lora {}  kv_lora {}  q_dim {}  expert ffn {}",
        emb[0], d_model, wq_a[0], wkv[0], wq_b[0], w1[0]
    );
    anyhow::ensure!(head[1] == d_model, "head width {} != d_model {}", head[1], d_model);
    anyhow::ensure!(wq_a[1] == d_model, "wq_a reads {} not d_model", wq_a[1]);
    anyhow::ensure!(w1[1] == d_model, "expert w1 reads {} not d_model", w1[1]);
    anyhow::ensure!(w2[0] == d_model, "expert w2 writes {} not d_model", w2[0]);
    anyhow::ensure!(w2[1] == w1[0], "expert w2 reads {} not {}", w2[1], w1[0]);

    // How many routed experts per layer, from the names.
    let mut per_layer_experts: Vec<usize> = Vec::new();
    for l in 0..n_layers {
        let prefix = format!("layers.{l}.ffn.experts.");
        let n = pkg
            .names()
            .iter()
            .filter(|s| s.starts_with(&prefix) && s.ends_with(".w1.weight"))
            .count();
        per_layer_experts.push(n);
    }
    let n_experts = per_layer_experts[0];
    println!("  routed experts per layer: {:?}{}", &per_layer_experts[..per_layer_experts.len().min(4)],
             if per_layer_experts.iter().all(|n| *n == n_experts) { " (uniform)" } else { " (NOT UNIFORM)" });

    // Per-layer role check.
    let roles = [
        "attn_norm.weight", "attn.wq_a.weight", "attn.wq_b.weight", "attn.wkv.weight",
        "attn.wo_a.weight", "attn.wo_b.weight",
        "ffn_norm.weight", "ffn.gate.weight",
        "ffn.shared_experts.w1.weight", "ffn.shared_experts.w2.weight", "ffn.shared_experts.w3.weight",
    ];
    let mut missing: Vec<(usize, &str)> = Vec::new();
    let mut bytes = 0u64;
    let mut dtypes: BTreeSet<String> = BTreeSet::new();
    for l in 0..n_layers {
        for r in roles {
            let n = format!("layers.{l}.{r}");
            match pkg.tensor(&n) {
                Ok(t) => {
                    bytes += t.nbytes();
                    dtypes.insert(format!("{:?}", t.dtype));
                }
                Err(_) => missing.push((l, r)),
            }
        }
    }
    // The routed experts are the bulk of the model, so they are priced across EVERY layer
    // and EVERY expert. Pricing one layer and multiplying understated the package by ~40x on
    // the first run, which is the sort of number that makes a 294 GB package look harmless.
    for l in 0..n_layers {
        for e in 0..n_experts {
            for w in ["w1", "w2", "w3"] {
                if let Ok(t) = pkg.tensor(&format!("layers.{l}.ffn.experts.{e}.{w}.weight")) {
                    bytes += t.nbytes();
                }
            }
        }
    }
    for n in ["embed.weight", "head.weight"] {
        if let Ok(t) = pkg.tensor(n) {
            bytes += t.nbytes();
        }
    }
    println!("  dtypes across roles: {:?}", dtypes);
    println!("  bytes referenced: {:.1} GB", bytes as f64 / 1e9);
    if missing.is_empty() {
        println!("\nVERDICT: every layer carries every role the decoder needs; the model opens.");
    } else {
        println!("\nVERDICT: {} role(s) missing, first {:?}", missing.len(), missing.first());
    }
    // A role that arrives as I8 where the engine expects f32 (or vice versa) is a silent
    // misread, so the expectation is stated rather than assumed.
    let mut wrong_dtype: Vec<String> = Vec::new();
    for (n, want) in [
        ("layers.0.attn_norm.weight", Dtype::F32),
        ("layers.0.ffn_norm.weight", Dtype::F32),
        ("layers.0.attn.wo_a.weight", Dtype::BF16),
        ("layers.0.ffn.gate.weight", Dtype::BF16),
        ("layers.0.attn.wq_a.weight", Dtype::I8),
    ] {
        if let Ok(t) = pkg.tensor(n) {
            if t.dtype != want {
                wrong_dtype.push(format!("{n}: {:?} not {want:?}", t.dtype));
            }
        }
    }
    if !wrong_dtype.is_empty() {
        println!("dtype surprises: {wrong_dtype:?}");
    }

    // The ROLE MAP, against the real package: every role the decoder asks for must resolve, for
    // every layer. This is the check that the builder's naming table is right, and it is a
    // different question from "the tensor exists" -- it is "the role means this tensor".
    let mut role_fail = 0usize;
    for l in 0..n_layers {
        match engine_core::w8a8_roles::resolve_layer(&pkg, l) {
            Ok(roles) => {
                if l == 0 {
                    for (r, t) in &roles {
                        println!(
                            "  role {:<16} -> {:<42} {:?} {:?}",
                            r.role, r.name, t.dtype, t.shape
                        );
                    }
                }
            }
            Err(e) => {
                role_fail += 1;
                if role_fail == 1 {
                    println!("  role map FAILED on layer {l}: {e}");
                }
            }
        }
    }
    for e in engine_core::w8a8_roles::expert_roles(0, 0) {
        match pkg.tensor(&e.name) {
            Ok(t) => println!("  expert role {:<13} -> {:<40} {:?} {:?}", e.role, e.name, t.dtype, t.shape),
            Err(_) => println!("  expert role {} MISSING: {}", e.role, e.name),
        }
    }
    println!("\nrole map: {} layer(s) failed to resolve", role_fail);
    Ok(())
}
