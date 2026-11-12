//! Cross-check one projection between the W8A8 deployment package and the GGUF build.
//!
//! Two independent quantisations of the same model must dequantise to nearly the same
//! numbers. This example is the only way to settle a question the kernel otherwise ASSUMES:
//! whether `weight_offset` means `(q - off) * scale` or `q * scale + off`. Both are printed,
//! and the smaller error wins -- convention by measurement rather than by naming.
//!
//! usage: w8a8_vs_gguf <package dir> <model.gguf> [layer] [role]

use engine_core::glm5_stream::dequant_blocks;
use engine_core::gguf::GgufFile;
use engine_core::layer_view::MmapSource;
use engine_core::safetensors::Package;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 {
        anyhow::bail!("usage: w8a8_vs_gguf <package dir> <model.gguf> [layer] [role]");
    }
    let dir = PathBuf::from(&a[1]);
    let gguf = PathBuf::from(&a[2]);
    let layer = a.get(3).map(|s| s.as_str()).unwrap_or("0");
    let role = a.get(4).map(|s| s.as_str()).unwrap_or("wq_a");

    // --- w8a8 side -------------------------------------------------------
    let pkg = Package::open(&dir)?;
    let base = format!("layers.{layer}.attn.{role}");
    let w = pkg.tensor(&format!("{base}.weight"))?;
    let sc = pkg.tensor(&format!("{base}.weight_scale"))?;
    let off = pkg.tensor(&format!("{base}.weight_offset"))?;
    println!("w8a8 {base}.weight {:?} {:?}", w.dtype, w.shape);
    let q: Vec<i8> = pkg
        .shard_for(&w.name)?
        .bytes(&w)
        .iter()
        .map(|b| *b as i8)
        .collect();
    let f32s = |t: &engine_core::safetensors::TensorRef| -> Vec<f32> {
        pkg.shard_for(&t.name)
            .unwrap()
            .bytes(t)
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    let scale = f32s(&sc);
    let offset = f32s(&off);
    let rows = w.shape[0] as usize;
    let cols = w.shape[1] as usize;
    println!("  rows={rows} cols={cols} scale={} offset={}", scale.len(), offset.len());

    // --- GGUF side -------------------------------------------------------
    let g = GgufFile::open(&gguf)?;
    let mm = MmapSource::open(&gguf)?;
    let name = format!("blk.{layer}.attn_q_a.weight");
    let t = g
        .tensors
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| anyhow::anyhow!("{name} not in {}", gguf.display()))?;
    let n_elems: usize = t.dims.iter().map(|d| *d as usize).product();
    let tsize = t.ttype.type_size();
    let bs = t.ttype.block_size();
    let nbytes = n_elems / bs * tsize;
    let start = (g.tensor_data_offset + t.offset) as usize;
    let gbytes = &mm.as_slice()[start..start + nbytes];
    println!("gguf {name} {:?} dims={:?} -> {} elems", t.ttype, t.dims, n_elems);
    let gw = dequant_blocks(t.ttype, gbytes, n_elems)?;

    // GGUF stores [in, out]; the w8a8 row-major [out, in] is the same array for a 2-D
    // projection, which is exactly what this asserts.
    anyhow::ensure!(
        gw.len() == q.len(),
        "size mismatch: gguf {} vs w8a8 {}",
        gw.len(),
        q.len()
    );

    // --- diagnostics, then the conventions -------------------------------
    //
    // Both conventions printed the SAME numbers on the first run, which settles the question
    // that mattered: this package's weight_offset is (near) zero, so (q-off)*s and q*s+off
    // coincide. `scale` has one entry per ROW, which also rules out a transposed layout.
    // What the run showed instead is that the arrays are UNCORRELATED, so these diagnostics
    // come first: they say whether the fault is mine (indexing) or the files' (different model).
    let n_off = offset.iter().filter(|v| **v != 0.0).count();
    let mean_g: f64 = gw.iter().map(|v| v.abs() as f64).sum::<f64>() / gw.len() as f64;
    // ABSOLUTE mean: a signed mean of a symmetric int8 array is ~0 by construction and
    // says nothing about the values, which is a mistake I made on the previous run.
    let mean_q: f64 = q.iter().map(|v| (*v as f64).abs()).sum::<f64>() / q.len() as f64;
    let mean_prod: f64 = (0..q.len())
        .map(|i| (q[i] as f64 * scale[i / cols] as f64).abs())
        .sum::<f64>()
        / q.len() as f64;
    let mean_s: f64 = scale.iter().map(|v| *v as f64).sum::<f64>() / scale.len() as f64;
    println!("\n  diagnostics:");
    println!("    nonzero offsets: {n_off} of {}", offset.len());
    println!("    mean|gguf| = {mean_g:.6e}   mean|q| = {mean_q:.3}   mean scale = {mean_s:.6e}");
    println!("    mean|q*scale| = {mean_prod:.6e}   (gguf/w8a8 ratio = {:.3})", mean_g / mean_prod);
    println!("    first scales: {:?}", &scale[..3]);
    println!("    first q row0: {:?}", &q[..6]);
    println!("    first gguf: {:?}", &gw[..4]);
    println!(
        "    first w8a8 (q*scale): {:?}",
        (0..4).map(|k| q[k] as f64 * scale[0] as f64).collect::<Vec<_>>()
    );

    let mut best: Option<(String, f64, f64)> = None;
    // Pearson correlation over the whole tensor: 1.0 means "same numbers", ~0 means
    // "different model or different indexing", which no convention can rescue.
    {
        let n = gw.len();
        let mg = gw.iter().map(|v| *v as f64).sum::<f64>() / n as f64;
        let mv = (0..n)
            .map(|i| {
                let row = i / cols;
                q[i] as f64 * scale[row] as f64
            })
            .sum::<f64>()
            / n as f64;
        let mut cov = 0.0f64;
        let mut vg = 0.0f64;
        let mut vv = 0.0f64;
        for i in 0..n {
            let row = i / cols;
            let v = q[i] as f64 * scale[row] as f64;
            let g = gw[i] as f64;
            cov += (g - mg) * (v - mv);
            vg += (g - mg) * (g - mg);
            vv += (v - mv) * (v - mv);
        }
        println!("    correlation = {:.6}", cov / (vg.sqrt() * vv.sqrt()));
    }

    for (label, deq) in [("(q - off) * scale", 0usize), ("q * scale + off", 1usize)] {
        let mut sum_abs = 0.0f64;
        let mut max_abs = 0.0f64;
        let mut sum_ref = 0.0f64;
        for i in 0..rows {
            for k in 0..cols {
                let qi = q[i * cols + k] as f64;
                let s = scale[i] as f64;
                let o = offset[i] as f64;
                let v = if deq == 0 { (qi - o) * s } else { qi * s + o };
                let r = gw[i * cols + k] as f64;
                let d = (v - r).abs();
                sum_abs += d;
                sum_ref += r.abs();
                if d > max_abs { max_abs = d; }
            }
        }
        let mean_rel = sum_abs / sum_ref;
        println!("  {label:<18} mean|diff|/sum|ref| = {mean_rel:.6e}  max|diff| = {max_abs:.6e}");
        if best.as_ref().map(|(_, m, _)| mean_rel < *m).unwrap_or(true) {
            best = Some((label.to_string(), mean_rel, max_abs));
        }
    }

    let (label, mean_rel, max_abs) = best.unwrap();
    println!("\ncloser: {label}  (mean rel {mean_rel:.6e}, max abs {max_abs:.6e})");
    // A convention that is WRONG is not marginally worse, it is wrong by the offset's own
    // scale, so this is a check with a wide margin rather than a precision argument.
    if mean_rel < 1e-2 {
        println!("VERDICT: the package and the GGUF agree -- same model, and the convention is {label}");
    } else {
        println!("VERDICT: no agreement; neither convention reproduces the GGUF weights");
    }
    Ok(())
}
