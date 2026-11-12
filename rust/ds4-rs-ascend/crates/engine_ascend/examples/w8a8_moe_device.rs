//! A REAL routed-expert MoE step through the engine, on the NPU, against f64.
//!
//! This is item (3)'s "routed experts routed through w8a8 kernels" end to end: the three expert
//! projections are read from the package in its own layout (int8 codes + per-output-channel scale and
//! offset, NOTHING dequantised), composed by `AscendDispatcher::moe_routed_step_w8a8` with the engine's
//! own clamp/fold, and compared against the same composition in f64.
//!
//! usage: w8a8_moe_device <package dir> <layer> <expert...> <elf>

use engine_ascend::dispatcher::{AscendDispatcher, W8a8Expert};
use engine_ascend::device::DeviceLaunch;
use engine_ascend::launch::KernelLaunch;
use engine_core::dispatch::CpuDispatcher;
use engine_core::safetensors::Package;
use std::path::PathBuf;

fn plane(pkg: &Package, prefix: &str) -> anyhow::Result<(W8a8Expert, usize, usize)> {
    let raw = pkg.raw(&format!("{prefix}.weight"))?.1;
    let codes: Vec<i8> = raw.iter().map(|&b| b as i8).collect();
    let scale = pkg.f32_tensor(&format!("{prefix}.weight_scale"))?;
    let offset = pkg.f32_tensor(&format!("{prefix}.weight_offset"))?;
    let d_out = scale.len();
    anyhow::ensure!(offset.len() == d_out, "offset {} != outputs {}", offset.len(), d_out);
    anyhow::ensure!(codes.len() % d_out == 0, "codes not divisible by outputs");
    let d_in = codes.len() / d_out;
    Ok((W8a8Expert { codes, scale, offset }, d_in, d_out))
}

fn cpu_ref(p: &W8a8Expert, x: &[f64], d_in: usize, d_out: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; d_out];
    for r in 0..d_out {
        let mut acc = 0.0f64;
        for c in 0..d_in {
            acc += (p.codes[r * d_in + c] as f64 - p.offset[r] as f64) * p.scale[r] as f64 * x[c];
        }
        out[r] = acc;
    }
    out
}

fn main() -> anyhow::Result<()> {
    let dir = std::env::args().nth(1).map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("usage: w8a8_moe_device <pkg> <layer> <elf>"))?;
    let layer: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    let elf = std::env::args().nth(3).ok_or_else(|| anyhow::anyhow!("need the device ELF"))?;

    let pkg = Package::open(&dir)?;
    let dev0 = DeviceLaunch::open(std::path::Path::new(&elf), 0)?;
    let dev = AscendDispatcher::new(CpuDispatcher).with_matvec_w8a8_experts(Box::new(dev0));

    let p = format!("layers.{layer}.ffn.experts.0");
    let (w1, d_in, d_ffn) = plane(&pkg, &format!("{p}.w1"))?;
    let (w3, _, _) = plane(&pkg, &format!("{p}.w3"))?;
    let (w2, d_ffn2, d_model) = plane(&pkg, &format!("{p}.w2"))?;
    anyhow::ensure!(d_ffn == d_ffn2, "w1 up {} != w2 in {}", d_ffn, d_ffn2);
    println!("w1 {d_in}->{d_ffn}, w2 {d_ffn2}->{d_model}");

    let x: Vec<f32> = (0..d_in).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();
    let weight = 0.75f32;
    const CLAMP: f32 = 10.0;

    // THE WIRING ITEM (3) NAMES: planes straight from the package, nothing dequantised, one call.
    let got = dev.moe_routed_step_from_package(&pkg, layer, 0, &x, weight, d_ffn)?;

    // The same composition in f64, through the package's own formula.
    let xf: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let g = cpu_ref(&w1, &xf, d_in, d_ffn);
    let u = cpu_ref(&w3, &xf, d_in, d_ffn);
    let mid: Vec<f64> = g.iter().zip(&u).map(|(&g, &u)| {
        let gc = g.min(CLAMP as f64);
        let uc = u.clamp(-(CLAMP as f64), CLAMP as f64);
        (gc / (1.0 + (-gc).exp())) * uc * weight as f64
    }).collect();
    let want = cpu_ref(&w2, &mid, d_ffn, d_model);

    let (mut abs_sum, mut sum_ref, mut max_abs) = (0.0f64, 0.0f64, 0.0f64);
    for r in 0..d_model {
        let d = (got[r] as f64 - want[r]).abs();
        abs_sum += d; sum_ref += want[r].abs();
        if d > max_abs { max_abs = d; }
    }
    let backward = abs_sum / sum_ref.max(1e-30);
    println!("device vs f64: max_abs {max_abs:.3e}  backward_rel {backward:.3e}");
    anyhow::ensure!(backward <= 1e-5, "device and f64 disagree: {backward}");
    println!("PASS: one real routed expert ran through the engine's w8a8 MoE step on the NPU");
    Ok(())
}
