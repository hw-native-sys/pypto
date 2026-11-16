//! One REAL routed expert's down projection, device vs CPU, on the package's own int8 weights.
//!
//! The package stores each expert as I8 codes plus a per-output-channel F32 scale and offset -- exactly
//! the w8a8 layout -- so nothing is dequantised before the launch. The CPU side dequantises through the
//! same (q - offset) * scale formula and computes the same dot in f64.
//!
//! usage: w8a8_expert_down <package dir> <layer> <expert> <device elf>

use engine_ascend::device::DeviceLaunch;
use engine_ascend::launch::KernelLaunch;
use engine_core::safetensors::Package;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let dir = std::env::args().nth(1).map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("usage: w8a8_expert_down <pkg> <layer> <expert> <elf>"))?;
    let layer: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    let expert: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(0);
    let elf = std::env::args().nth(4).ok_or_else(|| anyhow::anyhow!("need the device ELF"))?;
    let pkg = Package::open(&dir)?;
    let dev = DeviceLaunch::open(std::path::Path::new(&elf), 0)?;

    // 64 KiB of zero slack on every caller-side input (see the note below).
    const PAD: usize = 64 * 1024;
    let plane = std::env::args().nth(5).unwrap_or_else(|| "w2".to_string());
    let p = format!("layers.{layer}.ffn.experts.{expert}.{plane}");
    let raw = pkg.raw(&format!("{p}.weight"))?.1;
    let scale = pkg.f32_tensor(&format!("{p}.weight_scale"))?;
    let offset = pkg.f32_tensor(&format!("{p}.weight_offset"))?;
    let d_out = scale.len();
    let logical_codes = raw.len();
    let d_in = logical_codes / d_out;
    println!("{plane}: {logical_codes} codes, {d_out} outputs, k={d_in}  (caller buffers TIGHT)");
    anyhow::ensure!(offset.len() == d_out, "offset {} != outputs {}", offset.len(), d_out);
    anyhow::ensure!(logical_codes == d_out * d_in, "code count is not out*k");

    // PAD EVERY CALLER-SIDE INPUT. This is REQUIRED, and it is not the same as padding the device
    // allocation: padding in the launcher (a 64 KiB `INPUT_SLACK` on `aclrtMalloc`) still HUNG, while
    // padding the caller's vectors makes the launch return and match f64 to 1.130e-07. So the kernel is
    // sensitive to the SIZE OF THE BUFFER IT IS HANDED, not merely to the allocation behind it -- which
    // is also why the harness has never shown this: it allocates 9,000,000 floats for a kernel that
    // reads 8.4 MB of int8.
    let mut x: Vec<f32> = (0..d_in).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
    let mut scale = scale;
    let mut offset = offset;
    let mut codes: Vec<i8> = raw.iter().map(|&b| b as i8).collect();

    // ROW-SLICED LAUNCH. The arm faults above ~1024 outputs when k=4096 (1024 PASS, 1536 FAULT, 2048
    // FAULT), so a wider projection is decomposed here into 1024-row bands of the CERTIFIED kernel,
    // launched by symbol because a caller-side decomposition is not an engine slot.
    //
    // 1024-row band of a 4096-wide K: the band's weights are codes[lo*d_in .. hi*d_in], and its scale and
    // offset are the same rows. The certified kernel takes (x, w, scale, offset, out).
    let band: usize = if d_in == 4096 { 1024 } else { d_out };
    let dev_out: Vec<f32> = if d_out > band && d_in == 4096 {
        println!("slicing {d_out} outputs into {} bands of {band}", d_out.div_ceil(band));
        let mut out = Vec::with_capacity(d_out);
        for lo in (0..d_out).step_by(band) {
            let hi = (lo + band).min(d_out);
            let rows = hi - lo;
            anyhow::ensure!(rows == band, "band {lo}..{hi} is {rows} rows, not {band}: the certified kernel is fixed-width");
            let w = &codes[lo * d_in..hi * d_in];
            let sc = &scale[lo..hi];
            let of = &offset[lo..hi];
            // CONTIGUOUS: all four inputs inside ONE device allocation, like the harness that
            // certifies this kernel. Separate allocations hang at this shape; this is the test of
            // whether that is the difference.
            let mut got = dev.run_by_symbol_contiguous(
                "w8a8_slice_1x1024_k4096",
                "proj_w8a8_engine_1x1024_k4096_mix_aiv",
                engine_ascend::device::Partitioning::ByRow,
                &[
                    unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4) },
                    unsafe { std::slice::from_raw_parts(w.as_ptr() as *const u8, w.len()) },
                    unsafe { std::slice::from_raw_parts(sc.as_ptr() as *const u8, sc.len() * 4) },
                    unsafe { std::slice::from_raw_parts(of.as_ptr() as *const u8, of.len() * 4) },
                ],
                band,
            )?;
            out.extend_from_slice(&got);
            got.clear();
        }
        out
    } else {
        let spec = if d_in == 2048 { "w8a8_experts_2048_4096" } else { "w8a8_experts_4096_2048" };
        dev.matvec_w8a8_experts(spec, &codes, &scale, &offset, &x, d_out)?
    };

    let mut want = vec![0.0f64; d_out];
    for r in 0..d_out {
        let mut acc = 0.0f64;
        for c in 0..d_in {
            acc += (codes[r * d_in + c] as f64 - offset[r] as f64) * scale[r] as f64 * x[c] as f64;
        }
        want[r] = acc;
    }
    let (mut abs_sum, mut sum_ref, mut max_abs) = (0.0f64, 0.0f64, 0.0f64);
    for r in 0..d_out {
        let d = (dev_out[r] as f64 - want[r]).abs();
        abs_sum += d;
        sum_ref += want[r].abs();
        if d > max_abs { max_abs = d; }
    }
    let backward = abs_sum / sum_ref.max(1e-30);
    println!("device vs CPU: max_abs {max_abs:.3e}  backward_rel {backward:.3e}");
    anyhow::ensure!(backward <= 1e-5, "device and CPU disagree: {backward}");
    println!("PASS: a real expert's down projection ran through the w8a8 kernel");
    Ok(())
}
