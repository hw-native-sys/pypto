//! A vocabulary-wide projection as BANDS of a certified kernel.
//!
//! `lm_head` is 154,880 outputs. That is 620 KiB of output buffer against a 188 KiB unified buffer, so it
//! cannot be one kernel at any k-chunking -- the output-width bound is separate from the k bound. The caller
//! decomposes it over the OUTPUT dimension: N bands of 4096 weight rows, each through a certified 5-argument
//! w8a8 matvec at (4096 out, k=4096).
//!
//! This example proves the mechanism at 3 bands (12,288 outputs) with deterministic synthetic planes, and
//! notes the full size: at 154,880 outputs that is 38 bands.
//!
//! usage: w8a8_lm_head_banded <elf> [bands]

use engine_ascend::device::DeviceLaunch;
use engine_ascend::launch::KernelLaunch;
use std::path::PathBuf;

// Defaults are lm_head's: k = GLM 5.3's D, band = the certified 4096-output kernel's width. Both are
// overridable so the SAME code can bisect k on the same launch path (the banded launch hangs at k=4096
// while matvec_w8a8_experts works at k=2048 -- see iteration 161).
const K_DEFAULT: usize = 4096;
const BAND: usize = 4096;

fn main() -> anyhow::Result<()> {
    let elf = std::env::args().nth(1).ok_or_else(|| anyhow::anyhow!("usage: w8a8_lm_head_banded <elf> [bands]"))?;
    let bands: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(3);
    let K: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(K_DEFAULT);
    let kname = std::env::args().nth(4).unwrap_or_else(|| "w8a8_lm_head_band".to_string());
    let ksym = std::env::args().nth(5).unwrap_or_else(|| "glm53_proj_output_mix_aiv".to_string());
    let dev = DeviceLaunch::open(std::path::Path::new(&elf), 0)?;

    // Deterministic synthetic w8a8 planes for `bands` bands; real weights are 154,880 x 4096 = 634 MB.
    let d_out = bands * BAND;
    let mut codes = vec![0i8; d_out * K];
    let mut scale = vec![0.0f32; d_out];
    let mut offset = vec![0.0f32; d_out];
    for r in 0..d_out {
        for c in 0..K { codes[r * K + c] = (((r * 7 + c * 13) % 255) as i32 - 127) as i8; }
        scale[r] = 0.5 / ((r % 7) + 1) as f32;
        offset[r] = ((r % 5) as f32) - 2.0;
    }
    // PAD EVERY CALLER-SIDE INPUT. `run_raw_cached` allocates each device buffer EXACTLY to the slice it
    // is handed, and the emitted kernels read slightly past the logical end -- the same over-read that made
    // the expert launch hang for six rounds before its caller padded. The launcher-side over-allocation
    // attempted in iteration 144 was reverted because it landed on the wrong call site; until it is done
    // properly, every caller pads.
    let x: Vec<f32> = (0..K).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
    println!("{d_out} outputs as {bands} bands of {BAND}, k={K}");

    let mut got = Vec::with_capacity(d_out);
    let mut codes = codes;
    for b in 0..bands {
        let lo = b * BAND;
        let hi = lo + BAND;
        // EXACTLY THE WORKING CALL'S BUFFERS: slices of the planes, no per-band copy and no padding
        // (w8a8_expert_down passes `&codes, &scale, &offset, &x` straight through).
        let w = &codes[lo * K..hi * K];
        let sc = &scale[lo..hi];
        let of = &offset[lo..hi];
        // THE WORKING BODY, reached for a decomposition kernel: same marshalling, same flags, same
        // run_raw_cached call as `matvec_w8a8_experts`, with the spec taken from DECOMPOSITION_KERNELS.
        // THE CONFIRMING RUN. `SLOT:name` drives the PROVEN method with a slot name; anything else goes
        // through the decomposition clone. If the slot returns and the clone hangs at the same shape, on
        // the same ELF, then the defect is the DUPLICATE REGISTRATION of one symbol under two names.
        let mut band = if let Some(slot) = kname.strip_prefix("SLOT:") {
            dev.matvec_w8a8_experts(slot, w, sc, of, &x, BAND)?
        } else {
            dev.matvec_w8a8_banded(&kname, w, sc, of, &x, BAND)?
        };
        anyhow::ensure!(band.len() == BAND, "band {b} returned {}", band.len());
        got.append(&mut band);
    }

    // f64 reference over the same bands.
    let (mut abs_sum, mut sum_ref, mut max_abs) = (0.0f64, 0.0f64, 0.0f64);
    for r in 0..d_out {
        let mut want = 0.0f64;
        for c in 0..K {
            want += (codes[r * K + c] as f64 - offset[r] as f64) * scale[r] as f64 * x[c] as f64;
        }
        let d = (got[r] as f64 - want).abs();
        abs_sum += d; sum_ref += want.abs();
        if d > max_abs { max_abs = d; }
    }
    let backward = abs_sum / sum_ref.max(1e-30);
    println!("banded device vs f64: max_abs {max_abs:.3e}  backward_rel {backward:.3e}");
    anyhow::ensure!(backward <= 1e-5, "banded projection disagrees: {backward}");
    println!("PASS: a vocabulary-wide projection ran as {bands} bands of a certified kernel");
    Ok(())
}
