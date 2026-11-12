//! Loading one routed expert's three planes out of the checkpoint file.
//!
//! This is the prerequisite `decode_runner` names when it says "expert weights resident: zero until placement is wired".
//! `ExpertInfo` already carries each plane's file OFFSET and BYTE LENGTH, so placement is three `pread`s -- and because it
//! is pure I/O it can be tested against a synthetic file rather than a 96 GB checkpoint.

use crate::decode_runner::ExpertInfo;
use anyhow::{bail, Context, Result};
use std::fs::File;
use std::os::unix::fs::FileExt;

/// One expert's three quantised planes, exactly as they sit in the file.
pub struct ExpertPlanes {
    pub gate: Vec<u8>,
    pub up: Vec<u8>,
    pub down: Vec<u8>,
}

impl ExpertPlanes {
    /// Bytes held, which is what `DispatchedExperts::resident_bytes` should count.
    pub fn len(&self) -> usize {
        self.gate.len() + self.up.len() + self.down.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Where expert `e`'s copy of a stacked plane starts, and how long it is.
///
/// `plane` is `(file_offset, total_bytes)` for the WHOLE stack, so one expert's slice is
/// `total / n_experts` bytes at `file_offset + e * (total / n_experts)`.
pub fn expert_slice(plane: (usize, usize), n_experts: usize, e: usize) -> Result<(usize, usize)> {
    if n_experts == 0 {
        bail!("expert_slice: n_experts is zero");
    }
    if e >= n_experts {
        bail!("expert_slice: expert {e} is outside 0..{n_experts}");
    }
    let (start, total) = plane;
    if total % n_experts != 0 {
        bail!("expert_slice: {total} bytes do not divide by {n_experts} experts");
    }
    let per = total / n_experts;
    Ok((start + e * per, per))
}

fn read_at(f: &File, offset: usize, len: usize, what: &str) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; len];
    f.read_exact_at(&mut buf, offset as u64)
        .with_context(|| format!("pread {len} bytes at {offset} for {what}"))?;
    Ok(buf)
}

/// Load expert `e`'s gate, up and down planes.
pub fn load_expert(f: &File, info: &ExpertInfo, e: usize) -> Result<ExpertPlanes> {
    let (gs, gl) = expert_slice(info.gate, info.n_experts, e)?;
    let (us, ul) = expert_slice(info.up, info.n_experts, e)?;
    let (ds, dl) = expert_slice(info.down, info.n_experts, e)?;
    Ok(ExpertPlanes {
        gate: read_at(f, gs, gl, "gate")?,
        up: read_at(f, us, ul, "up")?,
        down: read_at(f, ds, dl, "down")?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// A SANITARY FILE: each byte encodes where it lives, so a slice read from the wrong offset is visible in the data and
    /// not merely in its length. This is the fixture rule -- the test must be able to express the bug.
    fn synthetic(dir: &std::path::Path, gate_at: usize, up_at: usize, down_at: usize, n_experts: usize, per: usize) -> File {
        let mut bytes = vec![0xEEu8; down_at + n_experts * per];
        for (base, tag) in [(gate_at, 0x61u8), (up_at, 0x62), (down_at, 0x63)] {
            for i in 0..n_experts * per {
                bytes[base + i] = tag;
            }
        }
        let path = dir.join("planes.bin");
        let mut f = File::create(&path).unwrap();
        f.write_all(&bytes).unwrap();
        File::open(&path).unwrap()
    }

    fn info(gate: usize, up: usize, down: usize, n: usize, per: usize) -> ExpertInfo {
        ExpertInfo {
            layer_idx: 0,
            ttype: engine_core::gguf::GgmlType::Q8_0,
            gate: (gate, n * per),
            up: (up, n * per),
            down: (down, n * per),
            n_experts: n,
            d_ffn: 4,
            d_model: 8,
        }
    }

    /// Each plane is read from ITS offset, with ITS length, and the bytes match the region they came from.
    #[test]
    fn each_plane_comes_from_its_own_region() {
        let dir = std::env::temp_dir().join(format!("hc_place_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (n, per) = (4usize, 16usize);
        let f = synthetic(&dir, 0, 64, 128, n, per);
        let inf = info(0, 64, 128, n, per);
        let p = load_expert(&f, &inf, 2).unwrap();
        assert_eq!(p.gate.len(), per, "one expert's gate slice");
        assert!(p.gate.iter().all(|b| *b == 0x61), "gate bytes came from the wrong region");
        assert!(p.up.iter().all(|b| *b == 0x62), "up bytes came from the wrong region");
        assert!(p.down.iter().all(|b| *b == 0x63), "down bytes came from the wrong region");
        assert_eq!(p.len(), 3 * per);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// An expert index outside the stack is REFUSED, not clamped: clamping would silently use another expert's weights.
    #[test]
    fn an_out_of_range_expert_is_refused() {
        let e = expert_slice((0, 64), 4, 4).expect_err("expert 4 of 4 must be refused");
        assert!(format!("{e}").contains("outside"), "got: {e}");
    }

    /// A stack whose length does not divide by the expert count is REFUSED: a fractional slice would misalign every plane.
    #[test]
    fn a_stack_that_does_not_divide_is_refused() {
        let e = expert_slice((0, 65), 4, 0).expect_err("65 / 4 must be refused");
        assert!(format!("{e}").contains("do not divide"), "got: {e}");
    }
}
