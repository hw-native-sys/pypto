// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Reading DS4-Flash weights out of the GGUF and into the layout the PTO kernels read.
//!
//! This is the first brick of the engine: until now every device check was fed by a probe
//! that fabricated its own planes, or by `mxfp4_repack_dump` writing files for a C++ runner.
//! Nothing read the real model and produced kernel-ready planes in one step.
//!
//! Granularity is ONE EXPERT, not one tensor, and that is deliberate. A layer's
//! `ffn_*_exps.weight` is ~1 GB packed; the residency work established that a token touches
//! 6 of 256 experts per layer and that placement is almost entirely a question of how much
//! expert POOL each chip gets. So the unit that gets read, repacked and made resident is the
//! expert — which also keeps this testable without materialising a gigabyte.
//!
//! Bytes are read with `read_exact_at` rather than mmap: the planes have to be copied to
//! device memory regardless, so mapping buys nothing and costs a dependency.

use crate::mxfp4_repack::{repack_weight, PackedWeight, BLK, QK};
use anyhow::{anyhow, bail, Context, Result};
use engine_core::gguf::{GgmlType, GgufFile, TensorInfo};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

/// The three MXFP4 expert projections in a DS4-Flash layer. These are exactly the 129
/// MXFP4 tensors in the model (43 layers x 3); everything else is Q8_0, F32, BF16 or I32.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertRole {
    Gate,
    Up,
    Down,
}

impl ExpertRole {
    pub fn tensor_name(self, layer: u32) -> String {
        let stem = match self {
            ExpertRole::Gate => "ffn_gate_exps",
            ExpertRole::Up => "ffn_up_exps",
            ExpertRole::Down => "ffn_down_exps",
        };
        format!("blk.{layer}.{stem}.weight")
    }
    pub const ALL: [ExpertRole; 3] = [ExpertRole::Gate, ExpertRole::Up, ExpertRole::Down];
}

/// One expert projection's shape inside its 3-D tensor.
///
/// GGUF dims are fastest-axis-first, so `[ne0, ne1, ne2]` means rows of `ne0` elements,
/// `ne1` rows per expert, `ne2` experts. `ne0` is the reduction width the matvec calls
/// `ne00`; `ne1` is how many output rows one expert owns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertShape {
    pub reduction: u64,
    pub rows_per_expert: u64,
    pub n_experts: u64,
}

pub struct WeightSource {
    file: File,
    gguf: GgufFile,
    index: HashMap<String, usize>,
}

impl WeightSource {
    pub fn open(path: &Path) -> Result<Self> {
        let gguf =
            GgufFile::open(path).map_err(|e| anyhow!("open {}: {e}", path.display()))?;
        let file = File::open(path).with_context(|| format!("reopen {}", path.display()))?;
        let index = gguf
            .tensors
            .iter()
            .enumerate()
            .map(|(i, t)| (t.name.clone(), i))
            .collect();
        Ok(Self { file, gguf, index })
    }

    pub fn gguf(&self) -> &GgufFile {
        &self.gguf
    }

    pub fn find(&self, name: &str) -> Result<&TensorInfo> {
        let i = *self
            .index
            .get(name)
            .ok_or_else(|| anyhow!("tensor {name} not in this GGUF"))?;
        Ok(&self.gguf.tensors[i])
    }

    /// Read a byte range of a tensor's data.
    ///
    /// Range-checked against the tensor's own size, not just the file's: a wrong shape
    /// assumption would otherwise read a neighbouring tensor and produce plausible garbage,
    /// which is the failure mode this whole effort keeps meeting.
    fn read_span(&self, t: &TensorInfo, rel_off: u64, len: u64) -> Result<Vec<u8>> {
        let total = self.gguf.tensor_byte_size(t);
        let end = rel_off
            .checked_add(len)
            .ok_or_else(|| anyhow!("{}: span overflow", t.name))?;
        if end > total {
            bail!(
                "{}: span {rel_off}..{end} exceeds the tensor's {total} bytes — shape assumption is wrong",
                t.name
            );
        }
        let abs = self
            .gguf
            .tensor_data_offset
            .checked_add(t.offset)
            .and_then(|b| b.checked_add(rel_off))
            .ok_or_else(|| anyhow!("{}: file offset overflow", t.name))?;
        let mut buf = vec![0u8; len as usize];
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            self.file
                .read_exact_at(&mut buf, abs)
                .with_context(|| format!("{}: read {len} B at {abs}", t.name))?;
        }
        #[cfg(not(unix))]
        {
            use std::io::{Read, Seek, SeekFrom};
            let mut f = &self.file;
            f.seek(SeekFrom::Start(abs))?;
            f.read_exact(&mut buf)?;
        }
        Ok(buf)
    }

    /// Shape of an expert projection, validated as MXFP4 and 3-D.
    pub fn expert_shape(&self, layer: u32, role: ExpertRole) -> Result<ExpertShape> {
        let name = role.tensor_name(layer);
        let t = self.find(&name)?;
        if t.ttype != GgmlType::MXFP4 {
            bail!(
                "{name}: expected MXFP4, found {:?} — the expert path assumes 4-bit weights",
                t.ttype
            );
        }
        if t.dims.len() != 3 {
            bail!(
                "{name}: expected 3 dims [reduction, rows_per_expert, n_experts], found {:?}",
                t.dims
            );
        }
        let s = ExpertShape {
            reduction: t.dims[0],
            rows_per_expert: t.dims[1],
            n_experts: t.dims[2],
        };
        if s.reduction % (2 * QK as u64) != 0 {
            bail!(
                "{name}: reduction {} must be a multiple of 64 — a 32-byte packed row holds a \
                 PAIR of 32-weight blocks",
                s.reduction
            );
        }
        Ok(s)
    }

    /// Read ONE expert and return the two planes the packed matvec reads.
    ///
    /// The expert's rows are contiguous in the tensor, so this is a single ranged read of
    /// `rows_per_expert * reduction / 32 * 17` bytes — about 4.4 MB at DS4-Flash's shape,
    /// against ~1 GB for the whole tensor.
    pub fn expert(&self, layer: u32, role: ExpertRole, expert: u32) -> Result<PackedWeight> {
        let s = self.expert_shape(layer, role)?;
        if expert as u64 >= s.n_experts {
            bail!(
                "{}: expert {expert} out of range (n_experts {})",
                role.tensor_name(layer),
                s.n_experts
            );
        }
        let t = self.find(&role.tensor_name(layer))?;
        let blocks_per_row = s.reduction / QK as u64;
        let bytes_per_row = blocks_per_row * BLK as u64;
        let span = s.rows_per_expert * bytes_per_row;
        let raw = self.read_span(t, expert as u64 * span, span)?;
        repack_weight(&raw, s.reduction as usize, s.rows_per_expert as usize)
            .map_err(|e| anyhow!("{}: expert {expert}: {e}", role.tensor_name(layer)))
    }

    /// Derive the placement [`Geometry`] from the FILE rather than the model card.
    ///
    /// The expert term is ~90% of the model, and it was once a literal constant nobody could
    /// attribute. Deriving it from `ModelDims` fixed that; deriving it from the tensor table
    /// closes the remaining gap, because the dims are then the model's own rather than a
    /// transcription of them. `layout` still has to be stated: it is a property of the kernel
    /// that will read the weights, not of the file.
    pub fn geometry(&self, layout: crate::placement::ExpertLayout) -> Result<crate::placement::Geometry> {
        use crate::placement::Geometry;
        // Layers = however many carry an expert tensor, counted rather than assumed.
        let mut layers = 0u32;
        while self.index.contains_key(&ExpertRole::Up.tensor_name(layers)) {
            layers += 1;
        }
        if layers == 0 {
            bail!("no blk.N.ffn_up_exps.weight tensors — not a DS4-Flash MXFP4 GGUF");
        }
        let mut weights_per_layer = 0u64;
        for role in ExpertRole::ALL {
            let s = self.expert_shape(0, role)?;
            weights_per_layer += s.reduction * s.rows_per_expert * s.n_experts;
        }
        let kv_latent = self
            .expert_shape(0, ExpertRole::Up)
            .map(|_| ())
            .ok()
            .and_then(|_| engine_core::gguf::meta_u32(&self.gguf, &["deepseek2.attention.key_length"]))
            .unwrap_or(512) as u64;
        let base = Geometry::ds4_flash(layout);
        Ok(Geometry {
            layers,
            nonexpert_per_layer: base.nonexpert_per_layer,
            experts_per_layer: (weights_per_layer as f64 * layout.bytes_per_weight()) as u64,
            globals: base.globals,
            kv_per_layer_token: kv_latent * 4,
            // From the tensor's own third dim, not the model card.
            n_experts: self.expert_shape(0, ExpertRole::Up)?.n_experts as u32,
            experts_used_per_token: base.experts_used_per_token,
        })
    }

    /// Total MXFP4 expert bytes in one layer, as STORED. Useful for cross-checking the
    /// placement geometry against the file rather than against the model card.
    pub fn layer_expert_stored_bytes(&self, layer: u32) -> Result<u64> {
        let mut n = 0;
        for role in ExpertRole::ALL {
            let t = self.find(&role.tensor_name(layer))?;
            n += self.gguf.tensor_byte_size(t);
        }
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mxfp4_repack::MXFP4_DOUBLED;
    use engine_core::glm5_stream::{dequant_mxfp4, e8m0_to_f32};
    use std::io::Write;

    /// Minimal GGUF holding one 3-D MXFP4 tensor, so the read/repack path is testable
    /// without the 145 GB model. Writing the container by hand is the point: it pins the
    /// layout this loader assumes (dims fastest-axis-first, data after an aligned pad)
    /// rather than trusting a reader and a writer to share a misunderstanding.
    fn synth_gguf(dir: &Path, name: &str, red: u64, rows: u64, nexp: u64, seed: u32) -> (std::path::PathBuf, Vec<u8>) {
        let nblk = (red / QK as u64) * rows * nexp;
        let mut data = vec![0u8; (nblk * BLK as u64) as usize];
        let mut s = seed | 1;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        };
        for b in 0..nblk as usize {
            // Spread exponents widely and include e == 0, the 2^-127 special case.
            data[b * BLK] = if b % 61 == 0 { 0 } else { 118 + (rnd() % 18) as u8 };
            for j in 1..BLK {
                data[b * BLK + j] = (rnd() & 0xFF) as u8;
            }
        }
        let mut out: Vec<u8> = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes()); // version
        out.extend_from_slice(&1u64.to_le_bytes()); // n_tensors
        out.extend_from_slice(&1u64.to_le_bytes()); // n_meta_kv
        let put_str = |o: &mut Vec<u8>, s: &str| {
            o.extend_from_slice(&(s.len() as u64).to_le_bytes());
            o.extend_from_slice(s.as_bytes());
        };
        // One metadata key: architecture. Enough to be a well-formed GGUF.
        put_str(&mut out, "general.architecture");
        out.extend_from_slice(&8u32.to_le_bytes()); // type 8 = string
        put_str(&mut out, "deepseek2");
        // Tensor table.
        put_str(&mut out, name);
        out.extend_from_slice(&3u32.to_le_bytes()); // n_dims
        for d in [red, rows, nexp] {
            out.extend_from_slice(&d.to_le_bytes());
        }
        out.extend_from_slice(&39u32.to_le_bytes()); // MXFP4
        out.extend_from_slice(&0u64.to_le_bytes()); // offset
        // Default alignment is 32; pad to it, then the data.
        let pad = (32 - (out.len() % 32)) % 32;
        out.extend(std::iter::repeat(0u8).take(pad));
        out.extend_from_slice(&data);
        let p = dir.join("synth.gguf");
        File::create(&p).unwrap().write_all(&out).unwrap();
        (p, data)
    }

    /// As `synth_gguf` but with all three roles across several layers.
    fn synth_gguf_multi(
        dir: &Path,
        layers: &[(u32, u64, u64, u64)],
        seed: u32,
    ) -> (std::path::PathBuf, Vec<u8>) {
        let mut names: Vec<(String, u64, u64, u64)> = Vec::new();
        for &(l, red, rows, nexp) in layers {
            for role in ExpertRole::ALL {
                names.push((role.tensor_name(l), red, rows, nexp));
            }
        }
        let mut s = seed | 1;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        };
        let mut blobs: Vec<Vec<u8>> = Vec::new();
        for (_, red, rows, nexp) in &names {
            let nblk = (red / QK as u64) * rows * nexp;
            let mut d = vec![0u8; (nblk * BLK as u64) as usize];
            for b in 0..nblk as usize {
                d[b * BLK] = 120 + (rnd() % 12) as u8;
                for j in 1..BLK {
                    d[b * BLK + j] = (rnd() & 0xFF) as u8;
                }
            }
            blobs.push(d);
        }
        let mut out: Vec<u8> = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&(names.len() as u64).to_le_bytes());
        out.extend_from_slice(&1u64.to_le_bytes());
        let put_str = |o: &mut Vec<u8>, s: &str| {
            o.extend_from_slice(&(s.len() as u64).to_le_bytes());
            o.extend_from_slice(s.as_bytes());
        };
        put_str(&mut out, "general.architecture");
        out.extend_from_slice(&8u32.to_le_bytes());
        put_str(&mut out, "deepseek2");
        let mut off = 0u64;
        for (i, (n, red, rows, nexp)) in names.iter().enumerate() {
            put_str(&mut out, n);
            out.extend_from_slice(&3u32.to_le_bytes());
            for d in [*red, *rows, *nexp] {
                out.extend_from_slice(&d.to_le_bytes());
            }
            out.extend_from_slice(&39u32.to_le_bytes());
            out.extend_from_slice(&off.to_le_bytes());
            off += blobs[i].len() as u64;
        }
        let pad = (32 - (out.len() % 32)) % 32;
        out.extend(std::iter::repeat(0u8).take(pad));
        let mut all = Vec::new();
        for b in &blobs {
            out.extend_from_slice(b);
            all.extend_from_slice(b);
        }
        let p = dir.join("synth_multi.gguf");
        File::create(&p).unwrap().write_all(&out).unwrap();
        (p, all)
    }

    fn tmpdir(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("ds4_ascend_weights_{tag}"));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// The contract end to end: locate the tensor, read ONE expert's byte range, repack, and
    /// agree with `engine_core`'s independent dequant over the same bytes.
    #[test]
    fn reads_one_expert_and_repacks_to_match_dequant() {
        let (red, rows, nexp) = (64u64, 3u64, 4u64);
        let d = tmpdir("one_expert");
        let (path, raw) = synth_gguf(&d, "blk.7.ffn_up_exps.weight", red, rows, nexp, 0xABCD);
        let ws = WeightSource::open(&path).unwrap();
        let s = ws.expert_shape(7, ExpertRole::Up).unwrap();
        assert_eq!(
            s,
            ExpertShape { reduction: red, rows_per_expert: rows, n_experts: nexp }
        );
        let bytes_per_row = red / QK as u64 * BLK as u64;
        for e in 0..nexp as u32 {
            let pk = ws.expert(7, ExpertRole::Up, e).unwrap();
            assert_eq!(pk.bytes_per_weight(red as usize, rows as usize), 0.5);
            // Independent path: dequant this expert's rows straight from the raw bytes.
            let base = (e as u64 * rows * bytes_per_row) as usize;
            for r in 0..rows as usize {
                let row = &raw[base + r * bytes_per_row as usize
                    ..base + (r + 1) * bytes_per_row as usize];
                let want = dequant_mxfp4(row);
                let nb = (red / QK as u64) as usize;
                let npair = nb / 2;
                for g in 0..npair {
                    for i in 0..QK {
                        let byte = pk.w[(r * npair + g) * QK + i];
                        let even = MXFP4_DOUBLED[(byte & 0x0F) as usize] * pk.d[r * nb + g];
                        let odd = MXFP4_DOUBLED[(byte >> 4) as usize] * pk.d[r * nb + npair + g];
                        // The loader halves the scale and doubles the code; the product must
                        // still equal the reference's value at the same element.
                        assert!(
                            (even - want[(2 * g) * QK + i]).abs() <= 1e-30 + want[(2 * g) * QK + i].abs() * 1e-6,
                            "expert {e} row {r} block {} elem {i}: {even} vs {}",
                            2 * g,
                            want[(2 * g) * QK + i]
                        );
                        assert!(
                            (odd - want[(2 * g + 1) * QK + i]).abs()
                                <= 1e-30 + want[(2 * g + 1) * QK + i].abs() * 1e-6,
                            "expert {e} row {r} block {} elem {i}",
                            2 * g + 1
                        );
                    }
                }
            }
        }
    }

    /// Reading expert `e` must read EXPERT e — an off-by-one in the row stride would still
    /// produce well-formed planes, so the experts are checked to differ from one another.
    #[test]
    fn experts_are_distinct_and_indexed_correctly() {
        let d = tmpdir("distinct");
        let (path, _) = synth_gguf(&d, "blk.0.ffn_gate_exps.weight", 64, 2, 3, 7);
        let ws = WeightSource::open(&path).unwrap();
        let a = ws.expert(0, ExpertRole::Gate, 0).unwrap();
        let b = ws.expert(0, ExpertRole::Gate, 1).unwrap();
        assert_ne!(a.w, b.w, "two experts returned identical codes");
        assert!(ws.expert(0, ExpertRole::Gate, 3).is_err(), "index 3 of 3 must be refused");
    }

    /// A wrong shape assumption must be reported, not read past into the next tensor.
    #[test]
    fn out_of_range_span_is_refused() {
        let d = tmpdir("span");
        let (path, _) = synth_gguf(&d, "blk.1.ffn_down_exps.weight", 64, 2, 2, 3);
        let ws = WeightSource::open(&path).unwrap();
        let t = ws.find("blk.1.ffn_down_exps.weight").unwrap().clone();
        let total = ws.gguf().tensor_byte_size(&t);
        let e = ws.read_span(&t, total - 4, 8).unwrap_err().to_string();
        assert!(e.contains("exceeds"), "expected a range refusal, got: {e}");
    }

    /// A reduction width that is not a multiple of 64 has no valid packed layout.
    #[test]
    fn odd_block_count_reduction_is_refused() {
        let d = tmpdir("odd");
        let (path, _) = synth_gguf(&d, "blk.2.ffn_up_exps.weight", 32, 1, 1, 5);
        let ws = WeightSource::open(&path).unwrap();
        let e = ws.expert_shape(2, ExpertRole::Up).unwrap_err().to_string();
        assert!(e.contains("multiple of 64"), "got: {e}");
    }

    /// The e8m0 zero exponent must survive the loader, since it appears in real tensors.
    #[test]
    fn e8m0_zero_exponent_reaches_the_plane() {
        let d = tmpdir("e0");
        let (path, raw) = synth_gguf(&d, "blk.3.ffn_up_exps.weight", 64, 2, 1, 11);
        let ws = WeightSource::open(&path).unwrap();
        let pk = ws.expert(3, ExpertRole::Up, 0).unwrap();
        // Block 0 was written with e == 0 by the generator.
        assert_eq!(raw[0], 0, "generator should have put e==0 in block 0");
        assert_eq!(pk.d[0], e8m0_to_f32(0) * 0.5);
        assert!(pk.d[0] > 0.0 && pk.d[0].is_finite());
    }

    /// The derived geometry must come from the file's own dims, and must agree with the
    /// hand-written `ModelDims` for the real model — a disagreement means one of the two is
    /// wrong about the model, which is worth failing over rather than averaging.
    #[test]
    fn geometry_is_derived_from_the_tensor_table() {
        use crate::placement::ExpertLayout;
        let d = tmpdir("geom");
        // Two layers, so the layer count is counted rather than assumed.
        let (path, _) = synth_gguf_multi(&d, &[(0u32, 64u64, 2u64, 4u64), (1, 64, 2, 4)], 21);
        let ws = WeightSource::open(&path).unwrap();
        let g = ws.geometry(ExpertLayout::PackedNibbleBlockF32).unwrap();
        assert_eq!(g.layers, 2, "layer count comes from the tensors present");
        // 3 roles x 64 x 2 x 4 weights at 0.625 B/weight.
        let w = 3 * 64 * 2 * 4;
        assert_eq!(g.experts_per_layer, (w as f64 * 0.625) as u64);
    }

    /// On the real model, the file-derived geometry and the transcribed one must agree.
    #[test]
    fn real_model_geometry_matches_the_model_card() {
        use crate::placement::{ExpertLayout, Geometry};
        let Ok(p) = std::env::var("DS4_ASCEND_TEST_GGUF") else {
            eprintln!("skip: set DS4_ASCEND_TEST_GGUF");
            return;
        };
        let ws = WeightSource::open(Path::new(&p)).expect("open");
        let from_file = ws.geometry(ExpertLayout::PackedNibbleBlockF32).expect("geometry");
        let from_card = Geometry::ds4_flash(ExpertLayout::PackedNibbleBlockF32);
        eprintln!("from file: {from_file:?}\nfrom card: {from_card:?}");
        assert_eq!(from_file.layers, from_card.layers, "layer count");
        assert_eq!(
            from_file.experts_per_layer, from_card.experts_per_layer,
            "expert bytes per layer disagree between the file and ModelDims"
        );
    }

    /// Opens the real model when one is present, and reports what it finds rather than
    /// asserting a shape from the model card. Skips without the env var so the suite stays
    /// free on a machine without 145 GB.
    #[test]
    fn real_model_expert_roundtrip() {
        let Ok(p) = std::env::var("DS4_ASCEND_TEST_GGUF") else {
            eprintln!("skip: set DS4_ASCEND_TEST_GGUF");
            return;
        };
        let ws = WeightSource::open(Path::new(&p)).expect("open");
        let s = ws.expert_shape(0, ExpertRole::Up).expect("layer 0 up shape");
        eprintln!("layer 0 ffn_up_exps: {s:?}");
        let pk = ws.expert(0, ExpertRole::Up, 0).expect("expert 0");
        assert_eq!(
            pk.bytes_per_weight(s.reduction as usize, s.rows_per_expert as usize),
            0.5
        );
        let stored = ws.layer_expert_stored_bytes(0).expect("layer bytes");
        eprintln!(
            "layer 0 expert tensors stored: {:.2} GiB",
            stored as f64 / 2f64.powi(30)
        );
    }
}
