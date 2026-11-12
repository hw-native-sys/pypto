//! A minimal safetensors reader, for the Ascend W8A8 deployment package.
//!
//! The package the engine is deployed with (`quant_model_weights-*.safetensors` +
//! `quant_model_description.json`) is NOT GGUF, so the GGUF loader cannot open it. This
//! module reads the safetensors framing only:
//!
//! ```text
//! [ u64 header_len ][ header_len bytes of JSON ][ tensor data, 8-byte aligned ]
//! ```
//!
//! and leaves the quantisation semantics to the caller. Deliberately narrow: no serde, no
//! writer, one JSON shape. The alternative -- pulling a JSON dependency into a crate that
//! parses GGUF by hand -- would be the only dependency in this file's path.

use anyhow::{anyhow, bail, Context, Result};
use std::collections::HashMap;
use std::path::Path;

use crate::layer_view::MmapSource;

/// The element types the W8A8 package uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32,
    F16,
    BF16,
    I8,
    U8,
    I32,
    Other,
}

impl Dtype {
    fn parse(s: &str) -> Dtype {
        match s {
            "F32" => Dtype::F32,
            "F16" => Dtype::F16,
            "BF16" => Dtype::BF16,
            "I8" => Dtype::I8,
            "U8" => Dtype::U8,
            "I32" => Dtype::I32,
            _ => Dtype::Other,
        }
    }

    /// Bytes per element, or `None` for a type this reader does not size.
    pub fn width(self) -> Option<usize> {
        match self {
            Dtype::F32 | Dtype::I32 => Some(4),
            Dtype::F16 | Dtype::BF16 => Some(2),
            Dtype::I8 | Dtype::U8 => Some(1),
            Dtype::Other => None,
        }
    }
}

/// One tensor's entry in a shard's header.
#[derive(Debug, Clone)]
pub struct TensorRef {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    /// Byte range within the DATA section (i.e. after the header).
    pub offsets: (u64, u64),
}

impl TensorRef {
    pub fn nbytes(&self) -> u64 {
        self.offsets.1 - self.offsets.0
    }

    pub fn n_elements(&self) -> u64 {
        self.shape.iter().product()
    }
}

/// One `*.safetensors` file, memory-mapped and indexed by its header.
pub struct Shard {
    mmap: MmapSource,
    data_start: u64,
    tensors: HashMap<String, TensorRef>,
}

impl Shard {
    pub fn open(path: &Path) -> Result<Self> {
        let mmap = MmapSource::open(path)
            .with_context(|| format!("opening {}", path.display()))?;
        let bytes = mmap.as_slice();
        if bytes.len() < 8 {
            bail!("{}: shorter than a safetensors length prefix", path.display());
        }
        let hlen = u64::from_le_bytes(bytes[..8].try_into().unwrap());
        let hlen = usize::try_from(hlen)
            .map_err(|_| anyhow!("{}: header length {hlen} does not fit usize", path.display()))?;
        if bytes.len() < 8 + hlen {
            bail!(
                "{}: header says {hlen} bytes but the file holds {}",
                path.display(),
                bytes.len() - 8
            );
        }
        let text = std::str::from_utf8(&bytes[8..8 + hlen])
            .with_context(|| format!("{}: header is not UTF-8", path.display()))?;
        let tensors = parse_header(text)
            .with_context(|| format!("{}: parsing the tensor header", path.display()))?;
        Ok(Self {
            mmap,
            data_start: 8 + hlen as u64,
            tensors,
        })
    }

    pub fn tensors(&self) -> &HashMap<String, TensorRef> {
        &self.tensors
    }

    pub fn get(&self, name: &str) -> Option<&TensorRef> {
        self.tensors.get(name)
    }

    /// The tensor's bytes, borrowed from the mapping.
    pub fn bytes(&self, t: &TensorRef) -> &[u8] {
        let lo = (self.data_start + t.offsets.0) as usize;
        let hi = (self.data_start + t.offsets.1) as usize;
        &self.mmap.as_slice()[lo..hi]
    }
}

// ---------------------------------------------------------------------------
// The JSON the header needs: {"name": {"dtype":"I8","shape":[2,3],
// "data_offsets":[0,12]}, "__metadata__": {...}}. Only that shape is accepted;
// anything else is an error rather than a silent skip, because a mis-read header
// would put every tensor at the wrong offset.
// ---------------------------------------------------------------------------

fn parse_header(text: &str) -> Result<HashMap<String, TensorRef>> {
    let mut p = P { b: text.as_bytes(), i: 0 };
    p.ws();
    p.expect(b'{')?;
    let mut out = HashMap::new();
    loop {
        p.ws();
        if p.eat(b'}') {
            break;
        }
        let name = p.string()?;
        p.ws();
        p.expect(b':')?;
        p.ws();
        if name == "__metadata__" {
            p.skip_value()?;
        } else {
            out.insert(name.clone(), p.tensor(&name)?);
        }
        p.ws();
        if p.eat(b',') {
            continue;
        }
        p.expect(b'}')?;
        break;
    }
    Ok(out)
}

struct P<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> P<'a> {
    fn ws(&mut self) {
        while self.i < self.b.len() && (self.b[self.i] as char).is_whitespace() {
            self.i += 1;
        }
    }

    fn eat(&mut self, c: u8) -> bool {
        if self.i < self.b.len() && self.b[self.i] == c {
            self.i += 1;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, c: u8) -> Result<()> {
        if self.eat(c) {
            Ok(())
        } else {
            bail!("expected {:?} at byte {}", c as char, self.i)
        }
    }

    fn string(&mut self) -> Result<String> {
        self.expect(b'"')?;
        let mut s = String::new();
        while self.i < self.b.len() {
            let c = self.b[self.i];
            self.i += 1;
            match c {
                b'"' => return Ok(s),
                b'\\' => {
                    let e = *self.b.get(self.i).ok_or_else(|| anyhow!("truncated escape"))?;
                    self.i += 1;
                    s.push(match e {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'n' => '\n',
                        b't' => '\t',
                        b'r' => '\r',
                        b'u' => {
                            // \uXXXX: the header only uses it for metadata keys, which are
                            // skipped, so a stub that consumes the 4 hex digits is enough.
                            self.i += 4;
                            '?'
                        }
                        other => other as char,
                    });
                }
                other => s.push(other as char),
            }
        }
        bail!("unterminated string")
    }

    fn number(&mut self) -> Result<u64> {
        let start = self.i;
        while self.i < self.b.len()
            && (self.b[self.i].is_ascii_digit() || self.b[self.i] == b'-')
        {
            self.i += 1;
        }
        std::str::from_utf8(&self.b[start..self.i])?
            .parse::<i64>()
            .map(|v| v.max(0) as u64)
            .map_err(|e| anyhow!("number: {e}"))
    }

    fn tensor(&mut self, name: &str) -> Result<TensorRef> {
        self.expect(b'{')?;
        let mut dtype = None;
        let mut shape = Vec::new();
        let mut offsets = None;
        loop {
            self.ws();
            if self.eat(b'}') {
                break;
            }
            let key = self.string()?;
            self.ws();
            self.expect(b':')?;
            self.ws();
            match key.as_str() {
                "dtype" => dtype = Some(Dtype::parse(&self.string()?)),
                "shape" => {
                    self.expect(b'[')?;
                    loop {
                        self.ws();
                        if self.eat(b']') {
                            break;
                        }
                        shape.push(self.number()?);
                        self.ws();
                        if !self.eat(b',') {
                            self.expect(b']')?;
                            break;
                        }
                    }
                }
                "data_offsets" => {
                    self.expect(b'[')?;
                    let a = self.number()?;
                    self.ws();
                    self.expect(b',')?;
                    self.ws();
                    let b = self.number()?;
                    self.ws();
                    self.expect(b']')?;
                    offsets = Some((a, b));
                }
                _ => self.skip_value()?,
            }
            self.ws();
            if self.eat(b',') {
                continue;
            }
            self.expect(b'}')?;
            break;
        }
        Ok(TensorRef {
            name: name.to_string(),
            dtype: dtype.ok_or_else(|| anyhow!("{name}: no dtype"))?,
            shape,
            offsets: offsets.ok_or_else(|| anyhow!("{name}: no data_offsets"))?,
        })
    }

    fn skip_value(&mut self) -> Result<()> {
        self.ws();
        match self.b.get(self.i).copied() {
            Some(b'"') => {
                self.string()?;
            }
            Some(b'[') | Some(b'{') => {
                let (open, close) = if self.b[self.i] == b'[' {
                    (b'[', b']')
                } else {
                    (b'{', b'}')
                };
                let mut depth = 0usize;
                while self.i < self.b.len() {
                    let c = self.b[self.i];
                    self.i += 1;
                    if c == b'"' {
                        // skip a string wholesale so a bracket inside it cannot count
                        self.i -= 1;
                        self.string()?;
                        continue;
                    }
                    if c == open {
                        depth += 1;
                    } else if c == close {
                        depth -= 1;
                        if depth == 0 {
                            return Ok(());
                        }
                    }
                }
                bail!("unterminated container")
            }
            Some(_) => {
                while self.i < self.b.len() {
                    let c = self.b[self.i];
                    if c == b',' || c == b'}' || c == b']' || (c as char).is_whitespace() {
                        break;
                    }
                    self.i += 1;
                }
            }
            None => bail!("unexpected end of header"),
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Quantisation
// ---------------------------------------------------------------------------

/// Dequantise a W8A8 (per-output-channel) matrix: `w[i,k] = (q[i,k] - off[i]) * scale[i]`.
///
/// `scale` and `offset` are per OUTPUT CHANNEL, i.e. one entry per row of `q`. This is the
/// arithmetic the certified `__tile_matvec_w8a8_f32` kernel performs on the device; this
/// function is the reference the host uses, and the two are expected to agree to f32.
pub fn dequant_w8a8(q: &[i8], scale: &[f32], offset: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(q.len(), rows * cols, "weights: {rows}x{cols}");
    assert_eq!(scale.len(), rows, "one scale per output channel");
    assert_eq!(offset.len(), rows, "one offset per output channel");
    let mut out = vec![0.0f32; rows * cols];
    for i in 0..rows {
        let s = scale[i];
        let o = offset[i];
        for k in 0..cols {
            out[i * cols + k] = (q[i * cols + k] as f32 - o) * s;
        }
    }
    out
}


// ---------------------------------------------------------------------------
// The package: an index of which shard holds which tensor, opened lazily.
// ---------------------------------------------------------------------------

/// A W8A8 deployment package: sharded safetensors plus the index that maps each
/// tensor name to its shard. `quant_model_description.json` says HOW each tensor is
/// quantised; the index says WHERE it lives.
pub struct Package {
    dir: std::path::PathBuf,
    shard_of: HashMap<String, String>,
    // Mutex, not RefCell: the dispatcher that will consume this holds its own `Mutex` fields and
    // must stay `Sync`. A `Rc` inside a `RefCell` would make the whole engine non-`Send`.
    open_shards: std::sync::Mutex<HashMap<String, std::sync::Arc<Shard>>>,
}

impl Package {
    /// Open a package directory by reading `quant_model_weights.safetensors.index.json`.
    pub fn open(dir: &Path) -> Result<Self> {
        let index_path = dir.join("quant_model_weights.safetensors.index.json");
        let text = std::fs::read_to_string(&index_path)
            .with_context(|| format!("reading {}", index_path.display()))?;
        let shard_of = parse_weight_map(&text)
            .with_context(|| format!("parsing {}", index_path.display()))?;
        if shard_of.is_empty() {
            bail!("{}: weight_map is empty", index_path.display());
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            shard_of,
            open_shards: std::sync::Mutex::new(HashMap::new()),
        })
    }

    pub fn n_tensors(&self) -> usize {
        self.shard_of.len()
    }

    /// Tensor names, sorted (a HashMap's order is not reproducible in a report).
    pub fn names(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self.shard_of.keys().map(|s| s.as_str()).collect();
        v.sort_unstable();
        v
    }

    /// The shard holding `name`, opened on first use and then reused.
    pub fn shard_for(&self, name: &str) -> Result<std::sync::Arc<Shard>> {
        let file = self
            .shard_of
            .get(name)
            .ok_or_else(|| anyhow!("no such tensor in the package: {name}"))?
            .clone();
        if let Some(s) = self.open_shards.lock().unwrap().get(&file) {
            return Ok(s.clone());
        }
        let shard = std::sync::Arc::new(Shard::open(&self.dir.join(&file))?);
        self.open_shards
            .lock()
            .unwrap()
            .insert(file, shard.clone());
        Ok(shard)
    }

    /// `(tensor ref)` for `name`, resolved through its shard.
    pub fn tensor(&self, name: &str) -> Result<TensorRef> {
        let shard = self.shard_for(name)?;
        shard
            .get(name)
            .cloned()
            .ok_or_else(|| anyhow!("{name}: in the index but not in its shard's header"))
    }
}

/// Extract `weight_map` (name -> shard file) from an index.json.
///
/// Narrow on purpose: the index also carries a `metadata` object of numbers, which is
/// skipped rather than parsed into anything.
fn parse_weight_map(text: &str) -> Result<HashMap<String, String>> {
    let mut p = P { b: text.as_bytes(), i: 0 };
    p.ws();
    p.expect(b'{')?;
    let mut out = HashMap::new();
    loop {
        p.ws();
        if p.eat(b'}') {
            break;
        }
        let key = p.string()?;
        p.ws();
        p.expect(b':')?;
        p.ws();
        if key == "weight_map" {
            p.expect(b'{')?;
            loop {
                p.ws();
                if p.eat(b'}') {
                    break;
                }
                let name = p.string()?;
                p.ws();
                p.expect(b':')?;
                p.ws();
                let file = p.string()?;
                out.insert(name, file);
                p.ws();
                if p.eat(b',') {
                    continue;
                }
                p.expect(b'}')?;
                break;
            }
        } else {
            p.skip_value()?;
        }
        p.ws();
        if p.eat(b',') {
            continue;
        }
        p.expect(b'}')?;
        break;
    }
    Ok(out)
}

impl Package {
    /// Raw bytes of `name`.
    pub fn raw(&self, name: &str) -> Result<(TensorRef, Vec<u8>)> {
        let shard = self.shard_for(name)?;
        let t = shard
            .get(name)
            .cloned()
            .ok_or_else(|| anyhow!("{name}: in the index but not in its shard's header"))?;
        Ok((t.clone(), shard.bytes(&t).to_vec()))
    }

    /// An unquantised tensor as f32. `F32` is reinterpreted and `BF16` widened; anything
    /// else is an error rather than a silent zero-fill, because a role that arrives as
    /// I8 here means the caller asked for the wrong tensor.
    pub fn f32_tensor(&self, name: &str) -> Result<Vec<f32>> {
        let (t, b) = self.raw(name)?;
        match t.dtype {
            Dtype::F32 => Ok(b
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()),
            Dtype::BF16 => Ok(b
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect()),
            Dtype::F16 => Ok(b
                .chunks_exact(2)
                .map(|c| half_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect()),
            other => bail!("{name}: expected a float tensor, got {other:?}"),
        }
    }

    /// A W8A8 tensor dequantised to f32: `w[i,k] = (q[i,k] - offset[i]) * scale[i]`.
    ///
    /// The scale/offset companions are per OUTPUT CHANNEL, which is the row axis here.
    /// This is the host-side reference for the certified device kernel.
    pub fn dequant_tensor(&self, name: &str) -> Result<Vec<f32>> {
        let (t, qb) = self.raw(name)?;
        if t.dtype != Dtype::I8 {
            bail!("{name}: expected I8 for a w8a8 weight, got {:?}", t.dtype);
        }
        let rows = t.shape.first().copied().unwrap_or(1) as usize;
        let cols: usize = t.shape.iter().skip(1).map(|d| *d as usize).product::<usize>().max(1);
        let q: Vec<i8> = qb.iter().map(|b| *b as i8).collect();
        if q.len() != rows * cols {
            bail!(
                "{name}: {} int8 for a {rows}x{cols} tensor",
                q.len()
            );
        }
        let scale = self.f32_tensor(&format!("{name}_scale"))?;
        // The offset is optional: many exports are symmetric and omit it, and a missing
        // companion must mean ZERO offset rather than an error (the certified kernel
        // subtracts offset[i], which is 0 for those packs).
        let offset = match self.f32_tensor(&format!("{name}_offset")) {
            Ok(v) => v,
            Err(_) => vec![0.0; rows],
        };
        Ok(dequant_w8a8(&q, &scale, &offset, rows, cols))
    }
}

/// IEEE binary16 -> f32.
fn half_to_f32(h: u16) -> f32 {
    let s = (h >> 15) & 1;
    let e = (h >> 10) & 0x1f;
    let m = h & 0x3ff;
    let v = if e == 0 {
        (m as f32) * 2f32.powi(-24)
    } else if e == 31 {
        if m == 0 { f32::INFINITY } else { f32::NAN }
    } else {
        (1.0 + (m as f32) / 1024.0) * 2f32.powi(e as i32 - 15)
    };
    if s == 1 { -v } else { v }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a safetensors file in memory: `n` tensors with the given dtypes/shapes.
    fn write_st(dtype: &str, shape: &[u64], payload: &[u8]) -> Vec<u8> {
        let header = format!(
            "{{\"w\":{{\"dtype\":\"{dtype}\",\"shape\":{shape:?},\"data_offsets\":[0,{}]}}}}",
            payload.len()
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(payload);
        out
    }

    fn open(bytes: &[u8]) -> Shard {
        let dir = std::env::temp_dir().join(format!(
            "st-{}-{}",
            std::process::id(),
            bytes.len()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("s.safetensors");
        std::fs::write(&p, bytes).unwrap();
        Shard::open(&p).unwrap()
    }


    /// A multi-tensor safetensors file: (name, dtype, shape, bytes) in order.
    fn write_multi(items: &[(&str, &str, Vec<u64>, Vec<u8>)]) -> Vec<u8> {
        let mut entries = Vec::new();
        let mut off = 0u64;
        let mut data = Vec::new();
        for (name, dtype, shape, bytes) in items {
            let end = off + bytes.len() as u64;
            entries.push(format!(
                "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":{shape:?},\"data_offsets\":[{off},{end}]}}"
            ));
            off = end;
            data.extend_from_slice(bytes);
        }
        let header = format!("{{{}}}", entries.join(","));
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&data);
        out
    }

    /// A package directory: one shard plus the index that points at it.
    fn package_dir(tag: &str, shard: &[u8], names: &[&str]) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("pkg-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("s-00001.safetensors"), shard).unwrap();
        let map: Vec<String> = names
            .iter()
            .map(|n| format!("\"{n}\":\"s-00001.safetensors\""))
            .collect();
        let idx = format!("{{\"metadata\":{{}},\"weight_map\":{{{}}}}}", map.join(","));
        std::fs::write(dir.join("quant_model_weights.safetensors.index.json"), idx).unwrap();
        dir
    }

    #[test]
    fn dequantises_a_package_tensor_with_its_scale_and_offset() {
        // 2x2: row 0 scale 1 offset 0; row 1 scale 10 offset 1 -> same numbers as the
        // unit test above, but now through the package reader.
        let q = vec![1u8, 2, 3, 4]; // read as i8: 1,2,3,4
        let mut sc = Vec::new();
        for v in [1.0f32, 10.0] { sc.extend_from_slice(&v.to_le_bytes()); }
        let mut of = Vec::new();
        for v in [0.0f32, 1.0] { of.extend_from_slice(&v.to_le_bytes()); }
        let shard = write_multi(&[
            ("w.weight", "I8", vec![2, 2], q),
            ("w.weight_scale", "F32", vec![2, 1], sc),
            ("w.weight_offset", "F32", vec![2, 1], of),
        ]);
        let dir = package_dir("w8a8", &shard, &["w.weight", "w.weight_scale", "w.weight_offset"]);
        let pkg = Package::open(&dir).unwrap();
        assert_eq!(pkg.dequant_tensor("w.weight").unwrap(), vec![1.0, 2.0, 20.0, 30.0]);
    }

    #[test]
    fn a_missing_offset_means_zero_not_an_error() {
        // Many exports are symmetric and omit `weight_offset`. The certified kernel
        // subtracts offset[i]; treating "absent" as an error would refuse those packs,
        // and treating it as anything but 0.0 would shift every weight.
        let mut sc = Vec::new();
        for v in [2.0f32, 2.0] { sc.extend_from_slice(&v.to_le_bytes()); }
        let shard = write_multi(&[
            ("w.weight", "I8", vec![2, 1], vec![3u8, 4]),
            ("w.weight_scale", "F32", vec![2, 1], sc),
        ]);
        let dir = package_dir("nooff", &shard, &["w.weight", "w.weight_scale"]);
        let pkg = Package::open(&dir).unwrap();
        assert_eq!(pkg.dequant_tensor("w.weight").unwrap(), vec![6.0, 8.0]);
    }

    #[test]
    fn bf16_widens_by_shifting_into_f32() {
        // 1.0 in bf16 is 0x3F80; widening must give exactly 1.0, and 0.5 (0x3F00) exactly
        // 0.5. A reader that byte-swapped would give ~3e-41 and this would fail loudly.
        let mut b = Vec::new();
        for h in [0x3F80u16, 0x3F00, 0x0000] { b.extend_from_slice(&h.to_le_bytes()); }
        let shard = write_multi(&[("n", "BF16", vec![3], b)]);
        let dir = package_dir("bf16", &shard, &["n"]);
        let pkg = Package::open(&dir).unwrap();
        assert_eq!(pkg.f32_tensor("n").unwrap(), vec![1.0, 0.5, 0.0]);
    }

    #[test]
    fn reads_an_int8_tensor_and_its_shape() {
        let payload: Vec<u8> = (0..12u8).collect();
        let bytes = write_st("I8", &[3, 4], &payload);
        let shard = open(&bytes);
        let t = shard.get("w").expect("tensor w");
        assert_eq!(t.dtype, Dtype::I8);
        assert_eq!(t.shape, vec![3, 4]);
        assert_eq!(t.n_elements(), 12);
        assert_eq!(shard.bytes(t), &payload[..]);
    }

    #[test]
    fn the_data_section_is_shifted_by_the_header() {
        // A reader that forgot the 8-byte prefix + header would return the header's tail.
        let payload = [1u8, 2, 3, 4];
        let bytes = write_st("U8", &[4], &payload);
        let shard = open(&bytes);
        let t = shard.get("w").unwrap();
        assert_eq!(shard.bytes(t), &payload[..]);
    }

    #[test]
    fn metadata_is_skipped_not_parsed_as_a_tensor() {
        let payload = [7u8; 4];
        let header = format!(
            "{{\"__metadata__\":{{\"format\":\"pt\",\"n\":[1,2]}},\"w\":{{\"dtype\":\"U8\",\
             \"shape\":[4],\"data_offsets\":[0,4]}}}}"
        );
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&payload);
        let shard = open(&bytes);
        assert!(shard.get("__metadata__").is_none());
        assert_eq!(shard.bytes(shard.get("w").unwrap()), &payload[..]);
    }

    #[test]
    fn dequant_is_per_output_channel_not_per_tensor() {
        // 2 rows of 2, distinct scale and offset per row. A per-tensor dequant (one scale
        // for everything) would give the same numbers for both rows.
        let q: [i8; 4] = [1, 2, 3, 4];
        let scale = [1.0f32, 10.0];
        let offset = [0.0f32, 1.0];
        let w = dequant_w8a8(&q, &scale, &offset, 2, 2);
        assert_eq!(w, vec![1.0, 2.0, 20.0, 30.0]);
    }

    #[test]
    fn a_truncated_header_is_an_error_not_a_panic() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&99u64.to_le_bytes());
        bytes.extend_from_slice(b"{}");
        let dir = std::env::temp_dir().join(format!("st-trunc-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.safetensors");
        std::fs::write(&p, &bytes).unwrap();
        assert!(Shard::open(&p).is_err());
    }
}
