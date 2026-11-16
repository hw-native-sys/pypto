//! Map the engine's weight ROLES onto a W8A8 package's tensor names.
//!
//! The decoder asks for `blk.N.attn_q_a.weight`; the deployment package stores
//! `layers.N.attn.wq_a.weight`. Both are DS4's own vocabulary, so this is a rename rather than an
//! architecture translation -- but a rename that has to be written down and tested, because a
//! wrong entry would load a real tensor with the wrong shape and fail three layers later with a
//! dimension error that looks like a kernel bug.
//!
//! The mapping is derived from the package itself (`examples/w8a8_model` walks it), not from a
//! document: every role below was checked to exist in the 0731 package.

use crate::safetensors::{Dtype, Package, TensorRef};
use anyhow::{anyhow, bail, Result};

/// A role the decoder needs, with the package name it corresponds to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoleTensor {
    /// The engine's role name, e.g. `mla_q_b` (the names are the decoder's, not the package's).
    pub role: &'static str,
    /// The package tensor that carries it, e.g. `layers.7.attn.wq_b.weight`.
    pub name: String,
}

/// One layer's roles, in the order the decoder asks for them.
pub fn layer_roles(layer: usize) -> Vec<RoleTensor> {
    let l = layer;
    // Both columns are FACT: the left from the decoder's own `deq("...")` calls, the right from
    // the package's tensor list. They are almost entirely different vocabularies -- `mla_q_a` vs
    // `attn.wq_a`, `moe_gate_shared` vs `ffn.shared_experts.w1` -- which is why this table exists
    // and why it was built by running the decoder, not by reading either side alone.
    let mut v = vec![
        RoleTensor { role: "attn_norm", name: format!("layers.{l}.attn_norm.weight") },
        RoleTensor { role: "mla_q_a", name: format!("layers.{l}.attn.wq_a.weight") },
        RoleTensor { role: "mla_q_b", name: format!("layers.{l}.attn.wq_b.weight") },
        RoleTensor { role: "mla_kv", name: format!("layers.{l}.attn.wkv.weight") },
        RoleTensor { role: "mla_q_a_norm", name: format!("layers.{l}.attn.q_norm.weight") },
        RoleTensor { role: "mla_kv_a_norm", name: format!("layers.{l}.attn.kv_norm.weight") },
        RoleTensor { role: "attn_output_a", name: format!("layers.{l}.attn.wo_a.weight") },
        RoleTensor { role: "attn_output_b", name: format!("layers.{l}.attn.wo_b.weight") },
        RoleTensor { role: "attn_sinks", name: format!("layers.{l}.attn.attn_sink") },
        RoleTensor { role: "ffn_norm", name: format!("layers.{l}.ffn_norm.weight") },
        RoleTensor { role: "moe_routing_table", name: format!("layers.{l}.ffn.gate.weight") },
        RoleTensor { role: "moe_gate_shared", name: format!("layers.{l}.ffn.shared_experts.w1.weight") },
        RoleTensor { role: "moe_down_shared", name: format!("layers.{l}.ffn.shared_experts.w2.weight") },
        RoleTensor { role: "moe_up_shared", name: format!("layers.{l}.ffn.shared_experts.w3.weight") },
        RoleTensor { role: "hc_attn_fn", name: format!("layers.{l}.hc_attn_fn") },
        RoleTensor { role: "hc_attn_base", name: format!("layers.{l}.hc_attn_base") },
        RoleTensor { role: "hc_attn_scale", name: format!("layers.{l}.hc_attn_scale") },
        RoleTensor { role: "hc_ffn_fn", name: format!("layers.{l}.hc_ffn_fn") },
        RoleTensor { role: "hc_ffn_base", name: format!("layers.{l}.hc_ffn_base") },
        RoleTensor { role: "hc_ffn_scale", name: format!("layers.{l}.hc_ffn_scale") },
    ];
    // The model-wide roles are carried on layer 0 so a single call site can resolve everything.
    if layer == 0 {
        v.push(RoleTensor { role: "token_embd", name: "embed.weight".to_string() });
        v.push(RoleTensor { role: "output", name: "head.weight".to_string() });
    }
    v
}

/// A routed expert's three planes. The package stores 256 SEPARATE tensors per layer, not one
/// stacked plane, so the engine's slice arithmetic (`e * stride`) does not carry over.
pub fn expert_roles(layer: usize, expert: usize) -> Vec<RoleTensor> {
    ["w1", "w2", "w3"]
        .iter()
        .zip(["ffn_gate_exps", "ffn_down_exps", "ffn_up_exps"])
        .map(|(w, role)| RoleTensor {
            role,
            name: format!("layers.{layer}.ffn.experts.{expert}.{w}.weight"),
        })
        .collect()
}

/// Resolve every role of one layer, checking it exists and reporting its dtype and shape.
///
/// A missing role is an ERROR here rather than a zero-filled placeholder: a silently absent
/// weight is a model that runs and produces nonsense.
pub fn resolve_layer(pkg: &Package, layer: usize) -> Result<Vec<(RoleTensor, TensorRef)>> {
    let mut out = Vec::new();
    for r in layer_roles(layer) {
        let t = pkg.tensor(&r.name).map_err(|_| {
            anyhow!("layer {layer}: role `{}` has no tensor `{}` in the package", r.role, r.name)
        })?;
        out.push((r, t));
    }
    Ok(out)
}

/// The roles that arrive quantised, and those that do not.
///
/// Recorded as data because it is a property of the EXPORT, not of a role: the same role could be
/// I8 in one package and BF16 in another, and a loader that hardcodes either will misread the other.
pub fn describe(t: &TensorRef) -> &'static str {
    match t.dtype {
        Dtype::I8 => "w8a8: (q - offset[i]) * scale[i] per output channel",
        Dtype::BF16 => "bf16: widened to f32",
        Dtype::F32 => "f32: read directly",
        Dtype::F16 => "f16: converted to f32",
        _ => "unsupported dtype",
    }
}


/// A package as a SOURCE of weight data, addressed by the decoder's roles.
///
/// This is the shape the converted `from_views_impl` will call: `dequant(layer, role)` returns
/// owned f32 (I8 dequantised per output channel, BF16 widened, F32 read), and `raw(layer, role)`
/// returns the tensor's bytes for the planes the engine keeps quantised on the host.
///
/// Both return OWNED data deliberately. `LayerViews::bytes_for` lends a slice into its single
/// mmap; a package spans 74 files, so lending is not available and pretending otherwise would
/// force a lifetime through a hot loop for no benefit.
pub struct PackageSource<'a> {
    pkg: &'a Package,
}

impl<'a> PackageSource<'a> {
    pub fn new(pkg: &'a Package) -> Self {
        Self { pkg }
    }

    fn name_for(&self, layer: usize, role: &str) -> Result<String> {
        layer_roles(layer)
            .into_iter()
            .find(|r| r.role == role)
            .map(|r| r.name)
            .ok_or_else(|| anyhow!("layer {layer}: no role named {role}"))
    }

    /// A role as f32, whatever its on-disk dtype.
    pub fn dequant(&self, layer: usize, role: &str) -> Result<Vec<f32>> {
        let name = self.name_for(layer, role)?;
        let t = self.pkg.tensor(&name)?;
        match t.dtype {
            Dtype::I8 => self.pkg.dequant_tensor(&name),
            _ => self.pkg.f32_tensor(&name),
        }
    }

    /// A role's raw bytes, for a caller that will dequantise on the device.
    pub fn raw(&self, layer: usize, role: &str) -> Result<Vec<u8>> {
        let name = self.name_for(layer, role)?;
        Ok(self.pkg.raw(&name)?.1)
    }

    /// One routed expert plane as RAW int8 bytes plus the per-output-channel scale and offset,
    /// i.e. exactly the three arguments the certified `__tile_matvec_w8a8_f32` kernel takes.
    ///
    /// This is the accessor the MoE path needs: 0.604 GB of f32 per token is avoided entirely by
    /// handing the device the bytes and letting it dequantise per row.
    pub fn raw_expert(
        &self,
        layer: usize,
        expert: usize,
        plane: u8,
    ) -> Result<(Vec<u8>, Vec<f32>, Vec<f32>)> {
        let role = match plane {
            1 => "ffn_gate_exps",
            2 => "ffn_down_exps",
            3 => "ffn_up_exps",
            _ => bail!("expert plane must be 1, 2 or 3"),
        };
        let e = expert_roles(layer, expert)
            .into_iter()
            .find(|r| r.role == role)
            .ok_or_else(|| anyhow!("layer {layer}: no expert role {role}"))?;
        let t = self.pkg.tensor(&e.name)?;
        if t.dtype != Dtype::I8 {
            bail!("{}: expected I8 for a w8a8 weight, got {:?}", e.name, t.dtype);
        }
        let (_, bytes) = self.pkg.raw(&e.name)?;
        // Scale and offset are companions of the same tensor; a missing offset means ZERO, which
        // is what the symmetric exports mean by omitting it.
        let scale = self.pkg.f32_tensor(&format!("{}_scale", e.name))?;
        let offset = self
            .pkg
            .f32_tensor(&format!("{}_offset", e.name))
            .unwrap_or_else(|_| vec![0.0; t.shape.first().copied().unwrap_or(1) as usize]);
        Ok((bytes, scale, offset))
    }

    /// One routed expert plane: `plane` is 1, 2 or 3 (`w1` gate, `w2` down, `w3` up).
    pub fn dequant_expert(&self, layer: usize, expert: usize, plane: u8) -> Result<Vec<f32>> {
        let role = match plane {
            1 => "ffn_gate_exps",
            2 => "ffn_down_exps",
            3 => "ffn_up_exps",
            _ => bail!("expert plane must be 1, 2 or 3"),
        };
        let e = expert_roles(layer, expert)
            .into_iter()
            .find(|r| r.role == role)
            .ok_or_else(|| anyhow!("layer {layer}: no expert role {role}"))?;
        let t = self.pkg.tensor(&e.name)?;
        match t.dtype {
            Dtype::I8 => self.pkg.dequant_tensor(&e.name),
            _ => self.pkg.f32_tensor(&e.name),
        }
    }
}


/// The two data operations the decoder loop performs on a weight source.
///
/// A GGUF view BORROWS from one mmap; a W8A8 package spans 74 files and can only OWN what it
/// returns. `bytes_for` therefore hands back a `Cow`, which is what lets one loop serve both
/// containers without the borrowed variant paying for an allocation.
pub trait WeightSource {
    /// A role as f32, whatever the on-disk dtype.
    fn dequant_f32(&self, h: &crate::layer_view::TensorHandle) -> Result<Vec<f32>>;
    /// The role's bytes, borrowed when the container can lend them.
    fn bytes_for(&self, h: &crate::layer_view::TensorHandle) -> Result<std::borrow::Cow<'_, [u8]>>;
    /// One layer's role-keyed handles. Borrowed for a GGUF view, owned for a package.
    fn layer(&self, idx: u32) -> Result<std::borrow::Cow<'_, crate::layer_view::LayerView>>;
    /// The model-wide roles (`token_embd`, `output`, the hc tails).
    fn globals(&self) -> Result<std::borrow::Cow<'_, std::collections::BTreeMap<&'static str, crate::layer_view::TensorHandle>>>;
}

impl WeightSource for crate::layer_view::LayerViews {
    fn dequant_f32(&self, h: &crate::layer_view::TensorHandle) -> Result<Vec<f32>> {
        self.dequant_f32_simple(h)
    }
    fn bytes_for(&self, h: &crate::layer_view::TensorHandle) -> Result<std::borrow::Cow<'_, [u8]>> {
        crate::layer_view::LayerViews::bytes_for(self, h)
    }
    fn layer(&self, idx: u32) -> Result<std::borrow::Cow<'_, crate::layer_view::LayerView>> {
        Ok(std::borrow::Cow::Borrowed(crate::layer_view::LayerViews::layer(self, idx)))
    }
    fn globals(
        &self,
    ) -> Result<std::borrow::Cow<'_, std::collections::BTreeMap<&'static str, crate::layer_view::TensorHandle>>>
    {
        Ok(std::borrow::Cow::Borrowed(&self.global))
    }
}

/// A package seen through the same trait.
///
/// The handles it is given carry PACKAGE names (`layers.7.attn.wq_a.weight`), which is why the
/// role map exists: the builder constructs handles with the names its own container understands,
/// and the loop below the trait never needs to know which container it is reading.
pub struct PackageViews<'a> {
    pkg: &'a Package,
}

impl<'a> PackageViews<'a> {
    pub fn new(pkg: &'a Package) -> Self {
        Self { pkg }
    }
}

impl WeightSource for PackageViews<'_> {
    fn dequant_f32(&self, h: &crate::layer_view::TensorHandle) -> Result<Vec<f32>> {
        // The handle's `ttype` is a GGML type; what matters here is the PACKAGE's own dtype
        // for that tensor, which is authoritative because it is what the bytes actually are.
        match self.pkg.tensor(&h.name)?.dtype {
            Dtype::I8 => self.pkg.dequant_tensor(&h.name),
            _ => self.pkg.f32_tensor(&h.name),
        }
    }
    fn bytes_for(&self, h: &crate::layer_view::TensorHandle) -> Result<std::borrow::Cow<'_, [u8]>> {
        Ok(std::borrow::Cow::Owned(self.pkg.raw(&h.name)?.1))
    }

    /// Build the layer from the ROLE MAP, naming each handle with the PACKAGE tensor that
    /// carries it. The loop keys on role names (`attn_q_a`), which are the same in both
    /// containers, so nothing below the trait has to translate.
    fn layer(&self, idx: u32) -> Result<std::borrow::Cow<'_, crate::layer_view::LayerView>> {
        use crate::layer_view::{LayerView, TensorHandle};
        let mut handles = std::collections::BTreeMap::new();
        for r in layer_roles(idx as usize) {
            let t = self.pkg.tensor(&r.name)?;
            handles.insert(
                r.role,
                TensorHandle {
                    name: r.name,
                    role: r.role,
                    // The package's own dtype is authoritative; `ttype` is a GGML type and is
                    // not consulted for package tensors.
                    ttype: crate::gguf::GgmlType::F32,
                    dims: t.shape.clone(),
                    abs_offset: 0,
                    byte_size: t.nbytes(),
                },
            );
        }
        Ok(std::borrow::Cow::Owned(LayerView { layer_idx: idx, handles }))
    }

    fn globals(
        &self,
    ) -> Result<std::borrow::Cow<'_, std::collections::BTreeMap<&'static str, crate::layer_view::TensorHandle>>>
    {
        use crate::layer_view::TensorHandle;
        let mut g = std::collections::BTreeMap::new();
        for (role, name) in [("token_embd", "embed.weight"), ("output", "head.weight")] {
            if let Ok(t) = self.pkg.tensor(name) {
                g.insert(
                    role,
                    TensorHandle {
                        name: name.to_string(),
                        role,
                        ttype: crate::gguf::GgmlType::F32,
                        dims: t.shape.clone(),
                        abs_offset: 0,
                        byte_size: t.nbytes(),
                    },
                );
            }
        }
        Ok(std::borrow::Cow::Owned(g))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_layer_role_has_a_distinct_package_name() {
        let roles = layer_roles(3);
        let mut names: Vec<&str> = roles.iter().map(|r| r.name.as_str()).collect();
        let n = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), n, "two roles map to the same tensor: {roles:?}");
        for r in &roles {
            assert!(r.name.starts_with("layers.3."), "layer 3 role escaped its layer: {r:?}");
        }
    }

    #[test]
    fn the_model_wide_roles_are_carried_once() {
        // They must appear when asked for layer 0 and NOT repeated for other layers, or a
        // caller summing per-layer bytes would count the 2 GB head 43 times.
        let z = layer_roles(0);
        assert!(z.iter().any(|r| r.role == "token_embd"));
        assert!(z.iter().any(|r| r.role == "output"));
        let other = layer_roles(1);
        assert!(!other.iter().any(|r| r.role == "token_embd"));
    }


    /// A one-tensor package: `layers.0.attn.wq_a.weight` as I8 with a per-channel scale.
    fn tiny_package(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("roles-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // two rows of two: q = [[1,2],[3,4]], scale [1,10], offset [0,1] -> [1,2,20,30]
        let q = [1u8, 2, 3, 4];
        let mut sc = Vec::new();
        for v in [1.0f32, 10.0] { sc.extend_from_slice(&v.to_le_bytes()); }
        let mut of = Vec::new();
        for v in [0.0f32, 1.0] { of.extend_from_slice(&v.to_le_bytes()); }
        let items: Vec<(String, String, u64, Vec<u8>)> = vec![
            ("layers.0.attn.wq_a.weight".into(), "I8".into(), 4, q.to_vec()),
            ("layers.0.attn.wq_a.weight_scale".into(), "F32".into(), 8, sc),
            ("layers.0.attn.wq_a.weight_offset".into(), "F32".into(), 8, of),
        ];
        let mut entries = Vec::new();
        let mut data = Vec::new();
        let mut off = 0u64;
        for (name, dtype, _n, bytes) in &items {
            let end = off + bytes.len() as u64;
            let shape = if name.ends_with(".weight") { "[2,2]" } else { "[2,1]" };
            entries.push(format!(
                "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":{shape},\"data_offsets\":[{off},{end}]}}"
            ));
            off = end;
            data.extend_from_slice(bytes);
        }
        let header = format!("{{{}}}", entries.join(","));
        let mut shard = Vec::new();
        shard.extend_from_slice(&(header.len() as u64).to_le_bytes());
        shard.extend_from_slice(header.as_bytes());
        shard.extend_from_slice(&data);
        std::fs::write(dir.join("s-00001.safetensors"), &shard).unwrap();
        let map: Vec<String> = items.iter().map(|(n, _, _, _)| format!("\"{n}\":\"s-00001.safetensors\"")).collect();
        std::fs::write(
            dir.join("quant_model_weights.safetensors.index.json"),
            format!("{{\"weight_map\":{{{}}}}}", map.join(",")),
        )
        .unwrap();
        dir
    }

    #[test]
    fn a_package_serves_the_same_trait_as_a_gguf_view() {
        // The point of the trait: the loop below it does not know which container it reads.
        // This exercises the package side with a handle whose NAME is a package name.
        let dir = tiny_package("trait");
        let pkg = Package::open(&dir).unwrap();
        let src = PackageViews::new(&pkg);
        let h = crate::layer_view::TensorHandle {
            name: "layers.0.attn.wq_a.weight".to_string(),
            role: "mla_q_a",
            ttype: crate::gguf::GgmlType::F32, // ignored: the package's own dtype is authoritative
            dims: vec![2, 2],
            abs_offset: 0,
            byte_size: 4,
        };
        assert_eq!(src.dequant_f32(&h).unwrap(), vec![1.0, 2.0, 20.0, 30.0]);
        let raw = src.bytes_for(&h).unwrap();
        assert_eq!(&*raw, &[1u8, 2, 3, 4]);
        assert!(matches!(raw, std::borrow::Cow::Owned(_)), "a package can only own its bytes");
    }

    #[test]
    fn expert_roles_are_per_expert_not_stacked() {
        let e = expert_roles(2, 17);
        assert_eq!(e.len(), 3);
        assert!(e.iter().all(|r| r.name.contains("experts.17.")));
        assert_eq!(e[0].name, "layers.2.ffn.experts.17.w1.weight");
        assert_eq!(e[1].name, "layers.2.ffn.experts.17.w2.weight");
        assert_eq!(e[2].name, "layers.2.ffn.experts.17.w3.weight");
    }
}
