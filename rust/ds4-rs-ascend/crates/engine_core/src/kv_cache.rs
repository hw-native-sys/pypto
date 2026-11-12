// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! KV cache for DS4 decode.
//!
//! DS4 uses MLA-style compressed KV: per layer we store a low-rank `kv_a` row
//! (length `kv_lora_rank`) and a short RoPE tail. The compressed cache is
//! decoded back to full-width K/V inside the attention kernel at decode time,
//! which is why the per-layer KV footprint is small enough to keep all 61
//! layers in RAM on 128 GB.
//!
//! Layout per layer:
//! - `kv_compressed`: `[max_seq_len][kv_lora_rank]` f32 (or f16 once we have a
//!   cast layer; CPU reference is f32 for simplicity).
//! - `k_rope`: `[max_seq_len][n_kv_heads * qk_rope_head_dim]` f32. The RoPE
//!   tail per head is concatenated row-wise.
//!
//! The Metal version will share storage with `Dsv4KvFp8Store` / `Dsv4Ratio4Shift`
//! kernels; this module is the CPU oracle.

#![allow(dead_code)]

use anyhow::{bail, Result};

#[derive(Debug, Clone, Copy)]
pub struct KvCacheShape {
    pub n_layers: usize,
    pub max_seq_len: usize,
    pub kv_lora_rank: usize,
    pub n_kv_heads: usize,
    pub qk_rope_head_dim: usize,
}

impl KvCacheShape {
    pub fn rope_row_dim(&self) -> usize {
        self.n_kv_heads * self.qk_rope_head_dim
    }

    pub fn bytes_per_layer(&self) -> usize {
        let f32_bytes = std::mem::size_of::<f32>();
        let comp = self.max_seq_len * self.kv_lora_rank * f32_bytes;
        let rope = self.max_seq_len * self.rope_row_dim() * f32_bytes;
        comp + rope
    }

    pub fn total_bytes(&self) -> usize {
        self.bytes_per_layer() * self.n_layers
    }
}

pub struct KvCache {
    pub shape: KvCacheShape,
    /// Flat storage per layer: `[layer][position * kv_lora_rank + lane]`.
    layers_compressed: Vec<Vec<f32>>,
    /// Flat storage per layer: `[layer][position * rope_row_dim + lane]`.
    layers_rope: Vec<Vec<f32>>,
    /// Number of valid positions stored so far (shared across layers — one
    /// decode step advances every layer in lockstep).
    pub len: usize,
}

impl KvCache {
    pub fn new(shape: KvCacheShape) -> Self {
        let layers_compressed = (0..shape.n_layers)
            .map(|_| vec![0.0f32; shape.max_seq_len * shape.kv_lora_rank])
            .collect();
        let layers_rope = (0..shape.n_layers)
            .map(|_| vec![0.0f32; shape.max_seq_len * shape.rope_row_dim()])
            .collect();
        Self {
            shape,
            layers_compressed,
            layers_rope,
            len: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.shape.max_seq_len
    }

    /// Append one row of compressed KV + rope key for every layer at the
    /// current position. After this call, `self.len` advances by 1.
    pub fn append(
        &mut self,
        kv_compressed_per_layer: &[Vec<f32>],
        k_rope_per_layer: &[Vec<f32>],
    ) -> Result<()> {
        if kv_compressed_per_layer.len() != self.shape.n_layers
            || k_rope_per_layer.len() != self.shape.n_layers
        {
            bail!(
                "append: expected {} layers, got compressed={} rope={}",
                self.shape.n_layers,
                kv_compressed_per_layer.len(),
                k_rope_per_layer.len()
            );
        }
        if self.len >= self.shape.max_seq_len {
            bail!(
                "append: KV cache full (len={}, max={})",
                self.len,
                self.shape.max_seq_len
            );
        }
        let pos = self.len;
        for l in 0..self.shape.n_layers {
            let c_src = &kv_compressed_per_layer[l];
            let r_src = &k_rope_per_layer[l];
            if c_src.len() != self.shape.kv_lora_rank {
                bail!(
                    "append: layer {l} compressed row has {} elements, expected {}",
                    c_src.len(),
                    self.shape.kv_lora_rank
                );
            }
            if r_src.len() != self.shape.rope_row_dim() {
                bail!(
                    "append: layer {l} rope row has {} elements, expected {}",
                    r_src.len(),
                    self.shape.rope_row_dim()
                );
            }
            let c_dst_start = pos * self.shape.kv_lora_rank;
            let c_dst =
                &mut self.layers_compressed[l][c_dst_start..c_dst_start + self.shape.kv_lora_rank];
            c_dst.copy_from_slice(c_src);

            let r_dst_start = pos * self.shape.rope_row_dim();
            let r_dst =
                &mut self.layers_rope[l][r_dst_start..r_dst_start + self.shape.rope_row_dim()];
            r_dst.copy_from_slice(r_src);
        }
        self.len += 1;
        Ok(())
    }

    pub fn compressed_layer(&self, layer: usize) -> &[f32] {
        &self.layers_compressed[layer][..self.len * self.shape.kv_lora_rank]
    }

    pub fn rope_layer(&self, layer: usize) -> &[f32] {
        &self.layers_rope[layer][..self.len * self.shape.rope_row_dim()]
    }

    /// Reset to position 0 without freeing storage (for prefill-snapshot loads).
    pub fn truncate_to(&mut self, new_len: usize) {
        self.len = new_len.min(self.shape.max_seq_len);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_shape() -> KvCacheShape {
        KvCacheShape {
            n_layers: 3,
            max_seq_len: 4,
            kv_lora_rank: 5,
            n_kv_heads: 2,
            qk_rope_head_dim: 6,
        }
    }

    #[test]
    fn fresh_cache_is_empty() {
        let kv = KvCache::new(tiny_shape());
        assert_eq!(kv.len, 0);
        assert_eq!(kv.capacity(), 4);
    }

    #[test]
    fn bytes_per_layer_is_sum_of_compressed_plus_rope() {
        let s = tiny_shape();
        // 4 * 5 * 4 = 80 (comp), 4 * (2*6) * 4 = 192 (rope) → 272
        assert_eq!(s.bytes_per_layer(), 272);
        assert_eq!(s.total_bytes(), 272 * 3);
    }

    #[test]
    fn append_one_row_advances_len() {
        let s = tiny_shape();
        let mut kv = KvCache::new(s);
        let c = vec![vec![1.0f32, 2.0, 3.0, 4.0, 5.0]; 3];
        let r = vec![vec![0.5f32; 12]; 3];
        kv.append(&c, &r).unwrap();
        assert_eq!(kv.len, 1);
        // Each layer should now have one row of [1,2,3,4,5] stored.
        for l in 0..3 {
            assert_eq!(kv.compressed_layer(l), &[1.0, 2.0, 3.0, 4.0, 5.0]);
            assert!(kv.rope_layer(l).iter().all(|&v| (v - 0.5).abs() < 1e-7));
        }
    }

    #[test]
    fn append_two_rows_concatenates() {
        let s = tiny_shape();
        let mut kv = KvCache::new(s);
        let c1 = vec![vec![1.0f32; 5]; 3];
        let c2 = vec![vec![2.0f32; 5]; 3];
        let r1 = vec![vec![0.0f32; 12]; 3];
        let r2 = vec![vec![1.0f32; 12]; 3];
        kv.append(&c1, &r1).unwrap();
        kv.append(&c2, &r2).unwrap();
        assert_eq!(kv.len, 2);
        // Layer 0 should have [1×5, 2×5] in compressed storage.
        let l0 = kv.compressed_layer(0);
        assert_eq!(l0.len(), 10);
        assert!(l0[..5].iter().all(|&v| v == 1.0));
        assert!(l0[5..].iter().all(|&v| v == 2.0));
    }

    #[test]
    fn append_rejects_wrong_layer_count() {
        let mut kv = KvCache::new(tiny_shape());
        let c = vec![vec![0.0f32; 5]; 2]; // wrong: 2 layers, expected 3
        let r = vec![vec![0.0f32; 12]; 3];
        assert!(kv.append(&c, &r).is_err());
    }

    #[test]
    fn append_rejects_wrong_row_dim() {
        let mut kv = KvCache::new(tiny_shape());
        let c = vec![vec![0.0f32; 4]; 3]; // wrong: 4, expected 5
        let r = vec![vec![0.0f32; 12]; 3];
        assert!(kv.append(&c, &r).is_err());
    }

    #[test]
    fn append_rejects_when_full() {
        let mut kv = KvCache::new(tiny_shape());
        let c = vec![vec![0.0f32; 5]; 3];
        let r = vec![vec![0.0f32; 12]; 3];
        for _ in 0..4 {
            kv.append(&c, &r).unwrap();
        }
        assert!(kv.append(&c, &r).is_err());
    }

    #[test]
    fn truncate_resets_len_without_realloc() {
        let mut kv = KvCache::new(tiny_shape());
        let c = vec![vec![0.0f32; 5]; 3];
        let r = vec![vec![0.0f32; 12]; 3];
        for _ in 0..3 {
            kv.append(&c, &r).unwrap();
        }
        kv.truncate_to(1);
        assert_eq!(kv.len, 1);
        // The underlying storage is unchanged; appending should resume at pos=1.
        kv.append(&c, &r).unwrap();
        assert_eq!(kv.len, 2);
    }
}
