// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! A `std` CPU reference for the tile API, so that a lifted kernel can RUN.
//!
//! # The problem this solves
//!
//! `ptoas-rs --verify miri` reported itself unavailable, and the reason was
//! structural rather than missing work: `cargo miri run` needs a binary, a
//! lifted kernel is a library whose entry points are `extern "C"` functions,
//! and the kernel-side DSL is `#![no_core]` -- which cannot host a `#[test]`
//! and cannot share a crate graph with `std`, because it declares its own lang
//! items. So there was nothing for Miri to execute, and no way to write a
//! driver that executed it.
//!
//! This crate is the way out. It mirrors the tile API's SIGNATURES exactly
//! while being an ordinary `std` crate with ordinary Rust bodies. A lifted
//! kernel's source, with its `use tile_std::tile::*` repointed here and its
//! `no_core` attributes dropped, compiles and runs as normal Rust -- and
//! therefore runs under Miri, which is what the verification arm wanted.
//!
//! # What running it under Miri does and does not prove
//!
//! **Does:** that the lifted kernel's POINTER DISCIPLINE is sound. Miri sees
//! every load and store the kernel makes through the raw pointers it was
//! handed, and reports out-of-bounds access, misalignment, use of dangling or
//! aliased pointers, and reads of uninitialised memory. That is "check the
//! safety using the Rustc and Miri toolchains", and it is the property a lifted
//! kernel most needs, because the lift is choosing extents and the kernel
//! dereferences on their basis.
//!
//! **Does not:** prove the lifted kernel computes what the PTO meant. The
//! numerics here are this crate's reading of each op, so a kernel that loads,
//! adds and stores will agree with this crate's `add` whatever either of them
//! does -- the trap of checking a model against itself. Pinning the numerics
//! against the PTO needs an interpreter for the PTO, which this is not. What
//! the tests DO pin is this crate against CLOSED FORMS computed by hand, so the
//! reference is not merely self-consistent.
//!
//! # Fidelity rules
//!
//! * The signatures must match `tile_std::tile` exactly, including const
//!   generic ORDER (`tile_matmul_f32<M, K, N>`, which is not `<M, N, K>` -- an
//!   emitter got that wrong and produced well-formed, wrong MLIR).
//! * `Tile` is deliberately NOT `Copy` and not `Clone`, because the real one is
//!   not: each tile is consumed by exactly one operation, and the lift's
//!   "computed tile read twice" refusal exists to respect that. A `Copy`
//!   reference would make the lift's output compile here and not there.
//! * Row-major, contiguous, no padding. The device's padded allocations are not
//!   modelled; a kernel whose correctness depends on padding is outside what
//!   this can check, and `ptoas-rs` refuses those separately.

use std::marker::PhantomData;

/// The tile API, mirrored.
pub mod tile {
    use super::*;

    /// A tile of `ROWS × COLS` values, row-major and owned.
    ///
    /// Holds the data, where the real `Tile` holds a `buf_id` handle into
    /// device memory. That is the whole difference, and it is why this crate
    /// can run a kernel while the real one can only be compiled for a device.
    ///
    /// NOT `Copy` and NOT `Clone`, matching the real type. The comment there
    /// reads "each tile must be consumed by exactly one store operation,
    /// preventing double-DMA bugs" -- so a reference that allowed copying
    /// would accept lifted kernels the device rejects, which is the one way a
    /// reference like this can do active harm.
    #[derive(Debug)]
    pub struct Tile<const ROWS: usize, const COLS: usize, T> {
        data: Vec<T>,
        _phantom: PhantomData<T>,
    }

    impl<const ROWS: usize, const COLS: usize, T: Copy> Tile<ROWS, COLS, T> {
        fn from_vec(data: Vec<T>) -> Self {
            assert_eq!(
                data.len(),
                ROWS * COLS,
                "a {ROWS}x{COLS} tile needs {} values, got {}",
                ROWS * COLS,
                data.len()
            );
            Tile {
                data,
                _phantom: PhantomData,
            }
        }
        /// Read the tile out. For a driver checking results, not for kernels.
        pub fn to_vec(&self) -> Vec<T> {
            self.data.clone()
        }
        /// `(row, col)`, for a driver's assertions.
        pub fn at(&self, row: usize, col: usize) -> T {
            assert!(row < ROWS && col < COLS, "({row}, {col}) is outside {ROWS}x{COLS}");
            self.data[row * COLS + col]
        }
    }

    /// Load `ROWS × COLS` f32 from global memory.
    ///
    /// # Safety
    /// `gm` must be valid for reads of `ROWS * COLS` f32. Miri is the point:
    /// when it is not, this is where that gets reported.
    pub unsafe fn tile_load_f32<const ROWS: usize, const COLS: usize>(
        gm: *const f32,
    ) -> Tile<ROWS, COLS, f32> {
        let n = ROWS * COLS;
        let mut data = Vec::with_capacity(n);
        for i in 0..n {
            // One element at a time, deliberately. `slice::from_raw_parts`
            // would let Miri see a single wide access and report the first
            // problem only; an element-wise read reports the exact index, which
            // is what tells you whether the lift chose the wrong extent.
            data.push(unsafe { *gm.add(i) });
        }
        Tile::from_vec(data)
    }

    /// Store a tile to global memory.
    ///
    /// # Safety
    /// `gm` must be valid for writes of `ROWS * COLS` f32.
    pub unsafe fn tile_store_f32<const ROWS: usize, const COLS: usize>(
        gm: *mut f32,
        tile: Tile<ROWS, COLS, f32>,
    ) {
        for (i, v) in tile.data.iter().enumerate() {
            unsafe { *gm.add(i) = *v };
        }
    }

    macro_rules! elementwise {
        ($name:ident, $op:tt) => {
            pub fn $name<const ROWS: usize, const COLS: usize>(
                a: Tile<ROWS, COLS, f32>,
                b: Tile<ROWS, COLS, f32>,
            ) -> Tile<ROWS, COLS, f32> {
                let data = a
                    .data
                    .iter()
                    .zip(b.data.iter())
                    .map(|(x, y)| x $op y)
                    .collect();
                Tile::from_vec(data)
            }
        };
    }

    elementwise!(tile_add_f32, +);
    elementwise!(tile_sub_f32, -);
    elementwise!(tile_mul_f32, *);
    elementwise!(tile_div_f32, /);

    pub fn tile_exp_f32<const ROWS: usize, const COLS: usize>(
        src: Tile<ROWS, COLS, f32>,
    ) -> Tile<ROWS, COLS, f32> {
        Tile::from_vec(src.data.iter().map(|x| x.exp()).collect())
    }

    pub fn tile_neg_f32<const ROWS: usize, const COLS: usize>(
        src: Tile<ROWS, COLS, f32>,
    ) -> Tile<ROWS, COLS, f32> {
        Tile::from_vec(src.data.iter().map(|x| -x).collect())
    }

    /// Broadcast a `1 x COLS` row down `ROWS` rows.
    pub fn tile_col_expand_f32<const ROWS: usize, const COLS: usize>(
        src: Tile<1, COLS, f32>,
    ) -> Tile<ROWS, COLS, f32> {
        let mut data = Vec::with_capacity(ROWS * COLS);
        for _ in 0..ROWS {
            data.extend_from_slice(&src.data);
        }
        Tile::from_vec(data)
    }

    /// Sum each column of a `ROWS x COLS` tile into a `1 x COLS` row.
    pub fn tile_col_sum_f32<const ROWS: usize, const COLS: usize>(
        src: Tile<ROWS, COLS, f32>,
    ) -> Tile<1, COLS, f32> {
        let mut data = vec![0.0f32; COLS];
        for r in 0..ROWS {
            for c in 0..COLS {
                data[c] += src.data[r * COLS + c];
            }
        }
        Tile::from_vec(data)
    }

    /// Divide every element by a scalar.
    pub fn tile_divs_f32<const ROWS: usize, const COLS: usize>(
        src: Tile<ROWS, COLS, f32>,
        scalar: f32,
    ) -> Tile<ROWS, COLS, f32> {
        Tile::from_vec(src.data.iter().map(|x| x / scalar).collect())
    }

    /// A tile-to-tile copy, by reference.
    ///
    /// Mirrors `tile_std::tile::tile_copy_f32`, which cannot consume its source
    /// or it would not be a copy. Present here so a lifted kernel that
    /// duplicates a tile can still be RUN under Miri -- the kernel side has no
    /// copy at all, and a tier that can only compile it would say nothing about
    /// the accesses it makes.
    pub fn tile_copy_f32<const ROWS: usize, const COLS: usize>(
        src: &Tile<ROWS, COLS, f32>,
    ) -> Tile<ROWS, COLS, f32> {
        Tile::from_vec(src.data.clone())
    }

    /// `M×K @ K×N -> M×N`.
    ///
    /// The const generic order is `<M, K, N>`. It matches `tile_std`, and it is
    /// written out here because the order is unguessable from the names and
    /// because an emitter in this repository got it wrong -- emitting
    /// `(M, N, K)` extents, which is well-formed and computes the wrong shape.
    pub fn tile_matmul_f32<const M: usize, const K: usize, const N: usize>(
        a: Tile<M, K, f32>,
        b: Tile<K, N, f32>,
    ) -> Tile<M, N, f32> {
        let mut data = vec![0.0f32; M * N];
        for i in 0..M {
            for j in 0..N {
                let mut acc = 0.0f32;
                for k in 0..K {
                    acc += a.data[i * K + k] * b.data[k * N + j];
                }
                data[i * N + j] = acc;
            }
        }
        Tile::from_vec(data)
    }

    /// Row-wise softmax, max-shifted.
    ///
    /// Max-shifted because the device kernel is: subtracting the row max before
    /// `exp` is what keeps a large logit from overflowing to `inf`, and a
    /// reference that skipped it would disagree with the device exactly on the
    /// inputs that matter.
    pub fn tile_softmax_f32<const ROWS: usize, const COLS: usize>(
        src: Tile<ROWS, COLS, f32>,
    ) -> Tile<ROWS, COLS, f32> {
        let mut data = vec![0.0f32; ROWS * COLS];
        for r in 0..ROWS {
            let row = &src.data[r * COLS..(r + 1) * COLS];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for (c, v) in row.iter().enumerate() {
                let e = (v - max).exp();
                data[r * COLS + c] = e;
                sum += e;
            }
            for c in 0..COLS {
                data[r * COLS + c] /= sum;
            }
        }
        Tile::from_vec(data)
    }

    /// Row-wise RMS norm: `x / sqrt(mean(x^2) + eps)`.
    pub fn tile_rms_norm_f32<const ROWS: usize, const COLS: usize>(
        src: Tile<ROWS, COLS, f32>,
        eps: f32,
    ) -> Tile<ROWS, COLS, f32> {
        let mut data = vec![0.0f32; ROWS * COLS];
        for r in 0..ROWS {
            let row = &src.data[r * COLS..(r + 1) * COLS];
            let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / COLS as f32;
            let inv = 1.0f32 / (mean_sq + eps).sqrt();
            for (c, v) in row.iter().enumerate() {
                data[r * COLS + c] = v * inv;
            }
        }
        Tile::from_vec(data)
    }

    /// The `safe` submodule the checked lift targets, re-exporting the same
    /// functions. `tile_std` has both spellings and the lift uses both, so a
    /// reference missing one would fail to compile exactly the kernels the
    /// checked path emits.
    pub mod safe {
        pub use super::{
            tile_add_f32, tile_div_f32, tile_exp_f32, tile_matmul_f32, tile_mul_f32,
            tile_neg_f32, tile_rms_norm_f32, tile_softmax_f32, tile_sub_f32,
        };
    }
}
