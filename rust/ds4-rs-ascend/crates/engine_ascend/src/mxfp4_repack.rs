// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Load-time repack for the PACKED 4-bit MXFP4 matvec.
//!
//! `__tile_mul_mv_id_mxfp4_pk_f32` reads 4-bit weights and expands them in-kernel,
//! costing 0.5 bytes per weight where the folded path's host-expanded int8 costs 1.0.
//! That halving is what brings DS4-Flash from five chips to three. It is not free:
//! measured on device the packed matvec is ~1.61x slower than the folded one, because
//! the kernel is vector-bound rather than memory-bound. So this transform exists to buy
//! MEMORY, and a deployment with room for the folded planes should prefer those.
//!
//! The kernel cannot read GGUF's native MXFP4 layout directly. A tile row must be
//! 32-byte aligned, so a packed row holds 64 weights — a PAIR of 32-weight blocks — and
//! the two blocks have different scales. The layout that makes this work puts one whole
//! block in each nibble plane:
//!
//! ```text
//! w[(r*npair + g)*32 + i] = code(r, blk 2g+1, i) << 4 | code(r, blk 2g, i)
//! ```
//!
//! so the low-nibble plane is block `2g` in order and the high-nibble plane is block
//! `2g+1` in order. `x` and `d` carry their even and odd planes CONCATENATED rather than
//! interleaved, which lets the kernel reach both halves by partition-view offset with no
//! stride — deliberate, because a strided TSTORE compiles on this toolchain and then
//! silently ignores the stride.
//!
//! Two details of GGUF MXFP4 that are easy to get wrong, both already documented by
//! `engine_core::glm5_stream::dequant_mxfp4` and both load-bearing here:
//!
//! 1. Within a block, `qs[j] & 0xF` is element `j` and `qs[j] >> 4` is element `j + 16`.
//!    Reading them as consecutive pairs decodes every block into noise.
//! 2. The e8m0 scale has a zero special case: `e == 0` is 2^-127, not a denormal.
//!
//! The scale written here is HALVED, because the in-kernel decode produces `2*MXFP4[c]`
//! (all 16 code points are half-integers, so the doubled value is an exact small
//! integer). That is the same convention the folded int8 path already uses.

use engine_core::glm5_stream::e8m0_to_f32;

/// Elements per MXFP4 block.
pub const QK: usize = 32;
/// Stored bytes per MXFP4 block: one e8m0 scale byte then 16 packed bytes.
pub const BLK: usize = 17;
/// Weights per packed tile row — a block PAIR, because a 32-byte row holds 64 nibbles.
pub const WEIGHTS_PER_ROW: usize = 2 * QK;

#[derive(Debug, PartialEq, Eq)]
pub enum RepackError {
    /// `ne00` must be a multiple of 64: a packed row holds a block pair, so the block
    /// count has to be even.
    OddBlockCount { ne00: usize },
    /// The GGUF tensor is not `ne0` rows of `ne00/32` 17-byte blocks.
    WrongLength {
        got: usize,
        want: usize,
        ne00: usize,
        ne0: usize,
    },
}

impl std::fmt::Display for RepackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RepackError::OddBlockCount { ne00 } => write!(
                f,
                "ne00 ({ne00}) must be a multiple of 64: a 32-byte packed tile row holds \
                 a PAIR of 32-weight blocks, so the block count must be even"
            ),
            RepackError::WrongLength {
                got,
                want,
                ne00,
                ne0,
            } => write!(
                f,
                "MXFP4 tensor is {got} bytes, expected {want} for [{ne00} x {ne0}] \
                 ({} blocks of {BLK} B)",
                ne00 / QK * ne0
            ),
        }
    }
}

impl std::error::Error for RepackError {}

/// The two device planes derived from one MXFP4 weight tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct PackedWeight {
    /// `[ne0*npair x 32]` u8 — nibble-plane repacked codes.
    pub w: Vec<u8>,
    /// `[ne0 x nb]` f32 — even-block scales then odd-block scales, each ALREADY halved.
    pub d: Vec<f32>,
}

impl PackedWeight {
    /// Bytes the kernel reads per weight, for comparison against the folded path's 1.0.
    pub fn bytes_per_weight(&self, ne00: usize, ne0: usize) -> f64 {
        self.w.len() as f64 / (ne00 * ne0) as f64
    }
}

/// The same weights with the scale plane kept at MXFP4's NATIVE one byte per block.
///
/// This is the layout that takes DS4-Flash from 179.6 GiB to 155.4: 0.53125 bytes per weight
/// against [`PackedWeight`]'s 0.625, because a 32-weight block carries one scale BYTE rather than
/// an expanded f32. It buys no chip on its own — three either way — but it turns 4.2 GiB of spare
/// HBM into 28.4, and since prefill streams every expert once it also cuts ~15% off the bytes
/// crossing the host link.
///
/// The exponent is stored **biased by -128** in a signed byte, which is a device-driven choice
/// and not cosmetic. The kernel's only route from a byte to a lane is `si8 -> f16`, a SIGNED
/// widen; with a raw e8m0 byte every scale above 1.0 has the high bit set and comes out negative,
/// needing a mask to undo. Biased, the widen is the identity on `e - 128` for the whole domain
/// and one `+128` recovers `e` exactly. The plane is one byte per block either way, so the bias
/// is free.
#[derive(Debug, Clone, PartialEq)]
pub struct PackedWeightNative {
    /// `[ne0*npair x 32]` u8 — nibble-plane repacked codes, identical to [`PackedWeight::w`].
    pub w: Vec<u8>,
    /// `[ne0 x nb]` i8 — `e - 128` per block, even-block exponents then odd-block.
    ///
    /// NOT halved. The halving that [`PackedWeight::d`] applies on the host is a multiply by
    /// 0.5 that the expansion does instead — an exponent byte cannot carry it (`e - 1` would
    /// turn `e == 0` into a negative shift).
    pub e: Vec<i8>,
}

/// The exact f32 value of MXFP4's `e == 0`, which means 2^-127 and not a denormal-encoded zero.
///
/// Named because it appears in three places that must agree: the reference dequant, this
/// crate's expansion, and the kernel's `tmaxs` fixup.
pub const E8M0_ZERO_VALUE: f32 = 5.877_471_754_111_438e-39;

impl PackedWeightNative {
    /// Bytes the kernel reads per weight, codes plus the block's share of the scale.
    pub fn bytes_per_weight(&self, ne00: usize, ne0: usize) -> f64 {
        (self.w.len() + self.e.len()) as f64 / (ne00 * ne0) as f64
    }

    /// Expand the native plane to the halved f32 scales the matvec already expects.
    ///
    /// This mirrors the DEVICE chain step for step rather than calling `e8m0_to_f32`, because it
    /// is the device chain that has to be right and this is what pins it:
    ///
    /// ```text
    ///   si8 -> f16 -> +128 -> f32 -> i32 -> << 23 -> reinterpret f32 -> max(2^-127) -> * 0.5
    /// ```
    ///
    /// Every step is verified on 910c over all 256 byte values (`tmov` between differently-typed
    /// tiles turned out to be a REINTERPRET, which is what makes the shift usable at all;
    /// `pto.tcast` does not exist). The `max` is what handles `e == 0`: bits `0 << 23` are exactly
    /// `+0.0`, and every `e >= 1` yields at least 2^-126, so the max lifts only that one case and
    /// is the identity everywhere else.
    pub fn expand_scales(&self) -> Vec<f32> {
        self.e
            .iter()
            .map(|&b| {
                let e = (b as i32 + 128) as u32; // the kernel's f16 +128, exact below 2048
                let bits = e << 23;
                f32::from_bits(bits).max(E8M0_ZERO_VALUE) * 0.5
            })
            .collect()
    }
}

/// Extract the 4-bit code of element `i` (`0..32`) from a block's 16 packed bytes.
///
/// This is the mapping from detail (1) above: the low nibble of `qs[j]` is element `j`
/// and the high nibble is element `j + 16`.
#[inline]
fn code_at(qs: &[u8], i: usize) -> u8 {
    if i < QK / 2 {
        qs[i] & 0x0F
    } else {
        qs[i - QK / 2] >> 4
    }
}

/// Validate `src`'s shape and return `(nb, npair)`. Shared so the two repacks cannot disagree
/// about what a well-formed tensor is.
fn repack_dims(src_len: usize, ne00: usize, ne0: usize) -> Result<(usize, usize), RepackError> {
    if ne00 % WEIGHTS_PER_ROW != 0 {
        return Err(RepackError::OddBlockCount { ne00 });
    }
    let nb = ne00 / QK;
    let want = nb * ne0 * BLK;
    if src_len != want {
        return Err(RepackError::WrongLength {
            got: src_len,
            want,
            ne00,
            ne0,
        });
    }
    Ok((nb, nb / 2))
}

/// Repack one MXFP4 weight tensor keeping the scale plane NATIVE.
///
/// Deliberately NOT implemented by calling [`repack_weight`] and re-encoding its f32 scales, nor
/// the reverse. The two compute their scale planes independently — this one from the stored byte,
/// that one through `e8m0_to_f32` — so a test comparing `expand_scales()` against `PackedWeight::d`
/// is comparing two different derivations. Sharing the code would make that test vacuous, which
/// this project has already been caught by once.
pub fn repack_weight_native(
    src: &[u8],
    ne00: usize,
    ne0: usize,
) -> Result<PackedWeightNative, RepackError> {
    let (nb, npair) = repack_dims(src.len(), ne00, ne0)?;
    let mut w = vec![0u8; ne0 * npair * QK];
    let mut e = vec![0i8; ne0 * nb];
    for r in 0..ne0 {
        let row = &src[r * nb * BLK..(r + 1) * nb * BLK];
        for g in 0..npair {
            let be = &row[(2 * g) * BLK..(2 * g + 1) * BLK];
            let bo = &row[(2 * g + 1) * BLK..(2 * g + 2) * BLK];
            let (qe, qo) = (&be[1..BLK], &bo[1..BLK]);
            let base = (r * npair + g) * QK;
            for i in 0..QK {
                w[base + i] = (code_at(qo, i) << 4) | code_at(qe, i);
            }
            // The stored byte, biased so the device's signed widen is the identity.
            e[r * nb + g] = (be[0] as i32 - 128) as i8;
            e[r * nb + npair + g] = (bo[0] as i32 - 128) as i8;
        }
    }
    Ok(PackedWeightNative { w, e })
}

/// Repack one MXFP4 weight tensor into the planes the packed matvec reads.
///
/// `src` is the tensor exactly as GGUF stores it: `ne0` rows, each `ne00/32` blocks of
/// `BLK` bytes.
pub fn repack_weight(src: &[u8], ne00: usize, ne0: usize) -> Result<PackedWeight, RepackError> {
    if ne00 % WEIGHTS_PER_ROW != 0 {
        return Err(RepackError::OddBlockCount { ne00 });
    }
    let nb = ne00 / QK;
    let npair = nb / 2;
    let want = nb * ne0 * BLK;
    if src.len() != want {
        return Err(RepackError::WrongLength {
            got: src.len(),
            want,
            ne00,
            ne0,
        });
    }

    let mut w = vec![0u8; ne0 * npair * QK];
    let mut d = vec![0f32; ne0 * nb];
    for r in 0..ne0 {
        let row = &src[r * nb * BLK..(r + 1) * nb * BLK];
        for g in 0..npair {
            let be = &row[(2 * g) * BLK..(2 * g + 1) * BLK];
            let bo = &row[(2 * g + 1) * BLK..(2 * g + 2) * BLK];
            let (qe, qo) = (&be[1..BLK], &bo[1..BLK]);
            let base = (r * npair + g) * QK;
            for i in 0..QK {
                w[base + i] = (code_at(qo, i) << 4) | code_at(qe, i);
            }
            // Halved, because the kernel decodes to 2*MXFP4[c].
            d[r * nb + g] = e8m0_to_f32(be[0]) * 0.5;
            d[r * nb + npair + g] = e8m0_to_f32(bo[0]) * 0.5;
        }
    }
    Ok(PackedWeight { w, d })
}

/// Reorder an activation vector into the even-blocks-then-odd-blocks plane order.
///
/// `O(ne00)` per matvec against a weight plane of `O(ne0*ne00/2)`, so this is
/// negligible — it is the price of keeping both weight halves stride-free.
pub fn permute_activation(x: &[f32], ne00: usize) -> Result<Vec<f32>, RepackError> {
    let mut out = vec![0f32; x.len()];
    permute_activation_into(x, &mut out, ne00)?;
    Ok(out)
}

/// As [`permute_activation`], writing into a caller-owned buffer so a decode loop can
/// reuse one allocation across steps.
pub fn permute_activation_into(
    x: &[f32],
    out: &mut [f32],
    ne00: usize,
) -> Result<(), RepackError> {
    if ne00 % WEIGHTS_PER_ROW != 0 {
        return Err(RepackError::OddBlockCount { ne00 });
    }
    assert_eq!(x.len(), ne00, "activation must be ne00 long");
    assert_eq!(out.len(), ne00, "output must be ne00 long");
    let npair = ne00 / QK / 2;
    for g in 0..npair {
        for i in 0..QK {
            out[g * QK + i] = x[(2 * g) * QK + i];
            out[(npair + g) * QK + i] = x[(2 * g + 1) * QK + i];
        }
    }
    Ok(())
}

/// The doubled MXFP4 code points — `2 * MXFP4[c]`, every one an exact small integer.
/// This is what the kernel's piecewise-linear decode produces, and the reason the scale
/// is halved.
pub const MXFP4_DOUBLED: [f32; 16] = [
    0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, -0.0, -1.0, -2.0, -3.0, -4.0, -6.0, -8.0, -12.0,
];

/// CPU replay of exactly what the packed kernel computes, given the same planes.
///
/// This is the executable form of the kernel's contract: if this disagrees with a direct
/// MXFP4 dequant-and-dot, the repack is wrong, and the tests below assert it does not.
/// Keeping it here means the contract is checked in the crate rather than only in a
/// device runner.
pub fn matvec_ref(pk: &PackedWeight, xperm: &[f32], ne00: usize, ne0: usize) -> Vec<f32> {
    let nb = ne00 / QK;
    let npair = nb / 2;
    let mut out = vec![0f32; ne0];
    for r in 0..ne0 {
        let mut acc = 0f64;
        for g in 0..npair {
            let base = (r * npair + g) * QK;
            let (mut se, mut so) = (0f64, 0f64);
            for i in 0..QK {
                let byte = pk.w[base + i];
                se += MXFP4_DOUBLED[(byte & 0x0F) as usize] as f64 * xperm[g * QK + i] as f64;
                so += MXFP4_DOUBLED[(byte >> 4) as usize] as f64
                    * xperm[(npair + g) * QK + i] as f64;
            }
            acc += se * pk.d[r * nb + g] as f64 + so * pk.d[r * nb + npair + g] as f64;
        }
        out[r] = acc as f32;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_core::glm5_stream::dequant_mxfp4;

    /// Deterministic pseudo-random MXFP4 tensor: `ne0` rows of `ne00/32` blocks.
    fn synth(ne00: usize, ne0: usize, seed: u32) -> Vec<u8> {
        let nb = ne00 / QK;
        let mut s = seed | 1;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        };
        let mut v = vec![0u8; nb * ne0 * BLK];
        for b in 0..nb * ne0 {
            // Keep exponents in a range where the products stay well inside f32, so a
            // mismatch here means a layout bug and not overflow.
            v[b * BLK] = 118 + (rnd() % 18) as u8;
            for j in 1..BLK {
                v[b * BLK + j] = (rnd() & 0xFF) as u8;
            }
        }
        v
    }

    fn xs(ne00: usize, seed: u32) -> Vec<f32> {
        let mut s = seed | 1;
        (0..ne00)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                ((s & 0xFFFF) as f32 / 65536.0 - 0.5) * 2.0
            })
            .collect()
    }

    /// The contract, end to end: repacking and replaying the kernel must agree with a
    /// direct dequant-and-dot through `engine_core`'s reference. That reference shares no
    /// code and no layout with the repack, which is what makes agreement meaningful.
    #[test]
    fn repack_then_replay_matches_direct_dequant_dot() {
        for &(ne00, ne0) in &[(64usize, 3usize), (128, 5), (1024, 7), (4096, 2)] {
            let src = synth(ne00, ne0, 0x1234_5678);
            let x = xs(ne00, 0x9abc_def0);
            let pk = repack_weight(&src, ne00, ne0).expect("repack");
            let xp = permute_activation(&x, ne00).expect("permute");
            let got = matvec_ref(&pk, &xp, ne00, ne0);

            let nb = ne00 / QK;
            for r in 0..ne0 {
                let row = &src[r * nb * BLK..(r + 1) * nb * BLK];
                let wf = dequant_mxfp4(row);
                assert_eq!(wf.len(), ne00);
                let want: f64 = (0..ne00).map(|i| wf[i] as f64 * x[i] as f64).sum();
                let err = (got[r] as f64 - want).abs();
                let tol = want.abs().max(1.0) * 1e-6;
                assert!(
                    err <= tol,
                    "[{ne00}x{ne0}] row {r}: replay {} vs dequant-dot {want} (err {err})",
                    got[r]
                );
            }
        }
    }

    /// The halving must be in the SCALE, not the codes: the packed bytes are the raw
    /// 4-bit codes, so a byte's nibbles must index the doubled table directly.
    #[test]
    fn packed_bytes_are_raw_codes_and_scale_carries_the_half() {
        let (ne00, ne0) = (64usize, 1usize);
        let src = synth(ne00, ne0, 7);
        let pk = repack_weight(&src, ne00, ne0).unwrap();
        let nb = ne00 / QK;
        for g in 0..nb / 2 {
            let be = &src[(2 * g) * BLK..(2 * g + 1) * BLK];
            let bo = &src[(2 * g + 1) * BLK..(2 * g + 2) * BLK];
            for i in 0..QK {
                let byte = pk.w[g * QK + i];
                assert_eq!(byte & 0x0F, code_at(&be[1..], i), "even code at {i}");
                assert_eq!(byte >> 4, code_at(&bo[1..], i), "odd code at {i}");
            }
            assert_eq!(pk.d[g], e8m0_to_f32(be[0]) * 0.5, "even scale halved");
            assert_eq!(pk.d[nb / 2 + g], e8m0_to_f32(bo[0]) * 0.5, "odd scale halved");
        }
    }

    /// Detail (1): the low nibble is element `j`, the high nibble element `j+16`.
    /// Reading them as consecutive pairs is the documented way to decode noise, so it is
    /// pinned rather than left to the reader.
    #[test]
    fn nibble_mapping_is_split_not_consecutive() {
        let mut qs = [0u8; 16];
        qs[0] = 0xA3; // element 0 -> 3, element 16 -> 0xA
        assert_eq!(code_at(&qs, 0), 0x3);
        assert_eq!(code_at(&qs, 16), 0xA);
        // The consecutive-pair misreading would put 0xA at element 1.
        assert_eq!(code_at(&qs, 1), 0x0, "element 1 comes from qs[1], not qs[0]>>4");
    }

    /// The e8m0 zero special case must survive the halving rather than collapse to a
    /// denormal or to zero.
    #[test]
    fn e8m0_zero_exponent_survives_the_halving() {
        let (ne00, ne0) = (64usize, 1usize);
        let mut src = synth(ne00, ne0, 11);
        src[0] = 0; // even block of the first pair
        let pk = repack_weight(&src, ne00, ne0).unwrap();
        let want = e8m0_to_f32(0) * 0.5;
        assert_eq!(pk.d[0], want);
        assert!(want > 0.0 && want.is_finite(), "2^-127/2 must stay finite: {want}");
    }

    /// A packed row is a block PAIR, so an odd block count has no valid layout and must
    /// be refused rather than silently truncated.
    #[test]
    fn odd_block_count_is_refused() {
        let e = repack_weight(&[], 32, 1).unwrap_err(); // 1 block
        assert_eq!(e, RepackError::OddBlockCount { ne00: 32 });
        assert!(e.to_string().contains("multiple of 64"));
        // 1056 = 33 blocks: the same shape the emitter refuses.
        assert_eq!(
            permute_activation(&vec![0.0; 1056], 1056).unwrap_err(),
            RepackError::OddBlockCount { ne00: 1056 }
        );
    }

    /// A truncated or mis-shaped tensor must be reported, not read past.
    #[test]
    fn wrong_length_is_reported_with_both_shapes() {
        let src = synth(64, 2, 3);
        let e = repack_weight(&src[..src.len() - 1], 64, 2).unwrap_err();
        match e {
            RepackError::WrongLength { got, want, .. } => {
                assert_eq!(want, 2 * 2 * BLK);
                assert_eq!(got, want - 1);
            }
            other => panic!("expected WrongLength, got {other:?}"),
        }
    }

    /// The DEVICE algorithm, checked against the reference dequant over the WHOLE byte domain.
    ///
    /// `expand_scales` mirrors the kernel chain (`<< 23`, reinterpret, `max(2^-127)`) rather than
    /// calling `e8m0_to_f32`, so this is what proves the two agree. Bit-for-bit, not to a
    /// tolerance: both sides are exact powers of two, so any tolerance would hide a real error.
    #[test]
    fn native_scale_expansion_matches_the_reference_for_all_256_bytes() {
        for e in 0u32..=255 {
            let biased = (e as i32 - 128) as i8;
            let pw = PackedWeightNative {
                w: Vec::new(),
                e: vec![biased],
            };
            let got = pw.expand_scales()[0];
            let want = e8m0_to_f32(e as u8) * 0.5;
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "e={e}: expansion gave {got:e}, reference {want:e}"
            );
        }
        // The two ends are the ones an implementation gets wrong, so name them.
        let at = |e: u32| {
            PackedWeightNative { w: Vec::new(), e: vec![(e as i32 - 128) as i8] }.expand_scales()[0]
        };
        assert_eq!(at(0), E8M0_ZERO_VALUE * 0.5, "e==0 must be 2^-127, halved");
        assert_eq!(at(127), 0.5, "e==127 is 2^0, halved");
        assert!(at(255).is_infinite(), "e==255 overflows to inf, as the reference does");
    }

    /// A lossless RE-ENCODING: the native layout must reproduce the f32 plane exactly, and the
    /// code plane must be untouched. Two independent derivations, compared bitwise.
    #[test]
    fn native_layout_is_a_lossless_reencoding_of_the_f32_one() {
        for (ne00, ne0) in [(256usize, 8usize), (128, 3), (4096, 2)] {
            let src = synth(ne00, ne0, 0xC0FFEE ^ ne00 as u32);
            let f32p = repack_weight(&src, ne00, ne0).expect("repack");
            let natp = repack_weight_native(&src, ne00, ne0).expect("repack native");
            assert_eq!(natp.w, f32p.w, "code plane must be identical at {ne00}x{ne0}");
            let expanded = natp.expand_scales();
            assert_eq!(expanded.len(), f32p.d.len());
            for (i, (a, b)) in expanded.iter().zip(f32p.d.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "scale {i} at {ne00}x{ne0}: native {a:e} vs f32 plane {b:e}"
                );
            }
        }
    }

    /// The reason for the layout: 0.53125 bytes per weight against 0.625.
    ///
    /// Also guards the bias, which is the part a reader would assume away: `e - 128` must ACTUALLY
    /// differ from the raw byte, or the device's signed widen is being relied on to do something
    /// it does not do.
    #[test]
    fn native_plane_is_thirty_four_thirty_seconds_of_a_byte_per_weight() {
        let (ne00, ne0) = (4096usize, 4usize);
        let src = synth(ne00, ne0, 7);
        let p = repack_weight_native(&src, ne00, ne0).expect("repack");
        assert_eq!(p.bytes_per_weight(ne00, ne0), 0.5 + 1.0 / 32.0);
        let f32p = repack_weight(&src, ne00, ne0).expect("repack");
        let f32_bpw = (f32p.w.len() + f32p.d.len() * 4) as f64 / (ne00 * ne0) as f64;
        assert_eq!(f32_bpw, 0.625);
        // synth() puts exponents in 118..136, so both signs of the bias must occur.
        assert!(
            p.e.iter().any(|&b| b < 0) && p.e.iter().any(|&b| b >= 0),
            "the test tensor must exercise both sides of the -128 bias, or the sign handling \
             is untested; got range {:?}..{:?}",
            p.e.iter().min(),
            p.e.iter().max()
        );
    }

    /// The bias must be recoverable for the extreme bytes too — `0` and `255` are where an
    /// `as i8` / `as u8` round trip goes wrong.
    #[test]
    fn the_bias_round_trips_for_every_byte() {
        for e in 0u32..=255 {
            let biased = (e as i32 - 128) as i8;
            let back = (biased as i32 + 128) as u32;
            assert_eq!(back, e, "byte {e} biased to {biased} came back as {back}");
        }
    }

    /// The whole point: 0.5 bytes per weight against the folded path's 1.0.
    #[test]
    fn packed_plane_is_half_a_byte_per_weight() {
        let (ne00, ne0) = (4096usize, 64usize);
        let pk = repack_weight(&synth(ne00, ne0, 5), ne00, ne0).unwrap();
        assert_eq!(pk.bytes_per_weight(ne00, ne0), 0.5);
        assert_eq!(pk.w.len(), ne0 * (ne00 / 64) * 32);
        assert_eq!(pk.d.len(), ne0 * (ne00 / 32));
    }

    /// The permutation must be a permutation — every activation present exactly once,
    /// with the even blocks first.
    #[test]
    fn activation_permutation_is_a_bijection() {
        let ne00 = 256usize;
        let x: Vec<f32> = (0..ne00).map(|i| i as f32).collect();
        let p = permute_activation(&x, ne00).unwrap();
        let mut seen = p.clone();
        seen.sort_by(f32::total_cmp);
        assert_eq!(seen, x, "not a permutation");
        // Block 0 stays first; block 1 moves to the start of the odd plane.
        let npair = ne00 / QK / 2;
        assert_eq!(&p[0..QK], &x[0..QK]);
        assert_eq!(&p[npair * QK..npair * QK + QK], &x[QK..2 * QK]);
    }
}
