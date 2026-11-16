// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Minimal AST for the subset of PTO we need to re-emit as Rust.
//!
//! Non-load-bearing attributes (v_row, blayout, slayout, fractal, pad) are
//! deliberately dropped -- they exist for `ptoas` codegen, not for
//! reconstructing a tile-API program. If S3 round-trip fidelity requires
//! them, they get folded back in as a pass-through struct.

use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    Bf16,
    I8,
    /// `i16` / `si16`. Carried so an artifact that mentions it round-trips;
    /// `tile_suffix` is `None`, because `tile_std` has NO `__tile_*_i16`
    /// intrinsic of any kind (measured 2026-10-31: 0 names).
    I16,
    /// `i32` / `si32`. Carried for round-trip only. `tile_suffix` is `None`
    /// even though `__tile_store_i32` exists, because `__tile_load_i32` does
    /// NOT -- the only i32 loader is `__tile_buf_load_i32`, which is the
    /// BUFFER api, not the tile api. A dtype whose load/store pair is half
    /// present cannot be lifted by an emitter that names both from one suffix.
    I32,
    /// `ui64`, MLIR's unsigned 64-bit spelling, which the repo's GMM
    /// artifacts use for their pointer arguments. Round-trip only: there is
    /// no `__tile_*_u64` intrinsic (measured 2026-10-31: 0 names).
    U64,
    /// `ui32`, MLIR's unsigned 32-bit spelling. Earns its variant for the
    /// sharpest possible reason: THIS PROJECT'S OWN PTO EMITTER WRITES IT, and
    /// this parser could not read it back. `pto.tsort32` requires its index
    /// tile to be `ui32`, so `rspto_emit` emits `dtype=ui32` for every topk and
    /// argsort kernel -- and the round-trip sweep then failed on
    /// `topk_v41_1x384_k6.pto` with "tile_buf missing/unknown dtype", a
    /// diagnostic that names neither the dtype nor the emitter that produced
    /// it. Round-trip only, like `U64`: there is no `__tile_*_u32` intrinsic
    /// (measured 2026-11-02 over the carried DSL: 0 names), so the lift still
    /// refuses it -- but now it refuses by NAME, which is a work item, rather
    /// than by failing to parse, which read as a corrupt file.
    U32,
}

impl DType {
    /// The Rust scalar a pointer to this dtype points at. Total, and total
    /// honestly: every variant has an exact Rust scalar, so this never has to
    /// substitute one width or signedness for another. What is NOT total is
    /// `tile_suffix` -- having a Rust spelling does not mean an intrinsic
    /// exists to pass it to.
    pub fn rust_scalar(&self) -> &'static str {
        match self {
            DType::F32 => "f32",
            // f16 / bf16 tiles are backed by `*const u16` at the intrinsic
            // boundary (the tile API uses u16 as the raw byte carrier).
            DType::F16 | DType::Bf16 => "u16",
            DType::I8 => "i8",
            DType::I16 => "i16",
            DType::I32 => "i32",
            DType::U64 => "u64",
            DType::U32 => "u32",
        }
    }

    /// The canonical PTO text for this dtype, for re-emitting `.pto`.
    ///
    /// Deliberately separate from `tile_suffix`. They coincide for f32/f16/
    /// bf16/i8 and that coincidence is why one function used to serve both --
    /// but `ui64` has a PTO spelling and no tile-API suffix at all, so a
    /// single function could only have served both by inventing one.
    ///
    /// Not a byte-for-byte echo of the input: `si8` and `si16` re-emit as
    /// `i8`/`i16`, which is the same type under MLIR's signless spelling, so
    /// the round-trip is lossless at the AST level even where it is not at the
    /// text level. `ui64` is NOT folded that way -- unsigned is a different
    /// type, not a different spelling.
    pub fn pto_spelling(&self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::Bf16 => "bf16",
            DType::I8 => "i8",
            DType::I16 => "i16",
            DType::I32 => "i32",
            DType::U64 => "ui64",
            DType::U32 => "ui32",
        }
    }

    /// Suffix for tile-API function names (`tile_load_f32`, `tile_add_f16`),
    /// or `None` when `tile_std` has no intrinsic family with this suffix.
    ///
    /// `None` is the whole point: a lift that cannot name a real intrinsic
    /// must REFUSE, and the one failure mode worth engineering against here is
    /// emitting a plausible name with nothing behind it. Callers surface the
    /// dtype in the diagnostic; see `unsupported_for_lift`.
    pub fn tile_suffix(&self) -> Option<&'static str> {
        match self {
            DType::F32 => Some("f32"),
            DType::F16 => Some("f16"),
            DType::Bf16 => Some("bf16"),
            DType::I8 => Some("i8"),
            DType::I16 => Some("i16"),
            DType::I32 | DType::U64 | DType::U32 => None,
        }
    }

    /// The diagnostic for a dtype that parses but cannot be lifted, naming the
    /// intrinsic that is missing rather than only the dtype that is unwanted.
    pub fn unsupported_for_lift(&self) -> String {
        let why = match self {
            DType::I16 => "tile_std has no __tile_*_i16 intrinsic",
            DType::I32 => {
                "__tile_load_i32 does not exist (only __tile_buf_load_i32, which is the buffer API)"
            }
            DType::U64 => "tile_std has no __tile_*_u64 intrinsic",
            DType::U32 => {
                "tile_std has no __tile_*_u32 intrinsic; `pto.tsort32` index tiles are \
                 ui32 by contract, so this needs an unsigned tile family, not a cast"
            }
            _ => "unsupported",
        };
        format!(
            "dtype `{}` parses but cannot be lifted to the tile API: {why}",
            self.pto_spelling()
        )
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim() {
            "f32" => Some(DType::F32),
            "f16" => Some(DType::F16),
            "bf16" => Some(DType::Bf16),
            // `si8` is MLIR's SIGNED 8-bit spelling of the same type: `i8` is
            // signless there, and this AST's I8 lowers to Rust `i8`. The e8 /
            // MXFP4 quant artifacts in the repo spell it `si8`, which the
            // parser rejected outright. `ui8` is deliberately NOT accepted --
            // unsigned is a different Rust scalar, and silently handing it the
            // signed lowering is the kind of substitution that compiles and
            // computes something else.
            "i8" | "si8" => Some(DType::I8),
            // Same signed/signless pairing as i8 above. These three are
            // accepted so the round-trip sweep stops rejecting 20 of the
            // repo's 47 tracked artifacts at the first type it reads; none of
            // them can be LIFTED, which `tile_suffix` is where that is said.
            "i16" | "si16" => Some(DType::I16),
            "i32" | "si32" => Some(DType::I32),
            // `ui8`/`ui16` stay rejected: nothing in the corpus needs them,
            // and the reason in the i8 note holds -- an unsigned type handed a
            // signed lowering compiles and computes something else. `ui64` and
            // `ui32` each earn a variant by APPEARING: `ui64` in the GMM
            // artifacts' pointer arguments, and `ui32` in this project's own
            // emitted PTO, where `pto.tsort32` requires it for index tiles.
            // Both are lift-refused, so parsing them adds a named refusal
            // rather than a silent substitution.
            "ui64" => Some(DType::U64),
            "ui32" => Some(DType::U32),
            _ => None,
        }
    }
}

/// An SSA identifier like `%pto3`, `%arg601`, `%c1024`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Ssa(pub String);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PtoType {
    /// `!pto.ptr<f32>`
    Ptr(DType),
    /// `!pto.tensor_view<?x?xf32>` (dynamic shape; concretized by the view that reads it)
    /// `!pto.tensor_view<?x?xf32>`, and the rank-5 NZ/fractal spellings the
    /// `gmm_*` cube artifacts use (`?x?x?x?x?xi8`).
    ///
    /// The rank is carried because re-emission has to print that many `?`s. It
    /// was hardcoded to 2 until 2026-11-01, so a rank-5 view came back rank-2
    /// -- and since the parser refused those files earlier for an unrelated
    /// reason, nothing had ever reached this printer with a rank-5 view.
    TensorView { rank: u32, dtype: DType },
    /// `!pto.partition_tensor_view<1x1024xf32>`
    PartitionView { rows: u32, cols: u32, dtype: DType },
    /// `!pto.tile_buf<loc=..., dtype=..., rows=R, cols=C, ...>` -- we keep only
    /// the fields the Rust emitter actually needs.
    TileBuf {
        rows: u32,
        cols: u32,
        /// The LOGICAL extent, from `v_row`/`v_col`. Equal to `rows`/`cols`
        /// unless the allocation is padded for the 512-byte fractal, which a
        /// row-reduce output is: `rows=8, cols=1` holding one `v_row=1,
        /// v_col=1` value. The Rust type a lifted kernel must name is this
        /// one, while the PTO allocation must keep the padded extent -- see
        /// `validate_active_tile_extents`.
        v_rows: u32,
        v_cols: u32,
        dtype: DType,
        loc: TileLoc,
    },
    /// `!pto.partition_tensor_view<1x4x16x16x32xi8>` — a view of rank other
    /// than 2, i.e. a fractal / NZ layout.
    ///
    /// The dims are kept as written because collapsing them to rows/cols would
    /// be a guess: in the corpus artifacts that use this, `1x4x16x16x32` has
    /// 32768 elements and the tile it loads into is 256x128, so neither "the
    /// last dim is cols" nor "the first is rows" recovers the tile shape.
    /// Faithful for a round-trip; refused by both lifts.
    ///
    /// Until 2026-11-01 `parse_rxcxty` refused rank > 2 outright, which cost
    /// five artifacts. (The refusal before THAT blamed the dtype: it reported
    /// "bad dtype in `1x4x16x16x32xi8`" because `16` is not a dtype, while the
    /// real dtype `i8` sat at the end and parses fine.)
    PartitionViewNd { dims: Vec<u32>, dtype: DType },
}

/// The tile-by-scalar ops. Separate from `BinKind`, and deliberately not a
/// superset of it: `pto.tsubs` and `pto.tdivs` do not appear in the corpus and
/// are not invented here.
/// Which core-identity query a `BlockQuery` is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockQueryKind {
    Idx,
    Num,
}

impl BlockQueryKind {
    pub fn pto_mnemonic(self) -> &'static str {
        match self {
            BlockQueryKind::Idx => "pto.get_block_idx",
            BlockQueryKind::Num => "pto.get_block_num",
        }
    }

    /// The `tile_std` function that answers this query.
    ///
    /// Both are extern fns at the CRATE ROOT (`tile_std/src/lib.rs:94-95`,
    /// returning `usize`), not in `tile_std::tile`, which is the only module
    /// the Rust emitter imports. The tile module's own `# Example` calls
    /// `get_block_idx()` while importing only three tile functions — and it is
    /// marked ```rust,ignore```, so that example has never been compiled.
    /// A documented pattern is not a verified one.
    pub fn tile_std_path(self) -> &'static str {
        match self {
            BlockQueryKind::Idx => "tile_std::get_block_idx",
            BlockQueryKind::Num => "tile_std::get_block_num",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarBinKind {
    /// `pto.tmuls`. The only one with a tile-API arm: `__tile_scale_f32` /
    /// `__tile_scale_f16` (measured 2026-10-31). Note the name -- there is no
    /// `__tile_muls_*` in the TILE api; the `__tile_muls_f32` that does exist
    /// lives in `lib.rs` over `UbBuf`, which is the BUFFER api, and reaching
    /// for it by name alone would cross the two.
    Muls,
    /// `pto.tadds` -> `tile_adds_f32`. It had no tile-API wrapper until
    /// 2026-11-04, when one was added because the PTO emitter grew the arm; the
    /// tier test in `tests/the_last_tier.rs` fired on the same day and said so
    /// ("may be liftable and this tier reopens"), which is the census working.
    Adds,
    /// `pto.tsubs` -> `tile_subs_f32`, added with `Adds` for symmetry.
    Subs,
    /// `pto.tmaxs`. No tile-API arm; parses and round-trips, refuses to lift.
    Maxs,
    /// `pto.tmins`. Absent from the corpus, included because the spelling is
    /// one the prefix dispatch would also have swallowed. No tile-API arm.
    Mins,
    /// `pto.tshls` — shift left by a scalar. 56 corpus instances, every one on
    /// an `i32` tile with an `i32` scalar.
    Shls,
    /// `pto.tshrs` — shift right by a scalar. 96 corpus instances, `i32`.
    Shrs,
    /// `pto.tands` — bitwise AND with a scalar. 128 corpus instances, `i32`.
    Ands,
}

impl ScalarBinKind {
    /// The PTO mnemonic, for re-emission.
    pub fn pto_mnemonic(&self) -> &'static str {
        match self {
            ScalarBinKind::Muls => "pto.tmuls",
            ScalarBinKind::Adds => "pto.tadds",
            ScalarBinKind::Subs => "pto.tsubs",
            ScalarBinKind::Maxs => "pto.tmaxs",
            ScalarBinKind::Mins => "pto.tmins",
            ScalarBinKind::Shls => "pto.tshls",
            ScalarBinKind::Shrs => "pto.tshrs",
            ScalarBinKind::Ands => "pto.tands",
        }
    }

    /// The tile-API base name for this op at `dtype`, or `None` when the tile
    /// API has no arm for it. `Muls` maps to `scale`, not to `muls`.
    pub fn tile_base(&self, dtype: DType) -> Option<&'static str> {
        match (self, dtype) {
            (ScalarBinKind::Muls, DType::F32 | DType::F16) => Some("scale"),
            (ScalarBinKind::Muls, DType::I16) => Some("scale"),
            (ScalarBinKind::Adds, DType::F32) => Some("adds"),
            (ScalarBinKind::Adds, DType::I16) => Some("adds"),
            (ScalarBinKind::Maxs, DType::I16) => Some("maxs"),
            (ScalarBinKind::Subs, DType::F32) => Some("subs"),
            // Not bf16: __tile_scale_bf16 does not exist, though
            // __tile_muls_bf16 does -- in the buffer API.
            //
            // The shift/mask family DOES have tile-API arms -- and only at i8.
            // Measured 2026-11-01 in `tile_std/src/tile.rs`:
            // `__tile_shls_i8`, `__tile_shrs_i8`, `__tile_ands_i8`, and no
            // other width. Every one of the 280 corpus instances is on an i32
            // tile, so the arm exists and does not fit: a half-present family,
            // the same shape as `__tile_store_i32` without a matching load.
            (ScalarBinKind::Shls, DType::I8) => Some("shls"),
            (ScalarBinKind::Shrs, DType::I8) => Some("shrs"),
            (ScalarBinKind::Ands, DType::I8) => Some("ands"),
            (ScalarBinKind::Shls, DType::I16) => Some("shls"),
            (ScalarBinKind::Shrs, DType::I16) => Some("shrs"),
            (ScalarBinKind::Ands, DType::I16) => Some("ands"),
            _ => None,
        }
    }

    /// Why this op cannot be lifted, naming what is missing.
    pub fn unsupported_for_lift(&self, dtype: DType) -> String {
        let dt = dtype.pto_spelling();
        match self {
            ScalarBinKind::Muls => format!(
                "`pto.tmuls` at dtype `{dt}` cannot be lifted: the tile API's scalar \
                 multiply is __tile_scale_*, which exists for f32 and f16 only"
            ),
            k @ (ScalarBinKind::Shls | ScalarBinKind::Shrs | ScalarBinKind::Ands) => {
                format!(
                    "`{}` at dtype `{dt}` cannot be lifted: the tile API has \
                     `__tile_{}_i8` and no other width, while every corpus instance \
                     of this op is on an i32 tile",
                    k.pto_mnemonic(),
                    match k {
                        ScalarBinKind::Shls => "shls",
                        ScalarBinKind::Shrs => "shrs",
                        ScalarBinKind::Ands => "ands",
                        _ => unreachable!(),
                    }
                )
            }
            k => format!(
                "`{}` cannot be lifted: the tile API has no scalar-operand arm for it \
                 (the __tile_{}_* that exist are buffer-API, over UbBuf)",
                k.pto_mnemonic(),
                match k {
                    ScalarBinKind::Adds => "adds",
                    ScalarBinKind::Maxs => "maxs",
                    ScalarBinKind::Mins => "mins",
                    _ => unreachable!(),
                }
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinKind {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnKind {
    Exp,
    Neg,
    Abs,
    /// `pto.tsqrt`. 10 corpus instances, all inside the rms-norm chain.
    /// `__tile_sqrt_f32` exists as an intrinsic with no safe wrapper, so this
    /// does not lift on its own.
    Sqrt,
    /// `pto.trsqrt`, the FUSED spelling, which `tile_std` can express and the
    /// sqrt/recip pair cannot.
    ///
    /// The note on `Recip` below says a fusion rule written against a `trsqrt`
    /// spelling "matches nothing in this corpus". That is true of the three MODEL
    /// corpora and false of the bench as a whole: `playground16/pto/rms_norm.pto`
    /// emits `pto.trsqrt ins(%x) outs(%y)` and refused on it being unmodelled.
    ///
    /// Unlike its two neighbours this one lifts on its own, because
    /// `tile_std::tile_rsqrt_f32` is a SAFE WRAPPER -- it takes `Tile<R, C, f32>`
    /// and returns one (tile.rs:2048) -- where `__tile_sqrt_f32` is a bare
    /// intrinsic and nothing in the tile API reciprocates. So the reason the
    /// pair cannot lift does not transfer to the fused op, which is why this is
    /// a separate kind rather than a fusion of the other two.
    Rsqrt,
    /// `pto.trecip`. 10 corpus instances, paired one-to-one with `tsqrt` --
    /// `mlir_to_pto` spells `rsqrt` as sqrt-then-reciprocal rather than using
    /// a fused op, which is why a fusion rule written against a `trsqrt`
    /// spelling matches nothing in this corpus. Nothing in the tile API
    /// reciprocates (`__tile_reciprocal_f32` is the buffer API over `UbBuf`).
    Recip,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowKind {
    Max,
    Sum,
}

/// The value of an `arith.constant`.
///
/// Floats are kept as the LITERAL TEXT plus its type, not as `f64`, for two
/// reasons. `f64` is not `Eq`, and this AST is compared for equality by the
/// round-trip verifier. And the comparison must not be fooled by
/// normalisation: the repo's own artifacts contain `0.00000000`, `0.000001`
/// and `128.0`, which a parse-and-reprint would silently rewrite. Keeping the
/// text means the round-trip proves the emitter preserved the constant rather
/// than that two floats happened to agree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstVal {
    /// `arith.constant N : index`, or an untyped one.
    ///
    /// Not "the only kind recorded in the parser's `consts` map" -- a width-typed
    /// integer is recorded there too, because a size operand may name it. It is
    /// the only kind whose SPELLING is `index`, which is the distinction this
    /// variant actually draws, and conflating the two re-emitted 172 corpus
    /// constants at the wrong type.
    Index(i64),
    /// Anything else, e.g. `arith.constant 128.0 : f16`.
    Typed { lit: String, ty: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PtoOp {
    /// `%dst = arith.constant N : index`, or a typed constant such as
    /// `arith.constant 128.0 : f16`.
    ScalarConst {
        dst: Ssa,
        value: ConstVal,
    },

    /// `%dst = pto.make_tensor_view %ptr, shape = [...], strides = [...] : !pto.tensor_view<?x?xTy>`
    /// `pto.barrier <PIPE_ALL>` — a pipeline barrier, 814 instances in this
    /// repo's corpus and the single most-dropped op.
    ///
    /// The pipe is carried as the verbatim text between the angle brackets
    /// rather than an enum: every corpus instance is `PIPE_ALL`, and a value
    /// this parser has not seen should round-trip rather than be refused or,
    /// worse, normalised to the one spelling it knows.
    ///
    /// A barrier is not decoration — dropping one changes device behaviour, and
    /// the repo's own codegen rule is that a `buf_store` in a loop must be
    /// followed by `pipe_barrier(PIPE_ALL)`. The Rust lift refuses it; see
    /// `emitter::emit_op`.
    Barrier { pipe: String },
    /// `%s = "pto.subview"(%f, %c2, %c0) {operandSegmentSizes = ..., sizes = [2, 32]} : (...) -> ...`
    ///
    /// The MLIR generic form, which is why the mnemonic is quoted. It was one
    /// of the three ops whose silent drop surfaced later and elsewhere as
    /// `no shape for SSA %s`, blaming the `tstore` that consumed the result.
    ///
    /// `operandSegmentSizes` is carried verbatim: it tells MLIR how the operand
    /// list splits across variadic groups, and while `1, 2, 0, 0` is derivable
    /// from one source plus two offsets in the single corpus instance, the two
    /// empty groups are not guaranteed to stay empty.
    Subview {
        dst: Ssa,
        src: Ssa,
        /// The dynamic offset operands, in order.
        offsets: Vec<Ssa>,
        sizes: [String; 2],
        segment_sizes: String,
        ty: PtoType,
    },
    /// `pto.tgatherb ins(%src, %idx : <tile>, <i32 tile>) outs(%dst : <tile>)`
    ///
    /// A block gather: `idx` holds offsets into `src`. The second operand is a
    /// tile of a different dtype from the first, which is why it cannot share
    /// the two-tile table's assumption that both match.
    GatherB { dst: Ssa, src: Ssa, idx: Ssa },
    /// `pto.tcvt ins(%src [{sat_mode = ...}] : <tile>) outs(%dst : <tile>)`
    ///
    /// A dtype conversion: `src` and `dst` differ in dtype, so the shapes map
    /// supplies both ends. 188 corpus instances across 8 pairs, measured
    /// 2026-11-01: f16->f32 ×71, si8->f16 ×26, f32->f16 ×26, i8->f16 ×22,
    /// i16->f16 ×16, f32->i32 ×14, f16->i16 ×8, f16->si8 ×5.
    ///
    /// `sat_mode` sits INSIDE the `ins` operand list, after the operand name
    /// (`ins(%div {sat_mode = #pto<saturation_mode ON>} : ...)`) -- a position
    /// no other op in this dialect uses. It decides whether an out-of-range
    /// value saturates or wraps, so it is carried, not dropped: 10 of the 188
    /// instances have it.
    Cvt {
        dst: Ssa,
        src: Ssa,
        sat_mode: Option<String>,
    },
    /// `%bidx = "pto.get_block_idx"() : () -> i64` and its `get_block_num`
    /// sibling — the kernel asking which core it is running on.
    ///
    /// These are how a partition offset becomes block-variant: the result is
    /// `arith.index_cast`-ed to an index and lands in
    /// `offsets = [%r, %c0]`, which is 42 of the corpus's partition views and
    /// the one offset shape no constant fold can resolve.
    ///
    /// The result type is carried verbatim (`i64` in every corpus instance)
    /// rather than assumed, since it is what the cast reads.
    BlockQuery {
        dst: Ssa,
        which: BlockQueryKind,
        ty: String,
    },
    /// `%i = arith.index_cast %bidx : i64 to index`
    IndexCast {
        dst: Ssa,
        src: Ssa,
        from: String,
        to: String,
    },
    /// `%dst = arith.muli %lhs, %rhs : index`
    MulI {
        dst: Ssa,
        lhs: Ssa,
        rhs: Ssa,
    },
    /// `%dst = arith.addi %lhs, %rhs : index`
    AddI {
        dst: Ssa,
        lhs: Ssa,
        rhs: Ssa,
    },
    /// `%dst = "pto.tgetval"(%tile, %idx) : (...) -> f32`
    GetVal {
        dst: Ssa,
        src: Ssa,
        idx: Ssa,
    },
    /// `%dst = arith.fptosi %src : f32 to i32`
    FpToSi {
        dst: Ssa,
        src: Ssa,
    },
    /// `%dst = pto.tci ins(%start : i32) outs(%dst : tile)`
    Tci {
        dst: Ssa,
        start: Ssa,
    },
    /// `%dst = pto.tsort32 ins(%values, %indices) outs(%dst)`
    TSort32 {
        dst: Ssa,
        values: Ssa,
        indices: Ssa,
    },
    /// `%dst = pto.tgather ins(%src, {maskPattern = #pto.mask_pattern<Pxxxx>}) outs(%dst)`
    TGather {
        dst: Ssa,
        src: Ssa,
        mask: u32,
        axis: String,
    },
    /// `%dst = pto.textract ins(%src, %row, %col) outs(%dst)`
    Textract {
        dst: Ssa,
        src: Ssa,
        row: Ssa,
        col: Ssa,
    },
    /// `%dst = pto.partition_view %src, offsets = [...], sizes = [...] : ... -> !pto.partition_tensor_view<1x4x16x16x32xi8>`
    ///
    /// The rank-N sibling of `PartitionView`. It binds no `TileShape`, because
    /// there is no honest rows/cols to bind, so anything that needs a shape
    /// from it fails closed.
    PartitionViewNd {
        dst: Ssa,
        src: Ssa,
        dims: Vec<u32>,
        dtype: DType,
        /// Rank-general, like `dims`. These were `[String; 2]` until
        /// 2026-11-01: the `gmm_*` cube artifacts declare rank-5 NZ partitions
        /// with five offsets and five sizes, two of which are loop-variant
        /// (`offsets = [%c0, %nz_n, %nz_k, %c0, %c0]`), so a fixed pair refused
        /// the file -- and had it not, it would have kept 2 of 5 and silently
        /// placed the window at the origin in the other three dims.
        offsets: Vec<String>,
        sizes: Vec<String>,
    },
    /// `scf.for %r = %lb to %ub step %st { ... }` — the first op in this AST
    /// with a REGION.
    ///
    /// 46 instances, and the dominant blocker: of the 33 artifacts still
    /// refused on 2026-11-01, **29 contain one**, and 7 contain nothing else
    /// unmodelled. A line-based parser reaches it with only its header, so the
    /// body is parsed by recursing into `parse_func_body`, which already stops
    /// at a closing brace.
    ///
    /// The bounds are SSA names, not folded integers: `%pto17`/`%pto18` are
    /// typically derived from `pto.get_block_idx`, so the loop extent is
    /// per-core and no constant map resolves it. The induction variable is the
    /// reason `offsets = [%r, %c0]` exists (42 corpus partition views), which
    /// is the one offset shape the lift cannot fold to zero.
    ///
    /// Body ops share the function's `consts` and `shapes` maps: MLIR scopes
    /// SSA names to the enclosing region, and the corpus reuses tile names
    /// freely across a loop boundary. There is no `scf.yield` anywhere in the
    /// corpus, so no loop-carried value is modelled; one would need its own
    /// variant rather than an extra field here.
    For {
        var: Ssa,
        lb: Ssa,
        ub: Ssa,
        step: Ssa,
        body: Vec<PtoOp>,
    },
    MakeTensorView {
        dst: Ssa,
        ptr: Ssa,
        dtype: DType,
        /// `shape = [...]` and `strides = [...]`, as the operand spellings
        /// appear in the text -- an SSA name (`%c1024`), a literal, or `?`.
        ///
        /// Carried verbatim rather than folded to integers because an operand
        /// may be loop-variant (`%r`), which no constant map can resolve.
        /// Until 2026-11-01 neither was in the AST at all and the reverse
        /// emitter wrote `[?, ?]`: two tensor views over different strides had
        /// identical ASTs, so the round-trip verifier called them equal, and
        /// the re-emitted text was not a valid program.
        ///
        /// `Vec`, not `[String; 2]`: the `gmm_*` cube artifacts declare rank-5
        /// NZ views with five shape and five stride operands. A fixed pair both
        /// refused those files and, had it not, would have kept 2 of 5 -- the
        /// same loss `PartitionViewNd` exists for, one op upstream. The LIFT
        /// still refuses any rank but 2; the round trip no longer has to.
        shape: Vec<String>,
        strides: Vec<String>,
        /// `{layout = #pto.layout<nz>}`, carried verbatim when present.
        ///
        /// Not decoration: NZ is a fractal layout, so the same pointer and the
        /// same dims describe a different memory order with and without it.
        /// Dropping it would have let an NZ view and an ND view compare equal.
        layout: Option<String>,
    },

    /// `%dst = pto.partition_view %src, offsets = [...], sizes = [%cR, %cC] : ... -> !pto.partition_tensor_view<RxCxTy>`
    PartitionView {
        dst: Ssa,
        src: Ssa,
        rows: u32,
        cols: u32,
        dtype: DType,
        /// `offsets = [a, b]` and `sizes = [c, d]`, verbatim.
        ///
        /// The offsets decide WHICH rows and columns of the parent this view
        /// addresses, and were dropped entirely until 2026-11-01. Corpus
        /// census that day: 119 partition views are `[%c0, %c0]` and **321 are
        /// not** -- `[%r, %c0]` ×42 (a loop-variant row offset), `[%c8, %c0]`
        /// through `[%c120, %c0]` ×9 each (a 16-way static row tiling),
        /// `[%c0, %r]` ×15. With the offset absent from the AST, those sixteen
        /// row tiles were the same AST, and `emitter::emit_op` lifted every one
        /// of them to `let p = v;` -- sixteen kernels all reading row 0, exit 0.
        ///
        /// Not reachable through the lift on 2026-11-01: all 8 fully-modelled
        /// artifacts use `[%c0, %c0]`, and every file with a non-zero offset
        /// also holds an op the parser refuses, so it never reaches the
        /// emitter. Modelling more ops is exactly what would have armed it,
        /// which is why this was fixed before any were added.
        offsets: [String; 2],
        sizes: [String; 2],
    },

    /// `%dst = pto.alloc_tile : !pto.tile_buf<rows=R, cols=C, dtype=Ty, ...>`
    AllocTile {
        dst: Ssa,
        ty: PtoType,
    },

    /// `pto.tload ins(%view : ...) outs(%tile : ...)`
    /// The partition view supplies rows/cols/dtype; the tile must agree.
    TLoad {
        view: Ssa,
        tile: Ssa,
        /// The view operand's OWN type.
        ///
        /// The reverse emitter used to rebuild it from the TILE's shape --
        /// printing `partition_tensor_view<{tile.rows}x{tile.cols}x{dtype}>`
        /// -- which is only right when the two agree. Measured 2026-11-01:
        /// **35 of the corpus's 487 tload/tstore lines have a view whose dims
        /// differ from its tile's**, e.g. a `16x1` view re-emitted as `16x8`
        /// and a `1x1` view as `8x1`. Reconstructing one operand's type from
        /// another operand's shape is the same substitution as every other
        /// loss found today, and a rank-N view cannot be reconstructed at all.
        view_ty: PtoType,
    },

    /// `pto.tstore ins(%tile : ...) outs(%view : ...)`
    TStore {
        tile: Ssa,
        view: Ssa,
        /// The view operand's own type; see `TLoad::view_ty`.
        view_ty: PtoType,
    },

    /// Pointwise binary: `pto.tadd ins(%a, %b : ...) outs(%dst : ...)`.
    BinOp {
        kind: BinKind,
        dst: Ssa,
        a: Ssa,
        b: Ssa,
    },

    /// Pointwise tile-by-SCALAR: `pto.tmuls ins(%a, %s : <tile>, f32) outs(%dst : <tile>)`.
    ///
    /// A separate variant rather than a `BinOp` whose `b` happens to be a
    /// scalar: the whole defect this models was a scalar SSA sitting in a tile
    /// operand slot, and sharing the variant would have kept that
    /// representable.
    BinOpScalar {
        kind: ScalarBinKind,
        dst: Ssa,
        a: Ssa,
        /// The scalar operand. Names an `arith.constant`, or any SSA the
        /// producing op left as a scalar; it is NOT a tile and has no shape.
        scalar: Ssa,
        /// The scalar operand's own spelling from the `ins` type list, verbatim.
        ///
        /// It is NOT always `f32`. Corpus census 2026-11-01: `pto.tadds` is
        /// `f16` in 30 instances, `pto.tmuls`/`pto.tmaxs` are `i16` in 24, and
        /// `pto.tshls`/`pto.tshrs`/`pto.tands` (not yet modelled) use `i32`.
        /// The first version of this variant did not carry the field and the
        /// reverse emitter wrote a literal `f32`, so every `f16` and `i16`
        /// scalar was re-emitted at the wrong width -- and NEITHER gate saw it,
        /// because `parse_ins2_outs1` discards the `ins` types, so the loss
        /// never reached the AST that the round-trip compares. A round-trip can
        /// only see what the AST carries; the fix is the field, not the test.
        ///
        /// It happens to equal the tile's dtype in every corpus instance, but
        /// deriving it from there is the same assumption that produced the bug.
        scalar_ty: String,
    },

    /// Pointwise unary: `pto.texp ins(%src : ...) outs(%dst : ...)`.
    UnOp {
        kind: UnKind,
        dst: Ssa,
        src: Ssa,
    },

    /// Row reduction (R×C -> R×1 col_major): `pto.trowmax ins(%src, %scratch) outs(%dst)`.
    /// The scratch tile is an implementation artifact of `ptoas`; we drop it on the Rust side.
    RowReduce {
        kind: RowKind,
        dst: Ssa,
        src: Ssa,
    },

    /// Broadcast-compute: `pto.trowexpandsub ins(%src, %red) outs(%dst)`
    /// means `dst[r,c] = src[r,c] - red[r,0]`. Same shape for Div.
    RowExpandSub {
        dst: Ssa,
        src: Ssa,
        red: Ssa,
    },
    RowExpandDiv {
        dst: Ssa,
        src: Ssa,
        red: Ssa,
    },
    /// `pto.trowexpandmul ins(%src, %red) outs(%dst)` -- the Mul arm of the
    /// same family. 16 corpus instances, in the u8/u32 quantized kernels,
    /// where the comment on the line calls it "scale each block's row".
    ///
    /// Not part of the rms-norm chain, despite being the op an rms-norm
    /// fusion rule would reach for: `mlir_to_pto` decomposes that chain into
    /// `trowexpand` followed by a plain `tmul` instead. Modelled here because
    /// the op occurs, not because the fusion uses it.
    RowExpandMul {
        dst: Ssa,
        src: Ssa,
        red: Ssa,
    },

    /// `pto.trowexpand ins(%red) outs(%dst)` -- broadcast a per-row reduction
    /// across the row. 20 corpus instances.
    ///
    /// Shape-changing with one operand: `%red` is 1xK (the reduce output) and
    /// `%dst` is 1xN, so neither end's shape can be taken from the other --
    /// which is why it is its own variant rather than a `UnOp` kind.
    RowExpand {
        dst: Ssa,
        src: Ssa,
    },

    /// `pto.tcolexpand ins(%row) outs(%wide)` — broadcast a `1 x COLS` row down
    /// `ROWS` rows. The other direction of `RowExpand`, and its own variant for
    /// the same reason: neither end's shape follows from the other.
    ColExpand {
        dst: Ssa,
        src: Ssa,
    },

    /// `pto.tcolsum ins(%tile) outs(%row)` — sum each column of a `ROWS x COLS`
    /// tile into a `1 x COLS` row.
    ColSum {
        dst: Ssa,
        src: Ssa,
    },

    /// `pto.tmatmul ins(%l, %r) outs(%acc)` — M×K @ K×N -> M×N.
    TMatmul {
        dst: Ssa,
        lhs: Ssa,
        rhs: Ssa,
    },

    /// `pto.tmov ins(%src) outs(%dst)` -- tile-to-tile copy.
    TMov {
        dst: Ssa,
        src: Ssa,
    },

    /// Synthetic: introduced by the S2 fusion pass when the parser observes
    /// a `trowmax → trowexpandsub → texp → trowsum → trowexpanddiv` chain
    /// whose intermediates don't escape. Lowers to `tile_softmax_fNN`.
    Softmax {
        dst: Ssa,
        src: Ssa,
    },

    /// Synthetic: the fused rms-norm, `dst = src * rsqrt(mean(src²) + eps)`.
    ///
    /// Introduced by the fusion pass on the eight-op chain `mlir_to_pto`
    /// emits. Fusion is not an optimisation here -- it is the only way this
    /// computation can be lifted at all. Four of the chain's eight ops have
    /// no `tile_std` counterpart (`tadds`, `tsqrt` and `trecip` have no safe
    /// wrapper, `trowexpand` nothing at any layer), so a piecewise lowering
    /// is impossible, while the whole IS `tile_rms_norm_f32`. The chain is the
    /// decomposition `mlir_to_pto` produced FROM that intrinsic, so matching
    /// it recovers information rather than inventing it.
    ///
    /// `eps` is the literal TEXT of the `arith.constant`, for the same reason
    /// `ConstVal::Typed` keeps one: the emitter must write the constant the
    /// artifact had, not a reparse of it.
    RmsNorm {
        dst: Ssa,
        src: Ssa,
        eps: String,
    },

    /// Synthetic: the fused INT8 quantiser,
    /// `dst = round(src / scale[0, 0])`, saturated to `[-128, 127]`.
    ///
    /// Introduced by the fusion pass on the form `mlir_to_pto` emits for
    /// `__tile_quantize_f32_i8` when its `scale` operand is a TILE:
    ///
    /// ```text
    ///   %ratio = pto.tcolexpand          %scale      (rows > 1 only)
    ///   %div   = pto.trowexpanddiv       %src, %ratio
    ///   %f16   = pto.tcvt sat_mode=ON    %div
    ///   %dst   = pto.tcvt sat_mode=ON    %f16
    /// ```
    ///
    /// Fusion is not an optimisation here. The pair of saturating casts is the
    /// decomposition of one quantise step -- the part has no f32->s8 convert,
    /// so the route is f32 -> f16 -> s8, exactly as the AscendC arm does it --
    /// and no per-op tile API cast takes a saturation mode. Lowering the two
    /// casts separately, or dropping `sat_mode`, would change what an
    /// out-of-range value becomes. The whole chain is
    /// `tile_quantize_f32_i8_tile_scale`, which carries the scale as a tile so
    /// the 1x1/TILED reading the artifact actually uses is preserved.
    Quantize {
        dst: Ssa,
        src: Ssa,
        /// The per-tensor scale, as a `1 x SC` f32 tile (the padded 1x1 load).
        scale: Ssa,
    },

    /// `scf.if %cond { ... } else { ... }`.
    ///
    /// 8 instances in 7 of the 49 corpus artifacts, every one of that exact
    /// shape, and NO `scf.yield` anywhere in the corpus -- so no branch yields a
    /// value and the two regions are pure effects. Both are modelled anyway,
    /// because an `scf.if` with only its then-branch read would be a different
    /// program.
    ///
    /// Why this is here rather than in the `Unmodelled` pile: an `scf.if` opens
    /// a REGION, and a line-based reader that does not know that reads `}
    /// else {` as the end of its enclosing body. In `gmm_blk16_nb128.pto` that
    /// silently truncated an `scf.for` body, dropping the three ops after the
    /// conditional -- a `partition_view`, its `tload` and a `tmov`. Carrying
    /// `scf.if` verbatim would not have been enough; the nesting has to be
    /// understood for the lines AFTER it to be read at all. The cross-check in
    /// `view_geometry::geometry_matches_the_text` is what caught it, by
    /// comparing the AST's partition offsets against a plain text re-read.
    ///
    /// `else_body: None` is reserved for the no-else form, which the corpus does
    /// not contain; it is distinguished from `Some(vec![])` so that an empty
    /// else-branch round-trips as one.
    If {
        cond: Ssa,
        then_body: Vec<PtoOp>,
        else_body: Option<Vec<PtoOp>>,
    },

    /// `pto.ttrans ins(%src, %scratch) outs(%dst)` -- transpose, so `dst` is
    /// `src` with rows and cols swapped.
    ///
    /// `scratch` is the workspace tile PTO threads through. `tile_transpose_f32`
    /// returns a fresh tile and needs no such operand, so the Rust lift ignores
    /// it -- but it is still carried, because `emitter_pto` has to put the
    /// two-operand form back or the re-emitted line would not parse as a
    /// `ttrans` at all. Ported from `pto-rs-viewpoint-rebase`, which modelled
    /// this op first.
    Transpose {
        dst: Ssa,
        src: Ssa,
        scratch: Ssa,
    },

    /// A PTO op this crate does not model, carried through verbatim.
    ///
    /// Ported from `pto-rs-viewpoint-rebase`, which reached the same conclusion
    /// from the other end: these lines used to be dropped on the floor, which
    /// quietly defeated the round-trip verifier -- dropping an op is the largest
    /// form of information loss it exists to detect, and the one form it could
    /// not see. There the miss surfaced only by accident, when a *later* op
    /// referenced the vanished result and reverse emit failed with `no shape for
    /// SSA`; here it surfaced as a census (1895 of 6158 op lines dropped).
    ///
    /// The two branches then diverged on what to do about it. This branch made
    /// the parser REFUSE an op not named in [`crate::parser::MODELLED_OPS`], so
    /// a dropped op became a loud error; that branch kept the text so the PTO
    /// round trip is faithful by construction and refused in the Rust lift
    /// instead. Keeping the text is the better half: a refusal in the parser
    /// also refuses the 25 ops the round trip could have carried losslessly,
    /// and the round trip is what proves the reader. So the text is kept here
    /// too -- and [`MODELLED_OPS`](crate::parser::MODELLED_OPS) survives as the
    /// declared COVERAGE list rather than a gate, checked against the dispatch
    /// by `every_modelled_op_round_trips` and reported per file by
    /// [`crate::verify::op_model_coverage`].
    ///
    /// The Rust and tile-DSL lifts refuse these by name, because an op with no
    /// AST meaning has no `tile_std` lowering either.
    Unmodelled {
        /// The op's leading token, e.g. `pto.subview`, for refusal messages.
        op: String,
        /// The original line, trimmed, re-emitted byte-for-byte.
        text: String,
    },
}

impl PtoOp {
    /// The TILE values this op reads.
    ///
    /// Views, pointers, scalars and loop bounds are deliberately absent: the
    /// one question this answers is which `Tile<R, C, T>` values a lifted call
    /// would CONSUME, because `tile_std::tile::Tile` is deliberately not
    /// `Copy` ("each tile must be consumed by exactly one store operation,
    /// preventing double-DMA bugs") and a PTO tile read by two ops therefore
    /// cannot be lifted as one Rust value.
    ///
    /// Exhaustive on purpose -- no `_` arm. A new `PtoOp` variant that reads a
    /// tile has to be listed here or the crate does not build, which is the
    /// only mechanism that keeps a linearity check from going quietly blind.
    /// `emitter_tile::op_inputs` is a separate, narrower table that omits
    /// `Cvt` and `Transpose`; it can, because that emitter refuses both before
    /// reaching them.
    pub fn tile_reads(&self) -> Vec<&Ssa> {
        match self {
            PtoOp::TStore { tile, .. } => vec![tile],
            PtoOp::BinOp { a, b, .. } => vec![a, b],
            PtoOp::TMatmul { lhs, rhs, .. } => vec![lhs, rhs],
            PtoOp::BinOpScalar { a, .. } => vec![a],
            PtoOp::GetVal { src, .. } => vec![src],
            PtoOp::TSort32 { values, indices, .. } => vec![values, indices],
            PtoOp::TGather { src, .. } => vec![src],
            PtoOp::Textract { src, .. } => vec![src],
            PtoOp::UnOp { src, .. }
            | PtoOp::RowReduce { src, .. }
            | PtoOp::RowExpand { src, .. }
            | PtoOp::ColExpand { src, .. }
            | PtoOp::ColSum { src, .. }
            | PtoOp::TMov { src, .. }
            | PtoOp::Softmax { src, .. }
            | PtoOp::RmsNorm { src, .. }
            | PtoOp::Cvt { src, .. }
            | PtoOp::GatherB { src, .. } => vec![src],
            // The scale is a TILE on this path, not a scalar literal, so it is
            // a read exactly like the data operand.
            PtoOp::Quantize { src, scale, .. } => vec![src, scale],
            PtoOp::RowExpandSub { src, red, .. }
            | PtoOp::RowExpandDiv { src, red, .. }
            | PtoOp::RowExpandMul { src, red, .. } => vec![src, red],
            // `scratch` is a tile the transpose writes through, and it is an
            // operand of exactly one op, so it is not a read.
            PtoOp::Transpose { src, .. } => vec![src],
            // Not tile reads: a view or pointer chain (`Subview`,
            // `PartitionView*`, `MakeTensorView`, `TLoad`'s view), an index
            // (`IndexCast`), or no operand at all.
            PtoOp::ScalarConst { .. }
            | PtoOp::MulI { .. }
            | PtoOp::AddI { .. }
            | PtoOp::FpToSi { .. }
            | PtoOp::Tci { .. }
            | PtoOp::Barrier { .. }
            | PtoOp::Subview { .. }
            | PtoOp::BlockQuery { .. }
            | PtoOp::IndexCast { .. }
            | PtoOp::PartitionViewNd { .. }
            | PtoOp::For { .. }
            | PtoOp::MakeTensorView { .. }
            | PtoOp::PartitionView { .. }
            | PtoOp::AllocTile { .. }
            | PtoOp::TLoad { .. }
            | PtoOp::If { .. }
            | PtoOp::Unmodelled { .. } => Vec::new(),
        }
    }

    /// `tile_reads`, mutably, so a rewrite can re-point one read at another
    /// SSA. Kept beside it rather than derived from it because Rust has no way
    /// to share one match arm between `&` and `&mut`; the two must be read
    /// together, and the test `tile_reads_and_mut_agree` asserts they do.
    pub fn tile_reads_mut(&mut self) -> Vec<&mut Ssa> {
        match self {
            PtoOp::TStore { tile, .. } => vec![tile],
            PtoOp::BinOp { a, b, .. } => vec![a, b],
            PtoOp::TMatmul { lhs, rhs, .. } => vec![lhs, rhs],
            PtoOp::BinOpScalar { a, .. } => vec![a],
            PtoOp::GetVal { src, .. } => vec![src],
            PtoOp::TSort32 { values, indices, .. } => vec![values, indices],
            PtoOp::TGather { src, .. } => vec![src],
            PtoOp::Textract { src, .. } => vec![src],
            PtoOp::UnOp { src, .. }
            | PtoOp::RowReduce { src, .. }
            | PtoOp::RowExpand { src, .. }
            | PtoOp::ColExpand { src, .. }
            | PtoOp::ColSum { src, .. }
            | PtoOp::TMov { src, .. }
            | PtoOp::Softmax { src, .. }
            | PtoOp::RmsNorm { src, .. }
            | PtoOp::Cvt { src, .. }
            | PtoOp::GatherB { src, .. } => vec![src],
            // The scale is a TILE on this path, not a scalar literal, so it is
            // a read exactly like the data operand.
            PtoOp::Quantize { src, scale, .. } => vec![src, scale],
            PtoOp::RowExpandSub { src, red, .. }
            | PtoOp::RowExpandDiv { src, red, .. }
            | PtoOp::RowExpandMul { src, red, .. } => vec![src, red],
            PtoOp::Transpose { src, .. } => vec![src],
            PtoOp::ScalarConst { .. }
            | PtoOp::MulI { .. }
            | PtoOp::AddI { .. }
            | PtoOp::FpToSi { .. }
            | PtoOp::Tci { .. }
            | PtoOp::Barrier { .. }
            | PtoOp::Subview { .. }
            | PtoOp::BlockQuery { .. }
            | PtoOp::IndexCast { .. }
            | PtoOp::PartitionViewNd { .. }
            | PtoOp::For { .. }
            | PtoOp::MakeTensorView { .. }
            | PtoOp::PartitionView { .. }
            | PtoOp::AllocTile { .. }
            | PtoOp::TLoad { .. }
            | PtoOp::If { .. }
            | PtoOp::Unmodelled { .. } => Vec::new(),
        }
    }
}

/// Which core a `func.func` runs on -- the `pto.kernel_kind` attribute on its
/// signature.
///
/// Carried because it is not decoration: it is the only thing in the file that
/// says which physical unit a function executes on, and the corpus's seven
/// cube/vector pairs are two functions in ONE module distinguished by nothing
/// else. The reverse PTO emitter dropped it until 2026-11-01, so re-emitting
/// `gmm_cubeonly.pto` produced two unqualified functions -- and
/// `verify_roundtrip` compared ASTs that did not carry the attribute on either
/// pass, so all seven reported faithful. Same defect class as the
/// `__global__` qualifier that decides whether a PTO kernel can be launched at
/// all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelKind {
    /// `#pto.kernel_kind<cube>` -- the AIC matrix unit.
    Cube,
    /// `#pto.kernel_kind<vector>` -- the AIV vector unit.
    Vector,
}

impl KernelKind {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "cube" => Some(KernelKind::Cube),
            "vector" => Some(KernelKind::Vector),
            _ => None,
        }
    }

    pub fn pto_spelling(self) -> &'static str {
        match self {
            KernelKind::Cube => "cube",
            KernelKind::Vector => "vector",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PtoFunc {
    pub name: String,
    /// `(%arg601, !pto.ptr<f32>)` in declaration order.
    pub args: Vec<(Ssa, PtoType)>,
    pub ops: Vec<PtoOp>,
    /// This function's OWN constant and shape maps. SSA names are scoped to a
    /// function, and emitted PTO reuses them freely across functions in one
    /// module.
    ///
    /// Both maps were module-level until 2026-11-01, keyed on the bare name, so
    /// the last function's binding won. Corpus census that day: 10 of 49
    /// artifacts hold more than one `func.func`, **3 of them bind one SSA name
    /// to two different tile shapes (36 names in all)** and one binds two
    /// different constants. `tile_softmax_bench_kernels.acl.pto` declares
    /// `%pto2` six times -- 1x1024, 1x4096, 1x8192, 4x256, 16x256, 16x512 --
    /// so the reverse emitter wrote `16x512` into all six kernels, corrupting
    /// five of them. That artifact is one of the eight the sweep reports as
    /// round-tripping, and the round-trip could not see it: the collided map is
    /// built identically on both passes, so the two ASTs agreed.
    pub consts: BTreeMap<Ssa, i64>,
    pub shapes: BTreeMap<Ssa, TileShape>,
    /// The `pto.kernel_kind` attribute, or `None` when the signature carried
    /// no attribute dictionary. An attribute this crate does not recognise is
    /// a PARSE ERROR rather than a `None` -- see `parse_func_signature`.
    pub kind: Option<KernelKind>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PtoModule {
    pub funcs: Vec<PtoFunc>,
    /// The module-wide UNION of every function's constants, with every
    /// ambiguous name REMOVED.
    ///
    /// A name bound to two different values in two functions is not in this
    /// map at all, so a consumer that reaches for it misses and fails closed,
    /// instead of silently receiving the last function's value. Emission must
    /// use `PtoFunc::consts`; this exists for whole-module queries that have no
    /// function in hand.
    pub consts: BTreeMap<Ssa, i64>,
    /// The same union, and the same rule, for shapes. Emission must use
    /// `PtoFunc::shapes`.
    pub shapes: BTreeMap<Ssa, TileShape>,
    /// Names that bind two different values across functions and were therefore
    /// dropped from the two unions above. Empty for a single-function module.
    pub ambiguous: BTreeSet<Ssa>,
}

/// The shape the emitter needs for any SSA it references.
/// The physical memory a tile lives in — the `loc=` field of `!pto.tile_buf`.
///
/// These are not interchangeable labels. On Ascend, `left`/`right`/`acc` are
/// the cube unit's L0A/L0B/L0C buffers (the `__ca__`/`__cb__`/`__cc__`
/// qualifiers `ptoas` emits), `mat` is L1, `scaling` holds quantization
/// scales, and `vec` is the vector unit's UB.
///
/// `loc` was not in the AST until 2026-11-01 and the reverse emitter wrote a
/// literal `loc=vec` for every tile. Corpus census that day: **233 tiles are
/// not `vec`** — `mat` ×77, `acc` ×50, `right` ×41, `left` ×38, `scaling` ×27
/// — so `matmul_32x64x32.pto` round-tripped with all three of its operands
/// relocated from the cube's L0A/L0B/L0C into the vector unit's UB, and the
/// verifier called that equal because the drop happened on both passes.
///
/// Whether the Rust lift may treat a cube-resident tile as a UB tile is NOT
/// settled here; this type only stops the information being thrown away.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TileLoc {
    Vec,
    Mat,
    Left,
    Right,
    Acc,
    Scaling,
}

impl TileLoc {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s.trim() {
            "vec" => TileLoc::Vec,
            "mat" => TileLoc::Mat,
            "left" => TileLoc::Left,
            "right" => TileLoc::Right,
            "acc" => TileLoc::Acc,
            "scaling" => TileLoc::Scaling,
            _ => return None,
        })
    }

    pub fn as_str(self) -> &'static str {
        match self {
            TileLoc::Vec => "vec",
            TileLoc::Mat => "mat",
            TileLoc::Left => "left",
            TileLoc::Right => "right",
            TileLoc::Acc => "acc",
            TileLoc::Scaling => "scaling",
        }
    }

    /// True for the four locations that are NOT the vector unit's UB.
    pub fn is_cube_side(self) -> bool {
        !matches!(self, TileLoc::Vec)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileShape {
    pub rows: u32,
    pub cols: u32,
    /// The logical extent (`v_row`/`v_col`); equals `rows`/`cols` for an
    /// unpadded tile and for every partition view. A lifted Rust value's type
    /// is this, not `rows`/`cols`.
    pub v_rows: u32,
    pub v_cols: u32,
    pub dtype: DType,
    /// `Some` for a `!pto.tile_buf`, `None` for a partition view — this map
    /// holds both, and a view has no `loc`.
    pub loc: Option<TileLoc>,
}
