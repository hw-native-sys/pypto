// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! A TYPED PTO emission core.
//!
//! The emitter has always built `Vec<String>`, so the type system stopped at the
//! boundary and every semantic invariant had to be re-checked by hand or by a gate
//! that read the text back afterwards. That is how a chunk loop with the wrong
//! bounds shipped: it was a string, it parsed, and nothing asked whether it covered
//! its width.
//!
//! This crate is the start of the fix: values that CANNOT be constructed in a
//! violating state, and a renderer that turns them into the same text the emitter
//! already writes.
//!
//! Stage 1 covers the chunk loop, which is where the defect was. The renderer is
//! deliberately byte-for-byte identical to the string emission so a converted site
//! can be proved equivalent before it is trusted.
//!
//! # How to convert a site
//!
//! The recipe, written down because each of its three steps was learned by getting it
//! wrong at least once:
//!
//! 1. **Clone every `String` operand.** They are moved into `Value::new`, and the
//!    surrounding emission usually reads them again. This has cost a build three
//!    times.
//! 2. **Pass the types the site actually has, not a recomputed common one.** Passing
//!    one type three times satisfies every constructor trivially and enforces
//!    nothing; the sites worth converting are the ones whose types are derived
//!    independently.
//! 3. **Make the control SEMANTIC.** Corrupt `v_col`, `rows` or `dtype` -- not the
//!    text of a type. `compatible` compares logical shape and dtype, so appending a
//!    character changes nothing and the "control" passes without testing the check.
//!
//! Then: build, run the corpus byte-identity gates, and confirm the control fires. A
//! conversion that cannot be proved byte-identical in its own commit should not be
//! made, which is also why this is done one site at a time rather than as a rewrite.
//!
//! # What must NOT be converted
//!
//! `pto.tmov` is the op that deliberately moves a tile BETWEEN memory locations and
//! layouts -- cube L1 into L0A, an accumulator into a vector tile. Requiring
//! `compatible` there is exactly backwards, and a batch of eight such patches had to
//! be thrown away for that reason. The same test applies to `tcvt` (changes dtype by
//! design) and `trowexpand`/`tcolexpand` (change shape by design): an op whose purpose
//! is to bridge two types cannot be checked by a constructor that requires one.
//!
//! # What a constructor must state
//!
//! The invariant the string emitter left implicit. Three have been corrected by the
//! corpus after looking obviously right: a row reduction keeps the LOGICAL row count
//! (`v_row`, not `rows`, because those tiles are physically padded), a store's view
//! need not equal its tile (one kernel pads deliberately, one truncated by bug -- so
//! `Store` is NOT adopted), and a col-major reduce tile legitimately bridges into a
//! row-major vector tile of the same logical shape.

/// A 1-D extent that a loop must tile exactly.
///
/// The type exists so that "covers the width with no tail" is a property of the
/// CONSTRUCTION and not of a later check: there is no way to make a `Covering` that
/// does not multiply out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Covering {
    width: u32,
    chunk: u32,
}

impl Covering {
    /// `Some` only when `chunk` divides `width` and is non-zero.
    ///
    /// A tail would need an epilogue that the emission does not have, so a
    /// non-dividing chunk is not a warning here -- it is unrepresentable.
    pub fn new(width: u32, chunk: u32) -> Option<Self> {
        if chunk == 0 || width % chunk != 0 {
            return None;
        }
        Some(Self { width, chunk })
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn chunk(&self) -> u32 {
        self.chunk
    }

    /// Iterations, exact by construction.
    pub fn iters(&self) -> u32 {
        self.width / self.chunk
    }

    /// The offset of iteration `i`, in elements.
    pub fn offset(&self, i: u32) -> u32 {
        i * self.chunk
    }
}

/// One emitted PTO operation: a name, its operand spellings, and nothing else.
///
/// The point is not the fields; it is that a rendered op goes through
/// [`Op::render`], so the shape of the text is written once. The typed tile values
/// this will eventually carry are the next stage; what matters now is that the
/// emission path has a place to put them.
#[derive(Clone, Debug)]
pub struct Op {
    name: String,
    args: String,
}

impl Op {
    pub fn new(name: impl Into<String>, args: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            args: args.into(),
        }
    }

    pub fn render(&self, ssa: &str, indent: usize) -> String {
        format!("{}{} {} = {}{}", " ".repeat(indent), ssa, "=", "", self.text())
    }

    fn text(&self) -> String {
        if self.args.is_empty() {
            self.name.clone()
        } else {
            format!("{} {}", self.name, self.args)
        }
    }
}

/// A chunk loop over a [`Covering`], rendered as `scf.for` with a one-step count
/// bound and an offset of `index * chunk`.
///
/// The arithmetic lives here and nowhere else: a caller holds a `Covering`, which
/// cannot exist unless the chunks close the width, and this renders the only loop
/// shape that walks it.
#[derive(Clone, Debug)]
pub struct ChunkLoop {
    covering: Covering,
}

impl ChunkLoop {
    pub fn new(covering: Covering) -> Self {
        Self { covering }
    }

    /// The prologue: the four constants and the loop header, in the exact form the
    /// string emitter writes, so a converted site is byte-identical.
    pub fn prologue(&self) -> Vec<String> {
        let c = self.covering;
        vec![
            format!("    %n_iters = arith.constant {} : index", c.iters()),
            format!("    %n_chunk = arith.constant {} : index", c.chunk()),
            "    %n_zero = arith.constant 0 : index".to_string(),
            "    %n_one = arith.constant 1 : index".to_string(),
            "    scf.for %n_i = %n_zero to %n_iters step %n_one {".to_string(),
            "      %n_off = arith.muli %n_i, %n_chunk : index".to_string(),
        ]
    }

    pub fn epilogue(&self) -> String {
        "    }".to_string()
    }

    /// Wrap a body: prologue, indented body, epilogue.
    pub fn wrap(&self, body: &[String]) -> Vec<String> {
        let mut out = self.prologue();
        out.extend(body.iter().map(|l| format!("      {l}")));
        out.push(self.epilogue());
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_covering_cannot_be_made_to_leave_a_tail() {
        assert!(Covering::new(12288, 4096).is_some());
        assert!(Covering::new(12288, 5000).is_none(), "a tail has no epilogue");
        assert!(Covering::new(12288, 0).is_none());
    }

    #[test]
    fn the_iteration_count_and_offsets_close_the_width() {
        let c = Covering::new(12288, 4096).unwrap();
        assert_eq!(c.iters(), 3);
        let offsets: Vec<u32> = (0..c.iters()).map(|i| c.offset(i)).collect();
        assert_eq!(offsets, vec![0, 4096, 8192]);
        // every chunk lands inside the width, and the last one ends exactly at it
        for (i, off) in offsets.iter().enumerate() {
            assert!(off + c.chunk() <= c.width());
            if i + 1 == offsets.len() {
                assert_eq!(off + c.chunk(), c.width());
            }
        }
    }

    #[test]
    fn the_rendered_loop_steps_by_one_over_the_count() {
        let rendered = ChunkLoop::new(Covering::new(12288, 4096).unwrap()).prologue().join("\n");
        // the bound is the COUNT and the step is ONE. Stepping by the chunk size over
        // the count runs exactly one iteration -- the defect this type exists for.
        assert!(rendered.contains("%n_iters = arith.constant 3 : index"), "{rendered}");
        assert!(rendered.contains("to %n_iters step %n_one"), "{rendered}");
        assert!(rendered.contains("%n_off = arith.muli %n_i, %n_chunk"), "{rendered}");
        assert!(!rendered.contains("step %n_chunk"), "{rendered}");
    }

    #[test]
    fn wrapping_indents_every_body_line_and_closes_the_block() {
        let lines = ChunkLoop::new(Covering::new(8192, 4096).unwrap())
            .wrap(&["%x = pto.tload".to_string()]);
        assert!(lines.iter().any(|l| l == "      %x = pto.tload"));
        assert_eq!(lines.last().unwrap(), "    }");
        assert_eq!(lines.iter().filter(|l| l.contains("scf.for")).count(), 1);
    }
}

/// A PTO operand type, as far as emission needs to distinguish them.
///
/// `Tile` carries the extents because that is what the emitter gets wrong: a tile
/// allocated at one width and rendered in an op at another type-checks in text and
/// fails in the IR. `Raw` is the escape hatch for shapes not modelled yet, and it is
/// deliberately explicit so the un-modelled surface is countable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ty {
    Tile { rows: u32, cols: u32, dtype: String },
    Raw(String),
}

impl Ty {
    pub fn tile(rows: u32, cols: u32, dtype: &str) -> Self {
        Ty::Tile { rows, cols, dtype: dtype.to_string() }
    }

    /// The extents, when the type has them.
    /// The extents, when they can be recovered.
    ///
    /// `Raw` PARSES them out of the type text rather than refusing, because the
    /// emitter still spells several tile flavours the typed model does not cover --
    /// a row-reduce tile is `col_major` with `v_col=1`, and its renderer is not the
    /// same one as a vector tile's. Reading `rows=` / `cols=` out of the string keeps
    /// the shape check working on real emissions while the migration is in progress;
    /// returning `None` for every raw type would make every converted site fail to
    /// build, which is a check that cannot be adopted.
    pub fn shape(&self) -> Option<(u32, u32)> {
        match self {
            Ty::Tile { rows, cols, .. } => Some((*rows, *cols)),
            Ty::Raw(s) => {
                let field = |name: &str| -> Option<u32> {
                    let at = s.find(name)? + name.len();
                    let digits: String = s[at..].chars().take_while(|c| c.is_ascii_digit()).collect();
                    digits.parse().ok()
                };
                Some((field("rows=")?, field("cols=")?))
            }
        }
    }

    /// The LOGICAL extents: `v_row`/`v_col` when present, else `rows`/`cols`.
    ///
    /// A row-reduce tile is declared `rows=8, v_row=1` -- physically eight lanes,
    /// logically one row -- so comparing physical rows against a 1-row source says
    /// the reduction loses rows when it does not. The padded form is the whole point
    /// of those types.
    pub fn logical_shape(&self) -> Option<(u32, u32)> {
        match self {
            Ty::Tile { rows, cols, .. } => Some((*rows, *cols)),
            Ty::Raw(s) => {
                let field = |name: &str| -> Option<u32> {
                    let at = s.find(name)? + name.len();
                    let digits: String = s[at..].chars().take_while(|c| c.is_ascii_digit()).collect();
                    digits.parse().ok()
                };
                let rows = field("v_row=").or_else(|| field("rows="))?;
                let cols = field("v_col=").or_else(|| field("cols="))?;
                Some((rows, cols))
            }
        }
    }

    /// The element dtype, recovered from the text for a raw type.
    ///
    /// This used to return `None` for every `Raw`, which made `compatible`'s dtype
    /// branch fall through to `true` -- so the dtype half of the check was INERT on
    /// every converted site, since all of them pass `Ty::Raw`. A control that changed
    /// `dtype=f32` to `f16` still emitted, which is how it was found.
    pub fn dtype(&self) -> Option<&str> {
        match self {
            Ty::Tile { dtype, .. } => Some(dtype),
            Ty::Raw(s) => {
                let at = s.find("dtype=")? + "dtype=".len();
                let end = s[at..].find(|c: char| c == ',' || c == '>' || c == ' ')? + at;
                Some(&s[at..end])
            }
        }
    }

    pub fn render(&self) -> String {
        match self {
            Ty::Raw(s) => s.clone(),
            Ty::Tile { rows, cols, dtype } => format!(
                "!pto.tile_buf<loc=vec, dtype={dtype}, rows={rows}, cols={cols}, v_row={rows}, \
                 v_col={cols}, blayout=row_major, slayout=none_box, fractal=512, pad=0>"
            ),
        }
    }
}

/// An SSA name with its type.
#[derive(Clone, Debug)]
pub struct Value {
    pub ssa: String,
    pub ty: Ty,
}

impl Value {
    pub fn new(ssa: impl Into<String>, ty: Ty) -> Self {
        Self { ssa: ssa.into(), ty }
    }
}

/// A binary elementwise PTO op (`pto.tadd`, `pto.tmul`, ...).
///
/// THE POINT: the constructor refuses operands whose types differ, so an op that
/// mixes a 4096-wide tile with a 12288-wide one cannot be built, let alone
/// rendered. The text emitter had no such rule -- it formatted whatever strings it
/// was handed, and the IR caught the mismatch only after emission, if at all.
#[derive(Debug)]
pub struct Binary {
    mnemonic: String,
    lhs: Value,
    rhs: Value,
    out: Value,
}

impl Binary {
    pub fn new(mnemonic: &str, lhs: Value, rhs: Value, out: Value) -> Result<Self, String> {
        if !compatible(&lhs.ty, &rhs.ty) {
            return Err(format!(
                "`{mnemonic}` operands differ: {} is {:?}, {} is {:?}",
                lhs.ssa, lhs.ty, rhs.ssa, rhs.ty
            ));
        }
        if !compatible(&out.ty, &lhs.ty) {
            return Err(format!(
                "`{mnemonic}` writes {:?} into {:?}",
                lhs.ty, out.ty
            ));
        }
        Ok(Self { mnemonic: mnemonic.to_string(), lhs, rhs, out })
    }

    /// Renders each operand's OWN type spelling.
    ///
    /// It used to print the left operand's type in all three slots, which was fine
    /// while the constructors required textual equality and wrong the moment the rule
    /// became `compatible`: two spellings of one logical shape (a col-major reduce
    /// tile and a row-major vector tile) are legal operands, and collapsing them
    /// would silently rewrite the emitted type. The renderer is faithful; the CHECK
    /// is what refuses a mismatch.
    pub fn render(&self) -> String {
        format!(
            "{} ins({}, {} : {}, {}) outs({} : {})",
            self.mnemonic,
            self.lhs.ssa,
            self.rhs.ssa,
            self.lhs.ty.render(),
            self.rhs.ty.render(),
            self.out.ssa,
            self.out.ty.render()
        )
    }
}

#[cfg(test)]
mod binary_tests {
    use super::*;

    #[test]
    fn matching_operands_render_as_the_emitter_writes_them() {
        let ty = Ty::tile(1, 4096, "f32");
        let op = Binary::new(
            "pto.tmul",
            Value::new("%a", ty.clone()),
            Value::new("%b", ty.clone()),
            Value::new("%c", ty),
        )
        .unwrap();
        let text = op.render();
        assert!(text.starts_with("pto.tmul ins(%a, %b : !pto.tile_buf<"));
        assert!(text.contains("outs(%c : !pto.tile_buf<"));
        assert_eq!(text.matches("cols=4096").count(), 3, "three operand types");
    }

    #[test]
    fn a_width_mismatch_cannot_be_constructed() {
        // This is the class that made me abandon the first chunking attempt: a tile
        // allocated at the chunk width and rendered at the original.
        let err = Binary::new(
            "pto.tmul",
            Value::new("%a", Ty::tile(1, 4096, "f32")),
            Value::new("%b", Ty::tile(1, 12288, "f32")),
            Value::new("%c", Ty::tile(1, 4096, "f32")),
        )
        .expect_err("mismatched widths must not build");
        assert!(err.contains("differ"), "{err}");
    }

    #[test]
    fn a_dtype_mismatch_cannot_be_constructed() {
        let err = Binary::new(
            "pto.tadd",
            Value::new("%a", Ty::tile(1, 64, "f32")),
            Value::new("%b", Ty::tile(1, 64, "f16")),
            Value::new("%c", Ty::tile(1, 64, "f32")),
        )
        .expect_err("mixed dtypes must not build");
        assert!(err.contains("differ"), "{err}");
    }

    #[test]
    fn an_output_of_the_wrong_shape_cannot_be_constructed() {
        let err = Binary::new(
            "pto.tadd",
            Value::new("%a", Ty::tile(1, 64, "f32")),
            Value::new("%b", Ty::tile(1, 64, "f32")),
            Value::new("%c", Ty::tile(1, 32, "f32")),
        )
        .expect_err("the output shape is checked too");
        assert!(err.contains("writes"), "{err}");
    }
}

/// A scalar-operand PTO op (`pto.tmuls`, `pto.tadds`, ...).
///
/// Same rule as [`Binary`]: the source and destination must agree. The scalar is a
/// literal and carries no type of its own beyond `f32`, which is why it is a `String`
/// here rather than a [`Value`].
#[derive(Debug)]
pub struct Scalar {
    mnemonic: String,
    src: Value,
    scalar: String,
    /// The scalar operand's dtype. NOT always `f32`: the rope arm passes an `f16`
    /// constant and the integer quantize path an `i16` one, so hardcoding f32 here
    /// would have rewritten those ops on conversion.
    scalar_dtype: String,
    out: Value,
}

impl Scalar {
    pub fn new(
        mnemonic: &str,
        src: Value,
        scalar: &str,
        scalar_dtype: &str,
        out: Value,
    ) -> Result<Self, String> {
        if !compatible(&src.ty, &out.ty) {
            return Err(format!(
                "`{mnemonic}` reads {:?} and writes {:?}",
                src.ty, out.ty
            ));
        }
        Ok(Self {
            mnemonic: mnemonic.to_string(),
            src,
            scalar: scalar.to_string(),
            scalar_dtype: scalar_dtype.to_string(),
            out,
        })
    }

    pub fn render(&self) -> String {
        format!(
            "{} ins({}, {} : {}, {}) outs({} : {})",
            self.mnemonic,
            self.src.ssa,
            self.scalar,
            self.src.ty.render(),
            self.scalar_dtype,
            self.out.ssa,
            self.out.ty.render()
        )
    }
}

/// A unary PTO op (`pto.texp`, `pto.tabs`, ...), where every operand shares a type.
#[derive(Debug)]
pub struct Unary {
    mnemonic: String,
    src: Value,
    out: Value,
}

impl Unary {
    pub fn new(mnemonic: &str, src: Value, out: Value) -> Result<Self, String> {
        if !compatible(&src.ty, &out.ty) {
            return Err(format!(
                "`{mnemonic}` reads {:?} and writes {:?}",
                src.ty, out.ty
            ));
        }
        Ok(Self { mnemonic: mnemonic.to_string(), src, out })
    }

    pub fn render(&self) -> String {
        format!(
            "{} ins({} : {}) outs({} : {})",
            self.mnemonic,
            self.src.ssa,
            self.src.ty.render(),
            self.out.ssa,
            self.out.ty.render()
        )
    }
}

#[cfg(test)]
mod scalar_unary_tests {
    use super::*;

    #[test]
    fn a_scalar_op_renders_the_literal_as_f32() {
        let op = Scalar::new(
            "pto.tmuls",
            Value::new("%g", Ty::Raw("T".into())),
            "%c",
            "f32",
            Value::new("%n", Ty::Raw("T".into())),
        )
        .unwrap();
        assert_eq!(op.render(), "pto.tmuls ins(%g, %c : T, f32) outs(%n : T)");
    }

    #[test]
    fn a_scalar_op_across_unequal_types_is_refused() {
        let err = Scalar::new(
            "pto.tadds",
            Value::new("%a", Ty::tile(1, 64, "f32")),
            "%c",
            "f32",
            Value::new("%b", Ty::tile(1, 32, "f32")),
        )
        .expect_err("the shapes differ");
        assert!(err.contains("reads"), "{err}");
    }

    #[test]
    fn a_unary_op_across_unequal_types_is_refused() {
        let err = Unary::new(
            "pto.texp",
            Value::new("%a", Ty::tile(1, 64, "f32")),
            Value::new("%b", Ty::tile(1, 64, "f16")),
        )
        .expect_err("the dtypes differ");
        assert!(err.contains("reads"), "{err}");
    }

    #[test]
    fn a_unary_op_renders_one_type_twice() {
        let op = Unary::new(
            "pto.texp",
            Value::new("%a", Ty::Raw("T".into())),
            Value::new("%b", Ty::Raw("T".into())),
        )
        .unwrap();
        assert_eq!(op.render(), "pto.texp ins(%a : T) outs(%b : T)");
    }
}

/// A row reduction (`pto.trowsum`, `pto.trowmax`, ...): `rows x cols` into one value
/// per row.
///
/// The rule is the SHAPE RELATION, not equality -- a reduction's output is
/// diamond-shaped relative to its input, and that is exactly the relation the text
/// emitter never stated. `dst` must have the same row count and a single column,
/// which is what the row-reduce tile types encode.
#[derive(Debug)]
pub struct Reduce {
    mnemonic: String,
    src: Value,
    tmp: Value,
    dst: Value,
}

impl Reduce {
    pub fn new(mnemonic: &str, src: Value, tmp: Value, dst: Value) -> Result<Self, String> {
        let (Some((srows, scols)), Some((drows, dcols))) =
            (src.ty.logical_shape(), dst.ty.logical_shape())
        else {
            return Err(format!(
                "`{mnemonic}` needs extents on both sides; got {:?} and {:?}",
                src.ty, dst.ty
            ));
        };
        if srows != drows {
            return Err(format!(
                "`{mnemonic}` reduces {srows} rows into {drows}: a row reduction keeps the row count"
            ));
        }
        if dcols != 1 {
            return Err(format!(
                "`{mnemonic}` writes {dcols} columns; a row reduction produces one value per row"
            ));
        }
        if scols < 1 {
            return Err(format!("`{mnemonic}` reduces a zero-width row"));
        }
        let _ = tmp;
        Ok(Self {
            mnemonic: mnemonic.to_string(),
            src,
            tmp,
            dst,
        })
    }

    pub fn render(&self) -> String {
        format!(
            "{} ins({}, {} : {}, {}) outs({} : {})",
            self.mnemonic,
            self.src.ssa,
            self.tmp.ssa,
            self.src.ty.render(),
            // THE SCRATCH HAS ITS OWN TYPE. It used to print the source's twice, and a
            // live site (rms_norm's trowsum) allocates scratch at HALF the source
            // width, so the conversion would have emitted a type the kernel does not
            // allocate -- wrong, and byte-different from what the corpus pins.
            self.tmp.ty.render(),
            self.dst.ssa,
            self.dst.ty.render()
        )
    }
}

#[cfg(test)]
mod reduce_tests {
    use super::*;

    fn v(name: &str, rows: u32, cols: u32) -> Value {
        Value::new(name, Ty::tile(rows, cols, "f32"))
    }

    #[test]
    fn a_row_reduction_keeps_the_row_count_and_yields_one_column() {
        let op = Reduce::new("pto.trowsum", v("%a", 1, 7168), v("%t", 1, 7168), v("%d", 1, 1)).unwrap();
        assert!(op.render().contains("outs(%d : "));
    }

    #[test]
    fn an_output_with_more_than_one_column_is_refused() {
        let err = Reduce::new("pto.trowsum", v("%a", 1, 64), v("%t", 1, 64), v("%d", 1, 8))
            .expect_err("a row reduction cannot produce eight columns");
        assert!(err.contains("one value per row"), "{err}");
    }

    #[test]
    fn losing_rows_is_refused() {
        let err = Reduce::new("pto.trowsum", v("%a", 8, 64), v("%t", 8, 64), v("%d", 1, 1))
            .expect_err("rows cannot disappear");
        assert!(err.contains("keeps the row count"), "{err}");
    }
}

/// A tile store: `pto.tstore ins(tile : T) outs(view : PV)`.
///
/// THE RELATION: the partition view's extents must equal the tile's. That is the
/// invariant the chunk port re-derived by rewriting `cols=` in the emitted text, and
/// the one that let a 4480-wide tile be stored through an 8960-wide view. Stating it
/// at construction makes the text rewriting unnecessary rather than merely checked.
#[derive(Debug)]
pub struct Store {
    src: Value,
    dst: Value,
}

/// The extents a `!pto.partition_tensor_view<R x C x dtype>` spells.
fn partition_extents(ty: &Ty) -> Option<(u32, u32)> {
    match ty {
        Ty::Tile { rows, cols, .. } => Some((*rows, *cols)),
        Ty::Raw(s) => {
            let at = s.find("partition_tensor_view<")? + "partition_tensor_view<".len();
            let body: String = s[at..].chars().take_while(|c| *c != '>').collect();
            let mut parts = body.split('x');
            let rows = parts.next()?.parse().ok()?;
            let cols = parts.next()?.parse().ok()?;
            Some((rows, cols))
        }
    }
}

impl Store {
    /// A store into a view that is EXACTLY the tile, or the tile PADDED UP to a block.
    ///
    /// The two cases that made `==` unusable are distinguishable by that second clause.
    /// The q4k unpack stores a 1x12 tile through a 1x16 view because 16 is the row
    /// padded to the block -- legitimate, and the four extra lanes are never written.
    /// The chunk defect stored a 1x4480 tile through a 1x8960 view, and 8960 is not
    /// 4480 rounded to any block. So the rule is: equal, or the tile rounded up.
    pub fn new(src: Value, dst: Value) -> Result<Self, String> {
        let Some((trows, tcols)) = src.ty.logical_shape() else {
            return Err(format!("`pto.tstore` needs a shaped source, got {:?}", src.ty));
        };
        let Some((vrows, vcols)) = partition_extents(&dst.ty) else {
            return Err(format!(
                "`pto.tstore` needs a partition view with extents, got {:?}",
                dst.ty
            ));
        };
        let padded = |n: u32| n.div_ceil(16) * 16;
        let ok = (trows, tcols) == (vrows, vcols)
            || (trows == vrows && vcols == padded(tcols))
            || (tcols == vcols && vrows == padded(trows));
        if !ok {
            return Err(format!(
                "`pto.tstore` stores a {trows}x{tcols} tile through a {vrows}x{vcols} view, \
                 which is neither the tile nor the tile padded to a block"
            ));
        }
        Ok(Self { src, dst })
    }

    pub fn render(&self) -> String {
        format!(
            "pto.tstore ins({} : {}) outs({} : {})",
            self.src.ssa,
            self.src.ty.render(),
            self.dst.ssa,
            self.dst.ty.render()
        )
    }
}

#[cfg(test)]
mod store_tests {
    use super::*;

    fn pv(rows: u32, cols: u32) -> Ty {
        Ty::Raw(format!("!pto.partition_tensor_view<{rows}x{cols}xf32>"))
    }

    #[test]
    fn a_store_whose_view_matches_the_tile_is_built() {
        let op = Store::new(Value::new("%t", Ty::tile(1, 4096, "f32")), Value::new("%p", pv(1, 4096)))
            .unwrap();
        assert_eq!(
            op.render(),
            format!(
                "pto.tstore ins(%t : {}) outs(%p : {})",
                Ty::tile(1, 4096, "f32").render(),
                pv(1, 4096).render()
            )
        );
    }

    #[test]
    fn a_tile_stored_through_a_wider_view_is_refused() {
        // The shape the chunk port produced before the fix: a chunk-sized tile stored
        // through the call's un-chunked extent.
        let err = Store::new(Value::new("%t", Ty::tile(1, 4480, "f32")), Value::new("%p", pv(1, 8960)))
            .expect_err("8960 is not 4480 padded to a block");
        assert!(err.contains("1x4480"), "{err}");
        assert!(err.contains("8960"), "{err}");
    }

    #[test]
    fn a_tile_stored_through_its_block_padded_view_is_accepted() {
        // The q4k unpack: a 1x12 tile into a 1x16 view, where 16 is 12 rounded to 16.
        let op = Store::new(Value::new("%t", Ty::tile(1, 12, "f32")), Value::new("%p", pv(1, 16)))
            .expect("a block-padded view is legitimate");
        let _ = op.render();
    }
}

/// Whether two types are compatible for an op that moves a tile unchanged.
///
/// LOGICAL SHAPE AND DTYPE, not type text. The text carries layout and padding --
/// `col_major` versus `row_major`, `rows=8, v_row=1` versus `rows=1` -- and the
/// emitter has five sites that deliberately bridge from a col-major reduce tile to a
/// row-major vector tile because ptoas rejects the reverse. Comparing text refuses
/// those by construction; comparing what the op actually requires accepts them.
///
/// What it still refuses is the defect class: 4480 against 8960, or f32 against f16.
pub fn compatible(a: &Ty, b: &Ty) -> bool {
    let shapes = match (a.logical_shape(), b.logical_shape()) {
        (Some(x), Some(y)) => x == y,
        // An unparseable type cannot be checked; do not pretend it passed.
        _ => a == b,
    };
    let dtypes = match (a.dtype(), b.dtype()) {
        (Some(x), Some(y)) => x == y,
        _ => true,
    };
    shapes && dtypes
}

#[cfg(test)]
mod compatible_tests {
    use super::*;

    #[test]
    fn a_col_major_reduce_tile_matches_a_row_major_vector_tile() {
        // The identity bridge the emitter needs: logically 1x1 either way.
        let cm = Ty::Raw("!pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=1, v_row=1, v_col=1, blayout=col_major>".into());
        let rr = Ty::Raw("!pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=1, blayout=row_major>".into());
        assert!(compatible(&cm, &rr), "same logical 1x1, different spelling");
    }

    #[test]
    fn a_padded_tile_matches_its_unpadded_twin() {
        let padded = Ty::Raw("!pto.tile_buf<dtype=f32, rows=1, cols=8, v_row=1, v_col=6>".into());
        let plain = Ty::Raw("!pto.tile_buf<dtype=f32, rows=1, cols=6, v_row=1, v_col=6>".into());
        assert!(compatible(&padded, &plain));
    }

    #[test]
    fn the_width_mismatch_is_still_refused() {
        assert!(!compatible(&Ty::tile(1, 4480, "f32"), &Ty::tile(1, 8960, "f32")));
    }

    #[test]
    fn a_dtype_mismatch_is_still_refused() {
        assert!(!compatible(&Ty::tile(1, 64, "f32"), &Ty::tile(1, 64, "f16")));
    }
}
