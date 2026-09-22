# FP4 in PyPTO

Canonical status page for MXFP4 / packed FP4. Pass algorithm details live in
[PackFp4](passes/07-pack_fp4.md); this page covers roles, pipeline, unit
convention, cast policy, and the three-layer support matrix.

## Roles

| Type | Meaning |
| ---- | ------- |
| `pl.FP4` | Frontend **logical** E2M1 nibble (`GetBit()==4`). Shapes / `valid_shape` count nibbles. |
| `pl.FP4E2M1X2` | **Packed** carrier (two nibbles per element, `GetBit()==8`). Physical shapes after `PackFp4`, or hand-written. Even last dims are the author's responsibility on the logical path; DN / NZ / col_major / cube are still rejected by PackFp4. Hand-written `FP4E2M1X2` is **ND row-major only**. |

`tensor.full` / `create` fill values: only **zero** is defined after packing; non-zero
fill semantics (duplicate nibble vs low-only) are undefined in this release.

PyPTO also rejects FP4-family `reshape`, scalar `read`/`write`, `tensor.dim`,
unwhitelisted coordinate ops, and cube / DN / NZ / col_major packing — see pass
docs and error strings (these are PyPTO-only limits, not a three-layer matrix
row).

`FP4E2M1X2` layout conversion is hard-rejected: non-ND annotations, implicit
column-vector `[M, 1]` DN, and explicit `tensor.view` layout flips.

## Pipeline

1. **PackFp4** (static-only): logical `pl.FP4` → `FP4E2M1X2`; last-axis covering
   sizes / offsets / ND leading strides must be even `ConstInt` and are halved.
2. **Local ExpandPackedFp4\*** (single-device codegen): expand GM
   `make_tensor_view` / partition last-axis geometry back to nibble units for
   pto-isa `GetByteSize`; EmitC expands Tile cols for TCVT. **Tile_buf stays in
   carrier units** (no expand). MX layouts are not expanded.
3. **Cast** (`LegalizeTileCast`): see [Cast policy](#cast-policy-ascend950).

## Unit convention

| Layer | Unit |
| ----- | ---- |
| Frontend `pl.FP4` shape / `valid_shape` | nibble |
| IR after PackFp4 / `tile_buf` / orch create | carrier |
| Hand-written `pl.FP4E2M1X2` IR / `tile_buf` / Torch ABI | carrier |
| `make_tensor_view` / `partition_view` (after Expand) | nibble (for pto-isa `GetByteSize`) |
| runtime Tensor / Torch `float4_e2m1fn_x2` | carrier element |

Write multi-row ND packed tensors with **carrier** last dims and leading strides
(for example `pl.Tensor[[2, 256], pl.FP4E2M1X2]` for 512 logical nibbles per row).
Codegen expands GM views to nibble units so multi-row pitch matches Tile /
pto-isa. Using logical widths on `FP4E2M1X2` can mis-size GM row strides — see
issue [#2754](https://github.com/hw-native-sys/pypto/issues/2754).

## Cast policy (Ascend950)

`LegalizeTileCast` treats the FP4 family as follows:

| Cast | Behavior |
| ---- | -------- |
| `FP4` / `FP4E2M1X2` ↔ `BF16` | Native TCVT (DSv4.1 c1a). Silent. Packed ↔ wider types adjust the last axis **2:1** (static even last dim required when packing). Result strides are rebuilt contiguous from the destination shape. |
| `FP4` / `FP4E2M1X2` → `FP8E4M3FN` / `FP8E5M2` | Legalized as FP4→BF16→FP32→FP8 with a **Warning** (prefer LUT / host precast). |
| `FP4` ↔ `FP4E2M1X2` | **Rejected** (`RejectFp4FamilyInternalCast`; geometry disagree). Cast via a wider type or rewrite shapes. |
| Wider → `FP4E2M1X2` with dynamic last dim | **Rejected** (static positive even only). |

### Hand-written `FP4E2M1X2` → BF16

Carrier last dim `32` expands to BF16 last dim `64`:

```python
import pypto.language as pl


@pl.function(type=pl.FunctionType.InCore)
def fp4x2_to_bf16(
    x: pl.Tensor[[16, 32], pl.FP4E2M1X2],
    out: pl.Out[pl.Tensor[[16, 64], pl.BF16]],
) -> pl.Tensor[[16, 64], pl.BF16]:
    t = pl.load(x, [0, 0], [16, 32])
    c = pl.cast(t, pl.BF16)
    return pl.store(c, [0, 0], out)
```

### Hand-written `FP4E2M1X2` → FP8

Same 2:1 last-axis expand; expect a LegalizeTileCast Warning:

```python
import pypto.language as pl


@pl.function(type=pl.FunctionType.InCore)
def fp4x2_to_fp8(
    x: pl.Tensor[[16, 32], pl.FP4E2M1X2],
    out: pl.Out[pl.Tensor[[16, 64], pl.FP8E4M3FN]],
) -> pl.Tensor[[16, 64], pl.FP8E4M3FN]:
    t = pl.load(x, [0, 0], [16, 32])
    c = pl.cast(t, pl.FP8E4M3FN)
    return pl.store(c, [0, 0], out)
```

## Support matrix (this release)

Legend: ✅ supported · ⚠️ partial / Warning · ❌ unsupported · ⏳ lower layer ready, PyPTO TODO

Rows where both PTOAS and pto-isa would be N/A are omitted (PyPTO-only limits live in prose above).

| Feature | PyPTO | PTOAS | pto-isa |
| ------- | ----- | ----- | ------- |
| Static Pack (even last-dim `ConstInt` → `FP4E2M1X2`) | ✅ | ✅ consumes `!pto.f4E2M1x2` | ✅ nibble `GetByteSize` |
| Dynamic logical FP4 Pack | ⏳ hard-reject | ✅ can take packed dynamic shapes | ✅ |
| `tensor.view` / tile load-store (static even last dims) | ✅ `valid_shape` packs with shape | ✅ | ✅ |
| FP4↔BF16 cast | ✅ | ✅ | ✅ TCVT |
| FP4→FP8\* cast | ⚠️ Warning (prefer LUT/host) | ✅ | ✅ |
| `FP4` ↔ `FP4E2M1X2` cast | ❌ rejected | — | — |
| `FP4E2M1X2` layout conversion (non-ND, `[M,1]` DN, layout `tensor.view`) | ❌ ND row-major only | — | — |
| `transpose` / `ttrans` | ❌ | ❌ no f4E2M1x2 `ttrans` | ❌ |
| `tensor.dim` on FP4 family | ❌ use static shapes | — | — |
| Distributed (remote / window / put / get) | ⏳ | ⚠️ comm/view for other dtypes | — |
| `matmul_mx` native FP4 data | ⏳ | ✅ MX path | ✅ TCVT |

## Recommended paths

1. DSv4.1-style **UINT8 half-width** cache ABI when block/slot sizes are dynamic.
2. Limited static `pl.FP4` (even last dims) + PackFp4 + local Expand.
3. Optional hand-written `pl.FP4E2M1X2` with physical **carrier** shapes (avoids
   `#2754`-class stride bugs; layout / cube still checked).
4. Prefer **LUT / host** for FP4→FP8; device cast is Warning-only.

## See also

- [PackFp4 pass](passes/07-pack_fp4.md) — algorithm only
- [Types](../user/language/00-types.md) — short FP4 blurb
- [Operators / MX](ir/05-operators.md) — matmul_mx FP4 note
