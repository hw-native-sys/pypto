# FP4 in PyPTO

Canonical status page for MXFP4 / packed FP4 in the current release. Short type-table
blurbs live in [Types](../user/language/00-types.md); this page covers roles, units,
cast policy, and what is safe to write by hand today.

Automatic `PackFp4` (logical `pl.FP4` → packed carrier) is **not** in this release
slice; prefer hand-written `pl.FP4E2M1X2` until that pass lands.

## Roles

| Type | Meaning |
| ---- | ------- |
| `pl.FP4` | Frontend **logical** E2M1 nibble (`GetBit()==4`). Shapes / `valid_shape` count nibbles. Resolving this short name emits a `UserWarning` (prefer `FP4E2M1X2`). |
| `pl.FP4E2M1X2` | **Packed** carrier (two nibbles per element, `GetBit()==8`). Physical last dim matches `torch.float4_e2m1fn_x2` / `!pto.f4E2M1x2`. |

Even last-axis extents for packed geometry are the author's responsibility. Cube /
DN / NZ / col_major FP4-family paths are rejected.

## Unit convention

| Layer | Unit |
| ----- | ---- |
| Frontend logical `pl.FP4` shape / `valid_shape` | nibble |
| Hand-written `pl.FP4E2M1X2` IR / `tile_buf` / Torch ABI | carrier |
| runtime Tensor / `torch.float4_e2m1fn_x2` | carrier element |

Write multi-row ND packed tensors with **carrier** last dims and leading strides
(for example `pl.Tensor[[2, 256], pl.FP4E2M1X2]` for 512 logical nibbles per row).
Using logical widths on `FP4E2M1X2` (or logical `pl.FP4` multi-row ND without
automatic pack) can mis-size GM row strides — see issue
[#2754](https://github.com/hw-native-sys/pypto/issues/2754).

## Cast policy (Ascend950)

`LegalizeTileCast` treats the FP4 family as follows:

| Cast | Behavior |
| ---- | -------- |
| `FP4` / `FP4E2M1X2` ↔ `BF16` | Native TCVT (DSv4.1 c1a). Silent. Packed ↔ wider types adjust the last axis **2:1**. |
| `FP4` / `FP4E2M1X2` → `FP8E4M3FN` / `FP8E5M2` | Legalized as FP4→BF16→FP32→FP8 with a **Warning** (prefer LUT / host precast). |

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

Legend: ✅ supported · ⚠️ partial / Warning · ❌ unsupported · ⏳ not in this slice

| Feature | PyPTO | Notes |
| ------- | ----- | ----- |
| Hand-written `pl.FP4E2M1X2` | ✅ | Preferred frontend for packed paths |
| Logical `pl.FP4` without PackFp4 | ⚠️ | Warning; in-core rejects logical FP4 on A5 |
| `FP4E2M1X2` ↔ BF16 cast | ✅ | Silent native hop |
| `FP4E2M1X2` → FP8\* cast | ⚠️ | Warning; prefer LUT / host |
| Automatic PackFp4 | ⏳ | Follow-up |
| `transpose` / cube / distributed FP4 | ❌ / ⏳ | Out of scope here |
| `matmul_mx` native FP4 data | ⏳ | Cast lhs to FP8 first when needed |

## Recommended paths

1. Hand-write `pl.FP4E2M1X2` with physical carrier shapes (avoids #2754-class stride bugs).
2. Cast to BF16 when a native wider float is enough.
3. Prefer **LUT / host** for FP4→FP8; device cast is Warning-only.
4. Keep logical `pl.FP4` only if you accept the incomplete path until PackFp4 lands.

## See also

- [Types](../user/language/00-types.md) — short FP4 / FP4E2M1X2 table
- [Operators / MX](ir/05-operators.md) — `matmul_mx` FP4 note
