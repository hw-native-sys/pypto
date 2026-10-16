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

Even last-axis extents apply to **logical nibble** geometry (and to wider→packed
casts that halve a static even last dim). A hand-written **carrier** last dim may
be odd. Prefer ND row-major + Vec; Cube / DN / NZ / col_major / distributed
FP4-family paths are unsupported in this slice (hard rejects land with PackFp4).

`tensor` / `tile` `reshape` and `transpose` reject the FP4 family (see error
strings and the support matrix).

## Unit convention

| Layer | Unit |
| ----- | ---- |
| Frontend logical `pl.FP4` shape / `valid_shape` | nibble |
| Hand-written `pl.FP4E2M1X2` IR / `tile_buf` / Torch ABI | carrier |
| Packed `make_tensor_view` / `partition_view` (PTOAS v0.67+) | carrier; PTOAS converts DMA descriptors to nibble units |
| runtime Tensor / `torch.float4_e2m1fn_x2` | carrier element |

Write multi-row ND packed tensors with **carrier** last dims and leading strides
(for example `pl.Tensor[[2, 256], pl.FP4E2M1X2]` for 512 logical nibbles per row).
Codegen preserves carrier units in GM views. PTOAS v0.67 converts transfer
descriptors to nibble units for pto-isa; expanding them in PyPTO as well doubles
the row pitch. Using logical widths on `FP4E2M1X2` (or logical `pl.FP4` multi-row ND
without automatic pack) can mis-size GM row strides — see issue
[#2754](https://github.com/hw-native-sys/pypto/issues/2754).

## Decode a UINT8 carrier through a UB alias (Ascend950)

Keep packed bytes as UINT8 in GM, then use a zero-copy FP4E2M1X2 view in UB:

```python
import pypto.language as pl


@pl.jit(auto_scope=False)
def decode_bytes(
    packed: pl.Tensor[[32, 256], pl.UINT8],
    output: pl.Out[pl.Tensor[[32, 512], pl.BF16]],
):
    with pl.spmd(1):
        row = pl.tile.get_block_idx() * 32
        payload = pl.tile.load(packed, [row, 0], [32, 256])
        fp4 = pl.reinterpret_view(payload, pl.FP4E2M1X2)
        decoded = pl.cast(fp4, pl.BF16)
        pl.tile.store(decoded, [row, 0], output)
    return output
```

`payload` and `fp4` both have carrier shape `[32,256]` and occupy the same
8,192 bytes at the same UB address. The numerical cast expands the last axis
and writes a fresh BF16 `[32,512]` tile through native `TCVT`. GM input addressing
continues to use UINT8 byte offsets (`row * 256` for this shape); there is no
FP4 GM partition and no software nibble-decoding sequence.

The alias accepts only UINT8 → FP4E2M1X2, 2D packed row-major Vec tiles with
`slayout=none_box`, no explicit strides, and zero tile `start_offset`. An
explicit `shape` must match the source shape exactly. Carrier `valid_shape`
is preserved; the subsequent cast expands its last axis. Reverse aliases,
logical FP4 aliases, tensor aliases, and reshaping through this alias are
rejected. GM load offsets may be nonzero; they are distinct from a UB tile's
`start_offset`.

The source must have an address-backed MemRef whose byte span equals the
static carrier element count. Noncontiguous UB column slices (including aliases
and reshapes of them) and MemRef-less inputs such as `tile.tpop_from_aic` are
rejected before emitting the FP4 conversion. Load the desired window directly
from UINT8 GM into a dense UB tile instead. The alias also requires a static UB
byte offset, including through intermediate views; a GM load may still use
runtime offsets.

PTOAS **0.65 is sufficient with PYPTO and DSA_RP memory planning**. Both emit
address-backed typed aliases and pass bitwise A5 tests. The PTOAS memory planner
is rejected for this alias: its `pto.treshape` primitive rejects packed FP4 in
0.65. Other reinterpret-view dtypes retain their existing planner support.
This path does not fix generic typed-FP4 GM partition offsets, introduce
PackFp4, or add tensor aliasing.

## Cast policy (Ascend950)

`LegalizeTileCast` treats the FP4 family as follows:

| Cast | Behavior |
| ---- | -------- |
| `FP4` / `FP4E2M1X2` ↔ `BF16` | Native TCVT (DSv4.1 c1a). Silent. Packed ↔ wider types adjust the last axis **2:1** (static even last dim required when packing). |
| `FP4` / `FP4E2M1X2` → `FP8E4M3FN` / `FP8E5M2` | Legalized as FP4→BF16→FP32→FP8 with a **Warning** (prefer LUT / host precast). |
| `FP4` ↔ `FP4E2M1X2` | **Rejected** (geometry disagree); cast via a wider type or rewrite shapes. |
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

Legend: ✅ supported · ⚠️ partial / Warning · ❌ unsupported · ⏳ not in this slice

| Feature | PyPTO | Notes |
| ------- | ----- | ----- |
| Hand-written `pl.FP4E2M1X2` | ✅ | Preferred frontend for packed paths; **ND only** |
| Logical `pl.FP4` without PackFp4 | ⚠️ | Prefer warning; A5 in-core still supports legacy logical FP4 alongside FP4E2M1X2 |
| Packed GM views in carrier units | ✅ | PTOAS v0.67 handles DMA conversion (ND) |
| UINT8 → FP4E2M1X2 UB alias | ✅ | 2D packed row-major Vec; same carrier shape; PYPTO / DSA_RP only |
| `FP4E2M1X2` ↔ BF16 cast | ✅ | Silent native hop; result strides rebuilt contiguous |
| `FP4E2M1X2` → FP8\* cast | ⚠️ | Warning; prefer LUT / host |
| `FP4` ↔ `FP4E2M1X2` cast | ❌ | Rejected |
| Automatic PackFp4 | ⏳ | Follow-up |
| `reshape` / `transpose` / DN / NZ / column-vector `[M,1]` / layout `tensor.view` for `FP4E2M1X2` | ❌ | ND row-major only; implicit DN and explicit layout conversion hard-rejected |
| `matmul_mx` native FP4 data | ⏳ | Cast lhs to FP8 first when needed |

## Recommended paths

1. Hand-write `pl.FP4E2M1X2` with physical carrier shapes (avoids `#2754`-class stride bugs).
2. Cast to BF16 when a native wider float is enough.
3. Prefer **LUT / host** for FP4→FP8; device cast is Warning-only.
4. Keep logical `pl.FP4` only if you accept the incomplete path until PackFp4 lands.

## See also

- [Types](../user/language/00-types.md) — short FP4 / FP4E2M1X2 table
- [Operators / MX](ir/05-operators.md) — `matmul_mx` FP4 note
