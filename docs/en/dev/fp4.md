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
| `make_tensor_view` shape / strides (after ExpandPackedFp4\*) | nibble (for pto-isa `GetByteSize`) |
| `partition_view` sizes | nibble |
| `partition_view` offsets (after the FP4 halve) | **pre-halved**: PTOAS advances the partition base pointer in carrier-byte units (`data_ptr + Σ offset·stride`) while pto-isa walks strides/extents in nibble units, so PyPTO emits every FP4-family window offset divided by two — rows halve from nibble rows and the packed last axis nets back to the carrier value |
| runtime Tensor / `torch.float4_e2m1fn_x2` | carrier element |

Window offsets must therefore be **even in nibble units** (even row starts, last-axis
windows starting on a carrier boundary); an odd offset has no integer byte encoding
under PTOAS's mixed-unit arithmetic and is rejected at codegen with that explanation.
Device-proven by `worktrees/fp8fp4/probe_fp4_noffset.py` (pre-halving: k0=32 reads row
32, nb=32 reads column 32; without it both read at 2×).

Write multi-row ND packed tensors with **carrier** last dims and leading strides
(for example `pl.Tensor[[2, 256], pl.FP4E2M1X2]` for 512 logical nibbles per row).
Codegen expands GM views to nibble units so multi-row pitch matches Tile /
pto-isa. Using logical widths on `FP4E2M1X2` (or logical `pl.FP4` multi-row ND
without automatic pack) can mis-size GM row strides — see issue
[#2754](https://github.com/hw-native-sys/pypto/issues/2754). When feeding a
compiled program, pass torch `float4_e2m1fn_x2` tensors whose **shape counts
carriers** (a logical `[K, N]` annotation compiles to the runtime carrier shape
`[K, N/2]`; dispatch rejects a logical-shaped tensor — a 2×-wide GM view used to
read every row past its true end before that check existed).

## Cast policy (Ascend950)

`LegalizeTileCast` treats the FP4 family as follows:

| Cast | Behavior |
| ---- | -------- |
| `FP4` / `FP4E2M1X2` ↔ `BF16` | Native TCVT (DSv4.1 c1a). Silent. Packed ↔ wider types adjust the last axis **2:1** (static even last dim required when packing). |
| `FP4` / `FP4E2M1X2` → `FP8E4M3FN` | **In-kernel LUT + tgather** on Ascend950 (`LowersFp4ToFp8InKernel`): one function-level 256-entry `u16` table (`LUT[i] = f8(i&15) \| f8(i>>4)<<8`, the exact E2M1→E4M3 bit embedding; the table is materialized once, at the first cast site) plus, per cast site, a `u8→u16` zero-extend (the one native TCVT) and a `tile.gather` lookup. At equal tiles it is never slower than the FP4→BF16→FP32→FP8 multi-hop: `[N,K]` rhs (cast followed by a Vec transpose) **41.6 ms vs 47.2 ms** at 256×256×128 tiles and **366 ms vs 378 ms** at the default 128×128×256 tiles (that config's overall cost comes from the transpose/NZ data path, not from the cast); `[K,N]` rhs 10.8 ms at the default tiles. Other backends keep the FP4→BF16→FP32→FP8 expansion with a **Warning**. |
| `FP4` / `FP4E2M1X2` → `FP8E5M2` | Legalized as FP4→BF16→FP32→FP8 with a **Warning** (the LUT/gather route covers E4M3 only). |
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

Same 2:1 last-axis expand. On Ascend950 the cast lowers to the
kernel-resident 256-entry u16 LUT + `tile.gather` route (no Warning);
backends without the in-kernel lowering expand multi-hop with a Warning:

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
| GM ExpandPackedFp4\* (carrier→nibble) | ✅ | `make_tensor_view` / partition last axis (ND) |
| `FP4E2M1X2` ↔ BF16 cast | ✅ | Silent native hop; result strides rebuilt contiguous |
| `FP4E2M1X2` → `FP8E4M3FN` cast | ✅ | Kernel-resident LUT + tgather on 950; ⚠️ multi-hop Warning elsewhere |
| `FP4E2M1X2` → `FP8E5M2` cast | ⚠️ | Warning; prefer LUT / host |
| `FP4` ↔ `FP4E2M1X2` cast | ❌ | Rejected |
| Automatic PackFp4 | ⏳ | Follow-up |
| `reshape` / `transpose` / DN / NZ / column-vector `[M,1]` / layout `tensor.view` for `FP4E2M1X2` | ❌ | ND row-major only; implicit DN and explicit layout conversion hard-rejected |
| `matmul_mx` native FP4 data | ⏳ | Cast lhs to FP8 first when needed |

## Recommended paths

1. Hand-write `pl.FP4E2M1X2` with physical carrier shapes (avoids `#2754`-class stride bugs).
2. Cast to BF16 when a native wider float is enough.
3. On Ascend950, `pl.cast` to `FP8E4M3FN` lowers to the in-kernel LUT + tgather (silent). On other backends prefer **LUT / host** — the device cast expands multi-hop with a Warning.
4. Keep logical `pl.FP4` only if you accept the incomplete path until PackFp4 lands.

## See also

- [Types](../user/language/00-types.md) — short FP4 / FP4E2M1X2 table
- [Operators / MX](ir/05-operators.md) — `matmul_mx` FP4 note
