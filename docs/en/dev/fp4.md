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

PyPTO also rejects FP4-family `reshape`, scalar `read`/`write`,
unwhitelisted coordinate ops, and cube / DN / NZ / col_major packing — see pass
docs and error strings (these are PyPTO-only limits, not a three-layer matrix
row). `tensor.dim` remains supported: non-last axes are unchanged; the last axis
stays in logical nibble units after PackFp4.

`FP4E2M1X2` layout conversion is hard-rejected: non-ND annotations, implicit
column-vector `[M, 1]` DN, and explicit `tensor.view` layout flips.

## Pipeline

1. **PackFp4** (static-only): logical `pl.FP4` → `FP4E2M1X2`; last-axis covering
   sizes / offsets / ND leading strides must be even `ConstInt` and are halved.
2. **PTO codegen**: GM `make_tensor_view`, `partition_view`, offsets, strides
   and `tile_buf` retain carrier units. PTOAS v0.67+ expands only the
   TLOAD/TSTORE transfer descriptor to nibble units for pto-isa `GetByteSize`.
   Pointer arithmetic continues to use carrier units.
3. **Cast** (`LegalizeTileCast`): see [Cast policy](#cast-policy-ascend950).

`reinterpret_view` allows only **byte-identical** `FP4E2M1X2` ↔ `UINT8` / `INT8`
aliases (same shape). Prefer `reinterpret_view` → `UINT8` → `reshape` when a
leading-dimension flatten is required; do not rely on packed-FP4 `reshape`.
PTOAS v0.67+ fixes multi-row TLOAD/TSTORE pitch by separating carrier
addressing from nibble DMA descriptors
([PTOAS #1575](https://github.com/hw-native-sys/PTOAS/issues/1575)).

Tile byte aliases require dense, address-backed 2D row-major Vec storage in UB
and the PYPTO or DSA_RP planner, which emits an addressed alias at the same
buffer. Strided subview provenance and addressless inputs remain rejected.
PTOAS v0.67 still rejects `treshape` / `bitcast` of
`!pto.f4E2M1x2`; PyPTO reports this explicitly for unresolved tile aliases.
InCore tensor reinterpret operations lower to these tile aliases and require
the same planners. Orchestration tensor reinterpretation remains unsupported.

## Unit convention

| Layer | Unit |
| ----- | ---- |
| Frontend `pl.FP4` shape / `valid_shape` | nibble |
| IR after PackFp4 / `tile_buf` / orch create | carrier |
| Hand-written `pl.FP4E2M1X2` IR / `tile_buf` / Torch ABI | carrier |
| `make_tensor_view` / `partition_view` / offsets / strides | carrier |
| PTOAS TLOAD/TSTORE transfer descriptor | nibble (for pto-isa `GetByteSize`) |
| runtime Tensor / Torch `float4_e2m1fn_x2` | carrier element |

Write multi-row ND packed tensors with **carrier** last dims and leading strides
(for example `pl.Tensor[[2, 256], pl.FP4E2M1X2]` for 512 logical nibbles per row).
PTOAS v0.67+ separates carrier addressing from nibble DMA descriptors, fixing
multi-row transfer pitch ([PTOAS #1575](https://github.com/hw-native-sys/PTOAS/issues/1575)).
Using logical widths on `FP4E2M1X2` can still mis-size GM row strides — see
issue [#2754](https://github.com/hw-native-sys/pypto/issues/2754).

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

The same-shape alias accepts FP4E2M1X2 ↔ UINT8 / INT8, 2D packed row-major Vec tiles with
`slayout=none_box`, no explicit strides, and zero tile `start_offset`. An
explicit `shape` must match the source shape exactly. Carrier `valid_shape`
is preserved; the subsequent cast expands its last axis. Reshape the byte view,
then reinterpret it back to FP4E2M1X2 before casting; each alias must preserve
its input shape and dense storage. InCore tensor aliases lower to the same tile
path. Logical FP4 aliases, native packed-FP4 reshape, and orchestration tensor
aliases remain rejected. GM load offsets may be nonzero; they are distinct
from a UB tile's `start_offset`.

The source must have an address-backed MemRef whose byte span equals the
static carrier element count. Noncontiguous UB column slices (including aliases
and reshapes of them) and MemRef-less inputs such as `tile.tpop_from_aic` are
rejected before emitting the FP4 conversion. Load the desired window directly
from UINT8 GM into a dense UB tile instead. The alias also requires a static UB
byte offset, including through intermediate views; a GM load may still use
runtime offsets.

Use **PTOAS v0.67+ with PYPTO or DSA_RP memory planning**. Both planners emit
address-backed typed aliases. The PTOAS memory planner is rejected for this
alias: its `pto.treshape` primitive still rejects packed FP4 in v0.67. Other
reinterpret-view dtypes retain their existing planner support. The UINT8 GM
path above keeps its byte addressing; typed FP4 GM transfers use the carrier
convention described above.

## Cast policy (Ascend950)

`LegalizeTileCast` treats the FP4 family as follows:

| Cast | Behavior |
| ---- | -------- |
| `FP4` / `FP4E2M1X2` ↔ `BF16` | Native TCVT (DSv4.1 c1a). Silent. Packed ↔ wider types adjust the last axis **2:1** (static even last dim required when packing). Result strides are rebuilt contiguous from the destination shape. |
| `FP4` / `FP4E2M1X2` → `FP8E4M3FN` / `FP8E5M2` | Legalized as FP4→BF16→FP32→FP8 with a **Warning** (prefer LUT / host precast). |
| `FP4` ↔ `FP4E2M1X2` | **Rejected** (`RejectFp4FamilyInternalCast`; geometry disagree). Cast via a wider type or rewrite shapes. |
| Wider → `FP4E2M1X2` with dynamic last dim | **Rejected** (static positive even only). |

The pinned pto-isa CPU simulator rounds fractional FP4 values to integers during
FP4→BF16 TCVT. Cast runtime tests therefore run on A5 hardware with exact
comparisons; byte-copy tests also run on the simulator. See the
[pinned CPU TCVT implementation](https://github.com/hw-native-sys/pto-isa/blob/418de18dde8fa33773dc5c384fdfe5506463b628/include/pto/cpu/TCvt.hpp).

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
| FP4↔BF16 cast | ✅ | ✅ | ✅ A5 TCVT; CPU simulator limitation above |
| InCore packed FP4 ↔ byte alias, byte `reshape`, alias back | ✅ same-shape aliases; dense addressed UB; PYPTO / DSA_RP | ❌ FP4 `treshape` in v0.67; addressed aliases bypass it | ✅ shared storage |
| FP4→FP8\* cast | ⚠️ Warning (prefer LUT/host) | ✅ | ✅ |
| `FP4` ↔ `FP4E2M1X2` cast | ❌ rejected | — | — |
| `FP4E2M1X2` layout conversion (non-ND, `[M,1]` DN, layout `tensor.view`) | ❌ ND row-major only | — | — |
| `transpose` / `ttrans` | ❌ | ❌ no f4E2M1x2 `ttrans` | ❌ |
| `tensor.dim` on logical FP4 | ✅ non-last unchanged; last axis = nibble | — | — |
| Distributed (remote / window / put / get) | ⏳ | ⚠️ comm/view for other dtypes | — |
| `matmul_mx` native FP4 data | ⏳ | ✅ MX path | ✅ TCVT |

## Recommended paths

1. DSv4.1-style **UINT8 half-width** cache ABI when block/slot sizes are dynamic.
2. Limited static `pl.FP4` (even last dims) + PackFp4 + PTOAS v0.67+.
3. Optional hand-written `pl.FP4E2M1X2` with physical **carrier** shapes (avoids
   `#2754`-class stride bugs; layout / cube still checked).
4. Use PYPTO / DSA_RP for packed FP4 ↔ byte aliases and reshape the byte view.
5. Prefer **LUT / host** for FP4→FP8; device cast is Warning-only.

## See also

- [PackFp4 pass](passes/07-pack_fp4.md) — algorithm only
- [Types](../user/language/00-types.md) — short FP4 blurb
- [Operators / MX](ir/05-operators.md) — matmul_mx FP4 note
