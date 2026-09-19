# FP4 in PyPTO

Canonical status page for MXFP4 / packed FP4. Pass algorithm details live in
[PackFp4](passes/07-pack_fp4.md); this page covers roles, pipeline, unit
convention, and the three-layer support matrix.

## Roles

| Type | Meaning |
| ---- | ------- |
| `pl.FP4` | Frontend **logical** E2M1 nibble. Shapes / `valid_shape` count nibbles. |
| `pl.FP4E2M1X2` | **Packed** carrier (two nibbles per element). Physical shapes after `PackFp4`, or hand-written. Even last dims are the author's responsibility; DN / NZ / col_major / cube are still rejected by PackFp4. |

`tensor.full` / `create` fill values: only **zero** is defined after packing; non-zero
fill semantics (duplicate nibble vs low-only) are undefined in this release.

PyPTO also rejects FP4-family `reshape`, scalar `read`/`write`, unwhitelisted
coordinate ops, and cube / DN / NZ / col_major packing — see pass docs and error
strings (these are PyPTO-only limits, not a three-layer matrix row).

## Pipeline

1. **PackFp4** (static-only): logical `pl.FP4` → `FP4E2M1X2`; last-axis covering
   sizes / offsets / ND leading strides must be even `ConstInt` and are halved.
2. **Local ExpandPackedFp4\*** (single-device codegen): expand GM
   `make_tensor_view` / partition last-axis geometry back to nibble units for
   pto-isa `GetByteSize`; EmitC expands Tile cols for TCVT. **Tile_buf stays in
   carrier units** (no expand).
3. **Cast** (`LegalizeTileCast`): FP4↔BF16 silent (DSv4.1 c1a); FP4→FP8\* allowed
   with a **Warning** (prefer LUT / host precast).

## Unit convention

| Layer | Unit |
| ----- | ---- |
| Frontend `pl.FP4` shape / `valid_shape` | nibble |
| IR after PackFp4 / `tile_buf` / orch create | carrier |
| `make_tensor_view` / `partition_view` (after Expand) | nibble (for pto-isa `GetByteSize`) |
| runtime Tensor / Torch `float4_e2m1fn_x2` | carrier element |

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
| `transpose` / `ttrans` | ❌ | ❌ no f4E2M1x2 `ttrans` | ❌ |
| Distributed (remote / window / put / get) | ⏳ | ⚠️ comm/view for other dtypes | — |
| `matmul_mx` native FP4 data | ⏳ | ✅ MX path | ✅ TCVT |

## Recommended paths

1. DSv4.1-style **UINT8 half-width** cache ABI when block/slot sizes are dynamic.
2. Limited static `pl.FP4` (even last dims) + PackFp4 + local Expand.
3. Optional hand-written `pl.FP4E2M1X2` with physical shapes (layout/cube still checked).
4. Prefer **LUT / host** for FP4→FP8; device cast is Warning-only.

## See also

- [PackFp4 pass](passes/07-pack_fp4.md) — algorithm only
- [Types](../user/language/00-types.md) — short FP4 blurb
- [Operators / MX](ir/05-operators.md) — matmul_mx FP4 note
