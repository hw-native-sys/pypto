# Packed FP4 types in PyPTO

PyPTO exposes two distinct packed FP4 data types. This page describes their
shared type and byte-carrier ABI, followed by the existing E2M1 operator support.
The short type table is in [Types](../user/language/00-types.md).

## Type and unit contract

| DSL type | Uppercase alias | Storage | PTO type |
| -------- | --------------- | ------- | -------- |
| `pl.fp4e2m1x2` | `pl.FP4E2M1X2` | Two E2M1 values per byte | `!pto.f4E2M1x2` |
| `pl.fp4e1m2x2` | `pl.FP4E1M2X2` | Two E1M2 values per byte | `!pto.f4E1M2x2` |

Both types have `get_bit() == 8` and `get_byte() == 1`. Tensor and tile shapes,
`valid_shape`, strides, and offsets count byte carriers. A `[2,256]` tensor
occupies 512 bytes and contains 1,024 FP4 values. A carrier extent or offset
need not be even; each carrier already contains a complete pair of values.
Operator-specific alignment and layout restrictions still apply.

There is no unpacked FP4 type or nibble-counting ABI. The old `pl.FP4`,
`DataType.FP4`, and `DT_FP4` names are removed. JIT specialization, function
arguments and returns, runtime allocation, and persisted argument metadata
preserve carrier geometry without doubling or halving a dimension.

The two formats are distinct even when their shapes and byte sizes match.
Changing a function annotation does not convert data: dependency calls,
including void calls and submits, reject E2M1/E1M2 format mismatches.
Numerical conversion between these formats is deferred to a later change.

## Torch and runtime carriers

- `torch.float4_e2m1fn_x2` maps directly to `FP4E2M1X2`, with its physical shape
  unchanged. An E1M2 annotation rejects this E2M1-specific Torch dtype.
- An explicit annotation for either packed type accepts `torch.uint8` carriers.
  Unannotated UINT8 data remains UINT8; the annotation supplies the FP4 format.
- E2M1 return allocation uses Torch's native x2 dtype when available, otherwise
  UINT8. E1M2 uses UINT8 because Torch has no corresponding native dtype.
- Compiled and reloaded argument metadata keeps the numerical format and the
  same carrier shape. Packed runtime tensors require rank at least one and a
  positive final carrier extent.

Raw carrier storage is not a host numerical implementation. `torch_codegen`
rejects packed FP4 references instead of treating their bytes as FP32 values.
Use a reference that explicitly decodes the selected format when checking
numerical results.

## Existing E2M1 operator support on Ascend950

The type foundation retains the existing `FP4E2M1X2` paths. It does not add
E1M2 numerical casts, new transfer layouts, or MXFP4 quantization.

| Operation | Existing support |
| --------- | ---------------- |
| Packed GM views | ND carrier dimensions and strides; PTOAS v0.67 handles DMA descriptor conversion |
| E2M1x2 ↔ UINT8/INT8 alias | Same-shape, dense addressed 2D row-major Vec storage; PYPTO or DSA_RP planning |
| E2M1x2 ↔ BF16 cast | Native TCVT; the last axis expands/contracts 2:1; packing requires a static positive even source extent |
| E2M1x2 → FP8E4M3FN cast | One exact native TCVT on A5; eligible private cast-plus-move chains write FP8 NZ directly |
| E2M1x2 → FP8E5M2 cast | Legalized through BF16 and FP32, with a warning |
| E1M2 numerical casts and E2M1↔E1M2 conversion | Deferred |
| Packed DN, NZ, column-major, column-vector `[M,1]`, or layout-changing packed views | Not enabled by this change |
| Direct packed `reshape` or `transpose` | Unsupported; reshape a byte alias where the existing alias contract permits it |
| Native packed FP4 `matmul_mx` and MXFP4 `quant_mx` | Unsupported; the existing E2M1-to-FP8 cast can prepare an FP8 matrix operand |
| Ascend910/A2A3 packed FP4 execution | Unsupported |

PTO GM views retain carrier coordinates. The conversion to the lower-level
instruction's value geometry belongs to PTOAS; scaling the GM shape again in
PyPTO would double its row pitch.

Pass torch `float4_e2m1fn_x2` tensors with shapes counted in carriers to compiled
programs. Dispatch checks the carrier shape and rejects a tensor that uses
logical nibble extents for a packed parameter.

## Decode a UINT8 carrier through a UB alias (Ascend950)

This existing E2M1 path keeps the GM tensor in UINT8 and aliases its UB storage:

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
        fp4 = pl.reinterpret_view(payload, pl.fp4e2m1x2)
        decoded = pl.cast(fp4, pl.BF16)
        pl.tile.store(decoded, [row, 0], output)
    return output
```

The alias keeps the `[32,256]` carrier shape and the same 8,192-byte allocation.
Only the numerical cast creates a fresh BF16 `[32,512]` tile.

The alias requires `slayout=none_box`, dense storage, zero tile `start_offset`,
and an address-backed MemRef with a static UB byte offset. Its source byte span
must equal the carrier element count. Noncontiguous UB slices, including
intermediate aliases or reshapes of them, and addressless inputs are rejected.
GM load offsets may be nonzero or dynamic; they are distinct from UB offsets.
An explicit alias shape must match the source. Reshape a byte view separately
before aliasing back. InCore tensor aliases lower to the same tile path;
orchestration tensor aliases remain unsupported.

PYPTO and DSA_RP planning emit addressed aliases. PTOAS planning remains
unsupported for this path because its `treshape` rejects the packed FP4 type.
These existing alias constraints do not establish additional E1M2 operator
support.

## Cast policy (Ascend950)

The ISA conversion is merged in canonical
[cann/pto-isa!1967](https://gitcode.com/cann/pto-isa/merge_requests/1967).
The closed [GitHub mirror #355](https://github.com/hw-native-sys/pto-isa/pull/355)
does not establish mirror or runtime-pin availability. The current runtime/ISA
pins still lack the implementation. Compiler support remains tracked by
[PTOAS #1637](https://github.com/hw-native-sys/PTOAS/pull/1637) and
[#1669](https://github.com/hw-native-sys/PTOAS/issues/1669).
The public PTOAS v0.67 pin and v0.69 release lack the direct path; a release
containing all prerequisites is not declared here.

A public cast from row-major ND E2M1 carriers returns a fresh row-major ND
FP8 tile. Both PYPTO and PTOAS memory planners use one native conversion,
without compiler LUT/index tiles, BF16/FP32 staging, or conversion scratch.
Source shape, valid extents, strides, and GM offsets retain byte-carrier units;
only the cast result expands columns and valid columns by two.
For example, a source with shape `[32,256]` and valid extent `[23,129]`
produces FP8 shape `[32,512]` and valid extent `[23,258]`.
Odd carrier valid-column extents are valid within aligned physical tile
geometry; they already represent complete pairs. Physical ND/NZ alignment
restrictions still apply.

Before allocation, `ExpandMixedKernel` may combine that cast with its sole
compatible NZ move. This private path writes the fresh FP8 NZ result at the
original cast position. Shared results, cross-region uses, allocated aliases,
or mismatching layout/valid metadata retain the ND cast and separate move.
A `[32,256]` packed source and `[32,512]` FP8 NZ result require disjoint
8,192-byte and 16,384-byte buffers, with no FP8 ND intermediate.
This enables NZ **output**, without adding packed NZ or column-major input.
NK matrix addressing decodes row-major packed `[N,K/2]` then transposes FP8;
packed reshape and transpose remain unsupported.

E1M2 numerical casts and E2M1/E1M2 conversion remain unsupported.
E2M1-to-E5M2 retains the BF16/FP32 bridge and warning. A2/A3 packed execution,
public byte-alias planner restrictions, and other cast/scratch rules are unchanged.

## See also

- [Types](../user/language/00-types.md) — public aliases and carrier ABI
- [Operators / MX](ir/05-operators.md) — existing MXFP8 matrix paths


## Opt-in padded NZ and MX L0 scheduling

The per-compilation options `enable_fp4_nz_row_plus_one` and
`enable_mx_l0_ping_pong` default to false and are independent. Pass them to
`ir.compile(...)`, or wrap a JIT invocation in
`passes.PassContext([], enable_fp4_nz_row_plus_one=True,
enable_mx_l0_ping_pong=True)`. Nested dump/profiling contexts inherit both;
JIT cache keys distinguish all four combinations. The emitted
`compiler_options.json` records resolved requested options, not a claim that
every operation was eligible.

Padded NZ selection runs after vector splitting and final cross-core adapters.
It applies only to A5 native E2M1x2-to-E4M3FN conversion with a full per-AIV
`128x256` output and exactly one `tpush_to_aic` consumer. Other shapes,
partial or dynamic valid extents, and additional consumers retain ordinary NZ.
The logical tile remains `128x256`; `CompactMode.row_plus_one` reserves
`129*256 = 33024` UB bytes. The producer, aliases and TPUSH retain this mode;
TINSERT removes each extra panel row into the ordinary `256x256` L1/FIFO
payload. Input carriers, numerical conversion, scales and FIFO size do not
change.

The option requires PTOAS direct-convert RowPlusOne verification and an ISA
TCVT implementation with padded storage sizing. It is disabled by default
while runtime dependency pins are brought forward. The independent MX option
requests supported K128 L0 ping/pong lowering; it does not change NZ eligibility.

MX ping/pong applies to full A5 M48/M64, N256, K256 FP8/E8M0
matrix operations whose four L0 staging moves have exclusive consumers.
It preserves the K256 Mat/FIFO source and splits each matrix operation into
two increasing-K128 operations with the original FP32 accumulation order.
Two independent pinned L0B slots hold 32768 bytes each; L0A slots use the
actual M*128-byte footprint. Each scale handle derives its address from its
paired data address shifted right by four. The producer FIFO is released only
after both B slices have been copied. Explicit user storage, unsupported
shapes and PTOAS memory planning retain the original lowering. This fallback
does not affect the independent padded NZ option. PyPTO memory planning still
uses PTOAS for verification, synchronization and C++ emission.

Explicit RowPlusOne Tile descriptors validate static positive physical rows
divisible by 16, columns divisible by 32, and padded offset bounds.
Buffer IR and multi-buffer slot descriptors currently reject this mode;
use independent Tile allocations. Automatic A5 kernels remain on Tile IR.
