# Packed FP4 types in PyPTO

PyPTO exposes two distinct packed FP4 data types. This page describes their
shared type and byte-carrier ABI, numerical casts, and current operator support.
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
including void calls and submits, require equal argument and parameter dtypes
whenever either is packed FP4. UINT8/INT8 values require an explicit
`reinterpret_view` where byte aliases are supported; a callee annotation alone
does not reinterpret them.
Use an explicit numerical `pl.cast` to convert between the formats.

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

## Operator support on Ascend950

Both formats support ND casts, full static dense DN casts, and the transfers
below. Native MXFP4 quantization supports E2M1 only.

| Operation | Support |
| --------- | ------- |
| Packed GM views and load/store | Both formats, ND/DN carrier dimensions, strides, and offsets; a one-carrier width stays ND; PTOAS v0.67 handles DMA descriptor conversion |
| Either packed format ↔ UINT8/INT8 alias | Same-shape orchestration GM aliases and dense 2D row-major Vec aliases with all three planners |
| Either packed format ↔ BF16 cast | Native TCVT on unboxed Vec; ND columns or full static dense DN rows expand/contract 2:1 |
| E2M1 → FP8E4M3FN cast | Direct A5 TCVT, ND or fused NZ output; no BF16/FP32 intermediate |
| Other packed FP4 → FP8 casts | Legalized through BF16 and FP32, with a warning |
| E2M1↔E1M2 conversion | Numerical cast through BF16; carrier shape is unchanged |
| Other wider-type casts | Existing legalization with lossless intermediates; see the encoding constraint below |
| Packed Vec layout conversion | ND→NZ; carrier geometry changes with the packing axis while preserving the value geometry |
| Packed Vec→Mat transfer | Matching layouts, explicit AIV function; generic InCore mixed-core expansion is rejected |
| Native packed `extract` / `assemble` | Restricted layout-preserving cube/NZ paths with zero offset on the packed axis; use INT8 aliases for general Vec extraction and UINT8 aliases for ND insertion |
| `fillpad`, `fillpad_expand`, `fillpad_inplace` | Both formats, ND Vec; zero/min/max padding preserves the valid carrier rectangle |
| DN numerical casts | Full static dense 2D BF16↔packed or packed↔packed; PYPTO/DSA_RP static addresses; partial, padded, and boxed cases are rejected |
| Direct packed `reshape` or `transpose` | Unsupported; reshape a byte alias where the existing alias contract permits it |
| Native packed FP4 `matmul_mx` | Tensor and Tile APIs: both formats and mixed E2M1/E1M2 pairs, FP32 accumulator; requires PTOAS v0.78 or a compatible newer build |
| MXFP4 `quant_mx` | E2M1 only, FP16/BF16 source, OCP or supported axis0 NV; PYPTO/DSA_RP static allocation |
| `pl.tile.quant_mx_raw` | Native E2M1 payload (INT8 for MXFP8) and raw UINT8 exponents with explicit workspaces; static partial validity and axis0 interleaving |
| Ascend910/A2A3 packed FP4 execution | Unsupported |

PTO GM views retain carrier coordinates. The conversion to the lower-level
instruction's value geometry belongs to PTOAS; scaling the GM shape again in
PyPTO would double its row pitch. Physical tiles still need ISA alignment,
including a 32-byte ND Vec row pitch; use padded storage with `valid_shape`
for a narrower carrier window.

Native FP4 matrix operands keep carrier dimensions through GM and Mat.
For logical `[M, K] @ [K, N]`, Left uses `[M/2, K]` carriers and Right uses
`[K, N/2]`; scales remain `[M, K/32]` and `[K/32, N]`, and Acc is FP32
`[M, N]`. Mat-to-Left/Right moves retarget the packed axis without a numeric
cast. A direct GM-to-Mat load avoids the unsupported mixed-core Vec transfer.
Orchestration FP4/byte aliases preserve buffer identity, offsets, strides,
and dependencies. Other orchestration dtype reinterpretations remain rejected.

PyPTO keeps the carrier ABI above and expands the packed axis of Mat/Left/Right
physical and valid dimensions when emitting PTO. GM and Vec descriptors retain
carrier dimensions; allocation byte sizes are unchanged. Executable generation
probes the selected assembler's logical operand, accumulator, and scale geometry
before enabling native FP4 MX. Select PTOAS v0.78 or a compatible newer build
via `PTOAS_ROOT`. IR-only generation (`skip_ptoas=True`) remains available.

## Numerical cast constraints

BF16 encoding supports `rint`, `round`, `floor`, `ceil`, and `trunc`; `odd` is
rejected. Packing requires complete pairs: a positive even source extent or a
provably even dynamic `valid_shape`, such as `2 * n`. Decoding expands the
carrier extent without requiring it to be even.

Packed cast sources require `none_box` storage. Wider cast chains require ND.
A5 E2M1-to-E4M3 tiles support fused NZ output:

```python
fp8_nz = pl.cast(packed, pl.FP8E4M3FN, layout=pl.NZ)
```

The source is an ND Vec tile with physical carrier columns divisible by 32
and static positive valid columns; valid rows may be runtime values.
`valid_shape` may describe a narrower rectangle. The direct path uses PYPTO
or DSA_RP address planning to bind static-column descriptors. Plain casts with
PTOAS planning or dynamic valid columns retain the BF16/FP32 bridge. Plain ND
casts from slice-derived views and standalone AIV functions awaiting lane
splitting also keep that bridge conservatively; authored per-lane kernels are
already final and remain eligible. An explicit NZ request requires the
addressed direct path. The result keeps logical
`[M, 2*K]` dimensions while its NZ descriptor controls physical storage.
NZ physical rows must be divisible by 16; use `valid_shape` for row tails.
The layout option is Tile-only. E1M2 and FP8E5M2 do not have this direct path.
PTOAS v0.77 or newer and the matching pinned A5 ISA are required.

For a direct NZ cast, the source `tile.slice` retains its parent row pitch and runtime row offset.
Its UB address must be provably 32-byte aligned: use static packed column
offsets divisible by 32. Runtime row offsets remain aligned when the parent
row pitch is a multiple of 32 bytes. Unknown runtime column offsets are
rejected; load that packed window into its own aligned tile instead.
A plain ND cast through BF16 and FP32 accepts unaligned carrier offsets, including
a runtime column. PyPTO extracts the byte window using the parent's row pitch
and decodes both nibbles from an aligned temporary of the window's size.
For one valid carrier per row, keep the physical width a multiple of 32 bytes.
For example, `pl.tile.slice(packed, [3, 32], [1, 1], valid_shape=[3, 1])`
decodes to three rows of two valid BF16 values.

Use `tpush_to_aic` to transfer the NZ result to the Cube without another
ND-to-NZ move. On A5, a plain Vec store can also copy it to a matching `pl.NZ`
FP8E4M3FN GM tensor. That store requires complete static 16x32 boxes and rejects
atomic, FIXPIPE epilogue, and explicit `st_phase` options. It preserves NZ box
order and does not convert NZ back to ND. The pinned CPU simulator loses
signed zero, so physical-byte acceptance uses A5 hardware.

RowPlusOne padding remains deferred to the assembler work tracked in
[PTOAS #1684](https://github.com/hw-native-sys/PTOAS/issues/1684) and
[PTOAS #1685](https://github.com/hw-native-sys/PTOAS/pull/1685); the public cast
currently emits ordinary NZ storage.

Decoding through BF16 preserves every FP4 value exactly. Implicit encoding chains must also preserve
the original source through BF16: FP16/FP32 and larger integer sources are
rejected because that intermediate can double-round. Write an explicit BF16
cast first when that intermediate rounding is intended. The pinned simulator
misrounds fractional FP4 conversions; numerical acceptance uses A5 hardware.

DN casts require a full static 2D rectangle, canonical column-major strides,
zero tile start offset, and BF16/packed endpoints. DN `[M,K]` carriers decode
to BF16 `[2*M,K]`. Use `pl.tile.transpose_view` on a full ND BF16 tile to
prepare a column-major BF16 source. Tensor casts preserve the same contract.
PYPTO or DSA_RP must supply static tile addresses; codegen binds full static
valid descriptors to select the safe native conversion path. Dynamic-valid
DN descriptors can fault even for a full rectangle and remain unsupported.

## Decode a UINT8 carrier through a UB alias (Ascend950)

Keep the GM tensor in UINT8 and alias its UB storage with the encoded format:

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
Only the numerical cast creates a fresh BF16 `[32,512]` tile. Use
`pl.fp4e1m2x2` instead when the source bytes encode E1M2.

The alias requires `slayout=none_box`, dense storage, zero tile `start_offset`,
and a locally planned Vec allocation. Its source byte span must equal the
carrier element count. Noncontiguous UB slices, including intermediate aliases
or reshapes of them, and inputs without a MemRef are rejected.
GM load offsets may be nonzero or dynamic; they are distinct from UB offsets.
An explicit alias shape must match the source. Reshape a byte view separately
before aliasing back. InCore tensor aliases lower to the same tile path;
orchestration tensor aliases support only same-shape packed FP4 ↔ UINT8/INT8 aliases.

PYPTO and DSA_RP bind both types to the same UB address. With the PTOAS
planner, PTOAS v0.78 lowers the same-shape byte alias through `pto.treshape`.
Aliasing preserves bits; it does not numerically convert one FP4 format into
the other.

Statically empty packed-FP4 tile slices reuse an initialized address-backed
result descriptor, avoiding `treshape`. They require PYPTO or DSA_RP planning,
a static offset, and a contiguous row-major Vec window directly over a local
allocation. PTOAS-planned or addressless sources, dynamic offsets, nested views,
and strided/fractal windows report a diagnostic. Slice the GM tensor before
loading when these restrictions apply; ordinary nonempty slices are unchanged.

Partial-transfer hardware tests compare every carrier byte, including untouched
output regions. The pinned simulator corrupts DN results, and its partial ND
case crashes during cleanup despite matching bytes; these cases run on A5 only.

## MX quantization

`pl.quant_mx(..., dtype=pl.fp4e2m1x2)` quantizes FP16/BF16 in groups of 32
values. Axis1 maps `[M,K]` to payload `[M,K/2]` and scales `[M,K/32]`;
axis0 maps `[N,K]` to payload `[K,N/2]` and scales `[K/32,N]`. Payload shapes
count byte carriers. Public quantization requires full valid tiles, `K%64==0`,
axis1 `M%16==0`, and FP4 axis0 `N%64==0` for its 32-byte carrier stride.

`quant_scale_alg="ocp"` is the default. `"nv"` supports the FP16/BF16 axis0
path; axis1 FP16/BF16 NV and E1M2 quantization are unsupported by PTOAS.
Packed quantization needs PYPTO or DSA_RP with static addresses because PTOAS
cannot reshape its custom FP4 type into the required static-valid descriptor.

`pl.tile.quant_mx_raw` keeps source orientation and returns native packed data
plus UINT8 exponent codes without composite scale boxing. Workspaces use the
source dtype. Axis0 workspaces retain the source physical column stride; axis1
FP16/BF16 OCP scaling capacity rounds the group count up to 128 elements.
Static partial valid regions are supported; axis1 requires tight physical/valid
columns and FP4 physical columns divisible by 64. Axis0 groups 32 valid rows;
`interleave=True` requires FP16/BF16 physical and valid rows divisible by 64.
See the [operator catalog](../user/ops/01-catalog.md) for the quantization entry
points. Native FP4 matrix multiplication is available through Tensor and Tile APIs.
`tensor.matmul_mx` consumes static oriented 2D ND/DN data: ND carriers are
`[M,K/2]` / `[K,N/2]`, DN carriers are `[M/2,K]` / `[K/2,N]`. Scales use
logical `[M,K/32]` / `[K/32,N]`, and the result is FP32 `[M,N]`. Logical
M/K/N must be multiples of 32/64/64 (M fills the existing boxed carrier rows). Accumulation and bias remain Tile-only. The generic mixed-core FP4 Vec-to-Mat
restriction still applies to quantization results.

## See also

- [Types](../user/language/00-types.md) — public aliases and carrier ABI
- [Operators / MX](ir/05-operators.md) — existing MXFP8 matrix paths
