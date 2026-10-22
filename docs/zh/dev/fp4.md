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
| E2M1x2 → FP8E4M3FN/FP8E5M2 cast | Legalized through BF16 and FP32, with a warning; a host reference or lookup table remains an alternative |
| E1M2 numerical casts and E2M1↔E1M2 conversion | Deferred |
| DN, NZ, column-major, column-vector `[M,1]`, or layout-changing packed views | Not enabled by this change |
| Direct packed `reshape` or `transpose` | Unsupported; reshape a byte alias where the existing alias contract permits it |
| Native packed FP4 `matmul_mx` and MXFP4 `quant_mx` | Unsupported; the existing E2M1-to-FP8 cast can prepare an FP8 matrix operand |
| Ascend910/A2A3 packed FP4 execution | Unsupported |

PTO GM views retain carrier coordinates. The conversion to the lower-level
instruction's value geometry belongs to PTOAS; scaling the GM shape again in
PyPTO would double its row pitch.

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

## See also

- [Types](../user/language/00-types.md) — public aliases and carrier ABI
- [Operators / MX](ir/05-operators.md) — existing MXFP8 matrix paths
