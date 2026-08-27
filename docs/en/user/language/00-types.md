# Types

Every value in a PyPTO program carries a type that says where it lives and how wide its
elements are. Getting the annotation right is how you tell the compiler what to allocate
and what it is allowed to do.

> **Prerequisites:** [Programming Model § memory hierarchy](../03-programming-model.md#memory-hierarchy).

## Concept

Three things are encoded in a type annotation, and it is worth separating them because
they fail differently.

**Where the value lives.** `pl.Tensor` is in DDR, `pl.Tile` is an on-chip buffer,
`pl.Scalar` is a register-width value. This is not a hint — a tensor operation on the
execution plane and a tile operation on the control plane are both rejected, and the
container type is how the compiler tells which is which.

**How wide the elements are.** The dtype constants (`pl.FP16`, `pl.INT32`, …) name a
hardware element format. Mixing them is legal but never implicit: there is no promotion,
so a `pl.cast` is required wherever the widths differ.

**How the caller may use it.** Parameter direction — `In` (the default), `pl.Out[...]`,
`pl.InOut[...]` — is part of the signature, not a convention. The compiler derives task
dependencies from directions, so a mis-declared direction produces a wrong dependency
graph rather than a compile error.

Shapes are static by default and checked at parse time. `pl.dynamic()` opts a dimension
out of that, at the cost of everything the compiler could have concluded from knowing it.

## Quickstart: reading a signature

<!-- doctest: setup -->
```python
import pypto.language as pl
import torch
from pypto.runtime import RunConfig

CFG = RunConfig(platform="__PLATFORM__")
torch.manual_seed(0)
```

<!-- doctest: run -->
```python
M = pl.dynamic("M")                       # an axis whose extent varies at run time


@pl.jit.incore
def scale_rows(
    x: pl.Tensor[[M, 128], pl.FP16],                    # In (default): read-only, DDR
    acc: pl.InOut[pl.Tensor[[M, 128], pl.FP32]],        # read-write, DDR
    out: pl.Out[pl.Tensor[[M, 128], pl.FP32]],          # write-only, DDR
    factor: pl.Scalar[pl.FP32],                         # scalar, passed by value
):
    tx = pl.load(x, [0, 0], [64, 128])
    scaled = pl.mul(pl.cast(tx, pl.FP32), factor)
    acc = pl.store(pl.add(pl.load(acc, [0, 0], [64, 128]), scaled), [0, 0], acc)
    out = pl.store(scaled, [0, 0], out)
    return acc, out


@pl.jit
def apply_scale(
    x: pl.Tensor[[64, 128], pl.FP16],
    acc: pl.InOut[pl.Tensor[[64, 128], pl.FP32]],
    out: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
):
    acc, out = scale_rows(x, acc, out, 2.0)             # the dynamic axis binds to 64 here
    return acc, out


x = torch.randn(64, 128, dtype=torch.float16)
acc = torch.randn(64, 128, dtype=torch.float32)
out = torch.zeros(64, 128, dtype=torch.float32)
acc_before = acc.clone()

apply_scale(x, acc, out, config=CFG)

scaled = x.float() * 2.0
torch.testing.assert_close(out, scaled, rtol=1e-2, atol=1e-2)               # Out was written
torch.testing.assert_close(acc, acc_before + scaled, rtol=1e-2, atol=1e-2)  # InOut was read *and* written
```

| Element | Reads as |
| ------- | -------- |
| `pl.Tensor[[M, 128], pl.FP16]` | 2D DDR array, `M` rows (runtime value), 128 columns, half precision |
| `pl.InOut[...]` | The kernel both reads and writes it — the compiler orders it against both earlier writers and earlier readers |
| `pl.Out[...]` | The kernel only writes it. Reading an `Out` parameter before writing it reads undefined memory |
| `pl.Scalar[pl.FP32]` | A single value, not a buffer |
| `M = pl.dynamic("M")` | The dimension is unknown at compile time and bound per launch |

## Mechanics

### Data types

| Constant | Bits | Notes |
| -------- | ---- | ----- |
| `pl.BOOL` | 1 | |
| `pl.INT4` / `pl.UINT4` | 4 | |
| `pl.INT8` / `pl.UINT8` | 8 | |
| `pl.INT16` / `pl.UINT16` | 16 | |
| `pl.INT32` / `pl.UINT32` | 32 | |
| `pl.INT64` / `pl.UINT64` | 64 | |
| `pl.FP16` | 16 | IEEE half |
| `pl.BF16` | 16 | Brain float |
| `pl.FP32` | 32 | IEEE single |
| `pl.FP4` | 4 | Packed MXFP4 E2M1×2 |
| `pl.FP8E4M3FN` / `pl.FP8E5M2` | 8 | MXFP8 data formats |
| `pl.FP8E8M0` | 8 | MX block-scale exponent |
| `pl.HF4` / `pl.HF8` | 4 / 8 | Hisilicon float formats |
| `pl.INDEX` | 64 | Index arithmetic — loop variables, dimensions |
| `pl.TASK_ID` | — | Producer handle for a launched task |

`dtype.get_byte()` returns the byte-addressable element size, rounded up to one byte.
Use it for byte-addressable dtypes whenever a byte count is computed rather than written
as a literal. Do not use `logical_elements * dtype.get_byte()` for a 4-bit buffer: PyPTO
packs every semantic 4-bit dtype two logical elements per byte, so the physical size is
`ceil(logical_elements / 2)`.

```python
nbytes = 256 * pl.FP32.get_byte()          # 1024, not 256
```

FP4 shapes inside PyPTO IR are logical nibble shapes, and `valid_shape` uses the same
logical units. At the Torch/runtime boundary, `torch.float4_e2m1fn_x2` uses a physical x2
carrier shape: its last dimension contains one byte per two logical FP4 values. JIT expands
that last dimension on entry, while compiled-call metadata and orchestration allocations
contract it by two; no separate `storage_shape` is stored in `TensorType` or `TileType`. Packed
FP4 requires a positive even logical last dimension, including static allocation and view
shapes; dynamic widths are checked before conversion. A 4-bit slice origin must land on a byte
boundary, so an odd linear nibble offset is rejected.

End-to-end 4-bit execution is backend-gated. Ascend950 supports `pl.FP4`; `INT4`, `UINT4`,
and `HF4` remain storage-accounted but are rejected by in-core codegen. Ascend910B/A2A3
rejects every 4-bit in-core dtype because its isolated FP16↔INT4 conversion has no matching
packed load/store carrier ABI.

### Container types

| Type | Lives in | Written as |
| ---- | -------- | ---------- |
| `pl.Tensor[[shape], dtype]` | DDR | `x: pl.Tensor[[64, 128], pl.FP32]` |
| `pl.Tile[[shape], dtype]` | On-chip buffer (Vec by default) | `t: pl.Tile[[64, 64], pl.FP32]` |
| `pl.Scalar[dtype]` | Value, not a buffer | `s: pl.Scalar[pl.FP32]` |
| `pl.Array[extent, dtype]` | On-core array | `a: pl.Array[16, pl.INT32]` |
| `pl.Tuple[T1, T2]` | — | Multi-value return annotation |

`pl.TaskId` is a convenience alias for `pl.Scalar[pl.TASK_ID]`.

`pl.Array` is normally created rather than annotated — arrays do not cross function
boundaries, so the annotation form is rare. The update is functional — it produces a new
array value and rebinds the name, so an array assignment inside a loop is a carried value
like any other.

```python
arr = pl.array.create(16, pl.INT32)
arr[i] = value          # array.update_element — functional, rebinds arr
x = arr[i]              # array.get_element
```

### Layouts

**Write `pl.Tensor` annotations with the runtime row-major shape and no layout marker.**
Layout is an IR-internal concern; passes derive it from the operations that produce and
consume each view.

```python
b: pl.Tensor[[N, K], pl.FP32]              # ✅ source shape, no marker
```

The layout-only shorthand `pl.Tensor[..., pl.DN]` is not supported: it raises
`ParserTypeError`. For a transposed matmul operand, pass `a_trans=True` / `b_trans=True` to
`pl.matmul`, or derive the transposed view at the use site with `pl.transpose(x, -2, -1)`.
A slice or reshape of a DN-producing operation inherits DN automatically.

`pl.ND` is the default row-major layout and never needs writing. `pl.NZ` asserts that the
tensor's bytes in global memory are *already* stored in PTO-native NZ fractal order, so a
matmul weight load can skip the online ND→NZ conversion. It is an assertion about existing
bytes, not a request to convert: the shape and slicing you write stay logical, and the
compiler derives the blocked physical descriptor. It currently requires a statically shaped,
fractal-aligned tensor with a whole-byte dtype (`shape[-2] % 16 == 0`,
`shape[-1] % (256 / dtype bits) == 0`) read
into a matmul operand; anything else is rejected with a diagnostic.

When a tensor's rows are not contiguous — a window into a larger buffer, a strided slice
handed in from outside — describe it with `pl.TensorView`, which makes the strides explicit
instead of leaving them to be inferred:

```python
view = pl.TensorView(stride=[1024, 1], layout=pl.TensorLayout.ND, valid_shape=[16, 64])
```

`layout=` is required whenever any of `stride`, `valid_shape`, or `pad` is given.
`pl.TensorLayout` is the enum those layout constants come from — `pl.ND` is
`pl.TensorLayout.ND`.

`pl.MX_A_ZZ` and `pl.MX_B_NN` are the two remaining layout constants. They tag the **GM
scale tensor** of an MX (microscaling) operand on Ascend950 — `MX_A_ZZ` for the left/A
scale pack, `MX_B_NN` for the right/B one — so that a Mat-to-scale `pl.move` can check the
source layout instead of byte-copying incompatible data into `LeftScale` / `RightScale`.
They are the one case where a layout marker on a `pl.Tensor` annotation is required rather
than discouraged. Current limitations: an MX `pl.load` must pass `target_memory=pl.Mem.Mat`
explicitly, MX subviews (`slice`, `reshape`, `transpose`, `reinterpret_view`, `view`) and
MX `remote_load` are rejected. The matmul itself is `pl.matmul_mx` and its `_acc` /
`_bias` variants, which take a data tile and a scale tile per operand. Both data tiles reaching
the op must be `FP8E4M3FN`. The supported FP4-input form is a left FP4 operand multiplied by a
right FP8 operand: write `pl.cast(fp4_tile, pl.FP8E4M3FN)` before `matmul_mx`. On A5 the cast
legalization pass expands that request to FP4→BF16→FP32→FP8E4M3FN. Native FP4×FP4 and the
reverse FP8×FP4 form are not supported; MXFP4 quantization is not exposed yet.

### Dynamic shapes

`pl.dynamic(name)` marks an axis whose extent is **not known when the kernel is compiled
and may differ from one launch to the next** — a batch that varies with the request, a
sequence length that grows as decoding proceeds. The extent becomes a run-time value, so
one compiled program serves every size that axis takes: dynamic dimensions collapse to
`None` in the JIT cache key, and changing the extent does not trigger a recompile.

<!-- doctest: run -->
```python
N = pl.dynamic("N")
TILE = 32                       # the physical tile the kernel moves per call


@pl.jit.incore
def rows(x: pl.Tensor[[N, 64], pl.FP32], out: pl.Out[pl.Tensor[[N, 64], pl.FP32]]):
    out = pl.store(pl.mul(pl.load(x, [0, 0], [TILE, 64]), 2.0), [0, 0], out)
    return out


@pl.jit
def drive(x: pl.Tensor[[N, 64], pl.FP32], out: pl.Out[pl.Tensor[[N, 64], pl.FP32]]):
    return rows(x, out)         # the entry is dynamic too, so both extents share it


# Two extents through one program: the dynamic dim collapses to None in the JIT
# cache key, so the second call is not a recompile.
for extent in (TILE, 3 * TILE):
    x = torch.randn(extent, 64, dtype=torch.float32)
    out = torch.zeros(extent, 64, dtype=torch.float32)
    drive(x, out, config=CFG)
    torch.testing.assert_close(out[:TILE], x[:TILE] * 2.0, rtol=1e-4, atol=1e-4)
```

The entry has to stay dynamic for that to hold. Giving it a concrete shape pins the program
to one extent, and doing orchestration-level work on a dynamic-shaped tensor (rather than
handing it to an InCore kernel) fails earlier still, in `InitMemRef`, which needs a constant
dim. The kernel above moves a fixed `TILE`; covering the whole of a larger input is the
chunking loop from [Control Flow](02-control-flow.md).

The same `DynVar` object used in several annotations refers to the same dimension — reuse
the object, do not create a second one with the same name if you mean the same value.

Keep a dimension static when it really is fixed. A static extent is a number the compiler
can plan around — tiling choices, unroll factors, static bound checks — and a dynamic one
withholds it.

### Parameter directions

| Direction | Syntax | The compiler concludes |
| --------- | ------ | ---------------------- |
| In (default) | `x: pl.Tensor[...]` | Read-only. Orders after producers |
| Out | `x: pl.Out[pl.Tensor[...]]` | Written, not read. Orders after prior readers and writers |
| InOut | `x: pl.InOut[pl.Tensor[...]]` | Both. Orders against everything touching it |

Directions are what the compiler reads to order tasks against each other. Declaring an
`InOut` buffer as `Out` tells the runtime nothing needs to finish before this task writes
it — which is a race, not a diagnostic.

## Edge Cases

> **Fatal pitfall:** a byte count written as an element count silently under-allocates.
> `pld.alloc_window_buffer(256)` reserves 256 **bytes** — room for 64 FP32 values, not
> 256. Any non-literal size must be spelled `n * pl.<DTYPE>.get_byte()`. Nothing warns;
> the symptom is corrupted data past the first 64 elements.

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| **`ParserTypeError` about the DN layout-only shorthand** | `pl.Tensor[..., pl.DN]` — removed, it forced two coordinate systems onto one annotation | Write the source shape with no marker; derive DN at the use site with `pl.transpose(x, -2, -1)`; or inherit it through a slice/reshape of a DN-producing op |
| **Results wrong only when two tasks overlap** | A read-write buffer declared `In` or `Out` instead of `InOut` | Declare the direction the kernel actually performs |
| **Reading an `Out` parameter returns garbage** | `Out` promises write-before-read | Use `pl.InOut[...]` if the prior contents matter |
| **`pl.cast` where you expected implicit promotion** | There is no implicit promotion | Insert the cast; check [LegalizeTileCast](../../dev/passes/15-legalize_tile_cast.md) for multi-hop pairs |
| **Two dimensions that should match are treated as independent** | Two separate `pl.dynamic("M")` calls | Create the `DynVar` once and reuse the object |

Not every `pl.cast` is one instruction. Whether a `(src, dst)` pair maps to a single
hardware `pto.tcvt` or expands into a chain depends on the target: `INT32 -> FP16` is one
instruction on Ascend910B and lowers to `INT32 -> FP32 -> FP16` on Ascend950. Each hop
costs a `tcvt`, and where an intermediate is narrower than the source the result can
differ from a directly rounded conversion by one ULP of the destination. This is expected
behaviour, not a defect — see
[LegalizeTileCast](../../dev/passes/15-legalize_tile_cast.md) for the per-architecture
tables.

## See Also

- [Functions and Programs](01-functions.md) — where these annotations appear, and what a signature means to the caller.
- [Memory and Data Movement](03-memory.md) — moving data between the spaces these types name.
- [Operations](../ops/index.md) — which operators accept `Tensor` versus `Tile`.
- [IR Types](../../dev/ir/02-types.md) — the IR-level type system these annotations build.
- [LegalizeTileCast](../../dev/passes/15-legalize_tile_cast.md) — per-architecture cast expansion and its precision consequences.
