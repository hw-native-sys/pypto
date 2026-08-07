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

```python
import pypto.language as pl

M = pl.dynamic("M")                       # an axis whose extent varies at run time

@pl.jit.incore
def scale_rows(
    x: pl.Tensor[[M, 128], pl.FP16],                    # In (default): read-only, DDR
    acc: pl.InOut[pl.Tensor[[M, 128], pl.FP32]],        # read-write, DDR
    out: pl.Out[pl.Tensor[[M, 128], pl.FP32]],          # write-only, DDR
    factor: pl.Scalar[pl.FP32],                         # scalar, passed by value
):
    ...
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

`dtype.get_byte()` returns the element size in bytes. Use it whenever a byte count is
computed rather than written as a literal — a raw element count passed where bytes are
expected is a silent under-allocation.

```python
nbytes = 256 * pl.FP32.get_byte()          # 1024, not 256
```

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

`pl.ND` is the default row-major layout and never needs writing. `pl.NZ` is tile-only — a
hardware tile layout, never a `pl.Tensor` annotation.

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
`_bias` variants, which take a data tile and a scale tile per operand.

### Dynamic shapes

`pl.dynamic(name)` marks an axis whose extent is **not known when the kernel is compiled
and may differ from one launch to the next** — a batch that varies with the request, a
sequence length that grows as decoding proceeds. The extent becomes a run-time value, so
one compiled program serves every size that axis takes: dynamic dimensions collapse to
`None` in the JIT cache key, and changing the extent does not trigger a recompile.

```python
M = pl.dynamic("M")

@pl.jit.incore
def rows(x: pl.Tensor[[M, 64], pl.FP32], out: pl.Out[pl.Tensor[[M, 64], pl.FP32]]):
    ...
```

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
