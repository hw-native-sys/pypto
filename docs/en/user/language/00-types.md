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

M = pl.dynamic("M")                       # symbolic dimension, fixed per compilation

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
| `pl.TASK_ID` | — | Producer handle from `pl.submit`; see [Scopes and Tasks](04-scopes-and-tasks.md) |

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
boundaries, so the annotation form is rare. See [Directives § arrays](05-directives.md#arrays).

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

```python
b: pl.Tensor[[K, N], pl.FP32, pl.DN]       # ⚠️ deprecated — DeprecationWarning at parse time
```

The DN shorthand is deprecated because it forces you to hold two coordinate systems in
your head at once: the IR-logical post-view shape and the runtime row-major shape. For a
transposed matmul operand, pass `a_trans=True` / `b_trans=True` to `pl.matmul`, or load
naturally and apply `pl.tile.transpose_view(...)`. A slice of a DN-producing operation
inherits the parent's layout automatically.

`pl.ND` is the default row-major layout and never needs writing. `pl.NZ` is tile-only — a
hardware tile layout, never a `pl.Tensor` annotation. To build a
DN tensor deliberately at the IR level (test fixtures, round-tripping printed IR), prefer
`pl.TensorView(stride=[...], layout=pl.TensorLayout.DN)`, which forces the stride to be
explicit and avoids the implicit coordinate flip.

### Dynamic shapes

`pl.dynamic(name)` creates a symbolic dimension. The same `DynVar` object used in several
annotations refers to the same dimension — reuse the object, do not create a second one
with the same name if you mean the same value.

```python
M = pl.dynamic("M")

@pl.jit.incore
def rows(x: pl.Tensor[[M, 64], pl.FP32], out: pl.Out[pl.Tensor[[M, 64], pl.FP32]]):
    ...
```

What you give up: any decision the compiler would have made from the extent. Tiling
choices, unroll factors, and static bound checks all lose information, so make a dimension
dynamic because it genuinely varies per launch — not to avoid writing a number down.

### Parameter directions

| Direction | Syntax | The compiler concludes |
| --------- | ------ | ---------------------- |
| In (default) | `x: pl.Tensor[...]` | Read-only. Orders after producers |
| Out | `x: pl.Out[pl.Tensor[...]]` | Written, not read. Orders after prior readers and writers |
| InOut | `x: pl.InOut[pl.Tensor[...]]` | Both. Orders against everything touching it |

Directions are what `DeriveCallDirections` propagates and what
`AutoDeriveTaskDependencies` reads to build the dependency graph. Declaring an `InOut`
buffer as `Out` tells the runtime nothing needs to finish before this task writes it —
which is a race, not a diagnostic.

To assert that a specific argument should *not* participate in dependency tracking (for
example, sibling tasks writing disjoint regions), use `pl.no_dep(t)` at the call site
rather than weakening the direction. See
[Scopes and Tasks § opting out](04-scopes-and-tasks.md#opting-out-of-dependency-tracking).

## Edge Cases

> **Fatal pitfall:** a byte count written as an element count silently under-allocates.
> `pld.alloc_window_buffer(256)` reserves 256 **bytes** — room for 64 FP32 values, not
> 256. Any non-literal size must be spelled `n * pl.<DTYPE>.get_byte()`. Nothing warns;
> the symptom is corrupted data past the first 64 elements.

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| **`DeprecationWarning` at parse time about layout** | `pl.Tensor[..., pl.DN]` annotation | Drop the marker, write the runtime shape, pass `b_trans=True` to `pl.matmul` |
| **Shape mismatch the numbers seem to satisfy** | A DN annotation flipped the coordinate system | Write the source shape; check whether the consumer wanted a `transpose_view` |
| **Results wrong only when two tasks overlap** | A read-write buffer declared `In` or `Out` instead of `InOut` | Declare the direction the kernel actually performs |
| **Reading an `Out` parameter returns garbage** | `Out` promises write-before-read | Use `pl.InOut[...]` if the prior contents matter |
| **`pl.cast` where you expected implicit promotion** | There is no implicit promotion | Insert the cast; check [LegalizeTileCast](../../dev/passes/14-legalize_tile_cast.md) for multi-hop pairs |
| **Two dimensions that should match are treated as independent** | Two separate `pl.dynamic("M")` calls | Create the `DynVar` once and reuse the object |

Not every `pl.cast` is one instruction. Whether a `(src, dst)` pair maps to a single
hardware `pto.tcvt` or expands into a chain depends on the target: `INT32 -> FP16` is one
instruction on Ascend910B and lowers to `INT32 -> FP32 -> FP16` on Ascend950. Each hop
costs a `tcvt`, and where an intermediate is narrower than the source the result can
differ from a directly rounded conversion by one ULP of the destination. This is expected
behaviour, not a defect — see
[LegalizeTileCast](../../dev/passes/14-legalize_tile_cast.md) for the per-architecture
tables.

## See Also

- [Functions and Programs](01-functions.md) — where these annotations appear, and what a signature means to the caller.
- [Memory and Data Movement](03-memory.md) — moving data between the spaces these types name.
- [Operations](../ops/index.md) — which operators accept `Tensor` versus `Tile`.
- [IR Types](../../dev/ir/02-types.md) — the IR-level type system these annotations build.
- [LegalizeTileCast](../../dev/passes/14-legalize_tile_cast.md) — per-architecture cast expansion and its precision consequences.
