# Register operations

Register operations describe logical vector and mask values within a non-nested
`pl.reg_scope()` on A5 (Ascend950). The frontend accepts an unspecialized InCore
function; after `ExpandMixedKernel`, a surviving region must be in an AIV
function. An explicitly configured non-A5 target is rejected. Without a selected
target, target validation is deferred until compilation selects one.

An explicit InCore scope in an Opaque, Orchestration, or Graph function also
provides a device context before outlining, including when a graph scope is an
ancestor. RegScope directly in orchestration code remains invalid. Orchestration
scopes inside RegScope remain forbidden; existing InCore scope rules are unchanged.

```python
@pl.function(type=pl.FunctionType.InCore)
def kernel(
    src: pl.Tile[[128], pl.FP32, pl.MemorySpace.Vec],
    dst: pl.Tile[[128], pl.FP32, pl.MemorySpace.Vec],
):
    with pl.reg_scope():
        x = pl.reg.load(src, offset=0, lanes=64)
        mask = pl.reg.active_mask(32, lanes=64)
        y = pl.reg.add(x, x, mask=mask, pmode="zero")
        out = pl.reg.store(dst, offset=0, value=y, mask=mask)
    z = pl.tile.add(out, out)
```

The scope has no `captures` argument, result list or yield. External Tile and
read-only Scalar values are implicit inputs. Reg values and local Scalars belong
to their defining region and cannot escape through subsequent uses, function
returns, loop results or assignments to an outer variable. Tile inputs originate
outside the region; only `reg.store` creates a Tile result inside it.
Wrapping a local value in a tuple does not permit it to escape; nested tuple
types are checked at function boundaries too.

Every register operation is a `Call` assigned with `AssignStmt`. A bare store is
rejected. Store operand order is `(tile, offset, value, mask?)`; its result has
exactly the destination Tile type and reuses operand 0 storage. This is an alias,
not a snapshot or a new allocation. Store does not enlarge `valid_shape`.

## P0 schemas

| Operations | Inputs | Result |
| ---------- | ------ | ------ |
| `load` | Vec Tile, element offset; `lanes` | VecReg |
| `store` | Vec Tile, element offset, VecReg, optional MaskReg | destination Tile |
| `broadcast`, `index_gen` | Scalar; `lanes`, `dtype` | VecReg |
| `full_mask` | `lanes` | MaskReg |
| `active_mask` | static active lane count; `lanes` | MaskReg |
| `add/sub/mul/div/minimum/maximum` | same-type VecRegs, optional MaskReg; `pmode` | VecReg |
| `convert` | VecReg; `dtype`, optional `rounding`, `saturate` | VecReg |
| `compare` | same-type VecRegs, seed MaskReg; `predicate` | MaskReg |
| `select` | MaskReg, same-type true/false VecRegs | VecReg |
| `mask_and/or/xor/not` | same-lane MaskRegs | MaskReg |

All lanes are static positive integers. Vec operands require identical dtype,
signedness and lane count; there is no promotion or lane broadcasting. Masks use
BOOL and must have the same lane count as controlled data. Arithmetic `pmode` is
`"zero"`; `"merge"` is unsupported. `index_gen` order is `"ascending"`.
Comparison predicates are `"eq"`, `"ne"`, `"lt"`, `"le"`, `"gt"`, `"ge"`.

The common dtype set is FP16/BF16/FP32 and signed/unsigned 8/16/32-bit integers.
`mul` excludes 8-bit integers; `div` accepts FP16/FP32 only; `compare` accepts
FP16/FP32/INT32/UINT32; `select` excludes BF16. Existing Scalar SSA values must
match the requested generator dtype. DSL literals are normalized to that dtype;
raw IR and Builder literals must already have the requested dtype.

Conversion pairs are FP16→FP32, FP32→FP16, FP32→INT32, INT32→FP32,
INT8→FP16, INT8→INT16, INT16→INT32, UINT8→UINT16, UINT16→UINT32 and
INT32→INT8. Other pairs are rejected. Same-dtype DSL conversion returns the
original value when no attributes are supplied. FP32→FP16 requires explicit
`saturate` and accepts rounding R/A/H/Z (default R); FP32→INT32 requires explicit
`saturate` and accepts R/A/F/C/Z (default Z). INT32→INT8 requires
`saturate=False` and no rounding. Other pairs accept neither attribute.

## Memory checks and effects

Offsets count elements and have INDEX dtype. Static offsets must be nonnegative
and the entire interval `[offset, offset + lanes)` must fit the Tile capacity,
even for masked stores. Loads additionally require the whole read interval to
lie in initialized `valid_shape`; a mask on a later operation cannot protect an
uninitialized load. One-dimensional Tiles are directly supported. Multidimensional
Tiles require an explicit contiguous row-major View; padded, boxed or compact
layouts are rejected. Static View offsets contribute to absolute effect ranges.

`ir.get_reg_op_spec(name)` reports the scope requirement and future VMI emitter.
`ir.get_reg_memory_effect(call)` reports the target, root allocation when known,
Read/Write access, absolute element interval and optional mask. Dynamic offsets
produce unknown endpoints, never an empty effect. They do not prove bounds and
require later analysis before emission. A dynamic offset load requires complete
valid-shape coverage. A mask is preserved without claiming whole-buffer
initialization or disjointness.

PR2 supplies schema effects and frontend validation only. SSA propagation,
initialization tracking across statements, alias-aware scheduling and VMI
emission belong to PR3. PTO codegen currently rejects RegScope with a stable
unsupported-backend diagnostic, regardless of whether PTOAS is installed.

## Builder

Python uses `with ib.reg_scope():`, `ib.emit_reg(name, op_name, args, kwargs)`
and `ib.emit_reg_store(name, tile, value, offset=0, mask=None)`. C++ exposes
`BeginRegScope`, `EmitReg`, `EmitRegStore` and `EndRegScope`. Both use the same
schema deduction as the DSL; completing a function validates region ownership.
`RegScopeValid` also validates manually constructed functions through the normal
property verifier. Final AIV placement is checked by `MixedKernelExpanded`.

The Python text syntax accepts no scope metadata. Printing a manually constructed
RegScope with a nonempty `name_hint` or `attrs` raises an error rather than losing
those fields; binary serialization preserves them. Empty scopes print as a `pass`
block.
