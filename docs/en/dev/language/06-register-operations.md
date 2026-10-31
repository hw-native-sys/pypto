# Register API reference

`pl.reg` provides logical vector and mask operations inside `pl.reg_scope()` on
A5 (Ascend950). This page lists every supported `pl.reg` operation, its parameters,
return type and dtype restrictions. All names in the tables have the `pl.reg.` prefix.

## Scope and register values

```python
import pypto.language as pl

@pl.function(type=pl.FunctionType.InCore)
def kernel(
    src: pl.Tile[[128], pl.FP32, pl.MemorySpace.Vec],
    dst: pl.Tile[[128], pl.FP32, pl.MemorySpace.Vec],
):
    with pl.reg_scope():
        x = pl.reg.load(src, lanes=64)
        mask = pl.reg.active_mask(32, lanes=64)
        y = pl.reg.add(x, x, mask=mask)
        out = pl.reg.store(dst, value=y, mask=mask)
    z = pl.tile.add(out, out)
```

`pl.reg_scope()` has no arguments, captures, result list or yield. Scopes cannot
nest. The frontend accepts InCore and AIV device contexts, including an explicit
InCore scope inside an Opaque, Orchestration or Graph function before outlining.
After `ExpandMixedKernel`, the region must belong to an AIV function. Direct use
in orchestration code and orchestration scopes inside RegScope are rejected.
Explicit non-A5 targets are rejected; target checks wait until compilation if no
target has been selected.

`pl.Reg[lanes, dtype]` describes a vector register (`VecReg`).
`pl.Reg[lanes, pl.BOOL, pl.RegKind.MaskReg]` describes a predicate mask (`MaskReg`).
The optional third annotation field is the register kind; its default is
`pl.RegKind.VecReg`. Register values are normally inferred from operations.
`lanes` is a static positive integer, not a tensor shape or a byte count.

External Tiles and read-only Scalars are implicit inputs. Registers and local
Scalars cannot escape their defining region through later uses, function returns,
loop results, tuples or assignments to outer variables. Tile operands must
originate outside the region; `store` returns an alias usable after the region.
Assign every operation result to a variable, including `store`; a bare store is
rejected. For loop APIs and hardware-loop policy, see [statements](01-statements.md).

## Common parameters and dtype notation

| Parameter | Meaning |
| --------- | ------- |
| `value`, `lhs`, `rhs` | Input VecRegs. Binary vector operands have identical dtype, signedness and lane count. |
| `mask` | Governing BOOL MaskReg with the same lanes as the input vector. Optional only where the signature has `mask=None`. |
| `seed` | Required governing MaskReg for comparison; inactive seed lanes produce false result lanes. |
| `scalar` | Scalar SSA value or numeric literal. Existing SSA dtype must match the required scalar dtype; there is no implicit cast. |
| `dtype` | A dtype constant such as `pl.FP32`; specifies generator output or conversion destination dtype. |
| `lanes` | Static positive logical element count, supplied by keyword for generators and loads. |
| `pmode` | Inactive-result policy. Only `"zero"` is accepted, so inactive vector lanes become zero. |

`*` in a signature starts keyword-only parameters. Omitted optional masks mean
all lanes participate. Except for conversion and reduction, results preserve the
input vector dtype and lanes. There is no implicit vector promotion or lane
broadcasting. DSL numeric literals are normalized to the requested scalar dtype;
Python `bool` is not accepted as an integer argument. Raw IR and Builder operands
must already have the required dtype.

The tables use the following exact dtype sets:

| Set | Dtypes |
| --- | ------ |
| `F` | FP16, FP32 |
| `I` | INT8, UINT8, INT16, UINT16, INT32, UINT32 |
| `S` | INT8, INT16, INT32 |
| `I16_32` | INT16, UINT16, INT32, UINT32 |
| `T` | FP16, BF16, FP32, INT8, UINT8, INT16, UINT16, INT32, UINT32 |
| `C` | FP16, FP32, INT32, UINT32 |

`+` denotes set union. BOOL is used only for masks; INDEX is used for offsets
and active-lane counts. No table entry implies support for other dtypes.

## Memory operations

| API signature | Parameters and behavior | Result | Dtypes |
| ------------- | ----------------------- | ------ | ------ |
| `load(tile, offset=0, *, lanes)` | `tile`: source Tile in Vec UB memory. `offset`: starting element index (INDEX Scalar or integer literal). Reads `lanes` consecutive elements. | VecReg with `lanes` and Tile dtype | `T` |
| `store(tile, offset=0, value=None, mask=None)` | `tile`: destination Vec Tile. `offset`: starting element index. `value`: required VecReg despite the Python default. `mask`: optional write mask. Only active lanes are written; other destination elements are preserved. | Tile alias of `tile` | `T`; value and Tile dtype must match |

Offsets count **elements**, not bytes. Static offsets must be nonnegative and the
whole interval `[offset, offset + lanes)` must fit the Tile capacity, including
for masked stores. Static loads must lie inside initialized `valid_shape`; a later
mask cannot protect an uninitialized load. One-dimensional Tiles are directly
supported. Multidimensional Tiles require an explicit contiguous row-major View;
padded, boxed or compact layouts are rejected. View start offsets are included in
address and allocation-bound checks.

Dynamic offsets and valid extents pass through when static checks cannot prove a
violation. The caller owns their runtime bounds and initialization. A store does
not enlarge `valid_shape`, allocate storage or create a snapshot: its result
aliases the destination and has exactly the destination Tile type.

## Vector and mask generators

| API signature | Parameters and behavior | Result | Dtypes |
| ------------- | ----------------------- | ------ | ------ |
| `broadcast(scalar, *, lanes, dtype)` | Repeat `scalar` into every lane. Scalar dtype must equal `dtype`. | VecReg | `T` |
| `index_gen(base, *, lanes, dtype, order="ascending")` | `base`: scalar start value; result lane `i` is `base + i`. `order` accepts only `"ascending"`. Base dtype must equal `dtype`. | VecReg | `F + I`; BF16 is rejected during type deduction, before codegen |
| `full_mask(*, lanes)` | Activate every logical lane. | BOOL MaskReg | BOOL |
| `active_mask(active_lanes, *, lanes)` | Activate the prefix `[0, active_lanes)`. `active_lanes` is a static INDEX integer in `[0, lanes]`; zero creates an all-false mask. Dynamic counts are unsupported. | BOOL MaskReg | BOOL |

## Vector arithmetic

Each API below has signature `(lhs, rhs, mask=None, *, pmode="zero")` and returns
a VecReg. `lhs` and `rhs` are same-type vectors; an optional `mask` governs both
inputs and zeroes inactive result lanes.

| API | Active-lane calculation | Dtypes |
| --- | ----------------------- | ------ |
| `add` | `lhs + rhs` | `T` |
| `sub` | `lhs - rhs` | `T` |
| `mul` | `lhs * rhs` | FP16, BF16, FP32, INT16, UINT16, INT32, UINT32 |
| `div` | `lhs / rhs` | `F` |
| `minimum` | Elementwise minimum | `T` |
| `maximum` | Elementwise maximum | `T` |

## Unary arithmetic

Each API below has signature `(value, mask=None, *, pmode="zero")` and returns
a VecReg. `value` is the input vector; `mask` is optional. These APIs do not
accept BF16 input.

| API | Active-lane calculation | Dtypes |
| --- | ----------------------- | ------ |
| `abs` | Absolute value | `F + S` |
| `neg` | `-value` | `F + I` |
| `relu` | `max(value, 0)` | FP16, FP32, INT32 |
| `exp` | Natural exponential `exp(value)` | `F` |
| `log` | Natural logarithm `ln(value)` | `F` |
| `sqrt` | Square root | `F` |

## Integer bitwise operations and vector shifts

The binary signature is `(lhs, rhs, mask=None, *, pmode="zero")`.
`bitwise_not` has signature `(value, mask=None, *, pmode="zero")`.
All return VecRegs, accept `I` only, and use optional same-lane masks.
For vector shifts, `rhs` supplies a separate integer shift amount for each lane
and must match `lhs` in dtype and lanes.

| API | Active-lane calculation | Dtypes |
| --- | ----------------------- | ------ |
| `bitwise_and` | Integer bitwise AND | `I` |
| `bitwise_or` | Integer bitwise OR | `I` |
| `bitwise_xor` | Integer bitwise XOR | `I` |
| `bitwise_not` | Integer bitwise complement | `I` |
| `shift_left` | Left shift | `I` |
| `shift_right` | Arithmetic right shift for signed input, logical right shift for unsigned input | `I` |

These are vector operations. For boolean predicate masks, use `mask_and`,
`mask_or`, `mask_xor` and `mask_not` below.

## Vector-scalar arithmetic and shifts

Each API below has signature `(value, scalar, mask, *, pmode="zero")` and returns
a VecReg. `value` is the vector, `scalar` is applied to every lane, and `mask` is
**required**. Unlike these APIs, `broadcast` creates a separate vector value.

| API | Active-lane calculation | Vector dtypes | Scalar dtype |
| --- | ----------------------- | ------------- | ------------ |
| `adds` | `value + scalar` | `F + I` | Same as vector |
| `muls` | `value * scalar` | `F + I16_32` | Same as vector |
| `maximums` | `max(value, scalar)` | `F + I` | Same as vector |
| `minimums` | `min(value, scalar)` | `F + I` | Same as vector |
| `shift_lefts` | Left shift each lane by `scalar` | `I` | INT16 |
| `shift_rights` | Signedness-aware right shift each lane by `scalar` | `I` | INT16 |

Scalar-shift amounts are INT16 in PyPTO and emitted as PTOAS signless `i16`.
Existing INT32 SSA shift amounts are rejected instead of being silently narrowed.
None of these vector-scalar APIs accepts BF16 input.

## Comparison, selection and mask logic

| API signature | Parameters and behavior | Result | Dtypes |
| ------------- | ----------------------- | ------ | ------ |
| `compare(lhs, rhs, seed, *, predicate)` | Compare same-type vectors under required `seed`. `predicate` chooses the comparison. | BOOL MaskReg with input lanes | `C` |
| `compares(value, scalar, seed, *, predicate)` | Compare each lane to `scalar`, whose dtype matches `value`, under required `seed`. | BOOL MaskReg with input lanes | `C` |
| `select(mask, true_value, false_value)` | Lane `i` is `true_value[i]` when `mask[i]` is true, else `false_value[i]`. True/false vectors must match in dtype and lanes. | VecReg | `F + I` |
| `gather_lanes(value, indices)` | Lane `i` is `value[indices[i]]`. `indices` is an integer vector with the same storage width and lanes as `value`; runtime indices must be within source lanes. | VecReg | `F + I`, with lane restrictions below |
| `mask_and(lhs, rhs)` | Boolean AND of same-lane masks | BOOL MaskReg | BOOL |
| `mask_or(lhs, rhs)` | Boolean OR of same-lane masks | BOOL MaskReg | BOOL |
| `mask_xor(lhs, rhs)` | Boolean XOR of same-lane masks | BOOL MaskReg | BOOL |
| `mask_not(value)` | Boolean inversion of a mask | BOOL MaskReg | BOOL |

`predicate` is required and accepts exactly `"eq"` (equal), `"ne"` (not equal),
`"lt"` (less), `"le"` (less or equal), `"gt"` (greater), or `"ge"` (greater or
equal). Integer comparisons follow the vector dtype signedness. A comparison's
seed has the same lane count as its input and result. Mask logic has no additional
predication mask or `pmode`; `select` uses both value operands and its mask is the
selection condition, not an inactive-lane policy.

`gather_lanes` maps to PTOAS `vselr`, a register permutation, and does not access
memory. Its supported source/index lane counts depend on element storage width:

| Element width | Supported lanes | Index dtypes |
| ------------- | --------------- | ------------ |
| 8 bits | 64, 128, 256 | INT8, UINT8 |
| 16 bits | 64, 128 | INT16, UINT16 |
| 32 bits | 64 | INT32, UINT32 |

## Reductions

| API signature | Parameters and behavior | Result | Dtypes |
| ------------- | ----------------------- | ------ | ------ |
| `reduce_sum(value, mask, *, reassoc=False)` | Sum active lanes. `reassoc=True` permits hardware tree reassociation and is required for floating input; integers accept either value. | One-lane VecReg, same dtype | `F + I16_32` |
| `reduce_max(value, mask)` | Maximum of active lanes | One-lane VecReg, same dtype | `F + I16_32` |
| `reduce_min(value, mask)` | Minimum of active lanes | One-lane VecReg, same dtype | `F + I16_32` |

`value` is the source vector; required `mask` has the **source** lane count, not
one lane. Inactive inputs contribute 0 for sum, the dtype minimum (or negative
infinity) for max, and the dtype maximum (or positive infinity) for min. No
implicit widening is performed. Results remain local registers; use `store` to
write the single output element. Grouped reductions are not exposed.

```python
with pl.reg_scope():
    x = pl.reg.load(src, lanes=64)
    mask = pl.reg.active_mask(32, lanes=64)
    y = pl.reg.adds(x, 1.0, mask)
    positive = pl.reg.compares(y, 0.0, mask, predicate="gt")
    selected = pl.reg.select(positive, y, x)
    total = pl.reg.reduce_sum(selected, mask, reassoc=True)
    out = pl.reg.store(dst, value=total)
```

## Dtype conversion

`convert(value, *, dtype, rounding=None, saturate=None)` returns a VecReg with the
same lanes and destination `dtype`. `value` is the source VecReg. `rounding`
selects rounding for supported narrowing conversions; `saturate` explicitly
chooses saturation. They are not universal flags: only the pairs below accept them.

| Source | Destination | `rounding` | `saturate` |
| ------ | ----------- | ---------- | ---------- |
| FP16 | FP32 | Must be omitted | Must be omitted |
| FP32 | FP16 | R/A/H/Z; DSL default R | Required: True or False |
| FP32 | INT32 | R/A/F/C/Z; DSL default Z | Required: True or False |
| INT32 | FP32 | Must be omitted | Must be omitted |
| INT8 | FP16 | Must be omitted | Must be omitted |
| INT8 | INT16 | Must be omitted | Must be omitted |
| INT16 | INT32 | Must be omitted | Must be omitted |
| UINT8 | UINT16 | Must be omitted | Must be omitted |
| UINT16 | UINT32 | Must be omitted | Must be omitted |
| INT32 | INT8 | Must be omitted | Required: False |

R rounds to nearest with ties to even; A rounds away from zero; F rounds
toward negative infinity; C toward positive infinity; Z toward zero. H selects
the PTOAS-specific H rounding mode for FP32-to-FP16 narrowing and is forwarded
to the backend unchanged.
`saturate=True` selects saturation (clamping to the target range);
`saturate=False` selects non-saturating conversion.
Same-dtype DSL conversion returns the original register if both optional
attributes are omitted. Other dtype pairs are rejected, including BF16
conversions. Raw IR/Builder calls must supply normalized rounding explicitly
where required; the defaults above are supplied by the DSL wrapper.

## IR Builder and inspection

| API | Meaning |
| --- | ------- |
| `with ib.reg_scope(span=None):` | Open a register region in Python IRBuilder. |
| `ib.emit_reg(name, op_name, args, kwargs=None, span=None)` | Create a named register operation. `op_name` is the registered name such as `"reg.add"`; `args` contains IR operands, `kwargs` operation attributes. |
| `ib.emit_reg_store(name, tile, value, offset=0, mask=None, span=None)` | Emit a named store and return the destination Tile alias. |
| `ir.get_reg_op_spec(name)` | Inspect the operation's region requirement and VMI emitter. |
| `ir.get_reg_memory_effect(call)` | Inspect a load/store's target Tile, known root allocation, read/write access, half-open element interval and optional mask. Unknown roots or interval endpoints remain unknown. |

C++ IRBuilder provides `BeginRegScope`, `EmitReg`, `EmitRegStore` and `EndRegScope`.
Builders use the same type and argument validation as the DSL. Completing a
function validates ownership; `RegScopeValid` validates manually constructed IR,
and `MixedKernelExpanded` checks final AIV placement. Memory-effect inspection
preserves masks without claiming full-buffer initialization or disjointness.

Python text syntax does not accept scope metadata. Printing a manually constructed
RegScope with nonempty `name_hint` or `attrs` raises an error to avoid losing
metadata; binary serialization preserves those fields. Empty scopes print as
`pass`. The API list above describes the supported frontend surface; PTOAS
instructions without a corresponding listed API are not exposed through `pl.reg`.
