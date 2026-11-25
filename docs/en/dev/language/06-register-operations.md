# Register API reference

`pl.reg` provides logical vector and mask operations inside `pl.reg_scope()` on
A5 (Ascend950). This page lists every supported `pl.reg` operation, its parameters,
return type and dtype restrictions. All names in the tables have the `pl.reg.` prefix.

The compiler outlines each register scope early into a `FunctionType.VF`
device helper. The public `reg_scope()` syntax is unchanged. Captured Tiles
and Scalars become function parameters; Tile aliases can be returned, while
registers and local Scalars remain inside the helper.

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

`pl.reg_scope(*, auto_sync=True)` has no captures, result list or yield.
`auto_sync` must be a compile-time bool; omitted or `True` retains automatic
UB barrier insertion, while `False` selects explicit synchronization. Scopes cannot
nest. The frontend accepts InCore and AIV device contexts, including an explicit
InCore scope inside an Opaque, Orchestration or Graph function before outlining.
After `ExpandMixedKernel`, the region must belong to an AIV function. Direct use
in orchestration code and orchestration scopes inside RegScope are rejected.
Explicit non-A5 targets are rejected; target checks wait until compilation if no
target has been selected.

### Explicit UB barriers

`pl.reg.mem_bar(kind: str) -> None` is a statement-only operation inside a
register scope. It emits exactly `pto.mem_bar "KIND"`; it cannot be assigned,
returned or used as an expression. `kind` is a case-sensitive compile-time
string with no default:

| Family | Accepted kinds |
| ------ | -------------- |
| VV | `VV_ALL`, `VST_VLD`, `VLD_VST`, `VST_VST` |
| VS | `VS_ALL`, `VST_LD`, `VLD_ST`, `VST_ST` |
| SV | `SV_ALL`, `ST_VLD`, `LD_VST`, `ST_VST` |
| SS | `SS_ALL`, `ST_LD`, `LD_ST`, `ST_ST` |

```python
# tile is initialized local Vec storage captured from outside the scope.
with pl.reg_scope(auto_sync=False):
    value = pl.reg.load(tile, lanes=64)
    pl.reg.mem_bar("VLD_VST")
    tile = pl.reg.store(tile, value=value)
    pl.reg.mem_bar("VST_VLD")
    value = pl.reg.load(tile, lanes=64)
```

Manual scopes require the author to provide all necessary RAW/WAR/WAW and
loop-carried barriers. Explicit barriers are also allowed in automatic scopes.
`ALL` covers only its family: `VV_ALL` does not cover scalar memory operations.
These barriers order UB accesses within one vecscope; they do not replace
cross-scope, DMA, GM or cross-core synchronization. See the
[VF compilation contract](../codegen/03-vpto-vf-integration.md).

Inline helpers may contain `reg_scope` and accept explicitly typed Vec Tiles.
They inherit their caller's execution context when expanded; final placement
must still be AIV. A register value cannot cross the helper boundary: return
Tiles written by `reg.store` instead.

`pl.Reg[lanes, dtype]` describes a vector register (`VecReg`).
`pl.Reg[lanes, pl.BOOL, pl.RegKind.MaskReg]` describes a predicate mask (`MaskReg`).
The optional third annotation field is the register kind; its default is
`pl.RegKind.VecReg`. Register values are normally inferred from operations.
`lanes` is a static positive integer, not a tensor shape or a byte count.

External Tiles and read-only Scalars are implicit inputs. Registers and local
Scalars cannot escape their defining region through later uses, function returns,
loop results, tuples or assignments to outer variables. Nested tuple types are
checked at function boundaries too. Tile operands must
originate outside the region; `store` and `scatter` return aliases usable after the region.
Assign register operation results to variables, including `store` and
`scatter`. Assign the state returned by `squeeze_store_step` and carry it
through `pl.yield_` inside a loop. See [statements](01-statements.md) for loop
APIs and hardware-loop policy.
`pl.hardware_loop()` (`HardwareLoopPolicy.Require`) is restricted to this lexical
scope, including when the loop has zero trips.
The A5 codegen boundary and backend execution limitations are
described in [VF execution](../codegen/03-vpto-vf-integration.md#required-hardware-loops).

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
all lanes participate. Except for conversion, reinterpretation, histograms,
reduction, `expdif` and byte `gather`, results preserve the input vector dtype
and lanes. There is no implicit vector promotion or lane broadcasting. DSL
numeric literals are normalized to the requested scalar dtype;
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
| `store(tile, offset=0, value=None, mask=None, *, block_stride=None)` | `tile`: destination Vec Tile. `offset`: starting element index. `value`: required VecReg despite the Python default. `mask`: optional write mask. Only active lanes are written; other destination elements are preserved. | Tile alias of `tile` | `T`; value and Tile dtype must match |

`block_stride` is an optional positive static integer in `[1, 32767]`, measured
in **32-byte blocks**; `offset` remains in elements. Omission preserves the
contiguous store. For element size `E`, block width `B = 32 / E`, active lane
`i` writes element `offset + (i // B) * block_stride * B + i % B`.
Inactive lanes and gaps retain their prior contents. Static bounds include the
last strided block; initialization checks credit only active written blocks.
Dynamic offsets remain caller-managed. The memory-effect query returns the
conservative bounding interval, including gaps, for dependency/lifetime analysis.
PyPTO emits the stride as an `i16` operand and a predicate with the vector's lane count.
If the mask is omitted, PyPTO emits `PAT_ALL`. PTOAS lowers BF16 block-stride stores to `vsstb`.
For example, `pl.reg.store(dst, 16, values, mask, block_stride=16)` writes
128 BF16 lanes into eight NZ panels at element offsets `16, 272, ..., 1808`.
Dynamic stride scalars are not exposed in this initial API.
Static initialization analysis records at most the first 256 blocks of each
strided store to bound compile-time work independently of the lane count.
Later blocks are not proved initialized by that store; initialize the destination
first when subsequent register loads need that larger region. This analysis
budget does not restrict the store's lane count or change its generated writes.

Offsets count **elements**, not bytes. Static offsets must be nonnegative and the
whole accessed footprint must fit the Tile capacity, including for masked
stores (`[offset, offset + lanes)` for contiguous stores). Static loads must lie inside initialized `valid_shape`; a later
mask cannot protect an uninitialized load. One-dimensional Tiles are directly
supported. Multidimensional Tiles require an explicit contiguous row-major View;
padded, boxed or compact layouts are rejected. View start offsets are included in
address and allocation-bound checks.

Dynamic offsets and valid extents pass through when static checks cannot prove a
violation. The caller owns their runtime bounds and initialization. A store does
not enlarge `valid_shape`, allocate storage or create a snapshot: its result
aliases the destination and has exactly the destination Tile type.

## Stable compaction and dense stores

`squeeze(value, mask)` packs selected lanes into the beginning of one register,
preserving their order. It accepts one full 256-byte UINT16 or UINT32 register
(128 or 64 lanes) and a matching mask. Only the selected prefix is defined; use
a prefix mask of the selected count when storing it. Integer bits are preserved,
including UINT32 values above `2**24`.

```python
with pl.reg_scope():
    value = pl.reg.load(src, lanes=64)  # UINT32 Tile
    keep = pl.reg.active_mask(9, lanes=64)
    packed = pl.reg.squeeze(value, keep)
    out = pl.reg.store(dst, value=packed, mask=keep)
```

Use the explicit `squeeze_store_init/step/flush` operations when successive
chunks must append into one dense UB sequence. Carry the opaque state through
the loop with `pl.range(..., init_values=...)` and yield each updated state:

```python
with pl.reg_scope():
    init_ureg = pl.reg.squeeze_store_init(dst, offset=7)
    for i, (curr_ureg,) in pl.range(0, 4, init_values=(init_ureg,)):
        value = pl.reg.load(src, offset=i * 64, lanes=64)
        keep = pl.reg.active_mask(9, lanes=64)
        next_ureg = pl.reg.squeeze_store_step(curr_ureg, dst, value, keep)
        final_ureg = pl.yield_(next_ureg)
    written = pl.reg.squeeze_store_flush(final_ureg, dst)
    # dst[7:43] contains four selected prefixes, without gaps.
```

The destination is a Tile, not a pointer. Only `init` accepts `offset`: it is
evaluated once and counts elements from the Tile's contiguous view origin,
including any existing View `start_offset`. For example, a view origin of 32
and `offset=7` start at element 39 of the backing allocation. `step` and `flush`
have no offset argument; they continue the existing chain. Each `step` must use the same
destination as `init`, and its mask must match the VecReg lane count. Use a
full or prefix mask when appending all lanes or a known prefix. A squeezed
register has an undefined tail, so use a mask that selects only its defined
prefix.

`init` initializes the UReg append state and returns an opaque
`RegKind.UReg` value; it does not initialize the destination Tile. Each
`step` must consume the current loop-carried UReg value and returns its
successor; the loop carries that value through `init_values` and `pl.yield_`.
Outside the loop, consume the result bound by `pl.yield_` (`final_ureg` above).
Call `flush` exactly once after the loop. The
backend lowers the chain to one alignment initialization, loop-carried
`!pto.align` passed through `vstur "POST_UPDATE"`, and one final `vstar`.
`flush` returns an alias of the destination Tile; it does not zero unwritten
capacity. The caller guarantees total output capacity and reads only the
written prefix.

An input can contain zero chunks, for example when a batch has no candidates.
Zero iterations forward the initial state unchanged. A chain that has been
initialized still requires exactly one flush, even when no step ran; it appends
zero elements and preserves the destination. The same rule covers an empty
straight-line sequence:

```python
with pl.reg_scope():
    ureg = pl.reg.squeeze_store_init(dst, offset=7)
    written = pl.reg.squeeze_store_flush(ureg, dst)
```

This keeps one init/step/flush contract for zero and nonzero runtime counts,
without a separate user-side conditional. Proven empty loops can be removed
while forwarding their initial state.

PTOAS 0.77 treats the stored form of VSQZ as pure, although it pushes a
SQZN count consumed by VSTUR. Until that effect is modeled in PTOAS, the verifier
rejects loop-invariant value/mask pairs: load or derive at least one input inside
the current loop. The input may come from a loop carry updated by such a value
through `pl.yield_`. Reusing equivalent value/mask producers for multiple steps in
one RegScope is also rejected to prevent commoning away a required SQZN producer.
This includes repeated writes of one value with equivalent full or prefix masks,
even when those masks were constructed separately.
These are conservative backend restrictions, not a change to append semantics.

With PTOAS 0.77 on A5, avoid using two loads from the same unchanged address
with the same mask in one RegScope. PTOAS can merge their stored VSQZ producers,
leaving a VSTUR without its SQZN count. Also avoid reopening a stream in the
same RegScope. PTOAS can schedule the second AR reset before the first VSTAR
flush. See [PTOAS #1779](https://github.com/hw-native-sys/PTOAS/issues/1779) for
the affected cases and follow-up validation.

The destination must be a static one-dimensional Vec Tile or a two-dimensional
Vec Tile that can be flattened in row-major order without copying. UINT16/UINT32
dtype must match the input. Both implicit dense layout and explicit dense strides
are accepted: `[2, 64]` with stride `[64, 1]` is valid; stride `[96, 1]` has row
gaps and is rejected. Singleton axes contribute no gap, including single-row or
single-column views. Boxed, padded, compact and nondegenerate column-major views
are unsupported. Changing shape alone does not make a strided view contiguous.
Offsets include the Tile View origin and are checked against static capacity at
entry. As with dynamic register stores, the caller
must ensure `offset + sum(selected_lanes) <= dst_capacity` at runtime.
There is no truncation or allocation growth. Bytes before and after the written
sequence retain their contents; initialize any tail that will later be read.

The state is linear and local to its RegScope: consume only the current state,
replace it with the step result, and flush once before leaving the region. It
cannot escape through a function return or container. Do not read, write or rebind
the destination or its aliases while the chain is active. Every for loop
enclosing a stream opening must be sequential. Carry state with `init_values` and
`pl.yield_`, with at most four nested loop levels. Use lane masks for filtering
while a chain is active. The active chain requires straight-line or sequential
for-loop control flow.

The current AR-based lowering supports one **active** append chain at a time.
The IR supports sequential chains after flushing, subject to the PTOAS 0.77
scheduling limitation above. Ascend C documents separate limits of four load
and four store unalign registers. Squeeze-store also uses shared AR state.
Supporting multiple live squeeze chains requires independent
cursor/count management and backend validation; increasing a resource limit alone
does not provide that support. See the [register limits](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/reg_vector_compute/register_data_types/UnalignRegForLoad-UnalignRegForStore.md)
and [squeeze-store AR semantics](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta2/API/ascendcopapi/atlasascendc_api_07_0324.html).

Squeeze-store uses physical micro instructions for its alignment state. A
register scope containing a stream uses physical register codegen throughout.
It currently supports full 256-byte loads/stores,
scalar broadcast, index generation, reinterpretation, comparisons, selection,
mask generation/logic, and squeeze. Other operations or partial register shapes
are diagnosed. The verifier rejects `reg.add` anywhere in a scope containing a
stream. Scopes without streams retain ordinary VMI lowering, including
pure `squeeze`. Future PTOAS stream-aware VMI can preserve this public API.

## Vector and mask generators

| API signature | Parameters and behavior | Result | Dtypes |
| ------------- | ----------------------- | ------ | ------ |
| `broadcast(scalar, *, lanes, dtype)` | Repeat a scalar or one-lane VecReg into every lane. Input dtype must equal `dtype`. | VecReg | `T` |
| `index_gen(base, *, lanes, dtype, order="ascending")` | `base`: scalar start value; result lane `i` is `base + i`. `order` accepts only `"ascending"`. Base dtype must equal `dtype`. | VecReg | `F + I`; BF16 is rejected during type deduction, before codegen |
| `full_mask(*, lanes)` | Activate every logical lane. | BOOL MaskReg | BOOL |
| `active_mask(active_lanes, *, lanes)` | Activate the prefix `[0, active_lanes)` using a static or runtime INDEX count. Constants are checked against `[0, lanes]`; callers must keep runtime counts in that range. Zero creates an all-false mask; the input count is unchanged. | BOOL MaskReg | BOOL |

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
| `reduce_sum(value, mask, *, reassoc=False, group_size=None)` | Sum active lanes. `reassoc=True` permits hardware tree reassociation and is required for floating input; integers accept either value. | One-lane VecReg, or one lane per group | `F + I16_32` |
| `reduce_max(value, mask, *, group_size=None)` | Maximum of active lanes | One-lane VecReg, or one lane per group | `F + I16_32` |
| `reduce_min(value, mask, *, group_size=None)` | Minimum of active lanes | One-lane VecReg, or one lane per group | `F + I16_32` |
| `group_broadcast(value, *, lanes)` | Expand a grouped result back across its groups | VecReg with `lanes` lanes, same dtype | `F + I16_32` |

`value` is the source vector; required `mask` has the **source** lane count, not
one lane. Inactive inputs contribute 0 for sum, the dtype minimum (or negative
infinity) for max, and the dtype maximum (or positive infinity) for min. No
implicit widening is performed. Results remain local registers; use `store` to
write the output elements.

Omitting `group_size` reduces the whole vector to one lane. Supplying it splits
the source into equal contiguous groups of that width and produces one lane per
group, in group order — PTOAS `vcadd`/`vcmax`/`vcmin` with a group count. The
group width must divide the source lane count, and the resulting **group count**
must be 1, 2, 4 or 8: A5 has no other grouped layout, so a 16-group request is
rejected at trace time rather than during layout assignment. MX-style group-32
quantization over 64 lanes is therefore `group_size=32`, giving two groups.

`group_broadcast` consumes exactly such a result and writes each group's value
back across that group's lanes, so a per-group scale can divide the activations
it came from. The group count is the source lane count — it is never passed
separately, since it could only disagree with the operand. `lanes` is the width
to expand back to and must be a multiple of the group count.

```python
with pl.reg_scope():
    x = pl.reg.load(src, lanes=64)
    mask = pl.reg.full_mask(lanes=64)
    # One amax per 32-element MX group, then back across its own 32 lanes.
    amax = pl.reg.reduce_max(pl.reg.abs(x, mask), mask, group_size=32)
    scale = pl.reg.group_broadcast(amax, lanes=64)
    scaled = pl.reg.div(x, scale, mask)
    out = pl.reg.store(dst, value=scaled)
```

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
| BF16 | FP32 | Must be omitted | Must be omitted |
| FP32 | FP16 | R/A/Z; DSL default R | Required: True or False |
| FP32 | BF16 | R/A/Z; DSL default R | Required: True or False |
| FP32 | FP8E4M3FN | R/A/Z; DSL default R | Required: True or False |
| FP32 | INT32 | R/A/F/C/Z; DSL default Z | Required: True or False |
| INT32 | FP32 | Must be omitted | Must be omitted |
| INT8 | FP16 | Must be omitted | Must be omitted |
| INT8 | INT16 | Must be omitted | Must be omitted |
| INT16 | INT32 | Must be omitted | Must be omitted |
| UINT8 | UINT16 | Must be omitted | Must be omitted |
| UINT16 | UINT32 | Must be omitted | Must be omitted |
| INT32 | INT8 | Must be omitted | Required: False |

R rounds to nearest with ties to even; A rounds away from zero; F rounds
toward negative infinity; C toward positive infinity; Z toward zero. H is not supported for FP32-to-FP16 narrowing and is rejected before codegen.
`saturate=True` selects saturation (clamping to the target range);
`saturate=False` selects non-saturating conversion.
Same-dtype DSL conversion returns the original register if both optional
attributes are omitted. Other dtype pairs are rejected.
FP8E4M3FN conversion preserves logical lanes as compact FP8 values.
Use `interpret_cast(result, dtype=pl.UINT8)` to store their raw bytes through
an integer Tile. FP8 arithmetic, direct register loads/stores and reverse numeric
conversion are not enabled by this conversion pair.

Raw IR/Builder calls must supply normalized rounding explicitly
where required; the defaults above are supplied by the DSL wrapper.

## Extended VMI operations

These APIs live under `pl.reg` and emit the corresponding `pto.vmi.v*` operation.
The masks of `mula`, `intlv`, and `dintlv` are optional; the other operations
require a governing mask. Arithmetic
`pmode` is `"zero"`; carry operations and `scatter` expose no `pmode` argument.
Vector operands must agree in dtype, signedness and lane count unless explicitly
specified below. Existing scalar SSA values are never implicitly cast; Python
scalar literals for `axpy`/`lrelu` are normalized to the vector dtype.

| API (positional arguments) | VMI op | Result and contract |
| -------------------------- | ------ | ------------------- |
| `expdif(x, max_value, mask)` | `vexpdif` | Fused `exp(x - max_value)`; FP16/FP32 inputs, always FP32 result |
| `axpy(x, acc, alpha, mask)` | `vaxpy` | Fused `alpha * x + acc`; FP16/FP32, scalar alpha |
| `mula(acc, lhs, rhs, mask=None)` | `vmula` | Fused `acc + lhs * rhs`; FP16/BF16/FP32 or signed/unsigned 16/32-bit integers |
| `lrelu(x, slope, mask)` | `vlrelu` | Leaky ReLU; FP16/FP32, scalar slope |
| `prelu(x, alpha, mask)` | `vprelu` | PReLU; FP16/FP32, per-lane alpha vector |
| `mull(lhs, rhs, mask)` | `vmull` | `(low, high)` 32-bit product halves; INT32/UINT32, 64/128/256 lanes |
| `addc(lhs, rhs, mask)` | `vaddc` | `(result, carry)`; INT32/UINT32 |
| `subc(lhs, rhs, mask)` | `vsubc` | `(result, not_borrow)`; INT32/UINT32 |
| `addcs(lhs, rhs, carry_in, mask)` | `vaddcs` | `(result, carry_out)` for `lhs + rhs + carry_in`; INT32/UINT32 |
| `subcs(lhs, rhs, carry_in, mask)` | `vsubcs` | `(result, not_borrow_out)` for `lhs - rhs - (1 - carry_in)`; INT32/UINT32 |
| `gather(tile, offsets, mask, base_offset=0)` | `vgather` | Masked indexed UB read; inactive result lanes are zero |
| `scatter(tile, offsets, value, mask, base_offset=0)` | `vscatter` | Masked indexed UB write; returns the destination Tile alias |
| `intlv(lhs, rhs, mask=None)` | `vintlv` | `(low, high)` interleaved vectors; even lane count |
| `dintlv(lhs, rhs, mask=None)` | `vdintlv` | `(even, odd)` deinterleaved vectors; even lane count |

Unknown indexed or dynamic reads do not disable initialization checks for known
static reads on the same allocation. Their own safety remains the caller's responsibility.

Paired results use one tuple-producing IR Call and two SSA results. The tuple,
its projections and all carry masks remain local to the defining RegScope.
Bind a paired call before indexing it: use `pair = pl.reg.mull(a, b, mask)`
followed by `low = pair[0]`. Direct `pl.reg.mull(a, b, mask)[0]` is rejected.
Unpack paired results before carrying them through `if` joins or loops: pass
individual registers (for example, `low, high = pl.reg.mull(a, b, mask)`),
not the tuple itself. The verifier rejects tuple-valued joins and loop carries.
`mull` preserves input signedness in both halves. Subtraction carry masks use
**true for no borrow**; they are distinct from the execution mask.
PyPTO emits masked interleave directly. Both outputs of `intlv`/`dintlv`
have zero in inactive lanes and preserve the bit patterns of active BF16
values. Omitting the mask enables all lanes and preserves fusion opportunities.

`tests/st/runtime/vpto/test_reg_interleave_onboard.py` checks both results for explicit and
omitted masks, including raw bits and guard regions. The default run covers
lengths at least as wide as one physical register. Short-vector code generation
has compile coverage with PTOAS v0.77. A5 numerical validation for those lengths
is pending. Use `--lanes 8` to run a targeted hardware regression. Build outside
device allocation, then execute the saved artifacts on A5:

```bash
task-submit --no-device --run \
  'python tests/st/runtime/vpto/test_reg_interleave_onboard.py --build build/interleave'
task-submit --device auto --run \
  'python tests/st/runtime/vpto/test_reg_interleave_onboard.py --worker build/interleave "$TASK_DEVICE"'
```

```python
# Inside a reg_scope, with matching UINT32 registers lo_a/lo_b/hi_a/hi_b:
lo, carry = pl.reg.addc(lo_a, lo_b, mask)
hi, carry_out = pl.reg.addcs(hi_a, hi_b, carry, mask)
product_lo, product_hi = pl.reg.mull(lo_a, lo_b, mask)
interleaved_lo, interleaved_hi = pl.reg.intlv(lo_a, lo_b, mask)
a, b = pl.reg.dintlv(interleaved_lo, interleaved_hi, mask)
```

For indexed memory, `offsets` is a UINT16/UINT32 VecReg with the same lane count
as the value/result and mask. Indices count **elements**, relative to the Tile
view origin. Use UINT16 offsets for 8/16-bit Tile elements and UINT32 offsets
for 32-bit Tile elements, matching the PTOAS physical indexed paths. INT8/UINT8
gather sources are zero-extended to INT16/UINT16 respectively. Other gather results retain the
Tile dtype. Scatter values must match the Tile dtype. The existing contiguous
Vec Tile geometry restrictions still apply.

The caller must ensure active indexed addresses are in bounds and gather reads
initialized elements. Active scatter indices must be pairwise distinct.
Inactive lanes perform no memory access, and scatter preserves every unwritten
element without enlarging `valid_shape`. Indexed effects retain the allocation
root and predicate but have unknown interval endpoints. Like dynamic offsets,
they do not establish static initialization proofs. VF helpers capture the
whole accessed Tile conservatively (`read` for gather, `readwrite` for scatter).
Mixed-dtype captures use typed pointers derived from the same UB base.

Indexed codegen relies on PTOAS to synchronize unknown indexed
footprints within a vecscope, including RAW, WAR and WAW dependencies. This
version includes the fix for [PTOAS #1730](https://github.com/hw-native-sys/PTOAS/issues/1730),
so PyPTO emits no per-access `VV_ALL` fences. PTOAS owns VF dependency analysis;
PyPTO does not recheck barrier coverage in emitted LLVM. Caller/helper boundary
and GM synchronization retain their existing mechanisms.

```python
with pl.reg_scope():
    offsets = pl.reg.index_gen(0, lanes=64, dtype=pl.UINT32)
    mask = pl.reg.active_mask(32, lanes=64)
    values = pl.reg.gather(src, offsets, mask)
    out = pl.reg.scatter(dst, offsets, values, mask)
```

## IR Builder and inspection

| API | Meaning |
| --- | ------- |
| `with ib.reg_scope(span=None, *, auto_sync=True):` | Open a register region in Python IRBuilder. |
| `ib.emit_reg(name, op_name, args, kwargs=None, span=None)` | Create a named register operation. `op_name` is the registered name such as `"reg.add"`; `args` contains IR operands, `kwargs` operation attributes. |
| `ib.emit_reg_mem_bar(kind, span=None)` | Emit a statement-only explicit UB barrier. |
| `ib.emit_reg_store(name, tile, value, offset=0, mask=None, span=None, *, block_stride=None)` | Emit a named store and return the destination Tile alias. |
| `ir.get_reg_op_spec(name)` | Inspect the operation's region requirement and VMI emitter. |
| `ir.get_reg_memory_effect(call)` | Inspect a load/store's target Tile, known root allocation, read/write access, half-open element interval and optional mask. Unknown roots or interval endpoints remain unknown. |

C++ IRBuilder provides `BeginRegScope`, `EmitReg`, `EmitRegStore` and `EndRegScope`.
Builders use the same type and argument validation as the DSL. Completing a
function validates ownership; `RegScopeValid` validates manually constructed IR,
and `MixedKernelExpanded` checks final AIV placement. Memory-effect inspection
preserves masks without claiming full-buffer initialization or disjointness.

Python printing emits `pl.reg_scope(auto_sync=False)` for a manual scope.
Other nonempty attributes and nonempty `name_hint` raise an error to avoid losing
metadata; binary serialization preserves those fields. Empty scopes print as
`pass`. The API list above describes the supported frontend surface; PTOAS
instructions without a corresponding listed API are not exposed through `pl.reg`.

## Bit reinterpretation, histograms and grouped masks

| API | PTOAS operation | Contract |
| --- | --------------- | -------- |
| `interpret_cast(value, *, dtype)` | `vinterpret_cast` | Preserve every bit, without numeric conversion. Source and destination use `T`; result lanes equal `input_lanes * input_bits / output_bits`, which must be integral. |
| `chist(acc, source, mask)` | `vchist` | `result[b] = acc[b] + count(active source <= b)`, modulo 65536. |
| `dhist(acc, source, mask)` | `vdhist` | `result[b] = acc[b] + count(active source == b)`, modulo 65536. |
| `create_group_mask(active_elems_per_group, *, num_groups, group_size)` | `create_group_mask` | Return a BOOL MaskReg with `num_groups * group_size` lanes. Lane `i` is active iff `i % group_size < active_elems_per_group`. |

Histogram `source` is a UINT8 VecReg; its governing mask has the **source**
lane count. The independent accumulator and result are UINT16 VecRegs with
128 or 256 bins, covering 0–127 or 0–255. The mask does not gate accumulator
bins: an empty mask returns the accumulator unchanged. Both operations return
new register values without mutating `acc`. Cumulative bins 128–255 include
counts from values below 128 as well. Results can be stored back over the loaded
accumulator with `pl.reg.store`; PTOAS determines ordering from memory and SSA
dependencies. The A5 regression covers both histogram operations with in-place
128/256-bin updates, including accumulator wraparound.

Grouped-mask geometry consists of positive static integers. The active count
is an INDEX scalar or integer literal in `[0, group_size]`; dynamic counts are
supported and callers must keep them in range. Constant counts are validated.
These four APIs have no `pmode` argument and access no UB memory themselves.

PTOAS 0.70 restricts **dynamic** grouped masks to power-of-two `group_size`
and 32-bit consumers (b32 mask granularity). Use constant active counts for
byte/halfword consumers, including histograms. Other dynamic granularities
are rejected by PTOAS lowering. VF helpers support mixed Tile dtypes through
typed pointers derived from one shared UB base; offsets use each Tile's own
element width.

The optional `base_offset` is a scalar INDEX offset in source/destination elements,
added to the Tile view origin before applying the unsigned vector indices.
It shifts the UB pointer without broadcasting the offset or widening the indices.
Dynamic active addresses and initialization remain the caller's responsibility;
masked-off lanes are not accessed. Existing calls default to a zero base offset.

`intlv` and `dintlv` accept an omitted mask for all-active rearrangement.
Explicit masks retain zero-inactive-result semantics; omitted masks do not need
result-clearing selects. Two adjacent BF16 loads followed by unmasked `dintlv`
can therefore lower to A5 `DINTLV_B16`, with `convert` selecting the required
EVEN/ODD parts automatically. Physical conversion parts remain a backend choice.

### Explicit physical register mode (A5)

For instruction-for-instruction ports of native VF kernels, use
`pl.reg.load_native`, `pl.reg.convert_native` and `pl.reg.store_native`. These
operations retain a full 256-byte physical register and select the PTO micro
lowering for the containing `reg_scope`. The ordinary logical register APIs
and default VMI lowering are unchanged. All register operands in a native
scope must describe full hardware vectors or masks.

- `load_native(..., dtype, lanes, mode="UNPK4")` reads 64 packed FP4/FP8 bytes
  into a 256-lane register. `BRC_B32` reads one 32-bit element and `E2B_B16`
  reads eight 16-bit group scales. `NORM` reads a complete register.
- `convert_native(..., mask, dtype, lanes, part="PART_P0")` converts native
  FP4 to BF16 or FP8 to FP32. FP32-to-BF16 uses `PART_EVEN` or `PART_ODD`,
  rounding `R` and disabled saturation, returning 128 physical lanes.
- Use `dintlv` on two physical BF16 registers to collect their populated even
  lanes into one full register, or `store_native(..., mode="PK_B32")` to
  write a single register's packed even lanes. `ONEPT_B32` writes lane zero.
  The mask for `PK_B32` is a 64-lane 32-bit mask, not 128 lanes.
- `reduce_max_native` and `reduce_sum_native` keep the result at physical
  lane zero of the full FP32 register; other lanes are not a defined result.

All offsets are in the backing Tile's elements. Bounds and initialization
are checked using the actual source/store footprint rather than counting
unused physical register lanes. Only the explicitly supported native
operations may be combined inside the scope; unsupported operations fail
codegen rather than being silently lowered through logical VMI.
