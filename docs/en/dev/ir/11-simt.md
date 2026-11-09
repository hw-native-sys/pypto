# Scalar SIMT IR

Use `pl.simt_scope` inside an explicit `@pl.jit.aiv` function to run scalar
code on a SIMT thread block. The compiler lowers that region to the canonical
void helper and explicit launch model described below.

## Public authoring

```python
import pypto.language as pl

@pl.jit.aiv
def copy_aiv(src: pl.Tensor[[1, 128], pl.FP32],
             dst: pl.InOut[pl.Tensor[[1, 128], pl.FP32]],
             valid: pl.Scalar[pl.INT32]):
    with pl.simt_scope(dims=(128, 1, 1)):
        tid = pl.simt.thread_idx(pl.simt.Axis.X)
        if tid < valid:
            value = pl.simt.load(src, (0, tid))
            pl.simt.store(dst, (0, tid), value)
    return dst

@pl.jit
def copy(src: pl.Tensor[[1, 128], pl.FP32],
         dst: pl.InOut[pl.Tensor[[1, 128], pl.FP32]],
         valid: pl.Scalar[pl.INT32]):
    return copy_aiv(src, dst, valid)
```

`pl.simt.Axis` is the shared `ir.SimtAxis` enum. Coordinate queries require one
typed axis, positional or `axis=`; `lane_id()` takes none. Dimensions are three
positive compile-time integers. They describe threads, not an element allocation
or grid; bounds checks remain explicit. GM Tensor and Vec/UB Tile operands use
their existing storage. The initial API supports the backend's scalar subset;
INT32/FP32 lane collectives, atomics and synchronization are described below. Expanded math remains a separate milestone.

Scalar assignments are thread-private, including shadowing an outer scalar.
A private scalar cannot be read after the scope. Stores automatically rebind the
outer storage handle, with one result per written capture. An alias made before
a Tile update retains the old value; Tensor aliases retain GM reference semantics.
`In` storage stays read-only through views, and partial updates require initialized
contents. Build views outside the region. No `as` clause, nested region, storage
allocation/rebinding, cross-boundary jump, VMI capture or manual resource limit is
accepted. Scopes may be placed inside outer loops and branches.

The transient `SimtScopeStmt` derives directly from `Stmt`, with `block_dims` and
`body` fields, plus `private_scalars` identifying reassigned outer scalar bindings.
It preserves the execution boundary through binding, printing,
serialization and cloning. [OutlineSimtScopes](../passes/01-outline_simt_scopes.md)
runs before SSA and emits canonical helpers/launches. Explicit named helper and
`pl.simt.launch` syntax remains available for canonical IR.

## Function and launch

`FunctionType.SIMT` is appended to the serialized function-kind enum (value 9).
A SIMT helper has no runtime task level, role, binding requirement, or return
values. `is_incore_type(SIMT)` is false: helpers are not dispatchable runtime tasks.
`is_device_function_type(SIMT)` is true.

```python
launch = ir.SimtLaunchStmt(
    ir.GlobalVar("helper"),
    [ir.ConstInt(n, DataType.INT32, span) for n in (64, 1, 1)],
    [src, dst],
    [updated_dst],
    [1],
    span,
)
```

`block_dims` contains three positive constant integers. Boolean and dynamic
values, and products exceeding INT32_MAX, are rejected at construction.
`args` are uses; `results` are definitions. `result_param_indices` is increasing
and maps each result to exactly one InOut helper parameter. Results retain the
corresponding argument's full type and storage metadata.

The statement participates in visitor/mutator dispatch, structural equality and
hashing, MessagePack serialization, deep cloning, use-before-definition checks,
and SSA conversion. Python printing uses a canonical statement spelling:

```python
updated_dst = pl.simt.launch(
    self.helper, src, dst, block_dims=(64, 1, 1), result_param_indices=(1,)
)
```

The text parser reads this spelling. It is an IR round-trip format; the higher
level programming interface derives output mappings during outlining.

## Scalar operations

| Operation | Result | Contract |
| --------- | ------ | -------- |
| `simt.thread_idx`, `block_idx`, `block_dim`, `grid_dim` | INT32 | Required typed `SimtAxis.X/Y/Z` kwarg |
| `simt.lane_id` | INT32 | No axis |
| `simt.load(storage, indices)` | Element scalar | Tile in Vec/UB or Tensor in GM |
| `simt.store(storage, indices, value)` | VoidType | Scalar dtype must match storage |

Python IR wrappers live in `pypto.ir.op.simt`. Indices normalize to `MakeTuple`
with one integer/index scalar per logical dimension. Store belongs in `EvalStmt`,
not a value-bearing assignment. Packed vectors and sub-byte storage are excluded.

Tensor accesses express direct GM access with no implicit Tile staging. ND/DN
views carry their existing strides and offsets; fractal layouts are rejected.
Explicitly non-Vec Tile storage is rejected. GM accepts PTOAS `l1cache` policies
`cache`/`uncache` and separate load/store `l2cache` policy sets. Cache attributes
on UB accesses are invalid.

## Warp collectives

The public `pl.simt` and IR `pypto.ir.op.simt` namespaces expose:

| Operation | Input → result | Meaning |
| --------- | -------------- | ------- |
| `vote_all`, `vote_any`, `vote_uni` | BOOL → BOOL | All true, any true, or all predicates equal |
| `vote_ballot` | BOOL → INT32 | Bit i is set for active lane i with a true predicate |
| `shuffle_idx(value, index, *, width=32)` | INT32/FP32 → same type | Read subgroup-relative source lane |
| `shuffle_up`, `shuffle_down` | INT32/FP32 → same type | Read lower/higher source lane by an offset |
| `shuffle_bfly(value, mask, *, width=32)` | INT32/FP32 → same type | Read source lane selected by local-lane XOR mask |
| `redux_add`, `redux_min`, `redux_max` | INT32/FP32 → same type | Reduce active lanes; return the result to each active lane |

The A5 warp contains 32 lanes. Participation is the current active execution
mask at the instruction; there is no explicit participation-mask argument.
Inactive lanes contribute nothing. These are register collectives, not memory
fences or block barriers. Their results depend on control flow, so they must
not be hoisted across branches or iterations that change active lanes.

Shuffle widths are **16 or 32**, defaulting to 32. Width splits the warp into
contiguous subgroups; it does not alter vote or redux participation. Lane controls
are INT32 (Python integer literals are normalized). A shuffle source must be
active and inside the same subgroup. Results from inactive or out-of-range
sources are unspecified; algorithms must not rely on boundary self-copy.

Ballot returns an INT32 bit carrier, so bit 31 can make its numeric value
negative. Use bit operations to inspect it. INT32 min/max use signed comparisons;
integer sums wrap at 32 bits. FP32 follows the target reduction order and
special-value rules, without a promise of sequential summation or portable
NaN/signed-zero selection. FP16, UINT32 and packed values remain unsupported
in this initial frontend/backend subset.

The native mapping is `pto.vote_*`, `pto.shuffle_*` and `pto.redux_{add,min,max}{i,f}`;
integer min/max explicitly carry `signed`. Tests compile all forms through the
public JIT and PTOAS LLVM export. Opt-in device acceptance lives in
`tests/st/runtime/vpto/test_simt_collectives_onboard.py`, covering two warps,
partial active sets, ballot bit 31, signed reductions, shuffle subgroups and
untouched inactive outputs. Compile-only tests do not establish device execution.

## Atomics and synchronization

`pl.simt.atomic_add/min/max/exch(storage, indices, value)` and
`pl.simt.atomic_cas(storage, indices, compare, value)` return the **old scalar**.
The same names exist in `pypto.ir.op.simt`. All operands must match the storage
INT32/FP32 dtype. Indices use the same logical coordinates and physical strides
as `load`/`store`; addresses must be naturally aligned to four bytes and in bounds.

| Operation | INT32 | FP32 | Storage / result |
| --------- | ----- | ---- | ---------------- |
| `atomic_add` | 32-bit wrapping addition | Target floating addition | GM and Vec/UB; old scalar |
| `atomic_min`, `atomic_max` | Signed comparison | Numeric operand wins over one NaN; both NaN yield NaN | GM and Vec/UB; old scalar |
| `atomic_exch` | Replace | Replace | GM and Vec/UB; old scalar |
| `atomic_cas` | Bitwise comparison | Bitwise comparison, including NaN payloads and signed zeros | GM and Vec/UB; old scalar |

On the validated A5 target, min selects negative zero and max positive zero when
comparing opposite signed zeros. NaN result payloads from arithmetic are not
promised. UINT32/INT64, FP16/BF16 and packed atomics are not part of this frontend
subset, even when PTOAS supports some of them. There is no selectable memory-order
or scope argument: each operation uses the native address-space atomic contract.
No acquire/release or sequential-consistency guarantee is added for other addresses.
Optional `l2cache=` uses store cache tokens and is allowed only on GM.

An atomic is always `ReadWrite`, including Exch and an unused return value.
Targets must already be initialized; an atomic cannot establish full initialization
of an `Out` capture. A prior full ordinary store can initialize it. Atomic updates
remain in the program when their old values are unused. Contending threads observe
a serialized order for the same location, without a guaranteed thread order.

```python
@pl.jit.aiv
def increment(counter: pl.InOut[pl.Tensor[[1], pl.INT32]]):
    with pl.simt_scope(dims=(64, 1, 1)):
        pl.simt.atomic_add(counter, (0,), pl.const(1, pl.INT32))
    return counter
```

The caller initializes `counter`; one invocation adds 64.

| Operation | Contract |
| --------- | -------- |
| `syncthreads()` | Current SIMT entry rendezvous; prior participating memory effects precede subsequent effects |
| `threadfence()` | Native target workitem fence; orders the calling thread's memory effects |
| `threadfence_block()` | Native block-scoped workitem fence; orders the calling thread's memory effects |

These operations return void and are retained as ordered calls. Fences do not
wait for peers. They are not a cross-AIV/global barrier, and the name `threadfence`
does not imply CUDA device/system-scope or CPU visibility semantics.
Use a rendezvous before reading another thread's writes; a fence alone does not
make an unsynchronized producer/consumer protocol safe.

This version conservatively requires `syncthreads` outside **all branches and
loops**, including apparently uniform ones. It may follow a reconverged branch.
Helpers with a barrier cannot contain early returns, break or continue. Warp votes
cannot establish block-wide arrival. Fences may appear in divergent control flow
because they do not rendezvous. Outer MTE/Vector → SIMT → MTE/Vector dependencies
still use the launch's storage effects and PTOAS synchronization; these builtins
do not replace that protocol.

## Effects and validation

`analyze_simt_access(helper)` reports physical reads/writes per parameter.
A physical store is `Write`; a writable launch parameter is logically `InOut`
because unwritten elements keep their contents. The launch result is an updated
storage version, not a scalar return or a runtime task handle.

The `SimtValid` structural verifier checks helper signatures, operation domains,
argument types, writable mappings, and duplicate writable arguments. Helpers
accept scalar parameters and borrowed Tile/Tensor handles. Storage cannot be
allocated, rebound, or carried through helper control-flow results. Only scalar
control flow and SIMT builtins are allowed. `Out` parameters, ordinary calls or
runtime submissions to helpers, and nested launches are rejected. Launch callers
are AIV or pre-expansion InCore functions. A VMI register is not a supported storage operand;
helpers borrow UB or GM storage instead.
Launching inside an active VMI register scope is also rejected. Mixed-kernel
expansion assigns SIMT launches to AIV rather than duplicating them across cores.

## Compiler preparation

The default pipeline now preserves storage semantics before memory planning:

- [MaterializeSimtOutputs](../passes/36-materialize_simt_outputs.md) initializes
  a dedicated copy for each writable Tile. Partial writes preserve untouched
  elements and old Tile values remain readable across branches and loops.
- [SpecializeSimtHelpers](../passes/37-specialize_simt_helpers.md) derives helper
  variants from full block dimensions, types/layouts, and the target. Dynamic
  metadata becomes scalar arguments; allocation addresses do not enter the key.
- GM results retain the original tensor identity. `simt.store` uses the `Simt`
  write channel, separate from scalar-controller and DMA stores. Kernel parameter
  directions propagate as InOut to outer Call and Submit sites, where ordinary
  task dependency analysis applies. Device statement scheduling preserves launch
  order even when a later read uses the old GM handle.
- Mixed-kernel splitting keeps launches on AIV and preserves their borrowed GM
  handles on AIC before cleanup. Tensor views retain their shape, strides and
  layout, including views and aliases carried through loops and branches.
  GM handle bookkeeping alone does not create a cube kernel.

Known overlapping writable arguments are rejected before copying. Distinct GM
arguments may still alias at runtime; guaranteeing nonoverlap remains the caller's
responsibility. Helper parameters borrow storage and receive no independent
allocations. SIMT kernels stay in Functional IR under automatic Buffer IR selection.

## Derived resources and implementation boundary

`infer_simt_resources(launch)` consults the current target's `BackendHandler`
through `PassContext`. Ascend950 accepts at most 2,048 threads, including the
product of all three dimensions, and derives these limits:

| Threads | Maximum registers |
| ------- | ----------------- |
| 1–256 | 128 |
| 257–512 | 64 |
| 513–1,024 | 32 |
| 1,025–2,048 | 16 |

Users do not provide `simt_max_threads` or `simt_max_regs`. Other backends report
SIMT resource inference as unsupported.

Canonical IR, validation, storage preparation, memory-planner integration,
resource specialization and the A5 executable subset are implemented.
[SIMT lowering and execution](../codegen/05-simt-lowering.md) describes its ABI,
synchronization, supported types and acceptance tests. The independent
[native SIMT probe](../codegen/04-simt-native-validation.md) retains the earlier
P0 ABI fixture. The public scope path uses the same backend and runtime contract.
