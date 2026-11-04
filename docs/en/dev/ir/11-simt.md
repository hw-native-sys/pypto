# Scalar SIMT IR

The canonical SIMT IR models a void device helper and an explicit launch
statement. Its primitives follow PTOAS. Inline `simt_scope` syntax and outlining
are a separate layer and are not implemented by this interface.

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
level programming interface will derive output mappings during outlining.

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

- [MaterializeSimtOutputs](../passes/35-materialize_simt_outputs.md) initializes
  a dedicated copy for each writable Tile. Partial writes preserve untouched
  elements and old Tile values remain readable across branches and loops.
- [SpecializeSimtHelpers](../passes/36-specialize_simt_helpers.md) derives helper
  variants from full block dimensions, types/layouts, and the target. Dynamic
  metadata becomes scalar arguments; allocation addresses do not enter the key.
- GM results retain the original tensor identity. `simt.store` uses the `Simt`
  write channel, separate from scalar-controller and DMA stores. Kernel parameter
  directions propagate as InOut to outer Call and Submit sites, where ordinary
  task dependency analysis applies. Device statement scheduling preserves launch
  order even when a later read uses the old GM handle.

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

Canonical IR, validation, storage preparation, memory-planner integration, and
resource specialization are implemented. PTOAS lowering and device execution
remain separate stages. PTO codegen still rejects SIMT explicitly; successful
pipeline verification does not establish executable kernel support.
