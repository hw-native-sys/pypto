# Logical synchronization IR

Logical synchronization expresses completion dependencies inside one execution
of a specialized AIC or AIV leaf function. It uses ordinary `Call`, `AssignStmt`,
`EvalStmt` and structured SSA control flow, with `SyncTokenType(pipe)` as a new
value type. The enclosing function supplies the core and execution-instance
identity; tokens do not carry a duplicated core attribute.

These operations support IR construction, verification and round trips. Hardware
lowering (`LowerSync`) is not implemented: code generation rejects unlowered
logical synchronization. This interface does not enable automatic insertion or
introduce a JIT option. Existing physical flag and FIFO operations remain distinct.

## Operations

The public namespace is `pl.sync`. `P` below denotes a concrete `pl.PipeType`;
`ALL` is forbidden. Each token is immutable and non-consuming.

| Signature | Result | Meaning |
| --------- | ------ | ------- |
| `record(pipe: PipeType)` | `SyncToken[pipe]` | Capture completion of the source pipeline's command prefix at this statement. |
| `completed(pipe: PipeType)` | `SyncToken[pipe]` | The already-satisfied, empty completion fact; it does not complete real work. |
| `join(first: SyncToken[P], second: SyncToken[P])` | `SyncToken[P]` | Conjunction: both facts must hold. Inputs have the same source pipeline and function execution. |
| `wait(token: SyncToken[P], dst_pipe: PipeType)` | `Void` | Order subsequent destination-pipeline commands after the captured completion. |
| `memory_overlaps(first, second, *, first_begin, first_end, second_begin, second_end)` | `BOOL` | Pure test of two runtime storage byte intervals. |
| `vf_barrier()` | `Void` | Complete preceding work in the current VF before subsequent VF work. |

`record` must be a direct assignment RHS. `wait` and `vf_barrier` are direct
`EvalStmt` calls, never assigned or nested as value operands. `completed`, `join`
and `memory_overlaps` are pure value operations.

## Command prefixes and waits

A record covers the complete prefix of its source execution domain, including
earlier waits imported into that domain. It excludes later submissions and work
on unrelated pipelines. A multi-stage operation's affected domains come from
its backend execution contract; textual precedence alone does not include it.

The following fragments are logical IR examples inside an AIV leaf function.
The comments stand for the original computation, which remains in the program.

```python
# Submit the MTE2 load of A.
old = pl.sync.record(pl.PipeType.MTE2)
# Submit the independent MTE2 load of B.
new = pl.sync.record(pl.PipeType.MTE2)
pl.sync.wait(old, pl.PipeType.V)
# V may now use A; this wait does not require B's completion.
pl.sync.wait(old, pl.PipeType.MTE3)
pl.sync.wait(old, pl.PipeType.V)
```

The same fact supports multiple targets and repeated waits. Later records or
physical resource reuse cannot change what `old` means. A token is not a flag
ID; successful waits do not consume its meaning. Hardware lowering must analyze
all uses and control flow rather than allocate one physical event per record.
Lowering must not widen an old prefix to include later work without a proof:
extra waiting can create a cycle around communication protocols.

```python
pl.sync.wait(old, pl.PipeType.V)
forwarded = pl.sync.record(pl.PipeType.V)
pl.sync.wait(forwarded, pl.PipeType.MTE3)
```

The second wait transitively includes `old`, even without an intervening V
computation. Same-pipeline waits are legal: ordered command submission does not
imply that each command completes before the next begins.

```python
pl.sync.wait(old, pl.PipeType.MTE2)
combined = pl.sync.join(old, new)
pl.sync.wait(combined, pl.PipeType.V)
```

`join` captures no new commands, performs no wait and allocates no event. For
same-pipeline tokens, conjunction is associative, commutative and idempotent;
`join(x, completed(P)) = x`. An optimization can select a later token only after
proving that its prefix includes the other's requirements.

`record` observes position-dependent execution state. Identical attributes do
not justify CSE or movement across commands, waits or control flow. A capture
whose result is entirely unused can be deleted. Waits and VF barriers retain
ordering effects even without tensor writes; pure token algebra does not make
them ordinary dead calls. `GetLogicalSyncKind` is the shared C++ identity query.

## Control flow and lifetime

Branch results select the actual path's token; they do not conjoin untaken
paths. Use `completed` only when that path has no corresponding dependency;
otherwise preserve the incoming token.

```python
if condition:
    # Submit the conditional MTE2 work.
    produced = pl.sync.record(pl.PipeType.MTE2)
    selected = pl.yield_(produced)
else:
    empty = pl.sync.completed(pl.PipeType.MTE2)
    selected = pl.yield_(empty)
pl.sync.wait(selected, pl.PipeType.V)
```

Loop-carried values identify the dynamic execution's completion facts. Zero
iterations return the initial fact; multiple iterations carry the previous one.

```python
initial = pl.sync.completed(pl.PipeType.V)
for i, (previous,) in pl.range(trips, init_values=(initial,)):
    pl.sync.wait(previous, pl.PipeType.MTE2)
    # Submit this iteration's work, with its other dependencies satisfied.
    current = pl.sync.record(pl.PipeType.V)
    final = pl.yield_(current)
pl.sync.wait(final, pl.PipeType.MTE3)
```

Tokens may pass through local SSA assignments, temporary SSA tuples, branch
results and loop carries. They cannot enter function parameters or results,
persistent containers, or a VF register scope. Recursive execution, cross-core
token transfer and token identity across function executions are not defined.
SSA dominance and source-pipeline agreement remain mandatory.

FIFO synchronization retains the native push/pop/free protocol. A borrowed slot
must not be freed before its asynchronous readers complete. Record those readers'
completion and place the release wait before `FREE`; a later independent `PUSH`
must not be pulled into that completion prefix merely to simplify lowering.
These are ordering requirements, not a general proof that every FIFO program is
deadlock-free or that a target has sufficient event resources.

## Runtime storage intervals

`memory_overlaps` accepts Tensor, Tile or Buffer values with the same **known**
address space. Their runtime data addresses are the interval origins. A current
`PtrType` or `MemRef` represents allocation identity, without a runtime-address
contract, and is rejected. An unknown Tile address space must be resolved first.

The four bounds are signed `INDEX` scalar SSA operands; Python integers become
INDEX constants. The low-level positional order is
`[first, second, first_begin, first_end, second_begin, second_end]`, with no
interval attributes. Bounds are byte offsets, and intervals are half-open.

```python
overlap = pl.sync.memory_overlaps(
    source, destination,
    first_begin=source_offset * 4,
    first_end=(source_offset + count) * 4,
    second_begin=destination_offset * 4,
    second_end=(destination_offset + count) * 4,
)
if overlap:
    pl.sync.wait(read_done, pl.PipeType.MTE3)
```

This FP32 example prevents an overlapping write from overtaking a read. The
predicate reads no payload, performs no wait and does not validate allocation
bounds. Empty intervals are disjoint. Inputs must satisfy `begin <= end`;
statically reversed constants are rejected and dynamic validity is a caller
precondition. Constants must fit signed INDEX. Address comparison must preserve
mathematical ordering without wrapping; overflow in earlier bound expressions
cannot be recovered by the predicate.

## VF boundaries

`vf_barrier` is legal only inside `pl.reg_scope()` in an AIV leaf function on an
A5 target. It completes preceding register/VMEM work before subsequent VF work.
Its first representation is a VF-wide boundary, not a per-register event.

```python
pl.sync.wait(input_ready, pl.PipeType.V)
with pl.reg_scope():
    value = pl.reg.load(tile, lanes=64)
    pl.sync.vf_barrier()
    doubled = pl.reg.add(value, value)
    pl.sync.vf_barrier()
    updated = pl.reg.store(tile, value=doubled)
vf_done = pl.sync.record(pl.PipeType.V)
pl.sync.wait(vf_done, pl.PipeType.MTE3)
```

The target must ensure that the outer V wait orders the VF's actual accesses,
and that the later V record includes completion of the full VF, including VMEM.
Syntactic scope exit alone is not an `ALL` drain. Lowering cannot assume that a
function's return completes asynchronous accesses without a target contract.
Removing an internal barrier requires preserving observable results, memory
effects and progress, not just register SSA uses.

## Representation and validation

Python exposes `ir.SyncTokenType(ir.PipeType.MTE2)` and the annotation
`pl.SyncToken[pl.PipeType.MTE2]`. C++ constructs the equivalent type with
`std::make_shared<ir::SyncTokenType>(ir::PipeType::MTE2)`. Reflection, structural
comparison, binary serialization and DSL printing preserve the source pipe.
Verification rechecks operator schemas and scope rules even for directly built
or deserialized calls. Tokens cannot be hidden in storage metadata, including
MemRef fields and WindowBuffer fields reached through distributed tensor types.
Shared expression DAGs are checked without expanding paths into trees; every
use still participates in lexical lifetime and execution-context validation.
See [types](02-types.md) and [operators](05-operators.md).
