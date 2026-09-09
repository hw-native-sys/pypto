# Operator effects

This reference covers Functional-stage argument effects, write channels, and
core placement/replication contracts. Buffer-stage operations use the separate
[data/metadata effect contract](02-types.md#buffer-operator-contracts).

> The whole chain that consumes these declarations is laid out in
> [Parameter Direction Inference](08-param-directions.md).

An operator that updates one of its arguments in place must say so. Direction
inference, dependency analysis and the parameter-direction verifier all ask the
registry the same question — *does this call write the buffer this argument
names?* — and an operator that never answered reads as a pure consumer:

```text
tile.mscatter writes output_tensor, but never declared it
  → the parameter it writes keeps direction In
  → no RAW edge is emitted against the kernel that reads it
  → the scheduler is free to run the reader first
  → stale data, or a deadlock waiting on a signal nobody wrote
```

| Effect | Meaning | Examples |
| ------ | ------- | -------- |
| `ArgEffect::Read` | Read, never written. Default for an unnamed argument | `tile.store`'s source tile |
| `ArgEffect::Write` | Overwritten without being read first | `tile.store`'s `output_tensor`, `pld.tile.get`'s `dst` |
| `ArgEffect::ReadWrite` | Read *and* written | `tile.matmul_acc`'s accumulator, an atomic store's destination |

**Partial overwrite is still `Write`.** A store that lands on a sub-region does
not read the untouched remainder — nothing moves *into* the kernel — so its
destination is a pure write. Declaring it `ReadWrite` is not a harmless
approximation: it makes the enclosing parameter `InOut`, which stages the buffer
host→device and, across ranks, invents a dependency between two ranks writing
disjoint rows.

Whether a destination is read is decided per operator, not per family: a
gather or exchange destination is pushed into and never loaded from, so it is
`Write`, while a reduce destination has its running value loaded back and is
`ReadWrite`.

**`ReadWrite` is for operators that genuinely read the slot**: an accumulator
(`out += x` reads the running sum), an atomic store or assemble, or a
destination-passing operator whose untouched positions flow through to its SSA
result (`tile.scatter`, `array.update_element`).

**Kwarg-dependent effects.** When a kwarg decides the answer, pass a resolver
instead of a constant. A kwarg can decide *whether* an argument is written at
all, not only how: `tile.mgather`'s third operand is a written GM scratch tensor
in Mat element mode and a read-only `valid_shape` in Mat row mode. The other
live cases are the `atomic` kwarg on the store family and the `op` kwarg on
`pld.system.notify`, whose default is atomic-add — so an unannotated notify
reads the slot it adds into:

```cpp
REGISTER_OP("tile.store")
    // ... arguments, memory spec ...
    .set_arg_effect(2,
                    [](const std::vector<std::pair<std::string, std::any>>& kwargs) {
                      return GetIntKwarg(kwargs, "atomic", static_cast<int>(AtomicType::kNone)) ==
                                     static_cast<int>(AtomicType::kNone)
                                 ? ArgEffect::Write
                                 : ArgEffect::ReadWrite;
                    })
    .set_write_channel(WriteChannel::Dma)
```

**Declared-read-only is not the same as unclassified.** `HasDeclaredArgEffects()`
distinguishes "a human decided this operator writes nothing" (`no_arg_writes()`,
e.g. `pld.system.wait`) from "nobody has looked at this operator yet". An
analysis that needs the answer can then refuse to guess instead of defaulting an
unclassified writer to read-only.

**Enforcement.** `OpRegistry::ValidateArgEffects()` runs at import and rejects
two shapes, naming every offender and the fix rather than failing on first use:

- an operator declaring `set_output_reuses_input(N)` — its SSA result *is*
  argument N's buffer, so it writes through it — without a verdict about
  argument N specifically. Classification is what is required, not a particular
  answer: an operator whose in-place slot is metadata may declare it read-only.
- an operator declaring a write channel while writing through no argument. A
  channel says *how* an operator writes, so one without a write is either a
  stray declaration or a missing one.

The second rule matters more than it looks. `set_write_channel()` creates the
effect spec as a side effect, so "the spec exists" cannot stand in for "a human
decided" — otherwise an operator that declared a channel and forgot its
`set_arg_effect` would pass the first rule with the argument it updates still
defaulting to `Read`. `no_arg_writes()` records the all-arguments verdict
explicitly, and combining it with `set_arg_effect` is rejected as
contradictory.

`set_write_channel` records whether the writes travel the MTE3/DMA path or the
scalar D-cache path. PyPTO cannot order the two against one GM tensor, so a
function mixing them on one buffer is rejected; the channel lets that diagnostic
read the registry instead of re-listing operators.

Declare it only for an operator whose writes really are one of those two paths,
and leave it unset otherwise — an unset channel keeps the operator out of that
diagnostic, which is where an operator belongs when neither path describes it:

- `pld.system.notify` emits `pto.comm.tnotify`, a distinct comm instruction.
  Claiming either channel would let the diagnostic reject a valid program.
- `system.set_ffts` hands the workspace *pointer* to the FFTS unit rather than
  moving data; the hardware writes that region on its own schedule, which no
  dependency edge models. It declares `no_arg_writes()`.
- A composite collective updates a data window and a signal through different
  mechanisms, and one operator-level channel cannot describe both. Per-argument
  channels would, but no case yet needs the distinction, and a wrong single
  answer is worse than none.

**`set_core_affinity` vs `set_no_duplicate`** — two orthogonal axes, and picking
the wrong one makes a false claim about the ISA:

- `set_core_affinity(...)` answers *which* core runs the op. Declare it only
  when the hardware really constrains the op to one side. When left unset,
  `ClassifyCallAffinity` derives placement from the call (memory spec, then the
  first tile argument's memory space), falling back to `SHARED`.
- `set_no_duplicate()` answers whether the op may run on a *second* core.
  `ExpandMixedKernel` replicates `SHARED` statements onto both the AIC and the
  AIV lane; mark an op for which that copy changes what the program means.

`pld.system.notify` is the canonical case, and the hazard is **premature release
from the wrong lane**, not non-idempotence: the AIC copy can publish the signal
before the AIV lane's TPUT has landed the data that signal releases, so the peer
reads stale bytes. That is why the flag is unconditional — a `NotifyOp::kSet`
fires the same race as an atomic-add, even though only the atomic-add
double-counts.

Its sibling `pld.system.wait` stays **unmarked**, and deliberately so: TWAIT
*blocks*, and its presence on the cube lane is load-bearing. Pinning it to the
vector lane would let the matmul race past the peer data it waits on. Read the
flag as "must not run on a second core", not as "not idempotent".

`IsNoDuplicate()` reads the axis. Its only consumer is `LowerAutoVectorSplit`'s
`pl.split_aiv` region placement stamp (pass 23), which pins exactly the
no-duplicate calls inside a region to the AIV lane. No verifier rejects anything
on this axis.

An op pinned to one lane by `set_core_affinity(...)` is never duplicated in the
first place and needs no flag. The flag exists precisely for the core-agnostic
ops, where no affinity value can express "may run on either core, but must not
run on both". Note that is a claim about *cores*, not about total executions: an
AIV function body still runs on both AIV sub-lanes under `dual_aiv_dispatch`, so
keeping an op off the cube lane does not make it happen once. That part is the
author's, and is documented in
[Scopes → pl.split_aiv](../../user/language/04-scopes.md).

For Buffer-stage calls, the shared `CallWriteTargets` query projects only the
registered data effects. `InParamWritten` therefore still diagnoses writes to
GM or Buffer parameters declared `In` after final lowering. A descriptor-only
`buffer.set_validshape` update does not count as a parameter data write.
