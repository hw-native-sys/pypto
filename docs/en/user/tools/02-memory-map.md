# Memory Map

Seeing what is on chip, where, and for how long.

## Concept

On-chip buffers are the scarce resource a kernel competes for. When a tile does not fit,
the compiler tells you — but the interesting question usually comes earlier: *what is
holding the space?* `pypto.tools.memory_map` answers it by drawing the allocation as HTML,
with address across, lifetime down, and the IR beside it.

Its input is a **pass dump**, not a run. Nothing is executed; this is a picture of what the
compiler decided.

## Quickstart

Produce a dump, then render it:

```python
from pypto.ir import PassDumpLevel
from pypto.runtime import RunConfig

compiled = kernel.compile(*args, config=RunConfig(dump_passes=PassDumpLevel.EXPLICIT))
```

`compile()` prints nothing by itself, so have it tell you where the dumps went:

```python
print(compiled.output_dir)
```

Then point the tool at that directory:

```bash
OUT=build_output/<program>_<timestamp>          # what the line above printed
python -m pypto.tools.memory_map "$OUT/passes_dump/NN_after_SomePass.py" -o map.html
```

Two things matter here. **`compile()`, not `lower()`** — `lower()` runs the passes and hands
back the `Program` without writing anything, so it produces no `passes_dump/` for this tool
to read. And **`EXPLICIT`**, which resolves implicit tile layouts and window buffers; that
is what the tool needs to size what it draws.

## Mechanics

### Which dump to open

The allocation is decided late, so the dumps worth opening are the ones after the memory
passes:

| Dump | Shows |
| ---- | ----- |
| after `MaterializeSemanticAliases` | The must-alias relationships that are semantics, not optimization |
| after `MemoryReuse` | What the opportunistic reuse pass merged |
| after `AllocateMemoryAddr` | Final offsets — the picture most questions want |

Opening an earlier dump is not an error; it just shows an allocation that has not been
decided yet.

### Reading it

Two things are worth looking for, and neither is "is it full":

- **A tile alive far longer than it is used.** A long bar with a short span of actual reads
  is a candidate for restructuring — that lifetime is what blocks reuse.
- **The headroom.** Whether another `pl.pipeline` stage or a deeper cross-core ring will fit
  is a question about the gap, not about the total.

### The PTOAS caveat

Under `memory_planner=PTOAS` the compiler **skips `AllocateMemoryAddr`** and leaves
addressing to ptoas. The pass dump then carries no assigned offsets, and this tool has
nothing to draw. That is a property of the planner, not a failure — compare end to end
instead, and see [Memory](../performance/05-memory.md).

## Edge Cases

| Symptom | Cause | Fix |
| ------- | ----- | --- |
| **Empty or near-empty map** | Dump predates allocation, or `memory_planner=PTOAS` | Open a later dump; or accept that PTOAS owns addressing |
| **No `passes_dump/` at all** | `lower()` writes no artifacts without `dump_passes=` | Pass `dump_passes=PassDumpLevel.EXPLICIT` |
| **Layouts render as unresolved** | Dump was `CONCISE` | Re-dump with `EXPLICIT` |

## See Also

- [Memory](../performance/05-memory.md) — the runtime-side rings, and the on-chip budget this tool draws.
- [Tuning the InCore function](../performance/04-incore.md) — what consumes that budget.
- [Debugging](00-debugging.md) — the other reader of pass dumps.
- [AllocateMemoryAddr](../../dev/passes/35-allocate_memory_addr.md) — the pass whose output this is.
