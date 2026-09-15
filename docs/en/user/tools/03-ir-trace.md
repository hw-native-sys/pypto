# IR Trace

The whole lowering pipeline as one navigable diff: which pass changed what, in order.

## Concept

A pass dump is a directory of numbered snapshots. Diffing two adjacent ones attributes a
change to a pass — but doing that fifty times by hand is how an afternoon disappears.
`pypto-ir-trace` renders the entire sequence as a single self-contained HTML report, with
the passes that changed nothing filtered out.

Its input is the same `passes_dump/` the [memory map](02-memory-map.md) reads. Nothing is
executed; this is a picture of the text the compiler printed at each step.

## Quickstart

Produce a dump, then render it:

```python
from pypto.ir import PassDumpLevel
from pypto.runtime import RunConfig

compiled = kernel.compile(*args, config=RunConfig(dump_passes=PassDumpLevel.EXPLICIT))
print(compiled.output_dir)
```

```bash
OUT=build_output/<program>_<unique>          # what the line above printed
pypto-ir-trace "$OUT/passes_dump" -o ir_trace.html
```

Open `ir_trace.html` in a browser. Styles, scripts, and data are all embedded — no server,
no network.

A source checkout that only sets `PYTHONPATH` has no console script; the module entry point
takes the same arguments and exit codes:

```bash
python -m pypto.tools.ir_trace "$OUT/passes_dump" -o ir_trace.html
```

## Mechanics

### CLI

| Argument | Meaning |
| -------- | ------- |
| `passes_dump` | The dump directory: `00_frontend.py`, `01_after_*.py`, … with no index gaps |
| `-o PATH`, `--output PATH` | Report path (default `ir_trace.html`); the directory must already exist |
| `--context N` | Unchanged lines kept around each change (default `3`, non-negative) |

The report is written to a temp file in the destination directory and atomically renamed, so
a failed write never leaves a partial report. Exit codes: `0` success, `1` bad dump contents
or I/O failure, `2` argument errors.

### Reading it

The sidebar lists passes in execution order with inserted / deleted line counts and a
warning badge where the pass emitted one. **Changed** and **No-op** filters toggle
independently; the first changed pass is selected on open.

| Control | Does |
| ------- | ---- |
| `j` / `k`, arrow keys | Next / previous visible pass |
| **Side by side** / **Stacked** | Layout; panes scroll in sync, both axes |
| **Function** selector | Focus the diff on one top-level function or class method |
| **Copy full source** | Copy a whole snapshot, including context-folded lines |
| **Expand all** / **Collapse all** | Override `--context` folding |

**The Function selector is the one worth knowing on day one.** A pass that touches one
InCore function still re-prints the whole program; scoping the diff to that function is the
difference between a readable change and a wall of noise. The selection is retained across
passes when the function exists on both sides.

### Which dump level

`CONCISE` is usually the clearest for textual comparison. Use `EXPLICIT` when the question
involves tile layouts or distributed window buffers — those are implicit in `CONCISE` and
therefore invisible in the diff.

### ptoas has its own dumps

`dump_passes` observes PyPTO's pipeline. The backend's own MLIR pipeline is a separate
switch:

```python
RunConfig(dump_ptoas_passes=True)     # -> <output_dir>/ptoas_passes/<codegen-unit>/
```

One directory per codegen unit, so parallel ptoas invocations do not collide. The files
inside are named by ptoas/MLIR, and `pypto-ir-trace` does not read them. No effect under
`skip_ptoas=True`, where no ptoas pipeline runs.

## Edge Cases

| Symptom | Cause | Fix |
| ------- | ----- | --- |
| **`input directory does not exist`** | Pointed at the output dir, not `passes_dump/` | Append `/passes_dump` |
| **No `passes_dump/` at all** | `lower()` writes no artifacts; `dump_passes=False` | Use `compile()` with `dump_passes=` |
| **Every pass shows as No-op** | Dump has one snapshot, or the pipeline genuinely changed nothing | Check the dump has `NN_after_*` files beyond `00_frontend.py` |
| **Function selector disabled for a pass** | A snapshot could not be parsed safely | Fall back to Whole file for that pass |
| **Layouts render as unresolved** | Dumped at `CONCISE` | Re-dump with `EXPLICIT` |

> **A textual diff is not a semantic diff.** A changed report means the printed text
> changed; it does not prove the program's meaning did, and textual similarity does not
> prove it did not. Use the trace to *locate* a lowering step, then
> [torch codegen](01-torch-codegen.md) or `validate_ir` to decide whether it was correct.

## See Also

- [Memory map](02-memory-map.md) — the other reader of `passes_dump/`.
- [Debugging](00-debugging.md) — dump levels, and reading the lowered form.
- [Precision](../precision/00-workflow.md) — where bisecting passes fits in the narrowing procedure.
- [IR Lowering Trace](../../dev/07-ir-lower-trace.md) — the viewer's full behaviour and diff-alignment rules.
- [Pass manager](../../dev/passes/00-pass_manager.md) — what the numbered passes are.
