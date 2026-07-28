# IR Lowering Trace

`pypto-ir-trace` turns a PyPTO `passes_dump/` directory into a deterministic,
self-contained HTML report. The report compares each pass output with its input
so that lowering changes can be inspected without a web server or network access.

## Generate pass dumps

Enable per-pass dumps when compiling a program. `dump_passes=True` emits concise
canonical IR, which is usually the clearest input for textual comparison:

```python
from pypto import ir

ir.compile(MyProgram, output_dir="build/my_program", dump_passes=True)
```

Use `PassDumpLevel.EXPLICIT` when the trace must include fully resolved tile
layouts and distributed window-buffer references:

```python
from pypto import ir
from pypto.ir import PassDumpLevel

ir.compile(
    MyProgram,
    output_dir="build/my_program",
    dump_passes=PassDumpLevel.EXPLICIT,
)
```

Both forms create `build/my_program/passes_dump/`, containing
`00_frontend.py` and consecutively numbered `NN_after_PassName.py` snapshots.
See the [pass manager documentation](passes/00-pass_manager.md) for dump levels
and pass-pipeline behavior.

## Generate a report

Run the installed command with the dump directory:

```bash
pypto-ir-trace build/my_program/passes_dump
```

The default output is `ir_trace.html` in the current directory. Output is
written to a temporary file in the destination directory and atomically replaces
the requested path, so a failed write does not leave a partial report.

### CLI options

| Argument | Description |
| -------- | ----------- |
| `passes_dump` | Input directory containing the ordered pass snapshots. |
| `-o PATH`, `--output PATH` | Output report path; defaults to `ir_trace.html`. |
| `--context N` | Unchanged lines shown around each change; defaults to `3` and must be non-negative. |

For example, keep one unchanged line around changes and choose an explicit
destination:

```bash
pypto-ir-trace build/my_program/passes_dump --context 1 -o build/ir-trace.html
```

## Use the viewer

Open the generated HTML file in a browser. All styles, scripts, and trace data
are embedded in the file; it does not load external resources.

### Sidebar and filters

The sidebar lists passes in execution order with inserted/deleted line counts,
change status, and warning badges. **Changed** and **No-op** filters independently
show or hide passes that changed the printed IR or left it unchanged. The first
changed pass is selected initially, falling back to the first pass when every
pass is a no-op.

### Navigate and inspect

Select a pass in the sidebar to compare its input and output side by side. Press
`j` or `Down Arrow` to move to the next visible pass, and `k` or `Up Arrow` to
move to the previous visible pass. Keyboard navigation is ignored while a
text-entry control has focus.

Replacement rows use light delete/insert backgrounds, with the exact changed
characters emphasized in stronger red and green. Lines present on only one side
remain full-line deletions or insertions. Scrolling either code pane keeps the
before and after views synchronized vertically and horizontally.

### Copy snapshots

Use **Copy full source** above either pane to copy the complete before or after
snapshot, including unchanged lines hidden by context folding. Copying uses the
browser clipboard API when available and a local fallback otherwise.

### Warnings

If a snapshot has a matching `.log` file, its text appears in the warning panel
and the pass receives a warning badge. The warning is diagnostic context from
the pass run; it does not change the textual diff.

### Theme and collapsed context

Use **Theme** to switch between light and dark colors. The initial theme follows
the browser's preferred color scheme. Long unchanged regions are collapsed
according to `--context`; click an individual collapsed region, or use **Expand
all** and **Collapse all**, to change its visibility.

## Error handling

Argument syntax and invalid `--context` values use argparse diagnostics and exit
with status `2`. Invalid dump contents and input/output I/O failures print one
concise diagnostic to standard error and exit with status `1`. A report written
successfully exits with status `0`.

```text
$ pypto-ir-trace missing/passes_dump
pypto-ir-trace: error: input directory does not exist: missing/passes_dump

$ pypto-ir-trace passes_dump --context -1
pypto-ir-trace: error: argument --context: must be non-negative, got -1
```

The input directory must contain valid UTF-8 snapshots named
`00_frontend.py`, `01_after_*.py`, `02_after_*.py`, and so on without index gaps.
The output directory must already exist.

## Interpretation and limitations

The viewer computes a line-oriented **textual diff of printed IR**. A changed
report means the serialized text changed; it does not prove that program
semantics changed. Conversely, textual similarity is not a proof of semantic
equivalence. Use the trace to locate lowering steps, then use IR verification and
behavioral tests to decide semantic correctness.
