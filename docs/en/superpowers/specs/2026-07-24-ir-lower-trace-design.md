# Interactive IR Lower Trace Design

**Issue:** #2134
**Date:** 2026-07-24
**Status:** Approved design for the first implementation phase

## Summary

Add a `pypto-ir-trace` developer CLI that reads the snapshots already produced
under `<output_dir>/passes_dump/` and writes a self-contained interactive HTML
report. The report uses an IDE-style layout: a pass sidebar on the left and a
side-by-side before/after textual IR diff on the right.

The first phase implements only the CLI and HTML report. Terminal rendering and
a public Python API remain follow-up work. The compiler and `dump_passes`
instrumentation are unchanged.

## Goals

- Discover pass snapshots in execution order and pair each output with its
  preceding snapshot.
- Classify passes as changed or no-op and compute insertion/deletion counts.
- Generate one deterministic HTML file with no server, CDN, or sidecar assets.
- Make long pipelines easy to navigate and unchanged regions easy to fold.
- Attach an existing per-pass warning `.log` when present.
- Diagnose missing, malformed, duplicated, unreadable, or non-contiguous files.
- Treat diffs as textual evidence only, not semantic equivalence.

## Non-Goals

- No terminal diff mode in the first phase.
- No public `pypto.tools.ir_trace.render(...)` API in the first phase.
- No new pass instrumentation or changes to `PassManager.run_passes`.
- No hosted application, local HTTP server, or runtime network access.
- No semantic IR comparison.
- No `compare-codegen` integration in the first phase.

## User Interface

### Command

```bash
pypto-ir-trace build_output/example/passes_dump \
  --output ir_trace.html
```

The initial contract is:

```text
pypto-ir-trace PASSES_DUMP [-o OUTPUT] [--context LINES]
```

- `PASSES_DUMP` is an existing `dump_passes` snapshot directory.
- `-o` or `--output` defaults to `ir_trace.html` in the current directory.
- `--context` defaults to `3` and must be a non-negative integer.

The command returns `0` on success, `1` for input/output failures, and
argparse's `2` for invalid syntax. It has no `--mode` while only HTML exists.

### Report layout

The report has two persistent regions:

1. A pass sidebar with summary counts, changed/no-op filters, pass names,
   insertion/deletion counts, and warning indicators.
2. A main pane with the pass header, before/after filenames, aligned textual
   diff rows, copy actions, context controls, and warnings.

The first changed pass is selected on load. If every pass is a no-op, the first
pass is selected. Navigation operates only on visible passes after filtering.
If the current selection becomes hidden, the closest visible pass is selected.

Keyboard controls are:

- `j` or `ArrowDown` selects the next visible pass.
- `k` or `ArrowUp` selects the previous visible pass.

The initial theme follows the system light/dark preference and can be toggled.
At narrow widths, before and after panes stack vertically.

Copy Before and Copy After copy the complete source snapshot, not only visible
rows. Each folded block can be expanded independently, with global Expand All
and Collapse All controls.

## Architecture

Use a focused internal package:

```text
python/pypto/tools/ir_trace/
├── __init__.py
├── model.py
├── discovery.py
├── diff.py
├── html.py
└── cli.py
```

### `model.py`

Defines immutable internal records:

```python
@dataclass(frozen=True)
class Snapshot:
    index: int
    pass_name: str | None
    path: Path
    text: str
    lines: tuple[str, ...]
    warning_text: str | None


@dataclass(frozen=True)
class DiffRow:
    kind: Literal["equal", "insert", "delete", "replace"]
    before_number: int | None
    before_html: str
    after_number: int | None
    after_html: str


@dataclass(frozen=True)
class PassTrace:
    index: int
    name: str
    before: Snapshot
    after: Snapshot
    inserted: int
    deleted: int
    hunks: tuple[DiffHunk, ...]
```

The command is the first-phase product API. These records remain internal.

### `discovery.py`

Owns filesystem validation, decoding, ordering, and warning attachment. It
returns ordered snapshots and does not compute diffs.

### `diff.py`

Uses `difflib.SequenceMatcher(autojunk=False)` to compare adjacent snapshots,
compute statistics, align lines, identify foldable equal regions, and prepare
syntax-highlighted fragments.

### `html.py`

Serializes precomputed trace data into a complete HTML document. CSS,
JavaScript, highlighted rows, raw before/after text, warnings, and metadata are
embedded. Browser JavaScript manages presentation only and never computes diffs.

### `cli.py`

Owns argparse, diagnostics, exit codes, and atomic output. `pyproject.toml`
registers:

```toml
[project.scripts]
pypto-ir-trace = "pypto.tools.ir_trace.cli:main"
```

## Snapshot Discovery

The input uses the existing naming convention:

```text
passes_dump/
├── 00_frontend.py
├── 01_after_InlineFunctions.py
├── 01_after_InlineFunctions.log
├── 02_after_UnrollLoops.py
└── fa_fused_EXTRACT.py
```

Rules:

1. `PASSES_DUMP` must exist and be a directory.
2. `00_frontend.py` is required.
3. Pass files match `^(?P<index>\d+)_after_(?P<name>.+)\.py$`.
4. Indices start at `1`, are unique, and are contiguous.
5. Numeric-prefix Python files resembling malformed snapshots are errors.
6. Non-numeric artifacts such as `fa_fused_EXTRACT.py` are ignored.
7. A same-stem `.log` becomes the pass warning when present.
8. Files must be valid UTF-8. `splitlines()` normalizes CRLF/LF and ignores a
   final-newline-only difference.
9. At least one pass snapshot is required.

Pass `N` compares snapshot `N - 1` with snapshot `N`. Therefore the first pass
uses `00_frontend.py` as its input.

## Diff Semantics

`SequenceMatcher.get_opcodes()` provides line-level edit groups:

- `equal` contributes no statistics.
- `insert` contributes after lines to `inserted`.
- `delete` contributes before lines to `deleted`.
- `replace` contributes before lines to `deleted` and after lines to `inserted`.

Replacement lines align positionally and the shorter side receives blank cells.
A pass is no-op only when normalized line tuples are equal.

Equal runs retain `--context` lines around changes. Longer hidden middles become
`DiffHunk` fold controls containing omitted counts and expandable rows.

Syntax highlighting uses standard-library `tokenize` over the full snapshot so
multiline token positions remain correct. Token ranges become escaped per-line
HTML spans. `TokenError` or unsafe range mapping falls back to escaped plain
text and never blocks report generation.

## HTML Data and Security

The output is a complete HTML5 document with embedded CSS and JavaScript. Trace
data uses sorted JSON keys and stable separators. `<`, `>`, `&`, U+2028, and
U+2029 are escaped before embedding, so content such as `</script>` cannot end
the data block or inject markup.

All filenames, pass names, IR text, and warnings are escaped. Unsafe
`innerHTML` is not used except for trusted highlighter output whose token text
was already escaped. Ordinary labels and warnings use `textContent`.

The report contains no remote URL, external font, image, script, stylesheet, or
dynamic import. Clipboard support uses `navigator.clipboard.writeText` with a
temporary local textarea fallback for restricted `file://` contexts.

Absolute paths and timestamps are excluded for deterministic output. Only
snapshot basenames and a generic source label are displayed.

## Error Handling

Domain failures use internal `IRTraceError`. `cli.main()` catches it, writes one
actionable message to stderr, and returns `1`. Programming errors are not hidden.

Examples:

```text
pypto-ir-trace: error: missing 00_frontend.py in build_output/example/passes_dump
pypto-ir-trace: error: no pass snapshots found in build_output/example/passes_dump
pypto-ir-trace: error: missing snapshot index 02 between 01_after_InlineFunctions.py and 03_after_ConvertToSSA.py
pypto-ir-trace: error: malformed snapshot name 02_ConvertToSSA.py; expected NN_after_PassName.py
pypto-ir-trace: error: 03_after_Simplify.py is not valid UTF-8
```

Output is written to a unique temporary sibling, flushed, and atomically
replaced with `Path.replace()`. Failed writes remove the temporary file.

## Testing

Add `tests/ut/tools/test_ir_trace.py` with pytest functions and the required
`pytest.main([__file__, "-v"])` footer.

Discovery coverage includes ordering, missing frontend, no passes, duplicate
and missing indices, malformed numeric names, ignored unrelated files, warning
attachment, and invalid UTF-8.

Diff coverage includes insert/delete/replace/no-op statistics, unequal
replacement alignment, context zero and boundary folding, newline
normalization, syntax highlighting, and safe fallback.

HTML coverage includes byte-for-byte determinism, embedded metadata, safe
`</script>` handling, no remote assets, filters, keyboard navigation, theme,
warnings, context controls, and clipboard fallback.

CLI coverage includes default and explicit output, invalid context, domain
errors, exit codes, and cleanup after failed writes.

Implementation verification runs:

```bash
PYTHONPATH="$(pwd)/python" python3.11 -m pytest tests/ut/tools/test_ir_trace.py -v
ruff check python/pypto/tools/ir_trace tests/ut/tools/test_ir_trace.py
ruff format --check python/pypto/tools/ir_trace tests/ut/tools/test_ir_trace.py
pyright
```

## Documentation

Implementation adds synchronized usage documentation:

- `docs/en/dev/07-ir-lower-trace.md`
- `docs/zh-cn/dev/07-ir-lower-trace.md`

It covers producing `passes_dump/`, invoking the CLI, using report controls,
interpreting statistics, and the textual-evidence limitation. Existing
pass-dump documentation remains authoritative for `PassDumpLevel`.

## Implementation Files

- `pyproject.toml`: register the console script.
- `python/pypto/tools/ir_trace/__init__.py`: declare the internal package.
- `python/pypto/tools/ir_trace/model.py`: immutable records.
- `python/pypto/tools/ir_trace/discovery.py`: discovery and validation.
- `python/pypto/tools/ir_trace/diff.py`: diffing, folding, and highlighting.
- `python/pypto/tools/ir_trace/html.py`: deterministic HTML renderer.
- `python/pypto/tools/ir_trace/cli.py`: command and atomic output.
- `tests/ut/tools/test_ir_trace.py`: unit and CLI tests.
- English and Chinese usage documents listed above.

No C++, binding, type-stub, pass-pipeline, or codegen file changes.

## Implementation Order

1. Add failing discovery tests, then implement `model.py` and `discovery.py`.
2. Add failing diff/statistics/folding tests, then implement `diff.py`.
3. Add failing escaping/determinism/report tests, then implement `html.py`.
4. Add failing CLI tests, implement `cli.py`, and register the script.
5. Add synchronized English and Chinese documentation.
6. Run focused tests, lint, formatting, type checking, and commit workflow.

Each step depends only on the preceding internal interface, supporting focused
test-driven iterations and keeping presentation separate from analysis.

## Rejected Alternatives

### Compute diffs in browser JavaScript

This moves correctness, statistics, and performance into harder-to-test client
code and increases browser work for large snapshots. Precomputation is preferred.

### Vendor a JavaScript diff or highlighting library

This adds versioning, licensing, bundle-size, and offline-packaging concerns.
The standard library is sufficient for the first phase.

### Put everything in one `ir_trace.py`

This mixes filesystem validation, edit semantics, HTML security, and CLI
behavior. Focused modules provide clearer contracts and isolated tests.

## Follow-Ups

Future work may add terminal rendering, a public Python API, and
`compare-codegen` integration. These should reuse the discovery and diff model
instead of introducing another snapshot path.
