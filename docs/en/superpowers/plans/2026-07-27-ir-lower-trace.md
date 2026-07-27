# IR Lower Trace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `pypto-ir-trace` CLI that converts existing `passes_dump/` snapshots into an offline, self-contained interactive HTML lowering trace.

**Architecture:** Python performs discovery, textual diffing, statistics, folding, and syntax highlighting at generation time. The browser only renders precomputed data and manages interaction. Focused model, discovery, diff, HTML, and CLI modules communicate through immutable dataclasses.

**Tech Stack:** Python 3.10+ standard library (`argparse`, `dataclasses`, `difflib`, `json`, `pathlib`, `tempfile`, `tokenize`), embedded HTML/CSS/JavaScript, pytest, ruff, and pyright.

## Global Constraints

- Phase one provides only an HTML CLI; no terminal mode or public Python API.
- Do not modify `PassManager.run_passes`, C++, bindings, type stubs, pass pipeline, or codegen.
- Add no runtime dependency. Reports load no CDN, remote script, font, image, or stylesheet.
- Diffs are textual evidence only and do not imply semantic equivalence.
- Exclude absolute paths, timestamps, and machine state; identical input produces byte-identical output.
- Escape all input safely, including `</script>`, `<`, `>`, `&`, U+2028, and U+2029.
- Use modern Python typing, f-strings, Google-style docstrings, and pytest conventions.
- Before Task 1, self-assign issue #2134 and set its project status to In Progress.

---

### Task 1: Snapshot Models and Discovery

**Files:**

- Create: `python/pypto/tools/ir_trace/{__init__,model,discovery}.py`
- Create: `tests/ut/tools/test_ir_trace.py`

**Interfaces:**

- Consumes: a user-provided `passes_dump/` as `Path`.
- Produces: `discover_snapshots(directory: Path) -> tuple[Snapshot, ...]`.
- Produces: `IRTraceError`, `Snapshot`, `DiffRow`, `DiffHunk`, and `PassTrace`.

- [ ] **Step 1: Add the package and failing happy-path discovery test**

```python
from pathlib import Path
import pytest
from pypto.tools.ir_trace.discovery import discover_snapshots
from pypto.tools.ir_trace.model import IRTraceError
def _write_dump(root: Path, files: dict[str, str]) -> Path:
    dump = root / "passes_dump"
    dump.mkdir()
    for name, text in files.items():
        (dump / name).write_text(text, encoding="utf-8")
    return dump
def test_discover_orders_snapshots_and_attaches_warning(tmp_path: Path):
    dump = _write_dump(tmp_path, {
        "02_after_UnrollLoops.py": "after two\n", "00_frontend.py": "frontend\n",
        "01_after_InlineFunctions.log": "unused variable\n",
        "01_after_InlineFunctions.py": "after one\n", "fa_fused_EXTRACT.py": "ignored\n",
    })
    snapshots = discover_snapshots(dump)
    assert [item.index for item in snapshots] == [0, 1, 2]
    assert [item.pass_name for item in snapshots] == [None, "InlineFunctions", "UnrollLoops"]
    assert snapshots[1].warning_text == "unused variable\n"
```

- [ ] **Step 2: Run the test and verify failure**

```bash
PYTHONPATH="$(pwd)/python" python3.11 -m pytest \
  tests/ut/tools/test_ir_trace.py::test_discover_orders_snapshots_and_attaches_warning -v
```

Expected: FAIL because `pypto.tools.ir_trace` does not exist.

- [ ] **Step 3: Implement immutable models**

```python
class IRTraceError(ValueError):
    """Report an actionable IR trace input or output error."""
@dataclass(frozen=True)
class Snapshot:
    index: int
    pass_name: str | None
    path: Path
    text: str
    lines: tuple[str, ...]
    warning_text: str | None = None
@dataclass(frozen=True)
class DiffRow:
    kind: Literal["equal", "insert", "delete", "replace"]
    before_number: int | None
    before_html: str
    after_number: int | None
    after_html: str
@dataclass(frozen=True)
class DiffHunk:
    rows: tuple[DiffRow, ...]
    collapsed: bool
@dataclass(frozen=True)
class PassTrace:
    index: int
    name: str
    before: Snapshot
    after: Snapshot
    inserted: int
    deleted: int
    hunks: tuple[DiffHunk, ...]
    @property
    def changed(self) -> bool:
        return self.inserted != 0 or self.deleted != 0
```

Export only `IRTraceError` from `__init__.py`; internal dataclasses are not a stable public API.

- [ ] **Step 4: Implement discovery**

Use `_PASS_RE = re.compile(r"^(?P<index>\d+)_after_(?P<name>.+)\.py$")` and
`_NUMERIC_PY_RE = re.compile(r"^\d+_.*\.py$")`. Validate directory existence/type,
require `00_frontend.py`, reject malformed numeric names, duplicate or missing indices,
and require indices starting at one. Ignore nonnumeric artifacts. `_read_utf8()` converts
`UnicodeDecodeError` to `IRTraceError(f"{path.name} is not valid UTF-8")`. Attach a
same-stem `.log` and populate `lines=tuple(text.splitlines())`.

- [ ] **Step 5: Add discovery error tests**

Parameterize missing directory, non-directory, missing frontend, no passes, start at two,
index gap, duplicate index, malformed `02_ConvertToSSA.py`, invalid snapshot UTF-8, and
invalid warning UTF-8. Assert the relevant filename or missing index in each message.

- [ ] **Step 6: Run Task 1 checks and commit**

```bash
PYTHONPATH="$(pwd)/python" python3.11 -m pytest tests/ut/tools/test_ir_trace.py -k discover -v
ruff check python/pypto/tools/ir_trace tests/ut/tools/test_ir_trace.py
git add python/pypto/tools/ir_trace tests/ut/tools/test_ir_trace.py
git commit -m "feat(tools): Discover IR pass snapshots"
```

---

### Task 2: Text Diffing, Statistics, Folding, and Highlighting

**Files:**

- Create: `python/pypto/tools/ir_trace/diff.py`
- Modify: `tests/ut/tools/test_ir_trace.py`

**Interfaces:**

- Consumes: Task 1 snapshots and non-negative context.
- Produces: `build_trace(snapshots: tuple[Snapshot, ...], context: int) -> tuple[PassTrace, ...]`.
- Internal: `highlight_python(text: str) -> tuple[str, ...]` returns safe per-line HTML.

- [ ] **Step 1: Add the failing statistics/alignment test**

```python
from pypto.tools.ir_trace.diff import build_trace
def test_build_trace_counts_and_aligns_replace(tmp_path: Path):
    dump = _write_dump(tmp_path, {
        "00_frontend.py": "a\nb\nc\n", "01_after_TestPass.py": "a\nx\ny\nc\n",
    })
    trace = build_trace(discover_snapshots(dump), context=3)[0]
    assert (trace.inserted, trace.deleted, trace.changed) == (2, 1, True)
    rows = [row for hunk in trace.hunks for row in hunk.rows if row.kind == "replace"]
    assert [(row.before_number, row.after_number) for row in rows] == [(2, 2), (None, 3)]
```

- [ ] **Step 2: Run it and verify the missing-module failure**

- [ ] **Step 3: Implement safe Python highlighting**

Use `tokenize.generate_tokens(io.StringIO(text).readline)` over the full source. Style only
keyword `NAME`, `STRING`, `NUMBER`, `COMMENT`, and `OP`; escape token and gap text with
`html.escape(..., quote=False)`. On `TokenError`, `IndentationError`, or invalid token
positions, return directly escaped lines.

- [ ] **Step 4: Implement opcode alignment and counts**

```python
matcher = difflib.SequenceMatcher(a=before.lines, b=after.lines, autojunk=False)
for tag, before_start, before_end, after_start, after_end in matcher.get_opcodes():
    before_count = before_end - before_start
    after_count = after_end - after_start
    if tag == "insert": inserted += after_count
    elif tag == "delete": deleted += before_count
    elif tag == "replace": inserted += after_count; deleted += before_count
```

Align `equal` one-to-one, leave the opposite side blank for insert/delete, and pad the
shorter replace side to `max(before_count, after_count)`. Line numbers are one-based.

- [ ] **Step 5: Implement folding and `build_trace()`**

Fold only equal runs. Between changes, keep `context` lines at both ends when the run is
longer than `2 * context`; at file boundaries keep context only near the change. Put hidden
rows in `DiffHunk(collapsed=True)`. With context zero, fold the entire equal middle. Reject
negative context with `IRTraceError(f"context must be non-negative, got {context}")`.

- [ ] **Step 6: Add edge tests**

Cover insert, delete, no-op, CRLF/LF, final-newline-only changes, file boundaries,
context zero, short/long equal runs, `<script>` highlighting, and invalid Python fallback.

- [ ] **Step 7: Run Task 2 checks and commit**

```bash
PYTHONPATH="$(pwd)/python" python3.11 -m pytest tests/ut/tools/test_ir_trace.py -k "trace or highlight or fold" -v
ruff check python/pypto/tools/ir_trace tests/ut/tools/test_ir_trace.py
git add python/pypto/tools/ir_trace/diff.py tests/ut/tools/test_ir_trace.py
git commit -m "feat(tools): Analyze IR pass differences"
```

---

### Task 3: Deterministic Self-Contained HTML

**Files:**

- Create: `python/pypto/tools/ir_trace/html.py`
- Modify: `tests/ut/tools/test_ir_trace.py`

**Interfaces:**

- Produces: `render_html(traces: tuple[PassTrace, ...], source_name: str) -> str`.
- It returns a complete HTML5 string and performs no filesystem I/O.

- [ ] **Step 1: Add the failing determinism and safety test**

```python
from pypto.tools.ir_trace.html import render_html
def test_render_html_is_deterministic_self_contained_and_safe(tmp_path: Path):
    dump = _write_dump(tmp_path, {
        "00_frontend.py": "value = '</script><b>'\n",
        "01_after_TestPass.py": "value = '<script>'\n",
        "01_after_TestPass.log": "warning </script>\n",
    })
    traces = build_trace(discover_snapshots(dump), context=3)
    first = render_html(traces, source_name="passes_dump")
    assert first == render_html(traces, source_name="passes_dump")
    assert first.startswith("<!doctype html>")
    assert "http://" not in first and "https://" not in first
    assert "</script><b>" not in first and "\\u003c/script\\u003e" in first
```

- [ ] **Step 2: Run it and verify the missing-module failure**

- [ ] **Step 3: Implement deterministic payload embedding**

Payload contains only basenames, index/name, changed/counts, warnings, complete before/after
text, and hunk/row data. Serialize with sorted keys and stable separators. Translate `<`,
`>`, `&`, U+2028, and U+2029 to JSON escapes. Embed under
`<script id="trace-data" type="application/json">`.

- [ ] **Step 4: Implement IDE layout and CSS**

Use stable DOM ids `pass-list`, `changed-filter`, `noop-filter`, `summary`, `pass-title`,
`before-pane`, `after-pane`, `warnings-panel`, `copy-before`, `copy-after`, `expand-all`,
`collapse-all`, and `theme-toggle`. Desktop grid is `18rem minmax(0, 1fr)` with equal diff
columns; under 800px, use one column. Define light/dark CSS variables.

- [ ] **Step 5: Implement browser interaction**

Implement `visiblePasses`, `selectPass`, `renderSidebar`, `renderDiff`, `copySnapshot`,
`setAllHunks`, and `toggleTheme`. Build rows through DOM APIs. Clipboard falls back to a
temporary textarea. Keyboard navigation ignores focused inputs/buttons and traverses only
visible passes. Select the first changed pass, or the first pass when all are no-op.

- [ ] **Step 6: Add structure/interaction contract tests**

Assert every DOM id, key, dark-mode query, fallback, filter, warning, and fold function;
assert payload counts and absence of absolute paths/timestamps.

- [ ] **Step 7: Run Task 3 checks and commit**

```bash
PYTHONPATH="$(pwd)/python" python3.11 -m pytest tests/ut/tools/test_ir_trace.py -k html -v
ruff check python/pypto/tools/ir_trace tests/ut/tools/test_ir_trace.py
git add python/pypto/tools/ir_trace/html.py tests/ut/tools/test_ir_trace.py
git commit -m "feat(tools): Render interactive IR trace HTML"
```

---

### Task 4: CLI, Atomic Output, Documentation, and Verification

**Files:**

- Create: `python/pypto/tools/ir_trace/cli.py`
- Modify: `pyproject.toml:38`, `tests/ut/tools/test_ir_trace.py`
- Create: `docs/{en,zh-cn}/dev/07-ir-lower-trace.md`

**Interfaces:**

- Consumes: `discover_snapshots()`, `build_trace()`, and `render_html()`.
- Produces: `main(argv: Sequence[str] | None = None) -> int` and `pypto-ir-trace`.

- [ ] **Step 1: Add failing CLI success/error tests**

```python
from pypto.tools.ir_trace.cli import main
def test_cli_writes_default_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dump = _write_dump(tmp_path, {"00_frontend.py": "a\n", "01_after_TestPass.py": "b\n"})
    monkeypatch.chdir(tmp_path)
    assert main([str(dump)]) == 0
    assert (tmp_path / "ir_trace.html").read_text(encoding="utf-8").startswith("<!doctype html>")
def test_cli_reports_domain_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    assert main([str(tmp_path / "missing")]) == 1
    assert "pypto-ir-trace: error: input directory does not exist" in capsys.readouterr().err
```

- [ ] **Step 2: Run them and verify the missing-module failure**

- [ ] **Step 3: Implement argparse and atomic output**

`_non_negative_int()` converts with `int()`, rejects negatives through
`ArgumentTypeError`, and returns the value. `_write_atomic()` validates the parent directory,
uses a UTF-8 `NamedTemporaryFile(delete=False)` in that directory, writes and closes it,
then calls `Path.replace()`. On `OSError`, unlink the temporary path and raise
`IRTraceError(f"failed to write {output}: {error}")`.

`main()` uses `ArgumentParser(prog="pypto-ir-trace")`, positional `passes_dump: Path`,
`-o/--output: Path` defaulting to `Path("ir_trace.html")`, and context defaulting to three.
Catch `IRTraceError`, print `pypto-ir-trace: error: {error}` to stderr, and return one.

- [ ] **Step 4: Register the command and complete CLI tests**

```toml
[project.scripts]
pypto-ir-trace = "pypto.tools.ir_trace.cli:main"
```

Add explicit output, negative context returning two, missing output directory, replace
failure, and temporary cleanup tests. End the test file with:

```python
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

- [ ] **Step 5: Run focused tests**

```bash
PYTHONPATH="$(pwd)/python" python3.11 -m pytest tests/ut/tools/test_ir_trace.py -v
```

- [ ] **Step 6: Write synchronized usage docs**

English is authoritative and Chinese mirrors its headings. Cover producing dumps with
`dump_passes=True` or `PassDumpLevel.EXPLICIT`, CLI arguments, sidebar, filters, keyboard,
copy, warnings, themes, folding, diagnostics, the textual-diff limitation, and a same-language
link to `passes/00-pass_manager.md`.

- [ ] **Step 7: Run static and tools regression checks**

```bash
ruff check python/pypto/tools/ir_trace tests/ut/tools/test_ir_trace.py
ruff format --check python/pypto/tools/ir_trace tests/ut/tools/test_ir_trace.py
PYTHONPATH="$(pwd)/python" python3.11 -m pytest tests/ut/tools -v
pyright
```

- [ ] **Step 8: Run full unit tests and pre-commit**

```bash
PYTHONPATH="$(pwd)/python" python3.11 -m pytest tests/ut/ -n auto --maxprocesses 8 -v
pre-commit run --all-files
```

Use `superpowers:systematic-debugging` for failures; never alter unrelated expectations.

- [ ] **Step 9: Commit Task 4**

```bash
git add pyproject.toml python/pypto/tools/ir_trace/cli.py tests/ut/tools/test_ir_trace.py \
  docs/en/dev/07-ir-lower-trace.md docs/zh-cn/dev/07-ir-lower-trace.md
git commit -m "feat(tools): Add IR lower trace CLI"
```

- [ ] **Step 10: Complete final PyPTO review**

Use `code-review`, `testing`, and `verification-before-completion`. Confirm clean status and
issue-only commits, then prepare a PR description containing `Fixes #2134`.
