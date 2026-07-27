# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for IR pass snapshot discovery."""

import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest
from pypto.tools.ir_trace.diff import build_trace, highlight_python
from pypto.tools.ir_trace.discovery import discover_snapshots
from pypto.tools.ir_trace.html import render_html
from pypto.tools.ir_trace.model import IRTraceError


def _write_dump(root: Path, files: dict[str, str]) -> Path:
    dump = root / "passes_dump"
    dump.mkdir()
    for name, text in files.items():
        (dump / name).write_text(text, encoding="utf-8")
    return dump


def _run_viewer_behavior(report: str, assertions: str) -> subprocess.CompletedProcess[str]:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to exercise the embedded viewer behavior")

    payload = report.split('<script id="trace-data" type="application/json">', 1)[1].split("</script>", 1)[0]
    viewer_script = report.rsplit("<script>", 1)[1].split("</script>", 1)[0]
    harness = textwrap.dedent(
        f"""
        class Element {{
          constructor(tagName = "div") {{
            this.tagName = tagName.toUpperCase();
            this.checked = true;
            this.children = [];
            this.className = "";
            this.dataset = {{}};
            this.disabled = false;
            this.hidden = false;
            this.listeners = {{}};
            this.style = {{}};
            this.textContent = "";
          }}
          addEventListener(name, callback) {{ this.listeners[name] = callback; }}
          appendChild(child) {{ this.children.push(child); return child; }}
          remove() {{}}
          replaceChildren(...children) {{ this.children = children; }}
          select() {{}}
          setAttribute(name, value) {{ this[name] = value; }}
        }}

        const ids = [
          "trace-data", "source-name", "changed-filter", "noop-filter", "pass-list", "summary",
          "pass-title", "before-pane", "after-pane", "before-title", "after-title", "warnings-panel",
          "copy-before", "copy-after", "expand-all", "collapse-all", "theme-toggle"
        ];
        const elements = Object.fromEntries(ids.map((id) => [id, new Element()]));
        elements["trace-data"].textContent = {json.dumps(payload)};
        const documentListeners = {{}};
        const document = {{
          body: new Element("body"),
          documentElement: new Element("html"),
          addEventListener(name, callback) {{ documentListeners[name] = callback; }},
          createElement(tagName) {{ return new Element(tagName); }},
          execCommand() {{ throw new Error("copy fallback must not run without a selected trace"); }},
          getElementById(id) {{ return elements[id]; }}
        }};
        const window = {{ matchMedia() {{ return {{ matches: false }}; }} }};
        Object.defineProperty(
          globalThis,
          "navigator",
          {{ value: {{ clipboard: null }}, configurable: true }}
        );

        {viewer_script}
        {assertions}
        """
    )
    return subprocess.run([node, "-e", harness], check=False, capture_output=True, text=True)


def test_build_trace_counts_and_aligns_replace(tmp_path: Path):
    dump = _write_dump(
        tmp_path,
        {
            "00_frontend.py": "a\nb\nc\n",
            "01_after_TestPass.py": "a\nx\ny\nc\n",
        },
    )

    trace = build_trace(discover_snapshots(dump), context=3)[0]

    assert (trace.inserted, trace.deleted, trace.changed) == (2, 1, True)
    assert trace.changed
    assert len(trace.hunks) == 1
    assert not trace.hunks[0].collapsed
    changed_rows = [row for hunk in trace.hunks for row in hunk.rows if row.kind == "replace"]
    assert [(row.before_number, row.after_number) for row in changed_rows] == [(2, 2), (None, 3)]


@pytest.mark.parametrize(
    ("before", "after", "kind", "before_number", "after_number"),
    [
        ("a\nc\n", "a\nb\nc\n", "insert", None, 2),
        ("a\nb\nc\n", "a\nc\n", "delete", 2, None),
    ],
)
def test_build_trace_aligns_insert_and_delete(
    tmp_path: Path,
    before: str,
    after: str,
    kind: str,
    before_number: int | None,
    after_number: int | None,
):
    dump = _write_dump(
        tmp_path,
        {
            "00_frontend.py": before,
            "01_after_TestPass.py": after,
        },
    )

    trace = build_trace(discover_snapshots(dump), context=3)[0]

    rows = [row for hunk in trace.hunks for row in hunk.rows if row.kind == kind]
    assert [(row.before_number, row.after_number) for row in rows] == [(before_number, after_number)]
    assert (trace.inserted, trace.deleted) == (int(kind == "insert"), int(kind == "delete"))


def test_build_trace_keeps_noop_and_normalizes_line_endings(tmp_path: Path):
    dump = _write_dump(
        tmp_path,
        {
            "00_frontend.py": "a\r\nb\r\n",
            "01_after_TestPass.py": "a\nb",
        },
    )

    trace = build_trace(discover_snapshots(dump), context=3)[0]

    assert not trace.changed
    assert (trace.inserted, trace.deleted) == (0, 0)
    assert [row.kind for hunk in trace.hunks for row in hunk.rows] == ["equal", "equal"]


@pytest.mark.parametrize(
    ("before", "after", "changed_number"),
    [
        ("old\na\nb\nc\nd\n", "new\na\nb\nc\nd\n", 1),
        ("a\nb\nc\nd\nold\n", "a\nb\nc\nd\nnew\n", 5),
    ],
)
def test_build_trace_folds_file_edge_equal_rows(tmp_path: Path, before: str, after: str, changed_number: int):
    dump = _write_dump(
        tmp_path,
        {
            "00_frontend.py": before,
            "01_after_TestPass.py": after,
        },
    )

    trace = build_trace(discover_snapshots(dump), context=1)[0]

    visible_rows = [row for hunk in trace.hunks if not hunk.collapsed for row in hunk.rows]
    assert any(
        row.kind == "replace" and row.before_number == changed_number and row.after_number == changed_number
        for row in visible_rows
    )
    assert any(hunk.collapsed for hunk in trace.hunks)


def test_build_trace_collapses_middle_equal_rows_with_zero_context(tmp_path: Path):
    dump = _write_dump(
        tmp_path,
        {
            "00_frontend.py": "old\na\nb\nold\n",
            "01_after_TestPass.py": "new\na\nb\nnew\n",
        },
    )

    trace = build_trace(discover_snapshots(dump), context=0)[0]

    collapsed = [hunk for hunk in trace.hunks if hunk.collapsed]
    assert [[row.before_number for row in hunk.rows] for hunk in collapsed] == [[2, 3]]


@pytest.mark.parametrize(
    ("middle", "collapsed_count"),
    [
        ("a\nb\nc\nd", 0),
        ("a\nb\nc\nd\ne", 1),
    ],
)
def test_build_trace_only_folds_long_middle_equal_runs(tmp_path: Path, middle: str, collapsed_count: int):
    dump = _write_dump(
        tmp_path,
        {
            "00_frontend.py": f"old\n{middle}\nold\n",
            "01_after_TestPass.py": f"new\n{middle}\nnew\n",
        },
    )

    trace = build_trace(discover_snapshots(dump), context=2)[0]

    assert sum(hunk.collapsed for hunk in trace.hunks) == collapsed_count


def test_build_trace_rejects_negative_context(tmp_path: Path):
    dump = _write_dump(
        tmp_path,
        {
            "00_frontend.py": "before\n",
            "01_after_TestPass.py": "after\n",
        },
    )

    with pytest.raises(IRTraceError, match="context must be non-negative, got -1"):
        build_trace(discover_snapshots(dump), context=-1)


def test_highlight_python_escapes_script_text_and_marks_tokens():
    highlighted = highlight_python('value = "<script>"  # <script>\n')

    assert len(highlighted) == 1
    assert "<script>" not in highlighted[0]
    assert "&lt;script&gt;" in highlighted[0]
    assert 'class="tok-string"' in highlighted[0]
    assert 'class="tok-comment"' in highlighted[0]


def test_highlight_python_escapes_unicode_line_separators():
    highlighted = highlight_python("value = '\u2028\u2029'\n")

    assert highlighted[0].count("&#x2028;") == 1
    assert highlighted[0].count("&#x2029;") == 1
    assert "\u2028" not in highlighted[0]
    assert "\u2029" not in highlighted[0]


def test_build_trace_preserves_unicode_line_separators_from_discovery(tmp_path: Path):
    source = "value = '\u2028\u2029'\n"
    dump = _write_dump(
        tmp_path,
        {
            "00_frontend.py": source,
            "01_after_TestPass.py": source,
        },
    )

    trace = build_trace(discover_snapshots(dump), context=3)[0]

    assert not trace.changed
    assert len(trace.hunks) == 1
    assert len(trace.hunks[0].rows) == 1
    row = trace.hunks[0].rows[0]
    assert "&#x2028;" in row.before_html and "&#x2029;" in row.before_html
    assert "&#x2028;" in row.after_html and "&#x2029;" in row.after_html


def test_highlight_python_escapes_every_line_after_tokenization_error():
    text = "if True:\n  value = (<script>\n"

    highlighted = highlight_python(text)

    assert highlighted == ("if True:", "  value = (&lt;script&gt;")
    assert all("<script>" not in line for line in highlighted)


def test_highlight_python_escapes_unicode_line_separators_after_tokenization_error():
    highlighted = highlight_python("value = (<script>\u2028\u2029")

    assert highlighted == ("value = (&lt;script&gt;&#x2028;&#x2029;",)


def test_render_html_is_deterministic_self_contained_and_safe(tmp_path: Path):
    dump = _write_dump(
        tmp_path,
        {
            "00_frontend.py": "value = '</script><b>&'\n",
            "01_after_TestPass.py": "value = '<script>'\n",
            "01_after_TestPass.log": "warning </script>\u2028\u2029\n",
        },
    )
    traces = build_trace(discover_snapshots(dump), context=3)

    first = render_html(traces, source_name="passes_dump")

    assert first == render_html(traces, source_name="passes_dump")
    assert first.startswith("<!doctype html>")
    assert "http://" not in first and "https://" not in first
    assert "</script><b>" not in first
    assert "\\u003c/script\\u003e" in first
    assert "\\u003e" in first
    assert "\\u0026" in first
    assert "\\u2028" in first and "\\u2029" in first


def test_render_html_payload_has_only_portable_trace_data(tmp_path: Path):
    dump = _write_dump(
        tmp_path,
        {
            "00_frontend.py": "header\nold\ntail\n",
            "01_after_ChangedPass.py": "header\nnew\ntail\n",
            "01_after_ChangedPass.log": "lowering warning\n",
            "02_after_NoopPass.py": "header\nnew\ntail\n",
        },
    )
    traces = build_trace(discover_snapshots(dump), context=0)

    report = render_html(traces, source_name=str(dump))
    encoded = report.split('<script id="trace-data" type="application/json">', 1)[1].split("</script>", 1)[0]
    payload = json.loads(encoded)

    assert str(tmp_path) not in report
    assert payload["sourceName"] == "passes_dump"
    assert (payload["changedCount"], payload["noopCount"]) == (1, 1)
    assert [(item["index"], item["name"], item["changed"]) for item in payload["passes"]] == [
        (1, "ChangedPass", True),
        (2, "NoopPass", False),
    ]
    assert payload["passes"][0]["warning"] == "lowering warning\n"
    assert payload["passes"][1]["warning"] is None
    assert payload["passes"][0]["beforeName"] == "00_frontend.py"
    assert payload["passes"][0]["afterName"] == "01_after_ChangedPass.py"
    assert payload["passes"][0]["beforeText"] == "header\nold\ntail\n"
    assert payload["passes"][0]["afterText"] == "header\nnew\ntail\n"
    assert any(hunk["collapsed"] for hunk in payload["passes"][0]["hunks"])
    assert {row["kind"] for hunk in payload["passes"][0]["hunks"] for row in hunk["rows"]} == {
        "equal",
        "replace",
    }
    assert "timestamp" not in encoded.lower()
    assert '"path"' not in encoded.lower()


def test_render_html_contains_layout_and_interaction_contract(tmp_path: Path):
    dump = _write_dump(
        tmp_path,
        {
            "00_frontend.py": "before\n",
            "01_after_ChangedPass.py": "after\n",
            "01_after_ChangedPass.log": "warning\n",
            "02_after_NoopPass.py": "after\n",
        },
    )
    report = render_html(build_trace(discover_snapshots(dump), context=0), source_name=dump.name)

    for element_id in (
        "pass-list",
        "changed-filter",
        "noop-filter",
        "summary",
        "pass-title",
        "before-pane",
        "after-pane",
        "warnings-panel",
        "copy-before",
        "copy-after",
        "expand-all",
        "collapse-all",
        "theme-toggle",
    ):
        assert f'id="{element_id}"' in report

    assert "grid-template-columns: 18rem minmax(0, 1fr)" in report
    assert "grid-template-columns: minmax(0, 1fr) minmax(0, 1fr)" in report
    assert "@media (max-width: 800px)" in report
    assert ':root[data-theme="light"]' in report
    assert ':root[data-theme="dark"]' in report
    assert "var(--" in report
    assert 'matchMedia("(prefers-color-scheme: dark)")' in report
    assert 'document.getElementById("source-name").textContent = data.sourceName' in report

    for function_name in (
        "visiblePasses",
        "selectPass",
        "renderSidebar",
        "renderDiff",
        "copySnapshot",
        "setAllHunks",
        "toggleTheme",
    ):
        assert f"function {function_name}(" in report

    assert 'document.getElementById("changed-filter")' in report
    assert 'document.getElementById("noop-filter")' in report
    assert 'document.addEventListener("keydown"' in report
    assert 'event.key === "j"' in report and 'event.key === "ArrowDown"' in report
    assert 'event.key === "k"' in report and 'event.key === "ArrowUp"' in report
    assert 'target.tagName === "INPUT"' in report and 'target.tagName === "BUTTON"' in report
    assert "data.passes.find((trace) => trace.changed) || data.passes[0]" in report
    assert "navigator.clipboard.writeText(text)" in report
    assert 'document.createElement("textarea")' in report
    assert 'trace[side + "Text"]' in report
    assert "trace.warning" in report


def test_viewer_clears_details_when_filters_hide_every_pass(tmp_path: Path):
    dump = _write_dump(
        tmp_path,
        {
            "00_frontend.py": "before\n",
            "01_after_ChangedPass.py": "after\n",
        },
    )
    report = render_html(build_trace(discover_snapshots(dump), context=0), source_name=dump.name)

    result = _run_viewer_behavior(
        report,
        """
        if (selectedIndex !== 1) throw new Error("changed pass was not initially selected");
        elements["changed-filter"].checked = false;
        elements["changed-filter"].listeners.change();
        if (selectedIndex !== null) throw new Error("hidden pass remained selected");
        if (elements["pass-title"].textContent !== "No passes match the filters.") {
          throw new Error("empty filter detail message was not rendered");
        }
        if (!elements["copy-before"].disabled || !elements["expand-all"].disabled) {
          throw new Error("snapshot controls remained enabled");
        }
        """,
    )

    assert result.returncode == 0, result.stderr


def test_empty_viewer_disables_snapshot_controls():
    result = _run_viewer_behavior(
        render_html((), source_name="passes_dump"),
        """
        for (const id of ["copy-before", "copy-after", "expand-all", "collapse-all"]) {
          if (!elements[id].disabled) throw new Error(`${id} remained enabled`);
        }
        if (elements["pass-title"].textContent !== "No passes in this report.") {
          throw new Error("empty report detail message was not rendered");
        }
        """,
    )

    assert result.returncode == 0, result.stderr


def test_empty_viewer_copy_is_a_safe_noop():
    result = _run_viewer_behavior(
        render_html((), source_name="passes_dump"),
        """
        copySnapshot("before");
        """,
    )

    assert result.returncode == 0, result.stderr


def test_discover_orders_snapshots_and_attaches_warning(tmp_path: Path):
    dump = _write_dump(
        tmp_path,
        {
            "02_after_UnrollLoops.py": "after two\n",
            "00_frontend.py": "frontend\n",
            "01_after_InlineFunctions.log": "unused variable\n",
            "01_after_InlineFunctions.py": "after one\n",
            "fa_fused_EXTRACT.py": "ignored\n",
        },
    )

    snapshots = discover_snapshots(dump)

    assert [snapshot.index for snapshot in snapshots] == [0, 1, 2]
    assert [snapshot.pass_name for snapshot in snapshots] == [None, "InlineFunctions", "UnrollLoops"]
    assert snapshots[1].warning_text == "unused variable\n"
    assert snapshots[2].warning_text is None


def test_discover_rejects_zero_index_pass_snapshot(tmp_path: Path):
    dump = _write_dump(
        tmp_path,
        {
            "00_frontend.py": "frontend\n",
            "00_after_InlineFunctions.py": "after zero\n",
        },
    )

    with pytest.raises(IRTraceError, match="00_after_InlineFunctions.py"):
        discover_snapshots(dump)


@pytest.mark.parametrize("path_name", ["01_after_InlineFunctions.py", "01_after_InlineFunctions.log"])
def test_discover_rejects_non_file_snapshot_paths(tmp_path: Path, path_name: str):
    dump = _write_dump(
        tmp_path,
        {
            "00_frontend.py": "frontend\n",
            "01_after_InlineFunctions.py": "after one\n",
        },
    )
    if path_name.endswith(".py"):
        (dump / path_name).unlink()
    (dump / path_name).mkdir()

    with pytest.raises(IRTraceError, match=path_name):
        discover_snapshots(dump)


@pytest.mark.parametrize(
    ("case", "expected_message"),
    [
        ("missing_directory", "does not exist"),
        ("not_directory", "not a directory"),
        ("missing_frontend", "00_frontend.py"),
        ("no_pass_snapshots", "no pass snapshots"),
        ("starts_at_two", "01"),
        ("index_gap", "02"),
        ("duplicate_index", "01"),
        ("malformed_name", "02_ConvertToSSA.py"),
        ("invalid_snapshot_utf8", "01_after_InlineFunctions.py"),
        ("invalid_warning_utf8", "01_after_InlineFunctions.log"),
    ],
)
def test_discover_rejects_invalid_dump_inputs(tmp_path: Path, case: str, expected_message: str):
    if case == "missing_directory":
        dump = tmp_path / "missing"
    elif case == "not_directory":
        dump = tmp_path / "not-a-directory"
        dump.write_text("not a directory", encoding="utf-8")
    elif case == "missing_frontend":
        dump = _write_dump(tmp_path, {"01_after_InlineFunctions.py": "after one\n"})
    elif case == "no_pass_snapshots":
        dump = _write_dump(tmp_path, {"00_frontend.py": "frontend\n"})
    elif case == "starts_at_two":
        dump = _write_dump(
            tmp_path,
            {
                "00_frontend.py": "frontend\n",
                "02_after_UnrollLoops.py": "after two\n",
            },
        )
    elif case == "index_gap":
        dump = _write_dump(
            tmp_path,
            {
                "00_frontend.py": "frontend\n",
                "01_after_InlineFunctions.py": "after one\n",
                "03_after_ConvertToSSA.py": "after three\n",
            },
        )
    elif case == "duplicate_index":
        dump = _write_dump(
            tmp_path,
            {
                "00_frontend.py": "frontend\n",
                "01_after_InlineFunctions.py": "after one\n",
                "01_after_UnrollLoops.py": "also after one\n",
            },
        )
    elif case == "malformed_name":
        dump = _write_dump(
            tmp_path,
            {
                "00_frontend.py": "frontend\n",
                "02_ConvertToSSA.py": "malformed\n",
            },
        )
    elif case == "invalid_snapshot_utf8":
        dump = _write_dump(
            tmp_path,
            {
                "00_frontend.py": "frontend\n",
                "01_after_InlineFunctions.py": "after one\n",
            },
        )
        (dump / "01_after_InlineFunctions.py").write_bytes(b"\xff")
    else:
        dump = _write_dump(
            tmp_path,
            {
                "00_frontend.py": "frontend\n",
                "01_after_InlineFunctions.py": "after one\n",
            },
        )
        (dump / "01_after_InlineFunctions.log").write_bytes(b"\xff")

    with pytest.raises(IRTraceError, match=expected_message):
        discover_snapshots(dump)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
