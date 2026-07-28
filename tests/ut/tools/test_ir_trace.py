# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for IR pass snapshot discovery."""

import errno
import json
import shutil
import subprocess
import sysconfig
import tempfile
import textwrap
import tokenize
from collections.abc import Iterator
from pathlib import Path

import pytest
from pypto.tools.ir_trace.cli import main
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


def test_cli_writes_default_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dump = _write_dump(
        tmp_path,
        {"00_frontend.py": "a\n", "01_after_TestPass.py": "b\n"},
    )
    monkeypatch.chdir(tmp_path)

    assert main([str(dump)]) == 0
    assert (tmp_path / "ir_trace.html").read_text(encoding="utf-8").startswith("<!doctype html>")


def test_cli_reports_domain_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    assert main([str(tmp_path / "missing")]) == 1
    assert "pypto-ir-trace: error: input directory does not exist" in capsys.readouterr().err


def test_cli_reports_directory_enumeration_error_without_path_or_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    dump = _write_dump(
        tmp_path,
        {"00_frontend.py": "a\n", "01_after_TestPass.py": "b\n"},
    )
    original_iterdir = Path.iterdir

    def fail_dump_enumeration(path: Path) -> Iterator[Path]:
        if path == dump:
            raise PermissionError(errno.EACCES, "Permission denied", str(path))
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", fail_dump_enumeration)

    assert main([str(dump)]) == 1
    error = capsys.readouterr().err
    assert "pypto-ir-trace: error: failed to enumerate passes_dump: Permission denied" in error
    assert str(tmp_path) not in error
    assert "Traceback" not in error


@pytest.mark.parametrize("path_name", ["01_after_TestPass.py", "01_after_TestPass.log"])
def test_cli_reports_snapshot_read_error_without_path_or_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    path_name: str,
):
    dump = _write_dump(
        tmp_path,
        {
            "00_frontend.py": "a\n",
            "01_after_TestPass.py": "b\n",
            "01_after_TestPass.log": "warning\n",
        },
    )
    original_read_text = Path.read_text

    def fail_snapshot_read(
        path: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        if path.name == path_name:
            raise OSError(errno.EIO, "Input/output error", str(path))
        return original_read_text(path, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", fail_snapshot_read)

    assert main([str(dump)]) == 1
    error = capsys.readouterr().err
    assert f"pypto-ir-trace: error: failed to read {path_name}: Input/output error" in error
    assert str(tmp_path) not in error
    assert "Traceback" not in error


def test_cli_writes_explicit_report(tmp_path: Path):
    dump = _write_dump(
        tmp_path,
        {"00_frontend.py": "a\n", "01_after_TestPass.py": "b\n"},
    )
    output = tmp_path / "custom.html"

    assert main([str(dump), "--output", str(output), "--context", "0"]) == 0
    assert output.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_cli_rejects_negative_context(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    assert main([str(tmp_path / "passes_dump"), "--context", "-1"]) == 2
    assert (
        "pypto-ir-trace: error: argument --context: must be non-negative, got -1" in capsys.readouterr().err
    )


def test_cli_reports_missing_output_directory(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    dump = _write_dump(
        tmp_path,
        {"00_frontend.py": "a\n", "01_after_TestPass.py": "b\n"},
    )
    output = tmp_path / "missing" / "trace.html"

    assert main([str(dump), "--output", str(output)]) == 1
    assert (
        f"pypto-ir-trace: error: output directory does not exist: {output.parent}" in capsys.readouterr().err
    )
    assert not output.exists()


def test_cli_cleans_up_temporary_file_when_replacement_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    dump = _write_dump(
        tmp_path,
        {"00_frontend.py": "a\n", "01_after_TestPass.py": "b\n"},
    )
    output = tmp_path / "trace.html"
    output.write_text("existing report", encoding="utf-8")

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError("replacement failed")

    monkeypatch.setattr(Path, "replace", fail_replace)

    assert main([str(dump), "--output", str(output)]) == 1
    assert "pypto-ir-trace: error: failed to write" in capsys.readouterr().err
    assert output.read_text(encoding="utf-8") == "existing report"
    assert list(tmp_path.glob(".trace.html.*.tmp")) == []


def test_cli_cleans_up_owned_temporary_file_when_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    dump = _write_dump(
        tmp_path,
        {"00_frontend.py": "a\n", "01_after_TestPass.py": "b\n"},
    )
    output = tmp_path / "trace.html"
    output.write_text("existing report", encoding="utf-8")

    with tempfile.NamedTemporaryFile() as probe:
        handle_type = type(probe)

    def fail_write(_handle: object, _content: str) -> int:
        raise OSError("write failed")

    monkeypatch.setattr(handle_type, "write", fail_write, raising=False)

    assert main([str(dump), "--output", str(output)]) == 1
    assert "pypto-ir-trace: error: failed to write" in capsys.readouterr().err
    assert output.read_text(encoding="utf-8") == "existing report"
    assert list(tmp_path.glob(".trace.html.*.tmp")) == []


def test_cli_cleanup_failure_does_not_mask_primary_write_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    dump = _write_dump(
        tmp_path,
        {"00_frontend.py": "a\n", "01_after_TestPass.py": "b\n"},
    )
    output = tmp_path / "trace.html"
    output.write_text("existing report", encoding="utf-8")

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError("replacement failed")

    def fail_unlink(_path: Path, *, missing_ok: bool = False) -> None:
        raise OSError(f"cleanup failed (missing_ok={missing_ok})")

    with monkeypatch.context() as cleanup_failure:
        cleanup_failure.setattr(Path, "replace", fail_replace)
        cleanup_failure.setattr(Path, "unlink", fail_unlink)
        assert main([str(dump), "--output", str(output)]) == 1

    error = capsys.readouterr().err
    assert "pypto-ir-trace: error: failed to write" in error
    assert "replacement failed" in error
    assert "cleanup failed" not in error
    assert output.read_text(encoding="utf-8") == "existing report"
    for temporary in tmp_path.glob(".trace.html.*.tmp"):
        temporary.unlink()


def test_installed_console_script_preserves_main_exit_codes(tmp_path: Path):
    script = Path(sysconfig.get_path("scripts")) / "pypto-ir-trace"
    assert script.is_file(), "install PyPTO before running the console-script smoke test"
    dump = _write_dump(
        tmp_path,
        {"00_frontend.py": "a\n", "01_after_TestPass.py": "b\n"},
    )
    output = tmp_path / "trace.html"

    success = subprocess.run(
        [str(script), str(dump), "--output", str(output)],
        check=False,
        capture_output=True,
        text=True,
    )
    domain_error = subprocess.run(
        [str(script), str(tmp_path / "missing")],
        check=False,
        capture_output=True,
        text=True,
    )
    argument_error = subprocess.run(
        [str(script), str(dump), "--context", "-1"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert success.returncode == 0
    assert output.read_text(encoding="utf-8").startswith("<!doctype html>")
    assert domain_error.returncode == 1
    assert "pypto-ir-trace: error: input directory does not exist" in domain_error.stderr
    assert argument_error.returncode == 2
    assert "argument --context: must be non-negative, got -1" in argument_error.stderr


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


def test_highlight_python_falls_back_to_escaped_text_on_syntax_error(
    monkeypatch: pytest.MonkeyPatch,
):
    def fail_tokenization(_readline: object) -> None:
        raise SyntaxError("invalid token stream")

    monkeypatch.setattr(tokenize, "generate_tokens", fail_tokenization)

    assert highlight_python("value = <script>\n") == ("value = &lt;script&gt;",)


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
    assert 'target.tagName === "INPUT"' in report
    assert 'target.tagName === "BUTTON"' not in report
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


def test_viewer_selects_closest_visible_pass_when_filter_hides_selection(tmp_path: Path):
    dump = _write_dump(
        tmp_path,
        {
            "00_frontend.py": "a\n",
            "01_after_NoopBefore.py": "a\n",
            "02_after_ChangedTwo.py": "b\n",
            "03_after_ChangedThree.py": "c\n",
            "04_after_ChangedFour.py": "d\n",
            "05_after_NoopAfter.py": "d\n",
        },
    )
    report = render_html(build_trace(discover_snapshots(dump), context=0), source_name=dump.name)

    result = _run_viewer_behavior(
        report,
        """
        selectPass(4);
        elements["changed-filter"].checked = false;
        elements["changed-filter"].listeners.change();
        if (selectedIndex !== 5) throw new Error("closest visible pass was not selected");

        elements["changed-filter"].checked = true;
        elements["changed-filter"].listeners.change();
        selectPass(3);
        elements["changed-filter"].checked = false;
        elements["changed-filter"].listeners.change();
        if (selectedIndex !== 1) throw new Error("lower-index pass did not win an equal-distance tie");
        """,
    )

    assert result.returncode == 0, result.stderr


def test_viewer_keyboard_navigation_works_from_focused_pass_button(tmp_path: Path):
    dump = _write_dump(
        tmp_path,
        {
            "00_frontend.py": "a\n",
            "01_after_First.py": "b\n",
            "02_after_Second.py": "c\n",
        },
    )
    report = render_html(build_trace(discover_snapshots(dump), context=0), source_name=dump.name)

    result = _run_viewer_behavior(
        report,
        """
        let prevented = false;
        documentListeners.keydown({
          target: new Element("button"),
          key: "j",
          preventDefault() { prevented = true; }
        });
        if (selectedIndex !== 2) throw new Error("focused pass button blocked keyboard navigation");
        if (!prevented) throw new Error("handled navigation did not prevent the browser default");
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


def test_discover_reports_gap_between_neighboring_snapshots(tmp_path: Path):
    dump = _write_dump(
        tmp_path,
        {
            "00_frontend.py": "frontend\n",
            "01_after_InlineFunctions.py": "after one\n",
            "03_after_ConvertToSSA.py": "after three\n",
        },
    )

    with pytest.raises(IRTraceError) as error:
        discover_snapshots(dump)

    assert str(error.value) == (
        f"missing snapshot index 02 in {dump} between "
        "01_after_InlineFunctions.py and 03_after_ConvertToSSA.py"
    )


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
