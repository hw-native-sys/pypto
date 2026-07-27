# IR Lower Trace 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 新增 `pypto-ir-trace` CLI，将现有 `passes_dump/` 快照转换为可离线打开的、自包含的交互式 HTML lowering trace。

**架构：** Python 在生成阶段完成快照发现、文本 diff、统计、折叠和语法高亮，浏览器只负责渲染与交互。实现拆分为 model、discovery、diff、html 和 cli 五个内部模块，模块之间通过不可变 dataclass 通信。

**技术栈：** Python 3.10+ 标准库（`argparse`、`dataclasses`、`difflib`、`json`、`pathlib`、`tempfile`、`tokenize`）、内嵌 HTML/CSS/JavaScript、pytest、ruff、pyright。

## 全局约束

- 第一阶段只提供 HTML CLI；不实现 terminal 模式或公开 Python API。
- 不修改 `PassManager.run_passes`、C++、binding、type stub、pass pipeline 或 codegen。
- 不新增第三方运行时依赖；报告不得加载 CDN、远程脚本、字体、图片或样式表。
- diff 仅表示文本差异，不表示 IR 语义等价性。
- 输出不得包含绝对路径、生成时间或其他机器相关信息；相同输入必须逐字节产生相同输出。
- 所有输入文本必须安全转义；`</script>`、`<`、`>`、`&`、U+2028 和 U+2029 不得逃逸嵌入数据块。
- Python 代码使用现代类型语法、f-string 和 Google-style docstring。
- 测试使用 pytest 函数、普通 `assert` 和 `pytest.raises()`；测试文件以 `pytest.main([__file__, "-v"])` 结束。
- 开始 Task 1 前，按 `fix-issue` 工作流自认领 issue #2134，并把项目状态改为 In Progress。

---

### Task 1：快照模型与发现

**文件：**

- 新建：`python/pypto/tools/ir_trace/{__init__,model,discovery}.py`
- 新建：`tests/ut/tools/test_ir_trace.py`

**接口：**

- 输入：用户提供的 `Path` 类型 `passes_dump/` 目录。
- 产出：`discover_snapshots(directory: Path) -> tuple[Snapshot, ...]`。
- 产出：后续任务使用的 `IRTraceError`、`Snapshot`、`DiffRow`、`DiffHunk`、`PassTrace`。

- [ ] **Step 1：添加 package 和失败的正常发现测试**

在 `tests/ut/tools/test_ir_trace.py` 添加：

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
```

- [ ] **Step 2：运行测试并确认按预期失败**

运行：

```bash
PYTHONPATH="$(pwd)/python" python3.11 -m pytest \
  tests/ut/tools/test_ir_trace.py::test_discover_orders_snapshots_and_attaches_warning -v
```

预期：FAIL，提示 `pypto.tools.ir_trace` 不存在。

- [ ] **Step 3：实现不可变模型**

在 `model.py` 中定义：

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

`__init__.py` 只导出 `IRTraceError`，不要把内部 dataclass 变成公开稳定 API。

- [ ] **Step 4：实现快照发现**

在 `discovery.py` 中使用：

```python
_PASS_RE = re.compile(r"^(?P<index>\d+)_after_(?P<name>.+)\.py$")
_NUMERIC_PY_RE = re.compile(r"^\d+_.*\.py$")
def discover_snapshots(directory: Path) -> tuple[Snapshot, ...]:
    if not directory.exists():
        raise IRTraceError(f"input directory does not exist: {directory}")
    if not directory.is_dir():
        raise IRTraceError(f"input path is not a directory: {directory}")

    frontend = directory / "00_frontend.py"
    if not frontend.is_file():
        raise IRTraceError(f"missing 00_frontend.py in {directory}")
    indexed: dict[int, tuple[str, Path]] = {}
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        match = _PASS_RE.fullmatch(path.name)
        if match:
            index = int(match.group("index"))
            if index in indexed:
                previous = indexed[index][1].name
                raise IRTraceError(f"duplicate snapshot index {index:02d}: {previous} and {path.name}")
            indexed[index] = (match.group("name"), path)
        elif _NUMERIC_PY_RE.fullmatch(path.name) and path.name != "00_frontend.py":
            raise IRTraceError(f"malformed snapshot name {path.name}; expected NN_after_PassName.py")
    if not indexed:
        raise IRTraceError(f"no pass snapshots found in {directory}")
```

增加私有 `_read_utf8(path: Path) -> str`，将 `UnicodeDecodeError` 转为
`IRTraceError(f"{path.name} is not valid UTF-8")`。按索引检查必须从 `1`
连续；读取同 stem `.log`；用 `tuple(text.splitlines())` 填充 `lines`。

- [ ] **Step 5：增加发现错误测试**

添加参数化测试，逐项覆盖：缺目录、非目录、缺 frontend、无 pass、从 2
开始、索引缺口、重复索引、`02_ConvertToSSA.py` 畸形名称、快照无效 UTF-8、
warning 无效 UTF-8。每个 case 同时断言异常消息中的文件名或缺失索引。

- [ ] **Step 6：运行 Task 1 测试并确认通过**

运行：

```bash
PYTHONPATH="$(pwd)/python" python3.11 -m pytest tests/ut/tools/test_ir_trace.py -k discover -v
ruff check python/pypto/tools/ir_trace tests/ut/tools/test_ir_trace.py
```

预期：全部 PASS，ruff 无输出。

- [ ] **Step 7：提交 Task 1**

```bash
git add python/pypto/tools/ir_trace tests/ut/tools/test_ir_trace.py
git commit -m "feat(tools): Discover IR pass snapshots"
```

---

### Task 2：文本 diff、统计、折叠与语法高亮

**文件：**

- 新建：`python/pypto/tools/ir_trace/diff.py`
- 修改：`tests/ut/tools/test_ir_trace.py`

**接口：**

- 输入：Task 1 的 `Snapshot` tuple 与非负 `context`。
- 产出：`build_trace(snapshots: tuple[Snapshot, ...], context: int) -> tuple[PassTrace, ...]`。
- 内部：`highlight_python(text: str) -> tuple[str, ...]` 返回逐行安全 HTML。

- [ ] **Step 1：添加失败的统计与对齐测试**

```python
from pypto.tools.ir_trace.diff import build_trace
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
    changed_rows = [row for hunk in trace.hunks for row in hunk.rows if row.kind == "replace"]
    assert [(row.before_number, row.after_number) for row in changed_rows] == [(2, 2), (None, 3)]
```

- [ ] **Step 2：运行测试并确认失败**

运行指定测试，预期 FAIL，提示 `pypto.tools.ir_trace.diff` 不存在。

- [ ] **Step 3：实现安全 Python 高亮**

实现 `highlight_python()`：按 `splitlines()` 创建每行片段，用 `tokenize.generate_tokens()` 获取 token，只给 NAME keyword、STRING、NUMBER、COMMENT、OP 添加固定 CSS class。
所有 token 与间隔文本先调用 `html.escape(..., quote=False)`；`TokenError`、`IndentationError` 或位置越界时，对每行直接 `html.escape()`。

- [ ] **Step 4：实现 opcode 对齐与统计**

实现私有 `_diff_rows(before, after) -> tuple[tuple[DiffRow, ...], int, int]`：

```python
matcher = difflib.SequenceMatcher(a=before.lines, b=after.lines, autojunk=False)
for tag, before_start, before_end, after_start, after_end in matcher.get_opcodes():
    before_count = before_end - before_start
    after_count = after_end - after_start
    if tag == "insert":
        inserted += after_count
    elif tag == "delete":
        deleted += before_count
    elif tag == "replace":
        inserted += after_count
        deleted += before_count
```

`equal` 逐行一一对应；`insert` 和 `delete` 的另一侧为空；`replace` 使用
`max(before_count, after_count)` 行并在较短侧补 `None`。行号从 `1` 开始。

- [ ] **Step 5：实现上下文折叠与完整 trace**

实现 `_fold_rows(rows, context) -> tuple[DiffHunk, ...]`。只折叠 `equal`
连续区间：若该区间位于变化之间并且长度大于 `2 * context`，保留首尾各
`context` 行，并把中段放入 `collapsed=True` 的 hunk；文件首部或尾部只
保留靠近变化的一侧 `context` 行。`context=0` 时整个相同中段折叠。

`build_trace()` 对 `snapshots[i - 1]` 与 `snapshots[i]` 逐对调用，并拒绝
负 context：`IRTraceError(f"context must be non-negative, got {context}")`。

- [ ] **Step 6：补齐边界测试**

新增测试覆盖 insert、delete、no-op、CRLF/LF、末尾换行、文件首尾变化、`context=0`、短/长 equal 区、高亮 `<script>` 和无效 Python 回退。
断言所有输出片段不含未转义的 `<script>`。

- [ ] **Step 7：运行 Task 2 测试并提交**

```bash
PYTHONPATH="$(pwd)/python" python3.11 -m pytest tests/ut/tools/test_ir_trace.py -k "trace or highlight or fold" -v
ruff check python/pypto/tools/ir_trace tests/ut/tools/test_ir_trace.py
git add python/pypto/tools/ir_trace/diff.py tests/ut/tools/test_ir_trace.py
git commit -m "feat(tools): Analyze IR pass differences"
```

---

### Task 3：确定性自包含 HTML 报告

**文件：**

- 新建：`python/pypto/tools/ir_trace/html.py`
- 修改：`tests/ut/tools/test_ir_trace.py`

**接口：**

- 输入：`render_html(traces: tuple[PassTrace, ...], source_name: str) -> str`。
- 产出：完整 HTML5 字符串；不读写文件。

- [ ] **Step 1：添加失败的确定性与安全测试**

```python
from pypto.tools.ir_trace.html import render_html
def test_render_html_is_deterministic_self_contained_and_safe(tmp_path: Path):
    dump = _write_dump(
        tmp_path,
        {
            "00_frontend.py": "value = '</script><b>'\n",
            "01_after_TestPass.py": "value = '<script>'\n",
            "01_after_TestPass.log": "warning </script>\n",
        },
    )
    traces = build_trace(discover_snapshots(dump), context=3)
    first = render_html(traces, source_name="passes_dump")
    assert first == render_html(traces, source_name="passes_dump")
    assert first.startswith("<!doctype html>")
    assert "http://" not in first and "https://" not in first
    assert "</script><b>" not in first
    assert "\\u003c/script\\u003e" in first
```

- [ ] **Step 2：运行测试并确认失败**

运行指定测试，预期 FAIL，提示 `pypto.tools.ir_trace.html` 不存在。

- [ ] **Step 3：实现稳定 payload 与安全嵌入**

实现 `_trace_payload()`，只放入 basename、pass index/name、changed、统计、
warning、完整 before/after 原文和 hunk/row 数据。实现：

```python
def _json_for_script(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    escapes = {ord("<"): "\\u003c", ord(">"): "\\u003e", ord("&"): "\\u0026"}
    escapes.update({0x2028: "\\u2028", 0x2029: "\\u2029"})
    return encoded.translate(escapes)
```

HTML 中使用 `<script id="trace-data" type="application/json">` 保存 payload。

- [ ] **Step 4：实现 IDE 双栏模板与样式**

模板必须包含以下稳定 DOM id：`pass-list`、`changed-filter`、`noop-filter`、
`summary`、`pass-title`、`before-pane`、`after-pane`、`warnings-panel`、
`copy-before`、`copy-after`、`expand-all`、`collapse-all`、`theme-toggle`。

CSS 使用 grid：桌面 `grid-template-columns: 18rem minmax(0, 1fr)`；diff 主区
为两个等宽列；`@media (max-width: 800px)` 下改为单列。颜色全部使用 CSS
variable，并为 `data-theme="light"` 与 `data-theme="dark"` 定义值。

- [ ] **Step 5：实现浏览器交互**

JavaScript 只从 `trace-data` 解析数据，并实现这些明确函数：

- `visiblePasses()`：应用 changed/no-op checkbox。
- `selectPass(index)`：更新选择并调用主区渲染。
- `renderSidebar()`：显示 pass 名、`+N -M`、no-op 和 warning 标识。
- `renderDiff(trace)`：使用 DOM API 创建行；折叠 hunk 使用 button 展开。
- `copySnapshot(side)`：复制完整 `beforeText` 或 `afterText`，失败时使用临时 textarea。
- `setAllHunks(expanded)`：实现全局展开或折叠。
- `toggleTheme()`：更新 `document.documentElement.dataset.theme`。

`keydown` 在 input/button 获得焦点时不处理；`j`/`ArrowDown` 和 `k`/`ArrowUp` 只遍历 `visiblePasses()`。
首次选择第一个 changed pass，全部 no-op 时选择第一项。

- [ ] **Step 6：补齐结构和交互契约测试**

断言上述所有 DOM id、键盘 key、`matchMedia("(prefers-color-scheme: dark)")`、
clipboard fallback、filter、warning、展开/折叠函数都存在；断言 payload 不含
输入目录绝对路径或时间字段；断言 changed/no-op 数量正确。

- [ ] **Step 7：运行 Task 3 测试并提交**

```bash
PYTHONPATH="$(pwd)/python" python3.11 -m pytest tests/ut/tools/test_ir_trace.py -k html -v
ruff check python/pypto/tools/ir_trace tests/ut/tools/test_ir_trace.py
git add python/pypto/tools/ir_trace/html.py tests/ut/tools/test_ir_trace.py
git commit -m "feat(tools): Render interactive IR trace HTML"
```

---

### Task 4：CLI、原子写入、文档与完整验证

**文件：**

- 新建：`python/pypto/tools/ir_trace/cli.py`
- 修改：`pyproject.toml:38`
- 修改：`tests/ut/tools/test_ir_trace.py`
- 新建：`docs/{en,zh-cn}/dev/07-ir-lower-trace.md`

**接口：**

- 消费：`discover_snapshots()`、`build_trace()`、`render_html()`。
- 产出：`main(argv: Sequence[str] | None = None) -> int`。
- 产出：已安装命令 `pypto-ir-trace`。

- [ ] **Step 1：添加失败的 CLI 成功与错误测试**

```python
from pypto.tools.ir_trace.cli import main
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
```

- [ ] **Step 2：运行 CLI 测试并确认失败**

运行两个指定测试，预期 FAIL，提示 `pypto.tools.ir_trace.cli` 不存在。

- [ ] **Step 3：实现参数解析与原子写入**

```python
def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"must be non-negative, got {value}")
    return parsed
def _write_atomic(output: Path, content: str) -> None:
    output_parent = output.parent
    if not output_parent.is_dir():
        raise IRTraceError(f"output directory does not exist: {output_parent}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output_parent,
            prefix=f".{output.name}.", suffix=".tmp", delete=False,
        ) as handle:
            handle.write(content)
            temporary = Path(handle.name)
        temporary.replace(output)
    except OSError as error:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise IRTraceError(f"failed to write {output}: {error}") from error
```

`main()` 使用 `ArgumentParser(prog="pypto-ir-trace")`，参数为 positional
`passes_dump: Path`、`-o/--output: Path` 默认 `Path("ir_trace.html")`、
`--context` 默认 `3`。捕获 `IRTraceError` 后打印
`pypto-ir-trace: error: {error}` 到 stderr 并返回 `1`。

- [ ] **Step 4：注册 console script 并补齐 CLI 测试**

在 `pyproject.toml` 的 `[project]` 结束后添加：

```toml
[project.scripts]
pypto-ir-trace = "pypto.tools.ir_trace.cli:main"
```

增加显式输出、负 context 返回 `2`、缺输出目录、模拟 `Path.replace()` 失败
并验证临时文件清理的测试。最后确保测试文件包含：

```python
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

- [ ] **Step 5：运行全部聚焦测试**

```bash
PYTHONPATH="$(pwd)/python" python3.11 -m pytest tests/ut/tools/test_ir_trace.py -v
```

预期：全部 PASS。

- [ ] **Step 6：编写同步使用文档**

英文 `docs/en/dev/07-ir-lower-trace.md` 为权威版本，中文文件保持相同标题结构。两份文档都必须说明如何用 `dump_passes=True` 或 `PassDumpLevel.EXPLICIT` 生成输入。
两份文档还必须包含 CLI 参数、侧栏、filter、键盘、复制、warning、主题、折叠、错误示例、文本差异非语义等价声明，以及同语言 pass manager 链接。

- [ ] **Step 7：运行静态检查和 tools 回归**

```bash
ruff check python/pypto/tools/ir_trace tests/ut/tools/test_ir_trace.py
ruff format --check python/pypto/tools/ir_trace tests/ut/tools/test_ir_trace.py
PYTHONPATH="$(pwd)/python" python3.11 -m pytest tests/ut/tools -v
pyright
```

预期：全部通过。

- [ ] **Step 8：运行完整单元测试与 pre-commit**

```bash
PYTHONPATH="$(pwd)/python" python3.11 -m pytest tests/ut/ -n auto --maxprocesses 8 -v
pre-commit run --all-files
```

若出现失败，先使用 `superpowers:systematic-debugging` 定位根因；不得通过
修改无关测试期望绕过失败。

- [ ] **Step 9：提交 Task 4**

```bash
git add pyproject.toml python/pypto/tools/ir_trace/cli.py \
  tests/ut/tools/test_ir_trace.py \
  docs/en/dev/07-ir-lower-trace.md docs/zh-cn/dev/07-ir-lower-trace.md
git commit -m "feat(tools): Add IR lower trace CLI"
```

- [ ] **Step 10：按 PyPTO 工作流做最终审查**

使用 `code-review`、`testing` 和 `verification-before-completion`；确认
`git status --short` 为空、所有 commit 只包含 issue #2134 范围内文件，
然后准备 PR 描述并包含 `Fixes #2134`。
