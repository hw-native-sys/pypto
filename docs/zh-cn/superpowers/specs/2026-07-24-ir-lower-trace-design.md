# 交互式 IR Lower Trace 设计

**Issue：** #2134
**日期：** 2026-07-24
**状态：** 第一阶段设计已批准

## 概述

新增开发者命令 `pypto-ir-trace`。该命令读取
`<output_dir>/passes_dump/` 中已有的逐 pass 快照，生成一个自包含的交互式
HTML 报告。报告采用 IDE 风格布局：左侧为 pass 导航栏，右侧为并排的
before/after 文本 IR diff。

第一阶段只实现 CLI 和 HTML 报告。终端渲染模式与公开 Python API 留待后续
迭代。编译器和现有 `dump_passes` 插桩路径保持不变。

## 目标

- 按执行顺序发现 pass 快照，并把每个 pass 输出与前一个快照配对。
- 将 pass 分类为 changed 或 no-op，并计算新增、删除行数。
- 生成单个确定性的 HTML 文件，不依赖服务器、CDN 或旁路资源文件。
- 让开发者能快速浏览长 pipeline，并折叠或展开大段未变化内容。
- 在存在同名 `.log` 文件时显示对应 pass 的 warning。
- 对缺失、畸形、重复、不可读或索引不连续的快照提供明确诊断。
- 明确 diff 仅是文本证据，不表示 IR 语义等价或不等价。

## 非目标

- 第一阶段不提供 terminal diff 模式。
- 第一阶段不提供公开的 `pypto.tools.ir_trace.render(...)` API。
- 不增加新的 pass 插桩，也不修改 `PassManager.run_passes`。
- 不构建托管 Web 应用、本地 HTTP 服务或任何运行时网络依赖。
- 不进行语义 IR 比较。
- 第一阶段不集成 `compare-codegen`。

## 用户接口

### 命令

```bash
pypto-ir-trace build_output/example/passes_dump \
  --output ir_trace.html
```

首版命令契约为：

```text
pypto-ir-trace PASSES_DUMP [-o OUTPUT] [--context LINES]
```

参数含义：

- `PASSES_DUMP`：由 `dump_passes` 生成的现有快照目录。
- `-o`、`--output`：输出文件；默认是当前目录下的 `ir_trace.html`。
- `--context`：每个变化块前后保留的未变化行数；默认值为 `3`。

`--context` 必须是非负整数。命令成功时返回 `0`，输入或写入失败时返回
`1`，命令行参数错误沿用 argparse 的返回码 `2`。

在只有一种输出模式时不提供 `--mode` 参数。

### 报告布局

页面包含两个固定区域：

1. 左侧 pass 导航栏：显示汇总数量、changed/no-op 过滤器、pass 名称、
   新增/删除行数和 warning 标识。
2. 右侧主区域：显示当前 pass 标题、before/after 文件名、对齐的文本 diff、
   复制按钮、上下文展开控件和 warning。

页面加载后默认选中第一个 changed pass。若所有 pass 均为 no-op，则选中
第一个 pass。过滤条件改变后，导航只遍历可见 pass；如果当前选择被隐藏，
则自动选择距离最近的可见项。

键盘操作：

- `j` 或 `ArrowDown`：选择下一个可见 pass。
- `k` 或 `ArrowUp`：选择上一个可见 pass。

页面首次打开时遵循系统浅色/深色主题，并提供手动切换按钮。窄屏下，
before 和 after 区域改为上下排列。

Copy Before 和 Copy After 始终复制完整原始快照，而不是只复制当前展开的
行。每个折叠块可以独立展开，同时提供 Expand All 和 Collapse All。

## 架构

采用职责单一的内部 package，避免把全部逻辑堆进一个模块：

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

定义不可变的内部数据结构。代表性结构如下：

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

CLI 命令是首版对外产品接口。上述数据类和模块均为内部实现，不提供兼容性
承诺。

### `discovery.py`

负责文件系统校验、快照解码、排序以及 warning 绑定。它返回按顺序排列的
`Snapshot`，不计算 diff。

### `diff.py`

负责文本分析。它使用 `difflib.SequenceMatcher(autojunk=False)` 比较相邻
快照，计算统计数据，对齐左右行，识别可折叠的相同行区域，并生成带语法
高亮的行片段。

### `html.py`

负责序列化预计算结果并生成完整 HTML。CSS、JavaScript、高亮后的行片段、
before/after 原文、warning 和元数据全部嵌入文件。浏览器端 JavaScript
只管理展示状态，不负责计算 diff。

### `cli.py`

负责 argparse、用户可见错误、退出码和原子写入。`pyproject.toml` 中的
console entry point 为：

```toml
[project.scripts]
pypto-ir-trace = "pypto.tools.ir_trace.cli:main"
```

## 快照发现

输入目录沿用现有 `PassManager` 命名约定：

```text
passes_dump/
├── 00_frontend.py
├── 01_after_InlineFunctions.py
├── 01_after_InlineFunctions.log
├── 02_after_UnrollLoops.py
└── fa_fused_EXTRACT.py
```

发现规则：

1. `PASSES_DUMP` 必须存在且必须是目录。
2. 必须存在且只能使用一个 `00_frontend.py`。
3. pass 快照匹配 `^(?P<index>\d+)_after_(?P<name>.+)\.py$`。
4. pass 索引必须从 `1` 开始、不可重复且必须连续。
5. 带数字前缀、看起来像快照但不符合命名格式的 Python 文件视为错误。
6. `fa_fused_EXTRACT.py` 等不带数字前缀的无关产物会被忽略。
7. 若存在同 stem 的 `.log` 文件，则将其解码为该 pass 的 warning 文本。
8. 所有文件必须是有效 UTF-8。使用 `splitlines()` 归一化换行，因此仅
   CRLF/LF 或末尾换行不同不会将 pass 标记为 changed。
9. 目录中至少要有一个 pass 快照。

pass `N` 比较快照 `N - 1` 与快照 `N`。因此
`01_after_InlineFunctions.py` 的输入是 `00_frontend.py`，
`02_after_UnrollLoops.py` 的输入是 `01_after_InlineFunctions.py`。

## Diff 语义

`SequenceMatcher.get_opcodes()` 生成稳定的行级编辑分组：

- `equal` 不计入统计。
- `insert` 将 after 侧行数计入 `inserted`。
- `delete` 将 before 侧行数计入 `deleted`。
- `replace` 将 before 侧行数计入 `deleted`，同时将 after 侧行数计入
  `inserted`。

替换块内部的 before 和 after 行按位置对齐，较短一侧使用空单元格补齐。
只有归一化后的行 tuple 完全一致时，该 pass 才是 no-op。

相同区域在变化块两侧保留 `--context` 行。较长的中间部分生成一个
`DiffHunk` 折叠项，其中保存省略行数以及浏览器展开时所需的行。未超过
折叠阈值的相同区域完整显示。

Python 语法高亮使用标准库 `tokenize`，并基于完整快照计算 token 位置，
以正确处理多行 token。生成 diff 行之前，将 token 区间转换为已经转义的
逐行 HTML span。如果出现 `TokenError`，或 token 区间无法安全映射，则
该快照回退为完整转义的纯文本。高亮失败不会阻止报告生成。

## HTML 数据与安全

输出为包含内嵌 CSS 和 JavaScript 的完整 HTML5 文档。Trace 数据使用排序
后的 JSON key 和稳定分隔符进行确定性编码。嵌入前转义潜在 HTML 终止字符，
包括 `<`、`>`、`&`、U+2028 和 U+2029，确保 `</script>` 等内容无法结束
数据块或注入可执行标记。

所有文件名、pass 名称、IR 文本和 warning 都来自本地文件，因此必须转义。
除 Python 高亮器在转义 token 内容后生成的受信任片段外，不使用不安全的
`innerHTML`。普通标签和 warning 使用 `textContent`。

报告不包含远程 URL、外部字体、图片、脚本、样式表或动态 import。
复制功能优先使用 `navigator.clipboard.writeText`；在 `file://` 场景下
Clipboard API 不可用时，回退为临时本地 textarea。

为保证确定性，报告数据不包含输入、输出的绝对路径，也不写入生成时间。
界面只显示快照 basename 和通用来源标签，避免机器相关信息。

## 错误处理

领域错误统一使用内部 `IRTraceError`。`cli.main()` 捕获该错误，在 stderr
输出一条可操作的信息并返回 `1`。意外的编程错误不应被吞掉。

示例：

```text
pypto-ir-trace: error: missing 00_frontend.py in build_output/example/passes_dump
pypto-ir-trace: error: no pass snapshots found in build_output/example/passes_dump
pypto-ir-trace: error: missing snapshot index 02 between 01_after_InlineFunctions.py and 03_after_ConvertToSSA.py
pypto-ir-trace: error: malformed snapshot name 02_ConvertToSSA.py; expected NN_after_PassName.py
pypto-ir-trace: error: 03_after_Simplify.py is not valid UTF-8
```

输出先写入输出目录中的唯一临时文件，刷新后通过 `Path.replace()` 原子替换
目标文件。写入失败时删除临时文件，绝不把半成品报告当作成功结果。

## 测试

新增 `tests/ut/tools/test_ir_trace.py`，使用 pytest 函数，并保留项目要求的
`pytest.main([__file__, "-v"])` 文件结尾。

### 快照发现测试

- 文件系统遍历顺序不影响快照顺序。
- 缺少 frontend 或没有 pass 快照时给出明确错误。
- 重复索引、不连续索引和数字前缀畸形名称会报告相关文件名。
- 不带数字前缀的无关 `.py` 文件会被忽略。
- 同 stem warning `.log` 能正确绑定到对应 pass。
- 快照或 warning 中的无效 UTF-8 会给出明确错误。

### Diff 测试

- insert、delete、replace 和 no-op 产生精确统计。
- 替换块左右长度不同时仍正确对齐。
- context 为 `0`、普通 context、文件开头变化和文件结尾变化均正确折叠。
- 仅 CRLF/LF 或末尾换行不同会被判定为 no-op。
- 有效 Python 生成 token span，无效 Python 安全回退为转义纯文本。

### HTML 测试

- 相同输入连续生成两次时，输出逐字节一致。
- 嵌入数据包含正确的 pass 顺序、warning、统计和复制原文。
- `</script>` 以及文件名、IR、warning 中的 HTML 无法注入标记。
- 不包含 `http://`、`https://`、CDN 引用或外部资源。
- 包含主题、过滤、键盘导航、warning、上下文展开和 clipboard fallback。

### CLI 测试

- 默认输出和显式 `--output` 都能生成报告。
- 无效 `--context` 返回 `2`。
- 快照发现和写入失败返回 `1`，并输出 `pypto-ir-trace: error:` 前缀。
- 写入失败时不留下半成品输出或临时文件。

实现阶段运行：

```bash
PYTHONPATH="$(pwd)/python" python3.11 -m pytest tests/ut/tools/test_ir_trace.py -v
ruff check python/pypto/tools/ir_trace tests/ut/tools/test_ir_trace.py
ruff format --check python/pypto/tools/ir_trace tests/ut/tools/test_ir_trace.py
pyright
```

## 文档

实现阶段新增：

- `docs/en/dev/07-ir-lower-trace.md`：英文权威文档。
- `docs/zh-cn/dev/07-ir-lower-trace.md`：结构和行为一致的中文文档。

文档说明如何生成 `passes_dump/`、如何调用 CLI、如何操作报告、如何解释
新增/删除统计，以及文本证据的限制。现有 pass dump 文档继续作为
`PassDumpLevel` 的权威来源，新页面通过链接引用，不重复其内容。

## 实现涉及的文件

- `pyproject.toml`：注册 `pypto-ir-trace` console script。
- `python/pypto/tools/ir_trace/__init__.py`：声明内部工具 package。
- `python/pypto/tools/ir_trace/model.py`：新增不可变 trace 数据结构。
- `python/pypto/tools/ir_trace/discovery.py`：新增快照发现与校验。
- `python/pypto/tools/ir_trace/diff.py`：新增文本 diff、折叠和高亮。
- `python/pypto/tools/ir_trace/html.py`：新增确定性的自包含 HTML renderer。
- `python/pypto/tools/ir_trace/cli.py`：新增 CLI 和原子写入。
- `tests/ut/tools/test_ir_trace.py`：新增单元测试和 CLI 测试。
- `docs/en/dev/07-ir-lower-trace.md`：新增英文使用文档。
- `docs/zh-cn/dev/07-ir-lower-trace.md`：新增中文镜像文档。

不修改 C++、binding、type stub、pass pipeline 或 codegen 文件。

## 实现顺序

1. 先添加失败的快照发现测试，再实现 `model.py` 和 `discovery.py`。
2. 添加失败的 diff、统计和折叠测试，再实现 `diff.py`。
3. 添加失败的转义、确定性和报告测试，再实现 `html.py`。
4. 添加失败的 CLI 测试，实现 `cli.py` 并注册 console script。
5. 添加同步的英文和中文文档。
6. 运行聚焦测试、lint、format、类型检查和项目提交工作流。

每一步只依赖前一步定义的内部接口，从而支持聚焦的测试驱动迭代，并保持
浏览器展示与文件系统、diff 逻辑解耦。

## 未采用的方案

### 在浏览器 JavaScript 中计算 diff

该方案可以减少 Python 代码，但会把正确性、统计和性能问题转移到更难测试的
客户端代码。大型 IR 快照也会增加页面加载成本，因此首版采用预计算。

### 内嵌第三方 JavaScript diff 或高亮库

该方案能更快获得视觉效果，但会引入版本、许可证、bundle 体积和离线打包
问题。首版使用标准库即可满足需求。

### 将全部逻辑放入单个 `ir_trace.py`

单文件起步较快，但会混合文件系统校验、编辑语义、HTML 安全和 CLI 行为。
拆分模块能形成更清晰的接口，也便于在不构造完整报告的情况下独立测试。

## 后续工作

后续可以增加 terminal 渲染、公开 Python API 和 `compare-codegen` 集成。
这些功能应复用本设计中的快照发现与 diff 模型，而不是引入第二套快照路径。
