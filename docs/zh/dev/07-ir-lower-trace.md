# IR 降低追踪

`pypto-ir-trace` 将 PyPTO 的 `passes_dump/` 目录转换为确定性的自包含 HTML
报告。该报告比较每个 Pass 的输入与输出，无需 Web 服务器或网络访问即可检查
降低过程的变化。

## 生成 Pass 转储

编译程序时启用逐 Pass 转储。`dump_passes=True` 会生成简洁的规范 IR，通常最适合
文本比较：

```python
from pypto import ir

ir.compile(MyProgram, output_dir="build/my_program", dump_passes=True)
```

当追踪必须包含完全解析的 tile 布局和分布式 window-buffer 引用时，使用
`PassDumpLevel.EXPLICIT`：

```python
from pypto import ir
from pypto.ir import PassDumpLevel

ir.compile(
    MyProgram,
    output_dir="build/my_program",
    dump_passes=PassDumpLevel.EXPLICIT,
)
```

两种形式都会创建 `build/my_program/passes_dump/`，其中包含 `00_frontend.py`
以及连续编号的 `NN_after_PassName.py` 快照。转储级别和 Pass 流水线行为详见
[Pass 管理器文档](passes/00-pass_manager.md)。

## 生成报告

使用转储目录运行已安装的命令：

```bash
pypto-ir-trace build/my_program/passes_dump
```

`pypto-ir-trace` 命令由 `pip install` 生成。若源码检出仅设置了 `PYTHONPATH`，
请改用模块入口点（module entry point），其参数与退出码完全一致：

```bash
python -m pypto.tools.ir_trace build/my_program/passes_dump
```

默认输出为当前目录中的 `ir_trace.html`。输出会先写入目标目录中的临时文件，
再原子替换（atomic replacement）指定路径，因此写入失败不会留下不完整报告。

### CLI 选项

| 参数 | 说明 |
| ---- | ---- |
| `passes_dump` | 包含有序 Pass 快照的输入目录。 |
| `-o PATH`, `--output PATH` | 输出报告路径；默认为 `ir_trace.html`。 |
| `--context N` | 每处变化周围显示的未变化行数；默认为 `3`，且必须为非负数。 |

例如，在变化周围保留一行未变化内容，并指定输出位置：

```bash
pypto-ir-trace build/my_program/passes_dump --context 1 -o build/ir-trace.html
```

## 使用查看器

在浏览器中打开生成的 HTML 文件。所有样式、脚本和追踪数据都嵌入该文件；
它不会加载外部资源。

### 侧栏和过滤器

侧栏（sidebar）按执行顺序列出 Pass，并显示插入/删除行数、变化状态和 warning
标记。**Changed** 与 **No-op** 过滤器（filter）可分别显示或隐藏改变了打印 IR
或未改变打印 IR 的 Pass。查看器最初选择第一个有变化的 Pass；如果所有 Pass
均为 no-op，则选择第一个 Pass。侧栏与对比面板可以独立上下滚动，侧栏标题和
过滤器会固定在侧栏顶部。

### 导航和检查

在侧栏中选择 Pass，可比较其输入和输出。按 `j` 或 `Down Arrow` 移至下一个可见
Pass，按 `k` 或 `Up Arrow` 移至上一个可见 Pass。当输入框或选择控件获得焦点时，
键盘导航（keyboard navigation）会被忽略。

使用 **Side by side** 将 Before 和 After 左右排列，或使用 **Stacked** 将 Before
放在 After 上方。默认使用 Side by side。打开报告期间切换 Pass 或 function 时会
保留所选布局；重新加载页面后恢复默认布局。工具栏保持可见，对比窗格占用视口剩余
高度。两个窗格各自可以滚动；在两种布局中，滚动任一窗格都会同步 Before 和 After
的上下及左右位置。

### 按 function 比对

使用 **Function** 选择器可聚焦某个顶层 function，或顶层 class 的直接 method。
**Whole file** 始终是第一个选项和默认选项。function 按完整限定 key 精确匹配；
class method 使用 `Program.run` 形式的 key。短名称无歧义时只显示短名称；存在同名
method 时显示限定 key。仅存在于 After 的 function 显示为新增，仅存在于 Before
的 function 显示为删除。

切换 Pass 时，如果所选 function 仍存在于新对比的任一侧，查看器会保留选择；否则
回退到 Whole file。嵌套 function 保留在所属 function 内，不单独列出。如果任一
快照无法安全解析，该 Pass 会禁用 function 选择并使用原有的 Whole file diff。

在替换块中，查看器会先忽略行首空白，对齐其余内容相同的行。仅缩进不同的行仍显示
为替换行，使结构变化保持可见。其余变化区域按单行 Python 调用的完整操作名对齐。
这样既能保持控制流头和相关操作的配对，也能将实际新增或删除的内容保留为单侧行。
替换行使用浅红色和浅绿色背景，并以更深的红色和绿色突出实际变化的字符。左右窗格
使用相同宽度的可滚动画布，因此行背景会覆盖完整代码行。

### 复制快照

使用任一窗格上方的 **Copy full source** 复制完整的 before 或 after 快照，
其中包括因上下文折叠而隐藏的未变化行。即使选择了单个 function，该操作仍复制
完整快照。复制（copy）会优先使用浏览器剪贴板 API，不可用时使用本地回退方式。

### Warning

如果快照具有匹配的 `.log` 文件，其文本会显示在 warning 面板中，并为该 Pass
添加 warning 标记。warning 是 Pass 运行产生的诊断上下文，不会改变文本差异。

### 主题和折叠上下文

使用 **Theme** 在浅色与深色主题（theme）之间切换。初始主题遵循浏览器首选
配色。较长的未变化区域会根据 `--context` 折叠（collapse）；可以单击某个折叠
区域，或使用 **Expand all** 与 **Collapse all** 改变其可见性。

## 错误处理

参数语法错误和无效的 `--context` 值使用 argparse 诊断，并以状态码 `2` 退出。
无效转储内容及输入/输出 I/O 失败会向标准错误打印一条简洁诊断，并以状态码
`1` 退出。成功写入报告时以状态码 `0` 退出。

```text
$ pypto-ir-trace missing/passes_dump
pypto-ir-trace: error: input directory does not exist: missing/passes_dump

$ pypto-ir-trace passes_dump --context -1
pypto-ir-trace: error: argument --context: must be non-negative, got -1
```

输入目录必须包含有效 UTF-8 快照，命名为 `00_frontend.py`、
`01_after_*.py`、`02_after_*.py` 等，且索引不能有缺口。输出目录必须已存在。

## 解释和限制

查看器计算打印 IR 的逐行**文本差异（textual diff）**。报告发生变化意味着序列化
文本发生了变化，并不能证明程序语义已经改变。反之，文本相似也不能证明语义
等价。应使用追踪定位降低步骤，再通过 IR 验证和行为测试判断语义正确性。
