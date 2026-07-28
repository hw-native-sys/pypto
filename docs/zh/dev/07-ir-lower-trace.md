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
均为 no-op，则选择第一个 Pass。

### 导航和检查

在侧栏中选择 Pass，可并排比较其输入和输出。按 `j` 或 `Down Arrow` 移至下一个
可见 Pass，按 `k` 或 `Up Arrow` 移至上一个可见 Pass。当文本输入控件获得焦点时，
键盘导航（keyboard navigation）会被忽略。

替换行使用浅红色和浅绿色的删除/新增背景，并以更深的红色和绿色突出实际变化的
字符。仅存在于一侧的行仍显示为整行删除或新增。滚动任一代码面板时，前后视图会
同时同步垂直和水平位置。

### 复制快照

使用任一窗格上方的 **Copy full source** 复制完整的 before 或 after 快照，
其中包括因上下文折叠而隐藏的未变化行。复制（copy）会优先使用浏览器剪贴板
API，不可用时使用本地回退方式。

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
