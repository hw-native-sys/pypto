# IR Trace

把整条降级流水线呈现为一份可导航的 diff：哪个 pass 改了什么，按顺序排列。

## 概念

一份 pass dump 是一个编号快照的目录。diff 相邻两份就能把一处改动归因到某个 pass —— 但手工做五十遍，一个下午就没了。`pypto-ir-trace` 把整个序列渲染成一份自包含的 HTML 报告，并把什么都没改的 pass 过滤掉。

它的输入就是[内存图](02-memory-map.md)读的那个 `passes_dump/`。不执行任何东西；这是编译器在每一步打印出的文本的画像。

## 快速上手

先产出 dump，再渲染：

```python
from pypto.ir import PassDumpLevel
from pypto.runtime import RunConfig

compiled = kernel.compile(*args, config=RunConfig(dump_passes=PassDumpLevel.EXPLICIT))
print(compiled.output_dir)
```

```bash
OUT=build_output/<program>_<timestamp>_<random>  # what the line above printed
pypto-ir-trace "$OUT/passes_dump" -o ir_trace.html
```

用浏览器打开 `ir_trace.html`。样式、脚本、数据全部内嵌 —— 不需要服务器，不需要网络。

只设了 `PYTHONPATH` 的源码 checkout 没有 console script；模块入口接受同样的参数和退出码：

```bash
python -m pypto.tools.ir_trace "$OUT/passes_dump" -o ir_trace.html
```

## 机制

### 命令行

| 参数 | 含义 |
| ---- | ---- |
| `passes_dump` | dump 目录：`00_frontend.py`、`01_after_*.py`、…… 编号不能有缺口 |
| `-o PATH`, `--output PATH` | 报告路径（默认 `ir_trace.html`）；该目录必须已存在 |
| `--context N` | 每处改动周围保留的未变行数（默认 `3`，非负） |

报告先写到目标目录下的临时文件再原子改名，所以写失败绝不会留下半份报告。退出码：`0` 成功，`1` dump 内容非法或 I/O 失败，`2` 参数错误。

### 怎么读

侧栏按执行顺序列出各 pass，带插入/删除行数，以及该 pass 发出警告时的徽标。**Changed** 与 **No-op** 两个过滤器互相独立；打开时默认选中第一个有改动的 pass。

| 控件 | 作用 |
| ---- | ---- |
| `j` / `k`、方向键 | 下一个 / 上一个可见 pass |
| **Side by side** / **Stacked** | 布局；两栏横竖双向同步滚动 |
| **Function** 选择器 | 把 diff 聚焦到某个顶层函数或类方法 |
| **Copy full source** | 复制整份快照，含被上下文折叠隐藏的行 |
| **Expand all** / **Collapse all** | 覆盖 `--context` 的折叠 |

**Function 选择器是第一天就值得知道的那个。** 一个只动了某个 InCore 函数的 pass 仍然会重新打印整个程序；把 diff 限定到那个函数，是「一处可读的改动」与「一面噪声墙」之间的差别。只要函数在新旧两侧都存在，选择会在切换 pass 时保留。

### 用哪个 dump 等级

做文本比较通常 `CONCISE` 最清晰。当问题涉及 tile layout 或 distributed window buffer 时用 `EXPLICIT` —— 这些在 `CONCISE` 下是隐式的，因此在 diff 里根本看不见。

### ptoas 有自己的 dump

`dump_passes` 观测的是 PyPTO 的流水线。后端自己的 MLIR 流水线是另一个开关：

```python
RunConfig(dump_ptoas_passes=True)     # -> <output_dir>/ptoas_passes/<codegen-unit>/
```

每个 codegen 单元一个目录，以免并行的 ptoas 调用互相覆盖。里面的文件由 ptoas/MLIR 命名，`pypto-ir-trace` 不读它们。`skip_ptoas=True` 下无效 —— 那时根本没有 ptoas 流水线在跑。

## 边界情况

| 现象 | 原因 | 处理 |
| ---- | ---- | ---- |
| **`input directory does not exist`** | 指到了输出目录，而不是 `passes_dump/` | 补上 `/passes_dump` |
| **根本没有 `passes_dump/`** | `lower()` 不落盘；或 `dump_passes=False` | 用 `compile()` 并带上 `dump_passes=` |
| **所有 pass 都显示 No-op** | dump 只有一份快照，或流水线确实什么都没改 | 检查 dump 里除 `00_frontend.py` 外是否还有 `NN_after_*` |
| **某个 pass 的 Function 选择器被禁用** | 该快照无法安全解析 | 这个 pass 退回 Whole file |
| **layout 显示为未解析** | dump 用的是 `CONCISE` | 改用 `EXPLICIT` 重新 dump |

> **文本 diff 不是语义 diff。** 报告有变化只说明打印出的文本变了，不能证明程序的含义变了；反过来，文本相似也不能证明含义没变。用 trace *定位*某个降级步骤，再用 [torch codegen](01-torch-codegen.md) 或 `validate_ir` 判断它对不对。

## 参见

- [内存图](02-memory-map.md) —— `passes_dump/` 的另一个读者。
- [调试](00-debugging.md) —— dump 等级，以及怎么读降级后的形态。
- [精度](../precision/00-workflow.md) —— 二分 pass 在那条收敛流程里的位置。
- [IR Lowering Trace](../../dev/07-ir-lower-trace.md) —— 查看器的完整行为与 diff 对齐规则。
- [Pass 管理器](../../dev/passes/00-pass_manager.md) —— 这些编号的 pass 分别是什么。
