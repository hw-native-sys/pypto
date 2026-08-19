# 调试

错误类型、日志级别，以及读懂编译器产出的 IR。

## 概念

PyPTO 把两类失败分开，而这个区分是你从一条报错里最先该读出的东西。**用户错误**意味着输入不合法 —— 一个不可能成立的形状、一个不支持该 dtype 的算子 —— 报错会说清该改什么。**内部错误**意味着编译器自己维护的某条不变量被打破了，那是编译器的 bug，与你的输入无关。

两者都以 Python 异常的形式到达，区别在类型与措辞。

## 快速上手：读编译器产出的 IR

<!-- doctest: setup -->
```python
import pypto.language as pl
import torch
from pypto.runtime import RunConfig

CFG = RunConfig(platform="__PLATFORM__")
torch.manual_seed(0)
A = torch.randn(64, 128, dtype=torch.float32)
```

<!-- doctest: run -->
```python
@pl.jit
def scale(a: pl.Tensor, out: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        out[:] = pl.mul(a, 2.0)
    return out


prog = scale.lower(A, torch.zeros(64, 128))   # passes only: no codegen, no artifacts
src = prog.as_python()                        # the lowered IR, as DSL source

# What comes back is post-pass IR, not what you wrote: the pl.at scope has been
# outlined into its own InCore function.
assert "@pl.program" in src
assert "def scale_incore_0(" in src
assert "pl.at" not in src

print(prog.as_python(concise=True)[:200])     # concise= drops intermediate type annotations
```

`lower()` 是便宜的那种形式 —— 只跑 pass 并返回 `Program`，什么都不写。`as_python()` 在 `Program` 或单个 `Function` 上都有；`JITFunction` 自己没有，因为在出现特化之前 IR 并不存在。

**要预期 IR 不长得像你的源码。** 上面那几条断言正是重点：pass 跑完之后 `pl.at` 已经不在了 —— 那个区域变成了 `scale_incore_0`，一个 role 为 `SubWorker` 的 `AIV` 函数，其 tile 带着显式的 `pl.MemRef` 分配。读 dump 就是读那种形态，而不是你写的形态。

## 机制

### 错误类型

| 类型 | 含义 | 该做什么 |
| ---- | ---- | -------- |
| `pypto.Error` | PyPTO 异常体系的基类 | — |
| `ValueError` / `TypeError` / `IndexError` | `CHECK` 抛出的用户错误 | 改输入；报错会写明期望什么、来了什么 |
| `pypto.InternalError` | 不变量被打破 —— 编译器 bug | 提 bug，附上可复现的 IR |
| `PartialCodegenError` | codegen 产出了一部分 kernel，另一部分失败 | 报告会点名是哪些；通常是 ptoas 拒绝 |

内部错误会在文本里自报家门（`Internal error: ...`），并带上失败那条检查的源位置。那个位置在编译器里，不在你的 kernel 里 —— 有 DSL span 时会一并打印。

### 日志级别

```python
import pypto

pypto.set_log_level(pypto.LogLevel.DEBUG)
print(pypto.get_log_level())
```

`NONE` / `FATAL` / `ERROR` / `WARN` / `INFO` / `EVENT` / `DEBUG`，依次更吵。默认是 `INFO`。

### 环境变量

| 变量 | 控制 |
| ---- | ---- |
| `PYPTO_VERIFY_LEVEL` | 没有 `PassContext` 指定时的默认 IR 校验级别 |
| `PYPTO_WARNING_LEVEL` | 默认的诊断阶段门 |
| `PYPTO_PROG_BUILD_DIR` | 产物的基准目录（默认 `build_output`） |
| `PYPTO_EMIT_PTO_LOC` | 把 DSL 源位置带进发出的 `.pto` |
| `PYPTO_COMPILE_PROFILING` | 逐阶段编译计时 |
| `PYPTO_EMIT_DEBUG_RUNNER` | 在产物旁边发出独立的 debug runner |

每一个都只是**默认值**：显式实参或活跃的 `PassContext` 会覆盖它。

### Pass dump

`dump_passes=PassDumpLevel.EXPLICIT` 会按执行顺序写出 `passes_dump/NN_after_<PassName>.py`。有两个页面读它：本页，回答「哪个 pass 改了我的 IR」；以及[内存图](02-memory-map.md)，回答分配长什么样。

把相邻两份 dump 做 diff，是把一处改动归因到某个 pass 的机械办法。`CompiledProgram.validate_ir` 把这件对比里语义的那一半自动化了 —— 见[精度](../precision/00-workflow.md)，包括那里关于它容差的提醒。

### 让 IR 经文本往返

一个程序可以写成 DSL 源码再读回来：

| 方向 | 调用 |
| ---- | ---- |
| IR → 文本 | `program.as_python()` |
| 文本 → IR | `pl.parse_program(code)` |
| 文件 → IR | `pl.loads_program(path)` |

`examples/utils/parse_from_text.py` 是完整版本。这条往返也正是 `VerificationLevel.ROUNDTRIP` 在每个 pass 上所做的事 —— 它慢到必须选择性开启，原因就在这里。

## 边界情况

| 现象 | 可能原因 | 修法 |
| ---- | -------- | ---- |
| **`JITFunction` 没有 `as_python`** | 特化之前 IR 并不存在 | `kernel.lower(*args).as_python()` |
| **报错点的是编译器文件，不是你的 kernel** | 那是 `InternalError` | 提 bug，不要绕过去 |
| **`ptoas compilation failed:` 后面是空的** | ptoas 二进制崩了 | 把 `PTOAS_ROOT` 指向一个能用的版本 |
| **没有 `passes_dump/`** | `lower()` 不写产物 | 用带 `dump_passes=` 的 `compile()` |

> **内部错误不该由你绕开。** 用改形状或换写法把它压下去，只是掩盖了一条被打破的不变量，它会在更不方便的地方再冒出来。

## 参见

- [Torch codegen](01-torch-codegen.md) —— 在 host 上跑 IR 的语义。
- [内存图](02-memory-map.md) —— pass dump 的另一个读者。
- [精度](../precision/00-workflow.md) —— 这些工具接入的那条流程。
- [错误处理](../../dev/02-error-handling.md) —— 上表类型背后的 `CHECK` / `INTERNAL_CHECK` 契约。
