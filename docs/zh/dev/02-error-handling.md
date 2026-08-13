# 错误处理（Error Handling）

PyPTO 的错误处理框架提供带 C++ 栈回溯的结构化异常、附带 IR 源码位置的断言宏，以及用于验证错误的诊断系统。

## 概述

| 组件 | 头文件 | 用途 |
| ---- | ------ | ---- |
| **异常体系** | `include/pypto/core/error.h` | 类型化异常（`ValueError`、`InternalError` 等），自动捕获栈回溯 |
| **断言宏** | `include/pypto/core/logging.h` | `CHECK` / `CHECK_SPAN`、`INTERNAL_CHECK_SPAN`、`UNREACHABLE` / `UNREACHABLE_SPAN` 等 |
| **诊断系统** | `include/pypto/core/error.h` | `Diagnostic` / `VerificationError`，用于验证 pass |
| **Span** | `include/pypto/ir/span.h` | IR 源码位置，附加到诊断和内部检查中 |

## 异常体系

所有异常继承自 `Error`，`Error` 在构造时通过 `libbacktrace` 自动捕获 C++ 栈回溯。

```text
std::runtime_error
  └── Error                  (基类：自动栈回溯捕获，→ Python pypto.Error)
        ├── ValueError       (→ Python ValueError)
        ├── TypeError        (→ Python TypeError)
        ├── RuntimeError     (→ Python RuntimeError)
        ├── NotImplementedError
        ├── IndexError
        ├── AssertionError
        ├── InternalError    (→ Python RuntimeError — 内部 bug)
        └── VerificationError (携带 vector<Diagnostic>，→ Python pypto.Error)
```

没有专门映射的子类会回退到 `pypto.Error`——这是一个真实的 Python 类型，而非裸 `Exception`，
因此 `VerificationError` 仍可按类型捕获。`pypto.Error` 继承自 `Exception`，所以 `except Exception`
依然有效。测试必须断言具体的异常类型而非 `Exception`；`tests/lint/check_no_broad_raises.py` 会强制这一点。

**Python 侧的映射是扁平的，而非嵌套的。** 上表中每一行都映射到一个*独立*的 Python 类型，因此
上面树状图中的 C++ 继承关系并不会传递过来：`pypto.InternalError` 继承自 Python 的 `RuntimeError`，
而不是 `pypto.Error`。所以 `except pypto.Error` 能捕获 `VerificationError`，但**捕获不到**
`InternalError`。如果两者都需要捕获，请捕获 `Exception`。

`Error::GetFullMessage()` 返回错误消息加上格式化的 C++ 栈回溯。

### 何时显示栈回溯

每个 `Error` 在构造时都会捕获栈回溯，但异常转换器（`python/bindings/modules/error.cpp`）
只在有帮助时才把它附加到 Python 消息中：

| 异常 | 抛出来源 | Python 消息中是否包含回溯 |
| ---- | -------- | ------------------------- |
| `InternalError`、`AssertionError` | `INTERNAL_CHECK` 系列 | 总是包含——内部不变量失败属于 PyPTO 的 bug，栈帧是首要排查依据 |
| `ValueError`、`TypeError`、`RuntimeError`、`IndexError`、`NotImplementedError`、`VerificationError` | `CHECK` 系列、用户输入 | 仅在 `PTO_BACKTRACE=1` 时包含 |

用户错误本身已经携带 DSL 源码片段；C++ 栈帧指向用户无法处理的 PyPTO 内部实现，夹在中间还会把
源码片段挤到更下方。`NotImplementedError` 同样属于这一类：尚未下降的特性是面向用户的、有文档
记录的能力限制，与 `CHECK` 覆盖的范畴相同，并非内部不变量失败。`PTO_BACKTRACE=1` 与 DSL 诊断
提示使用的是同一个开关，因此一个环境变量即可同时打开两种回溯。

```bash
PTO_BACKTRACE=1 python my_kernel.py   # 所有错误都显示 C++ 栈帧，而不仅仅是内部错误
```

只有精确取值 `1` 才会开启 C++ 回溯，其他取值一律视为关闭。DSL 解析器更严格——除 `0` 和 `1`
之外的取值都会报错——因此请只使用这两个值。

`Backtrace::FormatStackTrace` 会丢弃属于基础设施而非调用路径的栈帧——`libbacktrace`、`nanobind`、
libc、C++ 标准库，以及 `error.h` / `logging.h` 中的抛出点（见 `src/core/backtrace.cpp` 的
`kFileNameFilter`）。

### 栈回溯的平台支持

`3rdparty/libbacktrace` 跟随上游 [ianlancetaylor/libbacktrace](https://github.com/ianlancetaylor/libbacktrace)。
上游的 Mach-O 读取器只接受 `MH_EXECUTE`、`MH_DYLIB` 和 `MH_DSYM`，而 CPython 扩展模块是
`MH_BUNDLE`——因此在 **macOS** 上符号化会失败，`GetFullMessage()` 退化为
`No stack trace available`。任何构建模式都无法改变这一点：这是文件类型（filetype）的限制，
而非缺少调试信息。Linux（ELF）不受影响，但仍然需要调试信息：`Debug` 和 `RelWithDebInfo`
（默认值）会传入 `-g`，可生成完整回溯；而纯 `Release` 构建为 `-O2 -DNDEBUG`、不含 `-g`，
栈帧因此没有源码位置，同样会退化为上述提示。

`Backtrace::ErrorCallback` 对每个不同的 `(消息, errno)` 组合只上报一次。否则在 macOS 上，每次
捕获回溯都会为**每个栈帧**打印一行 `no debug info in Mach-O executable`——因为 dyld 初始化
路径整体上是成功的，并会把 `macho_nodebug` 装为 fileline 处理函数。

## 断言宏

### 面向用户的检查 — `CHECK` / `CHECK_SPAN`

当违反用户可见的约定时抛出 `ValueError`。`CHECK_SPAN` 额外附加 IR 源码位置 —— 与 `INTERNAL_CHECK_SPAN` 对称,当有 `Span` 可达时优先使用,以便用户看到是哪一行 DSL 触发了检查:

```cpp
CHECK(args.size() == 2) << "op requires exactly 2 arguments, got " << args.size();
CHECK_SPAN(shape.size() == 2, span) << "tensor.matmul: only 2D inputs are supported";
```

`span` 参数遵循与 `INTERNAL_CHECK_SPAN` 相同的安全规则:它仅在失败时求值,但在失败路径中是无条件求值的。因此 span 来源必须在失败点可以安全求值(典型如局部 `Span` 变量或已确认非空的兄弟 IR 节点)。

### 内部不变式检查 — `INTERNAL_CHECK_SPAN`

当违反内部不变式时抛出 `InternalError`。始终附加 IR 节点的 `Span`，使错误消息包含用户源代码位置：

```cpp
INTERNAL_CHECK_SPAN(op->var_, op->span_) << "AssignStmt has null var";
INTERNAL_CHECK_SPAN(new_value, op->span_) << "AssignStmt value mutated to null";
```

检查失败时，错误消息同时包含 IR 源码位置和 C++ 位置：

```text
AssignStmt has null var [user_model.py:42:1]
Check failed: op->var_ at src/ir/transforms/mutator.cpp:301
```

还有 `INTERNAL_UNREACHABLE_SPAN` 用于不应到达的代码路径：

```cpp
INTERNAL_UNREACHABLE_SPAN(span) << "Unknown binary expression kind";
```

### 不带 span 的变体

`CHECK` / `UNREACHABLE` / `INTERNAL_CHECK` / `INTERNAL_UNREACHABLE` 不携带 IR 源码位置。它们适用于没有 `Span` 可用的场景（例如非 IR 上下文中的算术工具或注册表查找,或解析结构性失败发生在 span 字段被读取之前的场合）。当正在处理 IR 节点且 `op->span_` 可访问时，应优先使用 `_SPAN` 变体。

### 不可达代码路径 — `UNREACHABLE` / `UNREACHABLE_SPAN`

对于从用户角度不应到达的代码路径，抛出 `ValueError`。当有 IR span 可用时优先使用 `UNREACHABLE_SPAN`:

```cpp
UNREACHABLE << "Unsupported data type: " << dtype;
UNREACHABLE_SPAN(node->span_) << "Unsupported data type: " << dtype;
```

### 宏参考

| 宏 | 异常类型 | Span | 状态 |
| -- | -------- | ---- | ---- |
| `CHECK(expr)` | `ValueError` | 无 | 可用 |
| `CHECK_SPAN(expr, span)` | `ValueError` | 有 | **有 span 时推荐** |
| `UNREACHABLE` | `ValueError` | 无 | 可用 |
| `UNREACHABLE_SPAN(span)` | `ValueError` | 有 | **有 span 时推荐** |
| `INTERNAL_CHECK_SPAN(expr, span)` | `InternalError` | 有 | **推荐** |
| `INTERNAL_UNREACHABLE_SPAN(span)` | `InternalError` | 有 | **推荐** |
| `INTERNAL_CHECK(expr)` | `InternalError` | 无 | 可用（有 span 时用 `_SPAN`） |
| `INTERNAL_UNREACHABLE` | `InternalError` | 无 | 可用（有 span 时用 `_SPAN`） |

## 诊断系统

诊断系统由 [IR 验证 pass](passes/99-verifier.md) 使用，在报告前收集多个问题。

每个 `Diagnostic` 携带：

| 字段 | 类型 | 用途 |
| ---- | ---- | ---- |
| `severity` | `DiagnosticSeverity` | Error 或 Warning |
| `rule_name` | `string` | 检测到问题的验证规则名称 |
| `error_code` | `int` | 数字错误标识符 |
| `message` | `string` | 可读的错误描述 |
| `span` | `Span` | IR 源码位置 |

验证失败时会抛出 `VerificationError`，携带所有收集到的诊断。

## Span 与源码位置

每个 IR 节点从 `IRNode` 继承 `span_` 字段（见 [IR 概述](ir/00-overview.md)）。该字段跟踪用户的源码位置（文件名、行、列），用于两条错误路径：

1. **验证诊断** — 验证 pass 将 `op->span_` 记录到 `Diagnostic` 对象中
2. **断言检查** — `CHECK_SPAN` / `UNREACHABLE_SPAN` / `INTERNAL_CHECK_SPAN` / `INTERNAL_UNREACHABLE_SPAN` 将 `span.to_string()` 嵌入失败消息

当 `Span` 有效时，错误输出在消息末尾追加 `[file:line:col]`。使用 `Span::unknown()` 时，不显示源码位置。

### Pass 中的 span 归属

Pass 合成或重建 IR 节点时，必须赋予它**所代表节点**的 span，而不是外层函数的 span。
使用 `func->span_` 很方便——它在整个变换过程中都可见——但这会让 pass 触及的每个节点
都报告 `def` 行，从而悄悄降低所有读取 `Call` span 的消费方的精度：后续 pass 抛出的
`CHECK_SPAN` / `INTERNAL_CHECK_SPAN` 诊断、IR trace 报告，以及按 span 归并的验证检查
（PH001 性能提示按源码位置去重，span 被粗化后会把互不相关的搬运合并成一条错误的
"N occurrences" 提示）。

```cpp
// ❌ 每个合成的算子都报告 `def` 行
const auto& span = func->span_;
for (const auto& stmt : body) { /* ... */ op_registry.Create(name, args, span); }

// ✅ 将每个节点归属到正在重写的语句
for (const auto& stmt : body) {
  const Span& span = stmt->span_;
  /* ... */ op_registry.Create(name, args, span);
}
```

选择促成新节点的最近节点：正在重写的语句、正在转换的 `Call`（`call->span_`）、
前置 load 所读取的参数（`var->span_`），或后置 store 所服务的 `ReturnStmt`。
只有真正属于整个函数的节点——重建的 `Function` 本身及其函数体 `SeqStmts`——
才应使用 `func->span_`。

## Python API

```python
import pypto

# 面向用户的检查（抛出 ValueError）
pypto.check(condition, "error message")

# 带 span 的内部不变式检查（抛出 RuntimeError）
pypto.internal_check_span(condition, "error message", span)

# 带 span 抛出 InternalError（用于测试或无条件错误路径）
pypto.raise_internal_error_with_span("error message", span)

# 不带 span 的内部不变式检查
pypto.internal_check(condition, "error message")
```

## 迁移指南

在 IR 变换、pass 或 codegen 中编写或修改代码时:

1. 确定当前处理的 IR 节点（`op`、`stmt`、`expr` 等）
2. 将 `INTERNAL_CHECK(expr)` 替换为 `INTERNAL_CHECK_SPAN(expr, op->span_)`(以及 `INTERNAL_UNREACHABLE` 替换为 `INTERNAL_UNREACHABLE_SPAN`)
3. 同样地,当有 span 可达时,将面向用户的 `CHECK(expr)` 替换为 `CHECK_SPAN(expr, op->span_)`(以及 `UNREACHABLE` 替换为 `UNREACHABLE_SPAN`)
4. 如果函数参数中已有 `Span`（例如 `Reconstruct*` 辅助函数或算子转换 lambda），直接使用该参数

```cpp
// 之前：
INTERNAL_CHECK(op->body_) << "ForStmt has null body";
CHECK(args.size() == 2) << "tensor.matmul requires 2 args";

// 之后（当 span 可用时推荐）：
INTERNAL_CHECK_SPAN(op->body_, op->span_) << "ForStmt has null body";
CHECK_SPAN(args.size() == 2, span) << "tensor.matmul requires 2 args";
```

### Pass 内部:CHECK vs INTERNAL_CHECK

Pass 处理的 IR 已被早期 pass 验证过。Pass 中的失败不变式因此几乎总是表明**编译器 bug**,而非用户错误 —— 应使用 `INTERNAL_CHECK_SPAN` / `INTERNAL_UNREACHABLE_SPAN`。仅当确实需要将文档化的用户限制(例如 "4D scatter_update 尚未下沉,请使用 2D")作为用户错误暴露时,才使用 `CHECK_SPAN`。如果不确定,自问:消息读起来是 "这是 PyPTO bug,请上报" 还是 "请修改你的代码"?

## 相关文档

- [IR 概述 — 源码位置跟踪](ir/00-overview.md)
- [IR 验证器 — 诊断系统](passes/99-verifier.md)
- `include/pypto/core/error.h` — 异常类和 `Diagnostic`
- `include/pypto/core/logging.h` — 断言宏和 `FatalLogger`
- `include/pypto/ir/span.h` — `Span` 类
