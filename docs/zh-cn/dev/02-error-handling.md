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
  └── Error                  (基类：自动栈回溯捕获)
        ├── ValueError       (→ Python ValueError)
        ├── TypeError        (→ Python TypeError)
        ├── RuntimeError     (→ Python RuntimeError)
        ├── NotImplementedError
        ├── IndexError
        ├── AssertionError
        ├── InternalError    (→ Python RuntimeError — 内部 bug)
        └── VerificationError (携带 vector<Diagnostic>)
```

`Error::GetFullMessage()` 返回错误消息加上格式化的 C++ 栈回溯。

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
