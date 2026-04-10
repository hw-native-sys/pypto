# 错误处理（Error Handling）

PyPTO 的错误处理框架提供带 C++ 栈回溯的结构化异常、附带 IR 源码位置的断言宏，以及用于验证错误的诊断系统。

## 概述

| 组件 | 头文件 | 用途 |
| ---- | ------ | ---- |
| **异常体系** | `include/pypto/core/error.h` | 类型化异常（`ValueError`、`InternalError` 等），自动捕获栈回溯 |
| **断言宏** | `include/pypto/core/logging.h` | `CHECK`、`INTERNAL_CHECK_SPAN`、`UNREACHABLE` 等 |
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

### 面向用户的检查 — `CHECK`

当违反用户可见的约定时抛出 `ValueError`：

```cpp
CHECK(args.size() == 2) << "op requires exactly 2 arguments, got " << args.size();
```

### 内部不变式检查 — `INTERNAL_CHECK_SPAN`

当违反内部不变式时抛出 `InternalError`。始终附加 IR 节点的 `Span`，使错误消息包含用户源代码位置：

```cpp
INTERNAL_CHECK_SPAN(op->var_, op->span_) << "AssignStmt has null var";
INTERNAL_CHECK_SPAN(new_value, op->span_) << "AssignStmt value mutated to null";
```

检查失败时，错误消息同时包含 IR 源码位置和 C++ 位置：

```text
AssignStmt has null var
  Source location: user_model.py:42:1
Check failed: op->var_ at src/ir/transforms/mutator.cpp:301
```

还有 `INTERNAL_UNREACHABLE_SPAN` 用于不应到达的代码路径：

```cpp
INTERNAL_UNREACHABLE_SPAN(span) << "Unknown binary expression kind";
```

### 不带 span 的变体

`INTERNAL_CHECK` 和 `INTERNAL_UNREACHABLE` 不携带 IR 源码位置。它们适用于没有 `Span` 可用的场景（例如非 IR 上下文中的算术工具或注册表查找）。当正在处理 IR 节点且 `op->span_` 可访问时，应优先使用 `_SPAN` 变体。

### 不可达代码路径 — `UNREACHABLE`

对于从用户角度不应到达的代码路径，抛出 `ValueError`：

```cpp
UNREACHABLE << "Unsupported data type: " << dtype;
```

### 宏参考

| 宏 | 异常类型 | Span | 状态 |
| -- | -------- | ---- | ---- |
| `CHECK(expr)` | `ValueError` | 无 | 可用 |
| `UNREACHABLE` | `ValueError` | 无 | 可用 |
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
2. **内部断言检查** — `INTERNAL_CHECK_SPAN` / `INTERNAL_UNREACHABLE_SPAN` 将 `span.to_string()` 嵌入 `InternalError` 消息

当 `Span` 有效时，错误输出包含指向用户代码的 `Source location:` 行。使用 `Span::unknown()` 时，不显示源码位置行。

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

在 IR 变换、pass 或 codegen 中编写使用 `INTERNAL_CHECK` 的新代码时：

1. 确定当前处理的 IR 节点（`op`、`stmt`、`expr` 等）
2. 将 `INTERNAL_CHECK(expr)` 替换为 `INTERNAL_CHECK_SPAN(expr, op->span_)`
3. 将 `INTERNAL_UNREACHABLE` 替换为 `INTERNAL_UNREACHABLE_SPAN(op->span_)`
4. 如果函数参数中已有 `Span`（例如 `Reconstruct*` 辅助函数），直接使用该参数

```cpp
// 之前：
INTERNAL_CHECK(op->body_) << "ForStmt has null body";

// 之后（当 span 可用时推荐）：
INTERNAL_CHECK_SPAN(op->body_, op->span_) << "ForStmt has null body";
```

## 相关文档

- [IR 概述 — 源码位置跟踪](ir/00-overview.md)
- [IR 验证器 — 诊断系统](passes/99-verifier.md)
- `include/pypto/core/error.h` — 异常类和 `Diagnostic`
- `include/pypto/core/logging.h` — 断言宏和 `FatalLogger`
- `include/pypto/ir/span.h` — `Span` 类
