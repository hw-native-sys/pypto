# PTO 项目生态

## 概述

PTO（Parallel Tensor/Tile Operation）项目是一个多仓库工具链，用于 AI 加速器编程。它覆盖了从 Python 级别的张量程序到硬件指令执行的完整栈。

本文档描述**每个仓库的职责**、**它们之间的连接方式**以及**各仓库的边界**。

## 仓库列表

所有仓库位于 [github.com/hw-native-sys](https://github.com/hw-native-sys) 组织下。

| 仓库 | 角色 | URL |
| ---- | ---- | --- |
| **pypto** | 编译器框架 | [hw-native-sys/pypto](https://github.com/hw-native-sys/pypto) |
| **pypto-lib** | 模型库与实际案例 | [hw-native-sys/pypto-lib](https://github.com/hw-native-sys/pypto-lib) |
| **PTOAS** | PTO 汇编器与优化器 | [hw-native-sys/PTOAS](https://github.com/hw-native-sys/PTOAS) |
| **pto-isa** | 指令集架构（ISA）定义 | [hw-native-sys/pto-isa](https://github.com/hw-native-sys/pto-isa) |
| **simpler** | 任务运行时 | [hw-native-sys/simpler](https://github.com/hw-native-sys/simpler) |

## 编译流水线

```text
                    ┌─────────────────────────────────────────────┐
                    │              pypto-lib                      │
                    │  实际模型与原语张量函数                         │
                    │  （以 pypto 作为编译框架）                     │
                    └──────────────────┬──────────────────────────┘
                                       │ imports & compiles via
                    ┌──────────────────▼──────────────────────────┐
                    │                pypto                        │
                    │  Python DSL → IR → Passes → CodeGen         │
                    │                                             │
                    │  产出:                                       │
                    │   • .pto 文件（InCore 内核 → AICore）         │
                    │   • Orchestration C++（任务调度 → AICPU）     │
                    └───┬─────────────────────────────────────┬───┘
          .pto files    │                                     │  orchestration C++
        (仅 InCore)     │                                     │  (运行在 AICPU)
                    ┌───▼────────────────────┐                │
                    │       PTOAS            │                │
                    │  汇编器与优化器          │                │
                    │                        │                │
                    │  .pto MLIR → C++       │                │
                    │  （使用 pto-isa 头文件）  │                │
                    └───┬────────────────────┘                │
                        │ kernel C++                          │
                        │  (includes pto-isa)                 │
                    ┌───▼────────────────────┐                │
                    │       pto-isa          │                │
                    │  ISA 定义：             │                │
                    │  tile 指令 C++ 头文件    │                │
                    └───┬────────────────────┘                │
                        │ compiled AICore binaries            │
                    ┌───▼─────────────────────────────────────▼───┐
                    │              simpler                        │
                    │  运行时：设备上的任务图执行                     │
                    │  Host ↔ AICPU ↔ AICore 协调                  │
                    └─────────────────────────────────────────────┘
```

**pypto 的两条代码生成路径：**

- **InCore 函数**（tile 级计算）→ `.pto` → PTOAS → pto-isa → AICore 二进制
- **Orchestration 函数**（任务调度）→ 使用 PTO2 runtime API 的 C++ → 编译到 AICPU

## 组件详情

### pypto — 编译器框架

核心编译器。将 Python 张量程序编译为设备可执行代码。

**输入：** 使用 `pypto.language` DSL 编写的 Python 程序（`@pl.program`、`@pl.function`）

**输出：**

- `.pto` 文件 — PTO-ISA MLIR 方言，每个 InCore 内核函数一个文件（运行在 AICore 上）
- Orchestration C++ — 使用 PTO2 runtime API 的任务调度代码（运行在 AICPU 上）

**内部流水线：**

```text
Python DSL → IR（不可变树）→ Pass Pipeline（20+ passes）→ CodeGen
```

- **IR 层**：多级表示 — Tensor ops、Tile ops 和 system ops 共存于同一 IR 中
- **Pass pipeline**：逐步将 tensor 级 IR 降低为 tile 级 IR（循环展开、SSA 转换、tiling、内存分配等）
- **CodeGen**：两个后端 — PTO codegen（InCore → `.pto` MLIR，运行在 AICore）和 Orchestration codegen（→ C++，运行在 AICPU）

**关键目录：**

| 路径 | 内容 |
| ---- | ---- |
| `include/pypto/ir/` | C++ IR 节点定义 |
| `src/ir/transforms/` | 编译 passes |
| `src/codegen/` | PTO 和 Orchestration 代码生成器 |
| `python/pypto/language/` | Python DSL 前端 |
| `python/pypto/ir/` | Pass manager、compile API |

### pypto-lib — 模型库与原语

基于 pypto 构建的实际模型和原语张量函数库。作用包括：

1. **模型库** — 端到端模型示例（如 DeepSeek、FFN、LLaMA），覆盖完整编译流水线
2. **原语张量函数** — 可复用的张量级构建块（elementwise、reduction、matmul），由编译器 tiling 并降低到 PTO-ISA

**依赖：** pypto（导入 `pypto.language`，通过 `pypto.ir.compile` 编译）

**与 pypto 的接口：** pypto-lib 程序就是标准的 pypto 程序 — 使用相同的 `@pl.program`/`@pl.function` DSL，通过相同的流水线编译。两者之间没有特殊 API；pypto-lib 是 pypto 框架的使用者。

### PTOAS — PTO 汇编器与优化器

基于 MLIR 的汇编器，消费 pypto codegen 生成的 `.pto` 文件，产出优化后的 C++ 内核代码。

**输入：** `.pto` 文件（PTO-ISA MLIR 方言）

**输出：** `#include` pto-isa 头文件的 C++ 源文件

**职责：**

- 解析 PTO-ISA MLIR 方言
- 应用 PTO 级优化 passes（同步插入、内存规划）
- 将 PTO MLIR 降低为调用 pto-isa tile 指令的 C++ 代码

**与 pypto 的接口：** `.pto` 文件是两者的契约。pypto 的 PTO codegen 使用 PTO 方言发射 MLIR（如 `pto.tload`、`pto.tmul`、`pto.alloc_tile` 等 ops），PTOAS 解析该方言。两个仓库必须在 PTO MLIR 方言定义上保持一致。

### pto-isa — 指令集架构

定义目标硬件的 tile 级指令集。提供声明硬件 tile 指令（load、store、compute、sync 等）的 C++ 头文件。

**被以下仓库消费：**

- **PTOAS** — PTOAS 生成的 C++ 代码调用 pto-isa 指令
- **simpler** — 首次构建时克隆 pto-isa 头文件用于运行时编译

**接口：** 定义指令 API 的 C++ 头文件库。下游消费者 `#include` pto-isa 头文件；硬件厂商提供支撑这些头文件的目标特定实现。

### simpler — 任务运行时

在 Ascend 硬件上执行编译后的程序。管理三程序执行模型：Host、AICPU kernel 和 AICore kernel。

**输入：**

- 编译后的 AICore 内核二进制文件（InCore 路径：pypto → PTOAS → pto-isa → 设备编译器）
- 编译后的 AICPU orchestration 二进制文件（Orchestration 路径：pypto → 使用 PTO2 runtime API 的 C++ → 设备编译器）

**职责：**

- 构建和执行任务依赖图
- 协调 Host ↔ AICPU ↔ AICore 执行
- 管理设备内存、同步和握手协议

**与 pypto 的接口：** pypto 生成的 orchestration C++ 代码使用 PTO2 runtime API（`rt_submit_task`、`make_tensor_external` 等），simpler 实现该 API。运行时 API 是 pypto orchestration codegen 和 simpler 之间的契约。

## 接口总结

每个仓库边界都有明确定义的接口：

```text
pypto-lib ──[ Python API: pypto.language / pypto.ir ]──► pypto
     pypto ──[ .pto files: PTO-ISA MLIR dialect     ]──► PTOAS
   pto-isa ──[ C++ #include: tile instruction hdrs  ]──► PTOAS
   pto-isa ──[ C++ #include: ISA headers            ]──► simpler
     pypto ──[ C++ API: PTO2 runtime API calls      ]──► simpler
```

| 边界 | 格式 | 提供者 | 消费者 |
| ---- | ---- | ------ | ------ |
| pypto-lib → pypto | Python imports | pypto-lib | pypto 编译器 |
| pypto → PTOAS | `.pto` MLIR 文件 | pypto PTO codegen | PTOAS 解析器 |
| pto-isa → PTOAS | C++ `#include` | pto-isa 头文件 | PTOAS codegen |
| pto-isa → simpler | C++ `#include` | pto-isa 头文件 | simpler 构建 |
| pypto → simpler | Orchestration C++ | pypto orchestration codegen | simpler 运行时 |

## 跨仓库开发

当变更涉及多个仓库时，识别受影响的接口：

| 变更 | 涉及仓库 | 受影响接口 |
| ---- | -------- | ---------- |
| 新增 tile 指令 | pto-isa + PTOAS + pypto | ISA 头文件、PTO MLIR 方言、pypto op/codegen |
| 新增张量原语 | pypto-lib + pypto | Python DSL（如需新 ops） |
| 新增运行时特性 | simpler + pypto | PTO2 runtime API、orchestration codegen |
| 新增 PTO MLIR op | PTOAS + pypto | PTO MLIR 方言、pypto PTO codegen |
| 新增模型示例 | 仅 pypto-lib | 无（现有 API 的消费者） |
