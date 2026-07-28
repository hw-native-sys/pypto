# PyPTO 总览

本页是所有用户文档的唯一入口。它介绍 PyPTO 是什么、如何设计，以及各类任务应去哪里查阅。

## PyPTO 是什么？

PyPTO 是面向昇腾 NPU 的、基于 Python 的 kernel 编程框架。你用 `pypto.language`
模块（导入为 `pl`）编写计算 kernel，PyPTO 将其编译为优化后的设备代码：

```python
import pypto.language as pl
from pypto import ir
```

你描述*要*算什么 —— 在整张量级或 tile 级 —— 并用声明式 DSL 提示（流水线、切分、
scope）表达*如何*优化。编译器经由 pass 流水线下降你的程序，并为两种设备角色生成
代码：运行计算的 AI Core，以及负责调度的 AI CPU。

初次接触请从 **[快速入门](01-getting_started.md)** 开始，再回到本页看全局地图。

## 设计思路

- **Python DSL → PTO → 设备。** 你用 `pl` DSL 编写 kernel。PyPTO 构建不可变 IR，
  运行一条把 tensor 级 IR 渐进下降到 tile 级 IR 的 pass 流水线，并生成设备代码。
  无需手写汇编或调度。
- **两级抽象。** *Tensor 级*操作整张量（如 `pl.matmul`、`pl.add`）—— 计算的 tiling
  仍需你自己写，但编译器会自动完成数据搬运以及 matmul *核内*的 L0 tiling；*Tile 级*
  在需要时给你对 tile 的 load / compute / store 的显式控制。两者可在同一程序中共存；
  见[语言指南](02-language_guide.md)。
- **切图由你显式控制。** 计算如何切分为各个 InCore kernel —— 即每个 InCore 作用域的
  边界 —— 由你*显式*指定，而非编译器决定。你决定哪些计算进入每个核内函数；编译器在
  你划定的边界之内与之间做下降与调度。
- **声明式优化。** 优化以 DSL 构造（`pl.pipeline`、`pl.split`、`pl.spmd`、scope）
  表达，而非手写调度。你表达意图，编译器实现它。
  [性能手册](handbook/perf/00-workflow.md)覆盖每个旋钮。

## 执行模型

编译后的 PyPTO 程序在设备上跨三个协作角色运行：

| 角色 | 运行 | 由什么生成 |
| ---- | ---- | ---------- |
| **Host** | 启动程序、拥有 host 缓冲 | 你的 Python 驱动 / 运行时 |
| **AI CPU** | 任务调度与分发（*编排 orchestration*） | 编排函数 → 使用 PTO2 运行时 API 的 C++ |
| **AI Core** | 实际计算（*核内 InCore*） | InCore 函数 → `.pto` → AICore 二进制 |

```text
        ┌──────────────────────────────────────────────────────────┐
        │ Host —— 你的 Python 驱动                                   │
        │   启动程序 · 拥有 host 缓冲                                 │
        └───────────────────────────┬──────────────────────────────┘
                                     │ 启动
        ┌───────────────────────────▼──────────────────────────────┐
        │ AI CPU —— 编排 Orchestration  (任务调度与分发)            │
        │   编排函数 → C++ (PTO2 运行时 API)                        │
        │   构建任务图 · 解析依赖 · 分发 kernel                      │
        └───────────────────────────┬──────────────────────────────┘
                                     │ 分发 InCore kernel
        ┌───────────────────────────▼──────────────────────────────┐
        │ AI Core —— 核内 InCore  (计算)                            │
        │   InCore 函数 → .pto → AICore 二进制                      │
        │   ┌───────────────────┐   tpush   ┌───────────────────┐   │
        │   │ AIC (cube)        │ ────────▶ │ AIV (vector)      │   │
        │   │ 偏 matmul         │ ◀──────── │ 逐元素 / 归约      │   │
        │   │                   │   tpop    │                   │   │
        │   └───────────────────┘           └───────────────────┘   │
        └───────────────────────────┬──────────────────────────────┘
                                     │ load / store
        ┌───────────────────────────▼──────────────────────────────┐
        │ DDR —— 全局内存  (张量参数)                                │
        └──────────────────────────────────────────────────────────┘
```

- **编排 vs 核内。** 编排函数描述任务图与分发；InCore 函数描述逐核计算。编译器在
  下降时自动区分二者。
- **Cube / Vector 核。** 一个 AI Core 含 cube 单元（**AIC**，偏 matmul）和 vector
  单元（**AIV**，偏逐元素/归约）。部分 kernel 为*混合*，把工作切分到两者并跨核搬运数据。
- **片上并行 vs 多卡分布式。** `pl.spmd` / `pl.cluster` 在**单芯片**的多个核上分发
  多个 block —— 一个片上性能工具（[性能 › 切分与并行](handbook/perf/02-split-parallel.md)）。
  `pld.*` 家族（collective、远程 load/store）跨**多卡**
  （[分布式指南](distributed/00-guide.md)）。

## 内存层次

PyPTO 暴露分层内存模型：片外**全局内存（DDR）**承载张量参数，tile 被暂存到逐层
更小/更快的片上缓冲以供计算 —— 用于 vector 的统一 **Vec** 缓冲，以及 matmul 路径
**Mat**（L1）→ **Left**/**Right**（L0A/L0B）→ **Acc**（L0C）。tensor 级时编译器替你
插入数据搬运；tile 级时你显式搬运数据（`pl.load` / `pl.move` / `pl.store`）并用
`pl.Mem` 提示引导放置。见
[性能 › 内存放置](handbook/perf/04-memory-placement.md)与
[语言指南](02-language_guide.md#内存与数据搬运)。

## 编译流水线鸟瞰

```text
Python DSL  →  IR（不可变树）  →  Pass 流水线（tensor → tile 下降）  →  CodeGen
  @pl.program     @pl.function       inline · SSA · tiling · 内存 ·          ├─ InCore  → .pto → AICore
  @pl.function                       跨核 split · 调度                        └─ 编排    → C++（PTO2 运行时）→ AI CPU
```

整条流水线由一次调用触发：

```python
output_dir = ir.compile(MyProgram, backend_type=BackendType.Ascend910B)
```

`ir.compile(..., dump_passes=True)`（默认）在每个 pass 之后把 IR 快照写到
`output_dir/passes_dump/` —— 这是[精度定位](handbook/precision/00-workflow.md)
工作流的骨干。各个 pass 面向编译器开发者，文档在
[`dev/passes/`](../dev/passes/00-pass_manager.md)；作为用户你很少需要它们，但 pass
名会出现在 dump 与诊断里。

## 功能地图

一行一项能力索引；每项链接到对应章节。

| 能力 | 入口 |
| ---- | ---- |
| 编写第一个 kernel | [快速入门](01-getting_started.md) |
| 语言 / 类型系统 | [语言指南](02-language_guide.md) |
| 查询某个操作 | [操作参考](03-operation_reference.md) · [API 参考](api-reference/index.md) |
| 片上多 block 并行（`pl.spmd`/`pl.cluster`） | [性能 › 切分与并行](handbook/perf/02-split-parallel.md) |
| Mixed kernel —— cube + vector 切分（`pl.split`、`pl.split_aiv`） | [性能 › 切分与并行](handbook/perf/02-split-parallel.md) |
| 任务图与依赖控制 | [性能 › 依赖与分发](handbook/perf/03-dependency-dispatch.md) |
| 多卡分布式（`pld.*` collective） | [分布式指南](distributed/00-guide.md) |
| 调试结果错误 | [精度定位](handbook/precision/00-workflow.md) |
| 调试 kernel 太慢 | [性能调优](handbook/perf/00-workflow.md) |
| 诊断 / 日志 / 回放 | [DFX 功能](handbook/dfx/00-flag-matrix.md) |

## 文档地图

各文件夹的位置及阅读时机。

| 文件夹 / 文件 | 内含板块 | 何时阅读 |
| ------------- | -------- | -------- |
| `01-getting_started.md` | Hello World → 编译 → 上板 | 你的第一个 kernel |
| `02-language_guide.md` | 类型、控制流、内存 —— 概念教程 | 学习 DSL |
| `03-operation_reference.md` | `pl.*` 操作查找表（→ 将被 `api-reference/` 取代） | 查询某个 op |
| `handbook/precision/` | 精度定位工作流（torch 黄金对比、pass-IR 二分、选择性 dump） | 结果算错了 |
| `handbook/perf/` | 性能调优语法（循环/流水线、切分/并行、依赖/分发、内存放置、测量） | kernel 太慢 |
| `handbook/dfx/` | DFX 诊断（标志矩阵、HTML 渲染、回放、日志） | 检查调度/依赖或抓日志 |
| `distributed/` | 多卡 SPMD 编写（`DistributedTensor`、collective、远程 load/store、信号） | 写分布式 kernel |
| `api-reference/` | 自动生成的 `pl.*` 参考（`__all__` 全量符号） | 查 API 签名 |
| `troubleshooting.md` | 报错 → 可能原因 → 章节 | 遇到报错 |
| `glossary.md` | 术语定义（tile、scope、orchestration、AIV/AIC、TaskId…） | 名词不懂 |

## 下一步

- 初次接触 PyPTO？从 **[快速入门](01-getting_started.md)** 开始。
- 遇到问题？跳到 **[功能手册索引](handbook/00-index.md)** —— 它把症状映射到具体工具。
