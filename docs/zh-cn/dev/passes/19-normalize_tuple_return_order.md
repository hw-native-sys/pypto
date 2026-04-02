# NormalizeTupleReturnOrder Pass

对多值 `ReturnStmt` 的返回值（及对应的返回类型）进行重排，使每个返回槽与 **Out/InOut 形参的声明顺序**一致。

## 概述

InCore 内核可能返回张量句柄的元组，其**语法顺序**与 **Out/InOut** 形参顺序不一致（例如同时存在 `tile.store` 输出与循环携带的 `yield` 输出）。编排代码生成层曾逐一对返回值做参数追溯；该分析现由本 pass 完成，以便 codegen 可按顺序映射元组元素。

**主要职责**：

- 使用与原 codegen 辅助逻辑相同的规则扫描函数体：顶层 `tile.store` 的结果变量，以及 `ForStmt` 的 yield 链（`iter_arg` → `return_var`）
- 对具有多个 Out/InOut 形参且为多值返回的函数，置换 `ReturnStmt::value_`，使 `return[i]` 对应按声明顺序的第 *i* 个 Out/InOut 形参
- 同步置换 `Function::return_types_`（或单个 `TupleType` 内部的类型列表）

**运行时机**：在 `SplitVectorKernel` 之后、`InitMemRef` 之前执行（不依赖 MemRef 或已分配地址）。对返回元组的被调函数，编排 codegen 之前需要完成本 pass。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::NormalizeTupleReturnOrder()` | `passes.normalize_tuple_return_order()` | 函数级 |

**工厂函数**：

```cpp
Pass NormalizeTupleReturnOrder();
```

**Python 用法**：

```python
from pypto.pypto_core import passes

norm_pass = passes.normalize_tuple_return_order()
program = norm_pass(program)
```

## 前置条件

- 与 `SplitVectorKernel` 之后阶段一致：需具备 `VectorKernelSplit` 以及 `SSAForm`、`SplitIncoreOrch`、`IncoreTileOps`、`TileOps2D`、`TileMemoryInferred`（见 `pass_properties.h` 中的 `kNormalizeTupleReturnOrderProperties`）。

## 失败情况

若多值返回无法无歧义地映射到 Out/InOut 形参，本 pass 会以内部检查失败（无法安全规范化该 IR）。

## 相关

- Issue [#814](https://github.com/hw-native-sys/pypto/issues/814)
- [编排 codegen](../codegen/01-orchestration_codegen.md) — 对元组输出假定返回顺序已规范化
