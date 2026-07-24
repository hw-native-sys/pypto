# FuseCreateAssembleToSlice Pass

将 `tensor.create` 与 `tensor.assemble` 配对融合为单个 `tensor.slice` 视图，消除中间缓冲区。

## 概述

Orchestration 代码常见模式：用 `tensor.create` 分配一个小的暂存张量 (Tensor)，通过 InCore 调用填充内容，然后用 `tensor.assemble(target, source, offsets)` 写回到已有目标张量的某个子区域。当该暂存张量**恰好被一个** assemble 消费时，"先暂存再拷贝" 等价于直接写入目标的 `tensor.slice(target, shape, offsets)` 视图 —— 中间缓冲区是多余的。

该 Pass 识别这一模式并改写：`tensor.create` 变成对 assemble 目标的 `tensor.slice`，对应的 assemble 被删除。下游 codegen 通过 slice 视图直接写出，从而省掉临时分配和显式拷贝。

**前置条件**：

- 该 Pass 要求 IR 属性 (IRProperty) `SplitIncoreOrch` —— Orchestration 函数已从 InCore 代码中拆分出来（已运行 `OutlineHierarchyScopes` / `OutlineIncoreScopes` / `OutlineClusterScopes`）。
- 仅扫描 Orchestration 函数；InCore、AIC、AIV、Opaque 函数原样返回。

**使用时机**：在 `Default` 策略中作为第 27 个 Pass 运行，位于 `AllocateMemoryAddr` 之后（保证存活张量已经分配地址）、`DeriveCallDirections` 与尾部 `Simplify` 之前。这是 Call 方向 (Direction) 推断之前最后一个张量形状改写 Pass。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::FuseCreateAssembleToSlice()` | `passes.fuse_create_assemble_to_slice()` | Program 级 |

**Python 用法**：

```python
from pypto.pypto_core import passes

fuse_pass = passes.fuse_create_assemble_to_slice()
program_fused = fuse_pass(program)
```

## 算法

对每个 Orchestration 函数（其它原样返回）执行三个阶段：

1. **缓冲区根分析**：`BufferRootCollector` 遍历函数体，构建 `var → root` 映射。函数参数 (Parameter) 是自身的根；`tensor.create` 与 `tensor.slice` 的结果定义新根；变量别名 (Alias) 类的赋值继承其源的根；`tensor.assemble(target, source, offsets)` 的结果则继承 `target`（第 0 个参数）的根。该收集器还会沿 `ForStmt` / `WhileStmt` 的 iter args 传递根（关联 `iter_arg`、对应的 `return_var`、以及循环体中的使用），记录“返回 tuple 的函数调用”的输出 root，并在 `TupleGetItemExpr` 从这些调用结果解包时使用，还会沿 `Out` / `InOut` 方向的函数调用输出参数传递。最终结果是即使穿过循环、特定的 tuple 调用结果解包和跨函数别名，每个 `var` 也只有一个缓冲身份。

2. **模式检测**：`AssemblePatternCollector` 扫描可融合的配对：
   - 每个 `tensor.create`，若其根解析到自身（即该 create 是该缓冲的源头），记录在 `create_vars` 中。
   - 每个 `tensor.assemble(target, source, offsets)`，若其 `source` 解析回某个已记录的 create 根，则记录为候选融合：`FuseInfo{target_expr, offset_tuple}`。
   - 若同一个 create 根出现在 ≥ 2 个 assemble 中，则从 `fusible_roots` 移到 `non_fusible_roots`，不参与改写。

3. **改写**：`FuseCreateAssembleMutator` 完成 IR 变换：
   - `tensor.create(shape, dtype)` → `tensor.slice(target, shape_tuple, offset_tuple)`。当 assemble 目标的 rank 大于 create 出来 tile 的 rank（如把 2D tile 装配到 3D 张量的 `[b, p, q]` 处），slice 的 shape 元组前面补齐若干 `1` 维度，使 shape 与 offset 的 rank 对齐。
   - 匹配到的 `tensor.assemble` `AssignStmt` 替换为空的 `SeqStmts`；其原本绑定的变量 (Var) 被重映射到 slice 的 target，下游使用仍然看到同一身份。
   - 当在 `ForStmt` / `WhileStmt` body 中删除 assemble 后导致某个 yielded iter arg 变成 pass-through（`yield(iter_arg)` 而非 `yield(new_value)`），`StripPassThroughIterArgs` / `StripPassThroughWhileIterArgs` 会把该 iter arg 从循环中移除，丢掉对应的 `return_var`，并将 iter arg 的 `init_value` 替换进 body。携带真实循环状态的其它 iter args 保持不变。

| 源模式 | 行为 |
| ------ | ---- |
| `create` 恰被 1 个 assemble 消费写入某 target | 把 `create` 改写为 `slice`；删除 `assemble`；重映射别名变量 |
| `create` 被 ≥ 2 个 assemble 消费 | 标记为不可融合，IR 不变 |
| `create` 后没有 `assemble` 消费 | IR 不变 |
| `assemble` 的源是 `tensor.slice`（而非 `create`） | IR 不变 |
| 函数类型为 InCore / AIC / AIV / Opaque | 原样返回 |

## 示例

### 基础融合

**改写前**：

```python
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def fill_row(
        self,
        x: pl.Tensor[[4, 8], pl.FP32],
        r: pl.Scalar[pl.INDEX],
        out: pl.Out[pl.Tensor[[1, 8], pl.FP32]],
    ) -> pl.Tensor[[1, 8], pl.FP32]:
        row_tile: pl.Tile[[1, 8], pl.FP32] = pl.load(x, [r, 0], [1, 8])
        out_1: pl.Tensor[[1, 8], pl.FP32] = pl.store(row_tile, [0, 0], out)
        return out_1

    @pl.function(type=pl.FunctionType.Orchestration)
    def orch(
        self,
        x: pl.Tensor[[4, 8], pl.FP32],
        out: pl.Out[pl.Tensor[[4, 8], pl.FP32]],
    ) -> pl.Tensor[[4, 8], pl.FP32]:
        for r in pl.range(4):
            row: pl.Tensor[[1, 8], pl.FP32] = pl.create_tensor([1, 8], dtype=pl.FP32)
            row = self.fill_row(x, r, row)
            out = pl.assemble(out, row, [r, 0])
        return out
```

**改写后**：

```python
@pl.program
class After:
    @pl.function(type=pl.FunctionType.Orchestration)
    def orch(
        self,
        x: pl.Tensor[[4, 8], pl.FP32],
        out: pl.Out[pl.Tensor[[4, 8], pl.FP32]],
    ) -> pl.Tensor[[4, 8], pl.FP32]:
        for r in pl.range(4):
            row: pl.Tensor[[1, 8], pl.FP32] = pl.slice(out, [1, 8], [r, 0])
            row = self.fill_row(x, r, row)
        return out
```

`pl.create_tensor` 被替换为 `out` 的 `pl.slice` 视图；`pl.assemble` 被删除；尾部 `out = pl.assemble(...)` 对应的 iter arg 因变为 pass-through 而被剥离。

### 高 rank target 的形状补齐（2D tile 装配到 3D target）

当 assemble 目标比 create 多出一些 leading 维度时，slice 的 shape 会用 `1` 补齐前导维度，使其与 offset 对齐：

**改写前**：

```python
@pl.function(type=pl.FunctionType.Orchestration)
def orch(
    self,
    x: pl.Tensor[[4, 8], pl.FP32],
    out: pl.Out[pl.Tensor[[2, 4, 8], pl.FP32]],
) -> pl.Tensor[[2, 4, 8], pl.FP32]:
    for b in pl.range(2):
        for c in pl.range(2):
            col = c * 4
            chunk: pl.Tensor[[2, 4], pl.FP32] = pl.create_tensor([2, 4], dtype=pl.FP32)
            chunk = self.compute(x, chunk)
            out = pl.assemble(out, chunk, [b, 0, col])
    return out
```

**改写后**：

```python
@pl.function(type=pl.FunctionType.Orchestration)
def orch(
    self,
    x: pl.Tensor[[4, 8], pl.FP32],
    out: pl.Out[pl.Tensor[[2, 4, 8], pl.FP32]],
) -> pl.Tensor[[2, 4, 8], pl.FP32]:
    for b in pl.range(2):
        for c in pl.range(2):
            col = c * 4
            chunk: pl.Tensor[[1, 2, 4], pl.FP32] = pl.slice(out, [1, 2, 4], [b, 0, col])
            chunk = self.compute(x, chunk)
    return out
```

## 实现

**头文件**：`include/pypto/ir/transforms/passes.h`

**实现文件**：`src/ir/transforms/fuse_create_assemble_to_slice_pass.cpp`

- `BufferRootCollector`（IRVisitor）—— 沿赋值、循环 iter args、tuple、调用输出别名等路径分析缓冲身份。
- `AssemblePatternCollector`（IRVisitor）—— 检测 "恰好被一次 assemble 消费" 的 create；多重 assemble 的根被排除。
- `FuseCreateAssembleMutator`（IRMutator）—— 把 `create` 改写为 `slice`，删除匹配到的 `assemble`，并剥离 `for` / `while` 中变成 pass-through 的 iter arg。

**Python 绑定**：`python/bindings/modules/passes.cpp`

**测试**：`tests/ut/ir/transforms/test_fuse_create_assemble_to_slice.py`

- `test_basic_create_assemble_fused_to_slice` —— 基础融合 + iter arg 剥离
- `test_duplicate_assemble_not_fused` —— 多重 assemble 的反例
- `test_slice_source_not_fused` —— assemble 源是 slice 时不融合
- `test_multi_iter_arg_partial_fuse` —— 仅剥离参与 assemble 的 iter arg；其它承载真实状态的 iter args 保留
- `test_3d_target_2d_tile_offset_padded` —— target rank 大于 tile rank 时的前导 `1` 补齐
- `test_no_orchestration_function_noop` —— 没有 Orchestration 函数时为 no-op

## Pass 属性

| 属性 | 值 |
| ---- | -- |
| Required | `SplitIncoreOrch` |
| Produced | — |
| Invalidated | — |

该 Pass 不破坏任何输入属性：仅改写 Orchestration 函数体内语句 (Statement)，不会引入 `tensor.slice` 之外的新 IR 形式；并且具有幂等性 —— 在已融合过的 IR 上重复运行不再发现 `create + 单一 assemble` 模式，因此为 no-op。

## 作用范围

| 函数类型 | 行为 |
| -------- | ---- |
| Orchestration | 扫描；可融合的 `create + 单一 assemble` 配对改写为 `slice` |
| InCore（InCore、AIC、AIV） | 原样返回 |
| Opaque | 原样返回 |

当没有 Orchestration 函数包含可融合的 `create + assemble` 配对时，该 Pass 为 no-op。
