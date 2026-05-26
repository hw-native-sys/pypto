# ResolveBackendOpLayouts Pass

为后端有 layout 约束的 elementwise tile op 修复 layout：把 `[N, 1]` 的 col-major 向量 reshape 成 `[1, N]` 的 row-major 视图，并通过 `tile.move(..., blayout=row_major)` 修复一般非 row-major tile。该 Pass 在 tile-PTO 阶段运行，位于 `LowerTransposeLoadParamLayout` 之后、收尾的 `NormalizeStmtStructure` 之前。

## 概述

经过 `FlattenTileNdTo2D` 和 `LowerTransposeLoadParamLayout` 之后，所有 tile op 都已是 2-D 形式且带有明确的 layout。多个 PTO elementwise op（在 `src/backend/common/pto_ops_common.cpp` 中注册）要求其 tile 操作数与结果均为 `row_major`。本 Pass 在使用点局部修复这些约束违反：

1. 对每个 RHS 是 `Call` 的 `AssignStmt` / `EvalStmt`，调用 `Backend::GetTileLayoutSpec(op_name)` 查询约束。
2. 若没有注册约束，或者所有受约束的 tile 输入与输出都已经是 `row_major`，则跳过。
3. 对 `[N, 1]` col-major 输入，在 call 前插入 `tile.reshape(arg, [1, N])`。这是仅改变元数据的视图修复，因为 `[N, 1]` col-major 与 `[1, N]` row-major 的扁平内存顺序相同。
4. 对其他非 row-major tile 输入，在 call 前插入 `tile.move(arg, target_memory=<same>, blayout=row_major, slayout=none_box)`。
5. 对原始结果类型不是 row-major 的 `AssignStmt`，先把修复后的 call 赋给一个 row-major 临时变量，再用 `tile.reshape`（列向量）或 `tile.move`（一般矩阵 tile）恢复原始结果 layout。

本 Pass 是 **后端驱动** 的：被约束的 op 集合及其逐输入要求来自每个 op 的 `BackendOpRegistryEntry`（参见 `pto_ops_common.cpp` 中的 `set_input_layout` / `set_output_layout`）。Pass 自身保持后端无关——新增一个被约束的 op 只需登记它的 layout spec，无需修改本 Pass。

**前置要求**：

- 在 `FlattenTileNdTo2D` 之后运行（假定 tile op 已为 2-D）。
- 函数必须是 `InCore`；Orchestration / Group 函数被跳过。
- 必须通过 `BackendConfig::Set(...)` 配置后端，否则本 Pass 为 no-op。

**何时使用**：作为 `Default` tile-PTO pipeline 的一部分，在改变 layout 的若干 Pass（`FlattenTileNdTo2D`、`InferTileMemorySpace`、`LowerTransposeLoadParamLayout`）之后、`NormalizeStmtStructure` 之前运行。Pass manager 已经把它放在了正确的位置。

## API

| C++ | Python | 层级 |
| --- | ------ | ---- |
| `pass::ResolveBackendOpLayouts()` | `passes.resolve_backend_op_layouts()` | Function 级 |

**Python 用法**：

```python
from pypto.pypto_core import passes

repair = passes.resolve_backend_op_layouts()
program = repair(program)
```

## 算法

```text
对程序中的每个函数：
  若函数不是 InCore：跳过。
  若未配置后端：跳过。

  使用 IRMutator 遍历 body。对每个 RHS 是 Call 的
  AssignStmt / EvalStmt：
    spec = backend.GetTileLayoutSpec(call.op.name)
    若 spec 为空：跳过
    若没有受约束输入/输出需要 row_major 修复：跳过

    对每个 row_major 槽位上的输入 i：
      若输入非 tile 或已经是 row-major：跳过。
      reshape_var = 新临时变量
        （AssignStmt：名字基于结果变量；
          EvalStmt：名字基于字面量 "layout_fix"。
          两种形式都附加 "row_major" + "arg<i>" 限定符。）
      若输入是 [N, 1] col-major：
        发射  reshape_var = tile.reshape(arg_i, [1, N])
      否则：
        发射  reshape_var = tile.move(arg_i, target_memory=<same>,
                                      blayout=row_major, slayout=none_box)
      把 reshape_var 替换 call 中对应的实参

    repaired = OpRegistry.Create(call.op.name, new_args, call.kwargs)

    若语句是 AssignStmt 且 result_type 受约束但不是 row-major：
      tmp = 新的 row-major 临时变量（结果名字加 "row_major" 限定符）
      发射  tmp = repaired
      若原始结果是列向量：
        发射  result_var = tile.reshape(tmp, original_result_shape)
      否则：
        发射  result_var = tile.move(tmp, target_memory=<same>,
                                     blayout=<original>, slayout=<original>)
    否则：
      发射  result_var = repaired   （或 EvalStmt 中以 repaired 替换）
```

非 tile 输入（标量、shape）以及对应槽位 `required_layout` 为 `nullopt` 的输入不会被改写。修复局限在受约束 op 附近：下游代码仍然看到原始变量类型和 layout，因为 `AssignStmt` 结果会在 row-major op 之后恢复。

## 示例

（改编自 `tests/ut/ir/transforms/test_resolve_backend_op_layouts_pass.py::test_rewrites_column_vector_add_through_row_major_reshape`，启用 Ascend910B 后端）

**Before**：

```python
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def repro(
        self,
        data: pl.Tensor[[16, 256], pl.FP32],
        out: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
    ) -> pl.Tensor[[16, 1], pl.FP32]:
        acc_0: pl.Tile[[16, 1], pl.FP32, pl.MemorySpace.Vec] = pl.tile.create(
            [16, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        acc_1: pl.Tile[[16, 1], pl.FP32, pl.MemorySpace.Vec] = pl.tile.muls(acc_0, 0.0)
        chunk: pl.Tile[[16, 256], pl.FP32, pl.MemorySpace.Vec] = pl.load(data, [0, 0], [16, 256])
        tmp: pl.Tile[[16, 256], pl.FP32, pl.MemorySpace.Vec] = pl.tile.create(
            [16, 256], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        partial: pl.Tile[[16, 1], pl.FP32, pl.MemorySpace.Vec] = pl.tile.row_sum(chunk, tmp)
        updated: pl.Tile[[16, 1], pl.FP32, pl.MemorySpace.Vec] = pl.tile.add(acc_1, partial)
        stored: pl.Tensor[[16, 1], pl.FP32] = pl.store(updated, [0, 0], out)
        return stored
```

**After**：

```python
@pl.program
class After:
    @pl.function(type=pl.FunctionType.InCore)
    def repro(
        self,
        data: pl.Tensor[[16, 256], pl.FP32],
        out: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
    ) -> pl.Tensor[[16, 1], pl.FP32]:
        acc_0: pl.Tile[[16, 1], pl.FP32, pl.MemorySpace.Vec] = pl.tile.create(
            [16, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        acc_0_rm: pl.Tile[[1, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tile.reshape(acc_0, [1, 16])
        acc_1_rm: pl.Tile[[1, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tile.muls(acc_0_rm, 0.0)
        acc_1: pl.Tile[[16, 1], pl.FP32, pl.MemorySpace.Vec] = pl.tile.reshape(acc_1_rm, [16, 1])
        chunk: pl.Tile[[16, 256], pl.FP32, pl.MemorySpace.Vec] = pl.load(data, [0, 0], [16, 256])
        tmp: pl.Tile[[16, 256], pl.FP32, pl.MemorySpace.Vec] = pl.tile.create(
            [16, 256], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        partial: pl.Tile[[16, 1], pl.FP32, pl.MemorySpace.Vec] = pl.tile.row_sum(chunk, tmp)
        acc_1_rm2: pl.Tile[[1, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tile.reshape(acc_1, [1, 16])
        partial_rm: pl.Tile[[1, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tile.reshape(partial, [1, 16])
        updated_rm: pl.Tile[[1, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tile.add(acc_1_rm2, partial_rm)
        updated: pl.Tile[[16, 1], pl.FP32, pl.MemorySpace.Vec] = pl.tile.reshape(updated_rm, [16, 1])
        stored: pl.Tensor[[16, 1], pl.FP32] = pl.store(updated, [0, 0], out)
        return stored
```

`tile.muls`、`tile.add` 等 elementwise PTO op 要求输入和输出均为 `row_major`。每一个被约束的 call 都会被包裹：`[16, 1]` 操作数在 call 之前 reshape 为 `[1, 16]`，call 在 row-major 形式下执行，结果再 reshape 回 `[16, 1]`，使下游代码（`tile.store`、返回类型）继续看到用户可见的形状。`tile.row_sum` 不带约束，其输入和输出保持原样。

## 实现

| 文件 | 角色 |
| ---- | ---- |
| `include/pypto/ir/transforms/passes.h`（`ResolveBackendOpLayouts`） | 公共 C++ 工厂 |
| `src/ir/transforms/resolve_backend_op_layouts_pass.cpp` | Mutator 与 Pass 主体 |
| `include/pypto/ir/transforms/pass_properties.h`（`kResolveBackendOpLayoutsProperties`） | Pass 属性 |
| `python/bindings/modules/passes.cpp`（`resolve_backend_op_layouts`） | Python 绑定 |
| `python/pypto/pypto_core/passes.pyi`（`resolve_backend_op_layouts`） | 类型存根 |
| `tests/ut/ir/transforms/test_resolve_backend_op_layouts_pass.py` | 单元测试（`[N, 1]` 向量上的 binary、unary、tile×scalar，以及通过 `tile.move` 进行矩阵 layout 修复） |

Layout 约束通过 `BackendOpRegistryEntry::set_input_layout` / `set_output_layout` 在 `src/backend/common/pto_ops_common.cpp` 中按 op 注册（如 `RequiresRowMajorLayout` 列表中的 row-major elementwise op、`tile.rsqrt`、`tile.cmps`、`tile.sort32`、`tile.mscatter` 等）。

Pass 源文件中的关键 helper：

- `NeedsInputRepair` / `NeedsOutputRepair` —— 检测受约束的 `row_major` 槽位中当前 tile layout 不是 row-major 的情况。
- `CreateLayoutMoveCall` —— 发射用于一般矩阵 layout 修复和结果恢复的 `tile.move`。
- `BackendLayoutRepairMutator::VisitStmt_(const AssignStmtPtr&)` / `VisitStmt_(const EvalStmtPtr&)` —— 发射 call 前的 reshape/move，重建 call，并在 `AssignStmt` 需要时发射 call 后的 reshape/move。
- `RewriteFunction` —— 跳过非 `InCore` 函数和未配置后端的情况，再调用 mutator。

## Pass 属性

| 属性 | 取值 |
| ---- | ---- |
| Required | SSAForm、IncoreTileOps、SplitIncoreOrch、TileOps2D |
| Produced | SSAForm、IncoreTileOps、SplitIncoreOrch、TileOps2D |
| Invalidated | NormalizedStmtStructure |

`NormalizedStmtStructure` 被 invalidate 是因为每次修复都会把原本一条语句的 op 包裹成多条 `tile.reshape` 赋值，破坏了规范化的语句结构。默认 pipeline 在本 Pass 之后立刻重新跑 `NormalizeStmtStructure` 以恢复该不变量。

## 设计取舍

| 决策 | 理由 |
| ---- | ---- |
| 通过 `Backend::GetTileLayoutSpec` 获取 layout 要求，而不是在 Pass 中硬编码 op 列表 | 各后端在自己的 codegen 注册旁边声明约束。Pass 保持后端无关（参见 `pass-context-config.md`）；新增被约束的 op 只需要一次 `set_input_layout` 调用，不必改 Pass。 |
| 对 `[N, 1]` 向量优先使用 `tile.reshape` | `[N, 1]` col-major 与 `[1, N]` row-major 的扁平内存布局相同；reshape 成本更低，也保留了既有的向量修复行为。 |
| 对一般矩阵 layout 修复使用 `tile.move` | `[16, 256]` 这类完整 tile 无法通过只改变形状的向量 reshape 修复。相同 memory space 内的 `tile.move` 会在 `tile.exp` 等 row-major PTO op 之前物化 row-major 视图，并在需要时恢复原 layout。 |
| 未配置后端时直接 bypass | 大量测试在未选择后端的情况下构造 IR；no-op fast path 让这些测试仍然通过，避免无意义的改写。 |
| 跳过非 `InCore` 函数 | Layout 约束作用于每个核内的 elementwise 执行；Orchestration、Group 函数仅承载对低层 kernel 的调用，没有需要修复的内容。 |
