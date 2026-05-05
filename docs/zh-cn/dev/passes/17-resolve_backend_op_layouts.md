# ResolveBackendOpLayouts Pass

为后端有 layout 约束的 elementwise tile op 修复 layout：把 `[N, 1]` 的 col-major 向量输入 reshape 成 `[1, N]` 的 row-major 视图，必要时再把结果 reshape 回来。该 Pass 在 tile-PTO 阶段运行，位于 `ResolveTransposeLayout` 之后、收尾的 `NormalizeStmtStructure` 之前。

## 概述

经过 `FlattenTileNdTo2D` 和 `ResolveTransposeLayout` 之后，所有 tile op 都已是 2-D 形式且带有明确的 layout。多个 PTO elementwise op（在 `src/backend/common/pto_ops_common.cpp` 中注册）要求其 tile 操作数与结果均为 `row_major`。`[N, 1]` 列向量与 `[1, N]` 行向量在内存中的排布完全相同，因此本 Pass 在使用点局部修复约束违反：

1. 对每个 RHS 是 `Call` 的 `AssignStmt` / `EvalStmt`，调用 `Backend::GetTileLayoutSpec(op_name)` 查询约束。
2. 若没有注册约束，或者没有 `[N, 1]` col-major 输入违反 `row_major` 要求，则跳过。
3. 否则在 call 前插入 `tile.reshape(arg, [1, N])`，把 reshape 后的值代入 call；对 `AssignStmt`，只要结果类型不是 `[1, N]` row-major tile，就再追加 `tile.reshape(tmp, original_shape)` 把结果 reshape 回用户可见的形状。

本 Pass 是 **后端驱动** 的：被约束的 op 集合及其逐输入要求来自每个 op 的 `BackendOpRegistryEntry`（参见 `pto_ops_common.cpp` 中的 `set_input_layout` / `set_output_layout`）。Pass 自身保持后端无关——新增一个被约束的 op 只需登记它的 layout spec，无需修改本 Pass。

**前置要求**：

- 在 `FlattenTileNdTo2D` 之后运行（假定 tile op 已为 2-D）。
- 函数必须是 `InCore`；Orchestration / Group 函数被跳过。
- 必须通过 `BackendConfig::Set(...)` 配置后端，否则本 Pass 为 no-op。

**何时使用**：作为 `Default` tile-PTO pipeline 的一部分，在改变 layout 的若干 Pass（`FlattenTileNdTo2D`、`InferTileMemorySpace`、`ResolveTransposeLayout`）之后、`NormalizeStmtStructure` 之前运行。Pass manager 已经把它放在了正确的位置。

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
    若没有输入违反 spec.input_layouts（仅 [N,1] col-major 输入针对
       row_major 槽位时是可修复的）：跳过

    对每个 row_major 槽位上的输入 i：
      若输入非 tile、为 [1, N] row-major、或非 [N, 1]：跳过。
      reshape_var = 新临时变量
        （AssignStmt：名字基于结果变量；
          EvalStmt：名字基于字面量 "layout_fix"。
          两种形式都附加 "row_major" + "arg<i>" 限定符。）
      发射  reshape_var = tile.reshape(arg_i, [1, N])
      把 reshape_var 替换 call 中对应的实参

    repaired = OpRegistry.Create(call.op.name, new_args, call.kwargs)

    若语句是 AssignStmt 且 result_type 不是 [1, N] row-major：
      tmp = 新的 row-major 临时变量（结果名字加 "row_major" 限定符）
      发射  tmp = repaired
      发射  result_var = tile.reshape(tmp, original_result_shape)
    否则：
      发射  result_var = repaired   （或 EvalStmt 中以 repaired 替换）
```

非 tile 输入（标量、shape）以及对应槽位 `required_layout` 为 `nullopt` 的输入不会被改写。对 `row_major` 槽位上的 tile 输入，一旦 call 被判定为可修复，逐输入改写循环就会 reshape 任何 `[N, 1]` 操作数（无论 col-major 还是 row-major）；只有 `[1, N]` row-major 和非 `[N, 1]` 形状会被跳过。call 是否可修复由 `IsRepairableCall` 决定：仅当至少有一个输入违反 `row_major` 要求、且每个违反约束的输入都是 `[N, 1]` col-major 时才返回 true；若有违反约束的输入不属于该模式，整条语句保持不动。

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
| `tests/ut/ir/transforms/test_resolve_backend_op_layouts_pass.py` | 单元测试（`[N, 1]` 向量上的 binary、unary、tile×scalar） |

Layout 约束通过 `BackendOpRegistryEntry::set_input_layout` / `set_output_layout` 在 `src/backend/common/pto_ops_common.cpp` 中按 op 注册（如 `RequiresRowMajorLayout` 列表中的 row-major elementwise op、`tile.rsqrt`、`tile.cmps`、`tile.sort32`、`tile.mscatter` 等）。

Pass 源文件中的关键 helper：

- `IsRepairableCall` —— 当且仅当至少有一个 tile 输入违反 `row_major` 要求、且每个违反约束的输入都是 `[N, 1]` col-major tile 时返回 true。
- `BackendLayoutRepairMutator::VisitStmt_(const AssignStmtPtr&)` / `VisitStmt_(const EvalStmtPtr&)` —— 发射 call 前的 reshape、重建 call，并在结果是 col-major 列向量的情况下（仅 `AssignStmt`）发射 call 后的 reshape。
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
| 通过两次 `tile.reshape` 修复，而不是直接拒绝程序 | `[N, 1]` col-major 与 `[1, N]` row-major 的扁平内存布局相同；局部 reshape 既保留了用户可见的形状，又满足后端 ISA，不必让用户重写 kernel。 |
| 只修复 `[N, 1]` col-major 输入（不处理任意 layout 不匹配） | 这是当前 PTO row-major elementwise op 唯一观测到的不匹配模式。处理更一般情形需要真正的 layout 转换 Pass，本 Pass 不在该范围内；其他模式下 `IsRepairableCall` 返回 false，语句保持不变。 |
| 未配置后端时直接 bypass | 大量测试在未选择后端的情况下构造 IR；no-op fast path 让这些测试仍然通过，避免无意义的改写。 |
| 跳过非 `InCore` 函数 | Layout 约束作用于每个核内的 elementwise 执行；Orchestration、Group 函数仅承载对低层 kernel 的调用，没有需要修复的内容。 |
