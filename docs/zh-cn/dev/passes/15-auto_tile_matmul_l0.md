# AutoTileMatmulL0 Pass

针对 Mat-resident 的 `tile.matmul` / `tile.matmul_acc` 进行 L0 切分：从当前 backend 的 L0 容量中挑选 L0 tile 形状 `(m, n, k)`，并把这次 matmul 调用改写成一个 2 阶段流水化的 K-loop，每个迭代用 `tile.extract` 从 Mat 抽取 Left/Right 操作数。

## 概览

由 `ConvertTensorToTileOps` + [`FlattenTileNdTo2D`](14-flatten_tile_nd_to_2d.md) 生成的 Mat-resident matmul 通常带有完整的 `(M, N, K)` 操作数形状——几乎一定大于 cube unit 的 L0a/L0b/L0c 容量。本 pass 选取一个能放进 L0 的 `(m, n, k)`，并把该 matmul 改写成一个 K-loop：循环体内用 `tile.extract` 把 `[m, k]` 与 `[k, n]` 的切片送入 `Left` / `Right`，并把累加器写入 `Acc`-resident 的 iter-arg。该循环带有 `ForKind::Pipeline` 与 `pipeline_stages=2`，使下游 [`LowerPipelineLoops`](23-lower_pipeline_loops.md) 可对每次迭代的操作数 `tile.extract` 生成 2 级 ping-pong。

**Pipeline 位置**：紧跟在 [`FlattenTileNdTo2D`](14-flatten_tile_nd_to_2d.md) 之后，先于 [`InferTileMemorySpace`](16-infer_tile_memory_space.md)。此时 tile op 已是 2D，但 memory space 尚未推断。

**前置属性 (Required)**：`SSAForm`、`SplitIncoreOrch`、`IncoreTileOps`、`TileOps2D`、`NormalizedStmtStructure`。

**产出属性 (Produced)**：与前置属性相同（属性保持不变的改写）。

**失效属性 (Invalidated)**：无。

**何时使用**：一律在默认 tile 阶段流水线中运行。如果不存在超过 backend L0 容量的 Mat-resident matmul，本 pass 是 no-op。

## API

| C++ | Python | 层级 |
| --- | ------ | ---- |
| `pass::AutoTileMatmulL0()` | `passes.auto_tile_matmul_l0()` | Program 级 |

```python
from pypto.pypto_core import passes

l0_tile_pass = passes.auto_tile_matmul_l0()
program_tiled = l0_tile_pass(program)
```

## 算法

对每个 InCore 函数中的 `tile.matmul` 或 `tile.matmul_acc`：

1. **过滤** —— 操作数布局：`tile.matmul` 为 `(lhs, rhs)`，`tile.matmul_acc` 为 `(acc, lhs, rhs)`。`lhs` 与 `rhs` 必须是 `Var` / `IterArg`（通过 `AsVarLike` 识别）且为 `TileType`，形状必须是静态 2D 且 `memory_space == Mat`。其它情形（Vec/Acc 操作数、动态形状）直接静默跳过。`tile.matmul_bias` 暂不改写——只在最后一次迭代后做 bias-add 需要额外重写，目前尚未实现。
2. **选择 L0 tile 形状** —— 调用 `utils::ChooseL0Tile(cfg)`。`cfg` 来自当前 `BackendHandler` 的 `GetL0{a,b,c}CapacityBytes()` 与 `GetL0FractalAlignment()`，再加上从调用结果类型读出的元素字节宽 `bytes_a/b/c`，使 chooser 看到真实的累加器占用。`c_read = is_matmul_acc`：因为 `tile.matmul_acc` 把调用方的累加器穿过 K-loop iter-arg（chooser 流量模型中 γ_C = 2）。Chooser 返回 `(m, n, k)` —— 闭式 O(1) 算法，依据 L0 切分设计文档（连续最优 + 邻域对齐候选，按 `(traffic, padded_compute, k_blocks, area, k)` 打分）。
3. **若已是 L0 大小则跳过** —— `(m, n, k) == (M, N, K)`。
4. **不支持的形态以 `PerfHint` 跳过**：
   - 子字节 dtype（cube path 不支持）—— `PH-AT-003`。
   - `ChooseL0Tile` 拒绝该配置 —— `PH-AT-005`。
   - 需要 M/N 切分（`m != M || n != N`）—— `PH-AT-006`。M/N 切分需要 Mat-resident 的输出 scratch 与每次迭代的 assemble，目前尚未实现。
   - `K % k != 0` —— `PH-AT-007`。K 边界处理（最后一次迭代切 `valid_shape`）目前尚未实现。
5. **构造 K-loop**：
   - `tile.matmul` —— iter-arg 初值为 Acc-resident 的 `tile.create([m, n], dtype, target_memory=Acc)` 占位；循环体用 `IfStmt` 在 `ko == 0` 时走 `tile.matmul`（产生新的 Acc），其它迭代走 `tile.matmul_acc`（向 iter-arg 上累加）。`IfStmt` 物化一个 phi 形式的 `return_var`，由外层 yield 写回 iter-arg。
   - `tile.matmul_acc` —— iter-arg 初值就是调用方传入的累加器（其类型已经与每次迭代的 `tile.matmul_acc` 输出一致）；每次迭代统一是 `tile.matmul_acc`，无需 if-else。
   - 每次迭代的操作数抽取使用 `tile.extract(src, idx_row, idx_col, [shape], target_memory=Left|Right)` —— 这是旧版 `tile.slice`（Mat-resident 中间 tile）+ `tile.mov`（Mat→Left/Right）的 SSA 化合并。这样既消除了 Mat-resident 中间 slice tile，也使得 lower 后是 `pto.textract` 而不是 `pto.subview`，从而绕开后者的 `valid_row` codegen 不一致问题。
   - K-loop 标记为 `ForKind::Pipeline`，`pipeline_stages=2`。
6. **改写所在 `SeqStmts`** —— 把原 matmul 的 `Var` 用法改成新的 `ForStmt::return_var`。替换作用域只限当前 `SeqStmts`，不会泄漏到兄弟区域。

本 pass 是 `ProgramPass`，对每个函数走 `IRMutator`；当函数内没有触发任何改写时，返回原函数（不会发生 `MutableCopy` 开销）。

## 示例

### 普通 `tile.matmul`

**Before**（Mat-resident `tile.matmul`，`M = N = 128`，`K = 256`）：

```python
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def main(self, ...):
        ...
        c: pl.Tile[[128, 128], pl.FP32] = pl.tile.matmul(a_mat, b_mat)
        ...
```

**After**（chooser 选定 `m = 128, n = 128, k = 64`）：

```python
@pl.program
class After:
    @pl.function(type=pl.FunctionType.InCore)
    def main(self, ...):
        ...
        c_l0_init = pl.tile.create([128, 128], pl.FP32, target_memory=Acc)
        for ko, (c_iter,) in pl.pipeline(0, 256, 64, init_values=(c_l0_init,), stage=2):
            sa = pl.tile.extract(a_mat, 0, ko, [128, 64], target_memory=Left)
            sb = pl.tile.extract(b_mat, ko, 0, [64, 128], target_memory=Right)
            if ko == 0:
                c_first = pl.tile.matmul(sa, sb)
                c_phi = pl.yield_(c_first)
            else:
                c_acc = pl.tile.matmul_acc(c_iter, sa, sb)
                c_phi = pl.yield_(c_acc)
            c = pl.yield_(c_phi)
        # c（即 yield-LHS）持有累加得到的 Acc 类型结果。
        ...
```

### `tile.matmul_acc`

调用方的累加器直接穿过 iter-arg，无需 if-else：

```python
for ko, (c_iter,) in pl.pipeline(0, K, k, init_values=(acc_init,), stage=2):
    sa = pl.tile.extract(a_mat, 0, ko, [m, k], target_memory=Left)
    sb = pl.tile.extract(b_mat, ko, 0, [k, n], target_memory=Right)
    c_new = pl.tile.matmul_acc(c_iter, sa, sb)
    c = pl.yield_(c_new)
# c（即 yield-LHS）持有累加得到的 Acc 类型结果。
```

## Backend 约束

L0 容量与 fractal 对齐都来自当前 `BackendHandler`。Pass 优先从 `PassContext::Current()->GetBackendHandler()` 读取，若无活动 context 则回退到 `pypto::backend::GetBackend()->GetHandler()`（例如未包 `PassContext` 直接调用的测试场景）。

| Handler 调用 | 用途 |
| ------------ | ---- |
| `GetL0aCapacityBytes()` | chooser 中 L0a (Left) 容量 |
| `GetL0bCapacityBytes()` | chooser 中 L0b (Right) 容量 |
| `GetL0cCapacityBytes()` | chooser 中 L0c (Acc) 容量 |
| `GetL0FractalAlignment()` | chooser 中 M/N/K 对齐粒度 |
| `GetMinL0TileDim()` | 单轴最小 tile 尺寸 |

因此新增 backend 时，只需要提供这些 handler 接口；本 pass 自身与具体 backend 无关。

## 实现

**头文件**：`include/pypto/ir/transforms/passes.h`

**Properties 声明**：`include/pypto/ir/transforms/pass_properties.h`（`kAutoTileMatmulL0Properties`）

**实现**：`src/ir/transforms/auto_tile_matmul_l0_pass.cpp`

**Chooser 工具**：`src/ir/transforms/utils/l0_tile_chooser.cpp` —— 闭式 L0 形状选取，未来其它 tiler 也可复用。

**Python 绑定**：`python/bindings/modules/passes.cpp`

**测试**：`tests/ut/ir/transforms/test_auto_tile_matmul_l0.py`、`tests/ut/ir/transforms/test_l0_tile_chooser.py`

## Pass 属性

| 属性 | 值 |
| ---- | -- |
| Required | SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D, NormalizedStmtStructure |
| Produced | SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D, NormalizedStmtStructure |
| Invalidated | — |

## 适用范围

| Op | 处理方式 |
| -- | -------- |
| 操作数为 Mat-resident 静态 2D 的 `tile.matmul` | 改写为 2 阶段流水化 K-loop |
| 操作数为 Mat-resident 静态 2D 的 `tile.matmul_acc` | 改写为 2 阶段流水化 K-loop（循环体统一为 `matmul_acc`） |
| `tile.matmul_bias` | 跳过（待支持——「最后一次迭代后再 bias-add」的改写尚未实现） |
| 已经是 L0 大小（`(m, n, k) == (M, N, K)`）的 matmul | 不动 |
| 子字节 dtype / `m != M` / `n != N` / `K % k != 0` | 以 `PerfHint` 跳过 |
| 非 InCore 函数（Orchestration、Opaque） | 不动 |

## Diagnostics

当 pass 决定不改写时，会发出 `PerfHint`（而不是失败）；原 matmul 保持不变并继续走后续流水线。`PerfHint` 编码：

| 编码 | 含义 |
| ---- | ---- |
| `PH-AT-003` | 操作数或累加器使用了子字节 dtype |
| `PH-AT-005` | `ChooseL0Tile` 拒绝了该配置 |
| `PH-AT-006` | Chooser 选了需要 M/N 切分的形状（暂不支持） |
| `PH-AT-007` | `K % k != 0`（K 边界处理暂不支持） |
| `PH-AT-008` | `ChooseL0Tile` 返回了 fallback 配置并附带 perf hint |

## 相关 Pass

- [`FlattenTileNdTo2D`](14-flatten_tile_nd_to_2d.md) —— 上游 pass；产生本 pass 所需的静态 2D Mat-resident tile 形状
- [`InferTileMemorySpace`](16-infer_tile_memory_space.md) —— 下游 pass；负责桥接本 pass 故意保留下来的 Vec/Acc 累加器
- [`LowerPipelineLoops`](23-lower_pipeline_loops.md) —— 消费本 pass 产生的 `ForKind::Pipeline` + `pipeline_stages=2`
