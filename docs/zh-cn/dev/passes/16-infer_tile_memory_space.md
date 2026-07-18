# InferTileMemorySpace Pass

为 InCore 函数中每个 `TileType` 变量推导片上 `MemorySpace`，插入 `tile.move` 来弥合生产者与消费者约束之间残留的不匹配，并让可证明为循环不变量的 Mat 操作数跨顺序循环保持驻留。

## 概述

`FlattenTileNdTo2D` 之后，每个 InCore tile 都拥有静态的 2D shape，但其 `TileType::memory_space_` 仍未设置（或仅在通过 `target_memory` kwarg 显式标注的部分生产者上设置）。PTO-ISA 硬件暴露了多种不同的片上缓冲区——`Vec`（统一缓冲区 / 向量）、`Mat`（L1）、`Left` / `Right`（L0A / L0B 矩阵乘操作数缓冲区）、`Acc`（L0C 累加器）、`Bias`——大多数算子都对其输入和输出可使用的 memory space 施加约束。本 pass 就是这个约束求解器：它沿数据流前向传播 memory space，遵循显式的 `target_memory` kwarg，沿视图链反向传播需求，并在生产者与消费者无法在同一 space 上达成一致时插入 `tile.move`。

本 pass 运行后，InCore 函数中每个 `TileType` 都带有具体的 `memory_space_`，满足 `ExpandMixedKernel`、`InitMemRef` 以及下游 codegen 所要求的 `TileMemoryInferred` IR 属性。

**前置条件**：

- 输入 IR 必须为 SSA 形式（`SSAForm`）
- 输入 IR 必须包含 InCore tile 操作（`IncoreTileOps`）
- InCore / Orchestration 拆分必须已完成（`SplitIncoreOrch`）
- 语句结构必须已规范化（`NormalizedStmtStructure`）

**使用时机**：紧接 `FlattenTileNdTo2D` 之后运行，先于 `ResolveBackendOpLayouts` / `ExpandMixedKernel`。它是 tile memory 成为下游契约的标准时点——尤其是 `ExpandMixedKernel` 的混合 kernel 检测和 `InitMemRef` 的缓冲区分配都直接读取该结果。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::InferTileMemorySpace()` | `passes.infer_tile_memory_space()` | Program 级 |

**Python 用法**：

```python
from pypto.pypto_core import passes

infer_pass = passes.infer_tile_memory_space()
program_inferred = infer_pass(program)
```

本 pass 仅重写 `func_type_ == FunctionType::InCore` 的函数。Orchestration 与 Opaque 函数原样返回。

## 算法

每个 InCore 函数依次经历五个阶段，均由 IR Visitor / Mutator 实现。阶段 4 只构建一次自底向上的循环清单和完整的语法使用关系，并且每个循环的原始直接循环体只分析一次。嵌套循环会独立重写，该阶段因此保持 O(N) 复杂度。每次 pass 调用中，一条链最多跨过一层词法循环，而不会反复沿新建 preheader 向外移动。

### 阶段 0 — 反向需求收集（`DemandCollector`）

对函数体执行一次遍历，记录两类信息：

1. 对于其算子在 `OpRegistry` 中注册了 `input_constraints` 的每一个 `Call`，把每个受约束输入的 *第一个* 允许 memory space 记录为该输入变量的 "需求"。后端会把规范的（无需 move、最便宜的）space 排在第一位——例如 `tile.store` 列出 `{Vec, Acc}`，因此 Vec 生产者无需 move，Acc 来源的 tile 也保留原 space。
2. 对于标记了 `OutputMemoryInheritsInput()` 的算子（如 `tile.fillpad`、`tile.slice`、`tile.reshape`），按程序顺序记录一条从输出变量指向第一个 tile 类型输入的 `dst → src` 边。

随后在这些边上**反向**传播需求：单次反向序遍历即可达到不动点。这是因为 SSA 中 inherit-input 算子的 `dst` 总在 `src` 之后定义，一次反向扫描即可完成 O(N) 的不动点。当同一变量上两个需求冲突时，非 `Vec` 的需求获胜（`ShouldOverrideDemand`）——`Vec` 是宽松的默认值，应被来自 compute 算子的特化需求覆盖。

正是这一阶段使 `slice(tensor) → fillpad → matmul` 链能把 matmul 的 `Left` / `Right` 需求一直传回 `tile.slice` 的输出，从而让阶段 1 把该生产者直接解析为 `Left` / `Right`，而无需绕道 `Vec`。

### 阶段 1 — 前向分析（`TileMemorySpaceAnalyzer`）

遍历函数体，为每个 TileType 变量分配一个 `MemorySpace`，结果存入 `var_memory_` map。

对每个 LHS 为 `TileType` 的 `AssignStmt`，分析器按 RHS 的形式分派：

- **调用 `tile.*` 算子的 `Call`** → `InferFromOp`（见下文解析表）。
- **调用非 `tile.*` 算子但产出 TileType 的 `Call`** → 默认为 `Vec`。
- **普通 SSA 别名 `y = x`** → 继承 `x` 的 memory space。Python 前端在消除已经具备匹配 `valid_shape` 的输入上的空操作 `tensor.fillpad(pad=zero)` 时会发出此种别名；别名在值上等同于源，必须保持一致的 memory space。

对每个带 `return_vars_` 的 `ForStmt`，访问完函数体后，分析器把每个 yield 变量的 memory space 拷贝到对应的 `return_var_`。同样的 space 还会被强制写到：

- 对应的 `iter_arg_` —— 用于覆盖累加器模式：`tile.create` 保守地默认 `Vec`，但循环体写入了不同 space（如来自 `matmul_acc` 的 `Acc`）。如果不做这一步反向传播，最终的 `tile.store` 读到的是 Vec 类型 tile，会导致 `ExpandMixedKernel` 误判为混合 kernel，进而生成错误的 AIC/AIV IR。
- `iter_arg_` 下面的 TileType `init_var_` 载体 —— 处理 `IfStmt` 的 `return_var`（永远不会作为 `AssignStmt` 被访问）作为循环 init 的情形。

#### 阶段 1 的逐算子解析表

| 生产者类型 | 解析得到的 memory space |
| ---------- | ----------------------- |
| 未注册的 cube 算子（`tile.matmul_mx*`） | `Acc` |
| 其他未注册算子 | `Vec` |
| 已注册但无 `MemorySpec` 的算子 | 若 `Call` 返回类型已设置且非 `DDR`，则使用之；否则 `Vec` |
| `deduce_output_memory` 返回 `Some(s)` 的已注册算子（如 `tile.matmul → Acc`） | `s` |
| `output_inherits_input` 算子（如 `tile.slice`、`tile.fillpad`、`tile.reshape`），且解析器返回 `None` | 第一个 tile 输入的 space；否则 `Vec` |
| `HasRetargetableMemoryKwarg()` 算子（如 `tile.load`、`tile.create`），且解析器返回 `None`（kwarg 缺失） | 阶段 0 的需求若为 `Vec` 或 `Mat` 则使用之；否则继承输入；否则 `Vec` |
| `tile.*` 算子，`deduce_output_memory` 返回 `None`，且既非 retargetable 也非 inherit | 继承输入；否则 `Vec` |

对 retargetable 生产者执行 "夹逼到 `{Vec, Mat}`" 是有意为之：面向 DDR 的 `tile.load` 不能直接产出 `Left` / `Right` / `Acc` / `Bias`；即便下游需求是这些 space 之一，生产者也必须停在 `Mat`（或 `Vec`），由阶段 2 插入 `tile.move` 抵达特化 space。

阶段 1 **从不**覆盖已有的 `target_memory` kwarg。如果用户写了 `pl.load(..., target_memory=Mat)`，而下游 `matmul` 需要 `Left`，则 load 仍保持 `Mat`，并由后续插入 `tile.move`。

### 阶段 2 — Move 收集（`MoveCollector`）

再次遍历函数体。对每个其算子带 `input_constraints` 的 `Call`，检查每个受约束输入变量在 `var_memory_` 中的解析结果是否在允许列表内。任何不匹配都会记录为 `MoveKey = (producer_var, target_space)` 加入 `needed_moves_`，其中 `target_space` 取该输入槽允许列表的第一个。阶段 3 会在每个外层 `SeqStmts` 作用域（即每个插入点缓存作用域）内最多为每个唯一 key 物化一个 `tile.move`，因此同一 `(producer_var, target_space)` 仍可能在兄弟作用域（如 `then` / `else` 分支）中分别物化。

### 阶段 3 — 重写（`TileMemorySpaceMutator`）

完整的 `IRMutator` 重写，产出新的函数体：

1. **变量重写（`VisitExpr_(Var)`）** —— 对每个解析到 space 的 TileType 变量，构造一个新的 `Var`，其 `TileType` 携带 `memory_space_`。当 space 改变时，同时把 `tile_view_` 刷新为目标 space 的隐式视图（例如 `Acc` 期望 col_major / row_major / fractal=1024，而非 Vec 风格的 row_major / none_box / fractal=512）。结果缓存在 `var_cache_`，使得对同一变量的多次引用保持身份一致。
2. **`tile.move` 插入（`VisitStmt_(SeqStmts)` → `InsertMovesForConsumer`）** —— 在每个 RHS 为受约束 `Call` 的 `AssignStmt` / `EvalStmt` 处，对每个挂着待处理 `MoveKey` 的输入，在消费者**之前**新增一条 `tile.move` 形式的 `AssignStmt`。新 `Var`（`<orig>_<TargetSpace>`）记入 `created_moves_`，作用域绑定到外层 `SeqStmts`，从而 `IfStmt` `then` 分支里发出的 move 不会泄漏到 `else` 分支（否则会留下悬空 SSA 引用）。当后端已配置时，会查询 `BackendTileLayoutSpec::input_layouts`，让插入的 `tile.move` 携带消费者所需的 `blayout`（`Vec` 目标还会带上 `slayout=none_box`），避免后续 `ResolveBackendOpLayouts` 的修复。
3. **参数替换（`VisitExpr_(Call)`）** —— 用 `created_moves_` 中已有的项替换每个受约束的输入参数。
4. **Retargetable 生产者 kwarg 重写（`VisitStmt_(AssignStmt)`）** —— 对注册了 `HasRetargetableMemoryKwarg()` 的算子，若阶段 1 把输出解析到与 kwarg 不同的 space（或 kwarg 缺失），则重写 `Call` 的 `target_memory` kwarg 与结果 `TileType`，使之匹配。这让 codegen 与赋值左侧 `Var` 的注解保持一致；这是必要的，因为阶段 1 可能基于反向需求做出解析，而 kwarg 永远看不到这些需求。
5. **LHS / RHS 类型同步** —— 当 `VisitExpr_(Call)` 在替换被 move 后的参数后，借由 `OpRegistry` 重建 `Call`，结果类型可能与 LHS `Var` 的原类型不同（重建的 call 会看到布局变化后的输入）。Mutator 把 LHS `Var` 的 `TileType` 同步到重建 call 的 shape / dtype / memref / view，同时保留变量重写阶段选定的 `memory_space_`，保证 roundtrip 等价。

### 阶段 4 — 循环不变量 Mat 驻留（`loop_invariant_mat_residency`）

所有 space 显式化后，一个独立的内部 transform 会识别形如 `tile.load(GM → Mat) → tile.transpose_view* → tile.move/tile.extract(Mat → Left/Right)` 的不变量前缀，并把该前缀移到循环 preheader。该链必须只有一个终止 `Left` / `Right` 值，并且只在一个 `tile.matmul`、`tile.matmul_bias` 或 `tile.matmul_acc` 的对应操作数位置使用。这样可将优化严格限定为驻留 matmul 操作数，而不是通用 tile LICM。对于这一受支持的单一使用形态，tensor-level matmul 的驻留操作数只加载一次，而依赖循环变量的另一操作数仍留在循环内正常流式加载。

这是 issue #2077 所要求的更广泛驻留行为中的保守首个子集，并不是通用的 tensor-level residency contract。直接进入或由程序外部进入的 InCore 函数没有可分析的调用者证据，因此会拒绝该优化。不同的外部 tensor 参数同样会被拒绝：PyPTO 没有运行时 `noalias` 契约来保证其底层分配互不重叠。当前只有 root orchestration IR 内由 `tensor.create` 创建的存储能够提供正向调用者 provenance。若要覆盖外部操作数，必须增加可强制执行的 no-alias 契约或带非提升回退路径的运行时检查；本 transform 不会自行假设这一点。当 `AutoTileMatmulL0` 将一个 Mat panel 展开为多个依赖 K 的 L0 extract 时，当前要求单一使用的链识别器同样会拒绝；此时整个 panel 的 GM→Mat load 仍位于外层用户循环内，而依赖 K 的 extract 会正确保留在内层 K 循环中。支持这种情况需要把 GM→Mat panel 驻留与可选的 Mat→L0 前缀移动分开分析。

候选资格首先依赖编译器私有的 provenance。`ConvertTensorToTileOps` 会标记其生成的所有 `GM → Mat` bridge load；该标记经过打印、flatten 和 L0 自动分块后一直保留到本阶段。阶段 4 随后证明带标记的 load 符合上文所述的精确静止 matmul 操作数链。用户手写的 `tile.load(..., target_memory=Mat)` 不带此标记，因此本优化绝不会提升它，从而保证显式 tile 程序仍由用户控制。

首版合法性规则有意保持严格：

- 循环必须为 `Sequential`，边界是常量，step 为正，且至少执行一次；
- 被移动的赋值必须是循环体顶层、无条件执行的语句；
- GM 源必须是方向为 `ParamDirection::In` 的直接 tensor 参数，且 load 带有编译器生成的 Mat bridge 标记；
- InCore 函数必须至少有一个来自 root orchestration 函数（即没有程序内调用者的 orchestration 函数）的直接 `Call`，并且该 InCore 函数的每个调用点都必须是这种 root-orchestration 直接 `Call`；`Submit` 调用点总会使候选失效，因为异步提交不能作为正向别名证据；
- 在每个此类调用点，候选 `Tensor In` 实参必须解析到由 `tensor.create` 创建、归编译器所有的分配；普通别名以及 `tensor.slice` / `tensor.assemble` / `tensor.view` 别名会规范化到该存储 root，所有可写 `Tensor Out` / `Tensor InOut` root 都必须已知且均不得与候选 root 重叠；InCore 函数自身也不能写入该 root，而无关 scalar 和其他只读 `Tensor In` root 不参与此过滤；
- offset、shape 以及整个被移动的依赖前缀都必须是循环不变量；
- 循环头（边界或 loop-carried 初始值）或循环体子树内出现任何函数调用、任务提交、跨核操作、同步、缓存维护或未知 builtin 时都会拒绝驻留，因为移动到 preheader 可能使 load 在迭代之间越过未知或隐藏的顺序效应；其他直接控制流或有副作用语句若出现在 candidate 之前，则会关闭可提升前缀；
- 链中每个值只能有预期的单一语法使用；普通 SSA 别名、`Submit` 实参、嵌套表达式、循环初始值、yield、return 以及额外的 call 实参都计为使用，因此会使候选失效；
- 被移动的结果不能是 loop-carried 值或 yield 值；
- 函数中所有实际拥有分配的 `Mat`、`Left`、`Right` tile 都必须具有静态大小，且按分配器对齐后的全函数上界不得超过后端容量；
- 函数中不得存在尚未表示为 tile 分配、因而无法计入容量的显式保留缓冲区区域。

`InOut` / `Out` 源、外部输入分配、手写 tile load、直接或从程序外进入的 InCore 函数、`Submit` 调用点、经过 InCore wrapper 或被调用的 orchestration helper 的调用、未知的候选或可写调用点 root、候选/写入别名、额外语法使用、条件 load、动态或零次循环、容量未知的情形、被 yield 或循环携带的结果，以及依赖循环变量的 extract 都会安全拒绝并保持 IR 不变。即使其他调用安全，只要有一个调用点不安全或不是 root 调用也会使候选失效。首版实现有意不传播 wrapper 证据：语法上不同的 wrapper 参数仍可能在 wrapper 自身的调用者处发生别名。容量检查只统计实际拥有分配的值，而不重复统计零拷贝 view 或 SSA 别名；它采用与 `InitMemRef` / `AllocateMemoryAddr` 相同的字节大小和地址对齐规则，并包含循环外已存活的分配。若某个 memory space 中的分配可能被后续 pipeline lowering 复制，也会拒绝该 space 的驻留，除非被移动的前缀位于不受影响的 space。该全函数上界刻意强于任一规划器的生命周期复用，因此 residency 重写不会在 PyPTO 或 PTOAS 规划器下引入后续容量失败。嵌套循环会独立处理；一次 pass 调用只会将链移到其直接词法 preheader。本阶段不会全局重映射参数，也不会把依赖 K 的 L0 extract 移出 `AutoTileMatmulL0` 的 pipeline 循环。

#### 驻留示例

对于 root orchestration 函数先创建全新 LHS 存储、再调用 InCore kernel 的 tensor 程序，静止 LHS 的 bridge 会移到用户循环之前，而依赖 N 的 RHS 仍在循环中流式加载：

```python
# Tensor 源程序
for n, (acc,) in pl.range(0, 256, 128, init_values=(out,)):
    rhs_n = pl.slice(rhs, [128, 128], [0, n])
    c_n = pl.matmul(lhs, rhs_n, out_dtype=pl.FP32)
    result = pl.yield_(pl.assemble(acc, c_n, [0, n]))
```

```python
# ConvertTensorToTileOps、L0 自动分块和 InferTileMemorySpace 之后
lhs_mat = pl.tile.load(lhs, [0, 0], [16, 128], target_memory=pl.Mem.Mat)
lhs_left = pl.tile.move(lhs_mat, target_memory=pl.Mem.Left)
for n, (acc,) in pl.range(0, 256, 128, init_values=(out,)):
    rhs_mat = pl.tile.load(rhs, [0, n], [128, 128], target_memory=pl.Mem.Mat)
    rhs_right = pl.tile.move(rhs_mat, target_memory=pl.Mem.Right)
    c_n = pl.tile.matmul(lhs_left, rhs_right)
    result = pl.yield_(pl.tile.store(c_n, [0, n], acc))
```

为便于阅读，上例省略了内部 provenance 属性和 root orchestration 调用。调用者先执行 `fresh_lhs = pl.create_tensor([16, 128], dtype=pl.BF16)`，再把它作为 `lhs` 传入；因此编译器能够证明该分配与外部可写 `out` 不同。仅仅传入不同的外部 `lhs` 与 `out` 参数并不充分。另一个只读 `rhs` root 与写别名过滤无关。缺少可信存储 provenance 时，仍保留原来的循环内放置方式。

## 通用 memory-space 推导示例

来源：`tests/ut/ir/transforms/test_infer_tile_memory_space.py::test_matmul_gets_acc`。

**优化前**：

```python
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def main_incore_0(
        self,
        x: pl.Tensor[[16, 128], pl.BF16],
        y: pl.Tensor[[128, 128], pl.BF16],
        out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
    ) -> pl.Tensor[[16, 128], pl.FP32]:
        x_tile: pl.Tile[[16, 128], pl.BF16] = pl.load(x, [0, 0], [16, 128])
        y_tile: pl.Tile[[128, 128], pl.BF16] = pl.load(y, [0, 0], [128, 128])
        z_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul(x_tile, y_tile)
        out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_tile, [0, 0], out_0)
        return out_0
```

**优化后**：

```python
@pl.program
class After:
    @pl.function(type=pl.FunctionType.InCore)
    def main_incore_0(
        self,
        x: pl.Tensor[[16, 128], pl.BF16],
        y: pl.Tensor[[128, 128], pl.BF16],
        out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
    ) -> pl.Tensor[[16, 128], pl.FP32]:
        x_tile: pl.Tile[[16, 128], pl.BF16, pl.MemorySpace.Vec] = pl.load(x, [0, 0], [16, 128])
        y_tile: pl.Tile[[128, 128], pl.BF16, pl.MemorySpace.Vec] = pl.load(y, [0, 0], [128, 128])
        x_tile_L: pl.Tile[[16, 128], pl.BF16, pl.MemorySpace.Left] = pl.move(
            x_tile, target_memory=pl.MemorySpace.Left
        )
        y_tile_R: pl.Tile[[128, 128], pl.BF16, pl.MemorySpace.Right] = pl.move(
            y_tile, target_memory=pl.MemorySpace.Right
        )
        z_tile: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Acc] = pl.matmul(x_tile_L, y_tile_R)
        out_0: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_tile, [0, 0], out_0)
        return out_0
```

发生的变化：

- 两个 `tile.load` 的输出都得到 `pl.MemorySpace.Vec`（无 `target_memory` kwarg，且这两个输入也未传播到可达的 Mat 需求）。
- `tile.matmul` 的 `deduce_output_memory` 把输出解析为 `Acc`。
- `tile.matmul` 的输入约束（`Left`、`Right`）与生产者的 `Vec` 不匹配，因此阶段 2 记录了两个 move key，阶段 3 在消费者前插入了 `x_tile_L`、`y_tile_R`。

如果用户改写为 `pl.load(..., target_memory=pl.MemorySpace.Mat)`，阶段 1 将遵循 kwarg，`tile.load` 输出已为 `Mat`。matmul 仍然需要 `Left` / `Right`，因此会从 `Mat` 出发插入 move——这也正是 `test_matmul_full_pipeline` 测试的标准全流程。

## 实现

**头文件**：`include/pypto/ir/transforms/passes.h`

**实现**：`src/ir/transforms/infer_tile_memory_space_pass.cpp`

**Python 绑定**：`python/bindings/modules/passes.cpp`

**测试**：`tests/ut/ir/transforms/test_infer_tile_memory_space.py`

本 pass 还在同一 `.cpp` 中注册了 `TileMemoryInferred` `PropertyVerifier`，在需要校验 `TileMemoryInferred` IR 属性时运行。它在每个 InCore 函数上检查两条不变量：

1. 由 `AssignStmt` 定义的每个 TileType `Var` 都已设置 `memory_space_`。
2. 每个具有已注册 `input_constraints` 的 `Call` 输入所引用的 tile，其 `memory_space_` 都在允许集合中。

## Pass Properties

| 属性 | 取值 |
| ---- | ---- |
| Required | `SSAForm`、`IncoreTileOps`、`SplitIncoreOrch`、`NormalizedStmtStructure` |
| Produced | `SSAForm`、`TileMemoryInferred`、`NormalizedStmtStructure` |
| Invalidated | — |

`TileMemoryInferred` 属性是本 pass 建立的契约。下游 pass（尤其 `ExpandMixedKernel` 与 `InitMemRef`）依赖该契约，配套的属性 verifier 守护回归。

## 作用范围

| 函数类型 | 行为 |
| -------- | ---- |
| `InCore`（含 `AIC`、`AIV`） | 进行变换 |
| `Orchestration` | 不变 |
| `Opaque` | 不变 |

本 pass 还断言任何 InCore 函数的参数都不能是 `TileType` —— InCore 参数必须是 `TensorType`。该断言在阶段 1 起始处检查，违反时触发 `CHECK` 失败。
