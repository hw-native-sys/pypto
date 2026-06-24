# OptimizeOrchTensors Pass

优化编排函数与 InCore 函数之间的张量缓冲区使用，消除冗余分配、改善布局信息，并把静态可证明的局部张量窗口在 orchestration callsite 显式化。

## 概述

`ConvertTensorToTileOps` 之后，编排函数在每个 InCore 调用点分配输出张量（`tensor.create`），即使在循环内同一缓冲区可以复用。本 pass 应用六个优化/降级模式来减少分配、改善缓冲区布局信息、在 orchestration callsite 显式化可静态证明的局部张量窗口，并降级可证明 linked-flow 产生的 runtime-current marker。

**前置条件**：

- 输入 IR 必须已完成 InCore 作用域提取和 tile 转换（需先运行 `ConvertTensorToTileOps`）

**使用时机**：在 `ConvertTensorToTileOps` 之后、`FlattenTileNdTo2D` 之前运行。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::OptimizeOrchTensors()` | `passes.optimize_orch_tensors()` | Program 级 |
| `pass::OptimizeOrchTensors("exact", "local")` | `passes.optimize_orch_tensors(window_policy="exact", window_flow="local")` | Program 级 |

**Python 用法**：

```python
from pypto.pypto_core import passes

opt_pass = passes.optimize_orch_tensors()
program_opt = opt_pass(program)
```

多数用户应先用 `window_option` 选择一个 preset：

```python
passes.optimize_orch_tensors(window_option="stable")
```

`window_option` 取值：

- `"none"`：展开为 `window_policy="none", window_flow="local"`。默认不
  window；显式 kernel attrs 仍可 opt in。
- `"stable"`：展开为 `window_policy="stable", window_flow="local"`。默认值；
  使用稳定 exact window，不扩张 runtime submit 实参。
- `"exact"`：展开为 `window_policy="exact", window_flow="local"`。允许会扩张
  submit 实参的 exact window；不允许 boundingBox 空洞，也不启用 linked flow。

`window_option="none"` 不是 kill switch；它只改变全局默认。带有显式
`window_outputs`、`window_inputs` 或 `window_flow` attrs 的 kernel 仍可 opt in。

高级用户可以使用显式两轴 API：

```python
passes.optimize_orch_tensors(window_policy="stable", window_flow="local")
```

不要同时传 `window_option` 和 `window_policy` / `window_flow`。二者分别表示
preset API 和显式两轴 API，混用会报错。

`window_policy` 控制单个 callsite 的 coverage 形状和 submit 参数预算。
`window_flow` 控制可证明的 coverage 关系能否跨 writer/reader callsite 传播。
Carrier/base/extent/current/remat 不是用户配置项，只是某些 proven linked flow
的 lowering 结果。

`boundingBox` 和 `linked` 故意不放进 `window_option`。需要更大的
coverage/flow 语义时，应显式使用 `window_policy="boundingBox"` 或
`window_flow="linked"`。

`window_policy` 取值：

- `"none"`：默认不 window；显式 kernel attrs 仍可 opt in。
- `"stable"`：默认值。只做 ABI-safe exact window；不扩张 submit
  tensor/scalar 参数。
- `"exact"`：使用可证明正确的 exact pieces；submit 签名可以扩张；不允许
  boundingBox 空洞。
- `"boundingBox"`：exact-first，但当 exact pieces 超过表达预算、linked proof
  需要连续 coverage，或机械 cost 规则证明 view/参数数量下降时，允许连续
  boundingBox。用户显式接受空洞覆盖。

`window_flow` 取值：

- `"local"`：每个 callsite 独立改写，不产生跨 callsite current/carrier
  lowering。
- `"linked"`：当 writer 和 reader 两端都 opt in 且 proof 成立时，允许跨
  callsite 复用 coverage；proof 失败自动 fallback，不报错。

常用全局组合：

| 全局配置 | Submit 参数 | Coverage | 空洞 | Linked flow | Dynamic carrier/base/extent | Aggregate barrier |
| -------- | ----------- | -------- | ---- | ----------- | --------------------------- | ----------------- |
| `none + local` | 默认不变 | 默认不 window | 否 | 否 | 否 | 否 |
| `none + linked` | 默认不变 | 只影响显式 opt-in kernel | 取决于 opt-in | 只有 writer/reader 双方 effective flow 都是 linked 的 opt-in edge | 仅 proof 成立时可能出现 | 仅 proof 成立时可能出现 |
| `stable + local` | 不扩张 | ABI-safe exact | 否 | 否 | 否 | 否 |
| `stable + linked` | 不扩张 | ABI-safe exact | 否 | 可尝试 proof，但多数 dynamic reader fallback | 需要新增参数时不会出现 | 需要新增参数时不会出现 |
| `exact + local` | 可扩张 | exact pieces | 否 | 否 | 否 | 否 |
| `exact + linked` | 可扩张 | exact pieces | 否 | 只允许 single dense exact linked coverage | 不会偷偷合成 boundingBox carrier | proven aggregate-current join 可出现 |
| `boundingBox + local` | 可扩张 | exact 或 boundingBox | 是 | 否 | 否 | 否 |
| `boundingBox + linked` | 可扩张 | exact 或 boundingBox | 是 | 允许 proven writer-reader linked coverage | 可能出现 | 可能出现 |

Linked flow 有两类相互独立的 lowering 结果：

- **动态范围转发**可能物化 carrier base/extent，并为 dynamic reader remat
  writer 的连续窗口。
- **聚合 current 汇合**可能在循环产出的 tensor 与后续 full-parent 或 inexact
  consumer 之间插入 runtime-current barrier。

聚合 barrier 不要求存在 dynamic carrier。关闭一条 writer/reader carrier 路径，
不会关闭其他已经证明成立的聚合汇合。`window_flow="local"` 下两类 lowering 都不
允许；`stable + linked` 下，任何需要扩张 submit 参数的 lowering 都会 fallback。

**Per-kernel 覆盖**（通过 function attrs）：

```python
@pl.function(
    type=pl.FunctionType.InCore,
    attrs={
        "window_outputs": "exact",       # off | stable | exact | boundingBox
        "window_inputs":  "off",         # off | stable | exact | boundingBox
        "window_flow":    "linked",      # local | linked
    }
)
def my_kernel(...): ...
```

缺失 attrs 继承全局设置。显式 side attrs 优先，包括全局默认为 `"none"` 时的
显式 opt-in。`window_outputs` 和 `window_inputs` 是相互独立的 coverage
permission。`InOut` 的依赖 coverage 可以拆成 read coverage 和 write coverage
分析，但 runtime submit 仍只能给这个参数一个 TensorView；如果读写 submit
view 冲突，pass 会选择保守 view 或 fallback。

如果要完全关闭某个 kernel 的本地 window externalization，需要两个 side attr
都设为 `"off"`：

```python
attrs={"window_outputs": "off", "window_inputs": "off"}
```

`boundingBox` 是 permission，不是强制模式。Pass 默认优先 exact。只有当 exact
pieces 超出 submit/view 表达预算、linked proof 需要连续 coverage，或机械 cost
规则证明 runtime tensor view/参数数量下降时，才会选择 boundingBox。如果
boundingBox 等于 full parent 或没有预期收益，pass 会 fallback 到 exact 或 full
parent，不报错。

```python
attrs={"window_outputs": "boundingBox"}
```

`exact` 同样是 permission，而不是强制模式。Separated exact pieces 可能分别变成
runtime tensor 参数。Pass 会拒绝超过 runtime 32 个 tensor 参数或 16 个 scalar
参数上限的改写，但合法的 multi-piece rewrite 仍可能显著增加调度开销。把
multi-piece kernel 显式设为 `exact` 后，应检查生成的 orchestration。

`window_flow="linked"` 不是 carrier 开关。它只允许双方都 effective linked 且
proof 成立的 writer-reader edge 复用 coverage。当前 lowering 可能表现为
carrier/base/extent/remat；未来也可能换成别的 lowering。

```python
attrs={"window_inputs": "boundingBox", "window_flow": "linked"}
```

例如，在 Qwen prefill 风格的 dynamic-indexed KV-cache writer/reader 链中：

- `stable + local`：只保留 ABI-safe local exact window。Dynamic KV-cache
  reader 保持 full tensor。
- `exact + local`：允许 exact pieces 扩张 submit 签名，但不产生跨 callsite
  carrier lowering。
- `boundingBox + local`：允许本地连续 coverage 和空洞，但仍不会有
  carrier/base/extent。
- `boundingBox + linked`：允许 qk/sv 风格 dynamic reader 使用可证明的
  output-derived linked flow。Proof 失败时 reader 保持 full parent。

linked dynamic-reader 路径依赖以下所有条件：

- producer 的有效 `window_outputs` 能提供需要的 exact 或 boundingBox coverage
- reader 的有效 `window_inputs` 允许需要的 coverage
- writer 和 reader 两端 effective `window_flow` 都是 `"linked"`
- writer/reader pair 满足 shared-root、loop-order 和 min/max proof 检查

如果任一条件不满足，dynamic reader 会保持 full parent tensor；pass 不能仅因为
coverage attr 存在就插入 runtime-current barrier。

### 根据泳道图调参

Window externalization 是调度 trade-off，不是单调加速开关。建议用泳道图判断
瓶颈到底来自设备侧依赖串行，还是来自 host 侧 orchestration dispatch 开销：

| 泳道图现象 | 可能瓶颈 | 可以尝试 |
| ---------- | -------- | -------- |
| 理论上应并发的 device tasks 被粗粒度 tensor 依赖串行化 | 依赖 coverage 太粗 | 开启 window，通常先试 `window_option="stable"`，再试 `window_option="exact"` 或局部 kernel attrs。 |
| 本地 window 后 dynamic reader 仍是 full-parent | 真正有用的是跨 callsite 关系 | 对全局或相关 producer/consumer kernel 尝试 `window_policy="boundingBox", window_flow="linked"`。 |
| 开 window 后 orchestrator 泳道变密，后续 task dispatch 变慢 | 改写引入了太多 slice、assemble、tensor view、scalar arg、carrier 或 barrier | 回退到 `stable`，把噪声 kernel 设为 `window_outputs="off"` / `window_inputs="off"`，或保持 `window_flow="local"`。 |
| Multi-piece kernel 改善了依赖精度，但生成的 `build_output/.../orchestration/*.cpp` 里 submit 实参数量明显增加（`add_input`、`add_inout`、`add_output` 或 `add_scalar`） | exact pieces 对该 workload 太碎。改写可能超过 runtime 侧 `MaxSig` 限制，也可能虽然合法但让 orchestrator 花太多时间准备 dispatch。 | 如果可接受带空洞的连续 range，并且能减少 submit 签名，局部允许 `boundingBox`；否则关闭该 kernel 的 window。 |

调参时优先选择能解决泳道图串行问题的最保守配置。`stable + local` 是安全默认；
`exact + local` 适合 fragmented exact coverage 能真实移除依赖的场景；
启用 exact multi-piece window 后，应检查 `build_output/.../orchestration/` 下生成的
C++：如果改写后的 callsite 新增大量 `add_input`、`add_inout`、`add_output` 或
`add_scalar`，runtime 签名可能超过 `MaxSig` 限制；即使仍然合法，也可能让
orchestrator 成为新的计时瓶颈。这时可以对噪声 kernel 局部允许 `boundingBox`，
用一个带空洞的连续 range 合并 pieces、减少 runtime views；如果空洞不可接受或仍无
收益，则关闭该 kernel 的 input/output window。`boundingBox + linked` 适合已证明的
writer-reader flow，并且可以接受更宽连续 coverage 与 linked lowering 带来的额外
orchestration 工作。如果只有单个 kernel 造成 orchestrator 压力，优先对该 kernel
做覆盖，而不是全局关闭仍有收益的 window。

旧配置拼写只有在映射无歧义时才兼容：全局 `auto` 映射为 `stable`，side
`coalesce` 映射为 `boundingBox`。新代码应使用规范名称。含义不明确的 `all`、
`carrier`、`coalesce_carrier` 以及 kernel 级 `window_policy` attr 会直接报错；
应改用显式的 `window_outputs`、`window_inputs` 和 `window_flow`。

parent shape 含 dynamic dim 的 output window 会保守处理。对 dynamic parent 的
static-offset/static-size 静态 partial window 会保持 full-tensor，因为同一个编译图可能用更小的
dynamic extent 运行。动态 offset 的 window，例如运行时 slot 上的 KV-cache 写入，
只要静态证明和策略门槛允许，仍可以改写。

调试本 pass 时，可以使用诊断专用环境变量：
`PYPTO_WINDOW_EXTERNALIZE_LOG=1` 打印 stable/coverage accept/reject 决策。这个变量
不是公开 policy 语义的一部分，也不会改变哪些候选会被改写。

## 优化模式

本 pass 按顺序应用六个模式。每个模式可以看到前一个模式的结果。

### 模式 1：迭代参数复用（IterArgReuseOptimizer）

**问题**：在 `for`/`while` 循环内，每次迭代都通过 `tensor.create` 分配新的输出张量，即使 InCore 结果作为 iter-arg 反馈到下一次迭代。

**方案**：将 `Out` 参数合并到对应的 `In` 参数（提升为 `InOut`），移除 `tensor.create`，并重定向 `tile.store` 写入复用的缓冲区。

**优化前**：

```python
for i in pl.range(N, init_values=[init_buf]):
    out: pl.Tensor = pl.tensor.create(shape, dtype=pl.FP32)  # 冗余分配
    result: pl.Tensor = self.incore_fn(iter_arg, out)          # In + Out 参数
    pl.yield_(result)
```

**优化后**：

```python
for i in pl.range(N, init_values=[init_buf]):
    result: pl.Tensor = self.incore_fn(iter_arg)  # InOut 参数（复用 iter-arg 缓冲区）
    pl.yield_(result)
```

### 模式 2：Assemble 父张量步长（AssembleParentStridesOptimizer）

**问题**：当编排函数通过 `tensor.assemble` 将 InCore 结果分散到更大的张量时，InCore 函数的 `tile.store` 不知道父张量的步长，可能导致次优的内存布局。

**方案**：分析编排函数中的 `tensor.assemble(parent, incore_result, offset)` 模式，将父张量的形状作为 `TensorView` 步长附加到 InCore 函数的 `Out` 参数类型上，使 `tile.store` 能使用正确的内存布局。

### 模式 3：Assemble 循环重写（AssembleLoopRewriter）

**问题**：InCore 函数包含一个通过 `tile.assemble` 将结果累积到 iter-arg 的 `for` 循环，然后存储最终结果。`tile.assemble` 每次迭代都创建中间 tile 副本。

**方案**：将循环体重写为直接使用 `tile.store`（写入 `Out` 参数），用 `Out` 参数初始化 iter-arg 代替 `tile.create`。

### 模式 4：切片输入步长（SliceInputStridesOptimizer）

**问题**：当编排函数将切片张量（`tensor.slice`）作为 `In` 参数传递给 InCore 函数时，InCore 函数的参数使用连续步长（从自身形状计算），而非父张量的步长。当切片是父张量的非连续视图时，这会导致错误的内存访问。

**方案**：分析编排函数中的 `tensor.slice(parent, size, offset)` 模式。当切片结果作为 `In` 参数传递给 InCore 调用时，将父张量形状推导出的步长通过 `TensorView` 附加到 InCore 函数的 `In` 参数类型上，使 `tile.load` 能使用正确的内存布局。

### 模式 5：静态窗口外提（OutWindowExternalizer）

**问题**：某些 outlined callee 实际只写入大 `Out` 张量中的一个静态可证明局部窗口，或只消费大 `In` 张量中的一个静态可证明局部窗口，但调用点仍传入整块张量。后续依赖分析会把它视为整块缓冲区访问，从而引入不必要的串行化。

**方案**：为 callee 克隆出 `__windowed` 版本，收窄被改写的张量参数类型，并局部化内部 offset。然后在 orchestration callsite 物化局部 slice。输出窗口使用 `slice + __windowed call + assemble`：

```python
out_window = pl.tensor.slice(out, shape, offset)
out_window_next = self.kernel__windowed(..., out_window)
out = pl.tensor.assemble(out, out_window_next, offset)
```

输入窗口使用相同的 callsite 局部 slice 物化，但不需要 assemble：

```python
in_window = pl.tensor.slice(inp, shape, offset)
result = self.consumer__windowed(in_window, ...)
```

如果待物化 slice 的 parent 是循环返回 alias，pass 会对 `ForStmt` 和 `WhileStmt`
都把这个 parent 改写为该循环 codegen 可见的 init tensor，避免生成的
orchestration C++ 在作用域外引用 loop-return SSA 名字。循环体内部的
loop-carried iter-arg 不会被这样折叠。

本 pass 有意保持保守的 window eligibility。它不会按 `topk` 等算子名字做特判；只有 callee 函数体能证明满足下面的访问模式时，才会 window 化。

静态 eligibility 之后，默认 `stable + local` 策略还会再做一层保守 cost gate。
它只保留不会增加 `add_inout`、`add_input`、`add_output` 或 `add_scalar`
submit 预算的 exact window。Multi-piece output rewrite、boundingBox coverage、
dynamic carrier args 和跨 callsite current/remat 都不属于 stable 默认行为。
如果 exact pieces 可以扩张 submit 签名，使用 `window_policy="exact"`；如果可以
接受带空洞的连续 coverage，使用 `window_policy="boundingBox"`；只有在允许
writer-reader coverage 跨 callsite 传播时才使用 `window_flow="linked"`。

支持的改写形态：

- `FinalStore`：callee 返回一次写入局部窗口的最终 `tile.store(...)` 结果
- `AggregateWindowLoop`：callee 在循环中携带一个或多个 `Out`，并写入静态可证明的聚合窗口，例如 outlined `kv_proj` 分组形态
- `PureInputWindowConsumer`：有数据返回的 callee 中，某个 `In` 张量参数只通过同一个局部输入窗口被使用
- `AggregateInputWindowLoop`：与 `AggregateWindowLoop` 输出改写配套使用；某个 `In` 张量参数只通过内部 loop 的局部 `tile.load`/`tensor.slice` 窗口读取，并且这些 offset 能沿同一个内部 loop 展开为一个静态可证明的 parent-shaped region，例如 qk norm 的 q/k 输入
- Linked-flow lowering：当 writer 和 reader 两端都 opt into linked flow 且
  proof 成立时，pass 可能用 carrier/base/extent/remat lowering 这条关系。
  Runtime-current marker 只由显式 linked-flow plan 发出；不能根据
  callee 名称或 `__windowed` clone 的存在做 post-pass heuristic 推断。

输出窗口 eligibility：

- 写入必须是静态可证明的局部 `tile.store` 窗口或聚合窗口循环
- window shape 和 offset 必须足够静态，能够物化为 `tensor.slice`
- offset 必须是该 pass 可接受的外层循环变量仿射表达式
- multi-`Out` 改写采用全有或全无策略
- 如果同一 callsite 中多个被 externalize 的 `Out` 参数解析到同一个 parent tensor，该 callsite 保持 full-tensor；Pattern 5 不尝试把多个 `tensor.assemble` 串成同一个 parent state
- 顺序循环 sibling 只有在每个被改写 `Out` 都能证明跨 sibling iteration 不重叠时才改写
- 同一 scope 内写入同一 parent 或 alias parent tensor 的 sibling writer，只要每个 writer 自身满足静态 output-window eligibility，仍然可以 externalize；但如果同一个 parent 还存在无法 externalize 成 output window 的 sibling full writer（`Out` 或 `InOut`），则写同一 parent 的其他 writer 也保持 full-tensor，避免这个非 window writer 掩盖只被部分初始化的区域
- 对剩余 windowed writer，写写/写读顺序交给 runtime TensorMap 对实际 submit 的 window descriptor 做 overlap 建边
- sibling-writer alias 收集会递归进入嵌套 `SeqStmts`、`ForStmt`、`WhileStmt` 和 `IfStmt` body，因此 loop return、tuple projection 这类 tensor alias 会先折叠到 codegen 可见的 parent，再生成 call-site slice
- 后续 full-parent read 不会关闭输出 window；callsite 暴露真实窗口张量之后，正确性依赖 runtime TensorMap overlap dependence

输入窗口 eligibility：

- 参数必须是 `In` 张量
- callee 内部对该参数的每一次引用都必须匹配同一个局部窗口
- 支持的引用只有 `tile.load` 和 `tensor.slice`
- 拒绝 transpose load
- `tile.load` 的 read shape 必须等于候选 window shape
- 如果 `tile.load` 携带 `valid_shape`，它必须等于 read shape；masked load 保持 full-tensor
- 所有匹配引用必须具有相同 window shape 和 offset
- 如果存在任何 unsupported ref，则整个输入参数保持 full-tensor
- pure input-window 的 shape 和 callee-local offset 表达式只能引用 callee 参数；callsite 替换后这些参数可以携带外层 loop-affine 值，windowed callee 内部再相对 `[0, ...]` 读取
- 对 `PureInputWindowConsumer`，如果匹配出的窗口其实是 zero offset 的 full shape，则跳过，因为 slice 不能暴露更窄依赖
- 对 `PureInputWindowConsumer`，如果 callee 没有数据返回，则保持 full-tensor；这类 consumer 可能是 side-effect 或 fence task，full input 本身用于表达更宽的依赖
- input-only 的 `Submit` callsite 保持 full-tensor；在 `manual_scope` 中，即使 callee body 只读局部窗口，full input 也可能有意表达更宽的依赖
- 如果同一个 callee 同时满足 output-window 改写，已经证明成立的 pure input window 会被保留，并在同一个 callsite 一起物化
- 对 `AggregateInputWindowLoop`，所有引用必须位于同一个静态 `ForStmt` 内，至少一个 offset 维度必须随该 loop 变化，并且聚合窗口必须等于输入 parent shape；权重子窗口这类 partial aggregate read 仍保持 full-tensor
- dynamic-indexed reader window 需要可证明的 min/max scan、coverage 实际保留为
  `__windowed` 调用或 linked boundingBox variant 的 writer、允许所需 coverage
  的 reader policy，以及 writer/reader 两端 effective linked flow。没有该
  proven flow 时，dynamic reader 保持 full parent tensor（无 private dynamic
  reader）。

非目标与依赖模型：

- pass 不添加 explicit dependency edges
- pass 不重新引入 later full-parent-read guard
- pass 不预生成全局 window descriptor 数组
- pass 不拆分 SPMD launch，也不 externalize per-block SPMD window
- unsupported consumer，包括 full-tensor reader，保持 baseline/full-tensor input
- runtime-current marker 只由显式 linked-flow analysis 发出；后续 lowering pass
  只负责把 marker 物化为 runtime barrier 并删除 marker call
- `DeriveCallDirections` 保持现有 sound 的顺序 `Out -> InOut` 规则；Pattern 5 只是在该 pass 运行前显式化可证明的局部窗口

## 示例（模式 1）

**优化前**：

```python
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def compute(self, x: pl.Tensor[[64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
        x_tile = pl.load(x, (0,), (64,))
        y_tile = pl.tile.add(x_tile, x_tile)
        ret = pl.store(y_tile, (0,), out_0)
        return ret

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, buf: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        for i in pl.range(10, init_values=[buf]):
            out_0 = pl.tensor.create((64,), dtype=pl.FP32)
            result = self.compute(iter_arg, out_0)
            pl.yield_(result)
        return loop_result
```

**优化后**（模式 1 将 Out 合并到 In，提升为 InOut）：

```python
@pl.program
class After:
    @pl.function(type=pl.FunctionType.InCore)
    def compute(self, x: pl.InOut[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
        x_tile = pl.load(x, (0,), (64,))
        y_tile = pl.tile.add(x_tile, x_tile)
        ret = pl.store(y_tile, (0,), x)
        return ret

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, buf: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        for i in pl.range(10, init_values=[buf]):
            result = self.compute(iter_arg)
            pl.yield_(result)
        return loop_result
```

`tensor.create` 被消除；iter-arg 缓冲区跨迭代复用。

## 实现

**头文件**：`include/pypto/ir/transforms/passes.h`

**实现**：`src/ir/transforms/optimize_orch_tensors_pass.cpp`

**Python 绑定**：`python/bindings/modules/passes.cpp`

**测试**：`tests/ut/ir/transforms/test_optimize_orch_tensors.py`

## Pass 属性

| 属性 | 值 |
| ---- | -- |
| Required | SplitIncoreOrch, IncoreTileOps |
| Produced | SplitIncoreOrch, IncoreTileOps |
| Invalidated | — |

## 关键组件

| 组件 | 作用 |
| ---- | ---- |
| `IterArgReuseOptimizer` | 模式 1 — 合并 Out 参数到 In 参数以复用循环携带缓冲区 |
| `AssembleParentStridesOptimizer` | 模式 2 — 通过 TensorView 附加父张量步长 |
| `SliceInputStridesOptimizer` | 模式 4 — 通过 TensorView 为切片输入的 In 参数附加父张量步长 |
| `AssembleLoopRewriter` | 模式 3 — 将 tile.assemble 循环重写为 tile.store 循环 |
| `OutWindowExternalizer` | 模式 5 — 将 eligible 的局部 Out 写和 eligible In-window consumer 改写为显式 callsite slice |
| `RuntimeCurrentAggregator` | 模式 6 — 将显式 linked-flow runtime-current marker 降级为 barrier 并删除 marker call |
| `BuildOutParamReturnMappings` | 共享辅助函数 — 通过 tile.store 映射 Out 参数到返回索引 |
| `ComputeRowMajorStrides` | 共享辅助函数 — 从形状计算行主序步长 |

## 作用范围

| 函数类型 | 操作 |
| -------- | ---- |
| InCore / outlined non-builtin callee | 参数/函数体重写（模式 1、3、4、5） |
| Orchestration / Opaque | 调用点重写（模式 1、2、5） |
| Group | 不变 |
