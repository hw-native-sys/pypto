# RFC：支持 InCore Load 的显式 GM 子视图

## 1. 状态与设计决策

**状态：当前 worktree 已实现；设备数值与物理 DMA 验收仍待完成。**
创建日期：2026-10-22；设计更新：2026-10-26。关联需求：[issue #3042](https://github.com/hw-native-sys/pypto/issues/3042)。
分析基线：`acc125cec76c06f3f629bd4274b9a51e102a4b06`。
英文版本为真源，本文与其保持一致。

完整 API → IR → pass → codegen 示例、可运行 PTO fixture 及验证状态见
[完整示例](gm-subview-worked-example.md)。

实现新增显式 `pl.tensor.subview` API，以下选择定义当前实现契约：

1. 复用 `TensorType` 和 `TensorView`；在 `TensorView` 中增加明确的寻址种类，
   不增加新的 Tensor 类型。现有 canonical packed view 与 affine subview 必须能
   被 verifier 和消费者区分。
2. 将公共算子定义为 N 维仿射 GM 视图。第一版可执行 backend profile 可以更窄
   （只读、函数内、ND、正步长），但 profile 不能重新定义算子的语义契约。
3. 保持现有 `slice` 行为。显式 GM 描述符构造与隐式 Tensor-to-Tile 物化是不同操作。
4. 地址起点由定义算子和 codegen 描述符状态表达，不给 `TensorView` 新增 offset 字段。
   stride 和寻址种类存放在结果类型中。
5. 输入输出限定普通 Tensor；必须证明严格包含关系，未知则拒绝，包括没有可用范围事实的动态几何量。
6. 不承诺任意步长 DMA，也不通过隐式读取大包络模拟该能力。原生路径必须经过
   backend 校验，并提供独立的硬件证据。

公共算子有意比第一版 A2/A3 lowering profile 更宽。backend 可以因为 DMA 或
layout 契约无法表示一个语义合法的视图而拒绝它，但必须指出不支持的维度、dtype
或 memory space，不能静默物化一个更大的包络。

## 2. 问题与已验证的现状

需求是将 GM 地址描述与数据搬运分离。视图可以跨越距离很远的行，片上只保存显式
load 得到的结果 Tile。

Issue 原始模式：

```python
span = pl.tensor.slice(cache, [2 * delta, 512], [lo, 0])
pitched = pl.tensor.view(span, shape=[2, delta * 512])
pair = pl.load(pitched, [0, 0], [2, 512])
```

当前基线把 GM 输入上的 `tensor.slice` 转成 `tile.load`，把 Tile 输入上的 slice
转成 `tile.slice`。`tensor.view` 只有在改写后的输入仍是 Tensor-like 时才能透传。
原始模式会报错：

```text
tensor.view in an InCore function requires a GM Tensor input that remains tensor-like
through ConvertTensorToTileOps; viewing the result of an op lowered to Tile is not supported
```

单独执行 pass 的 `slice -> load` 复现打印了以下函数体片段，已省略输出 store 和签名：

```python
# Before ConvertTensorToTileOps
span: pl.Tensor[[4, 512], pl.BF16] = pl.tensor.slice(x, [4, 512], [1, 0])
pair: pl.Tile[[2, 512], pl.BF16] = pl.tile.load(span, [0, 0], [2, 512], [2, 512])

# After ConvertTensorToTileOps: observed, invalid load input
span__tile: pl.Tile[[4, 512], pl.BF16, pl.Mem.Vec] = pl.tile.load(
    x, [1, 0], [4, 512], [4, 512], target_memory=pl.Mem.Vec
)
pair: pl.Tile[[2, 512], pl.BF16] = pl.tile.load(span__tile, [0, 0], [2, 512], [2, 512])
```

第二个 load 要求 GM Tensor，却收到了 Tile。单独调用 pass 没有立即失败，因为
Call 参数替换不会重新执行算子类型推导。这不是“两次 DMA 成功执行”的证据，
也不能据此推断最终片上分配量。根 Tensor 的 `view -> load` 可以通过同一转换测试。

现有预扫描会分析消费端内存空间需求、预测 Tile 值、写入安全性和部分 GM 分配来源，
但没有从显式 load 反向保留任意视图链。默认流水中 `OptimizeOrchTensors`
位于该转换之后。本 RFC 采用显式算子语义，不改变全局 materialization 策略。

仓库依据：

- `src/ir/transforms/op_conversion_registry.cpp`：`RegisterMemoryOps` 的 slice 规则。
- `src/ir/transforms/convert_tensor_to_tile_ops_pass.cpp`：`TensorToTileMutator`、
  `ConsumerSpaceCollector`、`TensorConversionAnalysis`、`TransformIncoreFunction`。
- `src/ir/transforms/mutator.cpp`：Call 参数替换。
- `src/ir/op/tile_ops/memory.cpp`：`DeduceTileLoadType` 的 GM 输入契约。
- `python/pypto/ir/pass_manager.py`：outlining、转换及后续 pass 的顺序。

后续示例均为拟实现行为，不是当前编译器已经生成的输出。

## 3. 目标与非目标

必须实现：

- subview 是零拷贝别名（zero-copy alias）。构造时没有 GM/UB 数据分配和 payload
  读取；允许产生描述符标量运算。
- 设备端产生的 offset、行步长保持为设备端运行时值，不回读 host。
- 嵌套 subview 保留父视图 pitch、字节起点、有效数据和根 allocation 的生命周期。
- load 只为结果 Tile 分配片上空间，不按 subview 的地址包络分配。
- 保持现有 slice、真实 Tile 拒绝规则、packed-view 校验和已有程序语义。
- 必须证明每个 subview 都包含于直接父窗口；无法证明边界则编译拒绝，不能作为调用方前置条件。

在父窗口包含关系可证明的前提下，语义上支持 rank >= 1、每个维度正且非零的 step、
符号整数 offset/step/shape，
以及能够用每维一个显式 stride 表达逻辑元素坐标到地址的仿射映射的源 layout。
NZ/MX 等分块或 swizzle 布局，除非能证明选中逻辑域可由该映射表示，否则拒绝；
仅知道 layout 名称并不足够。本 RFC 中它是只读的。
零 step（广播同一个源元素）、负 step（反向访问）、padding/clamp、drop-dims
以及经别名写入属于不同操作，因为它们改变 validity、alias effect 或硬件契约；
本算子明确拒绝这些情况，不给它们静默赋予另一种含义。

输入和结果都仅限普通 `TensorType`。拒绝 `DistributedTensor`（包括其本地存储）、
远端/collective view、host tensor、Tile 值和编译器拥有的片上 allocation。
类型推导和参数改写后都必须精确检查普通 Tensor，不能只用 `AsTensorTypeLike`。backend profile 可以增加更严格的 rank、layout、dtype、对齐和作用域限制。

### 3.1 能力矩阵

| 维度 | API 语义契约 | 第一版可执行 profile |
| ---- | ------------ | -------------------- |
| Rank | 任意 rank >= 1；三个 geometry 序列 rank 相同 | 二维 ND |
| Offset | 每维整数 scalar，可静态或可证明有界的设备动态 | 可证明有界的设备动态行 offset；其他维度待探针确认 |
| Step | 每维可证明正且非零的整数 scalar，嵌套时组合 | 可证明为正且有界的动态行 step；列 step 为 1 |
| Shape | 可证明正的 extent，符号值必须证明包含关系 | Tile load shape 遵守现有分配规则 |
| Layout | 保留仿射元素 stride 映射；不隐式转换 layout | 带显式外层 pitch 的 packed ND |
| Dtype | 任意有地址/load 契约的 dtype | BF16 静态及有界动态 PTOAS 路径已验证；其他 dtype 拒绝 |
| Source | 仅普通 Tensor GM allocation | 调用方拥有的 Tensor GM 参数 |
| Use | 元数据 alias、显式 load、批准的只读消费者 | `subview -> load`；指定的 `set_validshape` |
| Scope | 函数内描述符；不允许几何未证明的 phi | 相同；不允许跨边界逃逸 |
| Hardware | backend 决定 native DMA 支持，不隐含语义 fallback | PTOAS v0.67 A2/A3 静态与动态路径；设备 DMA 证据待补 |

该表分开三个不能混淆的问题：IR 能否描述、pass 能否安全保留、目标能否执行
一次 native load。backend 不支持时给编译期诊断，不能因此授权更大的隐式搬运。

## 4. 公共 API 与使用示例

### 4.1 接口

```python
def subview(
    source: Tensor,
    shape: Sequence[IntLike],
    offsets: Sequence[IntLike],
    *,
    steps: Sequence[IntLike] | None = None,
) -> Tensor:
    """Create a function-local GM alias; do not load or allocate payload data."""
```

第一版只暴露 `pl.tensor.subview` 和 `pypto.ir.op.tensor.subview`，不增加
`pl.subview`、Tile 重载或 Tensor 方法。Python IR builder 将省略的 steps
归一化成与 source rank 相同的全 1 序列。拒绝 bool、浮点数、嵌套序列、rank
不一致、零 step、Tile 和 DistributedTensor source。接受运行时整数 DSL Scalar，在地址乘加之前
转换到足够宽的类型。负 step 不属于 subview，应由将来的 reverse view 操作承载。

| 参数 | 含义 |
| ---- | ---- |
| `source` | 普通 GM Tensor 或其支持的函数内描述符别名 |
| `shape` | 结果各维实际包含的采样元素数量 |
| `offsets` | 在直接父视图坐标系中的第一个采样点 |
| `steps` | 结果坐标加一时，父视图坐标前进多少；默认 1 |

所有几何量以元素为单位，不是字节。API 不要求某个特定维度 step 为 1。
描述符 shape 可以是符号表达式，但不意味着允许动态 UB 分配。
源 allocation 和调用方提供的逻辑到字节映射本身必须合法。第一版 backend profile
可以要求 `steps[1] == 1`，但这属于 profile 限制，不是 API 规则。

### 4.2 动态两行需求

```python
# Accepted only where proven: rows > 2, delta > 0, lo >= 0, lo + delta < rows.
pair_view = pl.tensor.subview(
    cache, shape=[2, 512], offsets=[lo, 0], steps=[delta, 1]
)
pair = pl.load(pair_view, [0, 0], [2, 512])
```

对于 packed `[rows, 512]` BF16 源：

```text
origin = lo * 512 elements from cache's current origin
shape  = [2, 512]
stride = [delta * 512, 1]
pair[0, c] = cache[lo, c]
pair[1, c] = cache[lo + delta, c]
```

逻辑 payload 是 2048 字节；实际 UB 大小遵守 Tile 布局和对齐要求。
不存在随 delta 增长的中间数据缓冲区。旧大窗口要求 `lo + 2 * delta <= rows`，
新视图只要求 `lo + delta < rows`。如果原始索引为 a、b，取 lo、delta 会按行号
排序；a 大于 b 时，消费端需要恢复原始顺序。a 等于 b 时使用单行 load 并复用结果，
首版不支持零步长。

### 4.3 部分读取、嵌套与 slice

```python
window = pl.tensor.subview(x, [16, 512], [base_row, 0])
tile = pl.load(window, [2, 64], [4, 128])

selected = pl.tensor.subview(window, [2, 128], [3, 64], steps=[4, 1])
pair = pl.load(selected, [0, 0], [2, 128])
# pair rows: x[base_row + 3, 64:192], x[base_row + 7, 64:192]

block = pl.tensor.slice(window, [4, 128], [2, 64])
result = pl.add(block, block)  # existing implicit Tile materialization
```

支持 `subview -> slice -> compute`；`subview -> subview -> load` 全程保持 GM
身份。InCore `slice -> subview` 在 slice 已变为 Tile 时拒绝。
不把 `slice -> load` 自动解释成描述符链；为改写后收到 Tile 输入的 load 增加明确诊断。

### 4.4 view 与其他消费者

```python
whole_rows = pl.tensor.subview(x, [4, 512], [base_row, 0])
reshaped = pl.tensor.view(whole_rows, shape=[2, 1024])
pair = pl.load(reshaped, [0, 0], [2, 512])
# Reads original rows base_row and base_row + 2.
```

仅当选中的父视图行是 packed 时成立。`[N,512]` 父张量的 `[4,256]` 列窗口
保留 `[512,1]` stride，不能直接 reshape 成 `[2,512]`。带行采样步长的视图
同样不能在 reshape 时丢弃 pitch。符号目标 shape 必须在上下文中证明元素总数相等；
普通 view builder 延迟等量校验的行为不能用于扩大借用窗口。

| 消费者 | 第一版契约 |
| ------ | ---------- |
| `dim`、普通 SSA alias、嵌套 `subview` | 仅处理元数据，保留 GM 身份 |
| `view` | 保留起点；沿用现有 packed source 和 validity 限制 |
| `set_validshape` | 只能收窄有效区域，保留 pitch 和来源 |
| `slice`、支持的 Tensor 计算 | 可以在消费点加载，但不能全局把 GM 视图替换成 Tile |
| `load` | 按描述符读取，结果是新 Tile |
| 经别名执行 `store`、`assemble`、标量写或其他 writer | 首版拒绝 |
| `reshape`、`transpose`、`reinterpret_view` | 不新增 GM 保证；首版要求先显式 load 再调用 |
| return、未内联调用、几何变化的描述符 phi | 拒绝；在消费者作用域重新构造描述符 |

最后两行的非法消费者应在无关转换器开始改写前被拒绝。消费者 allowlist
只约束 subview 来源的别名，不约束普通 Tensor；计算结果是新值，不继承只读别名属性。

### 4.5 贯穿实现的示例与接口决策

规范静态案例从普通 packed ND BF16 Tensor [8,512] 读取第 1、4 行，写入 packed
[2,512] 输出。builder、pass、codegen 和 runtime 验收都采用同一几何信息。
完整拟议用户程序如下：

```python
import torch
import pypto.language as pl
from pypto.runtime import RunConfig


@pl.jit
def copy_selected_rows(cache: pl.Tensor, output: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        v = pl.tensor.subview(cache, [2, 512], [1, 0], steps=[3, 1])
        pair = pl.load(v, [0, 0], [2, 512])
        pl.store(pair, [0, 0], output)
    return output


if __name__ == "__main__":
    rows = torch.arange(8).reshape(8, 1)
    cols = torch.arange(512).reshape(1, 512)
    cache = (16 * rows + cols % 16).to(torch.bfloat16)
    output = torch.zeros((2, 512), dtype=torch.bfloat16)
    copy_selected_rows(cache, output, config=RunConfig())
    torch.testing.assert_close(output, cache[[1, 4], :], rtol=0, atol=0)
```

新接口承载以下独立语义，不是 view 的另一种拼写：

| 对同一源的操作 | 含义 | 结果 / InCore 行为 |
| -------------- | ---- | ------------------ |
| tensor.subview(cache, [2,512], [1,0], steps=[3,1]) | 不读取数据，在父坐标系采样严格更小的窗口 | GM Tensor，origin 512，stride [1536,1] |
| tensor.view(cache, shape=[2,2048]) | 对原有 4096 个元素做 canonical shape/layout 重新解释 | GM Tensor，不是所选两行 |
| tensor.view(cache, shape=[2,512]) | 尝试把 4096 个元素重新解释为 1024 个 | 已有静态元素数检查拒绝 |
| tensor.slice(cache, [2,512], [1,0]) | 选连续第 1、2 行 | InCore 中 GM 输入 slice 变为 Tile load |
| load(cache, [1,0], [2,512]) | 显式搬运连续第 1、2 行 | 新 Tile |
| load(subview, [0,0], [2,512]) | 显式搬运采样后的第 1、4 行 | 新 Tile，描述符保持 GM |

本例行包含证明为 1+(2-1)*3=4<8，列包含证明为 0+(512-1)*1=511<512，
严格缩小证明为 2<8。几何量均为常量，不允许把未知边界隐藏为调用方前置条件。
把 offset 从 1 改为 5 必须拒绝，因为最后一行是 8；无约束动态 offset 也必须因
unknown 拒绝。现有 view 对部分符号 shape 等式的乐观接受，不能沿用到 subview 的包含证明。

## 5. 坐标、边界和有效区域语义

设源元素 stride 为 S，源元素起点为 B：

```text
V[j]      = source[offsets + j * steps]
V.shape   = shape
V.stride[i] = S[i] * steps[i]
V.origin  = B + sum(offsets[i] * S[i])
byte_address(V[j]) = root_pointer + sizeof(dtype) *
                     (V.origin + sum(j[i] * V.stride[i]))
```

offsets 相对于直接父视图，不是根 allocation。嵌套 subview 使用父视图真实 stride
累加起点、相乘步长，禁止只按结果 shape 重建地址。普通 alias/view 保持起点，
步长不改变 dtype。

每一维必须满足，E 为源的有效 extent，而非仅物理 shape：

```text
shape[i] > 0; steps[i] > 0; offsets[i] >= 0
offsets[i] + (shape[i] - 1) * steps[i] < E[i]
```

选中区域必须全部有效。首版不把 padding 当作有效 payload，不提供
`valid_shape`、`pad_value` 或 `clamp` 参数。部分有效的源只有在所有采样点都处于
有效前缀内时才合法。之后可以用 `set_validshape` 再收窄。

父窗口必须**严格大于并包含**新视图，计入 offset 和采样 step。整个窗口相同的
identity subview 拒绝；不缩小窗口时使用现有普通 alias/view。仅比较
`shape <= source.shape` 或总元素数是不够的。
例如源 `[8,512]`、结果 `[2,512]`、行 offset 为 1：行 step 为 6 时最后一行为 7，
合法；step 为 7 时最后一行为 8，越界，尽管结果元素更少。嵌套时先证明新视图
严格包含于直接父窗口的有效 extent，再组合 origin 和 stride；根存储尚有物理空间
不能授权访问父窗口以外的区域。

上述每维四个条件都必须**证明为真**，并且至少一个维度必须证明 `shape[i] < E[i]`。
结合正整数 step 和包含关系，可知选中坐标集是父窗口的真子集；即使包络的两个端点
相同，带采样的视图仍可能是严格子视图。证明为假时报告越界或非法几何量；
未知时报告 `tensor.subview cannot prove`，包含未通过的证明条件和定义位置。不提供 unchecked 模式，不信任调用方约定，不隐式插入
runtime assert 或 clamp。subview 派生描述符上的 `set_validshape` 收窄、
load/slice 消费者的有效读取区域也采用“必须证明，否则拒绝”的规则。严格缩小仅要求于
subview 构造；load 可以读取整个 subview，`set_validshape` 可以保持当前有效 extent。
普通 Tensor 的已有操作保持不变。

### 5.1 证明阶段与动态几何量

Wrapper 先将整数输入提升为 INDEX 再计算几何量。已有整数表达式和标量定义也必须
能证明在原始 dtype 中不溢出；窄整数乘法溢出后再提升类型不能修复，即使后续抵消
使数学上的最终 offset 看似合法，也必须拒绝。分支条件和循环边界也必须先证明
原始运算不溢出，才能提供范围事实。

类型推导负责结构检查和常量非法值拒绝。由于算子注册回调没有所在分支/循环上下文，
它可以构造临时符号描述符，但上下文 verifier 通过前不能接受编译。新增共享
subview 边界 visitor，复用 `ir::arith::Analyzer`
（`include/pypto/ir/arith/analyzer.h`），在 Tensor-to-Tile 转换前执行，并在
最终变换后、PTO 生成前再次执行。单独调用转换 pass 也必须检查；直接 codegen
入口不能绕过。允许临时构造 IR 不等于允许生成代码。

visitor 绑定支配当前位置的 SSA scalar 定义、已证明的循环边界及分支条件，
通过作用域内的 `GetConstraintContext` 管理事实，退出分支时撤销；phi 仅保留
所有输入路径都成立的事实。用 `Analyzer::CanProve` 检查每维四个条件以及至少一维严格缩小的条件。文档约定、
未经检查的用户 assertion、scalar 的 shape annotation 和无关分支不能充当证明。
采用有界局部化简和区间查询，不引入无界全局 solver；超出证明预算即未知并拒绝，
无法处理的非线性关系同样拒绝。

动态不等于不检查：常量折叠/特化值、有界循环索引、处于支配且可识别 guard 内的
设备 scalar 可以通过。任意设备读取的 lo/delta 没有范围事实时失败。
Issue 的调用可以改写为以下示意：

```python
# rows is a known positive source extent; arithmetic is widened and proven safe.
if rows > 2 and delta >= 1 and delta < rows:
    if lo >= 0 and lo < rows - delta:
        v = pl.tensor.subview(cache, [2, 512], [lo, 0], steps=[delta, 1])
        pair = pl.load(v, [0, 0], [2, 512])
        # Consume pair here; the GM descriptor does not escape the branch.
```

这是拟实现的编译器能力，不表示当前编译器已经能够证明该 guard。若 analyzer
不能推导 `lo + delta < rows` 或追踪对应 SSA 定义，该 specialization 仍然拒绝。
因此 issue 仅在动态输入边界可证明时覆盖；接受任意设备几何量会违反本次包含校验要求。
fallback 不能通过重新构造未经检查的 subview 绕过证明。

乘加前归一化地址算术，证明 geometry、组合 stride、origin 和最大选中字节 offset
不超出选定 index/pointer 位宽；常量溢出或动态范围未知均拒绝。数学不等式不能
证明机器整数回绕后的运算安全。证明使用扩宽后的操作数，lowering 保留该位宽，
直到单独证明可转换到 backend 参数位宽。DMA pitch、对齐和参数限制也必须在
backend profile 单独证明；逻辑窗口包含不等于 DMA 合法。

## 6. IR 表示与注册

复用 Call，固定四个位置参数：

```text
tensor.subview(source, shape_tuple, offsets_tuple, steps_tuple)
  -> TensorType(shape, source.dtype, TensorView(stride=derived_stride, layout=source.layout, kind=AffineSubview))
```

不新增 ObjectKind 或 TensorType 子类；`TensorView` 增加明确的寻址种类，例如
`Canonical` 和 `AffineSubview`。确定的存储契约是：
**借用描述符不携带 MemRef**。类型推导将 subview 结果的 `memref_` 设置为
`nullopt`，即使源带有 MemRef。允许的元数据后继和普通 alias 经过 InitMemRef
及地址规划后也保持缺省。不能用 0、包络大小或父 size 充当 `MemRef.size_` 哨兵。
调用方拥有的 root 保持原存储表示，不重新分配，也不因该功能改变。

首版可执行源仅允许调用方拥有的 GM 参数 root 及其支持的函数内别名，不支持编译器
新建的局部 allocation。定义 Call 是额外相对 offset 的唯一来源。通过 alias 图
恢复 root 身份，codegen 对参数当前起点指针只叠加一次相对 origin。
print/serialize round-trip 后仍从这些 Call 重建状态，不依赖未序列化的隐藏信息。
普通 Tensor 和 Tile 不因此失去其 MemRef。

建议注册结构如下，其中 deduction helper 为新实现：

```cpp
REGISTER_OP("tensor.subview")
    .set_op_category("TensorOp")
    .add_argument("source", "Ordinary GM Tensor with an affine address map")
    .add_argument("shape", "Result extents in elements")
    .add_argument("offsets", "Immediate-source starting coordinates")
    .add_argument("steps", "Immediate-source sampling steps")
    .set_arg_effect(0, ArgEffect::Read)
    .set_output_reuses_input(0)
    .set_output_memory_inherit_input()
    .f_deduce_type(DeduceTensorSubviewType);
```

`ArgEffect::Read` 是现有 Functional 层的保守分类，不表示构造操作会读取 payload。
构造没有 write channel；真正数据读取发生在 load/compute。不能为了登记 alias
而虚构写 effect。其他几何参数也按注册表要求声明 read/no-write。
结果 alias 查询必须返回 argument 0。

### 6.1 Canonical 与 affine stride

当前 `CheckCanonicalView` 继续负责 `Canonical` view：ND view 的最内层 stride
为 1，并满足现有 layout-specific 外层 stride 下界；它不要求所有 ND view 都是
物理 packed。`AffineSubview` 使用单独的 verifier：rank 和整数 stride 向量必须
匹配，每个 step 必须为正，结果 stride 必须等于父 stride 乘以该 step，因此最内层
stride 可以大于 1。`TensorViewCanonical` 按寻址种类分派，不能全局放宽 canonical
不变量。

`MaterializeTensorStrides` 必须保留显式 stride 和寻址种类。packed-only 操作仍要求
stride 等于 packed stride；affine 操作只有在消费者契约允许时才能使用非 packed 映射。
合法 affine view 不自动意味着可用于 DMA 或 reshape；这些消费者必须分别证明布局条件。

反射、结构相等、hash、binding 和序列化都包含寻址种类。旧序列化 view 默认恢复为
`Canonical`；affine view 保存并恢复其种类和显式 stride。寻址种类描述地址映射，
不是新的 layout 或 allocation。通用 Call 序列化保存新算子与参数，并增加符号 stride、
offset 及旧数据的 round-trip 测试。重新解析 post-pass dump 仍须校验源是否合法。

### 6.2 Builder 与具体 IR 契约

DSL wrapper 解包普通 Tensor 和整数 Scalar，调用 IR builder，再将 Call 包装为
Tensor。公共参数归一化后，builder 复用现有 tuple helper 与 registry 路径：

```python
from pypto.pypto_core import ir as core
from pypto.ir.utils import _to_make_tuple


def subview_ir(source, shape, offsets, steps, span):
    return core.create_op_call(
        "tensor.subview",
        [
            source,
            _to_make_tuple(shape, span),
            _to_make_tuple(offsets, span),
            _to_make_tuple(steps, span),
        ],
        {},
        span,
    )
```

新增算子完成注册后此 builder 才能执行。规范示例构造以下完整 InCore 函数体；
这是操作记法，不是当前 printer 输出：

```text
func copy_selected_rows_incore(cache: Tensor<[8,512],BF16>,
                              output: Tensor<[2,512],BF16>):
  v    = tensor.subview(cache, (2,512), (1,0), (3,1))
  pair = tile.load(v, (0,0), (2,512), (2,512))
  out  = tile.store(pair, (0,0), output)
  return out
```

| 变量 | IR 类型 / 元数据 | 地址 / 分配 |
| ---- | ---------------- | ----------- |
| cache | TensorType([8,512], BF16)，canonical ND stride [512,1] | 调用方 GM pointer |
| v | TensorType([2,512], BF16)，TensorView stride [1536,1]，layout ND，寻址种类 AffineSubview | 无 MemRef，无新 payload 分配 |
| pair | TileType([2,512], BF16)，最终 Vec | 新结果 Tile 分配 |
| out | output 的 store 结果 | 普通输出 GM alias |

builder/type 测试校验四个 operand 及上述结果元数据。第四个 IR operand 总是显式的，
包括省略 steps 时归一化出的全 1 序列。结果 alias 解析到 cache；pair 是新存储。
起点 512 从 Call 恢复，不能加入 MemRef，也不能按更小 shape 再次推导。
初始 load target 可以未设置；InferTileMemorySpace 后放入 Vec，但不能改变 GM 描述符。

## 7. 编译器实现

### 7.1 Tensor-to-Tile 转换

将 `tensor.subview` 加入显式元数据 passthrough 集合。检查改写后的源仍然是普通 GM
Tensor；替换改变几何量时重新推导描述符。不能把它注册成返回 Tile 的转换，
不能在入口 preload 它的 payload，也不能把其结果放进全局 Tensor-to-Tile 映射表。

消费端内存空间分析可以穿过 subview，为之后的 load 选择内存空间，但不能把
subview 的 GM 结果转成 Tile。普通计算通过消费点 bridging 加载视图。
同一个视图同时被显式 load 和计算使用时，它本身保持 GM，各使用点分别得到所需表示。
不能把 load 移到根存储写入之前或跨越分支边界。cache 声明必须能经 subview
别名解析到原始 GM root。

Issue 的目标 IR，以下为示意：

```python
# Before and after ConvertTensorToTileOps: this descriptor chain stays intact.
v = pl.tensor.subview(cache, [2, 512], [lo, 0], steps=[delta, 1])
t = pl.tile.load(v, [0, 0], [2, 512], [2, 512])
```

pass 可以继续增加现有输出 store 和签名绑定。对参数改写后的 passthrough
`tile.load` 增加输入契约检查，使旧 `slice -> load` 在收到 Tile 时给出源码级诊断，
而非让非法 IR 流到后端。这是有界的诊断修复，不是自动改变旧 slice 语义。

当前 `tensor.set_validshape` 无条件简单转换成 `tile.set_validshape`。
仅对来源为 subview 的 GM alias 覆盖这一决策：在转换查询和 bridging 前保留
Tensor op，结果保持 shape/stride、无 MemRef，并收窄有效 extent。
类型预测和 preload 分析必须采用同一个 provenance 判定，不能把该结果标记为 Tile。
普通 Tensor set_validshape 保持现有转换。每维请求值与父有效 extent 比较，
静态扩大或无法证明收窄均拒绝。因此明确支持
`subview -> set_validshape -> subview/load`。
backend 增加仅描述符 handler，复用当前 pointer 和 tensor-view SSA；tile.load
继续从结果类型读取有效区域。该操作不产生数据 load、tile.set_validshape 或分配。

### 7.1.1 规范示例的 pass 检查点

第 6.2 节函数体已经包含 tile.load 和 tile.store；ConvertTensorToTileOps 必须保留
该操作顺序，不能为 v 新增 preload，不能把 v 本身降成 Tile。组件调整与 before/after
断言如下：

| Pass / 组件 | 本例 before → after | 必须调整 |
| ----------- | ------------------- | -------- |
| SSA / outlining | 局部描述符和 scalar geometry 留在消费 kernel 内 | 不能把描述符变成函数参数，不能丢失必要 guard |
| 边界验证 | 待证 geometry → 行列包含和严格缩小已证明 | conversion 前与 codegen 前增加共享上下文 visitor；false/unknown 拒绝 |
| ConvertTensorToTileOps | tensor.subview + tile.load → tensor.subview + tile.load | 显式元数据透传；预测/preload 分析保留 GM 身份；校验改写后的 load source |
| ConsumerSpaceCollector | load 的 Vec 需求 → 传播到对应消费点 | 不能全局把描述符替换成 Tile |
| MaterializeTensorStrides / canonical verifier | 显式 [1536,1]、AffineSubview → 保持 | 保留 affine kind/stride，分开 canonical 和 affine 校验 |
| InitMemRef / memory planning | 借用 GM 描述符 → 无 MemRef；结果 Tile → 正常 placement | 跳过描述符分配和继承 offset 的 MemRef 构造 |
| Composite / ND/layout passes | 二维 ND geometry → 相同坐标映射 | 不重建 packed stride；不支持的改写诊断拒绝 |
| 最终 backend 检查 | 已证明 geometry → 目标可执行 load | 单独验证 dtype、layout、pitch、对齐和整数位宽 |

conversion 和 placement 分别编写 Before/Expected 结构测试：conversion 保留 v 及其
load 输入 identity；内存空间推导后 pair 位于 Vec。stride materialization 保留
[1536,1] 与 AffineSubview；InitMemRef 后 v 仍没有 MemRef，pair 按普通 Tile/Buffer
规则分配。零大小描述符 MemRef 也违反契约。这些是独立验证点，不能假设一个转换
已经完成整条流水。

### 7.2 别名身份、生命周期与只读约束

嵌套 subview 和允许的元数据操作使用共享 result-alias 查询。通过索引和 worklist
一次性追踪 alias、tuple 和控制流边上的 subview 来源。经任何派生别名写入都拒绝，
包括先对 subview 做普通 view 再写，不能仅依靠 Python wrapper 限制。
load 产生的新 Tile 不属于 GM alias。

视图不是快照。通过其他根 handle 的写入仍遵守现有顺序和竞态规则，后续 load
观察有序内存状态。无法证明精确 strided region 重叠关系时，依赖分析保守归因到
根 allocation；不能用描述符的密集元素数推断不相交。根存储生命周期覆盖所有视图使用。

在 `InitMemRef` 前为借用描述符定义和允许的 alias 后继建立一次索引。
这些 Tensor 值绕过 `ProcessNormalVar` 的新 MemRef 构造，以及 `ShareMemRefFrom`
的继承 offset 路径；重映射类型几何信息，但保持 `memref_` 缺省。
当前 HasMemRefs 要求 Tile 有 MemRef，不要求 GM Tensor 有 MemRef；保留这一区别，
并覆盖借用描述符测试。memory reuse、address allocation 和 footprint 工具不能
把描述符视为分配候选，需要存储身份时查询 root alias，不构造描述符包络。
这与普通 tensor.slice 已有 view-MemRef 行为不同。load 结果仍获得普通 MemRef
以及物理 Tile 分配。

### 7.3 控制流与作用域

在上下文边界证明通过的条件下必须支持：任一分支内构造并 load；先用 if 合并标量几何信息再构造描述符；
循环每次迭代使用变化的 offset/step 构造并消费新描述符。
描述符可在定义作用域内跨普通使用保持存活。

首版不允许把 subview 描述符作为分支/循环结果带出作用域。即使 shape 相同，
origin/stride 不同的 phi 也必须诊断，仅合并 shape 不足以保证正确。
调用方应先合并/携带标量几何信息，在 load 点重新构造视图。未内联调用、return
和编排边界同样在 codegen 前拒绝。

### 7.4 PTO codegen

维护每个 SSA 值的描述符状态：root pointer、当前元素 origin、shape、stride。
对嵌套别名 memoize，不反复全表扫描。subview offset 按直接父视图 stride 计算，
在消费函数内生成 `pto.addptr` 和 `pto.make_tensor_view`，同时登记当前起点指针
和 tensor-view SSA，供 view/load 使用。
`GetTensorBasePtr(subview)` 必须表示当前起点指针；依赖分析另行保留 root 身份。
这样已有 `tensor.view` 会保留 subview 起点，而不是错误地回到根指针。

复用 `src/backend/common/pto_ops_distributed.cpp` 现有的 inline addptr +
make_tensor_view 模式，不引入返回描述符的 MLIR helper。以下是示意，不是已汇编证据：

```text
row_stride = wide_index(delta) * 512
origin     = wide_index(lo) * 512
p          = pto.addptr cache_pointer, origin
v          = pto.make_tensor_view p, shape=[2,512], strides=[row_stride,1]
window     = pto.partition_view v, offsets=[0,0], sizes=[2,512]
             pto.tload window -> tile[2,512]
```

一条 PTO tload 不等于已证明一条硬件 DMA。本 RFC 不凭空规定硬件限制或指令保证。

### 7.4.1 Lowering 阶段与最终 PTO 产物

`tensor.subview` 一直保留为显式 GM 描述符算子，直到 PTO codegen；不能转换成
`tile.slice`、`tile.load`，也不新增 PTO dialect op。现有流水需要以下契约：

| 阶段 | 必须处理 |
| ---- | -------- |
| SSA / 化简 / outlining | geometry 绑定到 SSA 定义；描述符留在消费 kernel 内。outlined kernel 外的 guard 不自动成为内部事实，必须在内部保留实际支配的 guard，否则拒绝。 |
| Tensor-to-Tile 前 | 证明父窗口包含关系；subview 保持普通 GM Tensor；仅在 slice/compute 消费点转换成 load。 |
| Composite lowering / ND flatten / layout legalization | 保留原源坐标的仿射映射。首版二维 ND 无需 flatten；不支持的 rank/layout 变化在改写前拒绝，不能重新推导 packed stride。 |
| Stride materialization / InitMemRef / memory planning | 保留 origin、显式 stride 和寻址种类；不分配借用描述符。仅 load 结果以 Tile 身份参与选定的 memory planner。 |
| 最终校验 / backend 能力判定 | 参数替换和控制流变化后重新证明包含关系；证明 dtype/layout、pitch、对齐和位宽限制。不能信任早期 IR 的缓存证明结果。 |
| PTO 生成 / PTOAS | 生成标量地址运算和标准描述符 op，之后复用 partition/tload 路径；使用安装的目标工具链汇编最终 PTO 并检查其 lowering。 |

仅扩展 codegen 查询表不够：新算子需要注册 backend handler，同时绑定结果 pointer
和 tensor-view。`GetTensorBasePtr(v)` 是 v 当前起点指针，根 allocation key 单独
保留；`tile.load` 不能重复加 origin。load 的 offset 相对 v，`pto.partition_view`
按 v 的采样 stride 应用 offset。load offset 为 q 时，起点为
`B + sum(offset*S) + sum(q*S*step)`。`set_validshape` 只改变有效读取大小，
不改变 pointer 或 pitch。嵌套 subview 的父 stride 只组合一次；可选折叠仅在证明
这些地址等式和边界后移除冗余描述符。

静态规范示例的完整目标 PTO 模块如下。它是手写、可执行的 backend fixture，
不是拟议 API 的当前输出。常量、分配、输出 store 和 return 都已定义：

```text
// RFC #3042 handwritten backend probe; real API output is tested separately.
// Input: packed BF16 [8,512]. Select rows 1 and 4, then copy to output [2,512].
!pair_tile = !pto.tile_buf<loc=vec, dtype=bf16, rows=2, cols=512, v_row=2, v_col=512, blayout=row_major, slayout=none_box, fractal=512, pad=0>
module attributes {pto.target_arch = "a2a3"} {
  func.func @subview_static(%cache: !pto.ptr<bf16>, %out: !pto.ptr<bf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c512 = arith.constant 512 : index
    %c1536 = arith.constant 1536 : index
    %p = pto.addptr %cache, %c512 : !pto.ptr<bf16> -> !pto.ptr<bf16>
    %v = pto.make_tensor_view %p, shape = [%c2, %c512], strides = [%c1536, %c1] {layout = #pto.layout<nd>} : !pto.tensor_view<?x?xbf16>
    %read = pto.partition_view %v, offsets = [%c0, %c0], sizes = [%c2, %c512] : !pto.tensor_view<?x?xbf16> -> !pto.partition_tensor_view<2x512xbf16>
    %pair = pto.alloc_tile : !pair_tile
    pto.tload ins(%read : !pto.partition_tensor_view<2x512xbf16>) outs(%pair : !pair_tile)
    %out_view = pto.make_tensor_view %out, shape = [%c2, %c512], strides = [%c512, %c1] {layout = #pto.layout<nd>} : !pto.tensor_view<?x?xbf16>
    %write = pto.partition_view %out_view, offsets = [%c0, %c0], sizes = [%c2, %c512] : !pto.tensor_view<?x?xbf16> -> !pto.partition_tensor_view<2x512xbf16>
    pto.tstore ins(%pair : !pair_tile) outs(%write : !pto.partition_tensor_view<2x512xbf16>)
    return
  }
}
```

fixture 维护于 [gm_subview_static.pto](../../../../tests/fixtures/gm_subview_static.pto)。
handler 必须为 v 调用 RegisterBasePtr(v, current_pointer)、RegisterTensorView 和
RegisterVarToMlir，使已有 view/load handler 取得描述符，而不是重建 packed root view。
最终 store 指向 output，不写入只读 v。常量可以折叠，但 origin 512 和行 pitch 1536
必须在语义上保持精确，不能再次叠加 origin。

产物具体增加当前起点 pointer、显式采样 pitch、tensor-view/partition 和标量运算
（常量情况可折叠）。`AffineSubview` 是内部 IR 区分，不输出为新的 PTO layout 或
属性。源 GM pointer ABI 不变；outlining 若需要捕获 geometry，可以使用普通 scalar
kernel 参数。kernel ABI 和 workspace 不增加描述符 buffer。本次读取对应的新数据
分配只有结果 Tile：`[2,512]` BF16 逻辑数据为 2048 字节，物理大小沿用已有布局和
对齐规则。不能生成 `[2*delta,512]` GM/UB 临时数据或包络 tload。即使根参数已有
自己的 tensor view，新描述符 op 仍可能保留。

partition extent 是请求的有效 load 区域，不是父窗口或地址包络。继续传递根存储的
cache policy/L2 offset 和已有 load 属性；geometry offset 与 bypass 的字节 offset
必须分开。无关 kernel 保持原 PTO 行为。测试需检查嵌套/部分 load 产物、真实采样
stride、pointer origin 组合、没有描述符数据分配及 Tile 物理大小。PTOAS 汇编成功
必须独立测试；一条文本 tload 与最终 DMA 数量/性能验收仍是不同证据。

### 7.5 原生 load 支持与 fallback 策略

首个可执行路径面向 A2/A3 的二维 Vec load、连续列，以当前安装的 PTOAS v0.67
进行验证。保留现有 target-memory 选择；如果 subview load 被请求或推导到尚未验证的
内存空间/backend，使用源码级 unsupported-path 诊断拒绝。

通过已注册 backend operator 或 PassContext backend handler 查询架构约束，
包括行字节对齐、行数、源 pitch/gap 范围、index/DMA 参数位宽。不能把架构常量
放进 Python API 或通用 IR pass。原生路径实施前必须从 backend/toolchain 证据
确定精确限制，本设计不猜测 gap 的数值上限。

动态值必须利用包含证明的同一作用域事实，证明原生路径的范围和对齐条件；
backend 条件未知则拒绝 specialization。分别测试可识别的设备 guard 和调用方
手写的两次单行 load，每条路径都必须有合法读取边界。不支持动态 pitch 的 backend
拒绝该 specialization，不能静默截断 pitch。
首版没有自动 fallback、隐藏 gather kernel 或整段大窗口搬运。静态不支持的几何量
在编译时拒绝。通用 API 不保证一条 DMA；关闭 issue 还必须有规定范围内的原生
单 DMA 证据。

## 8. 影响文件与实施顺序

下表是后续实现的目标或审计点，不是本 RFC 已修改的代码。不新增全局开关。

| 顺序 | 路径 | 工作 |
| ---- | ---- | ---- |
| 1 | `src/ir/op/tensor_ops/memory.cpp`；`src/ir/op/tensor_ops/transform.cpp`；`include/pypto/ir/type_inference.h`；`src/ir/op/type_inference.cpp` | 类型推导、几何归一化、有效性/溢出、set_validshape 收窄和注册 |
| 2 | `python/pypto/ir/op/tensor_ops.py`；`python/pypto/language/op/tensor_ops.py` | builder、export、归一化、诊断、docstring |
| 3 | `include/pypto/ir/transforms/utils/tensor_view_semantics.h`；`src/ir/verifier/verify_tensor_view_canonical.cpp`；`src/ir/transforms/materialize_tensor_strides_pass.cpp` | 分开 canonical/affine 校验；保留显式 stride 和寻址种类 |
| 4 | `src/ir/transforms/convert_tensor_to_tile_ops_pass.cpp`；`src/ir/transforms/op_conversion_registry.cpp` | GM 透传、消费点 bridging、load 输入检查、来源限制 |
| 5 | `src/ir/transforms/utils/result_alias_utils.cpp`；`src/ir/transforms/utils/op_predicates.cpp`；`src/ir/transforms/utils/return_lineage_utils.cpp`；`src/ir/transforms/init_memref.cpp` | alias/lifetime/只读审计；借用描述符不携带 MemRef |
| 6 | `include/pypto/ir/transforms/utils/memref_utils.h`；`src/ir/transforms/memory_reuse_pass.cpp`；`src/ir/transforms/allocate_memory_addr_pass.cpp`；`src/ir/transforms/optimize_orch_tensors_pass.cpp` | 不分配借用描述符；保留 origin/pitch、不做非法外提 |
| 7 | 新增 `src/ir/verifier/verify_tensor_subview.cpp`；`include/pypto/ir/arith/analyzer.h`；`src/ir/transforms/auto_derive_task_dependencies_pass.cpp`；`src/ir/verifier/type_check_pass.cpp`；`src/ir/verifier/warning_in_param_written.cpp` | 转换/codegen 入口的上下文包含证明；根依赖和诊断 |
| 8 | `src/backend/common/pto_ops_memory.cpp`；`src/codegen/pto/pto_codegen.cpp`；`include/pypto/codegen/pto/pto_codegen.h`；`include/pypto/backend/common/backend_handler.h`；backend handler 实现 | 正确 origin、set_validshape alias handler、backend 约束、原生 Vec load |
| 9 | `python/pypto/debug/torch_codegen.py` | 使用正步长 Python/Torch 切片实现 reference，保留别名和父 pitch |
| 10 | 现有 IR printer/serialization、binding 和 stub 路径 | 寻址种类 enum/字段的 binding、反射、打印和兼容序列化 |
| 11 | 测试及中英算子/pass/codegen 文档 | API、正确性、分配和硬件验收证据 |

Python operator 通过通用 `create_op_call` 路径构造，不要求专用 nanobind 函数。
新的寻址种类 enum 和 TensorView 字段必须同步更新 `include/pypto/ir/type.h`、
`python/bindings/modules/ir.cpp` 与 `python/pypto/pypto_core/ir.pyi`，
同时更新反射、serializer/deserializer 和类型 printer。
printer 保留第四个 IR operand，DSL 可打印为 `steps=`；序列化识别新注册算子，
不增加 node kind。

新分析保持 O(N+E)，使用有序映射时不超过 O(N log N)。定义和 alias use 只建索引一次，
控制流/alias 边用有界 worklist；不能为每个消费者重新遍历整个 alias 链。
复用表达式 DAG，避免反复展开符号运算树。

实施首先执行 **Gate 0：最小 PTOAS/backend 可行性探针**（静态 Gate 0a 与动态/目标
Gate 0b，见 9.1 节），放在 tests 下并使用
当前 worktree/toolchain。确定 BF16/FP16/FP32、动态/静态 pitch、对齐、行数、
参数位宽和可用 simulator/device trace 工具的支持矩阵。关闭 issue 必须支持
BF16 可证明有界的动态 pitch；可选 dtype/target 不支持时明确拒绝并记录。探针失败时先修改
backend 设计，不能直接进入原生路径实现。
之后增加语义/负例测试，实现 deduction/API，再实现转换和来源检查、保留存储身份、
实现已验证的 backend 描述符，最后执行端到端测试并更新参考文档。
在可执行链路贯通前，不发布半实现的公共 API。

## 9. 验证与验收

建议新增或更新的测试：

| 层次/路径 | 已实现覆盖 |
| --------- | ---------- |
| `tests/ut/ir/operators/test_tensor_subview.py` | 默认 steps、泛化 rank/内层 step、父 pitch/有效范围、非法几何与源类型、序列化、整数拓宽 |
| `tests/ut/codegen/test_tensor_subview.py` | DSL outlining、独立转换/codegen 安全门禁、有 guard 的动态 shape/offset/step、scalar phi/循环、嵌套起点和 view alias、slice 消费、分配、CPU reference、真实 PTOAS A2/A3 编译 |
| `tests/ut/codegen/test_gm_subview_rfc_fixture.py` | 独立手写静态/动态 PTOAS A2/A3 fixture |
| 现有 operator acceptance 和 backend inventory 测试 | 算子分类、排除 Buffer IR、backend 注册 |
| `tests/st/runtime/ops/test_tensor_subview.py` | 模拟器/设备 runtime harness 行选择用例；不进行数值执行时可验证 codegen-only |

输入用行列坐标编码，防止错误行列恰好相等。覆盖 delta=1、大合法 gap、最后合法行、
原始索引反序、相同行 fallback、列 offset、嵌套视图，同一 specialization 内的
两条设备分支和多次循环。非法索引 fallback 不得访问非法内存。
增加经 view/tuple alias 写入、非法作用域逃逸、真实 computed Tile source、
溢出与对齐边界测试。增加新旧窗口相等时拒绝、shape 更小但 offset/step 越界、
scalar 边界未知、有 guard 分支通过而无 guard 兄弟分支拒绝、phi/outlining 丢失事实、
以及变换使早期证明失效后必须拒绝的测试。

结果对比 `cache[[lo, lo + delta], :512]`。检查实际物理 `[2,512]` Tile 的 UB
分配产物，确保没有随 delta 增长的分配或搬运。必须通过指令、模拟器或真机 trace
证明原生条件下的一次 strided DMA，不能只检查 PTO 文本。无法获得证据时，
分别报告语义支持与性能证据，issue 的单 DMA 验收项保持未完成。

在当前 worktree 按 testing 技能执行：

```bash
source .claude/skills/testing/load-env.sh
cmake --build build --parallel "$PYPTO_BUILD_JOBS"
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest tests/ut/ -n "$PYPTO_TEST_JOBS" -v
python -m pytest tests/st/runtime/ops/test_tensor_subview.py --codegen-only -v
python -m pytest tests/st/runtime/ops/test_tensor_subview.py -v
```

上述 runtime 测试已落盘。数值执行需要匹配的固定 runtime 版本和模拟器/设备环境；
codegen-only 不能证明数值正确性。不在本机执行完整 ST suite。记录准确 toolchain/backend 版本与未完成的硬件检查。
实现阶段按仓库规则对相关路径执行 lint、类型检查和 pre-commit。

发布验收：

- [ ] 原 slice/view 程序语义保持；非法 Tile-input load 提前诊断。
- [ ] subview GM 描述符通过转换，不产生构造时 payload load 或数据分配。
- [ ] 嵌套 origin/pitch 经 view、显式 load 后仍然正确。
- [ ] 仅接受边界可证明的设备端动态几何量，未知则拒绝，无 host 回读。
- [ ] 分支/循环作用域内使用正确；不支持的逃逸和 writer 有明确诊断。
- [ ] 只分配结果 Tile，不物化包络。
- [ ] backend 限制具备真实 PTOAS/目标证据并写入文档。
- [ ] 原生两行路径满足单 DMA，或明确保持该验收项未完成。
- [ ] 中英文档、打印、序列化和 reference 执行一致。

### 9.1 从规范示例推导的证据门槛

Gate 0 分成独立跟踪的两部分。**Gate 0a** 为静态 BF16 backend fixture：
PTOAS 0.67 在 A2、A3 上接受该模块，产物含一个 [2,512] BF16 Vec Tile、
源行 stride 1536、一个 TLOAD 和一个 TSTORE。2048 字节逻辑 payload 与
4096 字节选中地址包络必须区分。**Gate 0b**：有界动态 BF16 pitch 已通过
PTOAS 0.67 A2/A3 编译。当前 native load 限定 rank-2 ND BF16 Vec、内层 stride=1、
物理行数最多 4095、有效/物理行按 32 字节对齐、index/地址不超过有符号 32 位、
源 gap 字节数不超过无符号 32 位；依据安装的 PTO-ISA ND-to-ND load 实现。
不支持或无法证明的条件拒绝。其他 dtype/target 及物理 DMA 证据仍待补。

已有 fixture 测试为
[test_gm_subview_rfc_fixture.py](../../../../tests/ut/codegen/test_gm_subview_rfc_fixture.py)，
它测试手写静态/动态 PTO。
[test_tensor_subview.py](../../../../tests/ut/codegen/test_tensor_subview.py)
通过真实 API 和默认 pass 检查分配及 load，并用 PTOAS 编译 A2/A3 静态与有界动态
产物（level3，因为默认分配 pass 提供显式 Tile 地址）。设备验收逐元素比较 output 与 cache[[1,4],:]。
完整 CPU reference 和命令见[示例](gm-subview-worked-example.md#5-独立数据地址验证与证据)。
API/pass、CPU reference 数值、当前 worktree 构建及 PTOAS 编译由实现测试覆盖。
设备数值和单硬件 DMA 仍未验证；一个 TLOAD 不能证明物理 DMA 次数。
2026-10-23 的构建/pytest 阻塞已在当前 worktree 解决。

## 10. 替代方案与待验证项

| 方案 | 未选原因 |
| ---- | -------- |
| `slice(..., gm_view=True)` | 用属性切换同一算子的 materialization 契约 |
| 根据显式 load 消费者全局延迟 slice | 改变已有语义和混合使用/控制流决策，应独立设计 |
| 直接给 load 增加任意 `strides=` | 缺少可组合逻辑窗口和有界父坐标映射 |
| 通用 `as_strided` 或新 Tensor 类型 | 不变量、安全与 ABI 范围大于本 issue 所需 |
| 总是读取整个包络 | 搬运和 UB 随 gap 增长，违背目标 |

API、IR 和作用域选择由本设计明确；builder 自行决定 helper 名称、现有文件内部放置
和诊断措辞。关闭原性能需求前，除上述编译器验证外，还必须补充设备数值正确性
和单 DMA 证据。toolchain 探测失败时应明确修改 backend 方案，
不能静默弱化语义或替换成大窗口 load。
