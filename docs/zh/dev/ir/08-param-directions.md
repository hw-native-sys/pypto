# 参数方向推导（Parameter Direction Inference）

`Function::param_directions_` 记录每个参数是被读、被写，还是二者皆有——`In`、
`Out`、`InOut`。下游几乎一切都依赖它：依赖分析据此发出 RAW 边，分布式 codegen
据此标记每个 per-rank 的芯片派发，host ABI 据此决定哪些 buffer 必须由调用方分配。

**它出错时是静默的。** 一个声明为 `In` 却被函数体写入的参数，在编译期既不报错、
也不产生错误数值——它需要的那条依赖边只是从未被发出，程序在设备上表现为竞争或
死锁。

本页把整条链放在一处。每个阶段各有专页；这里说明**有哪些阶段、各自回答什么问题、
答案从哪里来**。

## 唯一真相源

下面每个阶段问的都是同一个问题——*这次调用是否写了该实参所指的 buffer？*——
而且都从同一处取答案：算子自身在注册表上的声明。

```cpp
REGISTER_OP("tile.mscatter")
    .set_arg_effect(2, ArgEffect::Write)        // 固定效应
    .set_write_channel(WriteChannel::Dma);

REGISTER_OP("tile.mgather")
    .set_arg_effect(2, [](const auto& kwargs) { // 依 kwarg 而定：`scratch`
      ...                                       // 仅在 Mat elem 模式下被写
      return mat_output && elem_mode ? ArgEffect::Write : ArgEffect::Read;
    });

REGISTER_OP("system.set_ffts")
    .no_arg_writes();                           // "不写任何参数"必须显式声明
```

完整的声明面见[算子系统 — 参数效应](05-operators.md#参数效应argument-effects)。
查询面是 `GetArgEffect(i, kwargs)`、`HasDeclaredArgEffect(i)`、`WritesAnyArg()`
与 `GetOutputReusesInputArg()`。

过去并非如此。写入集合曾是各分析内部手工维护的算子名清单，而这些清单彼此不一致
——`pld.system.notify` 就这样带着一个没有任何分析知道的信号写入上了生产（#2391），
`tile.mscatter` 在其效应被声明之前也处于同样状态。

### 注册门禁覆盖什么、不覆盖什么

`OpRegistry::ValidateArgEffects()` 在注册期运行，拒绝两种形态：

- 声明了 `set_output_reuses_input(N)` 却没有对实参 `N` 分类——复用契约意味着对
  该实参必须有一个判断；
- 声明了 write channel 却不通过任何实参写入——channel 描述的是算子**如何**写，
  没有写入的 channel 要么是多余声明，要么是漏了声明。

**两者皆无**的算子——既无复用契约、也无 write channel——两道门都不碰。在那里漏掉
`set_arg_effect` 依旧是静默的。这正是最初那类故障，且尚未封闭，见[已知限制](#已知限制)。

## 整条链

| 阶段 | Pass | 回答的问题 |
| ---- | ---- | ---------- |
| [Outline](#1-outlinepass-79) | 7 / 8 / 9 | 我**正在创建**的这个函数，参数方向是什么？ |
| [调用方传播](#2-调用方传播pass-10) | 10 | 被调方写了这个实参——它背后我自己的参数是否要变成 `Out`？ |
| [Wrapper 恢复与调用点](#3-wrapper-恢复与调用点pass-37) | 37 | 每个 wrapper 的**有效**签名是什么？每个调用点实参的方向是什么？ |
| [一致性警告](#4-一致性警告postpipeline) | PostPipeline | 是否还有参数声明为 `In` 却被自身函数体写入？ |

### 1. Outline（pass 7–9）

`ScopeOutliner::InferParamDirections` 为刚被 outline 出来的 scope 函数确定签名。
四个步骤，每一步都只是访问集合的**下界**——任何一步都不得覆盖另一步的观测：

| 步骤 | 证据 |
| ---- | ---- |
| 0 | `ParamReadCollector` 扫描函数体中的读取 |
| 1a | 导出的 store 目标即写入 |
| 1b | `CallWriteTargets`——注册表声明的写入 |
| 2 | 内层被调函数声明的槽位 |

其中两处细节承载了大部分正确性。

**步骤 0 会跳过被调方纯覆写的槽位。** 把 capture 交给只写槽位并不会把数据搬进本
作用域，因此不算读取。指明这些槽位的是两份声明，每种被调方各一：内建算子的
`ArgEffect::Write`，以及用户函数的 `ParamDirection::Out`。

**步骤 2 不按 `In < Out < InOut` 合并。** `In` 是初始化时"尚无证据"的地板，因此它
不能同时表示"有人读过"——那样理解会把每个只写 capture 提升为 `InOut`，即把同一个
`pl.Out` 张量的互不相交的 per-rank 切片变成跨 rank 依赖的虚假读取（issue #2415）。
被调函数的槽位被累积为两个独立标志——`In`/`InOut` 记为读、`Out`/`InOut` 记为写
——最后一次性推导出方向。

参见 [Outline InCore Scopes](../passes/08-outline_incore_scopes.md)。

### 2. 调用方传播（pass 10）

当调用方把自己的参数转发给被调函数会写入的槽位时，`ConvertTensorToTileOps` 的
第 3 阶段把该参数提升为 `Out`/`InOut`。

实参**很少**就是参数本身。循环携带的张量以 `IterArg` 的身份到达，其值是参数的
buffer：

```python
acc = dst
for _ in pl.range(4):
    acc = self.kernel(x, acc)     # kernel(x, out: pl.Out[...])
```

因此在查参数表之前，实参会先经 `BufferRootCollector` 解析到它所属的 buffer。实参
本身就是参数时，它解析到自己——所以这是对指针恒等查找的**推广**，而非替换。

### 3. Wrapper 恢复与调用点（pass 37）

`DeriveCallDirections` 分两部分。

**Phase 0——把每个 Group/Spmd wrapper 的有效方向写回它自己的签名。** wrapper 把
参数 1:1 转发给内层 kernel，但它自己的 `param_directions_` 可能仍对一个内层 kernel
会写的参数读作 `In`：outliner 是按它提取出的函数体推导的，而 `ExpandMixedKernel` /
`SplitVectorKernel` 之后围绕新拆分出的被调函数重建 wrapper 函数体时，并不回头修改
签名。

这个恢复动作过去在四处被各自即时重算——本 pass、`AutoDeriveTaskDependencies`、
`CallDirectionsResolved` 验证器，以及 orchestration codegen。做一次并存入 IR，使
每个消费者只有一个依据：`callee->param_directions_`。

**然后是调用点。** 每个实参获得一个 `ArgDirection`——`Input`、`Output`、
`OutputExisting`、`InOut`、`NoDep`、`Scalar`——这才是依赖分析与 codegen 实际消费的
东西。

参见 [Derive Call Directions](../passes/38-derive_call_directions.md)。

### 4. 一致性警告（PostPipeline）

`DiagnosticCheck::InParamWritten` 报告仍声明为 `In` 却被自身函数体写入的参数。它读
的是推导所读的同样两份声明——注册表效应与被调函数的 `param_directions_`——并报告它们
与该参数相矛盾之处。

它是**警告而非 `IRProperty`**，且这是被迫而非选择：该检查必须在
`DeriveCallDirections`（pass 37）之后运行，而 `InitMemRef`（pass 31）作废了
`SSAForm` 且此后无人重建，因此流水线中不存在既在 pass 37 之后、又处于 SSA 形式的
位置。它的 buffer lineage 在控制流上因而是尽力而为的。

参见 [验证器 — InParamWritten](../passes/99-verifier.md#inparamwritten)。

## 共享分析

有三个 helper 被多个阶段共同读取——这正是各阶段不会彼此漂移的原因：

| Helper | 回答 |
| ------ | ---- |
| `BufferRootCollector` | 这个变量属于哪块 buffer？ |
| `ResultAliasedArgIndex` | 这次调用的结果指向哪个实参的 buffer？ |
| `op_predicates::IsBufferAliasingViewOp` | 这是否是对输入的零拷贝 view？（`OutputMemoryInheritsInput() && IsInplaceSafe()`） |

第三个排除了 `tile.transpose`：它继承内存**空间**，但会把数据置换进一块全新
buffer（`pto.ttrans` 注册为 `not_inplace_safe()`），因此其输出并不别名输入。

## 已知限制

有两处缺口是被记录下来而非被封闭的。第一处只会漏报；第二处**两个方向都会出错**
——这一点需要写明，因为本页早先的版本声称并非如此。

**缺失的声明不可见。** 每个阶段都读注册表，因此从未声明效应的算子在任何地方都不
贡献写入，一致性警告同样看不见它——它读的正是那份缺失的声明。`ValidateArgEffects`
缩小了这个缺口但没有封闭，见上文。

**一致性警告的 lineage 不是按访问点维护的。** 它运行在非 SSA 的 IR 上（见上文），
那里"每个名字一个环境"并不精确。`BufferRootCollector` 预先扫描整个函数体，因此被
重新绑定的名字只有一份**最终**映射，却也被套用到更早的写入上：

```python
t = buf1
t = pl.tile.assemble(t, patch, [0, 0])   # 写的是 buf1
t = buf2                                  # ……但映射说 t -> buf2
```

`buf2` 被报告（尽管没有任何东西写它），`buf1` 被漏掉（尽管它确实被写）——同一个成因
同时造成误报与漏报。已作为 strict `xfail` 钉在
`tests/ut/ir/verifier/test_in_param_written.py` 中。

分支内建立的 view **不属于**这一类。它的 lineage 确实越过了汇合点，但汇合之后的写入
在被走到的那条路径上确实到达了那块 buffer，因此点它的名是正确的 may-write 答案；该
情形以一个通过的测试钉住。

## 参见

- [算子系统](05-operators.md) —— 完整的声明面。
- [Outline InCore Scopes](../passes/08-outline_incore_scopes.md) —— 阶段 1。
- [Convert Tensor to Tile Ops](../passes/10-convert_tensor_to_tile_ops.md) —— 阶段 2。
- [Derive Call Directions](../passes/38-derive_call_directions.md) —— 阶段 3。
- [IR 验证器](../passes/99-verifier.md) —— 阶段 4。
