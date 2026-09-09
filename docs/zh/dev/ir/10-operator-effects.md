# 算子效应（Operator effects）

本文介绍 Functional 阶段的参数效应、写通道及核放置/复制契约。Buffer 阶段
算子使用单独的[数据/元数据效应契约](02-types.md#buffer-算子契约)。

> 消费这些声明的整条链见
> [参数方向推导](08-param-directions.md)。

原地更新某个参数的算子必须显式声明。方向推导（direction inference）、依赖分析
（dependency analysis）和参数方向验证器都向注册表询问同一个问题——*这次调用是否
写入该参数所指的缓冲区？*——而从未回答过的算子会被读成纯消费者：

```text
tile.mscatter 写 output_tensor，却从未声明
  → 它写入的参数方向停留在 In
  → 不会对读取它的 kernel 发出 RAW 边
  → 调度器可以先运行读方
  → 读到陈旧数据，或等待一个无人写入的信号而死锁
```

| 效应 | 含义 | 示例 |
| ---- | ---- | ---- |
| `ArgEffect::Read` | 只读，从不写入。未声明参数的默认值 | `tile.store` 的源 tile |
| `ArgEffect::Write` | 覆盖写，写前不读 | `tile.store` 的 `output_tensor`、`pld.tile.get` 的 `dst` |
| `ArgEffect::ReadWrite` | 既读*又*写 | `tile.matmul_acc` 的累加器、原子 store 的目的操作数 |

**部分覆盖仍然是 `Write`。** 只写入子区域的 store 不会读取未触及的其余部分——没有
任何数据流*进*内核——所以其目的操作数是纯写。把它声明成 `ReadWrite` 并非无害的保守
近似：这会让外层参数变成 `InOut`，从而触发 host→device 搬运，并且在跨 rank 场景下
为两个写入不相交行的 rank 凭空造出一条依赖。

目的操作数是否被读取按**算子**判定，而非按家族：gather / exchange 的目的窗口只被推入、
从不被 load，因此是 `Write`；而 reduce 的目的窗口会把运行值 load 回来，因此是 `ReadWrite`。

**`ReadWrite` 留给真正会读取该槽位的算子**：累加器（`out += x` 会读取运行中的和）、
原子 store/assemble，或那些未触及位置会流入 SSA 结果的 destination-passing 算子
（`tile.scatter`、`array.update_element`）。

**由 kwarg 决定的效应。** 当答案取决于某个 kwarg 时，传入 resolver 而非常量。kwarg 不
仅能决定*怎么*写，还能决定某个实参*是否*被写：`tile.mgather` 的第三个操作数在 Mat elem
模式下是被写的 GM scratch，在 Mat row 模式下则是只读的 `valid_shape`。另外两处是 store
家族的 `atomic` kwarg，以及 `pld.system.notify` 的 `op` kwarg——后者默认是 atomic-add，
因此未加标注的 notify 会读取它累加的槽位：

```cpp
REGISTER_OP("tile.store")
    // ... arguments, memory spec ...
    .set_arg_effect(2,
                    [](const std::vector<std::pair<std::string, std::any>>& kwargs) {
                      return GetIntKwarg(kwargs, "atomic", static_cast<int>(AtomicType::kNone)) ==
                                     static_cast<int>(AtomicType::kNone)
                                 ? ArgEffect::Write
                                 : ArgEffect::ReadWrite;
                    })
    .set_write_channel(WriteChannel::Dma)
```

**"已声明为只读"不等于"未分类"。** `HasDeclaredArgEffects()` 区分"有人判定该算子不
写入任何参数"（`no_arg_writes()`，例如 `pld.system.wait`）与"还没有人看过这个算子"。
需要答案的分析因而可以拒绝猜测，而不是把未分类的写者默认成只读。

**强制约束。** `OpRegistry::ValidateArgEffects()` 在 import 时运行，拒绝两种形态，
一次性列出所有违规算子和修复方式，而不是等到首次使用才失败：

- 声明了 `set_output_reuses_input(N)`（其 SSA 结果*就是*第 N 个参数的缓冲区，因此会
  通过它写入），却没有专门对第 N 个参数作出裁决。要求的是"做出分类"而非某个特定答案：
  原地槽位是元数据的算子可以声明为只读。
- 声明了写通路却不通过任何参数写入。通路描述的是"怎么写"，因此没有写的通路要么是多余
  声明，要么是漏了声明。

第二条比看上去重要。`set_write_channel()` 会顺带创建效应 spec，因此"spec 存在"不能
等同于"有人做过判断"——否则一个声明了通路却忘了 `set_arg_effect` 的算子会通过第一条
检查，而它原地更新的那个参数仍然默认为 `Read`。`no_arg_writes()` 显式记录"对所有参数
的裁决"，且与 `set_arg_effect` 同时使用会被判为自相矛盾而拒绝。

`set_write_channel` 记录写入走的是 MTE3/DMA 通路还是标量 D-cache 通路。PyPTO 无法为
同一个 GM tensor 排序这两者，因此会拒绝在同一缓冲区上混用两者的函数；有了通路声明，
该诊断可以查询注册表而不必再列一遍算子清单。

只为写入确实走这两条通路之一的算子声明它，其余一律留空——留空会把该算子排除在该诊断
之外，而当两条通路都无法描述它时，这正是它该待的位置：

- `pld.system.notify` 发出的是 `pto.comm.tnotify`，一条独立的 comm 指令。声明任一通路
  都会让该诊断拒绝合法程序。
- `system.set_ffts` 是把 workspace *指针*交给 FFTS 单元，而非搬运数据；该区域由硬件按
  自己的节奏写入，没有任何依赖边能建模它。它声明 `no_arg_writes()`。
- 复合集合通信通过不同机制更新数据窗口与 signal，单个算子级通路无法同时描述两者。
  按参数记录通路可以做到，但目前没有任何用例需要这种区分，而一个错误的单一答案比没有
  答案更糟。

**`set_core_affinity` 与 `set_no_duplicate`** —— 两个正交的维度，选错会对 ISA
做出错误的断言：

- `set_core_affinity(...)` 回答算子在*哪个*核上运行。只有当硬件确实把算子限制在
  某一侧时才声明。未声明时，`ClassifyCallAffinity` 从调用本身推导放置位置
  （先看 memory spec，再看第一个 tile 参数的内存空间），最终回退到 `SHARED`。
- `set_no_duplicate()` 回答算子是否允许在*第二个*核上运行。`ExpandMixedKernel` 会把
  `SHARED` 语句复制到 AIC 和 AIV 两条通路上；当这份副本会改变程序语义时，标记该算子。

`pld.system.notify` 是典型例子，而其风险在于**从错误的通路上提前释放**，而非不幂等：
AIC 上的那份副本可能在 AIV 通路的 TPUT 尚未把该信号所释放的数据落盘之前就发布信号，
于是对端 rank 读到过期数据。正因如此该标记是无条件的 —— 尽管只有原子加会重复计数，
`NotifyOp::kSet` 触发的竞态与原子加完全相同。

兄弟算子 `pld.system.wait` **不标记**，且是有意为之：TWAIT 会*阻塞*，它出现在 cube
通路上是有实际作用的。把它钉在向量通路上，会让 matmul 越过它本应等待的对端数据。
请把该标记读作「不得在第二个核上运行」，而不是「不幂等」。

读取该维度的查询是 `IsNoDuplicate()`。它唯一的消费者是 `LowerAutoVectorSplit`
（pass 23）的 `pl.split_aiv` 区域放置标记：该 pass 恰好把区域内的 no-duplicate 调用
钉在 AIV 通路上。没有任何 verifier 在这个维度上做拒绝。

被 `set_core_affinity(...)` 固定在单条通路上的算子本来就不会被复制，因此无需该标记。
这个标记正是为核无关（core-agnostic）的算子准备的 —— 对它们而言，没有任何 affinity
取值能表达「两个核都能跑，但不能两个核都跑」。请注意这是关于**核**的断言，而不是关于
总执行次数的断言：在 `dual_aiv_dispatch` 下，AIV 函数体仍然会在两条 AIV sub-lane 上都
运行，因此把算子挡在 cube 通路之外并不意味着它只执行一次。那一部分属于作者的职责，
文档见[作用域 → pl.split_aiv](../../user/language/04-scopes.md)。

对于 Buffer 阶段调用，共享的 `CallWriteTargets` 查询只投影已注册的数据效果。
因此，最终降低之后，`InParamWritten` 仍能诊断对声明为 `In` 的 GM 或 Buffer 参数的数据写入。
仅更新描述符的 `buffer.set_validshape` 不计为参数数据写入。
