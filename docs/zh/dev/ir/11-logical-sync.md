# 逻辑同步 IR

逻辑同步描述已特化 AIC 或 AIV 叶子函数单次执行中的完成依赖。
它复用普通 `Call`、`AssignStmt`、`EvalStmt` 和结构化 SSA 控制流，
新增值类型 `SyncTokenType(pipe)`。所属函数确定核与执行实例的身份；
token 不重复携带核属性。

这些操作支持 IR 构造、验证和往返。硬件降低（`LowerSync`）尚未实现，
代码生成会拒绝未降低的逻辑同步。本接口不启用自动插同步，
也不引入 JIT 选项。已有物理 flag 和 FIFO 操作保持各自的语义。

## 操作

公开命名空间为 `pl.sync`。下表中的 `P` 表示具体的 `pl.PipeType`，
禁止使用 `ALL`。每个 token 都不可变，且不会被等待消耗。

| 签名 | 结果 | 含义 |
| ---- | ---- | ---- |
| `record(pipe: PipeType)` | `SyncToken[pipe]` | 捕获源流水在当前语句位置的命令前缀完成事实。 |
| `completed(pipe: PipeType)` | `SyncToken[pipe]` | 已经满足的空完成事实，不会使真实工作变为完成。 |
| `join(first: SyncToken[P], second: SyncToken[P])` | `SyncToken[P]` | 合取：两个事实都必须成立。输入属于同一源流水和函数执行实例。 |
| `wait(token: SyncToken[P], dst_pipe: PipeType)` | `Void` | 使目标流水后续命令依赖所捕获的完成事实。 |
| `memory_overlaps(first, second, *, first_begin, first_end, second_begin, second_end)` | `BOOL` | 对两个运行时存储字节区间进行纯重叠判断。 |
| `vf_barrier()` | `Void` | 当前 VF 的此前工作完成后，后续 VF 工作才能执行。 |

`record` 必须直接作为赋值右值。`wait` 和 `vf_barrier` 必须直接作为
`EvalStmt` 调用，不能赋值或嵌套为其他值操作数。`completed`、`join`
和 `memory_overlaps` 是纯值操作。

## 命令前缀与等待

记录覆盖源执行域的完整命令前缀，包括此前导入该域的等待。
它不包含后续提交，也不包含无关流水的工作。多阶段操作影响哪些执行域，
由 backend 执行契约决定，不能仅凭文本先后顺序将其包含进去。

以下片段是 AIV 叶子函数内的逻辑 IR 示例。
注释代表仍然保留在程序中的原始计算。

```python
# Submit the MTE2 load of A.
old = pl.sync.record(pl.PipeType.MTE2)
# Submit the independent MTE2 load of B.
new = pl.sync.record(pl.PipeType.MTE2)
pl.sync.wait(old, pl.PipeType.V)
# V may now use A; this wait does not require B's completion.
pl.sync.wait(old, pl.PipeType.MTE3)
pl.sync.wait(old, pl.PipeType.V)
```

同一事实可以供多个目标和重复等待使用。后续记录或物理资源复用不能改变
`old` 的含义。token 不是 flag ID，成功等待也不会消耗其含义。
硬件降低必须分析所有使用者和控制流，不能给每个记录机械分配一个物理事件。
没有证明时，降低不能把旧前缀扩大到后续工作：
额外等待可能在通信协议周围引入等待环。

```python
pl.sync.wait(old, pl.PipeType.V)
forwarded = pl.sync.record(pl.PipeType.V)
pl.sync.wait(forwarded, pl.PipeType.MTE3)
```

即使中间没有 V 计算，第二个等待也传递地包含 `old`。
允许等待同一流水：命令提交有序不意味着每条命令完全结束后下一条才开始。

```python
pl.sync.wait(old, pl.PipeType.MTE2)
combined = pl.sync.join(old, new)
pl.sync.wait(combined, pl.PipeType.V)
```

`join` 不捕获新命令、不执行等待，也不分配事件。
对同流水 token，合取满足结合律、交换律和幂等律；
`join(x, completed(P)) = x`。只有证明较晚凭证的前缀包含另一凭证的要求后，
优化才能选择较晚凭证代替合取。

`record` 观察与位置相关的执行状态。属性相同不能证明可以做公共子表达式消除
（CSE），也不能证明可以跨越命令、等待或控制流移动记录。
结果完全无用的捕获可以删除。等待与 VF 屏障即使不写 Tensor，
仍然具有顺序效果；纯凭证代数不会使它们成为普通死调用。
`GetLogicalSyncKind` 是共享的 C++ 身份查询接口。

## 控制流与生命周期

分支结果选择实际路径的 token，不合取未执行路径。
只有该路径不存在对应依赖时才能使用 `completed`，否则应保留传入的 token。

```python
if condition:
    # Submit the conditional MTE2 work.
    produced = pl.sync.record(pl.PipeType.MTE2)
    selected = pl.yield_(produced)
else:
    empty = pl.sync.completed(pl.PipeType.MTE2)
    selected = pl.yield_(empty)
pl.sync.wait(selected, pl.PipeType.V)
```

循环携带值标识动态执行中的完成事实。
零次执行返回初值，多次执行则将前一轮的凭证传递给下一轮。

```python
initial = pl.sync.completed(pl.PipeType.V)
for i, (previous,) in pl.range(trips, init_values=(initial,)):
    pl.sync.wait(previous, pl.PipeType.MTE2)
    # Submit this iteration's work, with its other dependencies satisfied.
    current = pl.sync.record(pl.PipeType.V)
    final = pl.yield_(current)
pl.sync.wait(final, pl.PipeType.MTE3)
```

token 可以通过局部 SSA 赋值、临时 SSA 元组、分支结果和循环携带值传递。
禁止进入函数参数或结果、持久容器，以及 VF 寄存器作用域。
不定义递归执行、跨核 token 传递或跨函数执行实例的 token 身份。
仍须满足 SSA 支配关系和源流水一致性。

FIFO 同步保留原生 push/pop/free 协议。借用槽位的异步读者完成之前，
不得释放槽位。应记录这些读者的完成点，并将释放所需的等待放在 `FREE` 前；
不能只为简化降低，就把后续独立 `PUSH` 纳入该完成前缀。
这些是顺序要求，不构成任意 FIFO 程序无死锁、或目标事件资源足够的普遍证明。

## 运行时存储区间

`memory_overlaps` 接受地址空间**已知且相同**的 Tensor、Tile 或 Buffer。
区间原点是各自运行时数据地址。目前的 `PtrType` 或 `MemRef` 表达分配身份，
没有运行时数据地址契约，因此被拒绝。Tile 的未知地址空间必须先解析。

四个边界是有符号 `INDEX` 标量 SSA 操作数；Python 整数会转换为 INDEX 常量。
底层位置参数顺序为
`[first, second, first_begin, first_end, second_begin, second_end]`，
没有区间属性。边界单位为字节，区间采用半开形式。

```python
overlap = pl.sync.memory_overlaps(
    source, destination,
    first_begin=source_offset * 4,
    first_end=(source_offset + count) * 4,
    second_begin=destination_offset * 4,
    second_end=(destination_offset + count) * 4,
)
if overlap:
    pl.sync.wait(read_done, pl.PipeType.MTE3)
```

这个 FP32 示例防止重叠写入抢在读取之前发生。
谓词不读取数据、不等待，也不检查分配边界。空区间不重叠。
输入必须满足 `begin <= end`；静态反向常量会被拒绝，动态合法性由调用方保证。
常量必须能用有符号 INDEX 表示。地址比较必须保持数学上的顺序，不能回绕；
谓词无法恢复此前边界表达式溢出而丢失的信息。

## VF 边界

`vf_barrier` 只允许出现在 A5 目标上 AIV 叶子函数的 `pl.reg_scope()` 内。
它使此前寄存器／VMEM 工作在后续 VF 工作之前完成。
首版采用 VF 整体边界，尚未表示逐寄存器事件。

```python
pl.sync.wait(input_ready, pl.PipeType.V)
with pl.reg_scope():
    value = pl.reg.load(tile, lanes=64)
    pl.sync.vf_barrier()
    doubled = pl.reg.add(value, value)
    pl.sync.vf_barrier()
    updated = pl.reg.store(tile, value=doubled)
vf_done = pl.sync.record(pl.PipeType.V)
pl.sync.wait(vf_done, pl.PipeType.MTE3)
```

目标必须保证外层 V 等待约束 VF 的实际访问开始，且后面的 V 记录覆盖
整个 VF 的完成，包括 VMEM。语法上的作用域退出本身不表示排空 `ALL`。
没有目标契约时，降低不能假设函数返回就意味着异步访问完成。
删除内部屏障需要保留可观察结果、内存效果和可前进性，
不能只依据寄存器 SSA 使用关系。

## 表示与验证

Python 提供 `ir.SyncTokenType(ir.PipeType.MTE2)` 和标注
`pl.SyncToken[pl.PipeType.MTE2]`。C++ 使用
`std::make_shared<ir::SyncTokenType>(ir::PipeType::MTE2)` 构造等价类型。
反射、结构比较、二进制序列化和 DSL 打印均保留源流水。
验证会重新检查操作 schema 和作用域规则，包括直接构造或反序列化的调用。
token 不能藏在存储元数据中，包括 MemRef 字段，以及通过分布式张量类型
引用的 WindowBuffer 字段。共享表达式 DAG 的检查不会把路径展开为树；
每次使用仍参与词法生命周期和执行上下文验证。
另见[类型](02-types.md)与[算子](05-operators.md)。
