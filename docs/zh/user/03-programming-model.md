# 编程模型

一个 PyPTO 程序背后的抽象：三个描述层次、在它们之间下降的编译流水，以及它们共同描述的内存层次。

> **前置**：你已经编译过[快速上手](02-quickstart.md)里那些张量级示例。本页解释它们到底在做
> 什么，并引入快速上手刻意留白的 tile 级。

## Concept

PyPTO 让你按**实际需要的控制粒度**去描述计算，其余部分由它替你 lower。同一个程序既可以命名
整个数组、让编译器决定放置，也可以命名单个片上缓冲区、由你手工搬运 —— 通常两者兼有，只是分布
在不同的函数里。

这种弹性建立在一条贯穿整个系统的分离之上：**算什么**由你的 Python 源码描述，**在哪运行**由
函数的 type 与 level 描述，**何时运行**由运行时依据编译器推导出的任务图决定。把这三者混为一谈，
是初期绝大多数误解的根源，因此读下文时值得始终带着这个区分。

PyPTO 程序里没有任何东西是在 Python 运行它时执行的。装饰器把源码解析成 IR；pass 重写 IR；
代码生成发射设备 kernel 与主机编排代码；运行时调度它们。Python 是撰写语言，不是执行引擎。

## Quickstart：一个程序里的三个层次

下面把同一个计算 —— `x * x` —— 写了两遍：一次在张量级，一次在 tile 级。两者都是
`@pl.jit.incore` 设备 kernel，由一个编排入口同时派发。

```python
import pypto.language as pl

@pl.jit.incore
def square_tensor(
    x: pl.Tensor[[128, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    # 张量级：命名整个数组。放置与搬运是编译器的事。
    out = pl.assemble(out, pl.mul(x, x), [0, 0])
    return out

@pl.jit.incore
def square_tile(
    x: pl.Tensor[[128, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    # Tile 级：命名片上缓冲区，数据由你自己搬。
    t = pl.load(x, [0, 0], [128, 128])
    y = pl.mul(t, t)
    pl.store(y, [0, 0], out)
    return out

@pl.jit
def levels(
    x: pl.Tensor[[128, 128], pl.FP32],
    out_t: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    out_k: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    # 控制面：不做计算，只派发。
    out_t = square_tensor(x, out_t)
    out_k = square_tile(x, out_k)
    return out_t, out_k
```

这两个 kernel 对调用方来说可以互换，算出来的数也一样。它们的差别只在于**你把多少事情说出了口**：

| 差别 | `square_tensor` | `square_tile` |
| ---- | --------------- | ------------- |
| 你命名的东西 | 整个数组 `x` | 片上缓冲区 `t` |
| 数据搬运 | 编译器插入 | 你写 `pl.load` / `pl.store` |
| 区域 | 隐式 —— 整个张量 | 显式 —— 偏移 `[0, 0]`、形状 `[128, 128]` |
| 内存空间 | 编译器选择 | 你可以传 `target_memory=` |
| 代码行数 | 1 | 3 |

`ConvertTensorToTileOps` 会把前者变成很接近后者的东西 —— 对比 pass dump 就能看到这个过程。
所以 tile 级不是另一门语言，而是同一个程序、把选择明说出来。当某个选择开始要紧时你才下去：
哪个区域、什么时候上片、落在哪个缓冲区、怎么复用。

编译 `levels` 会产出**两个**设备 kernel，每个 `.incore` 函数一个：

```text
kernels/aiv/square_tensor.cpp
kernels/aiv/square_tile.cpp
```

三个层次，以及它们各自在上面出现在哪：

| 层次 | 你命名的东西 | 在本例中 | 谁决定放置 |
| ---- | ------------ | -------- | ---------- |
| **张量（Tensor）** | DDR 中的整个数组 | `square_tensor`；以及只负责传递数组的 `levels` | 编译器 |
| **Tile** | 片上缓冲区 | `square_tile` —— `pl.load`、`pl.mul`、`pl.store` | 你 |
| **Block** | 核以及它们之间的协同 | 本例未用。`pl.at(level=...)` 指定一个核组；`pl.spmd`、`pl.cluster` 走得更远 | 你，显式地 |

`pl.at` 是你最先遇到的 Block 级旋钮，而本例不需要它的原因正是 `@pl.jit.incore`：`.incore`
函数本身就已经被放在核上了。单函数 kernel 没有子函数来承载这个放置，所以要用
`with pl.at(level=pl.Level.CORE_GROUP):` 就地开作用域 —— 见[快速上手](02-quickstart.md)。

快速上手完全停留在张量级 —— `out = pl.add(a, b)`，一个 `pl.load` 都不出现 —— 所以上面的
`square_tensor` 就是你已经熟悉的那个样子，`square_tile` 才是往下走的那一步。Block 级用于指明
哪个核做什么：多 block 派发、cluster 作用域、AIC/AIV 混合 kernel。

## Mechanics

### 控制面与执行面

执行被划分到两个面上。`Orchestration` 在控制面，InCore 家族（`InCore` 以及编译器由它派生出的
`AIC` / `AIV` / `Group` / `Spmd` 形态）在执行面：

```text
HOST / Orchestration          控制面
  │  创建张量、派发任务、携带循环状态
  │  从不碰 tile 内存
  ▼
InCore (AIC / AIV)            执行面
     load、计算、store
     从不分配张量、也不派发任务
```

`Opaque` 与 `Inline` 是**不归属任何面**的两个取值，原因正好相反：`Opaque` 是还没有确定，
`Inline` 则根本不会以函数形态到达代码生成。

| 取值 | 所属面 | 含义 |
| ---- | ------ | ---- |
| `Orchestration` | 控制面 | 主机侧编排者 —— 分配张量、派发 kernel |
| `InCore` | 执行面 | AICore 上的计算 kernel |
| `AIC` / `AIV` / `Group` / `Spmd` | 执行面 | 编译器拆分与外提你的代码时产生 —— 很少需要手写 |
| `Opaque` | 尚未归属 | 默认。无特定执行上下文；作为构件，其所属面由使用它的位置决定 |
| `Inline` | 无 | 由第一个 pass 展开到每个调用点，最终不留下函数，因此本身没有所属面 |

函数的 `level` 与 `role` 会进一步细化：`pl.Level.HOST` 配 `pl.Role.Orchestrator` 标记的是
分布式程序的主机编排者。（**不存在 `FunctionType.Host`**；"主机性"是用 level/role 这一对来
表达的。）

### 编译流水

```text
Python DSL          @pl.jit / @pl.program 把源码解析成 IR
     │
     ▼
IR                  不可变树，贯穿整个编译过程共享
     │
     ▼
Pass 流水线         默认策略，按序：内联、SSA、外提作用域、tensor->tile、
     │              layout、内存规划、任务依赖、……
     ▼
CodeGen             设备 kernel（.pto -> C++）+ 主机编排 C++
```

每个阶段都可观测。`lower()` 会特化 JIT 函数、运行配置对应的 Pass 流水线，并返回 Pass 后的
`ir.Program`；对该结果调用 `program.as_python()`，即可查看最终 lowering 后的 IR。与之不同，
`CompiledProgram.program` 保留的是特化后、Pass 前的 program，而不是 Pass 流水线的输出。
调用 `compile()` 时设置 `dump_passes=`，会在每个 Pass 之后写一份快照；`lower()` 不会写
Pass 快照。Pass 本身在 [Passes](../dev/passes/index.md) 中逐个有文档，并按执行顺序编号。

`lower()` 不会执行代码生成，也不会填充编译缓存。需要验证代码生成时请使用 `compile()`。
（`@pl.jit` 函数自身没有 `as_python()`；要查看 Pass 后的 IR，请检查 `lower()` 的结果；要查看
`compile()` 后保留的特化后、Pass 前 IR，请调用 `compiled.program.as_python()`。）

作为用户，IR 的两个性质与你直接相关：

- **它是 SSA 的。** 每个绑定只写一次。在 Python 源码里重复绑定同一个名字是允许的 —— parser
  会重命名，并把循环内被重复绑定的值作为携带值穿过该循环。这就是为什么 `pl.range` 里的
  `acc = pl.add(acc, ...)` 能work，尽管 IR 里并不存在就地修改。
- **它是不可变的。** pass 构建新 IR 而不是就地修改，这也是逐 pass 快照对调试有意义的原因。

### 内存层次

Tile 级代码之所以对内存如此显式，是因为硬件本身如此。`pl.load` 与 `pl.move` 接受
`target_memory=` 参数，指明数据应该落到哪里：

片上空间是**六个相互独立的缓冲区，不是嵌套关系**。`Left` 不是 `Mat` 内部的一块区域，`Acc`
也不在 `Right` 里面 —— 它们是各自独立的硬件缓冲区，数据在它们**之间**搬运。

| 空间 | 枚举 | 硬件 | 能从 DDR 直达吗 |
| ---- | ---- | ---- | --------------- |
| DDR | `pl.Mem.DDR` | 片外全局内存 | —— 它本身就是 DDR；`pl.Tensor` 参数在这里 |
| Vec | `pl.Mem.Vec` | 统一缓冲区 | **能** —— `pl.load` 的默认目标 |
| Mat | `pl.Mem.Mat` | L1 | **能** —— `pl.load(..., target_memory=pl.Mem.Mat)` |
| Left | `pl.Mem.Left` | L0A，matmul 左操作数 | 不能 —— 只能由 `Mat` / `Vec` 经 `pl.move` 到达 |
| Right | `pl.Mem.Right` | L0B，matmul 右操作数 | 不能 —— 只能经 `pl.move` |
| Acc | `pl.Mem.Acc` | L0C，matmul 累加器 | 不能 —— 由 `pl.matmul` 写入 |
| Bias | `pl.Mem.Bias` | AIC 核上的 bias 缓冲区 | 不能 —— 只能经 `pl.move` |

`pl.MemorySpace` 与 `pl.Mem` 是同一个枚举的两个名字。

最后一列是承重的约束：**面向 DDR 的 load 只能落到 `Vec` 或 `Mat`。** 当消费者需要
`Left` / `Right` / `Acc` / `Bias` 时，生产者先停在 `Mat`（或 `Vec`），再由
`InferTileMemorySpace` 插入一个 `tile.move` 到专用空间 —— 这就是下面 matmul 通路里那一步
显式 `pl.move` 的由来。

下面画的是**数据流**而非包含关系。matmul 的两个操作数**汇聚**到 `Acc`，所以它是一张图、
不是一棵树：

```text
       pl.load(target_memory=Mat)      pl.move(Left)
  DDR ────────────────────────► Mat ─────────────────► Left ┐
                                                            │  pl.matmul
                                                            ├──────────► Acc ──────► DDR
  DDR ────────────────────────► Mat ─────────────────► Right┘                pl.store
       pl.load(target_memory=Mat)      pl.move(Right)

       pl.load()                    逐元素算子                  pl.store()
  DDR ───────────► Vec ─────────────────────────────► Vec ───────────────► DDR
       （默认）
```

matmul 通路正是这些空间被暴露而非隐藏的原因：操作数必须经 L1 到达 L0A/L0B，结果在 L0C 累加。

```python
@pl.jit.incore
def mm(
    a: pl.Tensor[[32, 32], pl.FP16],
    b: pl.Tensor[[32, 32], pl.FP16],
    out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
):
    a_l1 = pl.load(a, [0, 0], [32, 32], target_memory=pl.Mem.Mat)
    b_l1 = pl.load(b, [0, 0], [32, 32], target_memory=pl.Mem.Mat)
    a_l0a = pl.move(a_l1, target_memory=pl.Mem.Left)
    b_l0b = pl.move(b_l1, target_memory=pl.Mem.Right)
    c_acc = pl.matmul(a_l0a, b_l0b)      # 落在 Acc
    pl.store(c_acc, [0, 0], out)         # Acc -> DDR
    return out
```

你并不总需要手写这条链 —— 张量级的 `pl.matmul` 会被 lower 成它。手写换来的是对分块与常驻的控制。

### 执行模型

编译产物不是一个从头跑到尾的单一二进制，而是一组设备 kernel 加上主机编排代码，后者向运行时
**提交任务**，由运行时依据依赖图调度。

```text
已编译的 program
 ├── orchestration/   主机 C++：提交任务、携带循环状态
 └── kernels/         设备 kernel，每个 InCore 函数一个
                          │
                    运行时调度器
                          │  推导 / 消费任务依赖图
                          ▼
                    AICore 执行
```

对写代码的直接影响是：**源码顺序不构成顺序保证。** 只有当程序里有东西确立了两个任务之间的
先后关系时，运行时才会按序执行它们 —— 可能是编译器从缓冲区重叠推导出的依赖，也可能是你显式
声明的依赖。仅仅把一次派发写在另一次之后，本身什么也没表达。需要某个先后次序时，要把它表达
出来，不要从语句位置去推断。

这些任务落到的硬件按集群组织：**1 个 Cube 核 + 2 个伙伴 Vector 核**，共享基于 flag 的同步机制。
混合 kernel 与跨核流水这些概念正是由这个形态而来，参见
[集群架构](../reference/pto-isa/00-cluster_architecture.md)。

## Edge Cases

> **致命陷阱：** Orchestration 函数里的语句顺序不约束执行顺序。如果两次派发必须按序执行，
> 这个次序必须被表达出来 —— 通过依赖，或通过编译器能看见的缓冲区关系。只依赖源码顺序，
> 运行时就可以自由地让它们重叠，结果是一个偶发复现、一上调试器就消失的竞争。

| Symptom | Likely Cause | Fix |
| ------- | ------------ | --- |
| **多次运行结果不一致** | 两个必须有先后的任务，没有任何东西表达了这个先后 | 显式声明依赖；仅有源码顺序不构成先后 |
| **`pl.load` 直接写在 `@pl.jit` 体里失败** | 在控制面上使用了 tile 操作 | 用 `with pl.at(level=...)` 包起来，或移进 `@pl.jit.incore` 子函数 |
| **`@pl.jit.incore` 函数里的 `pl.create_tensor` 失败** | 在执行面上分配张量 | 在控制面分配，或把缓冲区作为 `pl.Out[...]` 参数接收 |
| **循环里写的值在循环后是空的** | 携带值从未离开循环 | 每次迭代重新绑定它（`acc = pl.add(acc, ...)`），循环后再读 |
| **`pl.matmul` 拒绝其操作数** | 操作数不在 `Left` / `Right` | 先 `pl.load` 到 `Mat`，再 `pl.move` 到 `Left` / `Right` |

## See Also

- [快速上手](02-quickstart.md) —— 本页所解释的那些例子。
- [语言指南](language/index.md) —— 完整表面：类型、函数、控制流、内存、作用域与任务、编译期指令。
- [Passes](../dev/passes/index.md) —— 流水线中的每个 pass，按执行顺序编号。
- [IR 概览](../dev/ir/00-overview.md) —— IR 的结构与设计原则。
- [集群架构](../reference/pto-isa/00-cluster_architecture.md) —— 执行模型所面向的 Cube + Vector 集群。
