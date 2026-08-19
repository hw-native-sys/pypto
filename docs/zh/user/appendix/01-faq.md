# FAQ 与已知限制

反复出现的问题，以及它们背后的约束。

## 写 kernel

**为什么 kernel 跑了却什么都没写？**
`out = pl.add(a, b)` 只是重新绑定了一个 Python 名字，并不写输出参数。要用 `out[:] = pl.add(a, b)`，它是 `pl.assemble` 的语法糖。这是最常见的第一个 bug，而且它**静默失败** —— 运行成功，输出全是零。见[语法](../language/06-syntax.md)。

**为什么普通的 `range` 会被拒？**
DSL 需要知道一个循环究竟是编译期展开、设备侧循环，还是并行派发。请用 `pl.range` / `pl.unroll` / `pl.parallel` / `pl.pipeline` / `pl.spmd`。见[控制流](../language/02-control-flow.md)。

**为什么不能用 Python list 收集 TaskId？**
函数体是被**追踪**的，不是被执行的，所以 `tids.append(tid)` 不是一个 DSL 操作。用 `pl.array.create(n, pl.TASK_ID)` 并按下标赋值。见[声明一条边](../tasks/02-submit.md)。

**为什么 `pl.dynamic` 用在 `@pl.jit` 入口上会失败？**
动态轴属于 InCore kernel 的注解。在动态形状的张量上做编排级运算会走到 `InitMemRef`，而它需要常量维。让 kernel 保持 `@pl.jit.incore`，入口本身保持 dynamic 并把活交给它。见[类型](../language/00-types.md)。

## 编译

**`compile()` 报 `TypeError: got an unexpected keyword argument`。**
`compile()` 的位置参数是 kernel 自己的参数。编译选项走 `config=RunConfig(...)`。见[编译](../execution/00-compile.md)。

**worker 拒绝我编译出来的程序。**
产物的 `platform` 必须与 worker 一致。用你要派发的平台去编译。

**`ptoas compilation failed:` 后面是空的。**
那是 ptoas 二进制崩了，而不是它拒绝了你的 IR。把 `PTOAS_ROOT` 指向一个能用的版本。

## 运行

**为什么 `run()` 的时间比 kernel 大那么多？**
`RunResult.execution_time` 是整段墙上时间 —— 含编译、golden 生成与校验。要 device/host 拆分请用 `pypto.runtime.benchmark`，它的 `BenchmarkStats` 带有 `device_wall_us` 与 `host_wall_us`。见 [Host](../performance/06-host.md)。

**为什么把 `DeviceTensor` 传给 `@pl.jit` kernel 会失败？**
`DeviceTensor` 不携带 shape 或 dtype 供特化器读取。改为派发**已编译**的程序。见[运行](../execution/01-run.md)。

**容量错误点名了一个我从没配置过的环。**
环的选择由作用域深度决定 —— `min(scope_depth, 3)` —— 所以嵌套很深的 kernel 会把活集中到 ring 3，而扁平的 kernel 全压在 ring 0。改尺寸之前先用 `enable_scope_stats` 度量。见[内存](../performance/05-memory.md)。

## 数值

**split-K 的结果逐次运行不同。**
跨核的原子累加顺序不固定。要预期末位差异，那是正确行为而不是 bug。见[精度](../precision/00-workflow.md)。

**一个正确的 FP16 kernel 过不了 `allclose`。**
默认的 `rtol=1e-5` 对 FP16 输入是错的，后者只有约三位十进制有效数字。在排查 kernel 之前，先让容差匹配**输入**的精度。

**多跳 cast 会掉精度吗？**
不会。A5 上的 `INT32 -> FP32 -> FP16` 在相同舍入与溢出行为下与单步转换**逐位相同** —— 该页用一个可运行的检查证明了这一点。见[精度](../precision/00-workflow.md)。

## 已知限制

| 限制 | 细节 |
| ---- | ---- |
| **PTOAS 下两个 slot 同时存活** | codegen 拒绝：ptoas 只保护一次迭代里的第一个 `multi_tile_get`。该写的形状是一次迭代一个 slot 存活 |
| **硬 `syncall` 需要满占用** | 部分发射会在设备上死锁（507018）；PyPTO 在编译期就拒绝。部分占用下用 `mode="soft"` |
| **ring allreduce 不是加一个参数的事** | 它需要显式的 `[2*(NR-1)+1, NR]` INT32 signal、静态可知的 `src` 形状，以及 `numel(src)` 被 `NR` 整除。见[集合通信](../distributed/01-collectives.md) |
| **`memory_planner=PTOAS` 与内存图** | 分配 pass 被跳过，pass dump 里没有偏移供工具绘制 |
| **文档代码块是 backend 相关的** | 手册里的可运行块在 `a2a3sim` 上执行；A5 独有的行为以论证给出而非执行 |

## 参见

- [特性矩阵](00-feature-matrix.md) —— 各 backend 支持什么。
- [工具](../tools/index.md) —— 上面多数答案背后的调试面。
- [快速上手](../00-getting_started.md) —— 如果你同时撞上了好几条。
