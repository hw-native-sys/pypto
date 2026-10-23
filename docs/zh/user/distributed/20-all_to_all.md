# AllToAll：给每个对端一个不同的 slice

每个 rank 向每个对端发送一个*不同的* slice，并从每个对端收到一个不同的 slice——点对点模式中最通用的一种——然后内置原语一次调用完成。

> **前置：** [19-reduce_scatter](19-reduce_scatter.md)。任意 ≥ 2 个设备（示例使用 2 和 4 个 sim 设备）。

**建议阅读顺序：** 01 → … → 14 → **15** —— 本页是步骤 15。

## 思路

之前所有集合通信都移动*相同*形状的数据：broadcast 把一个 slice 送到所有地方，allgather 把每个 rank 的 slice 送到所有地方，reduce-scatter 做归约。All-to-all 则不同：rank `r` 向每个对端发送一个**不同的 `N/P` slice**——给目标 `d` 的 slice 不同于给目标 `e` 的。这就是个性化交换。

| 方面 | AllToAll |
| ---- | -------- |
| 输入 | 每个 rank 有 `P` 个不同分块（每个目标一个） |
| 输出 | 每个 rank 有 `P` 个不同分块（每个来源一个） |
| 模式 | push 每个目标分块（`put`）→ barrier → 读回 |
| 开销 | 收到 `(P-1)/P · N` 字节；没有任何两个 rank 想要相同的字节 |

这是真实 dispatch/combine 工作负载背后的模式——分布式 MoE 把每个 token 组发给不同专家的 rank；AllGather-GEMM 流水线按分片路由数据。本页 See-also 按名称链接了这些应用（不重述其内容）。

## 运行

```bash
# 手工：put 每个目标分块、barrier、读回。
python examples/distributed/15_all_to_all.py -p a2a3sim -d 0,1

# 揭示：一次调用完成 pld.tensor.all_to_all。
python examples/distributed/15_all_to_all.py -p a2a3sim -d 0,1 --mode builtin

# 同一源码在 P=4：
python examples/distributed/15_all_to_all.py -p a2a3sim -d 0,1,2,3
python examples/distributed/15_all_to_all.py -p a2a3sim -d 0,1,2,3 --mode builtin
```

预期输出：

```text
OK
```

golden 的构造使每个分块唯一：`input[r, d, j] = r*1000 + d*100 + j`——来源、目标、元素都编码在值中。任何路由错误（错误来源、错误目标）都会显示为错误值，而不是微妙的形状问题。

## 走读

两种模式共享一个 `[nr, SIZE]` window。Rank `r` 给目标 `d` 的分块最终落在目标
`d` 的 window 的第 `r` 行，然后 rank `d` 读取自己 window 的第 `src` 行——在手工
内核里，这次写入由调用方显式完成；而在下面的揭示中，push 发生在调用内部。手工
内核使用步骤 06 的 `put`：

```python
@pl.function(type=pl.FunctionType.InCore)
def hand_step(self, x, y, data, signal):
    ctx = pld.get_comm_ctx(data)
    my_rank = pld.rank(ctx)
    nranks = pld.nranks(ctx)

    # Phase 1 — push: write chunk-for-dest into dest's window at our row.
    for dest in pl.range(nranks):
        pld.tensor.put(data, dest, x, [my_rank, 0], [dest, 0], [1, SIZE])

    # Phase 2 — barrier: notify every peer, wait on every peer slot.
    for peer in pl.range(nranks):
        if peer != my_rank:
            pld.system.notify(signal, peer=peer, offsets=[my_rank, 0],
                              value=1, op=pld.NotifyOp.Set)
    for src in pl.range(nranks):
        if src != my_rank:
            pld.system.wait(signal, offsets=[src, 0], expected=1,
                            cmp=pld.WaitCmp.Ge)

    # Phase 3 — read-back: row src of our window holds src's chunk for us.
    for src in pl.range(nranks):
        chunk = pl.load(data, [src, 0], [1, SIZE])
        y = pl.store(chunk, [src, 0], y)
    return y
```

- **push 是对目标的循环。** `pld.tensor.put(data, dest, x, [my_rank, 0], [dest, 0], [1, SIZE])` 把 `x` 的行 `dest` 写入对端 `dest` 的 window、位于*我们的*行。每次迭代以不同的源行、不同的对端为目标——这就是“个性化”所在。
- **notify 用 `Set` 而非 `AtomicAdd`。** 每个 rank 是其行的唯一写入者（`[my_rank, 0]`），因此这里的 barrier 使用步骤 04 走读中的单写入者形式 `Set`/`Ge(1)`。
- **读回完成交换。** barrier 之后，我们自己 window 的行 `src` 保存着 rank `src` 发给我们的分块。

揭示用一次调用替换阶段 1–2；阶段 3 的读回循环仍由调用方完成：

```python
    result = pld.tensor.all_to_all(x, data, signal)
    for src in pl.range(nranks):
        chunk = pl.load(result, [src, 0], [1, SIZE])
        y = pl.store(chunk, [src, 0], y)
    return y
```

- **源是你的本地 `x`（普通 `pl.Tensor`）**，行 `d` = 给目标 `d` 的分块——与手工循环相同的布局。
- **window 变成结果**（行 `src` = 来自 rank `src` 的分块），且 `input`/`target` 必须是**不同**的缓冲。

### IR 对比（教学工件）

有趣的地方不在于搬运了哪些字节——两种模式都把同样的分块送到同样的目的地——而在于
内置实现替你写了多少序列，以及它在哪里停下。

- `--mode hand` lowering 为上述三个阶段：`P` 次 put（每个目标一次）、
  `Set`/`Ge(1)` barrier、以及读回循环——barrier 位于你自己的 push *之后*，但在你
  信任任何对端的数据*之前*，否则你的读取会与它们的 push 竞争。
- `--mode builtin` 展开为相同形状：你的逐目标分块经 `pld.tile.put` 被 push，然后
  是 barrier——但它到此为止。barrier 与手工版本处在传输的同一侧，因此含义也相同：
  它是一个**完成** barrier，确认每个对端的 push 都已落地。内置实现*不*替你读回
  这些行——window **就是**结果，读回循环留在你的代码里。
- composite 会在 barrier 之后追加**自清理尾声**，把本次调用的信用额度再减回去，
  因此同一个信号可以被后续集合通信复用
  （[21-putting_it_together](21-putting_it_together.md)）。HOST 内置原语也会对大
  传输分块，但那是在生成的 AIV kernel 内部、按固定大小的 tile 进行——不在编排层。
**成本卡（每 rank）：** 每个 rank 向每个对端发送一个*不同的* `N/P` slice——收到 `(P-1)/P · N` 字节，且没有两个 rank 收到相同的字节。这正是 all-to-all 是最难调度的点对点模式的原因：每对都交换不同的数据。

## 边界情况

> **致命陷阱——源与结果复用同一个 window。** All-to-all 的*结果*是就地交付的——`target` 以 window-as-result 的方式被重新绑定——但*源*是另一块缓冲：API 要求 `input` 与 `target` 必须不同——而在 InCore 路径上 `x` 是普通 `pl.Tensor`、`data` 才是 window，因此它们是两块不同种类的独立缓冲，而不是同一个 window 用两次。把同一 window 同时作为源与目标，会让 push 覆盖你仍需发送的分块。**修复：** 分配一个独立的目标 window（如示例所做）——内置原语也强制这一点。

| 症状 | 可能原因 | 修复 |
| ---- | -------- | ---- |
| 给目标 `d` 的分块落入错误的对端 | `put` 的对端 ≠ 目标行 | `put(data, dest, x, ..., [dest, 0], ...)` |
| 输出是我自己的分块而非对端的 | 读回自己的行 | barrier 后读行 `src` |
| 源/目标混叠损坏 | 输入与目标用同一缓冲 | 分开 `x` 与 `data` 两个 window |
| 某个 rank 挂起 | `Set` notify/wait 不匹配 | notify 写对端的行 `my_rank`；wait 读自己的行 `src` |
| 仅 P=4 时值错误 | 读回前缺少 barrier | put 循环与读循环之间加 barrier |

## 另请参阅

- [05-tutorials](05-tutorials.md) — 教程索引（本步 = 第 15 行）
- [01-collectives](../distributed/01-collectives.md) §AllToAll — 完整 API
- [03-execution](../distributed/03-execution.md) — 生产级 all-to-all 的 `DistributedWorker` 与设备端 staging
- `examples/runtime/distributed_callback.py` — 围绕 L3 分布式程序的宿主端运行时绑定回调
- [11-put_get](11-put_get.md) — 本步所依赖的 `put`/`get` 底层
- 更高级的应用（此处不重述）：pypto-lib [#869](https://github.com/hw-native-sys/pypto-lib/pull/869)（AllGather-GEMM，来自步骤 13 的 allgather 模式）与 DeepSeek-V4 分布式 MoE dispatch/combine（来自本步的 all-to-all 模式）
- [04-debugging](04-debugging.md) — 分布式程序的规范故障目录
- 下一步：[21-putting_it_together](21-putting_it_together.md) — 在一个内核中组合三种集合通信

## A5 URMA 索引 AllToAllV

```python
target = pld.tensor.indexed_all_to_all_v(
    input, target, signal, send_counts, recv_counts, index, core_num=2,
)
```

索引接口直接读取原始 `input[Nmax,H]` 行，无需完整 hidden 发送打包区。
`index` 为连续 INT32 `[P,C]`，独立接收窗口 `target` 为 `[P*C,H]`。
来源 s 将 `input_s[index_s[d,j]]` 写入目标 d 的 `target_d[s*C+j]`，
其中 `j < send_counts_s[d]`。重复索引保留重复槽位；尾部不写，保留原值。
消费者只读取各来源分区的 `recv_counts[s,0]` 行。MoE 的 token 到 rank
去重及接收端专家展开由上层实现。

首版仅支持 **A5 URMA、PER_PEER、L2**，payload 为 FP32 或 INT8。
L2 的 input/index 和 recv_counts 可以是本地 Tensor，recv_counts 为 INT32
`[P,1]`，只由本 rank 写入。send_counts 必须是窗口，
因为 peer 会远程读取。形状为静态，所有视图完整且连续。
不支持 HOST、InCore、远端 MTE 或 SHARED_POOL。

`core_num=B` 是正的精确启动核数，运行时按硬件 AIV 上限检查 sync-start。
`B <= P` 时 AIV b 负责目标 `b,b+B,...`；`B > P` 时只有前 P 个 AIV 提交
URMA，每个远端 rank 始终只有一个队列写入者。全部 B 个 AIV 平均分配本地接收行；
每个 AIV 先提交自己的远端发送，再搬本地行，最后等待远端最终 CQE。
一行 hidden 不跨 AIV 拆分。全部 B 个 AIV 都参与 barrier，包括分到零行的核。
signal 为 INT32 `[P,S]`，`S >= B`，首次使用前清零。L2 要求编译期核数。各 rank 必须使用一致的布局、启动核数和 collective 顺序。

counts 必须在 `[0,C]` 内，有效 index 必须在 `[0,Nmax)` 内。这些是设备数据
前置条件；非法值会中止执行，不做 clamp。索引尾部可以是 -1，不会被读取。
运行失败时需要终止整个通信组。

SIMT 构造多 SGE WRITE SQE，每个 SQE 最多包含十二条完整输入行，固定占四个
WQEBB，未使用的尾部 WQEBB 填 NOP。每个远端 peer 的整次发送必须装入其 SQ，
即 `ceil(send_counts[peer] / 12) * 4 <= sqDepth`，内核在提交前断言检查。
每段最多 64 个 SQE，构造完成后敲一次 doorbell，段间不等待。只有整次发送的
最后一个 WRITE 使用 strong ordering、fence 并请求 CQE，全部提交后统一等待一次。
SQ head 按 WQEBB 计数，ISA 的 submittedWqeCount 每次非空远端 peer 发送只加一。
最后执行完成 barrier。
self 直接使用 TLOAD/TSTORE，采用 8 个 8 KiB UB stage 的环形流水，每个本地
搬运 AIV 使用 64 KiB UB。连续来源索引合并为区间，打散行也跨 token 保持流水。
仅复用 stage 时等待旧 MTE3 store；后续 token 的 MTE2 load 可以与前面的 store
重叠，末尾统一 drain 后再发布完成。不再逐 token 调用同步 TPUT。远端 payload
直接通过 URMA 发送，运行时没有为 self rank 初始化 URMA loopback SQ/CQ。

接口返回建立任务完成依赖，不提供仅提交即退核的异步 handle。input/index/counts
生产者必须先完成，target/recv_counts 消费者必须等待 collective。输入保持存活到
任务完成；接收窗口复用还必须等待上一轮本地消费者。

通知与轮询均使用带退避的 SIMT 全局原子操作，不使用标量 TNOTIFY/TWAIT 实现，
仍保留同一组 signal lane 上的双轮 credit 协议。接收计数通过非缓存标量写发布，
避免控制窗口共用 cache line 时整行回写旧值。A5 系统测试覆盖输入变化后的持久
窗口复用；不增加遍历完整 payload 的缓存扫描。

共享 URMA 队列的交换必须由调用者保证串行。即使 payload 窗口不同，复用同一个
signal 也会通过 InOut 建立任务依赖；使用独立 signal 时必须显式建立任务依赖。
接口不会额外插入全局队列串行依赖。
