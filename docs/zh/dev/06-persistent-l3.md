# L3 持久执行

通常情况下，准备好的分布式程序会复用同一个 Simpler worker，但每次 dispatch
仍会重新进入一次 `Worker.run()`。因此，包含通信窗口的程序会在每次调用时申请并
释放 CommDomain。

使用 `persistent=True` 可以在 prepared worker 的整个生命周期内保留 generated
CommDomain：

```python
with decode.prepare(persistent=True) as worker:
    for _ in range(100):
        worker(x, weights, output)
```

持久模式需要显式开启，默认的 `prepare()` 行为保持不变。

## 生命周期

每次 PyPTO 提交都会在调用线程中同步构建持久 orchestration，并直接调用 Simpler
`Worker.submit()`。返回的 `DistributedRunHandle` 会持有该请求，直到 native
completion fence 和清理全部结束。首次使用某个 generated CommDomain 时会申请物理
window，后续调用则获得同一个 handle 的 retained lease。关闭 prepared worker 时
会先停止接收新请求、排空所有已发布 handle，再释放全部 retained domain。请求和
domain 释放错误都会抛给调用方。

生成的 HOST orchestration entry 接受内部参数 `_domain_provider`。普通 dispatch
不传该参数，仍然调用 `orch.allocate_domain`；持久 dispatch 则传入一个按 compiled
program 和 generated domain name 隔离的 provider。已有 generated artifact 必须
重新生成后才能使用持久模式。

## Window 内容

持久执行默认会在复用前将 retained CommDomain window 恢复为全零，使每次重复
dispatch 都获得 fresh-window 语义，包括通信 buffer 中保存同步状态的程序。
第一次 dispatch 使用 runtime 新申请并初始化的 window，不执行额外 reset。

只有程序能够自行管理复用的通信 buffer 内容时，才应关闭 reset：

```python
with decode.prepare(
    persistent=True,
    reset_persistent_windows=False,
) as worker:
    worker(x, weights, output)
```

关闭 reset 后，后续 dispatch 会原样看到上一次留下的内容。调用方必须在复用前
手动清零相关通信 buffer，或者使用 epoch 等协议安全管理所有 retained signal 和
data 状态；如果两者都没有，复用陈旧状态可能导致错误结果或死锁。

默认开启 reset 时，PyPTO 会在复用前同步清零所有参与 worker 上的具名本地 buffer。
每次 reset 请求会按不同的具名 buffer 大小各创建一个全零的 POSIX 共享内存 host
`Buffer`，并在所有相同大小的 domain 和 worker 之间复用。staging buffer 会保留到
`Worker.run()` fence 返回后再释放，因此 staging 内存峰值是这些不同大小之和。
reset copy 会计入每次重复请求的 host 开销。

PyPTO 使用 `copy_to(dst, src)` 重置每个具名 buffer，默认拷贝长度为源 `Buffer`
的大小。生成的 domain 由具名 buffer 完整覆盖请求的 window；如果请求范围内含有
未命名的空隙，reset 会直接拒绝，因为该区域没有可用于恢复内容的具名 buffer。
运行时报告的物理映射（physical mapping）可能因分配对齐而大于请求大小，A5 就属于
这种情况。reset 接受这部分额外容量，只拷贝具名 buffer，不暴露或重置物理映射尾部
的填充区。如果物理映射小于请求的 window，则会在重置该 domain 的 buffer 之前报错。

## 多 compiled program

持久模式支持现有的 multi-program prepared worker：

```python
with prefill.prepare(extra_compiled=[decode], persistent=True) as worker:
    worker.run(prefill, prefill_x, weights, kv_cache)
    worker.run(decode, decode_x, weights, kv_cache)
    worker.run(decode, decode_x, weights, kv_cache)
```

Domain 按 `(compiled program, generated domain name)` 隔离。因此，即使 prefill 和
decode 都生成了 `comm_d0`，它们仍然使用不同的物理 domain。所有 prepared
program 仍须满足原有的 platform、runtime 和 device ID 兼容性检查。

`DistributedWorker.submit()` 和 Simpler 仍会串行构建 graph。Simpler 提供完成事件和
身份容量时，字节预算和公开的 `Worker.run_identity_capacity` 目录上限共同限制 Host 帧池。
完成事件归还帧并唤醒等待的提交者。
资源策略、显式资源契约和前驱成功依赖见[有界分布式提交](distributed-dispatch.md)。

## 运行时依赖

按资源预算提交使用 Simpler 公开的身份容量、完成事件及前驱成功接口。每个请求使用公开的
`Worker.submit()`，其 native handle 由 PyPTO dispatch handle 持有至完成。使用域池的请求
在提交时声明各帧的 retained 域。域池之外的请求通过
`Orchestrator.get_or_allocate_domain(..., retain_after_run=True)` 获取通信域
（CommDomain）。Simpler 在复用时验证配置，将通信域登记到当前运行的完成栅栏
（completion fence），并在成功运行之间保留其分配。prepared worker 关闭时，
PyPTO 等待所有 dispatch handle 完成后释放通信域；失败时由 Simpler 负责清理。
PyPTO 不再修改私有的通信域 registry 或 run-resource journal。

## Worker 持有的外部通信域

已准备好的 worker 可以通过 `create_comm_domain` 建域，将命名窗口传给独立编译的
HOST 入口。每个入口以静态、连续 ND 的 `pld.DistributedTensor` 声明窗口。
每次调用传入的 `DomainTensor` 决定域和槽位。内部持久化窗口的 reset 仅作用于程序内分配。

schema 4 的分布式 metadata 按声明顺序记录形参，用 `is_distributed` 标记外部窗口。
冷加载时按这个顺序恢复 scalar 后缀。每个窗口占一个 scalar 槽，存放
buffer index。配对的隐藏 ctx 位于后缀，占三个 uint64 槽，依次为按小端编码的
8 字节 owner ID、domain ID 和 generation。这些 scalar 由 HOST 回调原有的 `TaskArgs` 传递。普通 tensor 和保留 Python 类型的 scalar 仍按名称供 HOST 切片和控制流使用。
每次运行通过当前 orchestrator 查询各个不同的 ref，按成员取命名 buffer 和设备 ctx，
每个 CHIP 派发为各个不同的域追加一个排序 buffer。HOST 编排从命名 buffer 的起点传递分布式窗口。
共用同一域的提交会等待前一次
使用完成。释放域和关闭 worker 也会等已接受的调用完成。冷加载时由调用方重新建立
资源，并传入对应窗口。

等待被中断时，worker 会继续持有该任务的参数和通信域。如果底层取消操作无法确认任务已经结束，worker 会保留这些资源，直到原生 worker 成功关闭。
