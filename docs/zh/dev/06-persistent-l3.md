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

这种整 buffer staging 是当前 Simpler Buffer API 的限制：`copy_to` 根据源 `Buffer`
决定拷贝长度，同时不提供目标 offset 或公开的 Buffer subview。PyPTO 生成的 domain
会由具名 buffer 完整覆盖 window；如果 artifact 含有未命名的 window 空隙，reset
会直接拒绝，因为当前 Buffer API 无法将该空隙恢复为 fresh-window 状态。

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

`DistributedWorker.submit()` 和 Simpler 仍会串行构建 graph。请求被接受后，持久
模式与普通模式遵循相同的有界异步分发约定：PyPTO 最多发布两个 metadata frame，
实际可重叠的 device run 数量由 backend 协商出的 depth 决定。

## Runtime 依赖

该实现不修改 Simpler。每个请求都使用公开的 `Worker.submit()` boundary，其
native handle 会一直挂在 PyPTO 的有界 dispatch handle 上直到完成。PyPTO 会让
retained CommDomain 脱离 Simpler 的 per-run release set，并在 prepared worker
关闭时统一释放。目前该保留机制仍依赖 Simpler 私有的 Worker 级 live-domain
registry 和当前 run-resource journal（`_live_domains` 和
`_building_run_resources.live_domains`）；后续应由公开的 retention API 封装该
lifecycle。

## CLUSTER 程序：显式共享全局窗口

CLUSTER_0 支持同步的多程序会话。启动前用 `extra_compiled` 准备全部程序，
通过 `run(compiled, ...)` 选择程序；每次调用都有独立的完成栅栏（completion
fence）。全局域默认按程序隔离，只有显式映射到同一个会话共享键才复用：

```python
from pypto.ir import SharedDomainBinding

with compute.prepare(
    extra_compiled=[communication],
    persistent=True,
    reset_persistent_windows=False,
    shared_domains=[
        SharedDomainBinding(compute, domain=0, key="activations", slots={0: "values"}),
        SharedDomainBinding(communication, domain=0, key="activations", slots={0: "values"}),
    ],
) as worker:
    worker.run(compute, lhs, rhs, echo)
    worker.run(communication, peer_values)
```

第一个程序向各 rank 的全局窗口写入计算结果，返回时所有写入已排空，第二个
程序随后通过 TLOAD 读取。通信程序没有接收已保存数据的 host 副本。一个编译
程序对应一次 run；会话不会在程序内部切分相位，也不增加前端窗口拷贝接口。

通过不可变的 `compiled.comm_domains` 描述查看域和槽的序号。生成的名字仅供
内部使用，不同程序的名字可以不同。`slots` 将程序内槽序号映射到规范物理槽名，
必须完整且一一对应；规范槽名允许重排槽次序。复用前校验有序成员、拓扑、
runtime/profile、窗口和对齐后的槽大小，以及 dtype、shape、strides、字节偏移。
相同大小不会自动共享，不同 key 和未配置的域继续隔离。共享布局要求槽大小
静态可知。prepare 会保存配置快照，之后修改原映射不会改变会话；共享配置不
进入 JIT 编译身份。

会话安装一个有界的聚合源码包，每节点使用一份规范 runtime helper 和一个
fork 前 token pool。各程序共享真实的 R1 retained domain，以及以
`(domain_id, generation)` 为键的 ordering 状态。槽翻译只选择实际 runtime
Buffer，不包装或重命名导入的全局域 view。helper/bootstrap 合同不匹配会在启动
前失败。冷 schema-4 产物保存域描述；缺少这些描述的旧产物须重新生成后才能
显式共享。

默认在本 run 的首个任务提交前清零复用的整个全局映射，包括 signals 和 padding；
runtime 控制拷贝每块最多 8 MiB。新分配不重复清零。
`reset_persistent_windows=False` 保留 DATA、signals 和用户 epoch 的原内容。
使用 signal/credit 的 kernel 协议须自行管理重置或 epoch 推进；上述先计算再
TLOAD 的用例仅依赖 run fence。ordering token 的存储身份跨程序保持稳定。

正常关闭在最后一次 fence 后显式释放 retained domain。失败关闭先排空 Worker
树，再释放域、I/O 和 fork 前状态的引用。无法确认关闭完成时保留 journals，拒绝
新的 run，并允许重试 close。CLUSTER 仅支持同步 run；保留机制使用 R1 公开的
`retain_after_run` 合同，不从私有 runtime journal 中摘除域。

验证时，SIM 全局域的 ranks 必须位于同一个 host/POSIX 共享内存命名空间。
SIM 后端通过 `shm_open` 导入 peer 窗口，不提供跨 host 的 SIM 全局窗口。
真实跨机验证使用 A3 fabric profile 和物理设备；普通远端张量 I/O 是另一项能力。
