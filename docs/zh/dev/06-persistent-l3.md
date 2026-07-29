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

PyPTO worker 会启动一个后台 dispatcher，并通过 Python Queue 向它发送请求。
每个请求都在独立的 Simpler `Worker.run()` completion fence 中执行。首次使用某个
generated CommDomain 时会申请物理 window，后续调用则获得同一个 handle 的
retained lease。关闭 prepared worker 时，dispatcher 停止并释放所有保留的
domain。请求或 domain 释放过程中发生的错误会抛给调用方，而不会被后台线程
静默丢弃。

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

默认开启 reset 时，PyPTO 会在复用前同步清零所有参与 worker 上的本地 window。
芯片 worker fork 前会准备一个只读的 1 MiB host zero chunk；更大的 window 会分块
重复拷贝。reset copy 会计入每次重复请求的 host 开销。

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

请求通过一个 Queue 串行执行。持久模式不会让同一个 worker 并发执行多个 L3 DAG。

## Runtime 依赖

该实现不修改 Simpler。每个 Queue 请求都使用公开的 `Worker.run()` completion
boundary。PyPTO 会让 retained CommDomain 脱离 Simpler 的 per-run release set，
并在 prepared worker 关闭时统一释放。目前该保留机制仍依赖 Simpler 私有的
live-domain 和 deferred-release hook；后续应由公开的 retention API 封装该
lifecycle。
