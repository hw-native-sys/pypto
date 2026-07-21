# L3 持久执行

通常情况下，准备好的分布式程序会复用同一个 Simpler worker，但每次 dispatch
仍会重新进入一次 `Worker.run()`。因此，包含通信窗口的程序会在每次调用时申请并
释放 CommDomain。

使用 `persistent=True` 可以让一次 L3 orchestration 在 prepared worker 的整个
生命周期内保持运行：

```python
with decode.prepare(persistent=True) as worker:
    for _ in range(100):
        worker(x, weights, output)
```

持久模式需要显式开启，默认的 `prepare()` 行为保持不变。

## 生命周期

PyPTO worker 会启动一个后台 orchestration，并通过 Python Queue 向它发送请求。
首次使用某个 generated CommDomain 时会申请物理 window，后续调用则获得同一个
handle 的 retained lease。关闭 prepared worker 时，orchestration 停止，所有保留
的 domain 都通过 Simpler 原有的 `Worker.run()` 清理路径释放。最终 drain 或 domain
释放过程中发生的错误会由 `close()` 抛出，而不会被后台线程静默丢弃。

生成的 HOST orchestration entry 接受内部参数 `_domain_provider`。普通 dispatch
不传该参数，仍然调用 `orch.allocate_domain`；持久 dispatch 则传入一个按 compiled
program 和 generated domain name 隔离的 provider。已有 generated artifact 必须
重新生成后才能使用持久模式。

## Fresh-window 语义

当前 `alloc_window_buffer` 的语义是：每次 dispatch 都获得全零的新 scratch
storage。复用物理 allocation 时必须保持这一语义，尤其要正确处理单调递增的
notify/wait counter。

每次复用某个 program 的 retained domain 前，PyPTO 会同步清零所有参与 worker
上的本地 window。芯片 worker fork 前会准备一个只读的 1 MiB host zero chunk；
更大的 window 会分块重复拷贝。第一次 dispatch 使用 runtime 新申请并初始化的
window，不执行额外 reset。

这样不需要模型增加特定的 epoch 参数，但 reset copy 会计入每次重复请求的 host
开销。在生产环境开启持久模式前，应将它和动态申请 CommDomain 的开销进行实测
比较。

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

该实现不修改 Simpler，而是使用公开的 domain、copy 和 scope API，并调用私有的
orchestration drain 与 Worker cleanup API，在保持外层 `Worker.run()` 存活的同时，
为每个 Queue 请求建立设备完成边界。后续应由公开的 Simpler request-boundary API
封装这段 drain 和 cleanup 流程。
