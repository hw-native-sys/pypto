# `pl.prefetch`

GM 到 L2 的异步预取。它们产出的句柄类型见[特性矩阵](../user/appendix/00-feature-matrix.md)。

Kernel 模式需要在第一次调用前显式初始化 SDMA：

```python
import pypto.torch

pypto.torch.init(runtime="tensormap_and_ringbuffer", enable_sdma=True)
# HBG 使用 runtime="host_build_graph"
```

先在 capture 外编译并准备 kernel，再进行 ACLGraph capture/replay。
初始化时未开启 SDMA 的 context 不能在调用时自动升级。
Program 模式的一次性执行按产物元数据开启 SDMA；复用 `ChipWorker` 时需传入
`enable_sdma=True`。

在支持 SDMA 的 a2a3 runtime 中，每个常驻 AIV worker 使用独立通道。
编译器为每个预取 context 预留 256 字节 UB，覆盖整个函数的生命周期，
避免异步提交和等待覆盖仍存活的计算 tile。预取只改变缓存状态，不改变 tensor 的数值。

`wait` 在设备端通过非阻塞 `Test` 检查完成状态，未完成时约间隔 1 微秒再检查，
每个事件的等待期限为 20 毫秒。超时返回 `False`，不会取消传输；
在途传输的 source/context 存储不能因此提前复用。该等待策略由 PyPTO 适配层实现，
不需要修改 PTO-ISA 的通用 `Wait`。

::: pypto.language.prefetch
    options:
      show_root_heading: false
      filters: ["!^_"]
