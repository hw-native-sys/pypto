# 有界分布式提交

按资源预算提交使用 Simpler 的完成事件、身份容量、`RunHandle.finalized`
及分层提交完成回调。当前钉住的运行时仍使用主线已有的两个 Host 元数据帧，
帧满时完成最早提交的请求。

## 所有权与完成

每个提交帧持有 Host 临时存储、参数包装、retained 通信域及清理回调。
模型参数和调用方拥有的可变状态只保留引用，不复制。一个后台观察线程等待任意
native parent 完成，不轮询，也不为每个请求创建线程。观察线程调用 `result()`，
完成 native 清理和 PyPTO 结果发布之后才归还帧。容量不足的提交等待帧条件变量，
不指定等待最老 handle。显式 `result()` 仍然幂等。

后台回收不代表调用方已经处理错误。查询 `done` 只返回是否终止；未观察的失败仍由
`result()` 抛出，调用方丢弃 handle 时则由 `close()` 报告。
等待被中断或拒绝不会归还帧：必须等 native handle 发布最终结果和清理结果。
缓存的终态 `TimeoutError` 属于任务失败，与等待期限耗尽不同。

持久化提交显式声明前驱成功依赖。后继可以 prepare，但必须等前驱 native 清理、
PyPTO 清理及结果后处理全部成功才能 activate。失败使持久化 worker 进入失败状态，
禁止复用与后继执行；该规则不改变普通独立 Simpler 提交的语义。

未知通信声明或需要重置窗口的请求，在发出分配/拷贝控制操作前仍保守等待前驱。
仅完成预热且通过声明合同校验的独立域池路径不需要这个回退等待。

调用方共享窗口沿用 `CommDomain` 和 `DomainTensor` 的生命周期。绑定同一共享域的
请求等待该域此前的使用结束。提交帧持有域及其缓冲区实参，直到请求完成或整个
worker 完成清理。自动域池分配只覆盖符合条件的程序内域。

## 容量策略

容量在初始化期间、允许提交之前确定：

- 额外 Host 帧预算为 `min(64 MiB, Host 空闲页字节数 / 64)`。
  空闲页使用 `SC_AVPHYS_PAGES * SC_PAGE_SIZE`，不是 Linux `MemAvailable`。
  共享张量存储按操作系统页计费，Python 对象开销属于估算；native identity directory
  的容量同时构成上限。
- 每个非空共享 Tensor storage 的额外副本还计入一个文件描述符；别名只计一次。
  这些可选副本最多使用当前 `RLIMIT_NOFILE` 空余量的四分之一；无法读取占用量时
  只保留基线。在 fork 前还按已声明域的大小和设备缓存预算限制 Host 副本数，
  初始化后再按真实设备 mapping 进一步收缩。
- 所有 rank 先建立一份域。每个 rank 的额外域预算为
  `min(128 MiB, device 空闲字节数 / 64)`。成本计入窗口实际 mapping 大小、
  每个 opaque context 保守预留的 2 MiB，以及预热期间若更大的空闲显存下降值。
  所有 rank 使用最小的可容纳份数。
- 基线的一份帧和域不受额外副本预算限制。没有显存查询能力的模拟器只保留这一份域。
  不需要的 Host 候选帧在允许提交前丢弃。

这些是内部保守缓存预算，不代表硬件极限或最佳性能容量。其他进程会影响显存观测，
Python 对象估算也不是严格的进程内存上限。分配失败仍报错，部分初始化的 worker
走原有清理流程。目前没有运行中扩容；增加容量不会复制模型权重或调用方状态。

## 完整通信资源合同

经审计的产物生产者显式写入 `orchestration/communication_resources.json`。
仅有新生成的 IR 不能证明 external/opaque leaf 的行为，不会自动获得合同授权。
这个带版本的生产者承诺声明：opaque 通信仅访问列出的域，其余内存访问由实际
`TaskArgs` 表达。Host 语法检查拒绝动态域规格、额外 workspace/initializer 控制，
以及 rank 和已声明域 context 之外的标量实参。

签发接口要求显式声明资源完整性、依赖闭包和 rank 数。

```python
from pathlib import Path
from pypto.runtime.communication_resources import write_resource_contract

write_resource_contract(
    Path(artifact_root), complete_resources=True,
    external_dependencies=(Path(custom_library),), world_size=8,
)
```

依赖元组必须覆盖完整外部 application/helper 代码依赖闭包；空元组明确承诺不存在
未绑定的应用代码，不能省略未知依赖。自动摘要范围仅为 `orchestration/`、
`next_levels/` 和 `distributed_meta.json`，包括最终二进制、object、本地头文件与配置。
其他目录下的自定义 include/helper 必须列入外部依赖；平台 runtime 沿用已有的独立
ABI/provenance 检查。没有最终 native 共享库的产物不能启用。
摘要用于绑定生产者承诺，不证明 kernel 行为，也不认证生产者身份。
合同缺失、变化或不受支持时保守回退；域池没有模型专属 hash 授权。
