# Host

host 侧唯一一个能压过其他所有开销的成本：拷贝了本来不需要动的数据。

> **前置**：能跑起一个 kernel 就够了。

## 这件事要先查，不是最后查

`run()` 分别报告 host 与 device 时间。当 host 那一段是大的那个时，[00](00-swimlane.md)–[05](05-memory.md) 页里没有任何东西能动你的数字 —— 它们调的全是并非瓶颈的设备侧工作。

常见成因并不隐晦。一个在循环里被调用、每次都带同一个大权重参数的 kernel，会**每次调用都上传**那份权重：

```text
每次调用：  H2D 权重 ──► 计算 ──► D2H 结果
            ▲──────────── 同样的字节，每一轮都传一遍
```

## 让常驻的数据真的常驻

`ChipWorker.alloc_tensor` 分配常驻设备内存，返回一个 `DeviceTensor` 句柄，编译后的程序在任何接受 `torch.Tensor` 的位置都接受它。运行时把这块 buffer 当作已经常驻，对该参数跳过 H2D 与 D2H。

```python
import torch
from pypto import ir
from pypto.runtime import ChipWorker, RunConfig

compiled = ir.compile(MyKernel)

with ChipWorker(config=RunConfig(platform="a2a3")) as w:
    weight = w.alloc_tensor((1024, 4096), torch.float16, init=host_weight)  # 只上传一次
    for batch in batches:
        out = torch.empty(batch.shape[0], 4096, dtype=torch.float16)
        compiled(batch, weight, out)                                        # 不再重传
    w.free_tensor(weight)
```

**代价 —— 三份从此归你的义务：**

- `DeviceTensor` **永远不会**被拷回 host。如果 kernel 往里写了，用同一个 worker 上的 `w.copy_from(host_ptr, t.data_ptr, t.nbytes)` 读出来。
- 在 worker 关闭之前用 `w.free_tensor(t)` 释放，否则这块内存会泄漏到 worker 的生命期结束。
- 只有分配它的那个 worker 能使用它。

**怎么确认：** host 那一段变小，device 那一段不动。如果 device 时间也变了，说明还有别的东西一起变了。

## 还有什么是常驻的

同样的道理适用于任何活得比一次调用更久的东西：

| 数据 | 为什么留下 | 例子 |
| ---- | ---------- | ---- |
| 权重 | 每次调用都读，从不写 | 上面那段代码 |
| KV cache | 上一次调用写，下一次调用读 | `examples/runtime/multi_program_kv_cache.py` |
| 临时 / workspace | 根本不离开设备 | 分配一次，每次调用都传进去 |

KV cache 是那种「常驻不只是优化」的情形 —— 把它来回拷贝会主导一整个 decode step。`rt.alloc_tensor(...)` 让一块 buffer 跨越注册在同一个 worker 上的多个程序存活。

## 只注册一次

第二项 host 开销是注册。每次调用都编译并注册一遍程序，等于反复付 setup；register-once、dispatch-many 的模式只付一次：

```python
from pypto.runtime import benchmark

stats = benchmark(compiled, [a, b, c], rounds=100, warmup=3,
                  platform="a2a3", device_id=0)
print(stats.device_wall_us_median, stats.device_wall_us_min)
```

`benchmark` 拥有这个循环：它把*已编译*的程序注册一次，然后派发 `rounds` 次廉价 launch，中间没有逐轮的注册或加载，并聚合每次的 `[STRACE]` 计时标记。这也是拿到稳定 device 数字的正确方式，因为它排除了你并不想测的那部分 setup。

`examples/runtime/explicit_dispatch.py` 用三种真实形态展示了同一个结构 —— 推理服务、训练循环，以及一次 register/dispatch 开销检查。

## 参见

- [快速上手 § DeviceTensor](../00-getting_started.md#在-worker-上复用权重devicetensor)
  —— 参考性说明，含显式派发 API。
- [多卡度量](07-distributed.md) —— 当拷贝发生在 rank 之间时。
