# 内存图

看清片上放了什么、在哪、活多久。

## 概念

片上缓冲是 kernel 争抢的稀缺资源。放不下时编译器会告诉你 —— 但真正有意思的问题通常在更早：*是什么占着这块空间？* `pypto.tools.memory_map` 把分配画成 HTML 来回答它：横轴地址、纵轴生命期，旁边是 IR。

它的输入是一份 **pass dump**，不是一次运行。什么都不会被执行；这是一张「编译器决定了什么」的图。

## 快速上手

先产出 dump，再渲染：

```python
from pypto.ir import PassDumpLevel
from pypto.runtime import RunConfig

compiled = kernel.compile(*args, config=RunConfig(dump_passes=PassDumpLevel.EXPLICIT))
```

`compile()` 自己不打印任何东西，所以让它告诉你 dump 落在哪：

```python
print(compiled.output_dir)
```

然后把工具指向那个目录：

```bash
OUT=build_output/<program>_<timestamp>          # 上一行打印出来的路径
python -m pypto.tools.memory_map "$OUT/passes_dump/NN_after_SomePass.py" -o map.html
```

这里有两点要紧。**要用 `compile()` 而不是 `lower()`** —— `lower()` 只跑 pass 并把 `Program` 交回来，什么都不写，因此不会产出这个工具要读的 `passes_dump/`。以及 **`EXPLICIT`**，它会解析出隐式 tile layout 与 window buffer，那正是工具确定要画多大所需要的。

## 机制

### 该打开哪一份 dump

分配是很晚才定下来的，所以值得打开的是内存相关 pass 之后的那几份：

| dump | 显示 |
| ---- | ---- |
| `MaterializeSemanticAliases` 之后 | 属于语义而非优化的 must-alias 关系 |
| `MemoryReuse` 之后 | 机会性复用这一遍合并了什么 |
| `AllocateMemoryAddr` 之后 | 最终偏移 —— 多数问题要看的就是这张 |

打开更早的 dump 不算错，只是那时的分配还没定下来。

### 怎么读

值得找的有两样，都不是「是不是满了」：

- **活得远比被用到的时间长的 tile。** 一根很长的条，实际读取只集中在很短一段 —— 这是重构的候选，正是那段生命期挡住了复用。
- **余量。** 再加一级 `pl.pipeline` 或更深的跨核环能不能放下，问的是缝隙，不是总量。

### PTOAS 的注意事项

`memory_planner=PTOAS` 下编译器**跳过 `AllocateMemoryAddr`**，把寻址交给 ptoas。于是 pass dump 里没有已分配的偏移，这个工具无从绘制。这是该规划器的性质而非故障 —— 改用端到端对比，并见[内存](../performance/05-memory.md)。

## 边界情况

| 现象 | 原因 | 修法 |
| ---- | ---- | ---- |
| **图是空的或几乎是空的** | dump 早于分配，或用了 `memory_planner=PTOAS` | 打开更晚的 dump；或接受寻址归 PTOAS 管 |
| **压根没有 `passes_dump/`** | 不带 `dump_passes=` 的 `lower()` 不写产物 | 传 `dump_passes=PassDumpLevel.EXPLICIT` |
| **layout 显示为未解析** | dump 是 `CONCISE` | 用 `EXPLICIT` 重新 dump |

## 参见

- [内存](../performance/05-memory.md) —— 运行时那四个环，以及本工具所画的片上预算。
- [InCore 函数调优](../performance/04-incore.md) —— 消耗这份预算的一侧。
- [调试](00-debugging.md) —— pass dump 的另一个读者。
- [AllocateMemoryAddr](../../dev/passes/35-allocate_memory_addr.md) —— 这张图所呈现的那个 pass 的输出。
