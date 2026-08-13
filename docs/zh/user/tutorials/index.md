# 教程

六篇手把手，每篇结束时你都能得到一个跑得起来的东西。

> **前置**：[语言](../language/index.md) —— 至少读过 [类型](../language/00-types.md)、[函数](../language/01-functions.md) 与 [作用域与放置](../language/04-scopes.md)。

## 本章与其余各章的区别

[语言](../language/index.md) 和 [算子](../ops/index.md) 是按**能力**组织的 —— 一页讲一个特性，方便你查。本章按**任务**组织：每页从零做出一件东西，而且沿途每一步都是一个你能跑的程序。

这意味着各页会**有意**重复彼此的特性。想要 `pl.at` 的完整面，去读 [作用域与放置](../language/04-scopes.md)；想写一个 matmul，读 [分块 matmul](02-matmul.md) —— 它会用 `pl.at`，但不会把它讲全。

## 两条线

**写算子**（00–03）—— 怎么表达计算。

| 页面 | 你最终得到 | 预计耗时 |
| ---- | ---------- | -------- |
| [第一个算子](00-elementwise.md) | 一个可跑的逐元素 kernel，与 torch 对拍 | 约 20 分钟 |
| [规约与 softmax](01-reduction-softmax.md) | 一个数值稳定的 softmax | 约 30 分钟 |
| [分块 matmul](02-matmul.md) | 一个 K 轴分块的 matmul | 约 40 分钟 |
| [混合 kernel](03-mixed-kernel.md) | cube 与 vector 在同一个作用域里并发工作 | 约 40 分钟 |

**塑形调度**（04–05）—— 怎么控制运行时拿它做什么。

| 页面 | 你最终得到 | 预计耗时 |
| ---- | ---------- | -------- |
| [塑形任务图](04-task-graph.md) | 一个依赖图完全可控的多任务程序 | 约 30 分钟 |
| [调度调优](05-scheduling-tuning.md) | 一套能套用到自己 kernel 上的度量流程 | 约 40 分钟 |

## 阅读顺序

```text
00-elementwise ──► 01-reduction-softmax ──► 02-matmul ──► 03-mixed-kernel
      │                                                          │
      └──────────────────────────► 04-task-graph ──► 05-scheduling-tuning
```

算子线是累积的 —— 每页都假定你已掌握前一页的 tile 词汇。调度线只需要 `00`：它讲的是 kernel **之间**那张图的形状，与任何单个 kernel 算什么无关。

## 你的算子跑在哪个单元上

一个 core group 配一个 **cube** 单元（AIC）与若干 **vector** 单元（AIV）。由哪个执行不是你逐次调用去选的，而是由算子本身决定：

| 算子族 | 单元 | 讲解位置 |
| ------ | ---- | -------- |
| `matmul`、`matmul_acc`、`gemv` | Cube（AIC） | [分块 matmul](02-matmul.md) |
| 逐元素、规约、广播、cast | Vector（AIV） | [00](00-elementwise.md)、[01](01-reduction-softmax.md) |
| `tpush_to_aiv`、`tpop_from_aic`、`aiv_shard`、`aic_gather` | 两者，按构造 | [混合 kernel](03-mixed-kernel.md) |

只由单一算子族构成的 kernel 会占住一个单元、让另一个闲着。这个观察正是 [混合 kernel](03-mixed-kernel.md) 存在的理由；在它之前的各页写的都是单单元 kernel。完整清单见 [算子](../ops/01-catalog.md)。

## 跑这些示例

每页都指名 `examples/` 下的一个文件。`RunConfig.platform` 默认就是 `"a2a3sim"`，所以它们都不需要真机：

```bash
python examples/beginner/02_elementwise.py
python examples/advanced/03_mixed_kernel.py --mode staged
```

多数配套文件直接沿用这个默认值。`03_mixed_kernel.py` 是例外：它用 `--mode` 在几种 split 形式间切换，用 `--platform` 改目标平台。

## 参见

- [语言](../language/index.md) —— 同样的特性，按查阅方式组织。
- [任务与定序](../tasks/index.md) —— [04](04-task-graph.md) 背后的参考资料。
- [算子](../ops/index.md) —— 算子目录。
