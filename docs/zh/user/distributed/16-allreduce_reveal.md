# 揭示（The Reveal）：`pld.tensor.allreduce` 一次调用

你已手工把全归约做了三种——mesh、two-phase、ring。现在看内置原语：
`pld.tensor.allreduce(data, signal, op=..., mode=...)` 用一次调用取代整个
调度，而 IR diff 会展示它 lower 成什么：你写过的手工模式，或更好的一个。

> **前置条件：** [13-allreduce_mesh](13-allreduce_mesh.md) ·
> [14-allreduce_two_phase](14-allreduce_two_phase.md) ·
> [15-allreduce_ring](15-allreduce_ring.md)。建议使用 4 个模拟设备。

**建议阅读顺序（Suggested reading order）：** 01 → … → 10 → **11** — 本页为步骤 11。

## 思路（The idea）

揭示纪律到此完成：步骤 04 在你手工构建后揭示了 barrier；本步骤在你构建了
三种全归约后揭示集合通信。手工步骤的意义不在于你应该手工写全归约——而在于
你应该知道*内置原语在什么之间做选择*。

`pld.tensor.allreduce` 是整个阶梯的内置原语：它接收你的窗口和信号，完成
barrier + 跨 rank 归约 + 写回，并把归约后的 slice 交还给你。`mode=` 选择
算法——`"mesh"`（默认）或 `"ring"`。golden 与步骤 08-10 相同；调度一概不见。

## 运行（Run it）

```bash
# 两种模式，P=4（以及 P=2）：
python examples/distributed/11_allreduce_reveal.py -p a2a3sim -d 0,1,2,3
python examples/distributed/11_allreduce_reveal.py -p a2a3sim -d 0,1,2,3 --mode ring
python examples/distributed/11_allreduce_reveal.py -p a2a3sim -d 0,1
```

预期输出：

```text
OK
```

## 走读（Walkthrough）

你的 stage-in 与 stage-out 保留；只有中间被替换：

```python
# Phase 1 — 把本 rank 的 slice 放入自己的窗口槽位。
local = pl.load(x, [0, 0], [1, SIZE])
data = pl.store(local, [0, 0], data)

# Phase 2 — 内置原语：barrier + 归约 + 写回，一次调用。
data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum, mode=mode)

# Phase 3 — 写出：把归约后的 slice 写回本地输出。
recv = pl.load(data, [0, 0], [1, SIZE])
y = pl.store(recv, [0, 0], y)
```

- **信号是你的，其形状告诉你处于哪种模式。** mesh 用 `[nr, 1]`；ring 用
  `[2*(nr-1), nr]`——正是步骤 10 教过的每轮一行信号。工厂把 `nr` 与 `mode`
  都折叠进来，于是一份源码可构建任一变体（步骤 08-10 的 class-form 模式）。
- **内置原语收窄了什么（请读两遍）：** 本示例调用的 InCore 组合形式（显式
  信号）在**两种模式**下都接受完整 `ReduceOp` 家族（`Sum`/`Max`/`Min`/`Prod`）
  与 `FP16`/`FP32`——`Sum`+`FP32` 的限制属于 **HOST 内置** ring 路径，不属于
  这种 InCore 形式。ring 内置 ST 套件已覆盖 `Max`/`Min`/`Prod` 与 `FP16`。

### IR diff（教学工件）

用 pass dump 编译，并把本步骤的 lower 后 IR 与你的手工程序做 diff：

- `--mode mesh` 展开为**步骤 08 的模式**：在 `[nr, 1]` 信号上的 ready
  barrier，然后 `remote_load` + 累加块——你写过的 mesh，可能被分块成
  UB 大小的 tile。
- `--mode ring` 展开为**步骤 10 的模式**：`[2*(nr-1), nr]` 信号上的
  `2*(nr-1)` 轮邻居就绪握手，`N/P` 分块传输——你写过的 ring，推送/读取侧
  由 lowering 决定。

同样的 golden、同样的信号约定、同样的四阶段形态——diff 就是"你的调度，由
编译器表达"，这正是整个教学点。

**成本卡（每 rank）：** 取决于所选模式——`(P-1) * N`（mesh）或
`2 * (P-1) / P * N`、`N/P` 每步（ring）。内置原语选择算法；你仍选择模式。

## 边界情况（Edge cases）

> **致命陷阱——信号形状与模式不匹配。** 用 `[nr, 1]` 信号配 `mode="ring"`
> （或用 ring 形状配 mesh），lowering 会拒绝——信号*就是*内置原语运行哪种
> 调度的契约。**修复：** mesh → `[nr, 1]`；ring → `[2*(nr-1), nr]`。

| 现象 | 可能原因 | 修复 |
| ---- | -------- | ---- |
| 关于信号形状的编译错误 | 信号形状与模式不匹配 | mesh `[nr, 1]`，ring `[2*(nr-1), nr]` |
| HOST 路径上 `mode="ring"` 配 `Max`/`Prod`/`FP16` 被拒 | `Sum`+`FP32` 限制仅限 HOST 内置 ring | 用本示例的显式信号 InCore 形式，或 mesh |
| 每个 rank 结果相同但与 torch 和不同 | 归约顺序不同（非 bug） | 用容差比较 |
| 结果是你自己的 slice，未归约 | 调用结果未重新绑定（`data =`） | `data = pld.tensor.allreduce(data, signal, ...)`——就地重绑定 |
| 内置原语比你的手工 mesh 慢 | 载荷过小：mesh 适合小消息 | 载荷 ≳ 16 KiB 用 ring（见 `01-collectives.md`） |

## 参见（See also）

- [05-tutorials](05-tutorials.md) — 教程索引（本步骤 = 第 11 行）
- [01-collectives](01-collectives.md) §AllReduce — 两种模式与信号形状的参考
- [13-allreduce_mesh](13-allreduce_mesh.md) / [15-allreduce_ring](15-allreduce_ring.md) — 每种模式 lower 成的样子
- [02-primitives](02-primitives.md) — 内置原语所基于的底层
- 下一步：[05-tutorials](05-tutorials.md) 中步骤 12-16 覆盖其余集合通信
