# 动态 Rank 数量（Dynamic Rank Count）：同一份源码，任意 P

把步骤 06 的环形移位做成**与 rank 数量无关**：`NR = pl.dynamic("NR")` 把
world 大小命名为一个运行期维度，于是同一份源码可以针对 P=2、P=3、P=4……
编译并运行——只需改 `-d`，永远不改程序。

> **前置条件：** [11-put_get](11-put_get.md)。任意 ≥ 2 个设备（本页示例用
> 2、3、4 个模拟设备）。

**建议阅读顺序（Suggested reading order）：** 01 → 02 → 03 → 04 → 05 → 06 → **07** — 本页为步骤 07。

## 思路（The idea）

此前每个步骤都把 `N_RANKS = 2` 写死：主机 world 张量是 `[N_RANKS, 1, SIZE]`，
golden 也只按两个 rank 写。但 kernel 其实从未依赖过这个数量——它在运行期从
`pld.nranks(ctx)` 读取、按它循环、并用 `% nranks` 计算对端。被钉死的只有
主机的 world 形状。

`pl.dynamic("NR")` 解开这个钉死。`NR` 是**具名运行期维度**：它告诉编译器
"这个 extent 在程序被调用时解析，而不是在书写时"。主机签名变成
`x: pl.Tensor[[NR, 1, SIZE], pl.FP32]`，于是同一份源码可以为任意 `-d` 编译
——rank 数量从程序中消失了。

为什么此刻重要：后续步骤会彼此比较不同的集合通信算法，而在两个 rank 时，
其中几种算法会坍缩为同一次交换——它们的差异只有在四个 rank 时才可观测。
本步骤就是那座桥：同一份源码服务任意 world 大小，正是那些 P=4 对比所依赖
的，且无需为每个程序做 rank 数量修改。

## 运行（Run it）

```bash
# 两个、三个或四个 rank——同一份源码，只改 -d：
python examples/distributed/07_dynamic_rank_count.py -p a2a3sim -d 0,1
python examples/distributed/07_dynamic_rank_count.py -p a2a3sim -d 0,1,2
python examples/distributed/07_dynamic_rank_count.py -p a2a3sim -d 0,1,2,3
```

预期输出：

```text
OK
```

## 走读（Walkthrough）

与步骤 06 相比，唯一的不同是 `NR` 声明与主机签名；kernel 原封未动。

```python
SIZE = 64
NR = pl.dynamic("NR")          # rank 数量是一个运行期维度
```

```python
@pl.jit.host
def ring_get(
    x: pl.Tensor[[NR, 1, SIZE], pl.FP32],
    y: pl.Out[pl.Tensor[[NR, 1, SIZE], pl.FP32]],
):
    src_buf = pld.alloc_window_buffer([1, SIZE], dtype=pl.FP32)
    dst_buf = pld.alloc_window_buffer([1, SIZE], dtype=pl.FP32)
    signal_buf = pld.alloc_window_buffer([1, 1], dtype=pl.INT32)
    for r in pl.range(pld.world_size()):
        src = pld.window(src_buf, [1, SIZE], dtype=pl.FP32)
        dst = pld.window(dst_buf, [1, SIZE], dtype=pl.FP32)
        signal = pld.window(signal_buf, [1, 1], dtype=pl.INT32)
        per_rank_get(x[r], y[r], src, dst, signal, device=r)
```

- **`NR` 是符号化的。** `pl.dynamic("NR")` 声明 leading world 维在运行期
  解析。`pld.world_size()` 已经*返回*运行期数量；`NR` 是形状侧对它的命名。
- **kernel 保持原样。** `get_step`（以及 `put_step`）用 `pld.nranks(ctx)`
  约束循环，并用 `peer = (my_rank ± 1) % nranks` 计算对端——全部运行期。
  程序里没有任何 `N_RANKS`。
- **`-d` 是唯一的旋钮。** `main()` 从 `-d` 取设备数量（`len(device_ids)`），
  据此生成 world 张量 `(P, 1, SIZE)`，并从实际 `P` 推导 golden：

```python
device_ids = [int(d) for d in args.device.split(",")]
x = torch.randn((len(device_ids), 1, SIZE), dtype=torch.float32)
...
expected = expected_ring(x, get_mode)   # y[r] = x[(r+1) % P]（get）、x[(r-1) % P]（put）
```

每次调用编译一次（本次运行的 `-d` 确定具体的 `P`），同一份源码服务任意
world 大小——这正是 P=4 集合通信对比所依赖的能力。

**成本卡片：** 与步骤 06 相同——一步，每个 rank 与一个对端交换一个 `SIZE`
字节的 slice。rank 数量只改变环在*哪里*回绕，不改变每 rank 的成本。

## 边界情况（Edge cases）

> **致命陷阱——在主机形状里留下写死的 rank 数量。** 主机注解里必须使用
> `NR = pl.dynamic("NR")`。如果 world 形状仍是带 `N_RANKS = 2` 的
> `[N_RANKS, 1, SIZE]`，程序又被钉死为两个 rank——更大的 `-d` 会在编译期
> 报形状不匹配。**修复：** 把主机签名里的常量全部替换为 `NR`。

| 症状 | 可能原因 | 修复 |
| ---- | -------- | ---- |
| P 增大时编译期形状不匹配 | 运行期维度未使用 `pl.dynamic` | 包裹它：`NR = pl.dynamic("NR")`，主机签名用 `[NR, ...]` |
| 只在 P > 2 时结果错误 | 对端算术出现负数被除数（`(my_rank - 1) % nranks`） | 使用 `(my_rank + nranks - 1) % nranks`——永不为负 |
| P > 2 时 `get` 读到陈旧数据 | 握手目标错误 | 通知读取你的 rank（上一个）；等待你读取的 rank（下一个） |
| golden 落后一步 | pull 与 push 混淆 | `get` 模式：`y[r] = x[(r+1) % P]`；`put` 模式：`y[r] = x[(r-1) % P]` |
| 每个 P 都要重新编译，感觉浪费 | 编译产物是 P 特定的 | 用新的 `-d` 重跑；*源码*从不改变 |

## 参见（See also）

- [05-tutorials](05-tutorials.md) — 教程总览（本步骤 = 第 07 行）
- [11-put_get](11-put_get.md) — 本环的固定 P=2 版本（步骤 06）
- [02-primitives](../distributed/02-primitives.md) §系统基座 + §Put 与 Get —
  `world_size`/`nranks`、分块
- [00-getting_started](../00-getting_started.md) — `pl.dynamic(...)` 动态维
- 下一步：[05-tutorials](05-tutorials.md) — 步骤 08–16（集合通信，P=4）为规划中
