# 性能调优：测量收益

每次调优改动前后都要测量。下面每个工具回答不同的问题 —— 端到端墙钟、编译耗时、
逐 kernel cycle、逐 task 调度，或一次改动在 codegen 里产出了什么。

## Benchmark —— 端到端设备墙钟

`pypto.runtime.benchmark` 是最核心的“到底有多快”数字：多次 launch 下的稳态设备墙钟。

```python
from pypto.runtime import benchmark

stats = benchmark(compiled, args, rounds=100, warmup=3)
print(stats.device_us_mean(), stats.device_us_median(), stats.device_us_stdev())
```

- **稳态。** 把 `compiled` 只**注册一次**，然后循环已绑定的句柄 —— 每次 launch 只
  重付参数 coercion + dispatch，不再付 register/load。
- **测什么。** `device_wall_us` —— orchestrator 的 `orch_start` / `orch_end` 之间的
  片上时间，不受 host 侧建参影响；同时也记录 `host_wall_us`。`warmup`（默认 3）次
  被丢弃；`rounds`（默认 100）次被测量。
- **返回** `BenchmarkStats` —— 逐轮样本（`per_round("device")`）加聚合
  （`device_us_mean/median/min/max/stdev`）与 `print_mean_tree()`。
- **L2 与 L3。** 支持 `CompiledProgram`（L2）与 `DistributedCompiledProgram`
  （L3，逐轮跨 rank 取最大）。

用 benchmark 拿总数字；用下面的工具解释*为什么*是这个数字。

## 编译期 Profiling

记录编译流水线各阶段的**编译**墙钟 —— 前端 parse、passes（逐 pass）、codegen
（逐 kernel + 编排）、以及设备执行。用它回答*“为什么编译慢”*、*“哪个 pass 占编译
时间最多”*，而非测量 kernel 的片上速度（那要用下面的 in-core profiling 与泳道图）。

**开启**（任选其一）：

```python
ir.compile(program, profiling=True)     # 写出 output_dir/report/pipeline_profile.{txt,json}
```

```bash
PYPTO_COMPILE_PROFILING=1 python3 my_program.py
```

**不同入口捕获的阶段：**

| 入口 | 记录的阶段 |
| ---- | ---------- |
| `ir.compile(profiling=True)` | 仅 `passes`（逐 pass）+ `codegen`（子阶段） |
| `runtime.run(config=RunConfig(compile_profiling=True))` | 完整层级 —— 另含 `parse`、`golden_write`、`device_execution` |

**解读摘要**（`pipeline_profile.txt`）—— 树状，逐阶段的秒数与占总比：

```text
PyPTO Compile Profile
Total: 2.847s
  parse                    0.023s ( 0.8%)
  passes                   1.204s (42.3%)
    UnrollLoops            0.012s
    ConvertToSSA           0.034s
    AllocateMemoryAddr     0.156s
  codegen                  0.418s (14.7%)
    kernel_codegen:my_kernel   0.312s
    orchestration_codegen      0.106s
  device_execution         1.202s (42.2%)   # 仅 runtime.run() 才有
```

`pipeline_profile.json` 携带同样的树（`total_seconds` + 嵌套 `stages`），便于跨次
运行的程序化 diff。注意 `device_execution` 只是一个粗粒度总时长 —— 它**不**按 kernel
或 task 分解；那种分辨率来自泳道图与 in-core profiling。

## In-Core Profiling（msprof op-simulator）

*TODO：*

- 通过昇腾 msprof op-simulator 得到 cycle 精确的逐 kernel trace。
- 参考 `incore-profiling` skill 工作流。

## 设备侧泳道图

泳道图是在真实硬件上捕获的**逐 task 时间线** —— 直观看到跨核任务是重叠还是停顿的
最快方式。用 `enable_l2_swimlane` 开启，然后从泳道里找空隙与不均衡。

完整参考 —— 开启方式、输出、如何解读 —— 见
**[DFX › 泳道图](../dfx/01-swimlane.md)**。

## 跨分支 Codegen 对比

*TODO：*

- 在 `origin/main` 与你的分支之间 diff `.pto` / pass dump。
- 参考 `compare-codegen` skill 工作流。

## 参见

- 开发者参考：[`dev/01-compile-profiling.md`](../../../dev/01-compile-profiling.md)、[`dev/04-simulator-trace-cleaning.md`](../../../dev/04-simulator-trace-cleaning.md)
