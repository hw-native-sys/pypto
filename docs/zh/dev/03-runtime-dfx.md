# 运行时 DFX（Design For X）开关

PyPTO 将 Simpler 的五项运行时诊断子功能以独立开关的形式暴露在
[`RunConfig`](../../../python/pypto/runtime/runner.py) 上。每个开关都映射到
Simpler 的 `CallConfig` 字段，以及 `tests/st/conftest.py` 中对应的 pytest
flag。字段名与 Simpler 保持一致；旧拼写 `enable_l2_swimlane` /
`--enable-l2-swimlane` 仍可用，见[已弃用别名](#已弃用别名)。

## 开关映射表

| `RunConfig` 字段 | pytest flag | `CallConfig` 成员 | `dfx_outputs/` 下产物 | 后处理工具 |
| ---------------- | ----------- | ----------------- | --------------------- | ---------- |
| `enable_chip_swimlane: int` | `--enable-chip-swimlane`（= `4`）/ `--chip-swimlane-level N` | `enable_chip_swimlane`（`0` 关 .. `4` 全量） | `chip_swimlane_records.json` | `swimlane_converter` → `merged_swimlane_*.json` |
| `enable_dump_args: int` | `--dump-args [LEVEL]`（裸 flag = `1`） | `enable_dump_args`（`0` 关，`1` 部分，`2` 全量） | `args_dump/{args_dump.json,bin}` | `dump_viewer`（手动） |
| `enable_pmu: int` | `--enable-pmu [N]`（裸 flag = `2`） | `enable_pmu`（`0` 关，`>0` 事件类型） | `pmu.csv` | — |
| `enable_dep_gen: bool` | `--enable-dep-gen` | `enable_dep_gen` | `deps.json` | `deps_viewer`（手动） |
| `enable_scope_stats: bool` | `--enable-scope-stats` | `enable_scope_stats` | `scope_stats/scope_stats.jsonl` | `scope_stats_plot`（手动） |

五个开关**完全正交**，可任意组合。任一开启时自动将
`RunConfig.save_kernels` 强制设为 `True`，确保 `<work_dir>/dfx_outputs/`
目录在 run 结束后保留。

### Swimlane 采集等级

`enable_chip_swimlane` 是**等级**而非开关。每个等级在 runtime 采集器里都是真实的
判断分支，低等级不会打点高等级才有的数据，事后也无法通过后处理补回：

| 等级 | 新增内容 | 解锁能力 |
| ---- | -------- | -------- |
| `0` / `False` | — | 关闭采集 |
| `1` | AICore 逐任务 start / end + task record buffer | 逐任务泳道 |
| `2` | + AICPU 打点的 dispatch / finish | `[dispatch, start]` 取任务间隙 |
| `3` | + scheduler 主循环 phase 记录 | `simpler_setup.tools.sched_overhead_analysis`、Toolkit 插件的 Scheduler View |
| `4` / `True` | + orchestrator phase 记录 | Toolkit 插件的 AICPU Orchestrator 视图 |

`True` 请求等级 `4` —— 与裸 `--enable-chip-swimlane` 请求的是同一件事，PyPTO
与 runtime harness 皆然。超出范围的等级由 `RunConfig` 抛 `ValueError`。

在 pytest 侧，裸 flag 与等级是**两个**选项（`--enable-chip-swimlane` 与
`--chip-swimlane-level N`），而不是一个可选带值的选项。可选带值的 flag 会吞掉
后面那个 token，`pytest --enable-chip-swimlane tests/st/runtime/` 会报
`invalid int value: 'tests/st/runtime/'`。拆成两个选项后，裸 flag 与参数顺序无关。

## 产物契约

runtime 把所有产物写到 `CallConfig.output_prefix` 指向的同一目录。
PyPTO 将该 prefix 设为 `<work_dir>/dfx_outputs/`，其下的子路径按上表
固定。多数产物是 prefix 下的扁平文件；`scope_stats` 例外——其采集器
写入 `scope_stats/` 子目录，内含 `scope_stats.jsonl`。Simpler 的
`CallConfig::validate()` 在任一 flag 开启但
`output_prefix` 为空时拒绝调用；PyPTO 在 Python 侧镜像该契约，
`execute_on_device` 会**先于** C++ 边界抛 `ValueError`，让 traceback
直接指向调用方代码。

### L3（分布式）：每次 dispatch 一个子目录

分布式 run 会向多张卡下发，且同一张卡在一次 host 编排中可能收到多次
dispatch——若共用同一 prefix，各次 dispatch 会互相覆盖同名产物。因此 L3
路径下 PyPTO 按 dispatch 对 prefix 做命名空间隔离：

```text
<work_dir>/dfx_outputs/
├── rank0/d0/          # rank 0 的第 0 次 dispatch
│   └── dispatch_program.json   # 本次 dispatch 运行的 next_levels/<program>
├── rank0/d1/          # rank 0 的第 1 次 dispatch
└── rank1/d0/
```

`d{k}` 是该卡在本次 run 内的第 k 次 dispatch，每次 run 从 `d0` 重新计数。
每次 dispatch 都归档在实际运行它的芯片下：带 `device=` 的按自身 rank，
comm-less 的（没有 `device=`）则按被分配到的芯片——这类 dispatch 按提交
顺序在程序的各芯片间轮询分配。每个叶子目录内是上表所述的扁平产物，
因此在单个 dispatch 目录内 L2 契约完全适用。

该路径只记录 dispatch 跑在**哪里**，不记录它跑的是**什么**，因此
`_submit_chip` 还会写一份 `dispatch_program.json`，指明背后的
`next_levels/<program>`。kernel 名称必须经由它解析：`func_id` 是
**每个 L2 program 独立的命名空间**——每个 program 的 kernel 都从 0 开始
编号，所以跨 program 合并的 name map 会把一个 program 的 task 标成另一个
program 的名字，既静默又看似合理。若某次 dispatch 的 program 无法解析，
转换时使用匿名标签，而不是猜一个名字。

## 泳道会把 workload 跑两遍（onboard）

泳道转换器需要把每个 task 的耗时和一张**只有 `deps.json` 才携带的任务图**做
join——device 热路径不再记录 per-task fanout，因此没有 dep_gen 抓取时，泳道会退化
成没有依赖箭头的匿名 `task(rXtY)`。但 dep_gen 采集开销很高，会污染泳道本身要测量的
耗时。所以这两份抓取来自两次独立运行（这正是 Simpler 文档描述的“抓一次图、计多次
时”工作流）。

于是在 **onboard L2** 平台上开启 `enable_chip_swimlane` 会透明地把 kernel 跑两遍：

1. **抓图趟** —— 仅开 dep_gen，产出 `deps.json`，在**独立子进程**里运行
   （`python -m pypto.runtime._dep_gen_capture`）。这是必须的、不只是为了整洁：
   runtime 的每趟 finalize 不能可靠回收 DFX collector 申请的 SVM host-register
   映射，所以同一进程内第二趟 DFX 会撞上注册上限（`halHostRegister` 返回 8）；子进程
   退出时操作系统会彻底回收这些状态。抓图是 best-effort——子进程失败时只打印告警、
   计时趟照常运行（泳道退化成匿名 `task(rXtY)`）。
2. **计时趟** —— 开泳道（以及 PMU / args-dump / scope-stats 等其它对时序敏感的
   DFX），强制关闭 dep_gen，产出耗时干净的 `chip_swimlane_records.json`，这一趟的耗时
   才会被上报，在本进程内运行。

两趟写入同一个 `dfx_outputs/`，因此 `swimlane_converter` 会自动把同目录的
`deps.json` 与记录 join 起来。额外加 `--enable-dep-gen` 不会改变这两趟（抓图趟已经
产出了 `deps.json`），只是让本次运行额外打印 `deps_viewer` 的渲染提示。仿真平台
（`*sim`）保持单趟——那里本来就会跳过泳道转换。

分布式 L3 使用相同的抓图/计时拆分，但不经过 L2 抓图子进程。one-shot
路径为每趟创建新的 Worker 生命周期；prepared `DistributedWorker` 保留
resident handle 和已 fork 的层级，但进入两个独立的 `Worker.run()` fence：
先仅开 dep-gen，再开泳道并强制关闭 dep-gen。两趟都会重置每张卡的 dispatch
计数，因此图和计时产物会合并到相同的 `rank{r}/d{k}` 目录。两趟都会实际
执行程序，且不会在中间恢复可写参数，这与现有 one-shot L3 replay 语义一致。

L2 子进程用两种方式重建编排实参：被 pytest harness 驱动时从 `golden.py` 重新生成
（确定性输入 → 图忠实），被编译产物 API（`compiled(...)`）驱动时从记录下来的
规格重建。任务图可能由张量**值**（而不只是 scalar）路由，例如 paged-attention 的
`block_tables` / `seq_lens`，所以规格会尽量保留真实数据：host `torch.Tensor`
原样存盘再加载、scalar 原样保留，只有驻留在设备上、子进程无法访问的 `DeviceTensor`
退化成按记录 shape 填零的张量。因此除非是某个**设备驻留**张量在路由图，否则捕获是
精确的；那种情况下捕获是近似的。

## 使用方式

### 从 Python（`RunConfig`）

```python
from pypto import ir
from pypto.runtime import RunConfig

config = RunConfig(
    platform="a2a3sim",
    enable_chip_swimlane=4,      # 全量 swimlane -> chip_swimlane_records.json
                                 # （True 等价于等级 4；需要更轻量时用 1-3）
    enable_dep_gen=True,         # 生成 deps.json（按需用 deps_viewer 渲染 HTML）
    enable_pmu=4,                # PMU 事件 = MEMORY
)
compiled = ir.compile(MyProgram, **config.compile_kwargs())
compiled(a, b, c, config=config)
```

### 从 pytest

```bash
# 裸 flag = 等级 4（全量）
pytest tests/st/runtime/framework_and_models/test_perf_swimlane.py \
    --platform a2a3sim --enable-chip-swimlane

# 仅 AICore 计时——最轻量的采集
pytest tests/st/runtime/ \
    --platform a2a3sim --chip-swimlane-level 1 --enable-dep-gen
```

## 选择性张量 Dump

`enable_dump_args` 是一个**级别**（`0`=off、`1`=partial、`2`=full；
`True`→`1`、`False`→`0`）。级别 `2` 会把每个 task 的每个绑定都写入
`args_dump/`。在大规模工作负载下，host 端 dump 收集器（约 42 MB/s 排空
速率）会被打满，进而 AICPU 会被 STARS 算子执行超时机制杀掉 —— 1 GB 量级的
KV-cache 等大绑定填充队列的速度远快于排空速度。可以用 **partial**（级别 `1`）
并标记只关注的张量把 dump 范围收窄。提供两种入口，底层都由 runtime 的
`Arg::dump(...)` API（simpler#844）支撑。选择性与全量由 dump 级别在 host 侧
锁定，因此不再发射 orch body 的开关（simpler#953）。二者与两种 `deps=` 入口
一一对应 —— 一个声明式标记（`pl.dump_tag`，对应自动推断的 deps），一个
显式 kwarg（`dumps=`，对应 `deps=`）：

**声明式（`pl.dump_tag(t)`）** —— 一条语句，标记 `t` 后每个**后续**消费
该值的 kernel 派发都会 dump 它，无论该派发降级为普通 `ir.Call`（典型的
`@pl.jit` / 张量算子路径）还是 `ir.Submit`：

```python
@pl.function(type=pl.FunctionType.Orchestration)
def orch(self, q: pl.Tensor[...], k_cache: pl.Tensor[...], out: pl.Out[...]):
    pl.dump_tag(q)
    pl.dump_tag(out)
    out = self.qk_pv(q, k_cache, out)   # q、out 被 dump；k_cache 被过滤掉
```

**显式 kwarg（`dumps=[...]`）** —— `pl.submit(...)` 和 `pl.at(...)` 接受
`dumps=[...]` kwarg（与 `deps=[...]` 对称），列出该次 task 启动要 dump 的张量。
每个条目必须是该 submit 的某个张量实参 / 该 scope 捕获的某个张量：

```python
with pl.manual_scope():
    out, tid = pl.submit(self.qk_pv, q, k_cache, out, deps=[prev], dumps=[q, out])
    # codegen → params_t0.dump(ext_q, ext_out);
```

**没有调用参数包装** —— 普通 `self.kernel(...)` 调用点不提供 `dumps=` 入口；
用 `pl.dump_tag` 标记它的输入，或用 `pl.submit(..., dumps=[...])` 提交它。
两种入口都写入消费 Call / `Submit` 的同一个 `dump_vars` attr，以 **Var 身份**
跟踪 —— 而非名字。它像 `Submit::deps_` 一样随 SSA、内联、codegen 流动，
因此没有模糊名字匹配、没有误报。这些标记仅在部分 dump（`enable_dump_args == 1`）
下生效；dump 关闭（`0`）时不起作用，全量 dump（`2`）下也无意义——后者会捕获每个
绑定。

`pl.dump_tag` 同样可以写在 Inline helper（`@pl.jit.inline` /
`FunctionType.Inline`）内，对两种 kernel 调用风格都生效：

- **显式 `self.kernel(...)` 派发** —— 标记在消费 Call 上记录为
  `dump_vars`；`InlineFunctions` pass 把该 call splice 进调用方，并把每个
  inline 形参替换为调用方实参，因此写在 inline 形参或 inline 体内的
  `pl.create_tensor(...)` 结果上的标记会在内联点生效。
- **`@pl.jit` / 张量算子风格（`with pl.at(level=...)`、`c = a + 1.0`）** ——
  此时 kernel 派发由 outline pass *合成*，而非在 parse 阶段写出。标记改为
  写入所在 scope 的 `dump_vars`（round-trip 成 `pl.at(..., dumps=[...])`）；
  写在内联调用点的标记先落在该 call 的 `dump_vars` 上，再由
  `InlineFunctions` 转移到它 splice 进来的 scope 上。outliner 随后按 Var
  身份把每个被 scope 捕获的 dump Var 翻译成合成派发的 `dump_vars` ——
  与 `no_dep_args=` 走的 scope-attr → Call-attr 路径相同。scope 实际未作为
  kernel 实参消费的标记会被静默丢弃。

两种情况都无需任何 tag 迁移；多层内联在 pass 的 fixpoint 内被正确处理。

### 限制

| 标记位置 / 目标 | 状态 |
| --------------- | ---- |
| `pl.dump_tag(t)` 写在 Orchestration 或 Inline 函数体内的独立语句 | 支持（声明式标记；影响每个后续消费的派发）。 |
| `dumps=[arg]` 写在 `pl.submit(...)` 上 | 支持 —— submit 侧的显式入口（与 `deps=` 对称）；每个条目必须是该 submit 的位置实参。 |
| `dumps=[t]` 写在 `pl.at(...)` 上 | 支持 —— scope 侧的显式入口（与 `deps=` 对称）；每个条目必须是该 scope 体捕获的张量。 |
| `dumps=` 写在普通 `self.kernel(...)` 调用上 | 不支持 —— 抛出 `ParserTypeError`。普通调用是 fire-and-forget；请用 `pl.dump_tag(t)` 声明目标，或用 `pl.submit(..., dumps=[...])` 提交。 |
| 标记被 outline 合成的派发消费（`@pl.jit` / `with pl.at(level=...)` / 张量算子风格） | 支持 —— 标记随 scope 级 `dump_vars` 载体（`dumps=`）传递，outliner 再把它映射到合成派发的实参上。 |
| `pl.dump_tag(t)` 写在 `@pl.function(type=pl.FunctionType.InCore/AIC/AIV/Group)` 函数体内 | 不支持 —— parse 阶段抛出 `ParserSyntaxError`。dump 过滤由编排层 codegen 在 kernel 调用点完成；kernel 函数体内没有对应的调用点实参可挂载标记。请将 `pl.dump_tag` 放在外层 `Orchestration`（或 `Inline`）函数里。 |
| `pl.submit(...)` 的合成输出（隐式 `Out`） | 不支持 —— 合成输出没有调用点实参可包装。 |
| HOST 层 Python `SubWorker` 张量 | 不支持 —— runtime 没有对应的 `Arg::dump` 接口。 |
| 对被标记的值重新赋值（如 `q = self.foo(q)`） | rebind 出来的是**新值**；前面的 `pl.dump_tag(q)` **不会**自动覆盖（以 Var 身份跟踪，而非名字）。若 kernel 消费的是新值，需要再标一次。 |
| 标记的值经过形状/类型变换后才被消费（`q2 = pl.reshape(q)`、`pl.cast`、逐元素算子等） | 变换会产生**新 Var**，所以 `pl.dump_tag(q)` **覆盖不到** `q2`。与重新赋值同源（以 Var 身份跟踪，而非名字）。请标记 kernel 实际接收的那个值 —— 例如 `pl.dump_tag(q2)`。 |
| 标记只通过动态、数据相关偏移读取的值（`q_flat[runtime_row : runtime_row + N, …]`） | 不支持 —— 该索引读会 lower 成 gather / 动态地址 load，而非静态的整张量 `Arg`。编排层 codegen 从该实参槽取不出整个 Var（`AsVarLike` 无可按身份匹配的对象），标记无从挂载。请将该值先经一个用**静态、编译期分块**偏移读取的 buffer 中转，再标记该 buffer。 |
| 标记由 `y = pl.assemble(y, tile, offset)` 填充的编排层 buffer | 不支持 —— 编排层的 `pl.assemble` 只 lower 成纯名字别名（`emit_name_map_[lhs] = target`，`HandleTensorAssembleAssign`），**不产生任何 kernel 派发**。该 buffer 从不作为整张量 `Arg` 进入 task，没有可供 `Arg::dump` 标记的对象（且 `assemble` 每次迭代都会 rebind 该 Var）。请改用静态原地切片写 `y[offset_slice] = tile` 并标记 `y`，或改为 dump 各生产者 kernel 的输出 Arg。 |
| 标记只被编排层标量读消费的张量（`pl.read(block_table_flat, […])`） | 不支持 —— 该张量在 orch/AICPU/HOST 层被逐元素读取（如计算 page 偏移），从不作为 Tensor `Arg` 进入设备 kernel。MVP runtime 的选择性 dump 路径只覆盖 per-task 的**设备** Arg。请将其中转进一个被设备 kernel 作为整 Arg 消费的张量。 |

## 将 `deps.json` 渲染为 HTML

`enable_dep_gen` 只产出原始的 `deps.json`；对应的 pan/zoom HTML 依赖图由
一个独立的离线工具生成。该工具**不会自动**被调用——多 thousand-node
图的 Graphviz 布局可能跑几分钟乃至更久，在 runner 的 hot path 上同步
等待曾导致外层调度器（如 taskqueue daemon）把整个任务 SIGKILL。
所以按需手动渲染：

```bash
# 文本摘要（默认）—— 便于 grep，无需 Graphviz。
python -m simpler_setup.tools.deps_viewer <work_dir>/dfx_outputs/deps.json

# HTML 图 —— Graphviz `dot` 引擎，层次化布局（<500 节点）。
python -m simpler_setup.tools.deps_viewer <work_dir>/dfx_outputs/deps.json \
    --format html

# 大图 —— 切换到可扩展的力导向引擎。
python -m simpler_setup.tools.deps_viewer <work_dir>/dfx_outputs/deps.json \
    --format html --engine sfdp
```

输出会写到输入旁边的 `deps_viewer.txt`（默认文本）或 `deps_viewer.html`
（`--format html`），用 `-o <path>` 改路径。`--engine` 仅对 HTML 生效，
支持的取值（沿用 Graphviz 命名）：`dot | sfdp | fdp | neato | circo |
twopi`。`dot` 是默认值，在 ~500 节点以内 DAG 风格最清晰；更大的图建议
用 `sfdp`（O(N log N) 布局，能扩展到 1 万节点以上）。每次 dep_gen-enabled
跑完时 runner 也会在日志末尾打印同样的提示。

需要 `PATH` 上有 Graphviz（`apt install graphviz` /
`brew install graphviz`）。生成的 HTML 用浏览器直接打开即可——
拖拽平移、滚轮缩放、`f` 自适应窗口、`r` 重置。

### 可读的 kernel 名称（`name_map_*.json`）

默认情况下，swimlane / 依赖图工具用数字 id 标注任务（`task(rXtY)` /
`func_<id>(...)`）。要恢复真实 kernel 名称（`matmul(rXtY)`），records
旁边必须有一份 name map。Simpler 自带的 SceneTest harness 会写这个文件；
pypto 不使用 SceneTest，因此当开启 `enable_chip_swimlane` 或
`enable_dep_gen` 时，runner 会从 `kernel_config.py` 已有的 `func_id` /
`name` 字段合成 `<work_dir>/dfx_outputs/name_map_<case>.json`。它会被自动
消费：`swimlane_converter` 通过 `--func-names <name_map>` 调用，
`deps_viewer` 则自动发现同目录下的 `name_map_*.json`。无需手动操作。

## 将 `scope_stats.jsonl` 渲染为 HTML

`enable_scope_stats` 产出原始的 `scope_stats/scope_stats.jsonl`（第 1
行为 run 元数据，其后每行是一条 per-scope 记录）。用离线渲染器把它转成
一份自包含的 HTML 报告——每个 ring 一条时间线，含 heap / task_window /
tensormap 峰值：

```bash
python runtime/tools/scope_stats_plot.py \
    <work_dir>/dfx_outputs/scope_stats/scope_stats.jsonl
```

报告写在输入文件旁边，命名为 `scope_stats.html`。与 `deps_viewer`
一样，它**不会**自动触发——每次 scope-stats-enabled 跑完时 runner 会在
日志末尾打印这条提示。

## 实现位置

| 关注点 | 文件 | 函数 / 成员 |
| ------ | ---- | ----------- |
| `RunConfig` 字段定义 | [runner.py](../../../python/pypto/runtime/runner.py) | `RunConfig` dataclass + `any_dfx_enabled()` |
| `CallConfig` 透传 | [device_runner.py](../../../python/pypto/runtime/device_runner.py) | `execute_on_device(..., enable_*, output_prefix)` |
| 流水线打包 | [runner.py](../../../python/pypto/runtime/runner.py) | `_DfxOpts` dataclass + `_DfxOpts.from_run_config` |
| 按 flag 后处理分发 | [runner.py](../../../python/pypto/runtime/runner.py) | `_collect_dfx_artifacts` |
| kernel 名称映射合成 | [runner.py](../../../python/pypto/runtime/runner.py) | `_write_name_map` |
| L3 每次 dispatch 的 program 标记 | [distributed_runner.py](../../../python/pypto/runtime/distributed_runner.py) | `_record_dispatch_program` / `_read_dispatch_program` |
| L3 每次 dispatch 的泳道转换 | [distributed_runner.py](../../../python/pypto/runtime/distributed_runner.py) | `_collect_l3_swimlane` / `_write_dispatch_name_map` |
| pytest 入口 | [tests/st/conftest.py](../../../tests/st/conftest.py) | `pytest_addoption` |
| Harness 流水线上下文 | [tests/st/harness/core/test_runner.py](../../../tests/st/harness/core/test_runner.py) | `start_pipeline(..., enable_*)` |

## 已弃用别名

`RunConfig.enable_l2_swimlane` 与 pytest flag `--enable-l2-swimlane` 是
`enable_chip_swimlane` / `--enable-chip-swimlane` 的旧拼写。Simpler 的
Worker/Chip/Core 命名迁移把 L2 这一层改名为 "chip"（`L2Swimlane*` ->
`ChipSwimlane*`，`l2_swimlane_records.json` ->
`chip_swimlane_records.json`），PyPTO 现在遵循该契约。

两个旧拼写仍可用，并会发出 `DeprecationWarning`，将在未来版本中移除。
取值与语义完全不变，迁移就是改个名字：

```python
RunConfig(enable_l2_swimlane=True)    # 已弃用
RunConfig(enable_chip_swimlane=4)     # 等价采集
```

几个值得知道的细节：

- `enable_l2_swimlane` **不是** dataclass 字段，而是构造参数 + property。
  这样 `dataclasses.replace(cfg, enable_chip_swimlane=N)` 才不会有歧义；
  若把别名做成字段，`replace()` 会从旧实例把它一并传回，可能静默覆盖你刚
  传入的值。
- 读取 `cfg.enable_l2_swimlane` 不告警（返回规范字段的等级）。使用旧构造
  参数、或对该属性赋值，才会告警。
- 同时传入两种拼写会抛 `ValueError`。

## 重放已有的 build_output

重跑、修改并重新测量已有的 `build_output/<jit_dir>/`（含 `debug/run.py`、
`.pto` 拼接、对目录重放做 `benchmark()`、以及 L3 构建）单独成页：
[重放已有的 `build_output`](03-runtime-replay.md)。上面记录的每个 DFX 开关
在那条路径上同样生效。

## 相关文档

- Simpler runtime 侧参考：`runtime/docs/dfx/{chip-swimlane-profiling,
  args-dump,pmu-profiling,dep-gen,scope-stats}.md`。
- 编译期 profiling（正交、单 PyPTO 进程）：
  [01-compile-profiling.md](01-compile-profiling.md)。
