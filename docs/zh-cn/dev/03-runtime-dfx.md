# 运行时 DFX（Design For X）开关

PyPTO 将 Simpler 的五项运行时诊断子功能以独立开关的形式暴露在
[`RunConfig`](../../../python/pypto/runtime/runner.py) 上。每个开关都
1:1 映射到 Simpler 的 `CallConfig` 字段，以及 `tests/st/conftest.py` 中
对应的 pytest flag，保持两侧命名一致。

## 开关映射表

| `RunConfig` 字段 | pytest flag | `CallConfig` 成员 | `dfx_outputs/` 下产物 | 后处理工具 |
| ---------------- | ----------- | ----------------- | --------------------- | ---------- |
| `enable_l2_swimlane: bool` | `--enable-l2-swimlane` | `enable_l2_swimlane` | `l2_swimlane_records.json` | `swimlane_converter` → `merged_swimlane_*.json` |
| `enable_dump_tensor: int` | `--dump-tensor [LEVEL]`（裸 flag = `1`） | `enable_dump_tensor`（`0` 关，`1` 部分，`2` 全量） | `tensor_dump/{tensor_dump.json,bin}` | `dump_viewer`（手动） |
| `enable_pmu: int` | `--enable-pmu [N]`（裸 flag = `2`） | `enable_pmu`（`0` 关，`>0` 事件类型） | `pmu.csv` | — |
| `enable_dep_gen: bool` | `--enable-dep-gen` | `enable_dep_gen` | `deps.json` | `deps_to_graph`（手动） |
| `enable_scope_stats: bool` | `--enable-scope-stats` | `enable_scope_stats` | `scope_stats/scope_stats.jsonl` | `scope_stats_plot`（手动） |

五个开关**完全正交**，可任意组合。任一开启时自动将
`RunConfig.save_kernels` 强制设为 `True`，确保 `<work_dir>/dfx_outputs/`
目录在 run 结束后保留。

## 产物契约

runtime 把所有产物写到 `CallConfig.output_prefix` 指向的同一目录。
PyPTO 将该 prefix 设为 `<work_dir>/dfx_outputs/`，其下的子路径按上表
固定。多数产物是 prefix 下的扁平文件；`scope_stats` 例外——其采集器
写入 `scope_stats/` 子目录，内含 `scope_stats.jsonl`。Simpler 的
`CallConfig::validate()` 在任一 flag 开启但
`output_prefix` 为空时拒绝调用；PyPTO 在 Python 侧镜像该契约，
`execute_on_device` 会**先于** C++ 边界抛 `ValueError`，让 traceback
直接指向调用方代码。

## L2 泳道会把 kernel 跑两遍（onboard）

泳道转换器需要把每个 task 的耗时和一张**只有 `deps.json` 才携带的任务图**做
join——device 热路径不再记录 per-task fanout，因此没有 dep_gen 抓取时，泳道会退化
成没有依赖箭头的匿名 `task(rXtY)`。但 dep_gen 采集开销很高，会污染泳道本身要测量的
耗时。所以这两份抓取来自两次独立运行（这正是 Simpler 文档描述的“抓一次图、计多次
时”工作流）。

于是在 **onboard** 平台上开启 `enable_l2_swimlane` 会透明地把 kernel 跑两遍：

1. **抓图趟** —— 仅开 dep_gen，产出 `deps.json`，在**独立子进程**里运行
   （`python -m pypto.runtime._dep_gen_capture`）。这是必须的、不只是为了整洁：
   runtime 的每趟 finalize 不能可靠回收 DFX collector 申请的 SVM host-register
   映射，所以同一进程内第二趟 DFX 会撞上注册上限（`halHostRegister` 返回 8）；子进程
   退出时操作系统会彻底回收这些状态。抓图是 best-effort——子进程失败时只打印告警、
   计时趟照常运行（泳道退化成匿名 `task(rXtY)`）。
2. **计时趟** —— 开泳道（以及 PMU / tensor-dump / scope-stats 等其它对时序敏感的
   DFX），强制关闭 dep_gen，产出耗时干净的 `l2_swimlane_records.json`，这一趟的耗时
   才会被上报，在本进程内运行。

两趟写入同一个 `dfx_outputs/`，因此 `swimlane_converter` 会自动把同目录的
`deps.json` 与记录 join 起来。额外加 `--enable-dep-gen` 不会改变这两趟（抓图趟已经
产出了 `deps.json`），只是让本次运行额外打印 `deps_to_graph` 的渲染提示。仿真平台
（`*sim`）保持单趟——那里本来就会跳过泳道转换。

子进程用两种方式重建编排实参：被 pytest harness 驱动时从 `golden.py` 重新生成
（确定性输入 → 图忠实），被编译产物 API（`execute_compiled`）驱动时从记录下来的
规格重建。任务图可能由张量**值**（而不只是 scalar）路由，例如 paged-attention 的
`block_tables` / `seq_lens`，所以规格会尽量保留真实数据：host `torch.Tensor`
原样存盘再加载、scalar 原样保留，只有驻留在设备上、子进程无法访问的 `DeviceTensor`
退化成按记录 shape 填零的张量。因此除非是某个**设备驻留**张量在路由图，否则捕获是
精确的；那种情况下捕获是近似的。

## 使用方式

### 从 Python（`RunConfig`）

```python
from pypto.runtime import run, RunConfig

run(
    MyProgram, a, b, c,
    config=RunConfig(
        platform="a2a3sim",
        enable_l2_swimlane=True,     # 生成 l2_swimlane_records.json
        enable_dep_gen=True,         # 生成 deps.json（按需用 deps_to_graph 渲染 HTML）
        enable_pmu=4,                # PMU 事件 = MEMORY
    ),
)
```

### 从 pytest

```bash
pytest tests/st/runtime/framework_and_models/test_perf_swimlane.py \
    --platform a2a3sim --enable-l2-swimlane

pytest tests/st/runtime/ \
    --platform a2a3sim --enable-l2-swimlane --enable-dep-gen
```

## 选择性张量 Dump

`enable_dump_tensor` 是一个**级别**（`0`=off、`1`=partial、`2`=full；
`True`→`1`、`False`→`0`）。级别 `2` 会把每个 task 的每个绑定都写入
`tensor_dump/`。在大规模工作负载下，host 端 dump 收集器（约 42 MB/s 排空
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
因此没有模糊名字匹配、没有误报。这些标记仅在部分 dump（`enable_dump_tensor == 1`）
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
# 默认 —— Graphviz `dot` 引擎，层次化布局（<500 节点）。
python -m simpler_setup.tools.deps_to_graph <work_dir>/dfx_outputs/deps.json

# 大图 —— 切换到可扩展的力导向引擎。
python -m simpler_setup.tools.deps_to_graph <work_dir>/dfx_outputs/deps.json \
    --engine sfdp
```

输出会写到输入旁边的 `deps_graph.html`（用 `-o <path>` 改路径）。
`--engine` 支持的取值（沿用 Graphviz 命名）：
`dot | sfdp | fdp | neato | circo | twopi`。`dot` 是默认值，在 ~500
节点以内 DAG 风格最清晰；更大的图建议用 `sfdp`（O(N log N) 布局，
能扩展到 1 万节点以上）。每次 dep_gen-enabled 跑完时 runner 也会在
日志末尾打印同样的提示。

需要 `PATH` 上有 Graphviz（`apt install graphviz` /
`brew install graphviz`）。生成的 HTML 用浏览器直接打开即可——
拖拽平移、滚轮缩放、`f` 自适应窗口、`r` 重置。

### 可读的 kernel 名称（`name_map_*.json`）

默认情况下，swimlane / 依赖图工具用数字 id 标注任务（`task(rXtY)` /
`func_<id>(...)`）。要恢复真实 kernel 名称（`matmul(rXtY)`），records
旁边必须有一份 name map。Simpler 自带的 SceneTest harness 会写这个文件；
pypto 不使用 SceneTest，因此当开启 `enable_l2_swimlane` 或
`enable_dep_gen` 时，runner 会从 `kernel_config.py` 已有的 `func_id` /
`name` 字段合成 `<work_dir>/dfx_outputs/name_map_<case>.json`。它会被自动
消费：`swimlane_converter` 通过 `--func-names <name_map>` 调用，
`deps_to_graph` 则自动发现同目录下的 `name_map_*.json`。无需手动操作。

## 将 `scope_stats.jsonl` 渲染为 HTML

`enable_scope_stats` 产出原始的 `scope_stats/scope_stats.jsonl`（第 1
行为 run 元数据，其后每行是一条 per-scope 记录）。用离线渲染器把它转成
一份自包含的 HTML 报告——每个 ring 一条时间线，含 heap / task_window /
tensormap 峰值：

```bash
python runtime/tools/scope_stats_plot.py \
    <work_dir>/dfx_outputs/scope_stats/scope_stats.jsonl
```

报告写在输入文件旁边，命名为 `scope_stats.html`。与 `deps_to_graph`
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
| pytest 入口 | [tests/st/conftest.py](../../../tests/st/conftest.py) | `pytest_addoption` |
| Harness 流水线上下文 | [tests/st/harness/core/test_runner.py](../../../tests/st/harness/core/test_runner.py) | `start_pipeline(..., enable_*)` |

## 已弃用别名

`RunConfig.runtime_profiling` 与 pytest flag `--runtime-profiling` 是
四项 DFX 独立化之前唯一启用 L2 swimlane 采集的入口，现作为
`enable_l2_swimlane` / `--enable-l2-swimlane` 的别名暂时保留，以兼容
仍在使用它们的外部脚本。两条路径都会发出 `DeprecationWarning`，并将
在未来版本中移除，请尽快迁移到新名称。

## 重放已有的 build_output

需要在改完 kernel cpp 之后重新跑一遍编译产物（典型场景：手调 kernel
后用 PMU / swimlane / tensor-dump 验证修改是否正确），使用 debug 专用
的 [`pypto.runtime.debug.replay`](../../../python/pypto/runtime/debug/replay.py)
模块。它复用与 `pypto.runtime.run` 相同的 `execute_compiled` 路径,
因此 DFX 开关的行为完全一致。

```python
from pypto.runtime.debug import replay
from pypto.runtime import RunConfig

replay(
    "build_output/_jit_xxx/",
    a, b, c,
    config=RunConfig(
        platform="a2a3sim",
        enable_pmu=2,
        enable_l2_swimlane=True,
    ),
)
```

CLI 形式（从目录里的 `golden.py` 加载输入）:

```bash
python -m pypto.runtime.debug.replay build_output/_jit_xxx/ \
    --pmu 2 --swimlane --log-level debug
```

默认 `recompile=True` 会清掉缓存的 `.so` / `.bin`,确保手改的 cpp
能被重新编译。如果没改 cpp、想跳过重编译,传 `recompile=False`
（或 CLI 的 `--no-recompile`）即可。`--log-level` 接受和
`PYPTO_RUNTIME_LOG` 相同的值（`debug`、`v0..v9`、`info`、`warn`、
`error`、`null`）;加上 `--log-sync-pypto` 可以把同一档位推到
PyPTO 的 C++ logger。

传 `validate=True`（或 `--validate`）会在执行结束后,用
`golden.py::compute_golden` 计算参考输出,并按 `golden.py` 里声明的
`RTOL` / `ATOL` 公差逐 output 比对;不一致会抛 `AssertionError`。
该开关需要目录里存在 `golden.py`（`ir.compile` 默认会产出）。

### 改 `.pto` 而不是 cpp

`replay`（以及自动生成的 `debug/run.py`）在清理 cpp 二进制之前会先
按 mtime 扫描 `ptoas/*.pto`：任何比同名 `ptoas/<unit>.cpp` 新的
`.pto` 都会触发一次 `ptoas` 重跑，新生成的 body 会 splice 到所有命中
的 `kernels/<core>/<func>.cpp` —— 也就是在两条 sentinel
`// --- ptoas-generated code ---` 与 `// --- Kernel entry point ---`
之间替换。随后照常走 cpp → `.so` 重编译。

| 改了哪些文件 | 实际触发的路径 |
| ------------ | -------------- |
| 只改 `kernels/<core>/<func>.cpp` | `cpp → .so`（保持原有行为） |
| 只改 `ptoas/<unit>.pto` | `pto → cpp → .so`（新增 —— splice + 重编译） |
| 两者都改 | `.pto` 决定 body 段；用户在 cpp wrapper / header 上的改动保留 |

需要 `ptoas` 可被发现（`PTOAS_ROOT` 或 `PATH`）；找不到时静默跳过。
关闭方式：`--no-rebuild-from-pto` 或 `PYPTO_REBUILD_FROM_PTO=0`。
若 `.pto` 编辑会改变 kernel 函数签名，**不在本特性范围**：保存的
wrapper 模板对不上,必须重新 `ir.compile()`。

### 自动生成的 `debug/run.py`

`ir.compile()` 会在 `<output_dir>/debug/run.py` 写一个自包含的
重跑脚本，用户只需要记住一条命令：

```bash
python build_output/<jit_dir>/debug/run.py
```

脚本是对上面 `replay` 流程的封装：

- 如果同目录有 `golden.py`，输入来自
  `golden.generate_inputs()`，并用 `compute_golden` 做数值校验。
- 否则（JIT 路径），输入由脚本内嵌的 shape / dtype 元数据构造，
  用户可自由修改用于实验。脚本还预留了一个
  `_user_compare(<参数名>)` 钩子，会在 `replay` 返回后自动调用 ——
  在里面手写 `assert torch.allclose(...)` 即可对 kernel 输出做
  自定义比对。
- 上面 "改 `.pto` 而不是 cpp" 一节描述的 `.pto` 重建流程在生成的
  脚本里同样生效：改一份 `ptoas/*.pto` 再跑一次,splice 自动发生。
  加 `--no-rebuild-from-pto` 可跳过。

生成过程是 **best-effort** —— 没有干净 orchestration 入口的程序
会静默跳过这一步，编译流程本身不受影响。

设置环境变量 `PYPTO_EMIT_DEBUG_RUNNER=0`（也接受 `false` / `no`，
大小写不敏感）可全局关闭。适合大型测试套件或 benchmark 流水线
（编译量大、不需要 runner）。关闭后底层的
`pypto.runtime.debug.replay` 模块 / CLI 仍可直接对 output 目录使用。

### 重放 L3 / 分布式构建

分布式（L3）程序——即 `@pl.jit.host` orchestrator 编译出的
`DistributedCompiledProgram`——支持同样的「改 `.pto` 再重跑」循环，
但它的 build 目录形态不同：**没有顶层 `kernel_config.py`**（每个 rank
的配置在 `next_levels/{rank}/` 下），host 驱动是
`orchestration/host_orch.py`，并且 `ir.compile()` 会额外写一个
`distributed_meta.json`：

```text
build_output/<jit_dir>/
  distributed_meta.json          # 参数元数据 + platform + DistributedConfig
  orchestration/host_orch.py     # L3 host 驱动
  next_levels/{rank}/            # 每个 rank 一个完整的单芯片子构建
      kernels/{aic,aiv}/*.cpp
      ptoas/*.pto
      kernel_config.py
```

`replay` 会自动识别这种布局（无顶层 `kernel_config.py` 但存在
`orchestration/host_orch.py`），并改用 simpler `Worker(level=3)` 派发，
而不是 `execute_compiled`。同样的 CLI / `debug/run.py` 流程无需改动：

```bash
python -m pypto.runtime.debug.replay build_output/<jit_dir>/
# 或
python build_output/<jit_dir>/debug/run.py
```

`.pto` → cpp 拼接和 `.so` 失效都会递归进每个 `next_levels/{rank}/`，
所以改 `next_levels/rank0/ptoas/<unit>.pto`（或直接改 kernel cpp）会
被识别，行为与单芯片完全一致。

底层只凭 `distributed_meta.json` 就能把目录重建成一个可调用的程序——
**不重新编译 pypto、不重跑 pass**。两个入口直接暴露这个能力：

```python
from pypto.runtime import execute_distributed_compiled
# 一次性（对标单芯片的 execute_compiled）：
execute_distributed_compiled("build_output/<jit_dir>/", [a, b, c])

# 可复用对象（需要时覆盖持久化的 platform / 设备）：
from pypto.ir.distributed_compiled_program import DistributedCompiledProgram, DistributedConfig
prog = DistributedCompiledProgram.from_dir(
    "build_output/<jit_dir>/",
    platform="a2a3",
    distributed_config=DistributedConfig(device_ids=[0, 1]),
)
prog(a, b, c)
```

`from_dir` 读取持久化的 HOST orchestrator 参数元数据（与 `host_orch.py`
匹配的 post-SSA 名字、方向、形状、dtype），并通过遍历 `next_levels/`
重建 chip callables；`platform` 与 `distributed_config` 默认取编译时
记录的值，可覆盖以在不同目标 / 设备集上重放。

**限制**：DFX 标志（`--pmu`、`--swimlane`、`--dump-tensor` 等）**尚未
透传到 L3 派发路径**——目前仅对单芯片重放生效。L3 的「改完再跑」循环
本身（改 `.pto`/cpp 后的正确性复检）是完整支持的。

## 相关文档

- Simpler runtime 侧参考：`runtime/docs/dfx/{l2-swimlane,
  tensor-dump,pmu-profiling,dep_gen,scope-stats}.md`。
- 编译期 profiling（正交、单 PyPTO 进程）：
  [01-compile-profiling.md](01-compile-profiling.md)。
