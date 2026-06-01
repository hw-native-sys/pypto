# 运行时 DFX（Design For X）开关

PyPTO 将 Simpler 的四项运行时诊断子功能以独立开关的形式暴露在
[`RunConfig`](../../../python/pypto/runtime/runner.py) 上。每个开关都
1:1 映射到 Simpler 的 `CallConfig` 字段，以及 `runtime/conftest.py` 中
对应的 flag，保持两侧命名一致。

## 开关映射表

| `RunConfig` 字段 | pytest flag | `CallConfig` 成员 | `dfx_outputs/` 下产物 | 后处理工具 |
| ---------------- | ----------- | ----------------- | --------------------- | ---------- |
| `enable_l2_swimlane: bool` | `--enable-l2-swimlane` | `enable_l2_swimlane` | `l2_swimlane_records.json` | `swimlane_converter` → `merged_swimlane_*.json` |
| `enable_dump_tensor: bool` | `--dump-tensor` | `enable_dump_tensor` | `tensor_dump/{tensor_dump.json,bin}` | `dump_viewer`（手动） |
| `enable_pmu: int` | `--enable-pmu [N]`（裸 flag = `2`） | `enable_pmu`（`0` 关，`>0` 事件类型） | `pmu.csv` | — |
| `enable_dep_gen: bool` | `--enable-dep-gen` | `enable_dep_gen` | `deps.json` | `deps_to_graph`（手动） |

四个开关**完全正交**，可任意组合。任一开启时自动将
`RunConfig.save_kernels` 强制设为 `True`，确保 `<work_dir>/dfx_outputs/`
目录在 run 结束后保留。

## 产物契约

runtime 把所有产物写到 `CallConfig.output_prefix` 指向的同一目录。
PyPTO 将该 prefix 设为 `<work_dir>/dfx_outputs/`，其下的子路径按上表
固定。Simpler 的 `CallConfig::validate()` 在任一 flag 开启但
`output_prefix` 为空时拒绝调用；PyPTO 在 Python 侧镜像该契约，
`execute_on_device` 会**先于** C++ 边界抛 `ValueError`，让 traceback
直接指向调用方代码。

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

## 实现位置

| 关注点 | 文件 | 函数 / 成员 |
| ------ | ---- | ----------- |
| `RunConfig` 字段定义 | [runner.py](../../../python/pypto/runtime/runner.py) | `RunConfig` dataclass + `any_dfx_enabled()` |
| `CallConfig` 透传 | [device_runner.py](../../../python/pypto/runtime/device_runner.py) | `execute_on_device(..., enable_*, output_prefix)` |
| 流水线打包 | [runner.py](../../../python/pypto/runtime/runner.py) | `_DfxOpts` dataclass + `_DfxOpts.from_run_config` |
| 按 flag 后处理分发 | [runner.py](../../../python/pypto/runtime/runner.py) | `_collect_dfx_artifacts` |
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

## 相关文档

- Simpler runtime 侧参考：`runtime/docs/dfx/{l2-swimlane,
  tensor-dump,pmu-profiling,dep_gen}.md`。
- 编译期 profiling（正交、单 PyPTO 进程）：
  [01-compile-profiling.md](01-compile-profiling.md)。
