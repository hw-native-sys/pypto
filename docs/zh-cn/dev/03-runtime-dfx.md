# 运行时 DFX（Design For X）开关

PyPTO 将 Simpler 的四项运行时诊断子功能以独立开关的形式暴露在
[`RunConfig`](../../../python/pypto/runtime/runner.py) 上。每个开关都
1:1 映射到 Simpler 的 `CallConfig` 字段，以及 `runtime/conftest.py` 中
对应的 flag，保持两侧命名一致。

## 开关映射表

| `RunConfig` 字段 | pytest flag | `CallConfig` 成员 | `dfx_outputs/` 下产物 | 后处理工具 |
| ---------------- | ----------- | ----------------- | --------------------- | ---------- |
| `enable_l2_swimlane: bool` | `--enable-l2-swimlane` | `enable_l2_swimlane` | `l2_perf_records.json` | `swimlane_converter` → `merged_swimlane_*.json` |
| `enable_dump_tensor: bool` | `--dump-tensor` | `enable_dump_tensor` | `tensor_dump/{tensor_dump.json,bin}` | `dump_viewer`（手动） |
| `enable_pmu: int` | `--enable-pmu [N]`（裸 flag = `2`） | `enable_pmu`（`0` 关，`>0` 事件类型） | `pmu.csv` | — |
| `enable_dep_gen: bool` | `--enable-dep-gen` | `enable_dep_gen` | `deps.json` | `deps_to_graph` → `deps_graph.html` |

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
        enable_l2_swimlane=True,     # 生成 l2_perf_records.json
        enable_dep_gen=True,         # 生成 deps.json + deps_graph.html
        enable_pmu=4,                # PMU 事件 = MEMORY
    ),
)
```

### 从 pytest

```bash
pytest tests/st/runtime/test_perf_swimlane.py \
    --platform a2a3sim --enable-l2-swimlane

pytest tests/st/runtime/ \
    --platform a2a3sim --enable-l2-swimlane --enable-dep-gen
```

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

## 相关文档

- Simpler runtime 侧参考：`runtime/docs/dfx/{l2-swimlane,
  tensor-dump,pmu-profiling,dep_gen}.md`。
- 编译期 profiling（正交、单 PyPTO 进程）：
  [01-compile-profiling.md](01-compile-profiling.md)。
