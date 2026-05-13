# 编译 Profiling（Compile Profiling）

PyPTO 内置编译管线 profiling，可记录编译管线各阶段的墙钟耗时——从前端解析、编译
Pass、代码生成到上板执行。

## 快速开始

### 方式 1：环境变量

```bash
PYPTO_COMPILE_PROFILING=1 python3 my_program.py
```

### 方式 2：`ir.compile()` 参数

```python
output_dir = ir.compile(program, profiling=True)
# 结果写入 output_dir/report/pipeline_profile.{txt,json}
```

### 方式 3：Context Manager

```python
from pypto.compile_profiling import CompileProfiler

with CompileProfiler() as prof:
    @pl.program
    class MyProgram:
        ...
    ir.compile(MyProgram, ...)

print(prof.summary())
prof.to_json("profile.json")
```

### 方式 4：`RunConfig`

```python
from pypto.runtime import run, RunConfig

result = run(
    program=MyProgram,
    tensor_specs=specs,
    golden=golden_fn,
    config=RunConfig(compile_profiling=True),
)
# result.profile 包含 profiling 数据（dict 格式）
```

## 输出格式

### 人类可读摘要（`pipeline_profile.txt`）

```text
PyPTO Compile Profile
======================
Total: 2.847s

  parse                            0.023s  ( 0.8%)
  passes                           1.204s  (42.3%)
    UnrollLoops                     0.012s  ( 0.4%)
    ConvertToSSA                    0.034s  ( 1.2%)
    ...
    AllocateMemoryAddr              0.156s  ( 5.5%)
  codegen                           0.418s  (14.7%)
    kernel_codegen:my_kernel        0.312s  (11.0%)
    orchestration_codegen           0.106s  ( 3.7%)
  device_execution                  1.202s  (42.2%)
```

### 结构化 JSON（`pipeline_profile.json`）

```json
{
  "total_seconds": 2.847,
  "stages": [
    {"name": "parse", "seconds": 0.023, "children": []},
    {"name": "passes", "seconds": 1.204, "children": [
      {"name": "UnrollLoops", "seconds": 0.012, "children": []},
      {"name": "ConvertToSSA", "seconds": 0.034, "children": []}
    ]},
    {"name": "codegen", "seconds": 0.418, "children": [
      {"name": "kernel_codegen:my_kernel", "seconds": 0.312, "children": []},
      {"name": "orchestration_codegen", "seconds": 0.106, "children": []}
    ]}
  ]
}
```

## 阶段层次结构

使用 `runtime.run()` 时，profiler 记录以下阶段：

| 阶段 | 说明 |
| ---- | ---- |
| `compile` | 完整编译（包裹 `ir.compile()`） |
| `parse` | `@pl.program` 装饰器 AST 解析 |
| `passes` | Pass 管线执行 |
| 各 Pass 阶段 | 单个 Pass 耗时（如 `UnrollLoops`、`AllocateMemoryAddr`） |
| `codegen` | 代码生成 |
| `kernel_codegen:<name>` | 每个内核的 PTO/ptoas 代码生成 |
| `orchestration_codegen` | 编排 C++ 代码生成 |
| `golden_write` | Golden 参考文件生成 |
| `device_execution` | 上板编译与执行（Simpler） |

直接使用 `ir.compile()` 时，仅记录 `passes` 和 `codegen`（含子阶段）。

## 编程接口

```python
from pypto.compile_profiling import CompileProfiler, get_active_profiler

# 检查 profiling 是否激活（显式或通过环境变量）
prof = get_active_profiler()

# 作为 context manager 使用
with CompileProfiler() as prof:
    # 记录自定义阶段
    with prof.stage("my_custom_stage"):
        do_something()

    # 访问结果
    data = prof.to_dict()      # dict
    text = prof.summary()      # 人类可读字符串
    json_str = prof.to_json()  # JSON 字符串
    prof.to_json("out.json")   # 写入文件
    prof.write_report("dir/")  # 同时写入 .txt 和 .json
```

## 开销

当 profiling **未启用**（默认），每个阶段边界仅执行一次
`CompileProfiler.current()` 空值检查，开销可忽略不计。

当 profiling **已启用**时，每个阶段记录两次 `time.perf_counter()`
调用（现代硬件上为亚微秒级）。

## 相关功能

- **运行时 DFX**（`RunConfig.enable_l2_swimlane` / `enable_dump_tensor` /
  `enable_pmu` / `enable_dep_gen`）驱动 Simpler 的每任务诊断产物 —— swimlane
  记录、tensor I/O dump、AICore PMU CSV、PTO2 dep_gen 边。四个 flag 互相
  独立，共用 `<work_dir>/dfx_outputs/` 作为输出根目录，与编译 profiling
  正交。详见 [03-runtime-dfx.md](03-runtime-dfx.md)。
