# 回放一次构建

重跑、手改、重新测量一个产物目录 —— 不必回到 DSL。

## 概念

一个 `build_output/<jit_dir>/` 是自包含的：生成的 `.pto`、kernel cpp、运行时配置、参数元数据全在里面。回放就是围绕它闭合的那个循环 —— 手改一个 kernel、重跑、测量 —— 它是回答「这个改动有没有用」最快的方式，不需要重新编译。

让它成立的关键是**缓存失效**。没有它，手改过的 cpp 会被静默地从旧 `.so` 里取出来，你以为改了，实际测的是旧 kernel。

## 快速上手

每次编译都会写一个 re-runner，所以只需要记一条命令：

```bash
python build_output/<jit_dir>/debug/run.py
```

有兄弟 `golden.py` 时，它用 `generate_inputs()` 加载输入并与 `compute_golden` 对拍。JIT 路径下没有 `golden.py`，输入从脚本里内嵌的 shape / dtype 元数据现造 —— 随便改。

## 机制

### 改了就重跑的循环

| 你改了 | 会跑什么 |
| ------ | -------- |
| `kernels/<core>/<func>.cpp` | `cpp → .so` |
| `ptoas/<unit>.pto` | `pto → cpp → .so` —— 重跑 ptoas，并把新 body 拼进去 |
| 两个都改了 | body 区域以 `.pto` 为准；你在 cpp 里对 wrapper / 头部的改动被保留 |

判别依据是 mtime，逐单元评估：某个 `.pto` 比它的兄弟 `ptoas/<unit>.cpp` 新就触发重跑。拼接会替换 `// --- ptoas-generated code ---` 与 `// --- Kernel entry point ---` 两个 sentinel 之间的全部内容。

需要 `ptoas` 二进制在 `PTOAS_ROOT` 或 `PATH` 上；否则静默跳过。用 `--no-rebuild-from-pto` 或 `PYPTO_REBUILD_FROM_PTO=0` 关闭。

> **在 `.pto` 里改动 kernel 的签名不在支持范围内。** 存下来的 wrapper 样板会对不上；请从 DSL 重新编译。

### replay 模块

`debug/run.py` 是它的包装；需要自己传参时可以直接用模块：

```python
from pypto.runtime.debug import replay
from pypto.runtime import RunConfig

replay(
    "build_output/_jit_xxx/",
    a, b, c,
    config=RunConfig(platform="a2a3", enable_pmu=2, enable_chip_swimlane=4),
)
```

```bash
python -m pypto.runtime.debug.replay build_output/_jit_xxx/ \
    --pmu 2 --swimlane --dep-gen --scope-stats --log-level debug
```

**[运行期 DFX](04-runtime-dfx.md) 里的每个开关在这里原样适用** —— 回放正是给一个你一直在手调的构建挂上 DFX 的正确方式。

| 选项 | 效果 |
| ---- | ---- |
| `recompile=True`（默认）/ `--no-recompile` | 强制失效缓存的 `.so` / `.bin`，让手改被真正编进去 |
| `validate=True` / `--validate` | 按 `golden.py` 声明的 `RTOL` / `ATOL` 与 `compute_golden` 逐个输出对比；不符抛 `AssertionError` |
| `--log-level` | 取值同 `PYPTO_RUNTIME_LOG`（`debug`、`info`、`timing`、`warn`、`error`、`null`） |
| `--log-sync-pypto` | 同时把等级推给 PyPTO 的 C++ logger |

`--no-recompile` 只关闭*强制*失效。运行时与 PTO-ISA 的兼容性检查照常执行，而且 PyPTO **fail closed**：任一身份无法确定时，它选择重建，而不是信任现有二进制。

### 测量一次回放

`benchmark()` 需要一个活的 `CompiledProgram` 来拿参数元数据，而目录本身不带 —— 所以 `ir.compile()` 还会写一个 `compiled_meta.json` sidecar，`from_dir()` 据此重建一个可调用的程序，不重跑任何 pass。

**`from_dir()` 只重载元数据，不重建源码。** 与 `replay` 不同，它既不跑 `.pto` 拼接也不做缓存失效，所以只用它的话，你的改动会被静默忽略。两件事都要显式做：

```python
from pypto.ir import CompiledProgram
from pypto.runtime import benchmark
from pypto.runtime.debug import invalidate_binary_cache, rebuild_kernel_cpp_from_pto

work_dir = "build_output/<jit_dir>/"
rebuild_kernel_cpp_from_pto(work_dir)   # only if you edited ptoas/*.pto
invalidate_binary_cache(work_dir)       # drop cached .o/.so so the edit is compiled

compiled = CompiledProgram.from_dir(work_dir, platform="a2a3")
compiled(a, b, c)                                     # correctness re-check
stats = benchmark(compiled, [a, b, c], rounds=100)    # and timing
```

`platform` / `backend_type` 默认取编译时记录的值，可以覆盖以换个地方回放（`a2a3sim` → `a2a3`）。结果上的 `program` 是 `None` —— IR 不会持久化 —— 但 `validate_ir()` 仍然可用，它从 `passes_dump/` 读。

### 分布式构建

L3 构建没有顶层 `kernel_config.py`（每 rank 的配置在 `next_levels/{rank}/` 下），由 `orchestration/host_orch.py` 驱动。`replay` 会自动识别这种布局，并通过重建的 `DistributedCompiledProgram` 派发；`.pto` 拼接与缓存失效会递归进每个 rank。上面那两条命令原样可用，DFX 产物落在 `dfx_outputs/rank{r}/d{k}/`。

```python
from pypto.ir import DistributedCompiledProgram, DistributedConfig

DistributedCompiledProgram.from_dir("build_output/<jit_dir>/")(a, b, c)
```

### 把产物当独立进程跑

`pypto.runtime.execute_artifact` 以子进程方式运行一个已编译目录，并打印机器可读的结果标记：

标记的形状取决于它跑在哪种模式下。单个目录只报告设备号，因为调用方本来就知道自己指的是哪个目录：

```text
PYPTO_EXEC_RESULT=PASS device=<N>    # ran, and validated
PYPTO_EXEC_RESULT=FAIL               # device or validation failure
PYPTO_EXEC_RESULT=INFRA              # reconstruction / setup failure
```

批量清单会跑多个目录，所以每一行都要指名它报告的是哪一个：

```text
PYPTO_EXEC_RESULT=PASS  work_dir=<wd> device=<N>
PYPTO_EXEC_RESULT=FAIL  work_dir=<wd>
PYPTO_EXEC_RESULT=INFRA work_dir=<wd>
```

两种模式下第三个值都是重点：它把「kernel 错了」和「环境坏了」分开，于是 CI 不会把一次设备短缺记成回归。

## 边界情况

| 现象 | 原因 | 处理 |
| ---- | ---- | ---- |
| **改了 cpp，什么都没变** | 从缓存里取了旧 `.so` | 用 `replay`（默认失效缓存），不要只用 `from_dir()` |
| **改了 `.pto`，什么都没变** | `ptoas` 不在 `PTOAS_ROOT` / `PATH` 上，拼接静默跳过 | 把 `PTOAS_ROOT` 指向可用的 ptoas |
| **没有 `debug/run.py`** | 生成是 best-effort，会跳过没有干净编排入口的程序；或 `PYPTO_EMIT_DEBUG_RUNNER=0` | 直接用 `python -m pypto.runtime.debug.replay <dir>` |
| **抛出指向 `compiled_meta.json` 的 `ValueError`** | sidecar 被手改或截断 | 重新编译；报错信息会指名文件和重新生成它的那次编译 |
| **明明是 L3 构建却按 L2 回放** | 同一目录里残留了更早一次编译的顶层 `kernel_config.py` | 编译到一个全新目录；每次编译都会丢掉不属于自己这一类的标记 |
| **`validate=True` 抛 `FileNotFoundError`** | 没有 `golden.py`（JIT 路径不写） | 改用 `debug/run.py` 里的 `_user_compare(...)` 钩子 |

## 参见

- [运行期 DFX](04-runtime-dfx.md) —— 该给回放挂上哪些开关。
- [In-core trace](06-incore-trace.md) —— 给刚手改过的 kernel 做剖析。
- [性能](../performance/index.md) —— 该改什么，以及怎么确认改对了。
- [Replaying an Existing `build_output`](../../dev/03-runtime-replay.md) —— sidecar 契约、构建类型标记，以及复用 `output_dir` 的规则。
