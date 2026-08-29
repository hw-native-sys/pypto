# 重放已有的 `build_output`

在不从 DSL 重新编译的前提下，重跑、修改并重新测量一个已编译的
`build_output/<jit_dir>/`。本页引用的各项诊断开关记录在
[运行时 DFX 开关](03-runtime-dfx.md)。

需要在改完 kernel cpp 之后重新跑一遍编译产物（典型场景：手调 kernel
后用 PMU / swimlane / args-dump 验证修改是否正确），使用 debug 专用
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
        enable_chip_swimlane=True,
    ),
)
```

CLI 形式（从目录里的 `golden.py` 加载输入）:

```bash
python -m pypto.runtime.debug.replay build_output/_jit_xxx/ \
    --pmu 2 --swimlane --log-level debug
```

默认 `recompile=True` 会强制清掉缓存的 `.so` / `.bin`,确保手改的 cpp
能被重新编译。`recompile=False`（或 CLI 的 `--no-recompile`）只关闭该
强制失效；runtime / PTO-ISA 兼容性检查仍会运行，并可能清理和重建缓存产物。
复用还要求 runtime 与 PTO-ISA 的身份都能确定。runtime 源码 checkout 必须
保持干净；安装版 runtime 也可以使用内嵌的 build commit。PTO-ISA 目前必须是
干净的 Git checkout。若任一身份无法确定，PyPTO 会按安全失败策略重新构建，
而不会信任已有二进制。
`--log-level` 接受和
`PYPTO_RUNTIME_LOG` 相同的值（`debug`、`info`、`timing`、`warn`、
`error`、`null`）;加上 `--log-sync-pypto` 可以把同一档位推到
PyPTO 的 C++ logger。

传 `validate=True`（或 `--validate`）会在执行结束后,用
`golden.py::compute_golden` 计算参考输出,并按 `golden.py` 里声明的
`RTOL` / `ATOL` 公差逐 output 比对;不一致会抛 `AssertionError`。
该开关需要目录里存在 `golden.py`（`ir.compile` 默认会产出）。

## 改 `.pto` 而不是 cpp

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

## 自动生成的 `debug/run.py`

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

## 对重放的单芯片构建做 benchmark

`execute_compiled(work_dir, ...)` 是目录驱动的，但 `benchmark()` 需要一个活的
`CompiledProgram` 来拿 orchestration 参数元数据——这份元数据由 IR `Program`
派生，而目录重放没有 IR。所以 `ir.compile()` 会在 `kernel_config.py` 旁边额外
写一个 `compiled_meta.json`（参数元数据 + platform + backend），
`CompiledProgram.from_dir()` 只凭它就能重建出完全可调用的程序——
**不重新编译 pypto、不重跑 pass**：

**`from_dir()` 只重载元数据，不重建源码。** 与 `replay` 不同，它既不做
`.pto` → cpp 拼接，也不失效二进制缓存，所以只调它可能让改动被静默忽略：
改了 `.pto` 不会传到 cpp，改了 cpp 也可能仍被缓存的 `.o` / `.so` 顶替。
按 `replay` 内部的做法显式补上这两步：

```python
from pypto.ir import CompiledProgram
from pypto.runtime import benchmark
from pypto.runtime.debug import invalidate_binary_cache, rebuild_kernel_cpp_from_pto

work_dir = "build_output/<jit_dir>/"
rebuild_kernel_cpp_from_pto(work_dir)  # 仅在改了 ptoas/*.pto 时需要
invalidate_binary_cache(work_dir)      # 丢弃缓存的 .o/.so，让改动真正被编译

compiled = CompiledProgram.from_dir(work_dir, platform="a2a3")
compiled(a, b, c)                                    # 正确性复检
stats = benchmark(compiled, [a, b, c], rounds=100)   # 以及计时
```

`platform` / `backend_type` 默认取编译时记录的值，可覆盖以在别的目标上重放
（例如 `a2a3sim` → `a2a3`）。运行时产物照常从 `kernel_config.py` 重新派生；
重载既不重写 sidecar，也不覆盖手改过的 `debug/run.py`。重建出的对象
`program` 为 `None`（IR 未持久化），`validate_ir()` 仍可从 `passes_dump/` 工作。
multi-orch 的父目录自身没有 sidecar——每个 `next_levels/<name>/` 子构建各有一份，
重载你要的那个子构建即可。分布式构建用
`DistributedCompiledProgram.from_dir`（见下一节）。

编译到**复用的 `output_dir`** 时，sidecar 始终描述本次编译的程序：写入是原子的；
若新程序在该目录下没有可记录的签名（multi-orch 父目录、无法提取签名的
orchestration、IR 中没有同名函数的子构建），则直接删除旧文件。
`ir.compile()` 本身不会清空 `output_dir`，正是这一步保证 `from_dir()` 不会
交出过期的参数 ABI。

一次构建究竟是哪种布局——顶层单个程序，还是每个 orchestration 一个子构建——
由本次 codegen 决定（`pto_backend.multi_chip_orch_names`），而不是扫描目录得出。
因此，上一次 multi-orch 编译遗留的 `next_levels/` 不会让随后编译到同一目录的
single-orch 构建被误判成 multi-orch：新的顶层产物照常可以通过 `compiled(...)`
和 `from_dir()` 访问。遗留的子构建不会被单芯片 codegen 改动，各自的产物与
sidecar 仍然成对匹配（只是整体过期），
`CompiledProgram.from_dir(next_levels/<name>)` 依旧能重放那次旧构建。

**判型标记不会跨构建类型存活。** 一个目录是 L2 还是 L3，由少数几个文件决定：
单芯片看顶层 `kernel_config.py` 与 `compiled_meta.json`，分布式看
`orchestration/host_orch.py` 与 `distributed_meta.json`；而 `replay()` 恰恰是在
「顶层没有 `kernel_config.py`」时才走 L3 路径。因此另一种类型遗留下来的标记不只是
「过期」，它会把整个目录重新指向那次旧构建（旧 L2 的 `kernel_config.py` 会让新的
L3 构建按 L2 重放；旧 L3 的 sidecar 会让 `DistributedCompiledProgram.from_dir`
在已是 L2 产物的目录上仍能加载）。所以每次新编译都会删除**本类型不写的**那些标记，
复用目录要么解析到刚编译的这次构建，要么显式报错。产物目录本身从不删除，只删这些标记。

## 重放 L3 / 分布式构建

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

重建方式与上面单芯片一节相同，只凭 `distributed_meta.json`。
两个入口直接暴露这个能力：

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

这里持久化的参数元数据是 HOST orchestrator 的（与 `host_orch.py` 匹配的
post-SSA 名字），chip callables 通过遍历 `next_levels/` 重建；
`distributed_config` 与 `platform` 一样可覆盖。

`distributed_meta.json` 与单芯片 sidecar 共用同一套加载逻辑，因而契约一致：
原子写入；当编译进该目录的程序没有可记录的签名时直接删除；加载时逐字段校验
（含 `distributed_config` 块）。所以手改或被截断的 sidecar 会以一个
`ValueError` 失败，并指明文件名与重新编译的修复方式。

### 在拿到卡之前就把二进制编好

芯片二进制默认在第一次 prepare / 调用时才编译，于是整条 ptoas/ccec + g++ 工具链
就压在一次已经占着卡的运行的关键路径上。`build_binaries()` 只做这部分工作、不做
别的，因此可以在一台没有 NPU 的 CPU 机器上跑：

```python
prog = DistributedCompiledProgram.from_dir("build_output/<jit_dir>/")
prog.build_binaries()          # 只有 ptoas/ccec + g++；不碰 NPU，不 fork
```

每个 `next_levels/<name>/` 子构建都会被编译，`.o`/`.so` 落在该子构建自己的
`cache/` 下，并盖上它们所依据的 runtime 与 PTO-ISA 身份图章。之后对同一目录的
`prepare()` / `__call__()` 会直接复用 —— 换一个进程也算，通过 `from_dir` 复用一个
并非本对象产出的目录也算。在一个三子构建的 host collective 上实测：冷编 ~1.6s，
热装配 ~0.05s。

这些二进制与 `device_ids` 无关：`ir.compile` 只在 codegen **之后**才消费
`distributed_config`，所以这样编出来的目录可以通过上面的 `distributed_config`
覆盖在任意卡集合上重放。这正是编译与执行得以在空间或时间上分离的原因 —— 既可以
在 serving worker 借卡之前预热它的二进制，也可以把整个测试套件在卡外预编译、再把
产物交给一次短暂的上卡运行。`tests/st/README.md` 描述了分布式系统测试采用的两趟
形式。

**这不是通用的编译缓存。** 复用是按目录为键的，而目录内部那层生成 kernel 的二进制
缓存又是按位置而非内容为键，所以把**不同的**源码编进一个已有二进制的目录会复用到
陈旧的产物。请为每个 commit 编进全新目录。

L3 replay 会把 `RunConfig` 中的运行时 DFX 字段透传到每个芯片派发，产物写入
`dfx_outputs/rank{r}/d{k}/`；onboard 泳道使用
[运行时 DFX 开关](03-runtime-dfx.md) 里描述的抓图/计时两趟协议。
因此「改完再跑」既支持正确性复检，也支持 L3 运行时诊断。
