# 编译程序

把一个程序或 `@pl.jit` kernel 变成设备 kernel 与 host 编排代码。

> **本页原有的语言内容已经迁走**，去了 [语言指南](language/index.md) 章：类型、函数、控制流、内存、作用域与任务、编译期指令各自独立成页，算子表面在 [算子](ops/index.md)。留在这里的是编译部分，等待它自己的章节。

## `ir.compile()`

```python
from pypto import ir
from pypto.backend import BackendType

output_dir = ir.compile(
    program,
    output_dir=None,                           # 为 None 时自动生成
    strategy=ir.OptimizationStrategy.Default,  # 唯一的优化策略
    dump_passes=True,                          # 将 IR 快照写入 output_dir/passes_dump/
    backend_type=BackendType.Ascend910B,
)
```

| 参数 | 取值 | 说明 |
| ---- | ---- | ---- |
| `program` | `ir.Program` | 必填的程序对象（来自 `@pl.program` 或等价物） |
| `strategy` | `OptimizationStrategy.Default` | `Default` = 完整的面向张量流水线（唯一策略） |
| `backend_type` | `BackendType.Ascend910B`、`BackendType.Ascend950` | pass 与 codegen 的目标硬件（从 `pypto.backend` 导入 `BackendType`） |
| `dump_passes` | `PassDumpLevel` | 每个 pass 之后把 IR 快照写到 `<output_dir>/passes_dump/` |
| `skip_ptoas` | `True` / `False` | 跳过 ptoas 步骤；输出裸 `.pto`（MLIR）而非编译好的 C++ 包装（默认 `False`） |
| `output_dir` | 路径或 `None` | 为 `None` 时使用 `<base>/<program_name>_<timestamp>`，其中 `<base>` 取自 `PYPTO_PROG_BUILD_DIR`，未设则为 `build_output`；目录按需创建 |
| `verification_level` | `None`、`ir.VerificationLevel.NONE`、`BASIC` | `None` = 使用默认值（`BASIC`，或由 `PYPTO_VERIFY_LEVEL` 覆盖） |

`ir.compile()` 的参数不止上面这些 —— 还包括诊断、内存规划器选择、profiling、平台以及分布式配置等。完整的表属于尚未编写的执行章节；在此之前请阅读 `python/pypto/ir/compile.py` 里的签名。

## `JITFunction.compile()`

`@pl.jit` kernel 通常把特化 + 编译 + 派发融合进一次 `kernel(*args)` 调用。`compile(*sample_args)` 在编译后停下并返回 `CompiledProgram`：

```python
@pl.jit
def my_kernel(x, w, out): ...

compiled = my_kernel.compile(sample_x, sample_w, sample_out)
print("artifacts in:", compiled.output_dir)

from pypto.runtime import ChipWorker, RunConfig

worker = ChipWorker(config=RunConfig(platform="a2a3sim"))
w_dev = worker.alloc_tensor(sample_w.shape, sample_w.dtype, init=sample_w)
handle = worker.register(compiled)
for batch in stream:
    handle(batch.x, w_dev, batch.out)
worker.free_tensor(w_dev)
worker.close()
```

- `compile()` 与 `__call__` 一样接受 `config=RunConfig(...)`：编译侧开关（`strategy`、`dump_passes`、诊断……）会转发给 `ir.compile()`。运行时侧字段（`device_id`、DFX 标志）影响的是派发，不影响产物。
- 返回的 `CompiledProgram` 就是 JIT 缓存持有的那个对象，因此之后用同一特化 key 调用会拿到完全相同的实例。
- 它暴露完整的提取接口 —— `chip_callable`、`runtime_name`、`runtime_config`、`build_orch_args`、`build_call_config`、`output_dir`、`platform`、`output_indices` —— 因此直接驱动运行时的测试框架无需再写一个 `@pl.program` 包装。

**`compile()` 的位置参数是 kernel 自己的参数，不是编译选项。** `compile(skip_ptoas=True)` 会被当作 kernel 实参并静默忽略；编译选项走 `config=RunConfig(...)`。

## 检视结果

`node.as_python()` 打印一个函数或程序的 IR；`concise=True` 会省略中间类型注解。`JITFunction` 本身没有 `as_python()` —— IR 在特化存在之后才存在，所以请打印 `compiled.program.as_python()`。

`dump_passes=` 会在每个 pass 之后把快照写到 `passes_dump/` 下，这是你定位"是哪个 pass 改动了某个东西"的手段。

## 参见

- [语言指南](language/index.md) —— 本页原先覆盖的语言表面。
- [算子](ops/index.md) —— 算子目录。
- [函数与程序](language/01-functions.md) —— 正在被编译的那些装饰器。
- [Pass Manager](../dev/passes/00-pass_manager.md) —— `strategy` 所选择的流水线。
- [Torch Codegen 调试指南](03-torch_codegen_debug.md) —— 把结果与 PyTorch 对拍。
