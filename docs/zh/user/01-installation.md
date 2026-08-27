# 安装

从源码安装 PyPTO、验证安装，并了解 examples 目录的组织。

## Concept

PyPTO 是一个带 C++ 编译核心的 Python 包。安装即构建：除 Python 外还需要 C++17 工具链与
CMake。[scikit-build-core](https://scikit-build-core.readthedocs.io/) 会从 `pip` 驱动
CMake，所以一条普通的 `pip install` 就能完成全部工作。

安装得到的是**编译器前端** —— 足以编写 kernel 并查看它们解析成的 IR。有两样东西 `pip`
**不会**安装，在执行需要它们的命令之前值得先知道：

| 要做这件事 | 还需要 |
| ---------- | ------ |
| 编写 kernel、跑 pass 流水线、读 IR | 除安装外无需其他 |
| 把 kernel 编译成生成的 C++ | 除安装外无需其他。**ptoas**（单独分发，版本固定在 `toolchain/versions.env`）负责其中的汇编步骤；`@pl.jit` 会检测它是否存在，不存在时自动跳过该步骤 |
| 运行已编译的 kernel | 运行时，加一块 NPU 或模拟器平台 |

### 可选的 AI agent skills

[pypto-skills](https://github.com/hw-native-sys/pypto-skills) marketplace 提供了一组插件，
让受支持的 AI 编程 agent 掌握项目的公共工作流。这些插件安装到 agent 中，与 PyPTO Python
包的安装相互独立：

| 插件 | 适用对象 | 包含的工作流 |
| ---- | -------- | ------------ |
| `pypto-user` | PyPTO 用户 | 生成 IR trace、分析 in-core kernel 性能 |
| `pypto-developer` | PyPTO 贡献者 | Git、PR、issue 与分支管理工作流 |

Codex 安装方式：

```bash
codex plugin marketplace add hw-native-sys/pypto-skills
codex plugin add pypto-user@pypto-skills

# 可选：同时安装贡献者工作流
codex plugin add pypto-developer@pypto-skills
```

Claude Code 安装方式：

```bash
claude plugin marketplace add hw-native-sys/pypto-skills
claude plugin install pypto-user@pypto-skills

# 可选：同时安装贡献者工作流
claude plugin install pypto-developer@pypto-skills
```

下面的验证步骤刻意只停留在第一行。

## Quickstart

```bash
git clone https://github.com/hw-native-sys/pypto.git
cd pypto

# 先装 CPU 版 torch —— 默认 wheel 会拉约 2GB 的 CUDA 依赖
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

验证：

```bash
python -c "import pypto.language as pl; from pypto import ir; print(len(pl.__all__), 'exports')"
```

预期输出 —— 这个数字会随算子增加而变化，所以数字不同属正常，真正的信号是有没有 traceback：

```text
226 exports
```

然后确认一个真实 kernel 能走通整条 pass 流水线。`lower()` 会特化 JIT 函数、运行配置对应的
Pass 流水线，并返回 Pass 后的 `ir.Program`。它不会执行代码生成，也不会填充编译缓存，因此既
不需要 ptoas 也不需要设备。需要验证代码生成时请使用 `compile()`。请运行仓库中已有的 lowering
示例，不要把函数管道给 `python -`：`@pl.jit` 需要读取被装饰函数的源码，而 stdin 上取不到。

```bash
python examples/intermediate/05_assemble.py
```

最后一行是：

```text
OK
```

只要这行打印出来，就说明 C++ 核心导入成功、parser 构建出了 IR、整条 pass 流水线都跑过了。
这里出现 traceback 才是真正的信号 —— 那行字的具体措辞不是。

## Mechanics

### 前置条件

| 要求 | 版本 | 说明 |
| ---- | ---- | ---- |
| Python | ≥ 3.10 | `pyproject.toml` 中的 `requires-python`；DSL 使用 3.10+ 语法 |
| CMake | ≥ 3.15 | 由 scikit-build-core 调用，不需要你手动执行 |
| C++ 编译器 | C++17 | GCC 或 Clang。`CMAKE_CXX_STANDARD 17` 是强制要求而非建议 |
| numpy | ≥ 2.0 | 自动安装 |
| torch | ≥ 2.0 | 自动安装，但请先装 CPU 版（见下） |
| nanobind | ≥ 2.0, < 3 | 仅构建期需要，自动获取 |
| scikit-build-core | ≥ 0.10 | 构建后端，自动获取 |

上表给出的是 pypto 兼容的版本范围。CI 实际构建所用的确切版本固定在仓库根目录的
`build-constraints.txt` 中。本地复现 CI 构建时，把它指给 pip：

```bash
PIP_BUILD_CONSTRAINT=$PWD/build-constraints.txt \
PIP_CONSTRAINT=$PWD/build-constraints.txt \
    pip install -e .
```

两个变量都要传，因为它们覆盖不同的 pip 版本：`PIP_BUILD_CONSTRAINT` 是 pip 26.2
起用于约束构建依赖的变量，`PIP_CONSTRAINT` 则是更早版本在该场景下认的那个。只传后者
的话，较新的 pip 会重新自由解析 `[build-system] requires`，构建出来的就不是 CI 验证
过的那一份了。

**先装 CPU 版 torch，再装 PyPTO。** `pip install -e .` 会把 `torch>=2.0.0` 解析到默认
wheel，它携带完整 CUDA 栈 —— 约 2GB，而 PyPTO 的工作流一点也用不到。先从 CPU 索引安装
`torch`，后续解析就变成空操作。

### 安装模式

```bash
pip install -e .            # 可编辑 —— 改 Python 代码无需重装
pip install .               # 常规安装
pip install -e ".[dev]"     # 可编辑 + pytest、ruff、pyright、clang-tidy
```

开发 PyPTO 本身时默认用可编辑模式。注意它只对 **Python** 可编辑：改动 `src/` 或 `include/`
下的 C++ 仍然需要重新构建。

### 构建选项

默认构建类型是 `RelWithDebInfo`（带调试符号的优化版本）。通过环境变量覆盖：

```bash
CMAKE_BUILD_TYPE=Release pip install .
```

`RelWithDebInfo` 携带完整调试信息 —— 调试器需要它，它也占了产物的绝大部分：扩展模块
304 MiB，其中 292 MiB 是 DWARF。设置 `PYPTO_DEBUG_INFO_LEVEL=1` 只保留 backtrace 会读的
部分（函数描述与行号表），扩展模块降到 62 MiB、其 wheel 降到 18 MiB，编译快约三分之一，
而错误信息里的 `C++ Traceback` 完全不变。失去的是局部变量和类型信息，所以准备挂调试器时
请保持默认值：

```bash
SKBUILD_CMAKE_DEFINE=PYPTO_DEBUG_INFO_LEVEL=1 pip install .
```

检测到 `ccache` 时会自动启用，能显著降低重复构建的成本：

```bash
sudo apt-get install ccache   # Debian / Ubuntu
brew install ccache           # macOS
```

### 编译产物的位置

编译一个 program 会把生成代码、报告和 pass dump 写到当前工作目录下 `build_output/` 里一个
带时间戳的目录。`PYPTO_PROG_BUILD_DIR` 可以改变这个基准目录 —— 它是**运行时环境变量**，
逐进程读取：

```bash
PYPTO_PROG_BUILD_DIR=/scratch/pypto-out python my_kernel.py
```

### examples 目录导览

`examples/` 按难度组织，是了解 PyPTO 惯用写法最快的途径。

**`examples/beginner/`** —— 一个文件一个概念。

| 文件 | 展示 |
| ---- | ---- |
| `01_hello_world.py` | 最小的完整程序 |
| `02_elementwise.py` | tile 加 / 乘，以及对分块的循环 |
| `03_scalar_ops.py` | 标量操作数 |
| `04_activation.py` | `relu`、SiLU |
| `05_matmul.py` | cube 上一次算完的 64x64 matmul |
| `06_concat.py` | 两个 tile 写进互不相交的列区间 |

**`examples/intermediate/`** —— 真实 kernel 模式。

| 文件 | 展示 |
| ---- | ---- |
| `01_fused_linear.py` | cube matmul + vector bias-add + relu |
| `02_softmax.py` | 按行 softmax |
| `03_normalization.py` | RMSNorm、LayerNorm |
| `04_matmul_acc.py` | K 维分块 + 累加器 |
| `05_assemble.py` | 按偏移把 tile 写进目标 |
| `06_dyn_valid_shape.py` | 运行时收窄的有效范围 |
| `07_task_graph.py` | 一条推断的边与一条声明的边 |

**`examples/advanced/`** —— 性能技巧。

| 文件 | 展示 |
| ---- | ---- |
| `01_split_k.py` | 切分规约维 |
| `02_auto_tile_matmul.py` | 编译器驱动的 L0 分块 |
| `03_mixed_kernel.py` | cube 与 vector 同作用域，三种 split 模式 |

**`examples/models/`** —— 多 kernel 模型。

| 文件 | 展示 |
| ---- | ---- |
| `01_ffn.py` | 一个 FFN 模块 |
| `02_vector_dag.py` | 三个 InCore kernel 连成 DAG |
| `03_flash_attention.py` | 循环携带状态、嵌套 `if` / `pl.yield_` |
| `04_paged_attention.py` | paged attention，在线 softmax，4 kernel 流水 |
| `05_paged_attention_batch.py` | batch 循环挪进 kernel 内部 |
| `06_paged_attention_dynamic.py` | `pl.dynamic()` 形状 |
| `07_paged_attention_multi_config.py` | unroll 分组 + 由 `pl.tensor.dim()` 得到的形状 |
| `08_llama_mini.py` | 一个完整的小型 LLaMA 风格模型 |
| `09_paged_attention_spmd.py` | batch 维分布到 SPMD block 上 |
| `qwen3_jit/` | 按模块拆成多个 kernel 文件的 `@pl.jit` decode 路径 |

**`examples/utils/`** —— 只用前端。

| 文件 | 展示 |
| ---- | ---- |
| `cross_function_calls.py` | `@pl.jit.inline` 辅助函数在调用点展开 |
| `error_handling.py` | 裸重绑定的代价，以及它怎么暴露出来 |
| `parse_from_text.py` | 从字符串或文件 `pl.parse()` / `pl.loads()` |
| `phase_fence_dep_compression.py` | 扇出阶段之间的整数组 TaskId 栅栏 |

**`examples/runtime/`** —— host 侧模式。

| 文件 | 展示 |
| ---- | ---- |
| `explicit_dispatch.py` | 注册一次，多次派发 |
| `multi_program_kv_cache.py` | 跨程序共享的常驻 buffer |
| `distributed_callback.py` | 作为 Python 回调的 HOST `SubWorker` |

**`examples/distributed/`** —— 每个集合通信 / 原语一个文件，在 [分布式](distributed/index.md)
里逐页讲解。

**这些例子多数会派发到硬件，而不只是编译。** `beginner/01_hello_world.py`、
`intermediate/02_softmax.py`、`models/01_ffn.py` 最后都以 `config=RunConfig()` 调用各自的
kernel，也就是经 ptoas 汇编后真正运行 —— 因此它们需要运行时和一块设备或模拟器平台，仅有上面的
`pip install` 是不够的：

```bash
python examples/beginner/01_hello_world.py     # 需要运行时 + 设备/模拟器
python examples/intermediate/02_softmax.py     # 需要运行时 + 设备/模拟器
python examples/models/01_ffn.py               # 需要运行时 + 设备/模拟器
```

如果你只有编译器前端，读它们而不要运行 —— `examples/utils/` 是最接近“仅解析与查看”的那部分。

### 运行测试

```bash
pip install -e ".[dev]"

python -m pytest tests/ut -n auto --maxprocesses 8 -v      # 单元测试
python -m pytest tests/ut/core/test_error.py -v            # 单个文件
```

系统测试位于 `tests/st/`，需要设备或模拟器，参见 `tests/st/README.md`。

## 边界情况

> **致命陷阱：** 在装 CPU 版 torch 之前先装 PyPTO，会静默拉取完整的 CUDA 版 torch ——
> 约 2GB 的包，而 PyPTO 工作流一个都不会加载。没有任何报错和警告，唯一的症状是安装极慢、
> 环境极大。请**先**从 CPU 索引安装 `torch`。

| Symptom | Likely Cause | Fix |
| ------- | ------------ | --- |
| **`pip install` 期间 C++ 编译报错** | 工具链版本低于源码所用的 C++17 特性 | 让 CMake 指向更新的编译器：`CMAKE_CXX_COMPILER=/path/to/g++ pip install -e .` |
| **改完 C++ 后 `pypto_core` 导入失败** | 可编辑安装只跟踪 Python | 重新构建：`pip install -e . --no-build-isolation` |
| **导入成功但新增的绑定不存在** | Python 源码旁的 `.so` 是旧的 | 重新构建，并确认 `python/pypto/` 下的 `.so` 比你的 C++ 改动更新 |
| **安装拉取了数 GB 的 nvidia 包** | torch 从默认索引解析 | 先执行 `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| **编译产物出现在意料之外的目录** | 环境里设了 `PYPTO_PROG_BUILD_DIR` | 取消该变量，或给 `ir.compile` 传 `output_dir=` |

**环境变量 vs 编译期宏。** `PYPTO_PROG_BUILD_DIR` 与 `PYPTO_VERIFY_LEVEL` 在运行时从进程
环境读取，所以 `VAR=value python kernel.py` 有效。`SIMPLER_HOST_STRACE` 与 `SIMPLER_DFX`
是**运行时的编译期宏**，在构建运行时时以 `-DXXX=1` 设置 —— 在 shell 里 export 它们不起任何作用。

## See Also

- [快速上手](02-quickstart.md) —— 导入能跑通之后，写你的第一个 kernel。
- [编程模型](03-programming-model.md) —— 这些 kernel 所依赖的抽象。
- [PTO 项目生态](../dev/00-ecosystem.md) —— PyPTO、PTOAS、pto-isa 与运行时的关系。
- [运行时文档](https://hw-native-sys.github.io/simpler/) —— 安装与操作执行已编译程序的运行时。
