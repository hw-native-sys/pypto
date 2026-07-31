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
不需要 ptoas 也不需要设备。需要验证代码生成时请使用 `compile()`。请写成文件再运行，不要管道
给 `python -`：`@pl.jit` 需要读取被装饰函数的源码，而 stdin 上取不到。

```bash
cat > /tmp/pypto_check.py <<'PY'
import pypto.language as pl
import torch

@pl.jit
def add(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        out = pl.add(a, b)
    return out

x = torch.zeros((128, 128), dtype=torch.float32)
program = add.lower(x, x, x)
print("pipeline OK:", type(program).__name__)
PY

python /tmp/pypto_check.py
```

```text
pipeline OK: Program
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
| nanobind | ≥ 2.0 | 仅构建期需要，自动获取 |
| scikit-build-core | ≥ 0.10 | 构建后端，自动获取 |

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

| 路径 | 内容 |
| ---- | ---- |
| `examples/hello_world.py` | 最简单的完整程序 —— 从这里开始 |
| `examples/kernels/` | 单 kernel 算子，按难度编号：逐元素、融合算子、矩阵乘、softmax、assemble |
| `examples/models/` | 多 kernel 模型，按难度编号：FFN、paged attention、LLaMA |
| `examples/utils/` | 解析、跨函数调用、错误处理 |
| `examples/runtime/` | 派发、显式 worker、分布式回调、多程序 KV cache |

**这些例子多数会派发到硬件，而不只是编译。** `hello_world.py`、`kernels/06_softmax.py`、
`models/01_ffn.py` 最后都以 `config=RunConfig()` 调用各自的 kernel，也就是经 ptoas 汇编后
真正运行 —— 因此它们需要运行时和一块设备或模拟器平台，仅有上面的 `pip install` 是不够的：

```bash
python examples/hello_world.py          # 需要运行时 + 设备/模拟器
python examples/kernels/06_softmax.py   # 需要运行时 + 设备/模拟器
python examples/models/01_ffn.py        # 需要运行时 + 设备/模拟器
```

如果你只有编译器前端，读它们而不要运行 —— `examples/utils/` 是最接近“仅解析与查看”的那部分。

### 运行测试

```bash
pip install -e ".[dev]"

python -m pytest tests/ut -n auto --maxprocesses 8 -v      # 单元测试
python -m pytest tests/ut/core/test_error.py -v            # 单个文件
```

系统测试位于 `tests/st/`，需要设备或模拟器，参见 `tests/st/README.md`。

## Edge Cases

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
