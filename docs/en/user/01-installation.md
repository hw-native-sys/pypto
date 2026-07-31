# Installation

Install PyPTO from source, verify the install, and find your way around the examples.

## Concept

PyPTO is a Python package with a compiled C++ core. Installing it builds that core, so
an install is a build: you need a C++17 toolchain and CMake in addition to Python.
[scikit-build-core](https://scikit-build-core.readthedocs.io/) drives CMake from `pip`,
so a plain `pip install` does the whole thing.

What you get from an install is the **compiler front end** — enough to write kernels and
inspect the IR they parse into. Two things are *not* installed by `pip` and are worth
knowing about before you follow a command that needs them:

| To do this | You also need |
| ---------- | ------------- |
| Write kernels, run the pass pipeline, read the IR | Nothing beyond the install |
| Compile a kernel to generated C++ | Nothing beyond the install. **ptoas** (distributed separately, versions pinned in `toolchain/versions.env`) adds the assembly step; `@pl.jit` detects whether it is present and skips that step when it is not |
| Run a compiled kernel | The runtime plus an NPU or a simulator platform |

The verification below deliberately stays in the first row.

## Quickstart

```bash
git clone https://github.com/hw-native-sys/pypto.git
cd pypto

# CPU-only torch first — the default wheel pulls ~2 GB of CUDA dependencies
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

Verify:

```bash
python -c "import pypto.language as pl; from pypto import ir; print(len(pl.__all__), 'exports')"
```

Expected output — the count moves as operators are added, so treat a different number as
fine and a traceback as the real signal:

```text
226 exports
```

Then check that a real kernel makes it through the pass pipeline. `lower()` specializes
the JIT function, runs the configured pass pipeline, and returns the post-pass
`ir.Program`. It performs no code generation and does not populate the compiled-program
cache, so this needs neither ptoas nor a device. Use `compile()` to verify code generation.
Write the check to a file rather than piping it to `python -`: `@pl.jit` reads the
decorated function's source, which is unavailable on stdin.

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

If that prints, the C++ core imported, the parser built IR, and the whole pass pipeline
ran. A traceback here is the real signal — the exact wording of the line is not.

## Mechanics

### Prerequisites

| Requirement | Version | Notes |
| ----------- | ------- | ----- |
| Python | ≥ 3.10 | `requires-python` in `pyproject.toml`; the DSL uses 3.10+ syntax |
| CMake | ≥ 3.15 | Invoked by scikit-build-core, not by you |
| C++ compiler | C++17 | GCC or Clang. `CMAKE_CXX_STANDARD 17` is required, not merely preferred |
| numpy | ≥ 2.0 | Installed automatically |
| torch | ≥ 2.0 | Installed automatically, but install the CPU wheel first (see below) |
| nanobind | ≥ 2.0 | Build-time only; fetched automatically |
| scikit-build-core | ≥ 0.10 | Build backend; fetched automatically |

**Install the CPU torch wheel before PyPTO.** `pip install -e .` resolves `torch>=2.0.0`
to the default wheel, which carries the full CUDA stack — around 2 GB that a PyPTO
workflow never uses. Installing `torch` from the CPU index first makes the later resolve
a no-op.

### Install modes

```bash
pip install -e .            # editable — Python edits take effect without reinstalling
pip install .               # regular install
pip install -e ".[dev]"     # editable + pytest, ruff, pyright, clang-tidy
```

Editable mode is the right default while working on PyPTO itself. Note that it is
editable for *Python* only: changing C++ under `src/` or `include/` still requires a
rebuild.

### Build options

The default build type is `RelWithDebInfo` — optimized, with debug symbols. Override it
through the environment:

```bash
CMAKE_BUILD_TYPE=Release pip install .
```

`ccache` is detected and used automatically when present, which makes repeated builds
substantially cheaper:

```bash
sudo apt-get install ccache   # Debian / Ubuntu
brew install ccache           # macOS
```

### Where compile output goes

Compiling a program writes generated code, reports, and pass dumps to a timestamped
directory under `build_output/` in the current working directory. `PYPTO_PROG_BUILD_DIR`
relocates that base — it is a **runtime environment variable**, read per process:

```bash
PYPTO_PROG_BUILD_DIR=/scratch/pypto-out python my_kernel.py
```

### A tour of the examples

`examples/` is ordered by difficulty, and is the fastest way to see idiomatic PyPTO.

| Path | What is in it |
| ---- | ------------- |
| `examples/hello_world.py` | The simplest complete program — start here |
| `examples/kernels/` | Single-kernel operators, numbered by difficulty: elementwise, fused ops, matmul, softmax, assemble |
| `examples/models/` | Multi-kernel models, numbered by difficulty: FFN, paged attention, LLaMA |
| `examples/utils/` | Parsing, cross-function calls, error handling |
| `examples/runtime/` | Dispatch, explicit workers, distributed callbacks, multi-program KV cache |

**Most of these dispatch to hardware, not just compile.** `hello_world.py`,
`kernels/06_softmax.py`, and `models/01_ffn.py` all end by calling their kernel with
`config=RunConfig()`, which assembles through ptoas and runs it — so they need the
runtime and a device or simulator platform, not only the `pip install` above:

```bash
python examples/hello_world.py          # needs runtime + device/simulator
python examples/kernels/06_softmax.py   # needs runtime + device/simulator
python examples/models/01_ffn.py        # needs runtime + device/simulator
```

If you only have the compiler front end, read them rather than running them —
`examples/utils/` is the subset that stays closest to parse-and-inspect.

### Running the test suite

```bash
pip install -e ".[dev]"

python -m pytest tests/ut -n auto --maxprocesses 8 -v      # unit tests
python -m pytest tests/ut/core/test_error.py -v            # one file
```

System tests live under `tests/st/` and need a device or simulator; see
`tests/st/README.md`.

## Edge Cases

> **Fatal pitfall:** installing PyPTO before the CPU torch wheel silently pulls the full
> CUDA torch distribution — roughly 2 GB of packages that nothing in a PyPTO workflow
> loads. There is no error and no warning; the only symptom is a very long install and a
> very large environment. Install `torch` from the CPU index *first*.

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| **C++ compile errors during `pip install`** | Toolchain older than the C++17 features the sources use | Point CMake at a newer compiler: `CMAKE_CXX_COMPILER=/path/to/g++ pip install -e .` |
| **`ImportError` on `pypto_core` after editing C++** | Editable installs track Python only | Rebuild: `pip install -e . --no-build-isolation` |
| **Import succeeds but a new binding is missing** | Stale `.so` copied next to the Python sources | Rebuild, and confirm the `.so` under `python/pypto/` is newer than your C++ edit |
| **Install pulls gigabytes of nvidia packages** | torch resolved from the default index | `pip install torch --index-url https://download.pytorch.org/whl/cpu` first |
| **Compile output appears in an unexpected directory** | `PYPTO_PROG_BUILD_DIR` set in the environment | Unset it, or pass `output_dir=` to `ir.compile` |

**Environment variables vs. compile-time macros.** `PYPTO_PROG_BUILD_DIR` and
`PYPTO_VERIFY_LEVEL` are read from the process environment at runtime, so
`VAR=value python kernel.py` works. `SIMPLER_HOST_STRACE` and `SIMPLER_DFX` are
**compile-time macros of the runtime**, set with `-DXXX=1` when the runtime is built —
exporting them in a shell does nothing.

## See Also

- [Quickstart](02-quickstart.md) — your first kernels, once the import works.
- [Programming Model](03-programming-model.md) — the abstractions those kernels are built from.
- [PTO Project Ecosystem](../dev/00-ecosystem.md) — how PyPTO, PTOAS, pto-isa, and the runtime relate.
- [Runtime documentation](https://hw-native-sys.github.io/simpler/) — installing and operating the runtime that executes compiled programs.
