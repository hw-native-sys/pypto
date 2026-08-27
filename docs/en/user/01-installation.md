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

### Optional AI agent skills

The [pypto-skills](https://github.com/hw-native-sys/pypto-skills) marketplace provides
plugins that teach supported AI coding agents the project's common workflows. These
plugins are installed into the agent, independently of the PyPTO Python package:

| Plugin | Intended for | Included workflows |
| ------ | ------------ | ------------------ |
| `pypto-user` | PyPTO users | Generate IR traces and profile in-core kernels |
| `pypto-developer` | PyPTO contributors | Git, pull request, issue, and branch workflows |

For Codex:

```bash
codex plugin marketplace add hw-native-sys/pypto-skills
codex plugin add pypto-user@pypto-skills

# Optional: add contributor workflows too
codex plugin add pypto-developer@pypto-skills
```

For Claude Code:

```bash
claude plugin marketplace add hw-native-sys/pypto-skills
claude plugin install pypto-user@pypto-skills

# Optional: add contributor workflows too
claude plugin install pypto-developer@pypto-skills
```

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
Run the checked-in lowering example rather than piping a function to `python -`:
`@pl.jit` reads the decorated function's source, which is unavailable on stdin.

```bash
python examples/intermediate/05_assemble.py
```

The final line is:

```text
OK
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
| nanobind | ≥ 2.0, < 3 | Build-time only; fetched automatically |
| scikit-build-core | ≥ 0.10 | Build backend; fetched automatically |

The ranges above are what pypto is compatible with. The exact versions CI
builds against are pinned in `build-constraints.txt` at the repository root. To
reproduce a CI build locally, point pip at it:

```bash
PIP_BUILD_CONSTRAINT=$PWD/build-constraints.txt \
PIP_CONSTRAINT=$PWD/build-constraints.txt \
    pip install -e .
```

Both variables, because they cover different pip versions:
`PIP_BUILD_CONSTRAINT` is the one pip applies to build dependencies from 26.2
on, and `PIP_CONSTRAINT` is what earlier pip honours there. With only the
latter, a recent pip resolves `[build-system] requires` freely again and the
build is no longer the one CI validated.

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

`RelWithDebInfo` carries full debug info, which is what a debugger needs — and also most
of the artifact: the extension is 304 MiB, of which 292 MiB is DWARF. Set
`PYPTO_DEBUG_INFO_LEVEL=1` to keep only what backtraces read, the function descriptions
and line-number tables. The extension drops to 62 MiB and its wheel to 18 MiB, the
compile is about a third faster, and the `C++ Traceback` in an error message is
unchanged. What goes is local variables and types, so keep the default whenever you
plan to attach a debugger:

```bash
SKBUILD_CMAKE_DEFINE=PYPTO_DEBUG_INFO_LEVEL=1 pip install .
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

**`examples/beginner/`** — one concept per file.

| File | Shows |
| ---- | ----- |
| `01_hello_world.py` | The smallest complete program |
| `02_elementwise.py` | Tile add / mul, and a loop over chunks |
| `03_scalar_ops.py` | A scalar operand |
| `04_activation.py` | `relu`, SiLU |
| `05_matmul.py` | One 64x64 matmul on the cube |
| `06_concat.py` | Two tiles into disjoint column ranges |

**`examples/intermediate/`** — real-kernel patterns.

| File | Shows |
| ---- | ----- |
| `01_fused_linear.py` | Cube matmul + vector bias-add + relu |
| `02_softmax.py` | Row-wise softmax |
| `03_normalization.py` | RMSNorm, LayerNorm |
| `04_matmul_acc.py` | K-dimension tiling with an accumulator |
| `05_assemble.py` | Writing a tile into a target at an offset |
| `06_dyn_valid_shape.py` | A runtime-narrowed valid extent |
| `07_task_graph.py` | An inferred edge and a declared edge |

**`examples/advanced/`** — performance techniques.

| File | Shows |
| ---- | ----- |
| `01_split_k.py` | Splitting the reduction dimension |
| `02_auto_tile_matmul.py` | Compiler-driven L0 tiling |
| `03_mixed_kernel.py` | Cube and vector in one scope, three split modes |

**`examples/models/`** — multi-kernel models.

| File | Shows |
| ---- | ----- |
| `01_ffn.py` | An FFN module |
| `02_vector_dag.py` | Three InCore kernels wired into a DAG |
| `03_flash_attention.py` | Loop-carried state, nested `if` / `pl.yield_` |
| `04_paged_attention.py` | Paged attention, online softmax, 4-kernel pipeline |
| `05_paged_attention_batch.py` | The batch loop moved inside the kernels |
| `06_paged_attention_dynamic.py` | `pl.dynamic()` shapes |
| `07_paged_attention_multi_config.py` | Unroll grouping + shapes from `pl.tensor.dim()` |
| `08_llama_mini.py` | A complete small LLaMA-style model |
| `09_paged_attention_spmd.py` | The batch dimension across SPMD blocks |
| `qwen3_jit/` | A `@pl.jit` decode path split into per-module kernel files |

**`examples/utils/`** — the front end on its own.

| File | Shows |
| ---- | ----- |
| `cross_function_calls.py` | `@pl.jit.inline` helpers spliced at the call site |
| `error_handling.py` | What a bare rebinding costs, and how it surfaces |
| `parse_from_text.py` | `pl.parse()` / `pl.loads()` from a string or a file |
| `phase_fence_dep_compression.py` | Whole-array TaskId fences between fan-out stages |

**`examples/runtime/`** — host-side patterns.

| File | Shows |
| ---- | ----- |
| `explicit_dispatch.py` | Register once, dispatch many |
| `multi_program_kv_cache.py` | A resident buffer shared across programs |
| `distributed_callback.py` | A HOST `SubWorker` as a Python callback |

**`examples/distributed/`** — one file per collective / primitive, covered page by page in
[Distributed](distributed/index.md).

**Most of these dispatch to hardware, not just compile.** `beginner/01_hello_world.py`,
`intermediate/02_softmax.py`, and `models/01_ffn.py` all end by calling their kernel with
`config=RunConfig()`, which assembles through ptoas and runs it — so they need the
runtime and a device or simulator platform, not only the `pip install` above:

```bash
python examples/beginner/01_hello_world.py     # needs runtime + device/simulator
python examples/intermediate/02_softmax.py     # needs runtime + device/simulator
python examples/models/01_ffn.py               # needs runtime + device/simulator
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
