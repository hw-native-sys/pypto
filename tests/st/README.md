# PyPTO System Tests

This directory contains system-level integration tests for PyPTO. The testing framework (`harness`) is included internally in `tests/st/harness/`. These tests validate the complete compilation and execution pipeline from PyPTO DSL programs to executable code on target platforms.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Running Tests](#running-tests)
- [Test Configuration Options](#test-configuration-options)
- [Advanced Usage](#advanced-usage)
- [Writing New Tests](#writing-new-tests)
- [Troubleshooting](#troubleshooting)

## Overview

System tests use the internal `harness` package to perform end-to-end validation of PyPTO programs:

1. **PyPTO Frontend**: Defines tensor operations using Python DSL
2. **Compilation Pipeline**: Transforms high-level IR through optimization passes to generate kernels
3. **Simpler Runtime**: Executes generated code on simulator or hardware platforms
4. **Validation**: Compares runtime results against PyTorch reference implementations

**Test Flow:**

```text
Test Case Definition → Build IR → Generate Kernels → Compile → Execute → Validate
   (Python DSL)       (PyPTO)   (Codegen)        (C++)    (Simpler)  (PyTorch)
```

## Prerequisites

### Required Software

- **Python**: Version 3.9 or higher
- **PyPTO**: Installed (`pip install -e .` from project root)
- **Simpler Runtime**: Bundled as a git submodule (`git submodule update --init`)

### Python Dependencies

- `pytest>=7.0.0` - Test runner (in dev dependencies)
- `pytest-forked` - Required for process isolation between tests
- `torch` - Reference computations

### Hardware Requirements

- **Simulation Mode** (default): No special hardware required
- **Hardware Mode**: Requires NPU device (e.g., Ascend AI Processor)

## Running Tests

**Important:** The `--forked` flag is required for running system tests.

### Basic Test Execution

Navigate to the PyPTO project root and run tests:

```bash
# Navigate to PyPTO project directory
cd /path/to/pypto-github

# Run all system tests (simulation mode by default)
pytest tests/st/ -v --forked

# Run specific test file
pytest tests/st/runtime/ops/test_matmul.py -v --forked

# Run specific test class
pytest tests/st/runtime/ops/test_matmul.py::TestMatmulOperations -v --forked

# Run specific test method
pytest tests/st/runtime/ops/test_matmul.py::TestMatmulOperations::test_matmul_shapes -v --forked
```

### Platform Selection

Every test that takes the `test_runner` fixture runs on the platforms named
by `--platform`, without declaring anything: the platform matrix expands the
test over the active allowlist and binds each variant's platform to the case
it builds. `--platform` accepts a comma-separated subset of `a2a3`, `a5`,
`a2a3sim`, `a5sim` and defaults to `a2a3` (matching legacy on-NPU CI).

A test that reaches the runner through a module- or session-scoped fixture is
not expanded — that fixture is built once and shared by every item in its
scope, so it cannot hold one artefact per platform. Such a test runs on the
first `--platform` id its `@pytest.mark.platforms` marker allows.

A test taking `backend_type` (there is no fixture of that name — the matrix
supplies it) gets it paired with the platform in a single parametrize, so the
backend a case compiles for and the toolchain it runs on can never disagree.

A single active platform is *not* parametrized, so `--platform=a2a3` keeps the
plain node ids; naming two or more grows the familiar `[a2a3]` / `[a5]`
suffixes.

Keep `--forked` when a run spans both architectures: a single process that
executes on `a2a3*` and `a5*` simulators through the pre-compile pipeline
segfaults while cleaning up at session end (pre-existing; every test still
passes first).

```bash
# Default: run a2a3 only (matches legacy CI; requires Ascend 910B hardware)
pytest tests/st/ -v --forked

# Run only the a2a3 simulator
pytest tests/st/ -v --forked --platform=a2a3sim

# Run only the Ascend 950 simulator
pytest tests/st/ -v --forked --platform=a5sim

# Run on real Ascend 910B hardware (requires NPU device)
pytest tests/st/ -v --forked --platform=a2a3 --device=0

# Run on real Ascend 950 hardware
pytest tests/st/ -v --forked --platform=a5 --device=0

# Run on multiple platforms in a single invocation
pytest tests/st/ -v --forked --platform=a2a3sim,a5sim,a2a3
```

A test case can additionally restrict itself to a subset of platforms via the
``@pytest.mark.platforms(...)`` marker, e.g. ``@pytest.mark.platforms("a5",
"a5sim", reason="...")`` to mark a test as Ascend 950 only. The intersection of
the CLI filter and the per-test whitelist determines which variants actually
run. Always pass `reason=` — it is the only record of *why* the case is
restricted. An unknown platform id in either marker fails collection instead of
quietly deselecting the test everywhere.

#### Expected failures on one platform

A case that runs everywhere but is known to fail on one platform belongs in
`@pytest.mark.platform_xfail(...)`, not in a CI `-k` exclusion:

```python
@pytest.mark.platform_xfail(
    "a5",
    reason="950 board: manual pl.tpush_to_aic gets no V->C fractal adapter",
)
def test_multiple_pipes(self, test_runner):
    ...
```

The case still runs on every platform, so the day the underlying fix lands the
run reports an XPASS (the marker is strict by default) instead of the verdict
staying frozen. Pass `strict=False` only for a genuinely flaky failure. This is
what keeps a guard job's command a plain directory: a newly-evaluated failure
gets a marker with a reason next to the test, rather than a name in a `-k`
expression nobody can trace back.

`platforms` vs `platform_xfail`: use `platforms` when the platform *cannot*
run the case (the feature does not exist on that arch) and `platform_xfail`
when it *ought to* and does not yet.

#### Cases that pin an architecture

A `PTOTestCase` subclass that redefines `get_backend_type()` pins its backend,
while the toolchain that assembles and executes the artefact follows the
platform — so such a case cannot honour a platform of the other architecture.
The harness detects this before compiling and **skips** the case with a reason
naming the conflict, instead of compiling for one architecture and executing on
the other. Make the case arch-agnostic by dropping the override (the platform
then decides the backend), or state the limitation with
`@pytest.mark.platforms(...)` so the variant is never generated. The legacy
`PTOTestCase(backend_type=...)` constructor argument pins the backend the same
way and is treated the same.

### Verbose Output

Control output verbosity for debugging:

```bash
# Standard verbose mode
pytest tests/st/ -v --forked

# Extra verbose mode (shows test function docstrings)
pytest tests/st/ -vv --forked

# Show print statements and logging
pytest tests/st/ -v -s --forked

# Show full diff for assertion failures
pytest tests/st/ -vv --tb=long --forked
```

### Filtering Tests

Use pytest's built-in filtering capabilities:

```bash
# Run tests matching keyword
pytest tests/st/ -v --forked -k "matmul"

# Run tests NOT matching keyword
pytest tests/st/ -v --forked -k "not matmul"

# Run tests with specific marker
pytest tests/st/ -v --forked -m "slow"

# Filter by parametrized platform id (works because variants are
# named after the platform, e.g. ``test_foo[a5sim]``)
pytest tests/st/ -v --forked -k "a5sim"
```

## Test Configuration Options

The test framework provides extensive configuration through pytest command-line options.

### Available Options

| Option | Default | Description |
| ------ | ------- | ----------- |
| `--platform` | `a2a3` | Comma-separated allowlist of target platforms. Each runtime test case is parametrized over `a2a3`, `a5`, `a2a3sim`, `a5sim`; only variants whose id appears here run. |
| `--device` | `0` | Device ID for hardware tests (0, 1, 2, ...) |
| `--strategy` | `Default` | PyPTO optimization strategy (`Default` is the only supported value) |
| `--save-kernels` | `False` | Save generated kernels and artifacts to disk |
| `--kernels-dir` | `build_output/{testName}_{timestamp}/` | Custom output directory for saved kernels |
| `--dump-passes` | `False` | Dump intermediate IR after each compiler pass |
| `--codegen-only` | `False` | Only generate code, skip runtime execution |

### Usage Examples

```bash
# Run hardware tests on device 1
pytest tests/st/ -v --forked --platform=a2a3 --device=1

# Save generated kernels for inspection
pytest tests/st/ -v --forked --save-kernels

# Save kernels to custom directory
pytest tests/st/ -v --forked --save-kernels --kernels-dir ./my_test_outputs

# Enable compiler pass dumps for debugging
pytest tests/st/ -v --forked --save-kernels --dump-passes

# Generate code without running (for code inspection)
pytest tests/st/ -v --forked --codegen-only --save-kernels

# Combine multiple options
pytest tests/st/ -v --forked --platform=a2a3sim --save-kernels --dump-passes
```

## Advanced Usage

### Saving Generated Code

By default, generated kernels are stored in temporary directories and cleaned up after tests. Use `--save-kernels` to persist them:

```bash
# Save to default location: build_output/{testName}_{timestamp}/
pytest tests/st/ -v --forked --save-kernels

# Save to custom directory
pytest tests/st/ -v --forked --save-kernels --kernels-dir ./test_artifacts

# Run single test and save outputs
pytest tests/st/runtime/ops/test_matmul.py::TestMatmulOperations::test_matmul_shapes -v --forked --save-kernels
```

**Output Structure:**

Each test gets its own timestamped directory under `build_output/`:

```text
build_output/
├── matmul_64x64_20260205_143022/
│   ├── kernels/
│   │   ├── aiv/
│   │   │   └── matmul.cpp          # Generated kernel code
│   │   ├── orchestration/
│   │   │   └── orch.cpp            # Orchestration skeleton
│   │   ├── kernel_config.py        # Simpler runtime configuration
│   │   └── golden.py               # PyTorch reference computation
│   └── pass_dump/                  # (if --dump-passes enabled)
│       ├── 001_initial.mlir
│       ├── 002_after_pass_x.mlir
│       └── ...
├── matmul_128x128_20260205_143023/
│   └── ...
└── tile_add_64x64_20260205_143024/
    └── ...
```

### Debugging with Pass Dumps

Dump intermediate IR representations after each compiler pass to debug transformations:

```bash
# Enable IR pass dumps
pytest tests/st/ -v --forked --save-kernels --dump-passes

# The pass_dump/ directory will contain IR snapshots at each optimization stage
# Files are numbered sequentially: 001_initial.mlir, 002_after_pass_x.mlir, etc.
```

This is useful for:

- Understanding how optimization passes transform your program
- Debugging unexpected codegen results
- Learning the PyPTO compilation pipeline
- Reporting compiler bugs with IR snapshots

### Code Generation Only

Generate code without executing on the runtime:

```bash
# Generate kernels without running
pytest tests/st/ -v --forked --codegen-only --save-kernels

# Useful for:
# - Validating code generation without hardware/simulator
# - Inspecting generated C++ kernel code
# - Manual orchestration development
# - CI/CD pipelines that only test compilation
```

### Using Optimization Strategies

`OptimizationStrategy.Default` is currently the only strategy, so `--strategy`
needs no explicit value:

```bash
pytest tests/st/ -v --forked --save-kernels --dump-passes
```

A test case can pin its strategy by implementing `get_strategy()`; the hook
exists so a future strategy can be opted into per test without touching the
harness.

### Parameterized Testing

Run tests with multiple configurations:

```bash
# The conftest.py defines standard test shapes
# Tests using the tensor_shape fixture will run with: (64,64), (128,128), (256,256)

# Run all shape variations
pytest tests/st/ -v --forked

# Filter to specific parameter
pytest tests/st/ -v --forked -k "64"
```

## Writing New Tests

### Test Structure

System tests inherit from `PTOTestCase` and implement required methods. See the example below:

```python
"""
Test file: tests/st/runtime/ops/test_my_operation.py
"""
from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.runtime.runner import RunConfig


class MyOperationTestCase(PTOTestCase):
    """Add two FP32 tensors element-wise."""

    __test__ = False

    def __init__(self, rows: int = 64, cols: int = 64, config: RunConfig | None = None):
        super().__init__(config)
        self.rows = rows
        self.cols = cols

    def get_name(self) -> str:
        return f"my_operation_{self.rows}x{self.cols}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("input_a", [self.rows, self.cols], DataType.FP32, init_value=2.0),
            TensorSpec("input_b", [self.rows, self.cols], DataType.FP32, init_value=3.0),
            TensorSpec("output", [self.rows, self.cols], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        rows = self.rows
        cols = self.cols

        @pl.program
        class MyOperationProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def my_kernel(
                self,
                a: pl.Tensor[[rows, cols], pl.FP32],
                b: pl.Tensor[[rows, cols], pl.FP32],
                c: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
            ) -> pl.Tensor[[rows, cols], pl.FP32]:
                a_tile = pl.load(a, [0, 0], [rows, cols], target_memory=pl.MemorySpace.Vec)
                b_tile = pl.load(b, [0, 0], [rows, cols])
                result = pl.add(a_tile, b_tile)
                return pl.store(result, [0, 0], c)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[rows, cols], pl.FP32],
                b: pl.Tensor[[rows, cols], pl.FP32],
            ) -> pl.Tensor[[rows, cols], pl.FP32]:
                c: pl.Tensor[[rows, cols], pl.FP32] = pl.create_tensor([rows, cols], dtype=pl.FP32)
                c = self.my_kernel(a, b, c)
                return c

        return MyOperationProgram

    def compute_expected(self, tensors, params=None):
        tensors["output"][:] = tensors["input_a"] + tensors["input_b"]


class TestMyOperationSuite:
    """Pytest test suite."""

    @pytest.mark.parametrize("rows,cols", [(64, 64), (128, 128)])
    def test_my_operation_shapes(self, test_runner, rows, cols):
        """Test my operation with various shapes."""
        test_case = MyOperationTestCase(rows=rows, cols=cols)
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for {rows}x{cols}: {result.error}"
```

### Tensor Initialization Patterns

`TensorSpec` supports flexible initialization:

```python
# Scalar initialization (broadcast to all elements)
TensorSpec("a", [128, 128], DataType.FP32, init_value=1.0)

# Torch tensor initialization
TensorSpec("b", [4, 4], DataType.FP32, init_value=torch.eye(4))

# Callable initialization (for random data)
TensorSpec("c", [256, 256], DataType.FP32,
           init_value=lambda shape: torch.randn(shape))

# Zero initialization (default for outputs)
TensorSpec("output", [128, 128], DataType.FP32, is_output=True)
```

### Existing Test Examples

Refer to existing tests for more examples:

- **Matrix Multiplication**: [`tests/st/runtime/ops/test_matmul.py`](runtime/ops/test_matmul.py)
  - Demonstrates matmul operation with L0A/L0B/L0C memory levels
  - Shows parameterized testing with pytest

### Test Fixtures

The [`conftest.py`](conftest.py) provides useful fixtures:

- `test_config`: Session-scoped `RunConfig` built from CLI options
- `test_runner`: `TestRunner` bound to this item's platform (the compile cache it consults is process-wide, so per-item instances still share compiled binaries)
- `optimization_strategy`: Current optimization strategy string from `--strategy`
- `tensor_shape`: Parameterized fixture yielding standard shapes `(64,64)`, `(128,128)`, `(256,256)`

### Custom Markers

Use pytest markers to categorize or restrict tests:

```python
# Restrict a test (or a whole class) to a subset of platforms.  The
# intersection with the --platform CLI filter decides which variants run.
@pytest.mark.platforms("a5", "a5sim", reason="uses an Ascend 950 only operand")
def test_ascend950_specific(test_runner):
    ...

# Expect a failure on one platform without dropping the case from the run.
@pytest.mark.platform_xfail("a5", reason="950 layout mismatch, see #1234")
def test_runs_everywhere_fails_on_950(test_runner):
    ...

@pytest.mark.slow  # Long-running test
def test_large_model(test_runner):
    ...
```

Declaring ``platform`` yourself is only needed when the **test body** uses the
value — to vary a dtype, a tolerance, or which case class to build. Parametrize
it with the canonical ``PLATFORMS`` list (or a narrower one) and forward it to
your ``PTOTestCase`` subclass; an explicit ``platform`` parametrize always wins
over the automatic expansion:

```python
from harness.core.harness import PLATFORMS

class TestFoo:
    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_foo(self, test_runner, platform):
        result = test_runner.run(FooTestCase(platform=platform))
        assert result.passed
```

### Test Framework Package

The testing framework lives at `tests/st/harness/`:

- `core/` — Core infrastructure: `harness.py` (base classes), `test_runner.py` (execution pipeline), `environment.py` (Simpler path setup)
- `adapters/` — Low-level adapters bridging PyPTO compilation to Simpler's CodeRunner

### Test Organization

Tests are organized by execution mode:

- `runtime/` - Tests that execute on hardware or simulator
  - Each test case is parametrized over `a2a3`, `a5`, `a2a3sim`, `a5sim`
  - Tests automatically skip when the requested platform set is onboard-only
    (`a2a3` and/or `a5`) but no NPU device nodes are present
- `codegen/` - Tests that only verify code generation
  - Automatically uses --codegen-only mode
  - Does not require Simpler runtime

## Troubleshooting

### Common Issues

#### Tests Fail or Hang Without --forked

**Problem:** Tests fail with unexpected errors, hang, or produce incorrect results when run without `--forked`.

**Solution:**

```bash
# Always use --forked to run each test in a separate process
pytest tests/st/ -v --forked

# Install pytest-forked if not available
pip install pytest-forked
```

#### ModuleNotFoundError: No module named 'pypto'

**Problem:** PyPTO is not in the Python path.

**Solution:**

```bash
# Install PyPTO in editable mode
cd /path/to/pypto-github
pip install -e .
```

#### ModuleNotFoundError: No module named 'harness'

**Problem:** The internal test package is not in the Python path.

**Solution:** Tests must be run from the project root with pytest:

```bash
cd /path/to/pypto-github
pytest tests/st/ -v --forked
```

The `conftest.py` automatically adds `tests/st/` to the Python path.

#### ModuleNotFoundError: No module named 'code_runner'

**Problem:** Simpler submodule is not checked out.

**Solution:** Initialize the git submodule:

```bash
git submodule update --init
pip install -v ./simpler
```

#### Fixtures Not Found

**Problem:** pytest can't find `test_runner` or other fixtures.

**Solutions:**

```bash
# Run from project root directory
cd /path/to/pypto-github
pytest tests/st/ -v --forked

# Check pytest discovers conftest.py
pytest tests/st/ -v --forked --collect-only
```

#### Hardware Tests Skipped

**Problem:** Runtime tests are auto-skipped because the requested platform
set only contains onboard platforms (`a2a3`, `a5`) and no NPU device nodes
were detected.

**Solution:**

```bash
# Either provide real hardware and re-run with the onboard platform...
pytest tests/st/ -v --forked --platform=a2a3 --device=0

# ...or include a simulator platform in the filter to run those variants.
pytest tests/st/ -v --forked --platform=a2a3sim
```

### Verification Checklist

Before running tests, verify your setup:

- [ ] PyPTO installed: `python -c "import pypto"`
- [ ] pytest-forked installed: `pip install pytest-forked`
- [ ] In correct directory: `pwd` shows PyPTO project root
- [ ] conftest.py exists: `ls tests/st/conftest.py`
- [ ] harness package exists: `ls tests/st/harness/`
- [ ] Simpler submodule checked out: `ls runtime/`

---

For questions or contributions, please refer to the main [PyPTO README](../../README.md).
