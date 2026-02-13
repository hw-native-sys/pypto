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
4. **Validation**: Compares runtime results against NumPy reference implementations

**Test Flow:**
```
Test Case Definition → Build IR → Generate Kernels → Compile → Execute → Validate
   (Python DSL)       (PyPTO)   (Codegen)        (C++)    (Simpler)  (NumPy)
```

## Prerequisites

### Required Software

- **Python**: Version 3.9 or higher
- **PyPTO**: Installed (`pip install -e .` from project root)
- **Simpler Runtime**: Set `SIMPLER_ROOT` environment variable

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
pytest tests/st/runtime/test_matmul.py -v --forked

# Run specific test class
pytest tests/st/runtime/test_matmul.py::TestMatmulOperations -v --forked

# Run specific test method
pytest tests/st/runtime/test_matmul.py::TestMatmulOperations::test_matmul_shapes -v --forked
```

### Platform Selection

Tests can run on simulation or hardware platforms:

```bash
# Run on simulator (default, no hardware required)
pytest tests/st/ -v --forked --platform=a2a3sim

# Run on real hardware (requires NPU device)
pytest tests/st/ -v --forked --platform=a2a3 --device=0

# Specify different device ID
pytest tests/st/ -v --forked --platform=a2a3 --device=1
```

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

# Skip tests with specific marker
pytest tests/st/ -v --forked -m "not hardware"
```

## Test Configuration Options

The test framework provides extensive configuration through pytest command-line options.

### Available Options

| Option | Default | Description |
|--------|---------|-------------|
| `--platform` | `a2a3sim` | Target platform: `a2a3sim` (simulator) or `a2a3` (hardware) |
| `--device` | `0` | Device ID for hardware tests (0, 1, 2, ...) |
| `--strategy` | `Default` | PyPTO optimization strategy: `Default` or `PTOAS` |
| `--save-kernels` | `False` | Save generated kernels and artifacts to disk |
| `--kernels-dir` | `build/tests/st/outputs/output_{timestamp}/` | Custom output directory for saved kernels |
| `--dump-passes` | `False` | Dump intermediate IR after each compiler pass |
| `--codegen-only` | `False` | Only generate code, skip runtime execution |

### Usage Examples

```bash
# Test with PTOAS optimization strategy
pytest tests/st/ -v --forked --strategy=PTOAS

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
pytest tests/st/ -v --forked --platform=a2a3sim --strategy=PTOAS --save-kernels --dump-passes
```

## Advanced Usage

### Saving Generated Code

By default, generated kernels are stored in temporary directories and cleaned up after tests. Use `--save-kernels` to persist them:

```bash
# Save to default location: build/tests/st/outputs/output_{timestamp}/
pytest tests/st/ -v --forked --save-kernels

# Save to custom directory
pytest tests/st/ -v --forked --save-kernels --kernels-dir ./test_artifacts

# Run single test and save outputs
pytest tests/st/runtime/test_matmul.py::TestMatmulOperations::test_matmul_shapes -v --forked --save-kernels
```

**Output Structure:**

Test directories are prefixed with sequential numbers (e.g., `001_`, `002_`) to avoid name collisions when multiple tests have similar names:

```
build/tests/st/outputs/output_20260205_143022/
├── 001_matmul_64x64/
│   ├── kernels/
│   │   ├── aiv/
│   │   │   └── matmul.cpp          # Generated kernel code
│   │   ├── orchestration/
│   │   │   └── orch.cpp            # Orchestration skeleton
│   │   ├── kernel_config.py        # Simpler runtime configuration
│   │   └── golden.py               # NumPy reference computation
│   ├── pass_dump/                  # (if --dump-passes enabled)
│   │   ├── 001_initial.mlir
│   │   ├── 002_after_pass_x.mlir
│   │   └── ...
│   └── metadata.json               # Test metadata
├── 002_matmul_128x128/
│   └── ...
└── 003_tile_add_64x64/
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

PyPTO supports different optimization strategies. Select at runtime:

```bash
# Use Default optimization strategy (default)
pytest tests/st/ -v --forked --strategy=Default

# Use PTOAS (PTO Accelerator Strategy) optimization
pytest tests/st/ -v --forked --strategy=PTOAS

# Combine with other options
pytest tests/st/ -v --forked --strategy=PTOAS --save-kernels --dump-passes
```

You can also override the strategy in individual test cases by implementing the `get_strategy()` method:

```python
from pypto.ir.pass_manager import OptimizationStrategy

class MyTest(PTOTestCase):
    def get_strategy(self):
        return OptimizationStrategy.PTOAS
```

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
Test file: tests/st/runtime/test_my_operation.py
"""
from typing import Any, List
import torch
import pytest
from harness.core.harness import DataType, PTOTestCase, TensorSpec


class TestMyOperation(PTOTestCase):
    def __init__(self, rows: int = 64, cols: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.rows = rows
        self.cols = cols

    def get_name(self) -> str:
        """Return a unique test name."""
        return f"my_operation_{self.rows}x{self.cols}"

    def define_tensors(self) -> List[TensorSpec]:
        """Define input and output tensors."""
        return [
            TensorSpec("input_a", [self.rows, self.cols], DataType.FP32, init_value=2.0),
            TensorSpec("input_b", [self.rows, self.cols], DataType.FP32, init_value=3.0),
            TensorSpec("output", [self.rows, self.cols], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        """Define the PyPTO program."""
        import pypto.language as pl

        @pl.program
        class MyOperationProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def my_operation(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
                c: pl.Tensor[[64, 64], pl.FP32],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                # Load data to L1 memory
                tile_a = pl.block.load(a, offsets=[0, 0], shapes=[64, 64], target_memory=2)
                tile_b = pl.block.load(b, offsets=[0, 0], shapes=[64, 64], target_memory=2)

                # Perform operation (example: element-wise add)
                tile_c = pl.block.add(tile_a, tile_b)

                # Store result back to global memory
                out = pl.block.store(tile_c, offsets=[0, 0], shapes=[64, 64], output_tensor=c)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                out = self.my_operation(a, b)
                return out

        return MyOperationProgram

    def compute_expected(self, tensors, params=None):
        """Compute expected results using torch."""
        tensors["output"][:] = tensors["input_a"] + tensors["input_b"]


class TestMyOperationSuite:
    """Pytest test suite."""

    @pytest.mark.parametrize("rows,cols", [(64, 64), (128, 128)])
    def test_my_operation_shapes(self, test_runner, rows, cols):
        """Test my operation with various shapes."""
        test_case = TestMyOperation(rows=rows, cols=cols)
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

- **Matrix Multiplication**: [`tests/st/runtime/test_matmul.py`](runtime/test_matmul.py)
  - Demonstrates matmul operation with L0A/L0B/L0C memory levels
  - Shows parameterized testing with pytest

### Test Fixtures

The [`conftest.py`](conftest.py) provides useful fixtures:

- `test_config`: Session-scoped configuration from CLI options
- `test_runner`: Session-scoped test runner (reused across tests)
- `optimization_strategy`: Current optimization strategy
- `tensor_shape`: Parameterized fixture for standard shapes

### Custom Markers

Use pytest markers to categorize tests:

```python
@pytest.mark.hardware  # Requires --platform=a2a3
def test_hardware_specific(test_runner):
    ...

@pytest.mark.slow  # Long-running test
def test_large_model(test_runner):
    ...
```

### Test Framework Package

The testing framework is located at `tests/st/harness/` with the following structure:

- `core/` - Core test infrastructure (harness, test_runner, environment)
- `adapters/` - Adapters bridging PyPTO to Simpler (program, config, golden generators)

### Test Organization

Tests are organized by execution mode:

- `runtime/` - Tests that execute on hardware or simulator
  - Includes hardware availability detection
  - Automatically skips when hardware unavailable (platform=a2a3)
  - Always runs on simulator (platform=a2a3sim)
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

**Problem:** Simpler runtime is not available.

**Solution:** Set the SIMPLER_ROOT environment variable:
```bash
export SIMPLER_ROOT=/path/to/simpler
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

**Problem:** Tests marked with `@pytest.mark.hardware` are automatically skipped.

**Solution:**
```bash
# Run hardware tests on device
pytest tests/st/ -v --forked --platform=a2a3 --device=0
```

### Verification Checklist

Before running tests, verify your setup:

- [ ] PyPTO installed: `python -c "import pypto"`
- [ ] pytest-forked installed: `pip install pytest-forked`
- [ ] In correct directory: `pwd` shows PyPTO project root
- [ ] conftest.py exists: `ls tests/st/conftest.py`
- [ ] harness package exists: `ls tests/st/harness/`
- [ ] Simpler is set up: `echo $SIMPLER_ROOT`

---

For questions or contributions, please refer to the main [PyPTO README](../../README.md).
