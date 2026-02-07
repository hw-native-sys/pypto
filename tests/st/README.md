# PyPTO System Tests

This directory contains system-level integration tests for PyPTO using the [pto-testing-framework](https://github.com/hw-native-sys/pto-testing-framework). These tests validate the complete compilation and execution pipeline from PyPTO DSL programs to executable code on target platforms.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Installing the Testing Framework](#installing-the-testing-framework)
- [Running Tests](#running-tests)
- [Test Configuration Options](#test-configuration-options)
- [Advanced Usage](#advanced-usage)
- [Writing New Tests](#writing-new-tests)
- [Troubleshooting](#troubleshooting)

## Overview

System tests use the `pto-testing-framework` to perform end-to-end validation of PyPTO programs:

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

Before running system tests, ensure you have the following installed:

### Required Software

- **Python**: Version 3.9 or higher
- **CMake**: Version 3.15 or higher
- **C++ Compiler**: Supporting C++17 standard (GCC, Clang, or MSVC)
- **Git**: For cloning dependencies

### Python Dependencies

The following Python packages will be automatically installed by the testing framework:

- `pytest>=7.0.0` - Test runner
- `numpy` - Reference computations and data generation
- `nanobind>=2.0.0` - Python-C++ bindings
- `scikit-build-core>=0.10.0` - Build system

### Hardware Requirements

- **Simulation Mode** (default): No special hardware required
- **Hardware Mode**: Requires NPU device (e.g., Ascend AI Processor)

## Installing the Testing Framework

### Option 1: Fresh Installation (Recommended)

Clone and build the testing framework with automatic dependency management:

```bash
# Clone the testing framework
git https://github.com/luohuan19/PTOTestingFramework.git
cd pto-testing-framework

# Build with automatic dependency detection
# This will auto-detect PyPTO and Simpler, or clone them if not found
./build_and_install.sh --clean --install

# The script will:
# 1. Detect or clone PyPTO from PYPTO_ROOT or GitHub
# 2. Detect or clone Simpler from SIMPLER_ROOT or GitHub
# 3. Build PyPTO with CMake
# 4. Build framework components
# 5. Generate build/setup_env.sh for environment setup
# 6. Optionally install pto-test package in editable mode (with --install)
```

### Option 2: Using Existing PyPTO Installation

If you already have PyPTO installed or want to test against your local development version:

```bash
# Option 2a: Set environment variable
export PYPTO_ROOT=/path/to/your/pypto
cd pto-testing-framework
./build_and_install.sh

# Option 2b: Use pip editable install (auto-detected)
cd /path/to/pypto
pip install -e .
cd /path/to/pto-testing-framework
./build_and_install.sh

# Option 2c: Place in 3rdparty/ (auto-detected)
mkdir -p pto-testing-framework/3rdparty
ln -s /path/to/pypto pto-testing-framework/3rdparty/pypto
cd pto-testing-framework
./build_and_install.sh
```

### Build Script Options

The `build_and_install.sh` script supports various options:

**Build Configuration:**
```bash
# Clean build (recommended for first-time setup)
./build_and_install.sh --clean

# Debug build for development
./build_and_install.sh --type Debug

# Release build for performance testing
./build_and_install.sh --type Release

# Install pto-test package in editable mode
./build_and_install.sh --install

# Parallel build with custom job count
./build_and_install.sh --jobs 8

# Combined options
./build_and_install.sh --clean --type Release --install --jobs 8
```

**Dependency Configuration:**
```bash
# Use custom repository URLs
./build_and_install.sh --pypto-repo https://your-mirror.com/pypto

# Use specific branch or tag
./build_and_install.sh --pypto-branch develop

# Combine repository and branch options
./build_and_install.sh --pypto-repo URL --pypto-branch feature-xyz
```

**Help:**
```bash
# Show all available options
./build_and_install.sh --help
```

### Install PyPTO for Test Development

For test development in this repository, install PyPTO in editable mode:

```bash
# Navigate to PyPTO project root (this repository)
cd /path/to/pypto-github

# Install in editable mode with development dependencies
pip install -e ".[dev]"

# This allows you to:
# - Modify PyPTO source code and see changes immediately
# - Import pypto.language in your test cases
# - Use PyPTO's optimization passes and code generation
```

### Post-Installation

After building the framework, always source the environment for new terminal sessions:

```bash
# Every time you open a new terminal
cd /path/to/pto-testing-framework
source build/setup_env.sh

# Or add to your shell profile (~/.bashrc, ~/.zshrc)
echo "source /path/to/pto-testing-framework/build/setup_env.sh" >> ~/.bashrc
```

## Running Tests

### Basic Test Execution

Navigate to the PyPTO project root and run tests:

```bash
# Navigate to PyPTO project directory
cd /path/to/pypto-github

# Run all system tests (simulation mode by default)
pytest tests/st/ -v

# Run specific test file
pytest tests/st/test_case/test_matmul.py -v

# Run specific test class
pytest tests/st/test_case/test_matmul.py::TestMatmulOperations -v

# Run specific test method
pytest tests/st/test_case/test_matmul.py::TestMatmulOperations::test_matmul_shapes -v
```

### Platform Selection

Tests can run on simulation or hardware platforms:

```bash
# Run on simulator (default, no hardware required)
pytest tests/st/ -v --platform=a2a3sim

# Run on real hardware (requires NPU device)
pytest tests/st/ -v --platform=a2a3 --device=0

# Specify different device ID
pytest tests/st/ -v --platform=a2a3 --device=1
```

### Verbose Output

Control output verbosity for debugging:

```bash
# Standard verbose mode
pytest tests/st/ -v

# Extra verbose mode (shows test function docstrings)
pytest tests/st/ -vv

# Show print statements and logging
pytest tests/st/ -v -s

# Show full diff for assertion failures
pytest tests/st/ -vv --tb=long
```

### Filtering Tests

Use pytest's built-in filtering capabilities:

```bash
# Run tests matching keyword
pytest tests/st/ -v -k "matmul"

# Run tests NOT matching keyword
pytest tests/st/ -v -k "not matmul"

# Run tests with specific marker
pytest tests/st/ -v -m "slow"

# Skip tests with specific marker
pytest tests/st/ -v -m "not hardware"
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
| `--kernels-dir` | `build/outputs/output_{timestamp}/` | Custom output directory for saved kernels |
| `--dump-passes` | `False` | Dump intermediate IR after each compiler pass |
| `--codegen-only` | `False` | Only generate code, skip runtime execution |
| `--fuzz-count` | `10` | Number of fuzz test iterations (for future use) |
| `--fuzz-seed` | `random` | Random seed for fuzz tests (for future use) |

### Usage Examples

```bash
# Test with PTOAS optimization strategy
pytest tests/st/ -v --strategy=PTOAS

# Run hardware tests on device 1
pytest tests/st/ -v --platform=a2a3 --device=1

# Save generated kernels for inspection
pytest tests/st/ -v --save-kernels

# Save kernels to custom directory
pytest tests/st/ -v --save-kernels --kernels-dir ./my_test_outputs

# Enable compiler pass dumps for debugging
pytest tests/st/ -v --save-kernels --dump-passes

# Generate code without running (for code inspection)
pytest tests/st/ -v --codegen-only --save-kernels

# Combine multiple options
pytest tests/st/ -v --platform=a2a3sim --strategy=PTOAS --save-kernels --dump-passes
```

## Advanced Usage

### Saving Generated Code

By default, generated kernels are stored in temporary directories and cleaned up after tests. Use `--save-kernels` to persist them:

```bash
# Save to default location: build/outputs/output_{timestamp}/
pytest tests/st/ -v --save-kernels

# Save to custom directory
pytest tests/st/ -v --save-kernels --kernels-dir ./test_artifacts

# Run single test and save outputs
pytest tests/st/test_case/test_matmul.py::TestMatmulOperations::test_matmul_shapes -v --save-kernels
```

**Output Structure:**
```
build/outputs/output_20260205_143022/
└── matmul_64x64/
    ├── kernels/
    │   ├── aiv/
    │   │   └── matmul.cpp          # Generated kernel code
    │   ├── orchestration/
    │   │   └── orch.cpp            # Orchestration skeleton
    │   ├── kernel_config.py        # Simpler runtime configuration
    │   └── golden.py               # NumPy reference computation
    ├── pass_dump/                  # (if --dump-passes enabled)
    │   ├── 001_initial.mlir
    │   ├── 002_after_pass_x.mlir
    │   └── ...
    └── metadata.json               # Test metadata
```

### Debugging with Pass Dumps

Dump intermediate IR representations after each compiler pass to debug transformations:

```bash
# Enable IR pass dumps
pytest tests/st/ -v --save-kernels --dump-passes

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
pytest tests/st/ -v --codegen-only --save-kernels

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
pytest tests/st/ -v --strategy=Default

# Use PTOAS (PTO Accelerator Strategy) optimization
pytest tests/st/ -v --strategy=PTOAS

# Combine with other options
pytest tests/st/ -v --strategy=PTOAS --save-kernels --dump-passes
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
pytest tests/st/ -v

# Filter to specific parameter
pytest tests/st/ -v -k "64"
```

## Writing New Tests

### Test Structure

System tests inherit from `PTOTestCase` and implement required methods. See the example below:

```python
"""
Test file: tests/st/test_case/test_my_operation.py
"""
from typing import Any, List
import numpy as np
import pytest
from pto_test.core.test_case import DataType, PTOTestCase, TensorSpec


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
                tile_a = pl.op.block.load(a, 0, 0, 64, 64, target_memory=2)
                tile_b = pl.op.block.load(b, 0, 0, 64, 64, target_memory=2)

                # Perform operation (example: element-wise add)
                tile_c = pl.op.block.add(tile_a, tile_b)

                # Store result back to global memory
                out = pl.op.block.store(tile_c, 0, 0, 64, 64, c)
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
        """Compute expected results using NumPy."""
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

# NumPy array initialization
TensorSpec("b", [4, 4], DataType.FP32, init_value=np.eye(4))

# Callable initialization (for random data)
TensorSpec("c", [256, 256], DataType.FP32,
           init_value=lambda: np.random.randn(256, 256))

# Zero initialization (default for outputs)
TensorSpec("output", [128, 128], DataType.FP32, is_output=True)
```

### Existing Test Examples

Refer to existing tests for more examples:

- **Matrix Multiplication**: [`tests/st/test_case/test_matmul.py`](test_case/test_matmul.py)
  - Demonstrates matmul operation with L0A/L0B/L0C memory levels
  - Shows parameterized testing with pytest

### Test Fixtures

The [`conftest.py`](conftest.py) provides useful fixtures:

- `test_config`: Session-scoped configuration from CLI options
- `test_runner`: Session-scoped test runner (reused across tests)
- `optimization_strategy`: Current optimization strategy
- `tensor_shape`: Parameterized fixture for standard shapes
- `fuzz_count`, `fuzz_seed`: For fuzz testing (future use)

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

### Further Documentation

For detailed information about the testing framework API:
- [pto-testing-framework README](https://github.com/hw-native-sys/pto-testing-framework/blob/main/README.md)
- Framework source: `/path/to/pto-testing-framework/src/pto_test/`

## Troubleshooting

### Common Issues

#### ModuleNotFoundError: No module named 'pypto'

**Problem:** PyPTO is not in the Python path.

**Solutions:**
```bash
# Solution 1: Source the environment setup
cd /path/to/pto-testing-framework
source build/setup_env.sh

# Solution 2: Reinstall PyPTO in editable mode
cd /path/to/pypto-github
pip install -e .

# Solution 3: Set PYTHONPATH manually
export PYTHONPATH="/path/to/pypto/python:${PYTHONPATH}"
```

#### ModuleNotFoundError: No module named 'pto_test'

**Problem:** Testing framework is not in the Python path.

**Solutions:**
```bash
# Solution 1: Rebuild with --install flag
cd /path/to/pto-testing-framework
./build_and_install.sh --install

# Solution 2: Source the environment setup
source /path/to/pto-testing-framework/build/setup_env.sh

# Solution 3: Add to PYTHONPATH manually
export PYTHONPATH="/path/to/pto-testing-framework/src:${PYTHONPATH}"
```

#### ModuleNotFoundError: No module named 'pto_compiler'

**Problem:** Simpler runtime is not in the Python path.

**Solutions:**
```bash
# Solution 1: Rebuild testing framework (will auto-detect/clone Simpler)
cd /path/to/pto-testing-framework
./build_and_install.sh --clean

# Solution 2: Set SIMPLER_ROOT environment variable
export SIMPLER_ROOT=/path/to/simpler
source /path/to/pto-testing-framework/build/setup_env.sh

# Solution 3: Add to PYTHONPATH manually
export PYTHONPATH="/path/to/simpler/python:${PYTHONPATH}"
```

#### Fixtures Not Found

**Problem:** pytest can't find `test_runner` or other fixtures.

**Solutions:**
```bash
# Ensure conftest.py exists and is readable
ls tests/st/conftest.py

# Run from project root directory
cd /path/to/pypto-github
pytest tests/st/ -v

# Check pytest discovers conftest.py
pytest tests/st/ -v --collect-only
```

#### Hardware Tests Skipped

**Problem:** Tests marked with `@pytest.mark.hardware` are automatically skipped.

**Reason:** Hardware tests only run with `--platform=a2a3`.

**Solution:**
```bash
# Run hardware tests on device
pytest tests/st/ -v --platform=a2a3 --device=0
```

#### Build Failures in pto-testing-framework

**Problem:** CMake configuration errors or compilation failures.

**Solutions:**
```bash
# Clean build from scratch
cd /path/to/pto-testing-framework
./build_and_install.sh --clean

# Check Python version (3.9+ required)
python3 --version

# Install required build tools
pip install nanobind scikit-build-core

# Check CMake version (3.15+ required)
cmake --version
```

#### Import Errors After PyPTO Changes

**Problem:** Modified PyPTO source code not reflected in tests.

**Solutions:**
```bash
# If installed normally, reinstall
cd /path/to/pypto-github
pip install -e .

# If using editable install, rebuild C++ extensions
pip install -e . --force-reinstall --no-deps

# Clear Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
```

#### Tests Fail with Compilation Errors

**Problem:** Generated kernels fail to compile.

**Debugging Steps:**
```bash
# Save kernels for inspection
pytest tests/st/test_case/test_matmul.py -v --save-kernels --codegen-only

# Check generated C++ code
ls build/outputs/output_*/matmul_64x64/kernels/aiv/

# Enable IR dumps to debug compiler passes
pytest tests/st/test_case/test_matmul.py -v --save-kernels --dump-passes

# Check pass_dump/ directory for IR issues
ls build/outputs/output_*/matmul_64x64/pass_dump/
```

#### Environment Variables Not Set

**Problem:** `$PYPTO_ROOT` or other variables are empty after sourcing setup script.

**Solutions:**
```bash
# Check if setup_env.sh exists
ls /path/to/pto-testing-framework/build/setup_env.sh

# If not, rebuild the framework
cd /path/to/pto-testing-framework
./build_and_install.sh

# Source the script in current shell (not in subshell)
source build/setup_env.sh  # Correct
bash build/setup_env.sh    # Wrong - runs in subshell

# Verify variables are set
echo $FRAMEWORK_ROOT
echo $PYPTO_ROOT
```

### Getting Help

If you encounter issues not covered here:

1. **Check Framework Logs**: Look for detailed error messages in pytest output
2. **Save Artifacts**: Use `--save-kernels` and `--dump-passes` to capture debug information
3. **Consult Documentation**: Review [pto-testing-framework README](https://github.com/hw-native-sys/pto-testing-framework)
4. **Report Issues**: File bug reports with saved artifacts and full error logs

### Verification Checklist

Before running tests, verify your setup:

- [ ] PyPTO installed and importable: `python -c "import pypto"`
- [ ] Testing framework installed: `python -c "import pto_test"`
- [ ] Simpler runtime accessible: `python -c "import pto_compiler"`
- [ ] Environment sourced: `echo $FRAMEWORK_ROOT` shows path
- [ ] In correct directory: `pwd` shows PyPTO project root
- [ ] conftest.py exists: `ls tests/st/conftest.py`

---

**Happy Testing!** 🧪

For questions or contributions, please refer to the main [PyPTO README](../../README.md).
