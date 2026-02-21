# PyPTO Fuzzing Test Framework

Automated fuzzing framework for generating and validating multi-kernel test cases for PyPTO IR.

## Overview

This framework tests PyPTO compiler and runtime correctness by randomly generating operator combinations:

- **Operator Fuzzing**: Random combinations of block-level operators (binary, unary, scalar, reduction, expand, etc.)
- **Multi-Kernel Generation**: Auto-generates test cases with multiple InCore kernels and Orchestration functions
- **Golden Reference**: Uses NumPy to generate expected results, validated through harness framework
- **Shape Tracking**: Supports dynamic shape inference and memory alignment checks

## Directory Structure

```
tests/st/fuzz/
├── src/                          # Core generators
│   ├── fuzzer.py                 # Operator fuzzing engine
│   ├── kernel_generator.py       # InCore kernel code generation
│   ├── orchestrator_generator.py # Orchestration function generation
│   └── multi_kernel_test_generator.py  # Multi-kernel test case generation
├── generated/                    # Generated test files
│   └── test_fuzz_multi_kernel.py
├── example_mutil_kernel.py       # Usage example
└── README.md                     # This document
```

## Quick Start

### Generate Test Cases

Use the example script to generate tests:

```bash
# Generate tests with default configuration
python tests/st/fuzz/example_mutil_kernel.py

# Specify configuration index
python tests/st/fuzz/example_mutil_kernel.py --config-index 0

# Custom parameters
python tests/st/fuzz/example_mutil_kernel.py \
    --atol 1e-4 \
    --rtol 1e-4 \
    --advanced-ops-prob 0.7 \
    --output tests/st/fuzz/generated/my_test.py
```

### Run Tests

```bash
# Run generated tests
pytest tests/st/fuzz/generated/test_fuzz_multi_kernel.py

# Run all fuzz tests
pytest tests/st/fuzz/
```

## Core Components

### OpFuzzer ([fuzzer.py](src/fuzzer.py))

Operator fuzzing engine that:
- Randomly generates operator chains (ensures all inputs and intermediates are used)
- Checks shape compatibility and performs automatic inference
- Tracks value ranges (avoids illegal inputs for sqrt/log/div operators)
- Supports basic and advanced operators (row_expand, col_expand, reduction, etc.)

### KernelGenerator ([kernel_generator.py](src/kernel_generator.py))

Generates Python code for InCore kernels, including:
- Tile allocation and memory management
- Operator call sequences
- Golden reference implementation (NumPy)

### OrchestratorGenerator ([orchestrator_generator.py](src/orchestrator_generator.py))

Generates Orchestration functions, supporting:
- Sequential: Execute multiple kernels sequentially
- Parallel: Execute independent kernels in parallel
- Pipeline: Execute kernels in pipeline mode

### MultiKernelTestGenerator ([multi_kernel_test_generator.py](src/multi_kernel_test_generator.py))

Top-level test case generator that integrates the above components to generate complete pytest test classes.

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--config-index` | Configuration index (starting from 0) | 0 |
| `--output` | Output file path | `generated/test_fuzz_multi_kernel.py` |
| `--atol` | Absolute error tolerance | 5e-5 |
| `--rtol` | Relative error tolerance | 5e-5 |
| `--advanced-ops-prob` | Advanced operator selection probability (0.0-1.0) | 0.5 |

## Test Configuration Example

Define test configurations in [example_mutil_kernel.py](example_mutil_kernel.py):

```python
all_configs = [
    {
        "name": "fuzz_sequential_simple",
        "num_instances": 1,           # Generate 1 test instance
        "seed": 40,                   # Random seed
        "enable_advanced_ops": True,  # Enable advanced operators
        "num_kernels": 1,             # Number of kernels
        "mode": "sequential",         # Execution mode
        "shape": (128, 128),          # Default shape
        "num_ops_range": (5, 5),      # Operator count range per kernel
        "tensor_init_type": "random", # Tensor initialization method
        "input_shapes_list": [
            [(128, 128), (128, 128)], # kernel_0: 2 inputs with same dimensions
        ],
        "description": "Simple sequential execution: 1 kernel, same dimension inputs",
    },
]
```

## Notes

- Generated test files will overwrite existing files
- Advanced operators (row_expand, reduction, etc.) require specific shape constraints
- Start with small-scale configurations for validation before expanding to complex scenarios
- Check `run.log` for detailed information when tests fail
