# PyPTO Fuzzing Test Framework

Automated fuzzing framework for generating and validating multi-kernel test cases for PyPTO IR.

## Overview

This framework tests PyPTO compiler and runtime correctness by randomly generating operator combinations:

- **Operator Fuzzing**: Random combinations of block-level operators (binary, unary, reduction, expand, matmul)
- **Control Flow**: Supports `for` loops (tiling and accumulation modes) and `if/else` branching
- **Multi-Kernel Generation**: Auto-generates test cases with multiple InCore kernels and Orchestration functions
- **Golden Reference**: Uses NumPy/PyTorch to generate expected results, validated through harness framework
- **Shape Tracking**: Dynamic shape inference with 32-byte memory alignment checks

## Directory Structure

```text
tests/st/fuzz/
├── src/                               # Core generators
│   ├── core/                          # Foundation layer
│   │   ├── op_specs.py                #   Operator specifications and constraint registry
│   │   ├── shape_utils.py             #   Shape alignment utilities
│   │   └── fuzzer.py                  #   Operator fuzzing engine (OpFuzzer, OpChainConfig)
│   ├── body/                          # Composable body AST layer
│   │   ├── ast.py                     #   BodyNode = OpBlock | ForBlock | IfElseBlock
│   │   ├── generator.py               #   Random body structure with depth control
│   │   ├── codegen.py                 #   Recursive PTO code generation from body AST
│   │   └── golden.py                  #   Recursive Torch golden reference from body AST
│   ├── kernel_generator.py            # InCore kernel code generation
│   ├── golden_generator.py            # PyTorch golden reference generation
│   ├── orchestrator_generator.py      # Orchestration function generation
│   └── multi_kernel_test_generator.py # Top-level test case generator
├── generated/                         # Generated test files
│   └── test_fuzz_multi_kernel.py
├── check_pass_rate.py                 # Parse JUnit XML to compute pass rates
├── generate_test.py                   # Test generation CLI
└── README.md
```

## Quick Start

### Generate Test Cases

```bash
# Generate tests with default configuration
python tests/st/fuzz/generate_test.py

# Specify a reproducible seed
python tests/st/fuzz/generate_test.py --seed 42

# Generate 5 test cases with custom tolerances
python tests/st/fuzz/generate_test.py \
    --num-cases 5 \
    --atol 1e-5 \
    --rtol 1e-5 \
    --advanced-ops-prob 0.7 \
    --output tests/st/fuzz/generated/my_test.py

# Select a specific configuration by index
python tests/st/fuzz/generate_test.py --config-index 0 --num-cases 3
```

### Run Tests

```bash
pytest tests/st/fuzz/generated/test_fuzz_multi_kernel.py
pytest tests/st/fuzz/
```

## Architecture

### Data Flow

```text
OpSpec (core/op_specs.py)     Operator definitions + constraints
    │
    ▼
OpFuzzer (core/fuzzer.py)     Generates random op chains (OpChainConfig → op dicts)
    │
    ▼
BodyGenerator (body/)         Composable body AST with nested control flow
    ├─ ast.py                   BodyNode = OpBlock | ForBlock | IfElseBlock
    ├─ generator.py             Random body structure with depth control
    ├─ codegen.py               Recursive PTO code generation
    └─ golden.py                Recursive Torch golden reference
    │
    ▼
KernelGenerator               Converts op chains → PyPTO kernel source code
  └─ golden_generator.py        Generates PyTorch golden reference
    │
    ▼
OrchestratorGenerator          Wraps kernels into orchestration (sequential/parallel/pipeline)
    │
    ▼
MultiKernelTestGenerator       Assembles complete pytest test file
```

### Key Classes

| Class | File | Purpose |
| ----- | ---- | ------- |
| `OpSpec` | `op_specs.py` | Operator definition: name, input types, constraints, NumPy equivalent, shape transform. Includes eligibility checks (`is_range_eligible`, `is_shape_eligible`, `has_enough_inputs`) |
| `OpChainConfig` | `fuzzer.py` | Configuration dataclass for op-chain generation |
| `OpFuzzer` | `fuzzer.py` | Generates random operator chains with shape tracking, value-range safety, and usage throttling |
| `KernelGenerator` | `kernel_generator.py` | Generates InCore kernel code with tile allocation, for-loops, and if/else branches |
| `OrchestratorGenerator` | `orchestrator_generator.py` | Generates orchestration functions (sequential, parallel, pipeline) |
| `MultiKernelTestGenerator` | `multi_kernel_test_generator.py` | Top-level generator that integrates all components into pytest test classes |

### Operator Categories

| Category | Pipe | Examples | Constraints |
| -------- | ---- | -------- | ----------- |
| Binary | V | `add`, `sub`, `mul`, `div`, `maximum`, `minimum` | `avoid_zero` (div) |
| Unary | V | `sqrt`, `rsqrt`, `exp`, `log`, `abs`, `relu`, `neg`, `recip` | `positive_only`, `avoid_zero` |
| Row Expand | V | `row_expand_sub`, `row_expand_mul`, `row_expand_div` | `row_vec_required` |
| Row Reduction | V | `row_sum`, `row_max`, `row_min` | `produces_row_vec`, `requires_tmp_tile` |
| Col Expand | V | `col_expand_sub`, `col_expand_mul`, `col_expand_div` | `col_vec_required` |
| Col Reduction | V | `col_sum`, `col_max`, `col_min` | `produces_col_vec`, `requires_params` |
| Matrix | M | `matmul` | `requires_memory_management` |

### For-Loop Modes

The framework supports two for-loop modes for generated kernels:

- **Tiling mode** (`use_tiling=True`): The entire kernel body (loads, ops, store) runs inside a `for i in pl.range(N)` loop with `i`-based offsets, processing one tile per iteration. Input/output tensors are scaled by iteration count.
- **Accumulation mode** (`use_tiling=False`): The op chain is split into pre-loop / loop body / post-loop segments. The loop body runs `N` times and accumulates into a pre-loop variable. Input/output tensors keep tile size.

### Extensibility

The framework uses a table-driven and method-delegation design for easy extension:

- **New operator**: Append an `OpSpec` to the appropriate `BLOCK_*_OPS` list in `op_specs.py`
- **New constraint**: Add logic to `OpSpec.is_range_eligible()` or `OpSpec.is_shape_eligible()`
- **New usage limit**: Add an entry to `OpFuzzer._USAGE_LIMITS` / `_USAGE_CAPS` tables
- **New config parameter**: Add a field to `OpChainConfig` dataclass

## Test Configuration

```python
config = {
    "name": "fuzz_sequential",
    "num_instances": 1,
    "seed": None,                       # None = random seed
    "enable_advanced_ops": True,
    "num_kernels": 1,
    "mode": "sequential",               # sequential | parallel | pipeline
    "shape": (64, 64),
    "num_ops_range": (7, 8),
    "tensor_init_type": "random",
    "enable_for_loop": True,            # Enable for-loop generation
    "max_for_loop_iterations": 3,
    "for_loop_probability": 1.0,        # Probability of using for-loop (0.0-1.0)
    "enable_if_else": False,            # Enable if/else branch generation
    "input_shapes_list": [
        [(64, 64), (64, 64)],
    ],
    "description": "2-kernel sequential chain with advanced ops",
}
```

### Config Parameters

| Parameter | Description | Default |
| --------- | ----------- | ------- |
| `mode` | Execution mode: `sequential`, `parallel`, `pipeline` | `sequential` |
| `enable_advanced_ops` | Enable reduction, expand, and matmul operators | `False` |
| `enable_for_loop` | Generate for-loop control flow in kernels | `False` |
| `max_for_loop_iterations` | Upper bound for random iteration count (capped at 4) | `4` |
| `for_loop_probability` | Probability of using for-loop when enabled (0.0-1.0) | `1.0` |
| `enable_if_else` | Generate if/else branching in kernels | `False` |
| `num_kernels` | Number of InCore kernels per test case | `3` |
| `num_ops_range` | (min, max) operators per kernel | `(3, 7)` |
| `tensor_init_type` | Tensor initialization: `random`, `constant`, `range`, `normal` | `constant` |

### CLI Arguments

| Argument | Description | Default |
| -------- | ----------- | ------- |
| `--seed` | Base seed for test generation (random if omitted) | random |
| `--config-index` | Configuration index to use (starting from 0) | `0` |
| `--num-cases` | Number of test cases to generate | per-config |
| `--output` | Output file path | `generated/test_fuzz_multi_kernel.py` |
| `--atol` | Absolute error tolerance | `5e-5` |
| `--rtol` | Relative error tolerance | `5e-5` |
| `--advanced-ops-prob` | Probability of selecting advanced ops (0.0-1.0) | `0.5` |

## Notes

- Generated test files overwrite existing files at the output path
- Advanced operators require specific shape constraints (e.g., row_expand needs `[M, 1]` vectors)
- `enable_if_else` and `enable_for_loop` are mutually exclusive for now
- Start with small-scale configurations for validation before expanding
- Check `run.log` for detailed information when tests fail
- Use `--seed <N>` in CI to reproduce failures
