# IR Test Suite Organization

This document describes the organization structure, testing tools, and coding conventions for the PyPTO IR test suite.

## Directory Structure

The test suite is organized by functional domain into the following subdirectories:

```
tests/ut/ir/
├── core/                       # Core IR primitives tests
│   ├── test_span.py            # Span class creation, validation, conversion
│   ├── test_op.py              # Op class registration and usage
│   ├── test_var.py             # Var variable creation and operations
│   └── test_tuple_type.py      # Tuple type
├── expressions/                # Expression nodes tests
│   ├── test_constants.py       # Constant expressions (ConstInt/Float/Bool)
│   ├── test_binary_ops.py      # Binary operations (add, subtract, multiply, etc.)
│   ├── test_comparison_ops.py  # Comparison operations (gt, lt, eq, etc.)
│   └── test_tensor_expr.py     # Tensor expression operations
├── statements/                 # Statement nodes tests
│   ├── test_assign.py          # Assignment statements
│   ├── test_for_stmt.py        # For loop statements
│   ├── test_if_stmt.py         # If conditional statements
│   ├── test_seq_stmts.py       # Sequential statement sequences
│   ├── test_op_stmts.py        # Operation statements
│   └── test_iter_arg.py        # Iteration arguments
├── operators/                  # Operator tests
│   ├── test_op_registry.py     # Operator registration system
│   ├── test_tensor_ops.py      # Tensor operations (create/view/matmul, etc.)
│   ├── test_block_ops.py       # Block-level operations
│   └── test_operator_spans.py  # Operator location information
├── high_level/                 # High-level constructs tests
│   ├── test_function.py        # Function definition and calling
│   ├── test_program.py         # Program-level constructs
│   └── test_builder.py         # IR Builder API
├── memory/                     # Memory-related tests
│   └── test_memref.py          # Memory references, memory spaces, TileView
├── transforms/                 # IR transformation tests
│   ├── test_hash.py            # Structural hash computation
│   ├── test_equality.py        # Structural equality checking
│   └── test_serialization.py   # Serialization and deserialization
└── printing/                   # Output and visualization tests
    └── test_python_printer.py # Python code printing
```

## Test Writing Guidelines

### Test File Naming

- Test file names start with `test_`
- Use lowercase letters and underscores
- File names should describe the functional module being tested
- Examples: `test_span.py`, `test_binary_ops.py`

### Test Class Naming

Use `Test` prefix + descriptive name (PascalCase):

```python
class TestSpan:
    """Tests for Span class."""
    pass

class TestBinaryOperations:
    """Tests for binary arithmetic operations."""
    pass
```

### Test Method Naming

Follow the pattern: `test_<feature>_<condition>_<expected_result>`

```python
def test_span_creation():
    """Test creating a Span with valid coordinates."""
    pass

def test_add_with_same_types():
    """Test Add expression with same operand types."""
    pass

def test_var_immutability_raises_error():
    """Test that modifying Var attributes raises AttributeError."""
    pass
```

### Docstrings

Each test method should have a clear docstring describing the test's purpose:

```python
def test_nested_expression_equality(make_var, make_const_int):
    """Test structural equality of nested expressions.

    Verifies that two independently constructed expressions with the
    same structure are considered structurally equal when auto_mapping
    is enabled.
    """
    # Test code...
```

### Assertion Best Practices

1. **Clear assertion messages**:
   ```python
   assert result == expected, f"Expected {expected}, got {result}"
   ```

2. **One concept per test**:
   ```python
   # Good practice
   def test_add_operands():
       """Test Add expression has correct operands."""
       span = ir.Span.unknown()
       dtype = DataType.INT64
       x = ir.Var("x", ir.ScalarType(dtype), span)
       y = ir.Var("y", ir.ScalarType(dtype), span)
       add_expr = ir.Add(x, y, dtype, span)
       assert add_expr.left.name == "x"
       assert add_expr.right.name == "y"

   # Avoid testing too many unrelated things in one test
   ```

## Test Organization Examples

### Complete Example

```python
# Copyright header...

"""Tests for binary operation expressions."""

from typing import cast

import pytest
from pypto import DataType, ir


class TestArithmeticOps:
    """Tests for arithmetic binary operations."""

    def test_add_creation(self):
        """Test creating an Add expression."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        add_expr = ir.Add(x, y, dtype, span)

        assert cast(ir.Var, add_expr.left).name == "x"
        assert cast(ir.Var, add_expr.right).name == "y"

    def test_add_is_binary_expr(self):
        """Test that Add is an instance of BinaryExpr."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        add_expr = ir.Add(x, y, dtype, span)

        assert isinstance(add_expr, ir.BinaryExpr)
        assert isinstance(add_expr, ir.Expr)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
```

### Parameterized Test Example

```python
@pytest.mark.parametrize("value,dtype", [
    (42, DataType.INT64),
    (10, DataType.INT32),
    (255, DataType.UINT8),
])
def test_const_int_with_types(value, dtype):
    """Test ConstInt with various types."""
    span = ir.Span.unknown()
    const = ir.ConstInt(value, dtype, span)
    assert const.value == value
    assert const.dtype == dtype
```

## Running Tests

### Run All IR Tests

```bash
pytest tests/ut/ir/
```

### Run Tests in Specific Subdirectory

```bash
pytest tests/ut/ir/core/          # Run only core tests
pytest tests/ut/ir/expressions/   # Run only expression tests
pytest tests/ut/ir/operators/     # Run only operator tests
```

### Run Tests in Specific File

```bash
pytest tests/ut/ir/core/test_span.py
```

### Run Specific Test Class or Method

```bash
# Run specific class
pytest tests/ut/ir/core/test_span.py::TestSpan

# Run specific test method
pytest tests/ut/ir/core/test_span.py::TestSpan::test_span_creation
```

### Common Options

```bash
# Verbose output
pytest tests/ut/ir/ -v

# Show print output
pytest tests/ut/ir/ -s

# Run only failed tests
pytest tests/ut/ir/ --lf

# Run tests in parallel (requires pytest-xdist)
pytest tests/ut/ir/ -n auto

# Generate coverage report
pytest tests/ut/ir/ --cov=pypto.ir --cov-report=html
```

## Adding New Tests

When adding new tests, follow these steps:

1. **Determine the test directory**: Choose the appropriate subdirectory based on functional domain
2. **Create or update test file**: Use a descriptive file name
3. **Organize test classes**: Group related tests by functionality
4. **Write clear documentation**: Add descriptive docstrings to each test
5. **Follow naming conventions**: Use consistent naming patterns
6. **Keep tests self-contained**: Each test should create its own test data

## Maintenance Guidelines

- **Keep test files under 500 lines**: If a file gets too large, consider splitting it
- **Avoid code duplication**: Extract common logic into helper functions when needed
- **Run full test suite regularly**: Ensure all tests pass
- **Update documentation**: Update this README when changing organization structure

## References

- [pytest official documentation](https://docs.pytest.org/)
- [pytest fixtures guide](https://docs.pytest.org/en/stable/fixture.html)
- PyPTO IR design documentation: `docs/dev/00-ir_definition.md`
