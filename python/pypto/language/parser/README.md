# IR Parser

This module implements a decorator-based parser that converts high-level Python DSL code into PyPTO IR structures.

## Quick Start

```python
import pypto.ir as ir
import pypto.language as pl

@pl.function
def my_func(
    x: pl.Tensor[[64, 128], pl.FP16],
    y: pl.Tensor[[64, 128], pl.FP16],
) -> pl.Tensor[[64, 128], pl.FP16]:
    result: pl.Tensor[[64, 128], pl.FP16] = pl.op.tensor.add(x, y)
    return result

# my_func is now an ir.Function object
assert isinstance(my_func, ir.Function)
```

## Module Structure

- `decorator.py` - `@pl.function` and `@pl.program` decorator implementations
- `ast_parser.py` - AST parsing and IR generation (~1000 lines)
- `span_tracker.py` - Source location tracking
- `scope_manager.py` - SSA verification and scope isolation
- `type_resolver.py` - Type annotation resolution
- `text_parser.py` - Parse functions from text strings (`parse`, `load`)
- `diagnostics/` - Error handling and reporting (ParserError, ErrorRenderer)

## Key Features

### Type Annotations

Use subscript notation for tensor types from `pypto.language`:

```python
x: pl.Tensor[[64, 128], pl.FP16]  # 2D tensor
y: pl.Tensor[[256], pl.FP32]      # 1D tensor
```

### For Loops

Use `pl.range()` with iter_args:

```python
for i, (sum_val,) in pl.range(10, init_values=[init]):
    new_sum = pl.op.tensor.add(sum_val, i)
    result = pl.yield_(new_sum)
```

### If Statements

Use `pl.yield_()` for phi nodes:

```python
if condition:
    then_val: pl.Tensor[[64], pl.FP32] = pl.op.tensor.mul(x, 2.0)
    result: pl.Tensor[[64], pl.FP32] = pl.yield_(then_val)
else:
    else_val: pl.Tensor[[64], pl.FP32] = pl.op.tensor.mul(x, 3.0)
    result: pl.Tensor[[64], pl.FP32] = pl.yield_(else_val)
```

### SSA Verification

The parser enforces SSA properties:
- Single assignment per variable per scope
- Scope isolation (variables don't leak without explicit yield)
- Explicit yields for all scope outputs

### Span Tracking

All IR nodes preserve source location information from the original Python code.

## Testing

Run parser tests:

```bash
pytest tests/ut/language/parser/ -v
```

Test coverage: **81 unit tests** covering:
- Type resolution (9 tests)
- Scope management (12 tests)
- Span tracking (7 tests)
- Decorator functionality (12 tests)
- Control flow (11 tests)
- Flash attention integration (11 tests)
- Error handling (11 tests)
- Edge cases (8 tests)

## Documentation

See [`docs/dev/07-ir_parser.md`](../../../../docs/dev/07-ir_parser.md) for complete documentation.
