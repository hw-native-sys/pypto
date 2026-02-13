# Python Coding Style

## Type Annotations

### `X | Y` vs `TypeVar` vs `@overload`

Use the simplest tool that captures the type relationship:

**`X | Y` (pipe union) is fine** when types are independent and both sides are normal Python typing/runtime types (no extension-bound runtime types) — no correlation between parameters or between input and output:

```python
# ✅ Pipe union — rhs type doesn't affect return type
def add(lhs: Expr, rhs: int | float | Expr) -> Call: ...
```

**`TypeVar` when input/output types are related** and the relationship is expressible as a generic:

```python
# ✅ TypeVar — return type matches input type
T = TypeVar("T", Tensor, Tile)
def clone(x: T) -> T: ...

# ✅ TypeVar — correlated parameters (same type required)
T = TypeVar("T", int, str)
def pair(a: T, b: T) -> tuple[T, T]: ...
```

**`@overload` when `TypeVar` can't express the relationship** — distinct call patterns with different arg counts, or complex input→output mappings:

```python
# ✅ @overload — different arg counts, different return types
@overload
def range(*args: int) -> RangeIterator[int]: ...
@overload
def range(*args: int, init_values: list[Any]) -> RangeIterator[tuple[int, tuple[Any, ...]]]: ...

# ✅ @overload — different input types produce different output types (not expressible as TypeVar)
@overload
def create(shape: list[int], dtype: DataType) -> Tensor: ...
@overload
def create(shape: list[int], dtype: DataType, memref: MemRef) -> BoundTensor: ...
```

**Decision guide:**

| Condition | Tool |
|-----------|------|
| Types unrelated, same return type | `X \| Y` |
| Input type = output type, or params must match | `TypeVar` |
| Different arg counts or complex type mapping | `@overload` |

### Use Modern Type Syntax (Python 3.10+)

**Use built-in generics and pipe union syntax:**

| Use | Don't use |
|-----|-----------|
| `list[int]` | `List[int]` |
| `dict[str, Any]` | `Dict[str, Any]` |
| `tuple[int, ...]` | `Tuple[int, ...]` |
| `set[str]` | `Set[str]` |
| `type[Foo]` | `Type[Foo]` |
| `int \| float` | `Union[int, float]` |
| `str \| None` | `Optional[str]` |

**Imports still needed from `typing`:** `Any`, `Sequence`, `Mapping`, `Callable`, `TypeVar`, `overload`, `Final`, `Iterator`, `Generic`

### Type Hint All Public APIs

- All public function parameters and return types must have type hints
- Private/internal functions should also have hints when non-obvious
- Use `X | None` for nullable parameters in normal Python-only types
- For runtime-bound extension types, use `Optional[X]` instead of `X | None`

## String Formatting

**Use f-strings exclusively.** No `.format()` or `%` formatting.

```python
msg = f"Invalid shape {shape}: expected {expected}D, got {len(shape)}D"  # ✅
msg = "Invalid shape {}: expected {}D".format(shape, expected)           # ❌
```

## Docstrings

**Google-style docstrings.** Sections in order: Summary → Args → Returns → Raises → Examples (all optional except summary).

```python
def transform(node: IRNode, config: Config) -> IRNode:
    """Apply transformation to an IR node.

    Args:
        node: The IR node to transform
        config: Transformation configuration

    Returns:
        Transformed IR node
    """
```

## Naming Conventions

| Element | Convention | Example |
|---------|-----------|---------|
| Classes | PascalCase | `TensorMeta`, `PassManager` |
| Functions/methods | snake_case | `get_rank()`, `set_shape()` |
| Constants | UPPER_SNAKE_CASE | `DT_FP32`, `DT_INT32` |
| Private | Leading underscore | `_normalize_expr()` |
| TypeVars | Uppercase letter or descriptive | `T`, `NodeT` |

## Imports

**Order:** standard library → third-party → local (relative)

```python
import os
from enum import Enum
from typing import Any, Sequence

from pypto.pypto_core import ir as _ir

from .printer import python_print
```

- Use absolute imports for cross-package, relative for within-package
- Alias internal bindings: `from pypto.pypto_core import ir as _ir`
- No `from __future__ import annotations` (project uses Python 3.10+)

## Error Messages

**Always include context — show what was received and what was expected:**

```python
raise ValueError(f"Tensor dimension {i} must be positive, got {dim}")   # ✅
raise ValueError("Invalid dimension")                                    # ❌
```

## Enum Usage

**Use `enum.Enum` for fixed sets of named values.**

## File Structure

1. Copyright header
2. Module docstring
3. Imports (standard → third-party → local)
4. Module-level constants
5. Classes and functions
6. `__all__` in `__init__.py` files
