# Testing and Examples Policy

## Core Principle

**DO NOT write examples or temporary test scripts unless explicitly requested.**

## Testing Guidelines

### Test Location and Organization

**All tests belong in `tests/`:**

- `tests/ut/core/` - Core functionality tests
- `tests/ut/ir/` - IR tests, split by topic: `core/`, `expressions/`, `operators/`,
  `parser/`, `printing/`, `statements/`, `transforms/`
- `tests/ut/pass/` - Pass manager tests
- `tests/lint/` - Linting and code quality checks

**NEVER create test files outside `tests/`:**

- ❌ No `test_quick.py` in project root
- ❌ No `example_usage.py` for exploration
- ❌ No temporary test scripts

### When to Add Tests

**Prefer adding to existing test files** when a related test file already exists. Only create a new test file when no existing file covers the topic. **Exception**: IR statement node tests should each have a dedicated file (e.g., `test_for_stmt.py`) for discoverability.

**Add tests for:**

- New features requiring validation
- Bug fixes (prevent regression)
- New public APIs
- Edge cases and boundary conditions
- Cross-layer functionality (C++ ↔ Python)

### When NOT to Create Tests

**Don't create** temporary "proof of concept" files, ad-hoc demo scripts, tests that only
show how something works (explain instead), or anything outside `tests/`. To verify
something, use the existing test structure or run existing tests.

### Test Framework

**Use pytest as the Python testing framework.** Do not use `unittest` or other testing packages.

- Write test functions, not test classes inheriting from `unittest.TestCase`
- Use plain `assert` statements, not `self.assertEqual()` etc.
- Use pytest fixtures for setup/teardown, not `setUp()`/`tearDown()` methods
- Use `pytest.raises()` for exception testing, not `self.assertRaises()`
- **Always use `assert` to verify results, never `print`.** Tests must fail on wrong output, not just display it.
- **Every test file must end with a `pytest.main` block:**

```python
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

```python
# ✅ Good - pytest style with assert
def test_tensor_shape():
    assert ir.TensorExpr().get_rank() == 3

# ❌ Bad - unittest style
class TestTensor(unittest.TestCase):
    def test_tensor_shape(self):
        self.assertEqual(ir.TensorExpr().get_rank(), 3)

# ❌ Bad - print style — passes even when the value is wrong
def test_tensor_shape():
    print(ir.TensorExpr().get_rank())
```

### Test Style: Before/After Pattern

**For IR transform and pass tests, use the before/after pattern:**

```python
def test_example_transform(self):
    @pl.program
    class Before:
        @pl.function
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            # Input IR before transformation
            ...

    @pl.program
    class Expected:
        @pl.function
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            # Expected IR after transformation
            ...

    After = passes.some_pass()(Before)
    ir.assert_structural_equal(After, Expected)
```

**Key rules:**

- Use `@pl.program` with `Before` and `Expected` classes (not helper functions)
- Compare with `ir.assert_structural_equal(After, Expected)`

## Authoring Surface: `@pl.jit` Outside `tests/ut/`

### Core Rule

**Every kernel written to be *run* — `examples/`, `tests/st/`, docs snippets — must use the
`@pl.jit` family. `@pl.program` + `@pl.function` is for `tests/ut/` only.**

`@pl.jit` is the surface users write; `@pl.program` is the shape `@pl.jit` specializes
*into* (printing a compiled program yields `@pl.program` source). A UT that asserts on
that lowered shape writes it directly; a UT of `@pl.jit` itself does not. ST and examples
are the user-facing corpus — they must demonstrate what users actually write.

| Location | Surface |
| -------- | ------- |
| `tests/ut/**` IR / pass transform tests | `@pl.program` + `@pl.function` (pattern above) |
| `tests/ut/**` where `@pl.jit` is itself under test (`tests/ut/jit/`) | `@pl.jit` family |
| `tests/st/**`, `examples/**` | `@pl.jit` family |
| `docs/**` snippets | `@pl.jit` family, unless the snippet documents `@pl.program` itself |

**Verify a new `@pl.jit` kernel with a full `compile()`, never `lower()` alone** — `lower()`
stops after the passes, so errors like "operators in an Orchestration body" never fire and
the kernel only fails when someone runs it.

### Translation

| `@pl.program` form | `@pl.jit` form |
| ------------------ | -------------- |
| `@pl.function(type=pl.FunctionType.InCore)` | `@pl.jit.incore` |
| `@pl.function(type=pl.FunctionType.Orchestration)` entry | `@pl.jit` |
| HOST orchestrator entry | `@pl.jit.host` |
| `@pl.inline` / `type=Inline` | `@pl.jit.inline` |
| `@pl.function(type=pl.FunctionType.Opaque)` | `@pl.jit.opaque` |

Deps are discovered from the entry body — call them by name, no class needed. `@pl.jit.host`
also discovers `@pl.jit` chip orchestrators, so distributed ST needs no `@pl.program`
either. See `docs/en/user/language/01-functions.md`.

### Existing Files

The pre-`@pl.jit` ST / example corpus is historical, not a violation. **New files must use
`@pl.jit`; migrate an existing file when you substantially modify it** — never as an
unrelated drive-by rewrite.

### Narrow Exceptions

Outside `tests/ut/`, `@pl.program` stays correct only when the case *is* about that form —
parser / printer round-trips consuming printed `@pl.program` text (e.g.
`examples/utils/parse_from_text.py`). If you believe `@pl.jit` cannot express a new ST
case, **tell the user before falling back to `@pl.program`** — that gap is itself the finding.

## Examples Policy

### Examples Directory

The `examples/` directory contains **user-facing examples only**, in a two-axis layout —
difficulty tiers for the teaching kernels, category folders for everything else:

- `examples/beginner/` - Language basics, one concept per file (01_hello_world.py through 06_concat.py)
- `examples/intermediate/` - Real-kernel patterns (01_fused_linear.py through 06_dyn_valid_shape.py)
- `examples/advanced/` - Performance and low-level techniques (01_split_k.py, 02_auto_tile_matmul.py, 03_mixed_kernel.py)
- `examples/models/` - Multi-kernel model examples, numbered by difficulty (01_ffn.py through 09_paged_attention_spmd.py)
- `examples/runtime/` - Host/runtime patterns (dispatch, distributed callbacks, KV cache)
- `examples/utils/` - Parsing, cross-function calls, error handling

### When to Write Examples

**Only create examples when** the user asks for one, a major new feature needs a
demonstration, or an API change invalidates an existing example.

### When NOT to Write Examples

**Don't create examples** to demonstrate code during development, to test functionality
(use `tests/`), or to show implementation details (use `docs/`). To demonstrate something,
explain it in conversation, add a snippet to the docs, or point at an existing example.

## Remember

**Tests validate, examples demonstrate, docs explain.** Don't conflate these purposes —
create files only when necessary, in the proper location, on the right authoring surface.
