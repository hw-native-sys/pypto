# Testing and Examples Policy

## Core Principle

**DO NOT write examples or temporary test scripts unless explicitly requested.**

## Testing Guidelines

### Test Location and Organization

**All tests belong in `tests/`:**

- `tests/ut/core/` - Core functionality tests
- `tests/ut/ir/` - IR (Intermediate Representation) tests
  - `core/` - Basic IR nodes
  - `expressions/` - Expression tests
  - `operators/` - Operator tests
  - `parser/` - Parser tests
  - `printing/` - Printer tests
  - `statements/` - Statement tests
  - `transforms/` - Transform tests
- `tests/ut/pass/` - Pass manager tests
- `tests/lint/` - Linting and code quality checks

**NEVER create test files outside `tests/`:**

- ❌ No `test_quick.py` in project root
- ❌ No `example_usage.py` for exploration
- ❌ No temporary test scripts

### When to Add Tests

**Add tests for:**

- New features requiring validation
- Bug fixes (prevent regression)
- New public APIs
- Edge cases and boundary conditions
- Cross-layer functionality (C++ ↔ Python)

**When user explicitly requests:**

- "Add tests for this feature"
- "Write a test to verify this works"
- "Create regression test for this bug"

### When NOT to Create Tests

**Don't create:**

- Temporary "proof of concept" test files
- Ad-hoc example scripts to demonstrate functionality
- Test files just to show how something works (explain instead)
- Tests outside the `tests/` directory structure

**If you need to verify something:**

- Use existing test structure
- Run existing tests
- Explain in comments or documentation

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

## Examples Policy

### Examples Directory

The `examples/` directory contains **user-facing examples only:**

- `examples/ir_builder/` - IR builder examples
- `examples/ir_parser/` - Parser examples
- `examples/block-level/` - Block-level operation examples

### When to Write Examples

**Only create examples when:**

- User explicitly requests: "Create an example showing X"
- Adding major new feature that needs demonstration
- Updating example due to API changes

### When NOT to Write Examples

**Don't create examples to:**

- Demonstrate how code works during development
- Test functionality (use `tests/` instead)
- Show implementation details (use docs instead)

**If you need to demonstrate something:**

- Explain it in conversation
- Add to documentation with code snippets
- Reference existing examples

## Documentation vs Examples vs Tests

| Purpose | Location | When to Create |
| ------- | -------- | -------------- |
| **Explain concepts** | `docs/` | Always keep updated |
| **Show usage** | `examples/` | User requests only |
| **Verify correctness** | `tests/` | For all new features |

## Summary

- ❌ No temporary test files or examples
- ✅ Add tests to `tests/ut/` for new features
- ❌ No examples unless explicitly requested
- ✅ Update `docs/` to explain, not create examples
- ✅ Use existing examples in `examples/` as reference

## Remember

**Tests validate, examples demonstrate, docs explain.**

Don't conflate these purposes. Keep the codebase clean by only creating files when necessary and in the proper location.
