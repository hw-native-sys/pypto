# Documentation Workflow

## Overview

PyPTO follows a **documentation-first development** approach. This ensures that code and documentation remain aligned, making the codebase easier to understand and maintain.

## Core Principles

### 1. Read Documentation Before Coding

**Always read relevant documentation before making any code changes.**

Key documentation files:
- **[`docs/dev/00-ir_definition.md`](../../docs/dev/00-ir_definition.md)** - Comprehensive IR system architecture
  - IR node hierarchy (IRNode → Expr, Stmt, Function)
  - Type system (ScalarType, TensorType, UnknownType)
  - Expression types (Var, ConstInt, BinaryExpr, UnaryExpr, Call)
  - Statement types (AssignStmt, IfStmt, ForStmt, YieldStmt, SeqStmts, OpStmts)
  - Python API usage examples

- **[`docs/dev/01-structural_comparison.md`](../../docs/dev/01-structural_comparison.md)** - Structural comparison utilities
  - `structural_equal()` and `structural_hash()` functions
  - Reflection system and field types (IgnoreField, UsualField, DefField)
  - Auto-mapping behavior for pattern matching
  - Implementation details

- **Other docs in [`docs/dev/`](../../docs/dev/)** - Check for additional relevant documentation

### 2. Review Documentation After Each Edit

**After making any code changes, review the documentation to ensure alignment.**

Ask yourself:
- Does the documentation still accurately describe how the code works?
- Are there any examples in the docs that might be affected?
- Did I change any APIs or behavior that's documented?
- Are field descriptors and reflection behavior still accurate?

### 3. Update Documentation When Needed

**If your code changes affect documented behavior, update the documentation.**

Update docs when you:
- Modify IR node structures or add new node types
- Change API signatures or usage patterns
- Add, remove, or modify fields in IR nodes
- Change reflection field types (IgnoreField, UsualField, DefField)
- Modify type system or expression handling
- Update build/test procedures

## Documentation Structure

```
docs/
└── dev/
    ├── 00-ir_definition.md        # IR system architecture
    ├── 01-structural_comparison.md # Comparison utilities
    └── ...                         # Other developer docs
```

## Documentation Style Guide

When updating documentation:

1. **Use clear, descriptive headings** - Make it easy to scan
2. **Provide code examples** - Show both C++ and Python usage
3. **Explain the "why"** - Don't just describe what code does, explain design decisions
4. **Use BNF or diagrams** - For complex structures like grammar
5. **Keep examples working** - Test that examples actually compile/run
6. **Link between docs** - Reference related documentation files
7. **Maintain consistency** - Follow the existing documentation style

## Common Documentation Tasks

### Adding a New IR Node Type

1. Read `docs/dev/00-ir_definition.md` to understand existing hierarchy
2. Implement the new node type in C++
3. Update the "IR Node Hierarchy" section
4. Add BNF grammar if applicable
5. Provide Python usage examples
6. Update structural comparison docs if reflection behavior is special

### Modifying Existing IR Node

1. Read current documentation for that node
2. Make code changes
3. Update node description and examples
4. Verify all examples still work
5. Update any affected sections in other docs

### Changing Build/Test Procedures

1. Make changes to CMake or test infrastructure
2. Test that instructions work from a clean state

## Documentation Quality Checklist

Before finalizing changes, verify:

- [ ] Code matches documented behavior
- [ ] All code examples in docs are valid and tested
- [ ] C++ implementation matches Python API documentation
- [ ] Field descriptors and reflection behavior are accurate
- [ ] Links between documentation files are correct
- [ ] No broken references to moved/renamed files
- [ ] Examples use current API (no deprecated patterns)
- [ ] Documentation follows existing style and formatting

## Remember

**Good documentation is as important as good code.**

Documentation is the primary way developers understand the system. Keeping it accurate and up-to-date prevents bugs, reduces confusion, and makes the codebase more maintainable.

When in doubt, update the docs!
