# Documentation Workflow

## Overview

PyPTO follows a **documentation-first development** approach. This ensures that code and documentation remain aligned, making the codebase easier to understand and maintain.

## Core Principles

### 1. No Standalone Markdown Documents

**Do NOT create markdown files outside of the `/docs` folder.**

This includes:

- Do NOT write changes summaries or CHANGES.md files
- Do NOT create README.md files in arbitrary locations
- Only create markdown files in `/docs` when explicitly needed for project documentation

Communicate changes and summaries directly in conversation, not as files.

### 2. Read Documentation Before Coding

**Always read relevant documentation before making any code changes.**

Before starting work:

- Read relevant documentation in `docs/dev/`
- Understand the architecture and components being modified
- Check for related documentation files
- Review existing examples and patterns

### 3. Review Documentation After Each Edit

**After making any code changes, review the documentation to ensure alignment.**

After completing changes:

- Verify documentation still matches code behavior
- Check if examples in docs are affected
- Update APIs or behavior that are documented
- Ensure field descriptors and reflection behavior are accurate
- Verify cross-references are valid
- Update affected sections in related docs

### 4. Update Documentation When Needed

**If your code changes affect documented behavior, update the documentation.**

Update docs when you:

- Modify IR node structures or add new node types
- Change API signatures or usage patterns
- Add, remove, or modify fields in IR nodes
- Change reflection field types (IgnoreField, UsualField, DefField)
- Modify type system or expression handling
- Update build/test procedures

## Documentation Structure

**The documentation is organized by topic in `docs/dev/`.**

**Before working with documentation, always read the current structure:**

```bash
ls -la docs/dev/                    # See topic folders
find docs/dev -name "*.md" | sort   # List all documentation files
```

**Current organization (topic folders)**:

- `ir/` - IR Foundation (types, operators, builders, parsers)
- `passes/` - Passes & Transforms (pass system, individual passes)
- `codegen/` - Code Generation (PTO, CCE)
- `language/` - Python DSL

**Do not assume specific file names - explore the structure to find relevant files.**

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

1. Explore `docs/dev/ir/` to find overview and hierarchy documentation
2. Read the relevant files to understand existing IR node hierarchy
3. Implement the new node type in C++
4. Update the IR node hierarchy documentation
5. Add BNF grammar if applicable
6. Provide Python usage examples
7. Update structural comparison docs if reflection behavior is special

### Modifying Existing IR Node

1. Read current documentation for that node
2. Make code changes
3. Update node description and examples
4. Verify all examples still work
5. Update any affected sections in other docs

### Adding a New Pass

1. Explore `docs/dev/passes/` to find pass system documentation
2. Read the pass manager docs to understand the pass system
3. Implement the pass in C++ (`src/ir/transforms/`) using `CreateFunctionPass`/`CreateProgramPass`
4. Add factory function to `include/pypto/ir/transforms/passes.h`
5. **Declare PassProperties** (required/produced/invalidated) in the factory call
6. Add Python binding to `python/bindings/modules/passes.cpp`
7. Update type stub in `python/pypto/pypto_core/passes.pyi`
8. Create per-pass documentation following existing pass doc patterns
9. Update pass manager documentation if adding to default strategy
10. If the pass produces a new IRProperty, add a PropertyVerifier and register in PropertyVerifierRegistry
11. Add tests in `tests/ut/ir/transforms/`

### Changing Build/Test Procedures

1. Make changes to CMake or test infrastructure
2. Test that instructions work from a clean state
3. Update relevant documentation if procedures change

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
