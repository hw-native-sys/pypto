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

```
docs/
└── dev/
    ├── 00-ir_overview.md          # IR system overview
    ├── 01-ir_hierarchy.md         # IR node hierarchy
    ├── 02-ir_types_examples.md    # Type system examples
    ├── 03-structural_comparison.md # Comparison utilities
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

1. Read `docs/dev/00-ir_overview.md` and `docs/dev/01-ir_hierarchy.md` to understand existing hierarchy
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
