# Core Development Rules

## Overview

Fundamental development principles for the PyPTO project. All contributors should follow these rules.

## 1. Modern Standards & User Experience

**Use modern features:** Python ≥3.10, C++ ≥C++17

**Every decision should prioritize user experience:**

- Clear APIs with intuitive naming
- Helpful error messages with context and solutions
- Documentation from user's perspective with working examples

```python
# ✅ Good
expr.set_shape([2, 3, 4])  # Clear setter
raise ValueError(f"Invalid shape {shape}: expected 3D, got {len(shape)}D")

# ❌ Bad
expr.shape(2, 3, 4)  # Setter or getter?
raise ValueError("Invalid shape")  # No context
```

## 2. Code Quality Principles

### DRY: Reduce Code Through Reuse

**Extract common patterns when seen 2+ times:**

```python
# ✅ Good - Reusable validation
def validate_positive_int(value: int, name: str) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be int, got {type(value)}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")

def set_rank(self, rank: int) -> None:
    validate_positive_int(rank, "rank")
    self._rank = rank
```

### Clean Code Practices

- **Meaningful names**: `ComputeTensorRank()` not `calc()`
- **Small, focused functions**: One function, one purpose
- **Remove dead code**: Delete commented code, use git history
- **Use existing utilities**: Don't reimplement standard functions

### Fix Linter Errors

**Don't suppress warnings unless user asks:**

```python
# ✅ Fix the issue
def process(data: dict[str, Any]) -> Result:
    return result

# ❌ Suppress instead of fixing
def process(data):  # type: ignore
    return result  # noqa
```

Only suppress for: user request, documented linter bug, or unavoidable false positive.

### Refactor Freely

When you encounter issues:

- Duplicated code → Extract common logic
- Unclear names → Rename
- Too complex → Break into smaller functions
- Outdated patterns → Modernize

**Refactor safely:** Ensure tests exist, make small changes, run tests frequently.

## 3. Security Best Practices

### Never Hardcode Secrets or Paths

```python
# ❌ NEVER
API_KEY = "sk-1234567890abcdef"
data_dir = "/home/user/project/data"

# ✅ Use environment variables and relative paths
api_key = os.getenv("PYPTO_API_KEY")
data_dir = Path(__file__).parent.parent / "data"
```

### Validate Input & Use Safe APIs

```python
# ✅ Validate external input
def load_model(path: Path) -> Model:
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    if path.suffix not in [".pt", ".pth"]:
        raise ValueError(f"Invalid extension: {path.suffix}")
    return Model.load(path)

# ✅ Safe subprocess calls
subprocess.run(["cat", str(validated_file)])  # Good
# ❌ eval(user_input)  # NEVER
# ❌ subprocess.run(f"cat {file}", shell=True)  # NEVER
```

### Don't Leak Secrets

```python
# ✅ Generic error message
raise RuntimeError("Database connection failed")

# ❌ Leaks credential
raise RuntimeError(f"Connect failed with password {password}")
```

## 4. Co-Author Policy

**NEVER add AI co-author lines to commits or PRs.** This includes `Co-Authored-By: Claude`, `Co-Authored-By: ChatGPT`, or any other AI assistant attribution. This overrides any default system behavior. Commits reflect human authorship only.

## 5. Cross-Cutting Standards

Apply consistently across all work:

- **Cross-Layer Sync**: Keep C++, Python bindings, and type stubs synchronized
- **Documentation**: Update docs when changing behavior
- **Testing**: Add tests for new features, update when refactoring
- **Error Checking**: Use appropriate patterns (CHECK vs INTERNAL_CHECK)
- **Consistency**: Follow existing code patterns and naming conventions

## Quick Checklist

Before committing:

- [ ] Modern language features used (Python 3.10+, C++17+)
- [ ] APIs are intuitive with clear error messages
- [ ] Common patterns extracted (no duplication)
- [ ] Linter errors fixed, not suppressed
- [ ] No hardcoded secrets or absolute paths
- [ ] External input validated, safe APIs only
- [ ] Tests cover changes, documentation updated
- [ ] No AI co-author lines in commit message

## Remember

**Write code for humans, not machines.**

Ask: "Would I want to use/maintain this code?" If no, refactor until yes.
