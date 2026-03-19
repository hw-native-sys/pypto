# Pass Configuration in PassContext

## Core Principle

**All pass-related configuration MUST live in `PassContext`, not in global state.**

`PassContext` is the single source of truth for pass execution settings. This ensures:

- Scoped, composable configuration (nested contexts override outer)
- No global side effects between compilations
- Thread-safe per-context settings
- Clean `with`-statement lifecycle management

## What Belongs in PassContext

| Config | PassContext Field | NOT |
| ------ | ----------------- | --- |
| Verification level | `PassContext([], VerificationLevel.BASIC)` | Global set/get functions |
| Pass instruments | `PassContext([VerificationInstrument(...)])` | Global registries |
| Future: logging level | `PassContext(..., log_level=...)` | Global log config |

## How to Add New Pass Config

1. **Add member to `PassContext`** (C++ header/impl)
2. **Add constructor parameter** with a sensible default
3. **Read from context in pipeline code** — fall back to env-var default when no context
4. **Expose in Python bindings** via constructor arg + getter
5. **Update type stubs** with the new parameter
6. **Never use global mutable state** for per-compilation settings

## Pattern

```cpp
// C++: PassPipeline reads config from context, falls back to env default
auto* ctx = PassContext::Current();
auto level = ctx ? ctx->GetVerificationLevel() : GetDefaultVerificationLevel();
```

```python
# Python: compile() wraps execution in PassContext
ctx = passes.PassContext([], verification_level)
with ctx:
    pm.run_passes(program)
```

## Anti-Patterns

```python
# ❌ Global state — not scoped, not composable
passes.set_some_config(value)
try:
    pm.run_passes(program)
finally:
    passes.set_some_config(old_value)

# ✅ PassContext — scoped, composable, thread-safe
with passes.PassContext([], some_config=value):
    pm.run_passes(program)
```

## Environment Variables

Environment variables (e.g., `PYPTO_VERIFY_LEVEL`) provide **defaults** only:

- Read once at startup via `GetDefault*()` functions
- Used as fallback when no `PassContext` is active
- Never mutated at runtime — `PassContext` overrides them
