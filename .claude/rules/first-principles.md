# First Principles

**This is the most important rule in the project. All sessions must follow it.**

## 1. IR Design Is the Source of Truth

All design choices derive from the IR definition itself — not from the current implementation.

When evaluating a bug, designing a feature, or resolving a conflict between code and spec:

- **Ask "what does the IR design require?"** — not "what does the current code do?"
- If the implementation diverges from the IR design, the implementation is wrong
- Pass behavior, codegen output, and Python APIs must all be consistent with the IR definition

```text
# Decision flow
Is the behavior correct per IR design?
├─ YES → Implementation is fine
└─ NO  → Fix the implementation, not the IR design (unless the IR design itself needs evolving)
```

## 2. Passes Are Replaceable; IR Definitions Are Not

- **Any pass implementation can be rewritten** — passes are transformations over the IR and can always be reimplemented with a better algorithm or approach
- **IR node definitions must be preserved** — they are the contract that all layers (C++, bindings, Python, tests, docs) depend on
- When fixing a pass bug, feel free to restructure or rewrite the pass logic entirely
- When a pass produces wrong output, the fix belongs in the pass — not in the IR definition (unless the IR design genuinely needs to evolve, which requires explicit user approval)

## 3. Never Hack Test Cases

When a test fails:

1. **Investigate the root cause** — is the code wrong, or is the test wrong?
2. **If the code is wrong** — fix the code, not the test
3. **If the test is genuinely incorrect** — **inform the user before editing it** and explain why the test expectation is wrong
4. **Never silently modify test expectations** to make a failing test pass

```text
Test fails
├─ Code bug → Fix the code
└─ Test bug → Tell the user FIRST, then fix with approval
    ├─ Explain what the test asserts
    ├─ Explain why that assertion is wrong
    └─ Propose the corrected expectation
```

This applies to all test types: unit tests, transform tests, round-trip tests, and integration tests.
