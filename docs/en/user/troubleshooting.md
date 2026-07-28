# Troubleshooting

> **Status:** DRAFT skeleton. Error message → likely cause → where to go.

## How to Use This Page

_TODO — search for the key phrase in your error message; each row points to a
fix or the relevant handbook chapter._

## Common Errors

_TODO — table, populated from real `CHECK` / `ValueError` messages:_

| Error message (fragment) | Likely cause | Go to |
| ------------------------ | ------------ | ----- |
| _e.g._ shape mismatch in `pl.matmul` | operand rank/shape wrong | [Language Guide](02-language_guide.md) |
| _e.g._ predicate reads stale value | `predicate=` without `deps=` | [Perf › Dependency & Dispatch](handbook/perf/03-dependency-dispatch.md) |
| … | … | … |

## FAQ

_TODO — recurring "how do I…" questions that aren't errors._

## See Also

- [Feature Handbook index](handbook/00-index.md)
- [Glossary](glossary.md)
