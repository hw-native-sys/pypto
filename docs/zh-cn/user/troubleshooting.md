# 疑难排查

> **状态：** 草稿骨架。报错信息 → 可能原因 → 前往何处。

## 如何使用本页

_TODO —— 在报错信息里搜索关键短语；每行指向一个修复或相关手册章节。_

## 常见报错

_TODO —— 表格，从真实的 `CHECK` / `ValueError` 信息填充：_

| 报错信息（片段） | 可能原因 | 前往 |
| ---------------- | -------- | ---- |
| _例如_ `pl.matmul` 形状不匹配 | 操作数 rank/shape 错误 | [语言指南](02-language_guide.md) |
| _例如_ predicate 读到 stale 值 | `predicate=` 未配 `deps=` | [性能 › 依赖与分发](handbook/perf/03-dependency-dispatch.md) |
| … | … | … |

## FAQ

_TODO —— 反复出现、并非报错的“如何……”问题。_

## 参见

- [功能手册索引](handbook/00-index.md)
- [术语表](glossary.md)
