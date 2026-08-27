# 语言

编译器视角下的 Python DSL。

| 页面 | 内容 |
| ---- | ---- |
| [Python IR 语法规范](00-python_syntax.md) | 模块结构、类型系统、表达式 —— 规范的入口页 |
| [语句与控制流](01-statements.md) | 赋值、if/for/while、作用域上下文管理器、yield、编译期指令、SSA phi 节点 |
| [手工依赖原语](02-manual_dependencies.md) | `pl.manual_scope`、显式 `deps=` 边、调度谓词、array-carry fence |
| [函数与程序结构](03-functions.md) | 函数类型、参数方向、跨模块复用、打印 IR |
| [集成手写 C++ Kernel](04-external-kernels.md) | 从 PyPTO 程序中调用已有的手写 C++ InCore kernel |
| [GM 缓存访问策略](05-cache-policy.md) | `pl.set_cache_policy` / `pl.load(cache=...)` —— 声明式 GM 缓存策略、它的一致性契约，以及它如何抵达 codegen |

## 另请参阅

- [IR Parser](../ir/07-parser.md) —— 该语法如何变成 IR。
- [语言指南](../../user/language/index.md) —— 从用户视角看同一门语言。
