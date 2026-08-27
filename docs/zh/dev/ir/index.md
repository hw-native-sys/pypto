# IR

PyPTO 的中间表示 —— 其他所有层都依赖的契约。

IR 定义是整个编译器的真源：pass 可以重写，IR 节点定义不可以。先读[概览](00-overview.md)，
再读与你所改动内容对应的页面。

| 页面 | 内容 |
| ---- | ---- |
| [概览](00-overview.md) | 从 Python DSL 到生成 kernel 的编译流水，以及 IR 在其中的位置 |
| [节点层次](01-hierarchy.md) | 所有 IR 节点类型的完整参考，按类别组织 |
| [类型与示例](02-types.md) | 类型系统及其实际用法示例 |
| [结构化比较](03-structural_comparison.md) | 按结构而非指针身份比较 IR 节点 |
| [序列化](04-serialization.md) | 基于 MessagePack 的 `.pto` 序列化 |
| [算子系统](05-operators.md) | 带自动类型推导的类型安全算子定义 |
| [IR Builder](06-builder.md) | 增量构造 IR —— Python 用上下文管理器，C++ 用 Begin/End |
| [IR Parser](07-parser.md) | 通过 `@pl.function` / `@pl.program` 把 Python DSL 转成 IR，以及它强制的 SSA 性质 |
| [参数方向](08-param-directions.md) | `In`/`Out`/`InOut` 如何被推导——各阶段共同读取的注册表声明，以及基于它的四个 pass |

## 另请参阅

- [Passes](../passes/index.md) —— 运行在该 IR 之上的各种变换。
- [Python IR 语法规范](../language/00-python_syntax.md) —— parser 接受的表层语法。
- [IR 验证器](../passes/99-verifier.md) —— 在 pass 之间检查 IR 合法性的属性验证器。
