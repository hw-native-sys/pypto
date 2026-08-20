# 语言指南

`pypto.language` 的完整表面，一页一个主题：你能写什么、每个构造是什么意思、它在什么地方会失败。

> **前置**：[快速上手](../02-quickstart.md) 与 [编程模型](../03-programming-model.md)。本章假设你已经编译过东西，并且分得清控制面（control plane）与执行面（execution plane）。

## 本章是什么

一份按**能力**组织的指南（guide）—— 每页覆盖语言的一个部分及其边界情况。它不是教程：没有任何一页从头到尾搭出一个完整 kernel。它也不是 API 参考：要看签名去 [API 参考](../../api/index.md)，按名字查则从[算子目录](../ops/01-catalog.md)进。

约定 `import pypto.language as pl` —— 本章所有名字都通过这个别名访问。

## 目录

| 页面 | 覆盖内容 |
| ---- | -------- |
| [类型](00-types.md) | dtype、`Tensor` / `Tile` / `Scalar` / `Array` / `Tuple`、布局、动态 shape、参数方向 |
| [函数与程序](01-functions.md) | `@pl.jit` 家族、`@pl.function`、`@pl.program`、`@pl.inline`、跨函数调用、外部 kernel |
| [控制流](02-control-flow.md) | `pl.range` / `parallel` / `unroll` / `pipeline` / `while_`、循环携带值、`yield_`、`cond`、SSA |
| [内存与数据搬运](03-memory.md) | 内存空间、`load` / `store` / `move`、`valid_shape` 与 `fillpad`、片上常驻 |
| [作用域与放置](04-scopes.md) | `at` / `cluster` / `spmd` / `split_aiv` —— 代码在哪里执行 |
| [编译期指令](05-directives.md) | `static_print` / `static_assert`、`const` |
| [语言语法](06-syntax.md) | 下标语法糖、Python 运算符、闭包捕获 |

## 阅读顺序

先读 [类型](00-types.md) 和 [函数与程序](01-functions.md) —— 其余每页都以这两页为前提。之后各页彼此独立：

```text
00-types ──► 01-functions ──┬─► 02-control-flow
                            ├─► 03-memory
                            ├─► 04-scopes   ← 如果你此前只写过单 kernel，
                            └─► 05-directives           这一页的缺口最大
```

[作用域与放置](04-scopes.md) 是多数读者在别处找不到对应材料的一页：它讲工作如何被放置到核上，以及运行时真正执行的那张任务图是怎么成形的。

## 另请参阅

- [算子](../ops/index.md) —— 某个算子属于哪个命名空间，以及完整目录。
- [编程模型](../03-programming-model.md) —— 本章所描述表面背后的抽象。
- [Python IR 语法规范](../../dev/language/00-python_syntax.md) —— 解析器自身的参考，包含本指南不推荐的写法。
- [Passes](../../dev/passes/index.md) —— 编译器如何处理每个构造。
