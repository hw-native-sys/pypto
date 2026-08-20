# API 参考

由源码 docstring 生成，因此不会与代码脱节。

| 页面 | 覆盖 |
| ---- | ---- |
| [`pl`](language.md) | 装饰器、类型、控制流，以及按类型分派的算子包装 |
| [`pl.tile`](tile.md) | Tile 级算子 —— InCore 函数内部 |
| [`pl.tensor`](tensor.md) | 张量级算子 —— 编排层，整张张量 |
| [`pl.system`](system.md) | 同步、cache 与跨核原语 |
| [`pl.array`](array.md) | 定长数组，主要用于 `pl.TASK_ID` 扇入 |
| [`pl.prefetch`](prefetch.md) | GM 到 L2 的异步预取 |
| [`pl.optimizations`](optimizations.md) | `pl.at(..., optimizations=[...])` 接受的条目 |

## 怎么用

**从[算子目录](../user/ops/01-catalog.md)开始，而不是从这里。** 目录按族列出每个算子并各配一句话，且每个名字都链到上面这些页里对应的签名。这些页回答「参数是什么」，目录回答「我要哪个算子」。

**这里用的是规范名，不是你书写的名字。** `pl.create_tensor` 是 `tensor.create` 的别名，因此在 `pl.tensor` 页上显示为 `create`。目录里的链接已经替你解析好了。

**导语是中文，API 正文是英文。** 正文由源码 docstring 生成，因此保持源码的语言；每页顶部的中文导语说明这个命名空间**是什么、何时用**。

## 构建会检查什么

`mkdocs build --strict` 会因两类问题失败：docstring 的 `Args:` 写了签名里没有的参数；目录链接指向一个并未被渲染的符号。两者都是真实缺陷 —— 而且这些页第一次构建时**两类都当场抓到了**。

## 参见

- [算子目录](../user/ops/01-catalog.md) —— 通向这些页面的分类索引。
- [选择命名空间](../user/ops/00-dispatch.md) —— `pl.` / `pl.tile.` / `pl.tensor.` 的取舍。
- [语言指南](../user/language/index.md) —— 这些签名所用类型背后的说明。
