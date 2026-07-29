# 后端

逐架构的差异行为，从 pass 中剥离出来。

pass 从不对 `BackendType` 做分支判断。所有与架构相关的内容 —— codegen 目标、运行时 API
名称、硬件冒险规避、跨核 layout 规则 —— 都由当前 `PassContext` 提供的 `BackendHandler`
回答。新增一个架构意味着新增一个 handler，而不是修改 pass。

| 页面 | 内容 |
| ---- | ---- |
| [BackendHandler：有原则的后端分派](00-backend_handler.md) | 该虚接口、pass 如何查询它，以及新增后端需要做什么 |

## 另请参阅

- [Pass、PassContext、PassPipeline 与 PassManager](../passes/00-pass_manager.md) —— handler 的来源。
- [PTO ISA 参考](../../reference/index.md) —— handler 所抽象掉的硬件差异。
