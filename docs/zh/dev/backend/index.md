# 后端

逐架构的差异行为，从 pass 中剥离出来。

pass 从不对 `BackendType` 做分支判断。所有与架构相关的内容 —— codegen 目标、运行时 API
名称、硬件冒险规避、跨核 layout 规则 —— 都由当前 `PassContext` 提供的 `BackendHandler`
回答。新增一个架构意味着新增一个 handler，而不是修改 pass。

| 页面 | 内容 |
| ---- | ---- |
| [BackendHandler：有原则的后端分派](00-backend_handler.md) | 该虚接口、pass 如何查询它，以及新增后端需要做什么 |

## 已注册算子清单

通过具体后端查询实际注册的算子：

```python
from pypto import backend

target = backend.get_backend_instance(backend.BackendType.Ascend910B)
names = target.get_registered_op_names()
```

返回列表已排序，与注册表相互独立。查询结果反映目标架构的排除项和覆盖项，
不会改变全局后端选择。检视降低（lowering）覆盖范围时应使用该清单；
解析注册源码可能误计注释中的示例，或遗漏通过共享表注册的名称。
算子已注册并不意味着它支持所有数据类型、形状或布局。
列表也包含 IR 算子定义已不存在的历史后端条目；检视当前 IR 时，
可通过 `ir.is_op_registered(name)` 区分这些条目。

`tests/ut/backend/test_buffer_migration_inventory.py` 中的 Buffer 迁移检查
将两个实际后端注册表与显式审核清单比较：22 个算子族、168 个有效名称和 8 个历史回调。
它会检测名称的新增与删除、重复分类、目标架构变化，以及重新获得 IR 定义的历史回调。
注册范围变化时，需要同步更新该测试的审核清单。

清单中的 `PLANNED` 和 `RESTRICTED` 状态描述已声明的迁移成熟度。
`RESTRICTED` 表示部分形式已实现，不保证所有数据类型、布局、属性或目标架构形式都受支持。
该检查让清单遗漏触发测试失败；实际实现覆盖范围仍由生产转换规则（conversion recipes）
及其转换、原生编译和数值测试确认。审核清单不参与降低过程，也不会启用默认切换。

## 另请参阅

- [Pass、PassContext、PassPipeline 与 PassManager](../passes/00-pass_manager.md) —— handler 的来源。
- [PTO ISA 参考](../../reference/index.md) —— handler 所抽象掉的硬件差异。
