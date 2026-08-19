# 工具

结果不对时，该拿什么。

四个问题，四件工具。顺序是有意义的 —— 每一件都比下一件便宜，而且第一个问题编译器通常已经替你回答过了。

| 问题 | 工具 | 页面 |
| ---- | ---- | ---- |
| 出了什么错，错在哪？ | 错误类型、日志级别、IR dump | [调试](00-debugging.md) |
| 是 **IR** 错了，还是设备错了？ | `pypto.debug.torch_codegen` | [Torch codegen](01-torch-codegen.md) |
| 片上放了什么，活多久？ | `pypto.tools.memory_map` | [内存图](02-memory-map.md) |
| 时间去哪了？ | L2 泳道图 | [性能](../performance/00-swimlane.md) |

## 先看便宜的那两样

每次编译都会写出两份产物，读它们不花任何代价：

- **`report/perf_hints.log`** —— 编译器注意到但没有拒绝的东西：低于硬件粒度的搬运、它没能 tile 的 matmul、没放下的流水深度。还会往 stderr 打一行摘要。
- **报错本身。** PyPTO 区分用户错误与内部错误，这个区分直接告诉你该改自己的代码还是该提 bug —— 见[调试](00-debugging.md)。

## 参见

- [精度](../precision/index.md) —— 数值不对时这些工具服务的那条流程。
- [性能](../performance/index.md) —— 数值对但慢时的同类流程。
- [执行](../execution/index.md) —— 被调试的编译与派发面。
