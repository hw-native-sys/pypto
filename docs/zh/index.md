# PyPTO

面向 tile 中心计算的高性能编程框架。

PyPTO 让你用 Python 编写 NPU kernel 及其编排逻辑，经过多层 IR 编译后，把生成的任务图
交给运行时在设备上执行。

## 从哪里开始

| 你是... | 从这里开始 |
| ------- | ---------- |
| 初次接触 PyPTO | [入门指南](user/00-getting_started.md) —— 安装、第一个程序、编译与运行 |
| 编写 kernel | [语言指南](user/language/index.md) —— 类型、函数、控制流、内存、作用域 |
| 查找某个算子 | [算子](user/ops/index.md) —— `pl.*` / `pl.tensor.*` / `pl.tile.*` 全貌 |
| 查某个名字的签名 | [API 参考](api/index.md) —— 全部 `pl.*` 名字，由 docstring 生成 |
| 排查结果不对 | [Torch Codegen 调试指南](user/tools/01-torch-codegen.md) —— 把 IR 跑在 PyTorch 上对拍 |
| 参与编译器开发 | [开发者文档](dev/index.md) —— IR、passes、代码生成 |
| 阅读生成的代码 | [PTO ISA 参考](reference/index.md) —— 集群架构与指令语义 |

## PyPTO 与运行时的分工

PyPTO 是**编译器与编程语言**。调度并执行其产物的**运行时**是另一个项目
[`hw-native-sys/simpler`](https://github.com/hw-native-sys/simpler)，在本仓库中作为
`runtime/` 子模块引入。

这条边界决定了各类内容的归属：

| 关注点 | 归属 | 文档位置 |
| ------ | ---- | -------- |
| `pl.*` 语言、`ir.compile()`、IR 与 passes | PyPTO | 本站 |
| `pypto.runtime` —— `ChipWorker`、`DeviceTensor`、`RunConfig`、`benchmark` | PyPTO（其自有 Python 层） | 本站 |
| 调度器实现、graph building、message queue、tensormap 与 ring buffer | simpler | <https://hw-native-sys.github.io/simpler/> |

`pypto.runtime` 是 PyPTO 自己的 Python 封装层，不是 simpler 的 API —— 它写在本站；
外链出去的是 simpler 的**内部机制**。

## 关于本文档

- **英文为准。** `docs/zh/` 与英文逐页镜像；用页头的语言选择器切换。
- `docs/` 下的 markdown 是唯一真源，在 GitHub 上可直接阅读。本站是构建产物，不入库。
- 用户手册正在从上述四篇指南扩展为完整的分章手册（教程、分布式编程、性能优化、精度
  定位）。方案见 [issue #2120](https://github.com/hw-native-sys/pypto/issues/2120)。
