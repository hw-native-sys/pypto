# 开发者文档

PyPTO 的构成：IR、pass 流水线、代码生成，以及围绕它们的基础设施。

本章面向**开发编译器本身**的人。如果你是在编写 PyPTO 程序，请从[用户手册](../user/index.md)开始。

## 子章节

| 章节 | 内容 |
| ---- | ---- |
| [IR](ir/index.md) | 节点层次、类型系统、算子、builder、parser、序列化、结构化比较 |
| [Passes](passes/index.md) | pass 框架与默认流水线中的每个 pass，按执行顺序编号 |
| [语言](language/index.md) | Python DSL 语法规范与外部 C++ kernel 接入 |
| [代码生成](codegen/index.md) | 把 IR lower 成 PTO-ISA 方言 MLIR 与编排层 C++ |
| [后端](backend/index.md) | 通过 `BackendHandler` 做逐架构分派 |
| [调试](debug/index.md) | 把 IR lower 成可执行的 PyTorch 脚本用于数值校验 |

## 顶层主题

| 页面 | 内容 |
| ---- | ---- |
| [PTO 项目生态](00-ecosystem.md) | 多仓库工具链 —— PyPTO、PTOAS、pto-isa、simpler、pypto-lib —— 及其组合方式 |
| [编译性能剖析](01-compile-profiling.md) | 内建的编译流水线墙钟计时 |
| [错误处理](02-error-handling.md) | `CHECK` 与 `INTERNAL_CHECK`、PyPTO 异常类型、失败信息中的 IR 源码位置 |
| [日志](03-logging.md) | 两套相互独立的日志子系统，以及如何判断一条消息来自哪一套 |
| [运行时 DFX 开关](03-runtime-dfx.md) | 通过 `RunConfig` 暴露的五个运行时诊断子特性 |
| [重放已有的 `build_output`](03-runtime-replay.md) | 不重新编译，直接重跑、修改并重新测量已编译的构建目录 |
| [模拟器 Trace 清洗](04-simulator-trace-cleaning.md) | 把 MindStudio Insight 二进制 dump 转成可读 trace |
| [逐任务 Ring Sizing](05-runtime-ring-sizing.md) | `RunConfig` 上的三个 ring 尺寸覆盖项及其调优时机 |
| [持久化 L3 执行](06-persistent-l3.md) | 在多个已 prepare 的分布式程序间复用同一个 worker |
| [内存图](07-memory-map.md) | 把 pass dump 渲染成可交互的片上内存 HTML 图 |
| [编译与执行入口](08-entry-points.md) | 全部编译与执行入口、各自所属的层，以及何时该用哪一个 |
| [分布式算子](distributed_ops.md) | N6 分布式算子家族 —— 对集合通信与低层原语的类型化 DSL 访问 |
| [PTOAS 算子状态矩阵](ptoas-op-status.md) | 编译器当前会发射哪些 PTOAS 公开与兼容算子 |
| [FP4](fp4.md) | 逻辑 vs packed FP4、cast 策略，以及手写 `FP4E2M1X2` 指引 |

## 自动 PR 审查

`PR Agent` GitHub Actions 工作流会在非草稿 PR 创建、重新打开、更新或标记为
ready 时进行审查，也支持来自 fork 的 PR。仓库所有者、组织成员和协作者可以在
打开的 PR 下发布内容完全为 `/review` 的评论来触发审查。审查结果会更新到同一条 PR 评论中。

管理员在 **Settings → Secrets and variables → Actions** 中启用工作流：

1. 将 DeepSeek API Key 保存为仓库密钥（Secret）`OPENAI_KEY`。
2. 按需设置仓库变量 `PR_AGENT_API_BASE`（默认 `https://api.deepseek.com`）和
   `PR_AGENT_MODEL`（默认 `deepseek-flash`）。模型 ID 不需要 `openai/` 前缀；
   工作流会添加此前缀，以使用 OpenAI 兼容接口。审查的上下文预算为 64,000 token。
3. 工作流合并到默认分支后，将仓库变量 `PR_AGENT_ENABLED` 设置为 `true`。
   删除该变量或将其设为 `false` 可关闭审查。

PR 内容会发送到所配置的模型服务，并消耗 API 额度。工作流通过 GitHub API 读取
PR 数据，不检出 PR 代码。它使用工作流中定义的配置，仅启用 review，不会自动
改写 PR 描述、应用代码修改或批准合并。机器人触发的事件会被跳过；维护者可以在
机器人创建的 PR 下通过 `/review` 请求审查。

## 另请参阅

- [PTO ISA 参考](../reference/index.md) —— 后端所面向的硬件模型。
- [运行时文档](https://www.pypto.ai/simpler/) —— 执行已编译程序的调度器。
