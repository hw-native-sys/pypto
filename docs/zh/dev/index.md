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

`Codex Review` GitHub Actions 工作流会在非草稿 PR 创建、重新打开、更新或标记为
可审查（ready for review）时进行审查。工作流定义取自主分支的受信任版本，将 PR
head 作为不可信输入检出，并由独立的 GitHub 托管任务（GitHub-hosted job）发布结果。
自动生成的审查结果仅供参考；工作流不会批准或合并 PR。

管理员通过仓库 Actions 变量 `CODEX_REVIEW_ENABLED` 启用或禁用审查。将其设为
`true` 即可启用；删除该变量或设为其他值可以立即禁用。

审查任务（review job）需要标签为 `Linux`、`ARM64` 和 `cpu-codex` 的专用自托管
运行器（self-hosted runner）。以下资源由宿主机管理，不取自 PR：

- `/home/ci-runner/.codex-ci/auth.json`，仅 runner 账号可读
- 本地镜像仓库中按 digest 固定的 review 和 proxy 镜像
- `pypto-codex-egress` Docker 网络
- 监听 relay 端口 `17895` 且健康的 `pypto-codex-proxy-relay` 容器
- 仅监听 loopback 的 mixed proxy `127.0.0.1:7895`

Codex 在只读容器中以 UID/GID `1002:1003` 运行；仓库以只读方式挂载，同时移除
能力（capability）、实施资源限制并使用内部 Docker 网络。拥有 PR 写权限的 GitHub
令牌（token）仅提供给独立的评论任务（comment job）。如果审查输出包含 Codex
凭据中的任何完整长字符串值，工作流会拒绝发布；正常评论也会明确标记为由不可信
PR 内容生成的自动化结果。

由 ChatGPT 管理（ChatGPT-managed）的 Codex 凭据通过可信的每周或手动维护任务
刷新。该任务不检出仓库，只在空的临时目录中运行，通过宿主机锁与审查任务串行，
并原地更新持久凭据。PR 审查容器只接收临时快照，绝不会挂载或写入持久凭据。

## 另请参阅

- [PTO ISA 参考](../reference/index.md) —— 后端所面向的硬件模型。
- [运行时文档](https://www.pypto.ai/simpler/) —— 执行已编译程序的调度器。
