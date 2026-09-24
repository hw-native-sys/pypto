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
令牌（token）仅提供给独立的发布任务（publishing job）。如果审查输出包含 Codex
凭据中的任何完整长字符串值，工作流会拒绝发布；正常评论也会明确标记为由不可信
PR 内容生成的自动化结果。

由 ChatGPT 管理（ChatGPT-managed）的 Codex 凭据通过可信的每周或手动维护任务
刷新。该任务不检出仓库，只在空的临时目录中运行，通过宿主机锁与审查任务串行，
并原地更新持久凭据。PR 审查容器只接收临时快照，绝不会挂载或写入持久凭据。

### 自动批准

设置 `CODEX_REVIEW_AUTO_APPROVE=true` 后，无问题的审查可以由
`github-actions[bot]` 提交 `APPROVE` 审批。仓库 Actions 设置也必须允许
GitHub Actions 创建和批准 PR。删除该变量或设为 `false` 可以单独关闭自动批准，
审查结果仍会发布。仓库原有的必需 CI 检查仍然生效。

工作流检出事件对应的精确 head SHA，通过 `codex exec --output-schema` 请求
结构化 JSON。Schema 和发布脚本取自受信任的 base 提交。发布器只有在完整审查返回
`pass`、发现列表为空、且 head 和 base 均未改变时才批准。无效、空白、超长、
相互矛盾或不完整的输出均不会批准。目标分支必须启用新提交后撤销旧审批的规则。
批准后再次核对提交；若发布期间发生更新，则撤销刚提交的审批。替代审查结果发布前
会先撤销本工作流之前的自动审批，不修改人工审批。

修改 `.github`、`.claude`、`.codex`、`.agents` 下的文件，或修改 `AGENTS.md`、
`CLAUDE.md`、`.gitmodules` 时，必须人工审查。重命名这些文件也不能绕过限制。
如果无法核实完整的变更文件列表，则不自动批准。

自动批准是 AI 判断，并非正确性证明；结构化输出不能消除提示注入和漏检风险。
允许 Actions 审批的设置对仓库中所有拥有 PR 写权限的工作流生效。现有隔离审查器内
的凭据快照和精确字符串泄露检测不能消除编码后的凭据泄露。公共仓库的审查基础设施
应优先采用隔离凭据的 API 代理。模型进程不会获得 GitHub 审批令牌。

工作流必须合入默认分支后才生效，因为 pull-request-target 使用受信任的 base
工作流。正式依赖自动批准作为合并条件前，应在受控 PR 上验证首次无问题审查，
核对实际 review 的 commit ID 和 `APPROVED` 状态。

## 另请参阅

- [PTO ISA 参考](../reference/index.md) —— 后端所面向的硬件模型。
- [运行时文档](https://www.pypto.ai/simpler/) —— 执行已编译程序的调度器。
