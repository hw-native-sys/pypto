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

位置通过 diff 校验的问题会发布为行级审查评论，可以独立回复和标记解决。
位于 diff 之外、没有位置或位置无效的问题会保留在审查总结中。diff 之外的
已报告位置会链接到被审查的 head 或 merge-base 提交。同一提交上，本工作流
已经发布的相同行级评论不会重复发布。位置校验不影响批准策略：只要存在问题，
就不会自动批准。

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
由 `github-actions[bot]` 创建的 PR 只接收评论而不批准，因为 GitHub 不允许作者
批准自己的 PR。

工作流检出事件对应的精确 head SHA，通过 `codex exec --output-schema` 请求
结构化 JSON。Schema 和发布脚本取自提供工作流文件的提交（`github.workflow_sha`），
与 PR 的 base 版本独立。发布器只有在完整审查返回
`pass`、发现列表为空、且 head SHA、base SHA 和 base 分支名均未改变时才批准。
即使另一分支指向相同 SHA，修改目标分支也会使审查失效，因为该分支可能有不同的保护规则。
无效、空白、超长、相互矛盾或不完整的输出均不会批准。
目标分支必须启用至少需要一次批准、且新提交后撤销旧审批的规则。
目标分支还必须配置至少一项必需状态检查，并启用严格的分支最新状态要求。
当 base 分支前进时，此合并限制要求更新 PR head，从而撤销旧审批并重新触发
CI 和审查。缺少任一规则时，发布器只发表评论，不会批准。
批准后再次核对提交；若发布期间发生更新，则撤销刚提交的审批。替代审查结果发布前
会先撤销本工作流之前的自动审批，不修改人工审批。

修改 PR 的 base 分支也会在审查前触发审批撤销。独立的 GitHub 托管任务只运行
受信任工作流版本的代码，即使审查已禁用、PR 为草稿或由机器人修改目标分支，
也会撤销本工作流之前的审批。符合条件的新审查只有在撤销成功后才启动。
仅修改标题或正文不会启动审查，也不会取消已有审查。撤销任务使用独立的并发组，
新事件不会取消正在执行的撤销。GitHub 事件投递和任务调度
是异步的，因此撤销审批与修改目标分支并非原子操作。

修改 `.github`、`.claude`、`.codex`、`.agents` 下的文件，或修改 `AGENTS.md`、
`AGENTS.override.md`、`CLAUDE.md`、`.gitmodules` 时，必须人工审查。重命名这些文件也不能绕过限制。
如果无法核实完整的变更文件列表，则不自动批准。

自动批准是 AI 判断，并非正确性证明；结构化输出不能消除提示注入和漏检风险。
允许 Actions 审批的设置对仓库中所有拥有 PR 写权限的工作流生效。现有隔离审查器内
的凭据快照和精确字符串泄露检测不能消除编码后的凭据泄露。公共仓库的审查基础设施
应优先采用隔离凭据的 API 代理。模型进程不会获得 GitHub 审批令牌。

工作流必须合入默认分支后才生效，因为 pull-request-target 使用默认分支的工作流。
正式依赖自动批准作为合并条件前，应在受控 PR 上验证首次无问题审查，
核对实际 review 的 commit ID 和 `APPROVED` 状态。

## 另请参阅

- [PTO ISA 参考](../reference/index.md) —— 后端所面向的硬件模型。
- [运行时文档](https://www.pypto.ai/simpler/) —— 执行已编译程序的调度器。
