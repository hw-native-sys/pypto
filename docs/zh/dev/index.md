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

## 另请参阅

- [PTO ISA 参考](../reference/index.md) —— 后端所面向的硬件模型。
- [运行时文档](https://www.pypto.ai/simpler/) —— 执行已编译程序的调度器。

## 文档站点

<https://www.pypto.ai/pypto/> 从
[`hw-native-sys.github.io`](https://github.com/hw-native-sys/hw-native-sys.github.io)
继承布局、配色、项目切换和主题偏好。本仓库维护页面、导航、翻译及 API 插件。
`docs/theme-revision.txt` 用完整 Git 提交 SHA 固定公共主题版本。
CI 与本地构建使用同一版本；生成的站点包含自己的主题资源副本。

### 本地预览

在 PyPTO 仓库根目录运行以下命令，使用 Python 3.10 或更新版本。先获取主题，再安装文档依赖：

```bash
set -euo pipefail
git init .site-theme
git -C .site-theme fetch --depth 1 \
  https://github.com/hw-native-sys/hw-native-sys.github.io.git \
  "$(cat docs/theme-revision.txt)"
git -C .site-theme checkout --detach FETCH_HEAD

python -m venv .venv
source .venv/bin/activate
python -m pip install -r docs/requirements.txt
python tests/lint/check_docs_nav.py
python tests/lint/check_docs_en_zh_parity.py
mkdocs build --strict
mkdocs serve
```

预览地址为 <http://127.0.0.1:8000/>。`.site-theme/` 和虚拟环境均由 Git 忽略。
文档工具链静态解析 API 源码，无需编译 PyPTO，也无需 Ascend 设备。

### 升级或回滚主题

将 `docs/theme-revision.txt` 中的 SHA 替换为公共仓库的目标提交，然后重新运行上述获取、
检出、依赖安装及检查命令。提交版本变更前，检查中英文页面、API 参考、窄屏导航和深浅色模式。

`mkdocs.yml` 继承 `.site-theme/docs-theme/base.yml`；本地值覆盖公共值，列表会替换继承的列表。
项目导航和插件保留在本仓库，共用外观在公共仓库修改。仅更新公共仓库不会改变已部署的本站：
需要在本仓库合入主题版本变更才会发布。回滚时恢复之前的 SHA，再通过 Docs 工作流重新构建。
