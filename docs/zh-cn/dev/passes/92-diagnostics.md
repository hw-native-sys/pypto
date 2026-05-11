# 诊断系统：警告与性能提示

Pass 流水线统一的建议性诊断通道。通过同一套注册表、Instrument 与输出路径，承载警告（疑似用户错误）和性能提示（建议性调优提示）。

## 概述

| 组件 | 作用 |
| ---- | ---- |
| **`Diagnostic` 结构体** | 携带 severity（`Error` / `Warning` / `PerfHint`）、`rule_name`、`error_code`、`hint_code`、消息、span。 |
| **`DiagnosticCheck` 枚举** | 标识具体的检查项（如 `UnusedVariable`、`TileInnermostDimGranularity`）。 |
| **`DiagnosticCheckRegistry`** | 将检查项映射到 verifier 工厂；每次注册都声明 severity、phase 和 hint 码。 |
| **`DiagnosticInstrument`** | 运行已注册检查并分发输出的 `PassInstrument`。 |
| **`DiagnosticPhase`** | 检查触发时机：`PrePipeline`、`PostPass` 或 `PostPipeline`。每个检查独立声明。 |

Severity 与 phase 解耦：`Warning` 可以在 `PrePipeline` 触发，`PerfHint` 可以在 `PostPipeline` 触发——按检查注册，而非按 severity。

## Severity 等级

| Severity | 何时使用 | 输出 | 抑制方式 |
| -------- | -------- | ---- | -------- |
| `Error` | IR 不合法 | 抛出 `VerificationError` | 不可抑制 |
| `Warning` | 疑似用户错误或 pass bug | 完整输出至 stderr（`LOG_WARN`） | `disabled_diagnostics` 集合 |
| `PerfHint` | 建议性调优提示 | 上下文中存在 `ReportInstrument` 时（经 `compile()` / `pl.jit` 时恒为真）每条 hint 写入 `${ReportInstrument.output_dir}/perf_hints.log`，stderr 仅打印一行指向该文件的 `LOG_INFO` 摘要；若无 `ReportInstrument`，则每条 hint 完整输出至 stderr（`LOG_INFO`）。 | `disabled_diagnostics` 集合 |

`PYPTO_LOG_LEVEL` release 默认值为 `INFO`，因此 `[perf_hint] N hints …` 摘要行（无 `ReportInstrument` 时为完整的 `[perf_hint PH…] …` 行）开箱可见。设置 `PYPTO_LOG_LEVEL=warn` 可在 stderr 上静音性能提示。上下文中存在 `ReportInstrument` 时，无论日志级别如何逐条详情仍写入 `perf_hints.log`（文件输出独立）；若无 `ReportInstrument` 则没有该文件，静音 stderr 会彻底丢弃性能提示。

## 触发流程

```text
PassPipeline::Run(program)
 ├─ phase != None：运行 PrePipeline 检查  → EmitDiagnostics
 ├─ 对每个 pass：
 │    ├─ 执行 pass
 │    ├─ phase != None：运行 PostPass 检查 → EmitDiagnostics
 ├─ phase != None：运行 PostPipeline 检查  → EmitDiagnostics
 └─ ctx.RunAfterPipeline(program)         （instrument 钩子）
```

`EmitDiagnostics` 将 Warning 以完整形式经 `LOG_WARN` 输出。对 PerfHint：上下文中存在 `ReportInstrument` 时把每条 hint 附加到 `perf_hints.log`，并向 stderr 输出一行 `LOG_INFO` 摘要（`[perf_hint] N hints across M sites; see <path>`）；否则每条 hint 完整经 `LOG_INFO` 输出。

## 注册新的检查项

```cpp
// 1. 实现 PropertyVerifier 子类（src/ir/verifier/...）
class MyCheck : public PropertyVerifier {
 public:
  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diags) override;
  std::string GetName() const override { return "MyCheck"; }
};
PropertyVerifierPtr CreateMyCheckVerifier() { return std::make_shared<MyCheck>(); }

// 2. 添加枚举值（include/pypto/ir/verifier/diagnostic_check_registry.h）
enum class DiagnosticCheck : uint32_t { ..., MyCheck = N };

// 3. 在 DiagnosticCheckRegistry::DiagnosticCheckRegistry() 中注册
Register(DiagnosticCheck::MyCheck,
         DiagnosticSeverity::PerfHint,
         DiagnosticPhase::PostPipeline,
         "PHnnn",
         CreateMyCheckVerifier);
```

注册表会将注册时声明的 severity 和 hint 码盖到 verifier 产生的每个 diagnostic 上。

## 性能提示（issue #1180）

性能提示是对可能未充分利用目标硬件的代码模式的最佳努力建议。默认在 `DiagnosticPhase::PostPipeline` 启用，在 IR 完全 lower 后运行。

### 各 backend 阈值

`BackendHandler` 暴露：

| 方法 | Ascend910B | Ascend950 |
| ---- | ---------- | --------- |
| `GetGmAccessGranularityBytes()` | 512 | 128 |
| `GetL2CacheLineBytes()` | 512 | 512 |
| `GetRecommendedInnermostDimBytes()` | 512 | 128 |

新增 backend 时实现这些虚函数；perf-hint 检查通过 `PassContext::Current()->GetBackendHandler()` 读取。

### 第一项检查：`TileInnermostDimGranularity` (PH001)

检查每个 `tile.load` / `tile.store` 操作。当最内层维度的字节数（`shape[-1] * sizeof(dtype)`）低于 `GetRecommendedInnermostDimBytes()` 时，对每个 op 发一条 diagnostic，指向用户源代码 span。上下文中存在 `ReportInstrument` 时全部写入 `perf_hints.log`，stderr 仅见摘要行；否则每条 hint 输出到 stderr。

示例：stderr 摘要 + `perf_hints.log` 内容（`examples/kernels/08_assemble.py`，Ascend950）：

```text
# stderr：
[perf_hint] 2 hints across 2 sites; see /tmp/build/perf_hints.log

# /tmp/build/perf_hints.log：
[perf_hint PH001] TileInnermostDimGranularity: tile.load has innermost dim = 64B; recommended >= 128B for backend a5 (L2 cache line = 512B). Consider increasing tile shape on the innermost axis. at examples/kernels/08_assemble.py:60:4
[perf_hint PH001] TileInnermostDimGranularity: tile.store has innermost dim = 64B; recommended >= 128B for backend a5 (L2 cache line = 512B). Consider increasing tile shape on the innermost axis. at examples/kernels/08_assemble.py:60:4
```

## 用户面 API

```python
# 无 report instrument：每条 perf hint 在 pipeline 末尾完整输出到 stderr
with passes.PassContext([]):
    pipeline.run(program)

# 有 report instrument（compile() / pl.jit 的默认行为）：完整详情写入与其他 report
# 同目录的 perf_hints.log，stderr 仅得到一行摘要
with passes.PassContext([passes.ReportInstrument("/tmp/build")]):
    pipeline.run(program)
# → /tmp/build/perf_hints.log （stderr："[perf_hint] N hints across M sites; see …"）

# 抑制单个 hint
disabled = passes.DiagnosticCheckSet()
disabled.insert(passes.DiagnosticCheck.TileInnermostDimGranularity)
with passes.PassContext([], disabled_diagnostics=disabled):
    pipeline.run(program)

# 关闭整个通道
with passes.PassContext([], diagnostic_phase=passes.DiagnosticPhase.NONE):
    pipeline.run(program)
```

`compile()` 与 `run()` 通过 `diagnostic_phase` / `disabled_diagnostics` 关键字参数接收同样的配置。

## 环境变量

| 变量 | 效果 | 默认 |
| ---- | ---- | ---- |
| `PYPTO_LOG_LEVEL` | stderr 输出阈值（`debug`/`info`/`warn`/`error`/`fatal`/`event`/`none`） | `info`（release）/ `debug`（非 release） |
| `PYPTO_WARNING_LEVEL` | 默认 `DiagnosticPhase`（`none`/`pre_pipeline`/`post_pass`/`post_pipeline`） | `pre_pipeline` |
| `PYPTO_VERIFY_LEVEL` | 校验级别——与诊断系统正交 | `basic` |
