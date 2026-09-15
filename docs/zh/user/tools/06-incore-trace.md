# In-Core Trace

Cycle 级、分 pipe、一次一个 kernel —— 核到底逐条指令做了什么。

## 概念

三个视图处在不同粒度上，而这是最细的那个：

| 视图 | 粒度 | 页面 |
| ---- | ---- | ---- |
| 性能提示 | 编译器静态怀疑到的东西 | [工具总览](index.md#the-cheap-checks-first) |
| 泳道图 | 任务在整块芯片上如何被调度 | [运行期 DFX](04-runtime-dfx.md) |
| **In-core trace** | 哪条 pipe、哪条指令、多少 cycle | 本页 |

当泳道图显示一根很宽、两边没有间隙的条时就该用它 —— 调度没问题，kernel 本身就是代价。

它跑在 **Ascend 算子仿真器**（cycle-accurate camodel）上，不在设备上，也不在普通的 PyPTO 运行里。两步：一个 skill 负责采集，一个仓内工具负责清洗。

## 快速上手

采集，用 `pypto-user` 插件里的 `incore-profiling` skill：

```text
/incore-profiling --build-dir build_output/<case> --target a2a3
```

然后把原始 dump 清洗成可读的东西：

```bash
TRACE="build_output/<case>/kernel_insight_all_funcs_<ts>/funcs/<kernel>/collect/out"
python -m pypto.tools.clean_sim_trace "$TRACE"/OPPROF_* -o trace-out
```

在 [Perfetto UI](https://ui.perfetto.dev) 或 `chrome://tracing` 里打开 `trace-out/trace.clean.json`。

采集脚本随插件发布，不在本仓库里，所以没有可以直接运行的仓内路径。安装前要先注册 marketplace，否则插件名解析不出来：

```bash
claude plugin marketplace add hw-native-sys/pypto-skills
claude plugin install pypto-user@pypto-skills
```

它的前置条件是实打实的 —— 一个带 `ptoas/` kernel 的已构建用例、支持 TL 的 CANN、以及 `msopprof` worker —— 三者都会被预检，失败时尽早给出具体消息。

## 机制

### 为什么需要清洗这一步

仿真器每次 kernel 运行会写两个产物，而官方那个是有损的：

| 文件 | 内容 |
| ---- | ---- |
| `trace.json` | 仿真器自己导出的 Perfetto trace —— 只有 trace 事件 |
| `visualize_data.bin` | 完整的 MindStudio Insight 容器：trace 事件**加上**每指令指标、源码映射和其他数据块 |

`clean_sim_trace` 直接读二进制，因此恢复了官方导出丢掉的每指令指标。它同时做了去噪：直接打开 `trace.json` 是读不下去的，因为 `SET_FLAG` / `WAIT_FLAG` 切片和标量地址运算把真正的流水线埋掉了。

### 命令行

```bash
python -m pypto.tools.clean_sim_trace <path> [-o OUTPUT_DIR] [--keep-scalar] [--raw-metrics] [--no-copy-raw]
```

`<path>` 是一个 `visualize_data.bin` 文件或一个 `OPPROF_*` 目录（工具会在里面找 `simulator/visualize_data.bin`）。

| 产物 | 内容 |
| ---- | ---- |
| `trace.clean.json` | 重建后的 Chrome Trace Event JSON |
| `instr_metrics.json` | 每核的指令记录 —— `address`、`pipe`、`cycles`、`vector_utilization_percentage`…… |
| `raw_simulator/` | 原始 dump 的副本，使 `-o` 目标目录自包含（`--no-copy-raw` 跳过） |

| 参数 | 效果 |
| ---- | ---- |
| `--keep-scalar` | 保留 `SCALAR` setup 泳道（默认丢弃） |
| `--raw-metrics` | 原样 dump `API_INSTR` 块，而不重整 |
| `--no-copy-raw` | 不把原始二进制 trace 复制进输出目录 |

### 清洗做了什么

1. **泳道筛选** —— 保留流水线泳道，丢掉 `CACHEMISS` / `FLOWCTRL` / `ALL`，以及默认丢掉的 `SCALAR`。
2. **事件过滤** —— 保留完整指令事件；丢掉 `SET_FLAG` / `WAIT_FLAG` / `BAR` 切片。
3. **泳道排序** —— 输出元数据，使泳道按数据流顺序呈现：**MTE2 → MTE1 → CUBE → VECTOR → FIXPIPE → MTE3**。
4. **子泳道打包** —— 一条 pipe 上被软件流水的指令常常同时有好几条在飞，而且只部分重叠。Chrome trace 同一泳道上的事件必须互不相交或严格嵌套，所以重叠的指令被拆进 `MTE1`、`MTE1#1`…… —— 每条同时存活的指令占一行。**没有这一步，可见的流水深度会塌缩到大约 2**，看起来像「没有重叠」，而实际上重叠得很充分。
5. **同步变箭头** —— 每对 `SET_FLAG` → `WAIT_FLAG` 变成一条流箭头，重新锚定到真正的生产和消费指令上。
6. **时间戳** —— 原样保留，所以清洗后的 trace 与原始 `trace.json` 能对齐。

### 怎么读

分 pipe 的 cycle 分解就是「这个 kernel 到底在干什么」的答案：

| 形态 | 含义 | 去哪 |
| ---- | ---- | ---- |
| 几乎全是 **MTE2** | 搬运受限 | [双缓冲](../performance/04-incore.md#double-buffer) |
| 几乎全是 **VECTOR**，利用率低 | 形状不适配向量单元 | [调优 InCore 函数](../performance/04-incore.md) |
| matmul kernel 上 **CUBE = 0** | trace 退化了，不是 matmul 免费 | 见下方注意事项 |

## 边界情况

| 现象 | 原因 | 处理 |
| ---- | ---- | ---- |
| **trace 近乎为空，`CUBE`/`VECTOR` 约 0 cycle** | 数据相关的 kernel —— 循环次数或工作表大小从输入张量读取 —— 在 skill 的零值输入下被剖析 | 这是合成输入的假象，不是 kernel 快；按 skill 的 caveats 接入全尺寸真实中间量 |
| **四级流水线看起来只有 2 层深** | 读的是原始 `trace.json` 而不是清洗后的 | 用 `trace.clean.json`；子泳道打包才是保住深度的那一步 |
| **没有 `instr_metrics.json`** | 从 `trace.json` 读的，它不带 `API_INSTR` | 把工具指向 `visualize_data.bin` / `OPPROF_*` 目录 |
| **`corrupt block at offset …`** | dump 被截断或只写了一半 | 重新采集；那次运行多半写到一半就死了 |
| **skill 在预检阶段失败** | 缺 `ptoas/` kernel、CANN 不支持 TL，或没有 `msopprof` | 消息会指名是哪一项；先把用例构建出来 |

## 参见

- [调优 InCore 函数](../performance/04-incore.md) —— 知道是哪条 pipe 之后该改什么。
- [运行期 DFX](04-runtime-dfx.md) —— 上一层；PMU 用更粗的粒度回答同一个问题，而且不需要仿真器。
- [回放一次构建](05-replay.md) —— 手改 kernel，然后重新剖析。
- [Simulator Trace Cleaning](../../dev/04-simulator-trace-cleaning.md) —— `visualize_data.bin` 的块格式与完整重建规则。
