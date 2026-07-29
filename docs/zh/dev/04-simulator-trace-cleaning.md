# 仿真器 Trace 清洗

`clean_sim_trace` 将算子仿真器生成的 MindStudio Insight 二进制文件
（`visualize_data.bin`）转换为去噪的、可在 Perfetto 中查看的 AI Core 流水线 trace
（Chrome Trace Event JSON）。

仿真器每次 kernel 运行都会生成一个 `OPPROF_*/simulator/` 目录，其中包含两个 profiling 产物：

- `trace.json` —— 仿真器**官方**生成的 Perfetto/Chrome trace。
- `visualize_data.bin` —— 供 MindStudio Insight 使用的二进制容器。

**官方的 `trace.json` 是有损的（lossy）—— 它携带的信息严格少于
`visualize_data.bin`。** 该二进制容器保存了完整的 MindStudio Insight 数据块：除了
trace 事件，还包括逐指令指标（`API_INSTR`）、源码映射以及其它细节块；而 `trace.json`
只导出 trace 事件。因此 `clean_sim_trace` 直接读取信息更丰富的二进制文件，恢复出官方
Perfetto 导出所丢弃的逐指令指标（写入 `instr_metrics.json` 旁车文件）。

它同时也对 trace 本身去噪：直接在 Perfetto UI 中打开 `trace.json` 很难阅读，因为
`SET_FLAG` / `WAIT_FLAG` 同步切片和标量地址运算指令淹没了真正的 AI Core 流水线。
`clean_sim_trace` 会将其重建为一个干净的流水线视图（见*重建规则*）。

## 生成 dump

本工具消费的 `OPPROF_*/simulator/visualize_data.bin` 由 Ascend `msprof op
simulator`（cycle-accurate 的 camodel）生成，普通的 PyPTO 运行不会产生它。
**`incore-profiling` skill** 会按 kernel 驱动它——对 `build_output/<case>/` 中的每个
PTOAS kernel 生成独立 testcase，用 `ccec`/`bisheng` 编译，并运行该 op-simulator：

```bash
python .claude/skills/incore-profiling/incore_profile.py \
  --build-dir build_output/<case> --func <kernel> --target a2a3
```

它会写出 `.../kernel_insight_all_funcs_<ts>/funcs/<kernel>/collect/out/OPPROF_*/`；
将 `clean_sim_trace` 指向该 `OPPROF_*` 目录即可。

> **数据相关（data-dependent）的 kernel**——其循环次数或 work-table 大小从输入张量读取
> ——在 skill 自动生成的零值输入下会得到一个近乎空的 trace（`CUBE`/`VECTOR` 仅记录
> ~0 cycles）。这是合成输入造成的假象，而非 kernel 真的很快；如何注入全尺寸真实
> 中间张量，参见该 skill 的 *Caveats*。

## 用法

```bash
python -m pypto.tools.clean_sim_trace <path> [-o OUTPUT_DIR] [--keep-scalar] [--raw-metrics] [--no-copy-raw]
```

`<path>` 是一个 `visualize_data.bin` 文件或一个 `OPPROF_*` 目录（工具会在其中
定位 `simulator/visualize_data.bin`）。会在输入旁边（或 `-o` 指定的目录）写出两个文件。
当 `-o` 指向一个独立的目标目录时，原始二进制 trace 也会一并拷贝过去，使该目录自包含：

| 输出 | 内容 |
| ---- | ---- |
| `trace.clean.json` | 重建后的 Chrome Trace Event JSON，可在 `chrome://tracing` 与 Perfetto UI 打开 |
| `instr_metrics.json` | 来自 `API_INSTR` 块的逐核指令指标 |
| `raw_simulator/` | 原始 `visualize_data.bin` 与逐核 trace 数据的拷贝（仅在使用 `-o` 时；源 `OPPROF_*` 目录保持不变） |

| 选项 | 作用 |
| ---- | ---- |
| `--keep-scalar` | 保留 `SCALAR` 准备车道（默认丢弃） |
| `--raw-metrics` | 原样输出 `API_INSTR` 块，而不做重塑 |
| `--no-copy-raw` | 跳过将原始二进制 trace 拷贝到 `<output-dir>/raw_simulator/` |

## `visualize_data.bin` 格式

该文件是一串带长度前缀的数据块。每个块有一个 12 字节、4 字节对齐的头部:

| 偏移 | 大小 | 字段 | 含义 |
| ---- | ---- | ---- | ---- |
| 0 | `uint64` LE | `contentSize` | 负载长度，含尾部填充 |
| 8 | `uint8` | `type` | 块类型（`2` = TRACE，`4` = API_INSTR，`1` = SOURCE，...） |
| 9 | `uint8` | `paddingLength` | 用于 4 字节对齐的尾部补零字节数 |
| 10 | `uint8` | `instrVersion` | API_INSTR 版本标识 |
| 11 | `uint8` | `reserve` | 恒为 `0x5a`（二进制格式幻数） |

每个块的负载都是纯 JSON。`SOURCE` 块在负载前有一个固定 4096 字节的文件路径。
工具只消费 `TRACE` 与 `API_INSTR` 块，其它块类型会被跳过。

## 重建规则

1. **车道选择** —— 保留流水线车道（`MTE2`、`MTE1`、`CUBE`、`VECTOR`、
   `FIXPIPE`、`MTE3`）；丢弃 `CACHEMISS`、`FLOWCTRL`、`ALL`。`SCALAR` 车道
   默认丢弃（`--keep-scalar` 可恢复）。
2. **事件过滤** —— 保留 `X`（完整）指令事件；丢弃 `SET_FLAG` / `WAIT_FLAG` /
   `BAR` 切片。
3. **车道排序** —— 发出 `process_*` / `thread_*` 元数据，使核与流水线车道按
   数据流顺序（加载 -> 计算 -> 存储）显示。
4. **子车道拆分（sub-lane）** —— 软件流水的指令在同一 pipe 上常常同时有多条在飞，
   且只是*部分*重叠。Chrome trace 的 `X` 事件在同一 `tid` 上必须互不相交或严格嵌套，
   因此把每个 pipe 中重叠的指令贪心拆分到多个子车道（`MTE1`、`MTE1#1`……）——
   每条同时存活的指令占一行——在查看器中保留真实的流水线深度（否则深度会塌缩到约 2）。
5. **同步转为箭头** —— 每个 `SET_FLAG` -> `WAIT_FLAG` 对变成一条流向箭头，
   从生产指令重新锚定到消费指令（落在其所在的子车道上）。
6. **着色** —— 切片按流水线车道重新着色。
7. **时间戳** —— 原样保留，使清洗后的 trace 与原始 `trace.json` 对齐。

## 指标旁车文件

`instr_metrics.json` 重塑 `API_INSTR` 块：原始块中每个按核索引的指标数组，
会被展平为逐核的指令记录列表（`address`、`pipe`、`cycles`、
`vector_utilization_percentage` 等）。原始的 `Instructions Dtype` 映射保留在
`column_types` 下。
