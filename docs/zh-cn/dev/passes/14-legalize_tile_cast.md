# LegalizeTileCast Pass

把目标 profile（A5 / A2A3）上 `pto.tcvt` **不支持** 的 `tile.cast` `(src, dst)` 对，展开为最短的原生 cast 链，避免静默发出非法 `tcvt`。

## 概述

`LegalizeTileCast` 是函数级 Pass。对每条 `var = tile.cast(...)`：

1. 通过 `GetTcvtAdjacency()` 向当前 `BackendHandler` 取原生转换表（转录自 pto-isa `tcvt`
   Supported Conversions）。本 Pass 自身不含任何架构知识，新增后端只需提供自己的表，
   无需改动此处；未配置 backend 时本 Pass 不做任何事。
2. 已原生：原样保留（含可被 `AutoTileMatmulL0` FIXPIPE-fold 的 `FP32→BF16/FP16` + `rint`）。
3. 非原生：在邻接图上 BFS 求最短路径；等长路径优先「同字节转浮点 → 再调宽度」。

典型结果（A5）：

| 用户 Cast | 分解 |
| --------- | ---- |
| INT32→FP16 | INT32→FP32 → FP32→FP16 |
| FP16→BF16 | FP16→FP32 → FP32→BF16 |

搜不到路径则硬失败（带 src/dst/arch）。

**Requires / Produces / Invalidates**：无（空 `PassProperties`）。

## 原生 cast 与展开链

`pl.cast` 并不总是编译成一条指令。某个 `(src, dst)` 组合究竟是一条硬件
`pto.tcvt`，还是被展开成多跳链，完全取决于目标架构，而两者在性能和数值上都有
差别。下表按架构列出哪些组合是原生的、哪些由本 Pass 展开，`(n)` 为发射的
`tcvt` 指令条数。

**开销**：展开链每一跳都要对整个 tile 发一条 `tcvt`，并额外占用一块中转 tile
（仍受常规的 buffer 复用约束）。因此同样形状下，3 跳链的向量工作量约为原生
cast 的三倍。在向量受限的 kernel 里这是实打实的代价——若热点循环里出现了意料之
外的链，可考虑换一条等价的 dtype 路径避开它。

**数值**：当每个中转类型都能精确表示*落在目标值域内*的源值时，整条链与直接转换
逐位相同，只有最后一跳发生舍入。`INT32 -> FP32 -> FP16` 属于这一类：FP16 在 65504
以上饱和，而该范围内的整数在 FP32 中都是精确的，因此 FP32 这一跳不会舍入。

当中转类型会先舍入时，整条链发生双重舍入，结果可能与直接舍入相差目标类型的
1 个 ULP：

| 链路 | 双重舍入的原因 | 实测 |
| ---- | -------------- | ---- |
| `INT32 -> FP32 -> BF16` | BF16 值域覆盖整个 INT32，但 FP32 只有 24 位有效数字，因此超过 2^24 的输入在第一跳就被舍入 | 20 万均匀 INT32 中有 3 个相差 1 个 BF16 ULP |
| `FP32 -> FP16 -> INT8` | FP16 在取整跳之前先舍入一次 | 仅边界值 |

这并不是相对更优方案的退化：这些组合在 ISA 上压根没有直连转换，展开链是唯一可行
的下降方式。它同样与参考实现一致——`torch` 自身的 `int32 -> bfloat16` 在 200 万个
均匀 INT32 样本上与该链完全相同，因为 torch 也是经 fp32 中转的。

**如何确认自己的 kernel 走了哪条**：每一跳都会在生成的 MLIR 里出现一条 `pto.tcvt`：

```bash
grep -n 'pto.tcvt' build_output/<case>/ptoas/<kernel>.pto
```

### Ascend950 (a5)

| 源类型 | 原生（1 条指令） | 展开链（n 条指令） |
| ------ | ---------------- | ------------------ |
| `bf16` | `fp16`, `fp32`, `fp4`, `int32` | `fp8e4m3`(2), `fp8e5m2`(2), `hf8`(2), `int16`(2), `int8`(2), `uint16`(2), `uint8`(2) |
| `fp16` | `fp32`, `hf8`, `int16`, `int32`, `int8`, `uint8` | `bf16`(2), `fp8e4m3`(2), `fp8e5m2`(2), `uint16`(2), `fp4`(3) |
| `fp32` | `bf16`, `fp16`, `fp8e4m3`, `fp8e5m2`, `hf8`, `int16`, `int32`, `int64` | `fp4`(2), `int8`(2), `uint16`(2), `uint8`(2) |
| `fp4` | `bf16` | `fp8e4m3`(3), `fp8e5m2`(3), `hf8`(3), `int8`(3), `uint8`(3) |
| `fp8e4m3` | `fp32` | `bf16`(2), `fp16`(2), `fp8e5m2`(2), `hf8`(2), `int16`(2), `fp4`(3), `int8`(3), `uint16`(3), `uint8`(3) |
| `fp8e5m2` | `fp32` | `bf16`(2), `fp16`(2), `fp8e4m3`(2), `hf8`(2), `int16`(2), `fp4`(3), `int8`(3), `uint16`(3), `uint8`(3) |
| `hf8` | `fp32` | `bf16`(2), `fp16`(2), `fp8e4m3`(2), `fp8e5m2`(2), `int16`(2), `fp4`(3), `int8`(3), `uint16`(3), `uint8`(3) |
| `int16` | `fp16`, `fp32`, `int32`, `uint32`, `uint8` | `bf16`(2), `fp8e4m3`(2), `fp8e5m2`(2), `hf8`(2), `int8`(2), `uint16`(2), `fp4`(3) |
| `int32` | `fp32`, `int16`, `int64`, `uint16`, `uint8` | `bf16`(2), `fp16`(2), `fp8e4m3`(2), `fp8e5m2`(2), `hf8`(2), `fp4`(3), `int8`(3) |
| `int64` | `fp32`, `int32` | `bf16`(2), `fp16`(2), `fp8e4m3`(2), `fp8e5m2`(2), `hf8`(2), `int16`(2), `uint16`(2), `uint8`(2), `fp4`(3), `int8`(3) |
| `int8` | `fp16`, `int16`, `int32` | `hf8`(2), `uint16`(2), `uint8`(2), `fp8e4m3`(3), `fp8e5m2`(3), `fp4`(4) |
| `uint32` | `int16`, `uint16`, `uint8` | `int8`(3) |
| `uint8` | `fp16`, `uint16` | `hf8`(2), `int8`(2), `fp8e4m3`(3), `fp8e5m2`(3), `fp4`(4) |

### Ascend910B (a2a3)

| 源类型 | 原生（1 条指令） | 展开链（n 条指令） |
| ------ | ---------------- | ------------------ |
| `bf16` | `fp32`, `int32` | `fp16`(2), `int16`(2), `int4`(3), `int8`(3), `uint8`(3) |
| `fp16` | `fp32`, `int16`, `int32`, `int4`, `int8`, `uint8` | `bf16`(2) |
| `fp32` | `bf16`, `fp16`, `int16`, `int32`, `int64` | `int4`(2), `int8`(2), `uint8`(2) |
| `int16` | `fp16`, `fp32` | `bf16`(2), `int4`(2), `int8`(2), `uint8`(2) |
| `int32` | `fp16`, `fp32`, `int16`, `int64` | `bf16`(2), `int4`(2), `int8`(2), `uint8`(2) |
| `int4` | `fp16` | `int8`(2), `uint8`(2) |
| `int64` | `fp32`, `int32` | `bf16`(2), `fp16`(2), `int16`(2), `int4`(3), `int8`(3), `uint8`(3) |
| `int8` | `fp16` | `int4`(2), `uint8`(2) |
| `uint8` | `fp16` | `int4`(2), `int8`(2) |

两列都没有出现的组合会被拒绝：本 Pass 会报出 src、dst 和 arch，而不是生成有损的链。

## 运行时机

Default 流水线：

```text
lower_composite_ops → flatten_tile_nd_to_2d → legalize_tile_cast → auto_tile_matmul_l0
```

放在 Flatten 之后以覆盖其新插入的 cast；放在 MatmulL0 之前以免拆开本可 fold 的原生降精度 cast。

## API

| C++ | Python |
| --- | ------ |
| `pass::LegalizeTileCast()` | `passes.legalize_tile_cast()` |
