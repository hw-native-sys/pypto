# LegalizeTileCast Pass

把目标 profile（A5 / A2A3）上 `pto.tcvt` **不支持** 的 `tile.cast` `(src, dst)` 对，展开为最短的原生 cast 链，避免静默发出非法 `tcvt`。

## 概述

`LegalizeTileCast` 是函数级 Pass。对每条 `var = tile.cast(...)`：

1. 查当前 backend 的原生邻接表（来自 pto-isa `tcvt` Supported Conversions）。
2. 已原生：原样保留（含可被 `AutoTileMatmulL0` FIXPIPE-fold 的 `FP32→BF16/FP16` + `rint`）。
3. 非原生：在邻接图上 BFS 求最短路径；等长路径优先「同字节转浮点 → 再调宽度」。

典型结果（A5）：

| 用户 Cast | 分解 |
|-----------|------|
| INT32→FP16 | INT32→FP32 → FP32→FP16 |
| FP16→BF16 | FP16→FP32 → FP32→BF16 |

搜不到路径则硬失败（带 src/dst/arch）。

**Requires / Produces / Invalidates**：无（空 `PassProperties`）。

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
