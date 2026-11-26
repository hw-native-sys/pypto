# 分块 Tile 尺寸校验

PyPTO 发射的每一条 `pto.alloc_tile` 都会按 PTOAS 将要检查的分块网格先行校验。
PTO 以「块」为单位寻址分块 tile，因此*物理*尺寸不是整数个块的 tile 根本没有地址。

该规则与 PTOAS 的 `verifyBoxedTileLayout` 完全一致：

| 布局 | 块尺寸（行 x 列） |
| ---- | ----------------- |
| fractal 1024（`Acc`） | 16 x 16 |
| fractal 512，`slayout = row_major`（`Mat` / `Left`） | 16 x (32 / sizeof(dtype)) |
| fractal 512，`slayout = col_major`（`Right`，转置对偶） | (32 / sizeof(dtype)) x 16 |
| `slayout = none_box` | 非分块，无此约束 |

并保留 PTOAS 自身的豁免：*行*方向的规则对 `Vec` 以及单行 tile（此时 NZ 映射退化）
跳过，而列方向的规则始终生效。MX scale fractal 与亚字节载体交由 PTOAS 自行诊断。

**为什么放在这里而不是交给 PTOAS。** PTOAS 会拒绝同样的形状，但它的报错只提及自身内部
概念，也不给出修复方式：

```text
'pto.alloc_tile' op expects result boxed tile rows to be a multiple of innerRows (16), but got 100
```

在发射点报错则能同时给出 tile、出问题的轴、需要达到的尺寸，以及达到它的方式：

```text
a Mat tile of physical shape [100, 128] and dtype fp16 must be a whole number of
16x16 fractal boxes, but its row extent 100 is not a multiple of 16. ...
allocate 112 on that axis and declare 100 as the tile's valid_shape ...
```

`ComputeAllocTileFields` 是所有分配的唯一收口——逐变量声明、被提升出来的
`extra_alloc_tiles`、以及控制流路径都经过它——因此校验看到的正是最终发射的内容，不会与之
漂移。张量层的 `pl.matmul` / `pl.matmul_acc` 不会因 *M 轴*触发它：M 轴已由
[`ConvertTensorToTileOps`](../passes/13-convert_tensor_to_tile_ops.md#cube-operand-m-axis-boxing)
自动对齐；仍需用户自行保证的是 `K` 与 `N`。
