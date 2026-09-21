# FoldFixpipeAccEpilogue

把向量端的反量化 / ReLU epilogue 折叠进 cube 的 `Acc → GM` 写回，使带 scale 和激活的
matmul 仍然是一个纯 cube kernel。

## 为什么这值得一个 pass，而不只是微优化 {#why-a-pass}

写成 `store(maximum(cast(acc, FP32) * scale, 0))` 的代价远不止那三条向量指令。
`InferTileMemorySpace` 会把 cast、muls、maximums 全部定到 `Mem.Vec`，于是一个 cube
kernel 被撕成 AIC 与 AIV 两个函数，中间夹一次跨核往返：

```text
kernel_aic:  tile.matmul -> tpush_to_aiv -> ... -> tpop_from_aiv -> tile.store
kernel_aiv:  tpop_from_aic -> cast -> muls -> maximums -> cast -> tpush_to_aic
```

这一对要付出 `tpush_to_aiv` / `tpop_from_aic` / `tpush_to_aic` / `tpop_from_aiv`
一整套、两个 `tfree`、`aic_initialize_pipe`，外加 C2V 环用的 GM slot buffer。在 a2a3
上 `[128, 128]` 这个形状，环的默认深度加上向量 tile **直接把 Vec 空间撑爆**，kernel
根本编译不过：

```text
ValueError: Vec buffer usage (196608 bytes) exceeds platform limit (188416 bytes).
The first 131072 bytes of that space are reserved by system.reserve_buffer — this
is the cross-core pipe ring.
```

而 fix-pipe 在排空 L0C 的同时就能完成乘法与激活。折叠之后，同一个 kernel 变成单个
AIC 函数，跨核算子数为 0。

## 位置 {#position}

在 `CanonicalizeTileSlice`（19）与 `InferTileMemorySpace`（21）之间。**必须**在后者
之前：正是那一趟把向量链定到 `Mem.Vec` 并催生跨核搬运，折叠晚于它就没有意义了。
也正因为跑在它之前，此时**内存空间尚未解析**，所以本 pass 通过**算子身份**
（`tile.matmul`、`tile.matmul_acc`、`tile.matmul_bias` 的结果天然在 Acc）而不是
`memory_space_` 来认定累加器。

## 识别的链 {#recognised-chain}

每一环都必须是前一环的**唯一**消费者，且是顶层赋值语句，这样折叠就绝不会删掉别人
还在读的值。除了至少要有一环之外，所有环都是可选的：

```text
acc = tile.matmul(...)                    # 天然在 Acc
  [ t = tile.maximums(acc, 0.0) ]         # -> pre_relu=True   （形态 A）
  [ t = tile.cast(t, FP32) ]              # 被吸收：FIXPIPE 本来就在 FP32 里乘
  [ t = tile.muls(t, <const s>) ]         # -> pre_quant=s
  [ t = tile.maximums(t, 0.0) ]           # -> pre_relu=True   （形态 B）
  [ t = tile.cast(t, DST) ]               # 写回自带的类型转换
  out = tile.store(t, offs, dst)          # 改写为直接读 `acc`
```

链上的语句被删除，store 改写为
`tile.store(acc, offs, dst, pre_quant=s, pre_relu=r)`，原 store 携带的所有 kwarg
（`atomic`、`st_phase` 等）原样保留。

### 两种 ReLU 位置并不等价 {#relu-position}

硬件算的是 `clamp(ReLU(acc) * s)` —— 激活属于*量化前*阶段，看到的是未缩放的累加器，
目标类型的 clamp 排在最后（a2a3 真机由 `acc_to_gm_negative_scale` 测得，见
`99-verifier.md`）。

| 形态 | 链 | 能否折叠 |
| ---- | -- | -------- |
| A | `muls(maximums(acc, 0), s)` | 恒可 —— 这**就是**硬件的顺序 |
| B | `maximums(muls(acc, s), 0)` | 仅当 `s >= 0` |

`s >= 0` 时两者是同一个函数；`s < 0` 时逐元素都不同。B 是用户实际会写的形态（先反
量化再激活），所以加符号守卫放行，而不是一概拒绝。

## 拒绝的情形 {#declines}

拒绝时 IR **一个字节都不改** —— 一个用不上的优化必须是 no-op，绝不能改一半。
用户可据以行动的拒绝会发 PerfHint。

| 情形 | 码 | 原因 |
| ---- | -- | ---- |
| scale 不是编译期常量 | `PH-FE-001` | 打包字是编译期构造的 |
| 形态 B 且 `s < 0` | `PH-FE-002` | 与硬件顺序不符（见上） |
| 后端没有该 `(acc, dst)` 的带 scale 模式 | `PH-FE-003` | `SupportsFixpipePreQuant`；硬折会产出 `AccToGmStoreValid` 拒绝的 IR |
| cast 用了前端默认的 `mode="round"` | `PH-FE-004` | FIXPIPE 是**四舍六入五成双**（`RINT`），而 `ROUND` 是逢五远离零，折叠会改变平局处的结果。写 `mode="rint"` 即可启用 |
| cast 显式要求了 `saturation_mode` | — | `pto.tstore` 没有 `satmode` 可承载 |
| 链上任何中间值被读多于一次 | — | 折叠会删掉整条链 |
| store 上已经带了 epilogue | — | 两者需要复合，不是本 pass 的职责 |
| 目标是 `Mat` | — | 两个 handler 都已关闭，等 PTOAS#1570 |

cast 的规则刻意与既有的无 scale 折叠 `CastFoldableToFixpipeMat`
（`auto_tile_matmul_l0_pass.cpp`）保持一致 —— 两者争夺同一段 IR，对「哪些 cast 能由
fix-pipe 复现」必须给出同一个答案。

## 适用范围 {#scope}

只折叠 `Acc → GM`（`pto.tstore`）写回。`Acc → Mat`（`pto.tinsert`）形态被两个后端
handler 关闭，因为 ptoas 在那里会把 scale 发错
（[PTOAS#1570](https://github.com/hw-native-sys/PTOAS/issues/1570)，机制见
`99-verifier.md`）；发出来会得到 `FixpipeEpilogueValid` 拒绝的 IR。等 ptoas 修好，
那一支只需 handler 改一行，再加一个改写目标。

## 复杂度 {#complexity}

一趟统计 use、一趟建消费者索引、一趟改写 —— 配合哈希表查找为 O(N log N)，符合
`pass-complexity.md`。

## 测试 {#tests}

`tests/ut/ir/transforms/test_fold_fixpipe_acc_epilogue.py` —— 覆盖两种可折叠形态，
以及每一种拒绝情形（均以 `assert_structural_equal(After, Before)` 断言无改动）。
