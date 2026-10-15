# PyPTO 中的 FP4

FP4 / packed FP4 的权威说明页。类型表短述见 [类型](../user/language/00-types.md)；
本页写角色、单位约定、cast 策略，以及当前可手写的安全路径。

本切片**尚未**提供自动 `PackFp4`（逻辑 `pl.FP4` → 打包 carrier）；在 pass
落地前请优先手写 `pl.FP4E2M1X2`。

## 角色

| 类型 | 含义 |
| ---- | ---- |
| `pl.FP4` | 前端**逻辑** E2M1 nibble（`GetBit()==4`）；shape / `valid_shape` 按 nibble 计数。解析短名会发 `UserWarning`（推荐 `FP4E2M1X2`）。 |
| `pl.FP4E2M1X2` | **打包** carrier（每元素两个 nibble，`GetBit()==8`）。物理末维对齐 `torch.float4_e2m1fn_x2` / `!pto.f4E2M1x2`。 |

偶数末维约束针对**逻辑 nibble** 几何（以及 wider→packed、需要静态偶数末维再减半的
cast）。手写 **carrier** 末维可以为奇数。请优先用 ND row-major + Vec；Cube / DN /
NZ / col_major / distributed 的 FP4 族路径在本切片不支持（硬拒随 PackFp4 落地）。

`tensor` / `tile` 的 `reshape` 与 `transpose` 会拒绝 FP4 族（见报错与支持矩阵）。

## 单位约定

| 层 | 单位 |
| -- | ---- |
| 前端逻辑 `pl.FP4` shape / `valid_shape` | nibble |
| 手写 `pl.FP4E2M1X2` IR / `tile_buf` / Torch ABI | carrier |
| `make_tensor_view` shape / strides（ExpandPackedFp4\* 之后） | nibble（供 pto-isa `GetByteSize`） |
| `partition_view` sizes | nibble |
| `partition_view` offsets（FP4 减半之后） | **预减半**：PTOAS 按 carrier 字节单位推进 partition 基址指针（`data_ptr + Σ offset·stride`），而 pto-isa 按 nibble 单位走 strides/extents，因此 PyPTO 把 FP4 族的窗口偏移全部除以 2 发射——行偏移按 nibble 行减半，packed 末轴（先 ×2 展开）净回到 carrier 值 |
| runtime Tensor / `torch.float4_e2m1fn_x2` | carrier 元素 |

窗口偏移必须**在 nibble 单位下为偶数**（偶数行起点、末轴窗口从 carrier 边界开始）；
奇数偏移在 PTOAS 的混合单位算术下没有整数字节编码，codegen 会带说明直接拒绝。真机
证据：`worktrees/fp8fp4/probe_fp4_noffset.py`（减半后 k0=32 读第 32 行、nb=32 读第
32 列；不减半则都落到 2× 处）。调用已编译程序时，torch `float4_e2m1fn_x2` 张量的
**shape 必须按 carrier 计数**（逻辑 `[K, N]` 注解编译为运行时 carrier 形状
`[K, N/2]`；dispatch 拒绝逻辑形状的张量——该检查存在之前，2× 宽的 GM 视图曾使每行
读越过真实末端）。

多行 ND packed 张量请用 **carrier** 末维与 leading stride（例如每行 512 逻辑
nibble 写成 `pl.Tensor[[2, 256], pl.FP4E2M1X2]`）。Codegen 会把 GM view 扩到
nibble 单位，使多行 pitch 与 Tile / pto-isa 一致。在 `FP4E2M1X2` 上误用逻辑宽度，
或在无自动打包时用逻辑 `pl.FP4` 多行 ND，可能导致 GM 行 stride 错位——见 issue
[#2754](https://github.com/hw-native-sys/pypto/issues/2754)。

## Cast 策略（Ascend950）

`LegalizeTileCast` 对 FP4 族如下处理：

| Cast | 行为 |
| ---- | ---- |
| `FP4` / `FP4E2M1X2` ↔ `BF16` | 原生 TCVT（DSv4.1 c1a）。静默。Packed ↔ 更宽类型按末轴 **2:1** 调整（打包时要求静态偶数末维）。 |
| `FP4` / `FP4E2M1X2` → `FP8E4M3FN` | Ascend950 上为**核内 LUT + tgather**（`LowersFp4ToFp8InKernel`）：一次函数级 256 项 `u16` 表（`LUT[i] = f8(i&15) \| f8(i>>4)<<8`，E2M1→E4M3 精确位嵌入，建表只在首个 cast 站点展开一次），每个站点 `u8→u16` 零扩展（唯一原生 TCVT）+ `tile.gather` 查表。同配置实测均不慢于 FP4→BF16→FP32→FP8 多跳：`[N,K]` rhs（cast 后接 Vec transpose）256×256×128 tile **41.6 ms vs 多跳 47.2 ms**，默认 128×128×256 **366 ms vs 多跳 378 ms**（该配置整体偏慢来自 transpose/NZ 数据路径，与 cast 路径无关）；`[K,N]` rhs 默认 tile 10.8 ms。其他后端展开 FP4→BF16→FP32→FP8 并带 **Warning**。 |
| `FP4` / `FP4E2M1X2` → `FP8E5M2` | 合法化为 FP4→BF16→FP32→FP8，并带 **Warning**（LUT/gather 路径仅覆盖 E4M3）。 |
| `FP4` ↔ `FP4E2M1X2` | **拒绝**（几何不一致）；经更宽类型绕行或显式改 shape。 |
| Wider → `FP4E2M1X2` 且末维动态 | **拒绝**（仅静态正偶数）。 |

### 手写 `FP4E2M1X2` → BF16

Carrier 末维 `32` 展开为 BF16 末维 `64`：

```python
import pypto.language as pl


@pl.function(type=pl.FunctionType.InCore)
def fp4x2_to_bf16(
    x: pl.Tensor[[16, 32], pl.FP4E2M1X2],
    out: pl.Out[pl.Tensor[[16, 64], pl.BF16]],
) -> pl.Tensor[[16, 64], pl.BF16]:
    t = pl.load(x, [0, 0], [16, 32])
    c = pl.cast(t, pl.BF16)
    return pl.store(c, [0, 0], out)
```

### 手写 `FP4E2M1X2` → FP8

同样 2:1 末轴展开。Ascend950 上合法化为**核内 256 项 `u16` LUT + `tile.gather`** 路径（无 Warning）；无核内下沉的后端展开多跳并带 Warning：

```python
import pypto.language as pl


@pl.function(type=pl.FunctionType.InCore)
def fp4x2_to_fp8(
    x: pl.Tensor[[16, 32], pl.FP4E2M1X2],
    out: pl.Out[pl.Tensor[[16, 64], pl.FP8E4M3FN]],
) -> pl.Tensor[[16, 64], pl.FP8E4M3FN]:
    t = pl.load(x, [0, 0], [16, 32])
    c = pl.cast(t, pl.FP8E4M3FN)
    return pl.store(c, [0, 0], out)
```

## 支持矩阵（本切片）

图例：✅ 支持 · ⚠️ 部分 / Warning · ❌ 不支持 · ⏳ 本切片未提供

| 特性 | PyPTO | 说明 |
| ---- | ----- | ---- |
| 手写 `pl.FP4E2M1X2` | ✅ | packed 路径的首选前端；**仅 ND** |
| 无 PackFp4 的逻辑 `pl.FP4` | ⚠️ | Prefer Warning；A5 in-core 仍支持逻辑 FP4，并与 FP4E2M1X2 并存 |
| GM ExpandPackedFp4\*（carrier→nibble） | ✅ | `make_tensor_view` / partition 末轴（ND） |
| `FP4E2M1X2` ↔ BF16 cast | ✅ | 静默原生 hop；结果 stride 重建为连续 |
| `FP4E2M1X2` → `FP8E4M3FN` cast | ✅ | 950 核内 LUT + tgather；其他后端 ⚠️ 多跳 Warning |
| `FP4E2M1X2` → `FP8E5M2` cast | ⚠️ | Warning；更推荐 LUT / 主机 |
| `FP4` ↔ `FP4E2M1X2` cast | ❌ | 拒绝 |
| 自动 PackFp4 | ⏳ | 后续 |
| `FP4E2M1X2` 的 `reshape` / `transpose` / DN / NZ / 列向量 `[M,1]` / layout `tensor.view` | ❌ | 仅 ND row-major；隐式 DN 与显式 layout 转化硬拒 |
| `matmul_mx` 原生 FP4 数据 | ⏳ | 需要时先 cast lhs 到 FP8 |

## 推荐路径

1. 手写 `pl.FP4E2M1X2` 并用物理 carrier shape（避免 `#2754` 类 stride 问题）。
2. 需要更宽浮点时 cast 到 BF16。
3. Ascend950 上 `pl.cast` 到 `FP8E4M3FN` 走核内 LUT + tgather（静默）；其他后端 FP4→FP8 优先 **LUT / 主机**——设备 cast 展开多跳并带 Warning。
4. 仅在接受不完整路径时保留逻辑 `pl.FP4`，直到 PackFp4 落地。

## 另见

- [类型](../user/language/00-types.md) — FP4 / FP4E2M1X2 短表
- [算子 / MX](ir/05-operators.md) — `matmul_mx` FP4 说明
