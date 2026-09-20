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

打包几何的偶数末维由作者保证。Cube / DN / NZ / col_major 的 FP4 族路径会被拒绝。

## 单位约定

| 层 | 单位 |
| -- | ---- |
| 前端逻辑 `pl.FP4` shape / `valid_shape` | nibble |
| 手写 `pl.FP4E2M1X2` IR / `tile_buf` / Torch ABI | carrier |
| runtime Tensor / `torch.float4_e2m1fn_x2` | carrier 元素 |

多行 ND packed 张量请用 **carrier** 末维与 leading stride（例如每行 512 逻辑
nibble 写成 `pl.Tensor[[2, 256], pl.FP4E2M1X2]`）。在 `FP4E2M1X2` 上误用逻辑宽度，
或在无自动打包时用逻辑 `pl.FP4` 多行 ND，可能导致 GM 行 stride 错位——见 issue
[#2754](https://github.com/hw-native-sys/pypto/issues/2754)。

## Cast 策略（Ascend950）

`LegalizeTileCast` 对 FP4 族的处理：

| Cast | 行为 |
| ---- | ---- |
| `FP4` / `FP4E2M1X2` ↔ `BF16` | 原生 TCVT（DSv4.1 c1a），静默。packed ↔ 更宽类型时末维按 **2:1** 调整。 |
| `FP4` / `FP4E2M1X2` → `FP8E4M3FN` / `FP8E5M2` | 展开为 FP4→BF16→FP32→FP8，并发 **Warning**（优先 LUT / host 预转）。 |

### 手写 `FP4E2M1X2` → BF16

carrier 末维 `32` 展开为 BF16 末维 `64`：

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

同样末维 2:1；会收到 LegalizeTileCast Warning：

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

## 支持情况（本版本）

标记：✅ 支持 · ⚠️ 部分 / Warning · ❌ 不支持 · ⏳ 本切片未提供

| 功能 | PyPTO | 说明 |
| ---- | ----- | ---- |
| 手写 `pl.FP4E2M1X2` | ✅ | packed 路径的推荐前端 |
| 无 PackFp4 的逻辑 `pl.FP4` | ⚠️ | Warning；A5 in-core 拒逻辑 FP4 |
| `FP4E2M1X2` ↔ BF16 cast | ✅ | 静默原生 hop |
| `FP4E2M1X2` → FP8\* cast | ⚠️ | Warning；优先 LUT / host |
| 自动 PackFp4 | ⏳ | 后续切片 |
| `transpose` / cube / 分布式 FP4 | ❌ / ⏳ | 本页范围外 |
| `matmul_mx` 原生 FP4 data | ⏳ | 需要时先把 lhs cast 到 FP8 |

## 推荐路径

1. 手写 `pl.FP4E2M1X2` 并使用物理 carrier shape（避开 #2754 类 stride 问题）。
2. 需要更宽浮点时 cast 到 BF16。
3. FP4→FP8 优先 **LUT / host**；设备 cast 仅 Warning。
4. 逻辑 `pl.FP4` 仅在接受不完善路径时使用，直到 PackFp4 落地。

## 参见

- [类型](../user/language/00-types.md) — FP4 / FP4E2M1X2 短表
- [算子 / MX](ir/05-operators.md) — `matmul_mx` 的 FP4 说明
