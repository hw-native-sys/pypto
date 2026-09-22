# PyPTO 中的 FP4

FP4 / packed FP4 的权威说明页。Pass 算法见 [PackFp4](passes/07-pack_fp4.md)；
本页写角色、管道、单位约定、cast 策略与三层支持矩阵。

## 角色

| 类型 | 含义 |
| ---- | ---- |
| `pl.FP4` | 前端**逻辑** E2M1 nibble（`GetBit()==4`）；shape / `valid_shape` 按 nibble 计数 |
| `pl.FP4E2M1X2` | **打包** carrier（每元素两个 nibble，`GetBit()==8`）；`PackFp4` 之后或手写物理 shape。逻辑路径的偶数末维由作者保证；DN / NZ / col_major / cube 仍由 PackFp4 拒绝。手写 `FP4E2M1X2` **仅 ND row-major** |

`tensor.full` / `create` 填充值：打包后仅 **0** 有定义；非零 fill（双 nibble 复制 vs 仅低 nibble）本版本未定义。

PyPTO 另拒绝 FP4 族的 `reshape`、标量 `read`/`write`、`tensor.dim`、白名单外坐标
op，以及 cube / DN / NZ / col_major 打包——见 pass 文档与报错（属仅 PyPTO 限制，
不进三列表）。

`FP4E2M1X2` 的 layout 转化硬拒：非 ND 注解、隐式列向量 `[M, 1]` DN、以及显式 `tensor.view` layout 翻转。

## 管道

1. **PackFp4**（仅静态）：逻辑 `pl.FP4` → `FP4E2M1X2`；末维 covering size / offset /
   ND leading stride 必须为偶数 `ConstInt` 并 `/2`。
2. **本地 ExpandPackedFp4\***（单卡 codegen）：GM `make_tensor_view` / partition
   末维几何扩回 nibble，供 pto-isa `GetByteSize`；EmitC 对 Tile 列做同样扩展以配合 TCVT。
   **tile_buf 保持 carrier 单位**（不回退）。MX layout 不扩展。
3. **Cast**（`LegalizeTileCast`）：见 [Cast 策略](#cast-策略ascend950)。

## 单位约定

| 层 | 单位 |
| -- | ---- |
| 前端 `pl.FP4` shape / `valid_shape` | nibble |
| PackFp4 之后 IR / `tile_buf` / orch create | carrier |
| 手写 `pl.FP4E2M1X2` IR / `tile_buf` / Torch ABI | carrier |
| `make_tensor_view` / `partition_view`（Expand 后） | nibble（供 pto-isa `GetByteSize`） |
| runtime Tensor / Torch `float4_e2m1fn_x2` | carrier 元素 |

多行 ND packed 张量请用 **carrier** 末维与 leading stride（例如每行 512 逻辑
nibble 写成 `pl.Tensor[[2, 256], pl.FP4E2M1X2]`）。Codegen 会把 GM view 扩到
nibble 单位，使多行 pitch 与 Tile / pto-isa 一致。在 `FP4E2M1X2` 上误用逻辑宽度
可能导致 GM 行 stride 错位——见 issue
[#2754](https://github.com/hw-native-sys/pypto/issues/2754)。

## Cast 策略（Ascend950）

`LegalizeTileCast` 对 FP4 族如下处理：

| Cast | 行为 |
| ---- | ---- |
| `FP4` / `FP4E2M1X2` ↔ `BF16` | 原生 TCVT（DSv4.1 c1a）。静默。Packed ↔ 更宽类型按末轴 **2:1** 调整（打包时要求静态偶数末维）。结果 stride 按目标 shape 重建为连续。 |
| `FP4` / `FP4E2M1X2` → `FP8E4M3FN` / `FP8E5M2` | 合法化为 FP4→BF16→FP32→FP8，并带 **Warning**（更推荐 LUT / 主机预转换）。 |
| `FP4` ↔ `FP4E2M1X2` | **拒绝**（`RejectFp4FamilyInternalCast`；几何不一致）；经更宽类型绕行或显式改 shape。 |
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

同样 2:1 末轴展开；预期 LegalizeTileCast Warning：

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

标记：✅ 支持 · ⚠️ 部分 / Warning · ❌ 不支持 · ⏳ 下层可用、PyPTO 待接

PTOAS 与 pto-isa 皆为不适用的行不写（仅 PyPTO 限制见上文）。

| 功能 | PyPTO | PTOAS | pto-isa |
| ---- | ----- | ----- | ------- |
| 静态 Pack（偶末维 `ConstInt` → `FP4E2M1X2`） | ✅ | ✅ 消费 `!pto.f4E2M1x2` | ✅ nibble `GetByteSize` |
| 动态逻辑 FP4 Pack | ⏳ 硬拒 | ✅ 可吃 packed 动态 shape | ✅ |
| `tensor.view` / tile load-store（静态偶末维） | ✅ `valid_shape` 一并 `/2` | ✅ | ✅ |
| FP4↔BF16 cast | ✅ | ✅ | ✅ TCVT |
| FP4→FP8\* cast | ⚠️ Warning（优先 LUT/host） | ✅ | ✅ |
| `FP4` ↔ `FP4E2M1X2` cast | ❌ 拒绝 | — | — |
| `FP4E2M1X2` layout 转化（非 ND、`[M,1]` DN、layout `tensor.view`） | ❌ 仅 ND row-major | — | — |
| `transpose` / `ttrans` | ❌ | ❌ 无 f4E2M1x2 `ttrans` | ❌ |
| FP4 族上的 `tensor.dim` | ❌ 请用静态 shape | — | — |
| Distributed（remote / window / put / get） | ⏳ | ⚠️ 其它 dtype 的 comm/view 有 | — |
| `matmul_mx` 原生 FP4 data | ⏳ | ✅ MX 路径 | ✅ TCVT |

## 推荐路径

1. 动态 block/slot：DSv4.1 风格 **UINT8 半宽** cache ABI。
2. 有限静态 `pl.FP4`（偶末维）+ PackFp4 + 本地 Expand。
3. 可选手写 `pl.FP4E2M1X2`（物理 **carrier** shape，避免 `#2754` 类 stride 问题；layout/cube 仍校验）。
4. FP4→FP8 优先 **LUT / host**；设备 cast 仅 Warning。

## 参见

- [PackFp4 pass](passes/07-pack_fp4.md) — 仅算法
- [类型](../user/language/00-types.md) — FP4 短述
- [算子 / MX](ir/05-operators.md) — matmul_mx FP4 说明
