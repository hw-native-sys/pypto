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
| 打包 `make_tensor_view` / `partition_view`（PTOAS v0.67+） | carrier；PTOAS 将 DMA 描述符转换为 nibble 单位 |
| runtime Tensor / `torch.float4_e2m1fn_x2` | carrier 元素 |

多行 ND packed 张量请用 **carrier** 末维与 leading stride（例如每行 512 逻辑
nibble 写成 `pl.Tensor[[2, 256], pl.FP4E2M1X2]`）。Codegen 保留 GM view 的 carrier 单位。
PTOAS v0.67 会将传输描述符转换为 pto-isa 所需的 nibble 单位；PyPTO 再次扩展会导致行步长翻倍。在 `FP4E2M1X2` 上误用逻辑宽度，
可能导致 GM 行 stride 错位；在自动 PackFp4 将 nibble 尺寸转换为 carrier 之前，in-core PTO codegen 会拒绝逻辑 `pl.FP4`——见 issue
[#2754](https://github.com/hw-native-sys/pypto/issues/2754)。

## 通过 UB alias 解码 UINT8 carrier（Ascend950）

GM 保持 UINT8 打包字节，在 UB 内使用零拷贝 FP4E2M1X2 视图（zero-copy view）：

```python
import pypto.language as pl


@pl.jit(auto_scope=False)
def decode_bytes(
    packed: pl.Tensor[[32, 256], pl.UINT8],
    output: pl.Out[pl.Tensor[[32, 512], pl.BF16]],
):
    with pl.spmd(1):
        row = pl.tile.get_block_idx() * 32
        payload = pl.tile.load(packed, [row, 0], [32, 256])
        fp4 = pl.reinterpret_view(payload, pl.FP4E2M1X2)
        decoded = pl.cast(fp4, pl.BF16)
        pl.tile.store(decoded, [row, 0], output)
    return output
```

`payload` 与 `fp4` 的 carrier shape 都是 `[32,256]`，在同一 UB 地址共享
8,192 字节。数值 cast 将末轴展开为新的 BF16 `[32,512]` tile，并使用原生
`TCVT`。GM 输入仍采用 UINT8 字节寻址（本例为 `row * 256`）；不会生成 FP4 GM
partition 或软件 nibble 解码序列。

alias 仅接受 UINT8 → FP4E2M1X2，要求二维 packed row-major Vec tile、
`slayout=none_box`、无显式 stride，且 tile `start_offset` 为零。显式 `shape`
必须与源 shape 完全相同。carrier `valid_shape` 保留，随后 cast 展开其末轴。
逆向 alias、逻辑 FP4 alias、tensor alias 及通过此 alias 改变 shape 均被拒绝。
GM load 可使用非零偏移；它与 UB tile 的 `start_offset` 不同。

源 tile 必须具有显式地址的内存引用（MemRef），其字节跨度必须等于静态 carrier
元素数。非连续 UB 列切片（包括其 alias 和 reshape）以及 `tile.tpop_from_aic`
等无 MemRef 的输入，会在生成 FP4 转换前被拒绝。请直接从 UINT8 GM 加载所需窗口，
得到连续 UB tile 后再创建 alias。包括中间视图在内，alias 的 UB 字节偏移必须
为静态值；GM load 仍可使用运行时偏移。

**PTOAS 0.65 足以支持 PYPTO 和 DSA_RP 内存规划器**：两者生成有显式地址的
类型 alias，并通过 A5 逐位测试。此 alias 会拒绝 PTOAS 内存规划器，因为 0.65
的 `pto.treshape` 不支持 packed FP4。其他 reinterpret-view dtype 保持原有
规划器支持。本路径不修复通用 typed FP4 GM partition 偏移、不引入 PackFp4、
不增加 tensor alias。

## Cast 策略（Ascend950）

`LegalizeTileCast` 对 FP4 族如下处理：

| Cast | 行为 |
| ---- | ---- |
| `FP4` / `FP4E2M1X2` ↔ `BF16` | 原生 TCVT（DSv4.1 c1a）。静默。Packed ↔ 更宽类型按末轴 **2:1** 调整（打包时要求静态偶数末维）。 |
| `FP4` / `FP4E2M1X2` → `FP8E4M3FN` / `FP8E5M2` | 合法化为 FP4→BF16→FP32→FP8，并带 **Warning**（更推荐 LUT / 主机预转换）。 |
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

## 支持矩阵（本切片）

图例：✅ 支持 · ⚠️ 部分 / Warning · ❌ 不支持 · ⏳ 本切片未提供

| 特性 | PyPTO | 说明 |
| ---- | ----- | ---- |
| 手写 `pl.FP4E2M1X2` | ✅ | packed 路径的首选前端；**仅 ND** |
| 无 PackFp4 的逻辑 `pl.FP4` | ❌ | In-core PTO codegen 拒绝 nibble 单位的 GM view；请使用 carrier 尺寸的 `FP4E2M1X2` |
| 打包 GM view 保留 carrier 单位 | ✅ | PTOAS v0.67 负责 DMA 转换（ND） |
| UINT8 → FP4E2M1X2 UB alias | ✅ | 二维 packed row-major Vec；carrier shape 不变；仅 PYPTO / DSA_RP |
| `FP4E2M1X2` ↔ BF16 cast | ✅ | 静默原生 hop；结果 stride 重建为连续 |
| `FP4E2M1X2` → FP8\* cast | ⚠️ | Warning；更推荐 LUT / 主机 |
| `FP4` ↔ `FP4E2M1X2` cast | ❌ | 拒绝 |
| 自动 PackFp4 | ⏳ | 后续 |
| `FP4E2M1X2` 的 `reshape` / `transpose` / DN / NZ / 列向量 `[M,1]` / layout `tensor.view` | ❌ | 仅 ND row-major；隐式 DN 与显式 layout 转化硬拒 |
| `matmul_mx` 原生 FP4 数据 | ⏳ | 需要时先 cast lhs 到 FP8 |

## 推荐路径

1. 手写 `pl.FP4E2M1X2` 并用物理 carrier shape（避免 `#2754` 类 stride 问题）。
2. 需要更宽浮点时 cast 到 BF16。
3. FP4→FP8 优先 **LUT / 主机**；设备 cast 仅 Warning。
4. PackFp4 落地前，in-core GM 访问请使用 `pl.FP4E2M1X2`。

## 另见

- [类型](../user/language/00-types.md) — FP4 / FP4E2M1X2 短表
- [算子 / MX](ir/05-operators.md) — `matmul_mx` FP4 说明
