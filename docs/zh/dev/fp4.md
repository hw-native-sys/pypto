# PyPTO 中的打包 FP4 类型

PyPTO 提供两种不同的打包 FP4 数据类型。本页介绍它们共用的类型与字节载体
（byte carrier）ABI、数值转换以及当前支持的算子。类型简表见[类型](../user/language/00-types.md)。

## 类型与单位约定

| DSL 类型 | 大写别名 | 存储方式 | PTO 类型 |
| -------- | -------- | -------- | -------- |
| `pl.fp4e2m1x2` | `pl.FP4E2M1X2` | 每字节存放两个 E2M1 值 | `!pto.f4E2M1x2` |
| `pl.fp4e1m2x2` | `pl.FP4E1M2X2` | 每字节存放两个 E1M2 值 | `!pto.f4E1M2x2` |

两种类型都满足 `get_bit() == 8` 和 `get_byte() == 1`。Tensor 和 tile 的 shape、
`valid_shape`、步长（stride）与偏移量均按字节载体计数。`[2,256]` 张量占用
512 字节，包含 1,024 个 FP4 值。载体维度或偏移量不必为偶数，因为每个载体已包含
完整的一对数值；各算子的对齐与布局限制仍然适用。

不再提供未打包的 FP4 类型或按半字节（nibble）计数的 ABI。旧的 `pl.FP4`、
`DataType.FP4` 和 `DT_FP4` 名称已移除。JIT 特化、函数参数和返回值、运行时分配以及
持久化参数元数据均保留载体维度，不会将某个维度隐式加倍或减半。

即使 shape 和字节数相同，这两种格式仍是不同的类型。修改函数注解不会转换数据：
依赖函数调用（包括无返回值调用和 submit）中，只要实参或形参任一方为打包 FP4，
双方 dtype 就必须相同。在支持字节别名的路径上，UINT8/INT8 值需要显式使用
`reinterpret_view`；仅修改被调用函数的注解不会重新解释字节。
两种 FP4 格式之间的数值转换应显式使用 `pl.cast`。

## Torch 与运行时载体

- `torch.float4_e2m1fn_x2` 直接映射为 `FP4E2M1X2`，物理 shape 不变。
  E1M2 注解会拒绝这种专用于 E2M1 的 Torch dtype。
- 两种打包类型的显式注解均接受 `torch.uint8` 载体。未标注的 UINT8 数据仍为
  UINT8；FP4 格式由注解指定。
- E2M1 返回值优先使用 Torch 原生 x2 dtype 分配，不可用时使用 UINT8。
  E1M2 使用 UINT8，因为 Torch 没有对应的原生 dtype。
- 编译后或重新加载的参数元数据保留数值格式和相同的载体 shape。
  打包运行时张量的秩至少为一，且最后一个载体维度必须为正。

原始载体存储不等于主机端数值实现。`torch_codegen` 会拒绝打包 FP4 参考代码，
而不会把这些字节当作 FP32 值。验证数值结果时，应使用显式解码对应格式的参考实现。

## Ascend950 上的算子支持

两种格式均支持 ND 转换、完整静态连续 DN 转换及下表中的数据搬运。
原生 MXFP4 量化仅支持 E2M1。

| 操作 | 支持范围 |
| ---- | -------- |
| 打包 GM 视图与 load/store | 两种格式均支持 ND/DN 载体维度、步长和偏移；宽度为一个载体时仍保持 ND；PTOAS v0.67 负责 DMA 描述符转换 |
| 任一打包格式 ↔ UINT8/INT8 别名 | shape 相同、连续且带地址的二维行主序 Vec 存储；使用 PYPTO 或 DSA_RP 内存规划器 |
| 任一打包格式 ↔ BF16 转换 | 无分块 Vec 上的原生 TCVT；ND 列或完整静态连续 DN 行按 2:1 展开或收缩 |
| 任一打包格式 → FP8E4M3FN/FP8E5M2 转换 | 经 BF16 和 FP32 合法化，并发出警告；也可使用主机参考实现或查找表 |
| E2M1↔E1M2 转换 | 经 BF16 进行数值转换，载体 shape 不变 |
| 其他宽类型转换 | 使用现有合法化路径，要求中间转换无损；编码限制见下文 |
| 打包 Vec 布局转换 | ND→NZ；载体维度随打包轴变化，数值维度保持不变 |
| 打包 Vec→Mat 搬运 | 布局相同，且显式使用 AIV 函数；通用 InCore 混合核展开会被拒绝 |
| 原生打包 `extract` / `assemble` | 仅支持保持布局且打包轴偏移为零的 cube/NZ 路径；通用 Vec 提取使用 INT8 别名，ND 插入使用 UINT8 别名 |
| `fillpad`、`fillpad_expand`、`fillpad_inplace` | 两种格式的 ND Vec；零值、最小值和最大值填充均保留有效载体矩形 |
| DN 数值转换 | 完整静态连续的二维 BF16↔打包或打包↔打包转换；使用 PYPTO/DSA_RP 静态地址；部分有效、带填充及分块布局均被拒绝 |
| 直接对打包类型执行 `reshape` 或 `transpose` | 不支持；在现有别名契约允许时，先对字节别名执行 reshape |
| 原生打包 FP4 `matmul_mx` | 固定版本 PTOAS 的维度契约不支持；先把 ND 操作数转换为 FP8 |
| MXFP4 `quant_mx` | 仅 E2M1，输入为 FP16/BF16，支持 OCP 和 axis0 上受支持的 NV 路径；使用 PYPTO/DSA_RP 静态分配 |
| `pl.tile.quant_mx_raw` | 显式提供工作区，返回原生 E2M1 数据（MXFP8 返回 INT8）及原始 UINT8 指数；支持静态部分有效区域和 axis0 交错 |
| Ascend910/A2A3 上执行打包 FP4 | 不支持 |

PTO GM 视图保留载体坐标。底层指令所用数值维度的转换由 PTOAS 负责；若 PyPTO
再次缩放 GM shape，会把行跨度加倍。物理 tile 仍须满足 ISA 对齐要求，包括 ND Vec
的 32 字节行跨度；较窄的载体窗口应使用带填充的存储，并通过 `valid_shape` 指定有效区域。

## 数值转换限制

BF16 编码支持 `rint`、`round`、`floor`、`ceil` 和 `trunc`，拒绝 `odd`。
打包要求完整的数值对：源维度为正偶数，或动态 `valid_shape` 可证明为偶数，
例如 `2 * n`。解码会展开载体维度，不要求该载体维度为偶数。

所有打包转换均要求 `none_box` 存储；更宽类型的转换链要求 ND 布局。
经 BF16 解码可精确保留每个 FP4 值。隐式编码链同样要求经 BF16 后保留原始源值：
FP16/FP32 及较宽整数源会被拒绝，因为该中间步骤可能导致两次舍入。
若确实需要中间舍入，应先显式转换为 BF16。固定版本模拟器对小数 FP4 转换的舍入有误，
因此数值验收使用 A5 硬件。

DN 转换要求完整静态二维矩形、规范的连续列主序步长、零 tile 起始偏移，且两端类型
均为 BF16 或打包 FP4。DN `[M,K]` 载体解码为 BF16 `[2*M,K]`。
可对完整 ND BF16 tile 使用 `pl.tile.transpose_view` 来准备列主序 BF16 源。
Tensor 转换遵守相同契约。PYPTO 或 DSA_RP 必须提供静态 tile 地址；代码生成绑定
完整静态有效描述符，以选择安全的原生转换路径。动态有效 DN 描述符即使针对完整矩形
也可能触发故障，因此仍不支持。

## 通过 UB 别名解码 UINT8 载体（Ascend950）

GM 张量保持 UINT8 类型，并按实际编码格式为其 UB 存储创建别名：

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
        fp4 = pl.reinterpret_view(payload, pl.fp4e2m1x2)
        decoded = pl.cast(fp4, pl.BF16)
        pl.tile.store(decoded, [row, 0], output)
    return output
```

该别名保留 `[32,256]` 载体 shape，并共享原有的 8,192 字节分配。
只有数值转换会创建新的 BF16 `[32,512]` tile。若源字节编码为 E1M2，
应改用 `pl.fp4e1m2x2`。

别名要求 `slayout=none_box`、连续存储、零 tile `start_offset`，以及具有静态 UB
字节偏移的带地址内存引用（MemRef）。源字节跨度必须等于载体元素数量。
非连续 UB 切片（包括其间接别名和 reshape）以及无地址输入会被拒绝。
GM 加载偏移可为非零或动态值，它不同于 UB 偏移。显式别名 shape 必须与源相同；
需要 reshape 时，应先单独变形字节视图，再重新创建别名。
InCore tensor 别名降低为相同的 tile 路径；编排层 tensor 别名仍不支持。

PYPTO 和 DSA_RP 内存规划器生成带地址别名。PTOAS 内存规划器的 `treshape`
拒绝打包 FP4 类型，因此不支持此路径。别名保留比特，不会将一种 FP4 格式数值转换为另一种。

部分搬运的硬件测试逐字节比较所有载体，包括未写入的输出区域。固定版本模拟器的 DN
结果会损坏；部分 ND 用例虽字节比较一致，仍在清理阶段崩溃，因此这些用例仅在 A5 上运行。

## MX 量化

`pl.quant_mx(..., dtype=pl.fp4e2m1x2)` 每 32 个 FP16/BF16 值分为一组进行量化。
axis1 将 `[M,K]` 转为数据 `[M,K/2]` 和缩放因子 `[M,K/32]`；axis0 将 `[N,K]`
转为数据 `[K,N/2]` 和缩放因子 `[K/32,N]`。数据 shape 按字节载体计数。
公开量化接口要求 tile 完全有效、`K%64==0`、axis1 的 `M%16==0`，以及 FP4 axis0
的 `N%64==0`，以满足 32 字节载体行跨度。

默认采用 `quant_scale_alg="ocp"`。`"nv"` 支持 FP16/BF16 的 axis0 路径；PTOAS
不支持 FP16/BF16 的 axis1 NV 路径或 E1M2 量化。打包量化要求 PYPTO 或 DSA_RP
提供静态地址，因为 PTOAS 无法将其自定义 FP4 类型 reshape 为所需的静态有效描述符。

`pl.tile.quant_mx_raw` 保持源朝向，返回原生打包数据及 UINT8 指数编码，不进行复合
缩放因子分块。工作区使用源 dtype；axis0 工作区保留源的物理列跨度，axis1 的
FP16/BF16 OCP 缩放工作区容量将组数向上取整到 128 个元素。
支持静态部分有效区域；axis1 要求物理列数等于有效列数，且 FP4 物理列数为 64 的倍数。
axis0 每 32 个有效行分组；`interleave=True` 要求 FP16/BF16 的物理行数和有效行数
均为 64 的倍数。量化入口见[算子目录](../user/ops/01-catalog.md)。
原生打包 FP4 `matmul_mx` 仍不支持；量化结果本身不会建立原生矩阵计算路径。

## 另见

- [类型](../user/language/00-types.md) — 公开别名与载体 ABI
- [算子 / MX](ir/05-operators.md) — 现有 MXFP8 矩阵路径
