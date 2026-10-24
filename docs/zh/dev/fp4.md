# Packed FP4 types in PyPTO

PyPTO exposes two distinct packed FP4 data types. This page describes their
shared type and byte-carrier ABI, followed by the existing E2M1 operator support.
The short type table is in [Types](../user/language/00-types.md).

## Type and unit contract

| DSL type | Uppercase alias | Storage | PTO type |
| -------- | --------------- | ------- | -------- |
| `pl.fp4e2m1x2` | `pl.FP4E2M1X2` | Two E2M1 values per byte | `!pto.f4E2M1x2` |
| `pl.fp4e1m2x2` | `pl.FP4E1M2X2` | Two E1M2 values per byte | `!pto.f4E1M2x2` |

Both types have `get_bit() == 8` and `get_byte() == 1`. Tensor and tile shapes,
`valid_shape`, strides, and offsets count byte carriers. A `[2,256]` tensor
occupies 512 bytes and contains 1,024 FP4 values. A carrier extent or offset
need not be even; each carrier already contains a complete pair of values.
Operator-specific alignment and layout restrictions still apply.

There is no unpacked FP4 type or nibble-counting ABI. The old `pl.FP4`,
`DataType.FP4`, and `DT_FP4` names are removed. JIT specialization, function
arguments and returns, runtime allocation, and persisted argument metadata
preserve carrier geometry without doubling or halving a dimension.

The two formats are distinct even when their shapes and byte sizes match.
Changing a function annotation does not convert data: dependency calls,
including void calls and submits, reject E2M1/E1M2 format mismatches.
Numerical conversion between these formats is deferred to a later change.

## Torch and runtime carriers

- `torch.float4_e2m1fn_x2` maps directly to `FP4E2M1X2`, with its physical shape
  unchanged. An E1M2 annotation rejects this E2M1-specific Torch dtype.
- An explicit annotation for either packed type accepts `torch.uint8` carriers.
  Unannotated UINT8 data remains UINT8; the annotation supplies the FP4 format.
- E2M1 return allocation uses Torch's native x2 dtype when available, otherwise
  UINT8. E1M2 uses UINT8 because Torch has no corresponding native dtype.
- Compiled and reloaded argument metadata keeps the numerical format and the
  same carrier shape. Packed runtime tensors require rank at least one and a
  positive final carrier extent.

Raw carrier storage is not a host numerical implementation. `torch_codegen`
rejects packed FP4 references instead of treating their bytes as FP32 values.
Use a reference that explicitly decodes the selected format when checking
numerical results.

## Existing E2M1 operator support on Ascend950

The type foundation retains the existing `FP4E2M1X2` paths. It does not add
E1M2 numerical casts, new transfer layouts, or MXFP4 quantization.

| Operation | Existing support |
| --------- | ---------------- |
| Packed GM views | ND carrier dimensions and strides; PTOAS v0.67 handles DMA descriptor conversion |
| E2M1x2 ↔ UINT8/INT8 alias | Same-shape, dense addressed 2D row-major Vec storage; PYPTO or DSA_RP planning |
| E2M1x2 ↔ BF16 cast | Native TCVT; the last axis expands/contracts 2:1; packing requires a static positive even source extent |
| E2M1x2 → FP8E4M3FN cast | One exact native TCVT on A5; eligible private cast-plus-move chains write FP8 NZ directly |
| E2M1x2 → FP8E5M2 cast | Legalized through BF16 and FP32, with a warning |
| E1M2 numerical casts and E2M1↔E1M2 conversion | Deferred |
| Packed DN, NZ, column-major, column-vector `[M,1]`, or layout-changing packed views | Not enabled by this change |
| Direct packed `reshape` or `transpose` | Unsupported; reshape a byte alias where the existing alias contract permits it |
| Native packed FP4 `matmul_mx` and MXFP4 `quant_mx` | Unsupported; the existing E2M1-to-FP8 cast can prepare an FP8 matrix operand |
| Ascend910/A2A3 packed FP4 execution | Unsupported |

PTO GM views retain carrier coordinates. The conversion to the lower-level
instruction's value geometry belongs to PTOAS; scaling the GM shape again in
PyPTO would double its row pitch.

## Decode a UINT8 carrier through a UB alias (Ascend950)

This existing E2M1 path keeps the GM tensor in UINT8 and aliases its UB storage:

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

The alias keeps the `[32,256]` carrier shape and the same 8,192-byte allocation.
Only the numerical cast creates a fresh BF16 `[32,512]` tile.

The alias requires `slayout=none_box`, dense storage, zero tile `start_offset`,
and an address-backed MemRef with a static UB byte offset. Its source byte span
must equal the carrier element count. Noncontiguous UB slices, including
intermediate aliases or reshapes of them, and addressless inputs are rejected.
GM load offsets may be nonzero or dynamic; they are distinct from UB offsets.
An explicit alias shape must match the source. Reshape a byte view separately
before aliasing back. InCore tensor aliases lower to the same tile path;
orchestration tensor aliases remain unsupported.

PYPTO and DSA_RP planning emit addressed aliases. PTOAS planning remains
unsupported for this path because its `treshape` rejects the packed FP4 type.
These existing alias constraints do not establish additional E1M2 operator
support.

## Cast 策略（Ascend950）

ISA 转换已合入规范仓库
[cann/pto-isa!1967](https://gitcode.com/cann/pto-isa/merge_requests/1967)。
[GitHub 镜像 #355](https://github.com/hw-native-sys/pto-isa/pull/355) 已关闭，
但关闭并不证明镜像或运行时 pin 已包含实现。当前 runtime/ISA pin 仍缺少此实现。
编译器支持仍由 [PTOAS #1637](https://github.com/hw-native-sys/PTOAS/pull/1637) 和
[#1669](https://github.com/hw-native-sys/PTOAS/issues/1669) 跟踪。
公开 PTOAS v0.67 pin 及 v0.69 发布版均缺少直接转换路径；
此处未声明具备全部前置条件的发布版本。

公开 cast 接受行优先 ND E2M1 carrier，并返回新分配的行优先 ND FP8 tile。
PYPTO 和 PTOAS 两种内存规划器均使用一次原生转换，不产生编译器 LUT/索引
tile、BF16/FP32 中转或转换 scratch。源 shape、有效范围、stride 和 GM
offset 保持字节 carrier 单位；仅 cast 结果的列和有效列按两倍展开。
例如 `[32,256]`、有效范围 `[23,129]` 转为 FP8 `[32,512]`、
有效范围 `[23,258]`。在对齐的物理 tile 几何范围内，奇数 carrier
有效列仍包含完整数值对，因此合法；物理 ND/NZ 对齐限制仍然适用。

分配前，`ExpandMixedKernel` 可将 cast 与其唯一兼容 NZ move 合并，
在原 cast 位置直接写入独立 FP8 NZ 结果。共享结果、跨区域使用、已分配别名
或不匹配的布局/有效元数据会保留 ND cast 和独立 move。
`[32,256]` packed 源与 `[32,512]` FP8 NZ 结果分别使用互不重叠的
8,192 和 16,384 字节缓冲区，不产生 FP8 ND 中间值。
此能力仅增加 NZ **输出**，不增加 packed NZ 或列优先输入。
NK 矩阵路径先解码行优先 packed `[N,K/2]`，再转置 FP8；
packed reshape 和 transpose 仍不支持。

E1M2 数值 cast 及 E2M1/E1M2 互转仍不支持。E2M1 到 E5M2 仍经
BF16/FP32 中转并发出警告。A2/A3 packed 执行、公开字节别名的内存规划器
限制和其他 cast/scratch 规则保持不变。

## See also

- [Types](../user/language/00-types.md) — public aliases and carrier ABI
- [Operators / MX](ir/05-operators.md) — existing MXFP8 matrix paths


## 可选的 NZ 错位写与 MX L0 双缓冲

每次编译可独立设置 `enable_fp4_nz_row_plus_one` 和
`enable_mx_l0_ping_pong`，两者默认均为 false。可以直接传给
`ir.compile(...)`，也可以用
`passes.PassContext([], enable_fp4_nz_row_plus_one=True,
enable_mx_l0_ping_pong=True)` 包裹 JIT 调用。嵌套的转储和性能分析
上下文会继承选项；JIT 缓存键区分四种组合。`compiler_options.json`
记录实际请求的选项，不代表每个算子都满足优化条件。

NZ 错位写选择在向量切分和跨核适配完成后进行，仅覆盖 A5 原生
E2M1x2 到 E4M3FN 转换，且要求每个 AIV 的输出完整为
`128x256`、唯一消费者为 `tpush_to_aic`。其他形状、部分或动态
有效范围以及多消费者情形继续使用普通 NZ。逻辑形状保持
`128x256`；`CompactMode.row_plus_one` 分配
`129*256 = 33024` 字节 UB。生产者、别名和 TPUSH 保留该模式，
TINSERT 在写入普通 `256x256` L1/FIFO 载荷时去除面板间的空行。
输入载体、数值转换、scale 和 FIFO 大小均不变。

该选项依赖 PTOAS 对直接转换 RowPlusOne 的校验，以及 ISA TCVT
对错位存储容量的支持。在运行时依赖版本更新前默认关闭。
独立的 MX 选项请求受支持的 K128 L0 双缓冲，不改变 NZ 选择条件。

MX 双缓冲适用于完整的 A5 M48/M64、N256、K256 FP8/E8M0
矩阵运算，且四个 L0 搬运结果必须由对应矩阵运算独占。
保留 K256 Mat/FIFO 源，将每次矩阵运算拆成按 K 顺序执行的两次
K128 运算，维持原来的 FP32 累加顺序。两个独立固定的 L0B 槽各占
32768 字节；L0A 槽按实际 M*128 字节分配。每个 scale 句柄的地址
由对应数据地址右移四位得到。两个 B 子块均完成拷贝后才释放生产者
FIFO。用户显式绑定的存储、不支持的形状以及 PTOAS 内存规划模式
保留原来的降低路径；该回退不影响独立的 NZ 错位写选项。
使用 PyPTO 内存规划时，仍由 PTOAS 执行校验、同步分析和 C++ 生成。

显式 RowPlusOne Tile 描述符要求物理形状为静态正数，行数为 16 的
倍数、列数为 32 的倍数，且错位地址不越界。Buffer IR 和多槽缓冲区
描述符当前拒绝该模式；应使用独立 Tile 分配。自动生成的 A5 内核
继续使用 Tile IR。
