# PyPTO 中的 FP4

FP4 / packed FP4 的权威说明页。Pass 算法见 [PackFp4](passes/07-pack_fp4.md)；
本页写角色、管道、单位约定、cast 策略与三层支持矩阵。

## 角色

| 类型 | 含义 |
| ---- | ---- |
| `pl.FP4` | 前端**逻辑** E2M1 nibble（`GetBit()==4`）；shape / `valid_shape` 按 nibble 计数 |
| `pl.FP4E2M1X2` | **打包** carrier（每元素两个 nibble，`GetBit()==8`）；`PackFp4` 之后或手写物理 shape。逻辑路径的偶数末维由作者保证；DN / NZ / col_major / cube 仍由 PackFp4 拒绝。手写 `FP4E2M1X2` **仅 ND row-major** |

`tensor.full` / `create` 填充值：打包后仅 **0** 有定义；非零 fill（双 nibble 复制 vs 仅低 nibble）本版本未定义。

PyPTO 另拒绝 FP4 族的 `reshape`、标量 `read`/`write`、白名单外坐标 op，以及 cube /
DN / NZ / col_major 打包——见 pass 文档与报错（属仅 PyPTO 限制，不进三列表）。
`tensor.dim` 仍支持：非末轴不变；末轴在 PackFp4 后仍按逻辑 nibble 计数。

`FP4E2M1X2` 的 layout 转化硬拒：非 ND 注解、隐式列向量 `[M, 1]` DN、以及显式 `tensor.view` layout 翻转。

## 管道

1. **PackFp4**（仅静态）：逻辑 `pl.FP4` → `FP4E2M1X2`；末维 covering size / offset /
   ND leading stride 必须为偶数 `ConstInt` 并 `/2`。
2. **PTO codegen**：GM `make_tensor_view`、`partition_view`、offset、stride 和
   `tile_buf` 均保持 carrier 单位。PTOAS v0.67+ 仅把 TLOAD/TSTORE 传输描述符扩成
   nibble，供 pto-isa `GetByteSize`；指针寻址继续使用 carrier。
3. **Cast**（`LegalizeTileCast`）：见 [Cast 策略](#cast-策略ascend950)。

`reinterpret_view` 仅允许 **等字节** 的 `FP4E2M1X2` ↔ `UINT8` / `INT8` 别名（shape
不变）。需要前导维 flatten 时，请走 `reinterpret_view` → `UINT8` → `reshape`，不要
依赖 packed-FP4 的 `reshape`。PTOAS v0.67+ 分离 carrier 寻址和 nibble DMA
描述符，修复多行 TLOAD/TSTORE pitch
（[PTOAS #1575](https://github.com/hw-native-sys/PTOAS/issues/1575)）。

Tile 字节别名要求 UB 中连续且有显式地址的二维 row-major Vec 存储，并需要
PYPTO 或 DSA_RP planner，以同一 buffer 地址生成别名。带 stride 的子视图来源和
无显式地址的输入仍被拒绝。
PTOAS v0.67 仍拒绝 `!pto.f4E2M1x2` 的 `treshape` / `bitcast`；PyPTO 对未分配地址的
tile 别名明确报错。InCore 中的 Tensor reinterpret 会降低为上述 tile 别名，因此有
相同的 planner 要求。Orchestration Tensor reinterpret 仍不支持。

## 单位约定

| 层 | 单位 |
| -- | ---- |
| 前端 `pl.FP4` shape / `valid_shape` | nibble |
| PackFp4 之后 IR / `tile_buf` / orch create | carrier |
| 手写 `pl.FP4E2M1X2` IR / `tile_buf` / Torch ABI | carrier |
| `make_tensor_view` / `partition_view` / offset / stride | carrier |
| PTOAS TLOAD/TSTORE 传输描述符 | nibble（供 pto-isa `GetByteSize`） |
| runtime Tensor / Torch `float4_e2m1fn_x2` | carrier 元素 |

多行 ND packed 张量请用 **carrier** 末维与 leading stride（例如每行 512 逻辑
nibble 写成 `pl.Tensor[[2, 256], pl.FP4E2M1X2]`）。PTOAS v0.67+ 分离 carrier 寻址和 nibble DMA 描述符，修复多行传输 pitch
（[PTOAS #1575](https://github.com/hw-native-sys/PTOAS/issues/1575)）。
在 `FP4E2M1X2` 上误用逻辑宽度仍可能导致 GM 行 stride 错位——见 issue
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

同 shape 的 alias 接受 FP4E2M1X2 ↔ UINT8 / INT8，要求二维 packed row-major Vec tile、
`slayout=none_box`、无显式 stride，且 tile `start_offset` 为零。显式 `shape`
必须与源 shape 完全相同。carrier `valid_shape` 保留，随后 cast 展开其末轴。
需要改变 shape 时，先 reshape 字节视图，再解释回 FP4E2M1X2 后 cast；每次 alias
都必须保持输入 shape 和连续存储。InCore tensor alias 降低到同一 tile 路径。
逻辑 FP4 alias、原生 packed-FP4 reshape 和 orchestration tensor alias 仍被拒绝。
GM load 可使用非零偏移；它与 UB tile 的 `start_offset` 不同。

源 tile 必须具有显式地址的内存引用（MemRef），其字节跨度必须等于静态 carrier
元素数。非连续 UB 列切片（包括其 alias 和 reshape）以及 `tile.tpop_from_aic`
等无 MemRef 的输入，会在生成 FP4 转换前被拒绝。请直接从 UINT8 GM 加载所需窗口，
得到连续 UB tile 后再创建 alias。包括中间视图在内，alias 的 UB 字节偏移必须
为静态值；GM load 仍可使用运行时偏移。

请使用 **PTOAS v0.67+ 及 PYPTO 或 DSA_RP 内存规划器**：两者生成有显式地址的
类型 alias。此 alias 会拒绝 PTOAS 内存规划器，因为 v0.67 的 `pto.treshape`
仍不支持 packed FP4。其他 reinterpret-view dtype 保持原有规划器支持。
上例的 UINT8 GM 路径保持字节寻址；typed FP4 GM 传输使用前述 carrier 约定。

## Cast 策略（Ascend950）

`LegalizeTileCast` 对 FP4 族如下处理：

| Cast | 行为 |
| ---- | ---- |
| `FP4` / `FP4E2M1X2` ↔ `BF16` | 原生 TCVT（DSv4.1 c1a）。静默。Packed ↔ 更宽类型按末轴 **2:1** 调整（打包时要求静态偶数末维）。结果 stride 按目标 shape 重建为连续。 |
| `FP4` / `FP4E2M1X2` → `FP8E4M3FN` / `FP8E5M2` | 合法化为 FP4→BF16→FP32→FP8，并带 **Warning**（更推荐 LUT / 主机预转换）。 |
| `FP4` ↔ `FP4E2M1X2` | **拒绝**（`RejectFp4FamilyInternalCast`；几何不一致）；经更宽类型绕行或显式改 shape。 |
| Wider → `FP4E2M1X2` 且末维动态 | **拒绝**（仅静态正偶数）。 |

当前 pin 的 pto-isa CPU 模拟器在 FP4→BF16 TCVT 中把小数值取整。因此 cast
运行时测试在 A5 真机上进行精确比较；字节拷贝测试同时覆盖模拟器。参见
[当前 pin 的 CPU TCVT 实现](https://github.com/hw-native-sys/pto-isa/blob/418de18dde8fa33773dc5c384fdfe5506463b628/include/pto/cpu/TCvt.hpp)。

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
| FP4↔BF16 cast | ✅ | ✅ | ✅ A5 TCVT；CPU 模拟器限制见上文 |
| InCore packed FP4 ↔ 字节别名、字节 `reshape`、转回 packed 别名 | ✅ 同 shape 别名；连续且有地址的 UB；PYPTO / DSA_RP | ❌ v0.67 的 FP4 `treshape`；有地址别名绕过该指令 | ✅ 共享存储 |
| FP4→FP8\* cast | ⚠️ Warning（优先 LUT/host） | ✅ | ✅ |
| `FP4` ↔ `FP4E2M1X2` cast | ❌ 拒绝 | — | — |
| `FP4E2M1X2` layout 转化（非 ND、`[M,1]` DN、layout `tensor.view`） | ❌ 仅 ND row-major | — | — |
| `transpose` / `ttrans` | ❌ | ❌ 无 f4E2M1x2 `ttrans` | ❌ |
| 逻辑 FP4 上的 `tensor.dim` | ✅ 非末轴不变；末轴按 nibble | — | — |
| Distributed（remote / window / put / get） | ⏳ | ⚠️ 其它 dtype 的 comm/view 有 | — |
| `matmul_mx` 原生 FP4 data | ⏳ | ✅ MX 路径 | ✅ TCVT |

## 推荐路径

1. 动态 block/slot：DSv4.1 风格 **UINT8 半宽** cache ABI。
2. 有限静态 `pl.FP4`（偶末维）+ PackFp4 + PTOAS v0.67+。
3. 可选手写 `pl.FP4E2M1X2`（物理 **carrier** shape，避免 `#2754` 类 stride 问题；layout/cube 仍校验）。
4. Packed FP4 ↔ 字节别名请用 PYPTO / DSA_RP，在字节 view 上 reshape。
5. FP4→FP8 优先 **LUT / host**；设备 cast 仅 Warning。

## 参见

- [PackFp4 pass](passes/07-pack_fp4.md) — 仅算法
- [类型](../user/language/00-types.md) — FP4 短述
- [算子 / MX](ir/05-operators.md) — matmul_mx FP4 说明
