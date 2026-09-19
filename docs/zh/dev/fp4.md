# PyPTO 中的 FP4

FP4 / packed FP4 的权威说明页。Pass 算法见 [PackFp4](passes/07-pack_fp4.md)；
本页写角色、管道、单位约定与三层支持矩阵。

## 角色

| 类型 | 含义 |
| ---- | ---- |
| `pl.FP4` | 前端**逻辑** E2M1 nibble；shape / `valid_shape` 按 nibble 计数 |
| `pl.FP4E2M1X2` | **打包** carrier（每元素两个 nibble）；`PackFp4` 之后或手写物理 shape。偶数末维由作者保证；DN / NZ / col_major / cube 仍由 PackFp4 拒绝 |

`tensor.full` / `create` 填充值：打包后仅 **0** 有定义；非零 fill（双 nibble 复制 vs 仅低 nibble）本版本未定义。

PyPTO 另拒绝 FP4 族的 `reshape`、标量 `read`/`write`、白名单外坐标 op，以及 cube / DN / NZ / col_major 打包——见 pass 文档与报错（属仅 PyPTO 限制，不进三列表）。

## 管道

1. **PackFp4**（仅静态）：逻辑 `pl.FP4` → `FP4E2M1X2`；末维 covering size / offset /
   ND leading stride 必须为偶数 `ConstInt` 并 `/2`。
2. **本地 ExpandPackedFp4\***（单卡 codegen）：GM `make_tensor_view` / partition
   末维几何扩回 nibble，供 pto-isa `GetByteSize`；EmitC 对 Tile 列做同样扩展以配合 TCVT。
   **tile_buf 保持 carrier 单位**（不回退）。
3. **Cast**（`LegalizeTileCast`）：FP4↔BF16 静默（DSv4.1 c1a）；FP4→FP8\* 允许但发
   **Warning**（优先 LUT / host 预转）。

## 单位约定

| 层 | 单位 |
| -- | ---- |
| 前端 `pl.FP4` shape / `valid_shape` | nibble |
| PackFp4 之后 IR / `tile_buf` / orch create | carrier |
| `make_tensor_view` / `partition_view`（Expand 后） | nibble（供 pto-isa `GetByteSize`） |
| runtime Tensor / Torch `float4_e2m1fn_x2` | carrier 元素 |

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
| `transpose` / `ttrans` | ❌ | ❌ 无 f4E2M1x2 `ttrans` | ❌ |
| Distributed（remote / window / put / get） | ⏳ | ⚠️ 其它 dtype 的 comm/view 有 | — |
| `matmul_mx` 原生 FP4 data | ⏳ | ✅ MX 路径 | ✅ TCVT |

## 推荐路径

1. 动态 block/slot：DSv4.1 风格 **UINT8 半宽** cache ABI。
2. 有限静态 `pl.FP4`（偶末维）+ PackFp4 + 本地 Expand。
3. 可选手写 `pl.FP4E2M1X2`（物理 shape；layout/cube 仍校验）。
4. FP4→FP8 优先 **LUT / host**；设备 cast 仅 Warning。

## 参见

- [PackFp4 pass](passes/07-pack_fp4.md) — 仅算法
- [类型](../user/language/00-types.md) — FP4 短述
- [算子 / MX](ir/05-operators.md) — matmul_mx FP4 说明
