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
| `make_tensor_view` / `partition_view`（ExpandPackedFp4\* 之后） | nibble（供 pto-isa `GetByteSize`） |
| runtime Tensor / `torch.float4_e2m1fn_x2` | carrier 元素 |

多行 ND packed 张量请用 **carrier** 末维与 leading stride（例如每行 512 逻辑
nibble 写成 `pl.Tensor[[2, 256], pl.FP4E2M1X2]`）。Codegen 会把 GM view 扩到
nibble 单位，使多行 pitch 与 Tile / pto-isa 一致。在 `FP4E2M1X2` 上误用逻辑宽度，
或在无自动打包时用逻辑 `pl.FP4` 多行 ND，可能导致 GM 行 stride 错位——见 issue
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

**PTOAS 0.65 足以支持 PYPTO 和 DSA_RP 内存规划器**：两者生成有显式地址的
类型 alias，并通过 A5 逐位测试。此 alias 会拒绝 PTOAS 内存规划器，因为 0.65
的 `pto.treshape` 不支持 packed FP4。其他 reinterpret-view dtype 保持原有
规划器支持。本路径不修复通用 typed FP4 GM partition 偏移、不引入 PackFp4、
不增加 tensor alias。
[#2979](https://github.com/hw-native-sys/pypto/issues/2979) 的独立
`safe_offset_probe.py` 仍复现读错行；退出 0 仅表示缺陷复现。

### 验证与性能

原始复现基于 PyPTO
`db9ac53ad0615fb3f3a81993266ab49fcb4f29a5`、simpler
`6e383fc57c12a8bd4dce045289b8b4f5d55a15f1`、PTOAS `0.65` 和 PTO ISA
`c0d7148e95ef73bd12a73165fdce4b723a3b7e72`。CANN runtime/devkit 的
`version.info` 报告 `9.2.0`；Python 为 `3.11.17`，CPU PyTorch 为 `2.14.1+cpu`。
PyPTO 扩展、simpler 扩展及 A5 runtime 二进制均在独立 worktree 内构建。
PR 基于上游 main `a0ff351c519a580e9523c9ba889b662459680f3f`，runtime 和
工具链版本保持一致。已在此基线加上修复后重新构建，并重跑正确性用例及全部三路
benchmark。没有包含 PR #2962 的改动。

A5 测试覆盖全部 256 种 packed byte、低/高 nibble 顺序、正负零、可区分的行、
非零 GM 行/列偏移、两轮循环复用 UB，以及更换输入的三次执行。PYPTO 与 DSA_RP
均与独立 BF16 位模式 LUT 完全一致，零 bit mismatch。

2026-10-14 在同一分配的 Ascend950DT（device 7）测量。每格为 median effective_us
**[P10, P90]**，预热 5 次、计时 100 轮。三路均使用 tile rows 32、workers 32、
`sync_start=False` 和相同输入字节。CPU 编译/汇编与设备执行分离；计时前后均逐位
验证输出，九个用例全部通过。

| N | cce_native (µs) | dsl_manual (µs) | dsl_native (µs) | Native DSL / CCE |
| - | --------------- | --------------- | --------------- | ---------------- |
| 1024 | 42.436 [38.491, 46.573] | 47.172 [43.155, 50.496] | 42.620 [38.385, 47.310] | 1.004× |
| 8192 | 49.759 [46.181, 54.205] | 81.528 [77.660, 85.331] | 46.830 [42.852, 52.476] | 0.941× |
| 32768 | 70.281 [67.162, 73.478] | 202.855 [199.162, 206.042] | 63.800 [59.334, 67.151] | 0.908× |

N=32768 时，native DSL 的 median 比同次 CCE 低 9.2%，比手工 DSL 快 3.18 倍。
这是无统一 L2 flush 的预热测量，仅衡量此精简转换，不代表模型吞吐。
生成的 native DSL 使用 MTE2→V 和 V→MTE3 事件，并在循环复用时加入 V→MTE2、
MTE3→V 依赖；CCE 基线使用 PIPE_ALL barrier。

在 worktree 内构建并运行三路 benchmark，`PTOAS_ROOT` 必须选择 0.65，
并安装所固定的 runtime：

```bash
source .claude/skills/testing/load-env.sh
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --parallel "$PYPTO_BUILD_JOBS"
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
python tests/issue_2979/repro.py --phase build
task-submit --device auto --run 'python tests/issue_2979/repro.py --phase run --device "$TASK_DEVICE"'
```

`tests/issue_2979/results-pypto.json` 保存全部 900 个样本、源文件及输入哈希、
规划器、设备和分位数。上表对应的
[main 基线重测数据](../../assets/issue_2979/results-a0ff351c.json) 与
[原始基线测量数据](../../assets/issue_2979/results-db9ac53a.json) 分别保存。
四个原始基线文件的 SHA-256 保存在
`tests/issue_2979/baseline/sources.json`。用以下命令从所链接的 issue 恢复原始文件；
写入任何文件前会校验全部四个哈希，且不会覆盖已修改的本地文件。原始文件由 Git
忽略，以保留原始格式。

```bash
python tests/issue_2979/fetch_baseline.py
```

离线使用时，传入 `--issue-json path/to/2979.json`，文件内容为 GitHub issue API
响应。原始基线应在原始 PyPTO 版本上运行；三路对比使用修复版本上的
`tests/issue_2979/repro.py`。

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
| 无 PackFp4 的逻辑 `pl.FP4` | ⚠️ | Prefer Warning；A5 in-core 仍支持逻辑 FP4，并与 FP4E2M1X2 并存 |
| GM ExpandPackedFp4\*（carrier→nibble） | ✅ | `make_tensor_view` / partition 末轴（ND） |
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
4. 仅在接受不完整路径时保留逻辑 `pl.FP4`，直到 PackFp4 落地。

## 另见

- [类型](../user/language/00-types.md) — FP4 / FP4E2M1X2 短表
- [算子 / MX](ir/05-operators.md) — `matmul_mx` FP4 说明
