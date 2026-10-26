# GM Subview RFC：完整可验证示例

本文配合[设计 RFC](gm-subview-rfc.md)，用同一个案例贯穿接口、IR、pass 和 PTO codegen。
实现测试已覆盖该链路，并编译真实生成的 PTO。独立 PTO fixture 继续作为 backend
探测；设备数值与物理 DMA 证据单独跟踪。

## 1. 输入、输出及与已有接口的差异

输入是普通 packed ND BF16 Tensor，shape 为 [8,512]；选取第 1、4 行，复制到
普通 packed 输出 Tensor [2,512]。

| 对同一源的操作 | 含义 | 结果 / InCore 行为 |
| -------------- | ---- | ------------------ |
| tensor.subview(cache, [2,512], [1,0], steps=[3,1]) | 不读取数据，在父坐标系采样严格更小的窗口 | GM Tensor，origin 512，stride [1536,1] |
| tensor.view(cache, shape=[2,2048]) | 对原有 4096 个元素做 canonical shape/layout 重新解释 | GM Tensor，不是所选两行 |
| tensor.view(cache, shape=[2,512]) | 尝试把 4096 个元素重新解释为 1024 个 | 已有静态元素数检查拒绝 |
| tensor.slice(cache, [2,512], [1,0]) | 选连续第 1、2 行 | InCore 中 GM 输入 slice 变为 Tile load |
| load(cache, [1,0], [2,512]) | 显式搬运连续第 1、2 行 | 新 Tile |
| load(subview, [0,0], [2,512]) | 显式搬运采样后的第 1、4 行 | 新 Tile，描述符保持 GM |

Subview 在保留父地址映射的前提下改变起点和采样步长。View 对已有存储重新解释
shape/layout，没有 offsets/steps 参数，不是有界采样操作。Slice 保持当前物化语义，
本 RFC 不增加隐式 slice-to-subview 改写。

完整的拟议用户程序如下：

~~~python
import torch
import pypto.language as pl
from pypto.runtime import RunConfig


@pl.jit
def copy_selected_rows(cache: pl.Tensor, output: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        v = pl.tensor.subview(cache, [2, 512], [1, 0], steps=[3, 1])
        pair = pl.load(v, [0, 0], [2, 512])
        pl.store(pair, [0, 0], output)
    return output


if __name__ == "__main__":
    rows = torch.arange(8).reshape(8, 1)
    cols = torch.arange(512).reshape(1, 512)
    cache = (16 * rows + cols % 16).to(torch.bfloat16)
    output = torch.zeros((2, 512), dtype=torch.bfloat16)
    copy_selected_rows(cache, output, config=RunConfig())
    torch.testing.assert_close(output, cache[[1, 4], :], rtol=0, atol=0)
~~~

此程序需在 API 和可执行流水实现后，在支持的 runtime/device 上运行。输入值都是
BF16 可精确表示的整数；行列编码能检测读错行。测试必须完整编译并执行，不能只到 lower()。

## 2. API 如何构造 IR

DSL wrapper 解包 source 和整数 Scalar 参数，调用 Python IR builder，再把 Call
包装为 Tensor。公共签名见 RFC 4.1；省略 steps 时补成与 rank 一样长的全 1 序列。
IR builder 复用现有 tuple helper 和通用 op 构造：

~~~python
from pypto.pypto_core import ir as core
from pypto.ir.utils import _to_make_tuple


def subview_ir(source, shape, offsets, steps, span):
    return core.create_op_call(
        "tensor.subview",
        [
            source,
            _to_make_tuple(shape, span),
            _to_make_tuple(offsets, span),
            _to_make_tuple(steps, span),
        ],
        {},
        span,
    )
~~~

此底层 builder 示例说明四个 IR 操作数；公共 wrapper 还会将整数归一化为 INDEX
后再运算。实现已注册 C++ 算子。
不引入任意 Python callback 或新 IR 节点。DeduceTensorSubviewType 按父 stride
乘以 step 推导结果 stride；origin 保留在定义 Call 中，不增加类型 offset 字段。

完整 InCore 函数体预期操作顺序如下，仅省略外层 printed-program class。
这是 IR 记法，不是 printer 的实际语法：

~~~text
func copy_selected_rows_incore(cache: Tensor<[8,512],BF16>,
                              output: Tensor<[2,512],BF16>):
  v    = tensor.subview(cache, (2,512), (1,0), (3,1))
  pair = tile.load(v, (0,0), (2,512), (2,512))
  out  = tile.store(pair, (0,0), output)
  return out
~~~

初始/default target-memory 可以暂时未设置，直到 InferTileMemorySpace；显式搬运 Call
已经是 tile.load。JIT 特化、SSA 命名和外层 orchestration 沿用现有机制。

| 变量 | IR 类型 / 元数据 | 地址 / 分配 |
| ---- | ---------------- | ----------- |
| cache | TensorType([8,512], BF16)，canonical ND stride [512,1] | 调用方 GM pointer |
| v | TensorType([2,512], BF16)，TensorView stride [1536,1]，layout ND，寻址种类 AffineSubview | 无 MemRef，无新 payload 分配 |
| pair | TileType([2,512], BF16)，最终 Vec | 新结果 Tile 分配 |
| out | output 的 store 结果 | 普通输出 GM alias |

算子测试需精确校验：Call 名为 tensor.subview，四个 operand 按 source/shape/offsets/steps
排序，结果是普通 TensorType，元数据如表，以及只读 alias root 为 cache。

边界证明如下：

~~~text
Row:    shape=2 > 0; step=3 > 0; offset=1 >= 0;
        1 + (2-1)*3 = 4 < 8.
Column: shape=512 > 0; step=1 > 0; offset=0 >= 0;
        0 + (512-1)*1 = 511 < 512.
Strict reduction: 2 < 8.
Address(v[r,c]) = cache_pointer + (512 + r*1536 + c) BF16 elements.
~~~

若把行 offset 改成 5，最后一行为 8，拒绝。无范围约束的设备动态 offset/step
会得到 unknown，拒绝。同窗口 [8,512] identity subview 也违反严格缩小规则。
动态 guard 场景还需 RFC 5.1 的作用域证明。

## 3. IR 是否需要额外处理，以及 pass 的调整

| Pass / 组件 | 本例 before → after | 必须调整 |
| ----------- | ------------------- | -------- |
| SSA / outlining | 局部描述符和 scalar geometry 留在消费 kernel 内 | 不能把描述符变成函数参数，不能丢失必要 guard |
| 边界验证 | 待证 geometry → 行列包含和严格缩小已证明 | conversion 前与 codegen 前增加共享上下文 visitor；false/unknown 拒绝 |
| ConvertTensorToTileOps | tensor.subview + tile.load → tensor.subview + tile.load | 显式元数据透传；预测/preload 分析保留 GM 身份；校验改写后的 load source |
| ConsumerSpaceCollector | load 的 Vec 需求 → 传播到对应消费点 | 不能全局把描述符替换成 Tile |
| MaterializeTensorStrides / canonical verifier | 显式 [1536,1]、AffineSubview → 保持 | 保留 affine kind/stride，分开 canonical 和 affine 校验 |
| InitMemRef / memory planning | 借用 GM 描述符 → 无 MemRef；结果 Tile → 正常 placement | 跳过描述符分配和继承 offset 的 MemRef 构造 |
| Composite / ND/layout passes | 二维 ND geometry → 相同坐标映射 | 不重建 packed stride；不支持的改写诊断拒绝 |
| 最终 backend 检查 | 已证明 geometry → 目标可执行 load | 单独验证 dtype、layout、pitch、对齐和整数位宽 |

Pass UT 按第 2 节操作序列构造 Before / Expected，使用 ir.assert_structural_equal
比较。内存空间推导前显式 target 可未设置；推导后 Expected 将 pair 放在 Vec。
另外校验 v 仍是普通 Tensor、仅有一个 tile.load，且输入为 v。这是多个独立检查点，
不能假设执行一个转换就已经达到最终状态。

元数据/内存测试需检查 InitMemRef 后 v.memref 缺省，输入侧新 payload 分配只有 pair。
Tile 物理大小服从选定的 Tile/Buffer memory planner；不能用零大小描述符 MemRef 替代。

其他路径也必须有独立 before/after 测试：

| 路径 | 必须具备的行为 / 产物 |
| ---- | --------------------- |
| subview → nested subview → load | 父 stride 和 origin 仅组合一次 |
| subview → slice → compute | slice 变为从 v 读取的 load，使用 v 的采样 pitch |
| subview → set_validshape → load | 保留收窄的 GM 描述符，不产生 tile.set_validshape 或 preload |
| subview → view → load | 证明已有 packed-view 限制；本例 [1536,1] 不能 flatten 丢掉行间 gap |
| InCore slice → subview / slice → load | 不改变已有 slice 解释；改写成 Tile 输入后诊断 |

## 4. 最终 codegen 与可运行的完整 PTO

在 src/backend/common/pto_ops_memory.cpp 的 memory handler 注册 tensor.subview：

1. 解析普通 Tensor source 的当前起点 pointer 和真实 stride。
2. 计算 origin = 1*512 + 0*1 = 512 个元素，结果 stride 为 [3*512,1]。
3. 在当前 InCore 函数内生成 pto.addptr 和 pto.make_tensor_view。
4. 像现有 view handler 一样，通过 RegisterBasePtr 绑定 v 当前起点 pointer，
   通过 RegisterTensorView / RegisterVarToMlir 绑定描述符。
5. 复用 tile.load，生成 partition_view(v,[0,0],[2,512]) 和 tload，不能重复叠加
   origin。复用 tile.store 生成输出。

**完整且可独立汇编的模块**见
[gm_subview_static.pto](../../../../tests/fixtures/gm_subview_static.pto)。它包括常量、
结果 Tile 分配、输出描述符/store 和 return；没有未定义 SSA 或 tile-buffer-type 占位符。
这是独立手写 backend fixture；新 API 真实生成的产物由实现测试单独编译验证。

在仓库根目录执行：

~~~bash
/usr/local/ptoas/0.67/bin/ptoas tests/fixtures/gm_subview_static.pto \
  --pto-arch=a2 --pto-level=level2 --enable-insert-sync \
  -o /tmp/gm_subview_static.cpp
~~~

A3 编译检查把 a2 替换成 a3。模块 a2a3 family attribute 与命令行接受的 a2/a3
是不同概念。Level2 把结果 Tile placement 交给 PTOAS，不分配 GM 描述符 buffer。

PTOAS 0.67 的 A2 输出包含一个二维 2×512 的 BF16 Vec Tile 声明、一个 TLOAD 和
一个 TSTORE。GM 描述符起点为 cache + 512 个元素，行 stride 为 1536 个元素。
payload 为 2048 逻辑字节；采样地址包络为 4096 字节。不能分配 4096 字节中间 Tile，
也不能搬运整个包络。

自动 fixture 测试见
[test_gm_subview_rfc_fixture.py](../../../../tests/ut/codegen/test_gm_subview_rfc_fixture.py)。
构建当前 worktree 并加载测试环境后运行：

~~~bash
source .claude/skills/testing/load-env.sh
cmake --build build --parallel "$PYPTO_BUILD_JOBS"
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest tests/ut/codegen/test_gm_subview_rfc_fixture.py -v
~~~

此 fixture 检查编译器接受及 TLOAD/TSTORE 路径，不能证明最终机器 DMA 数量或设备正确性。
实现后的 codegen 测试必须编译真实 API 程序，对比该模块的地址/stride 和分配语义，
再对真实生成的模块调用 PTOAS；仅手写模块通过，不能证明 API 到 backend 链路正确。

## 5. 独立数据/地址验证与证据

下面完整 CPU reference 仅依赖 Python 标准库。它逐元素对比地址计算与直接行索引，
不调用 subview：

~~~python
source = [[16 * r + c % 16 for c in range(512)] for r in range(8)]
flat = [value for row in source for value in row]
selected = [[flat[512 + r * 1536 + c] for c in range(512)] for r in range(2)]
assert selected == [source[1], source[4]]
assert len(selected) * len(selected[0]) * 2 == 2048
assert (512 + 1536 + 511 - 512 + 1) * 2 == 4096
~~~

2026-10-26 更新的验证记录：

| 检查 | 状态 |
| ---- | ---- |
| 静态边界 / CPU 地址映射 | 已逐元素验证 |
| 完整 PTO fixture，PTOAS 0.67 A2/A3 | 汇编成功；一个 TLOAD、一个 TSTORE |
| 新 API / 默认 pass / 真实生成 PTO | 静态及有界动态 BF16 通过 A2/A3 编译；一个结果 Tile、一个 TLOAD/TSTORE |
| Worktree 构建 / pytest | Rebase 后构建及 67 项算子/codegen/backend inventory 专项通过；C++ 测试 5/5 通过 |
| Runtime 用例 codegen-only | 通过，覆盖 orchestration 和 InCore 生成 |
| Simulator 数值执行 | RuntimePinMismatch 阻塞：已安装 runtime 与固定源码版本不一致 |
| 全量 UT | 935ed13c 上 19,873 通过、94 失败（69 项混合 shape 构造、23 项 runtime 依赖/导入、2 项打印）；ced0e5ab 上构建、专项、C++、codegen-only 验证通过，未重复全量 |
| 设备结果 / 单硬件 DMA | 未执行，不做结论 |

实现验收必须同时满足：用户程序、算子断言、pass 检查点、真实生成 PTO 的编译和运行时
数据比对一致。

构建当前 worktree 后执行实现链路：

~~~bash
source .claude/skills/testing/load-env.sh
export PYTHONPATH="$PWD/python:$PYTHONPATH"
export PTOAS_ROOT=/usr/local/ptoas/0.67
python -m pytest tests/ut/ir/operators/test_tensor_subview.py \
  tests/ut/codegen/test_tensor_subview.py tests/ut/codegen/test_gm_subview_rfc_fixture.py \
  tests/ut/backend/test_buffer_migration_inventory.py -n "$PYPTO_TEST_JOBS" -q
~~~

默认流水线生成的 PTO 已指定 Tile 地址，使用 `--pto-level=level3`；上述手写 fixture
未指定地址，使用 level2。定向 runtime 用例为
`tests/st/runtime/ops/test_tensor_subview.py`，需要 simulator/device runtime；它不能
代替物理单 DMA 证据。
