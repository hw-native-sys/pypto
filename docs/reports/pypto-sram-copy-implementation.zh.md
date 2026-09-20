# PyPTO DDR↔SRAM copy 接口实现说明

日期：2026-09-17

> 2026-09-20 接口更新：copy 已改为无张量返回值的原地写入算子。当前用法是
> `pl.copy(dst, src, ...)`，调用后直接使用 dst；不再写 `dst = pl.copy(...)`。
> 下文保留此前实现记录，其中输出别名描述和返回 copy 的示例属于旧接口。
> 当前接口以 [算子文档](../zh/dev/ir/05-operators.md) 为准。

## 后续更新：create_tensor 与 DDR → DDR 验证

以下为同日后续开发结果；后文各节保留第一阶段实现及当时测试记录。

- `pl.create_tensor` 新增关键字参数 `memory_type=pl.Mem.DDR`，接受 DDR/SRAM。
  默认行为及已有 layout 位置参数保持兼容。SRAM 保存为算子属性，仍使用 DDR 地址类型和
  运行时分配；当前环境不存在真实 SRAM。SRAM 声明应在 orchestration 中创建后传入 InCore。
- `pl.copy(dst, src, ...)` 增加 DDR → DDR 路径，底层沿用有界 Vec 临时块的 load/store。
  保留已有 DDR → SRAM 默认端点；DDR → DDR 必须显式传入 `target_memory=pl.Mem.DDR`。
- 增加整张量简写：省略全部 offsets/shape 即复制同形状张量；区域拷贝仍需同时传入三个区域参数。
- `examples/beginner/05_matmul.py` 先创建独立 DDR 缓冲区 a1/b1，在第一段 InCore scope
  复制 A/B，再在第二段 scope 中使用 a1/b1 做矩阵乘法。新增 `--platform` 参数。
- 数值测试直接导入该示例，覆盖随机矩阵、左单位阵、右单位阵，并检查输入未被改写。
  生成的 PTO IR 包含两组 GM load/store，确认未将复制优化成输入别名。

验证位置：`101.245.68.6:/data/pyptouser/s00832235/pypto-sim/pypto`。
进入仓库后执行 `source ../activate_sim.sh`（服务器上的实际脚本名称），并设置
`PYTHONPATH` 指向当前仓库。仅激活 conda 环境会导入 `site-packages` 中的旧版 PyPTO，
导致 `create() got an unexpected keyword argument 'memory_type'`。

```bash
cd /data/pyptouser/s00832235/pypto-sim/pypto
source ../activate_sim.sh
source .claude/skills/testing/load-env.sh
export PYTHONPATH="$PWD/python:$PYTHONPATH"
# C++ 源码修改后重新构建；只运行已构建版本时可跳过此行。
cmake --build build --parallel "$PYPTO_BUILD_JOBS"
python examples/beginner/05_matmul.py --platform a5sim
```

示例在 a2a3sim 和 a5sim 均输出 `OK`；两平台分别通过上述 3 组数值测试。
相关算子、转换、代码生成、JIT roundtrip 回归共 **808 passed**，另有 6 条现有弃用提示。
这些结果证明 DDR 数据搬运及复制后矩阵乘法的正确性，不代表真实 SRAM 硬件验证。

```python
a1 = pl.create_tensor(a.shape, dtype=a.dtype, memory_type=pl.Mem.DDR)
with pl.at(level=pl.Level.CORE_GROUP):
    a1 = pl.copy(a1, a, source_memory=pl.Mem.DDR, target_memory=pl.Mem.DDR)
```

接口详情同步维护于 [算子文档](../zh/dev/ir/05-operators.md)。

## 1. 实现结果

本次新增 `pl.copy` / `tensor.copy`，用于 DDR 与 SRAM 之间的张量区域搬运。支持 DDR → SRAM 和 SRAM → DDR，支持独立的源、目的偏移，以及静态或动态搬运范围。

实现沿用仓库已有的 `source_memory` / `target_memory` 端点模型：张量仍使用全局 DDR 地址空间，通过算子参数声明物理介质。**没有将 SRAM 空间信息迁入 `TensorType`，也没有新增独立 SRAM 地址分配器。**

当前工具链没有直接的 DDR↔SRAM 搬运指令，因此将 copy 降低为通过有界 Vec 临时块执行的 MTE2 load 和 MTE3 store。SRAM 端点标记保留到生成的 PTO IR 中。

## 2. 已确认的设计边界

| 问题 | 本次处理 |
| --- | --- |
| SRAM 所属层级 | 放在 chip，即 `SoC` 层，记录一块 256 MiB 共享 SRAM；不按 core 或 die 重复计数 |
| 同步 | 不新增跨核同步接口或语义；生成的 load/store 沿用已有流水线同步机制 |
| 搬运引擎 | 复用 MTE；当前实际降低路径同时包含 MTE2 和 MTE3 |
| SRAM 地址由谁提供 | 暂不确定；先复用现有全局张量寻址和调用者/运行时提供的存储 |
| 跨 InCore 生命周期 | 同一张量存储可跨 InCore 调用传递，不要求新增显式 SRAM alloc/free |
| 与策略文档的差异 | 经确认保留现有端点参数接口，不进行 `TensorType` 内存空间字段改造 |

仓库原有的 `MemorySpace.SRAM`、SRAM load/store 端点参数和相关拓扑支持是本次工作的基础，并非本次新增。

## 3. 接口与使用方式

### 3.1 接口签名

```python
pl.copy(
    dst,
    src,
    dst_offsets,
    src_offsets,
    shape,
    *,
    source_memory=pl.Mem.DDR,
    target_memory=pl.Mem.SRAM,
)
```

| 参数 | 含义 |
| --- | --- |
| `dst` | 具有实际存储的目的张量 |
| `src` | 源张量，dtype 和 rank 必须与目的张量一致 |
| `dst_offsets` | 目的张量各维的起始偏移 |
| `src_offsets` | 源张量各维的起始偏移 |
| `shape` | 各维搬运长度，与两端张量 rank 一致 |
| `source_memory` | 源介质，DDR 或 SRAM |
| `target_memory` | 目的介质，必须与源组成 DDR↔SRAM 搬运方向 |

返回值与 `dst` 复用存储，不创建新的目的分配；DSL 包装保留目的张量的具体 Python 类型。即使忽略返回值，写入仍然保留。

当前只接受 DDR → SRAM 或 SRAM → DDR，不接受 DDR → DDR、SRAM → SRAM 或以 Vec 等 tile 空间为端点。

### 3.2 双向搬运

以下为 InCore 函数的完整写法；调用者向 `dst` 传入已有存储。

```python
import pypto.language as pl


@pl.jit.incore
def stage(
    src: pl.Tensor[[4, 600], pl.FP32],
    dst: pl.Out[pl.Tensor[[4, 600], pl.FP32]],
) -> pl.Tensor[[4, 600], pl.FP32]:
    return pl.copy(dst, src, [0, 0], [0, 0], [4, 600])


@pl.jit.incore
def restore(
    src: pl.Tensor[[4, 600], pl.FP32],
    dst: pl.Out[pl.Tensor[[4, 600], pl.FP32]],
) -> pl.Tensor[[4, 600], pl.FP32]:
    return pl.copy(
        dst, src, [0, 0], [0, 0], [4, 600],
        source_memory=pl.Mem.SRAM,
        target_memory=pl.Mem.DDR,
    )
```

跨 InCore 使用时，可在外层通过已有 `pl.create_tensor(..., dtype=...)` 创建中间张量，将 `stage` 返回的张量传给 `restore`。这条路径已通过模拟器数值测试，但不代表已经实现真实硬件上的独立 SRAM 分配。

子区域接口使用相同语法。例如，偏移分别为 `[1, 8]`、`[0, 16]`，范围为 `[2, 577]` 时，目的区域对应 `dst[1:3, 8:585]`，源区域对应 `src[0:2, 16:593]`。如果需要把目的张量原有内容传入执行环境以保留周边数据，应使用适当的 `InOut` 参数声明。

### 3.3 输入检查

IR 类型推导检查两端是否为 tensor-like 类型、dtype/rank 是否匹配、端点组合是否合法，以及 offsets/shape 是否为整数元组。复用已有区域校验逻辑，检查元组长度、静态负偏移、非正静态长度和可静态判断的越界。

动态区域由调用者保证运行时合法，不新增运行时边界检查。源、目的区域不得重叠；当前没有实现重叠区域的 memmove 语义。

## 4. 各层实现

### 4.1 Python DSL 与 IR 算子

新增底层 `tensor.copy` 包装和用户侧 `pl.copy`，并完成导出。新算子携带以下语义：

- 返回结果复用第 0 个输入，即目的张量的存储。
- 第 0 个输入标记为写入，写入通道为 DMA。
- 核亲和性为 VECTOR，避免将 copy 误归为 CUBE 计算。
- `source_memory`、`target_memory` 作为算子属性保存。

### 4.2 Tensor→Tile 降低

在 `OpConversionRegistry` 中注册 copy 转换规则，并将其作为自行处理张量读取的算子，避免转换框架提前插入默认 load。

转换按张量各维生成循环：外层维度逐坐标遍历，最内层每次处理最多 256 个元素。每块依次执行：

```text
源 tensor 区域
    ↓ tile.load，source_memory 保留 DDR/SRAM 声明
Vec 临时块（最多 256 个元素）
    ↓ tile.store，target_memory 保留 SRAM/DDR 声明
目的 tensor 区域
```

尾块通过 `min(块宽, 剩余长度)` 计算有效范围。非二维 load 结果显式 reshape 为二维 tile，以适配后续代码生成。循环携带目的张量，维持写入结果与返回值之间的 SSA 关系。

临时存储大小不随整个搬运区域增长；256 是元素数，而不是字节数。当前策略优先保证可执行性与内存有界，尚未进行自动分块调优、双缓冲或多核分工优化。

### 4.3 SoC 与 Backend

为 `SoC` 增加 chip 级 `mems_` 和 `GetMems()`，两个已有模拟拓扑工厂均记录一块 256 MiB SRAM。Python 通过 `soc.mems` 查询该信息。

同步修改以下路径：

- `GetMemSize` / `GetMemAlignment` 先查询 chip 级内存，再查询 core 级内存。
- SoC 序列化写入 `mems`；反序列化允许旧数据缺少此字段。
- 逻辑 DDR → SRAM 路由推断为 MTE2，SRAM → DDR 路由推断为 MTE3。
- `tensor.copy` 的逻辑 pipe 推断读取显式端点属性。实际降低后仍由各个 load/store 指令分别决定其 pipe。

256 MiB 容量和 32 字节对齐是当前拓扑描述，不表示已经实现硬件 SRAM 地址预留、容量分配或释放。

### 4.4 打印器修复

连续 copy 会生成同名提示的循环变量。测试发现，打印 tile 类型中的复合 `valid_shape` 表达式时，临时打印器没有继承函数局部变量的重命名信息，导致打印后重新解析时引用错误变量。

本次让类型表达式打印器继承局部变量重命名表和自由变量标记，使连续 copy 的 IR 打印、解析和结构比较保持一致。

## 5. 修改文件清单

以下路径相对 `D:\code\pypto` 仓库根目录；清单不包含这份总结文档。

| 文件 | 修改内容 |
| --- | --- |
| `python/pypto/ir/op/tensor_ops.py` | 新增底层 copy 调用包装 |
| `python/pypto/language/op/tensor_ops.py` | 新增用户 DSL 接口及说明 |
| `python/pypto/language/op/__init__.py` | 导出 copy |
| `python/pypto/language/__init__.py` | 提供 `pl.copy` |
| `src/ir/op/tensor_ops/memory.cpp` | 注册算子、类型推导、区域及端点校验、写入和别名语义 |
| `src/ir/transforms/op_conversion_registry.cpp` | 分块循环及 load/store 转换 |
| `src/ir/transforms/convert_tensor_to_tile_ops_pass.cpp` | 避免为 copy 插入多余的默认 load |
| `src/ir/transforms/python_printer.cpp` | 修复类型表达式中的变量重命名 |
| `include/pypto/backend/common/soc.h` | 增加 chip 级内存字段和访问接口 |
| `src/backend/common/soc.cpp` | 构造与 256 MiB SRAM 拓扑描述 |
| `src/backend/common/backend.cpp` | chip 内存查询、序列化兼容、copy pipe 推断 |
| `python/bindings/modules/backend.cpp` | 暴露 `SoC.mems` |
| `python/pypto/pypto_core/backend.pyi` | 同步类型声明 |
| `tests/ut/ir/operators/test_tensor_copy.py` | 新增 copy 单元及完整流水线测试 |
| `tests/ut/ir/test_memory_space.py` | SRAM 容量预期由 0 更新为 256 MiB |
| `tests/st/runtime/ops/test_tensor_copy.py` | 新增跨 InCore 往返、子区域数值测试 |
| `docs/en/dev/ir/05-operators.md` | 英文接口与实现边界说明 |
| `docs/zh/dev/ir/05-operators.md` | 对应中文说明 |

共涉及 18 个源码、测试和接口文档文件。本次没有修改原先已有工作区差异 `3rdparty/libbacktrace`。

## 6. 服务器适配与验证

服务器：`101.245.68.6`。

验证目录：`/data/pyptouser/s00832235/pypto-sim/pypto`。

实际找到的环境脚本为上一级目录的 `activate_sim.sh`，不是 `active_sim.sh`。验证前执行该脚本和仓库资源限制加载脚本：

```bash
cd /data/pyptouser/s00832235/pypto-sim/pypto
source ../activate_sim.sh
source .claude/skills/testing/load-env.sh
cmake --build build --parallel "$PYPTO_BUILD_JOBS"
```

服务器已有未提交的 SRAM load/store 改动，适配时先备份差异，再应用增量补丁。最终核对了本次 18 个文件的内容校验值；忽略 Windows/Linux 换行差异后，本地与服务器源码一致。

| 验证项 | 结果与范围 |
| --- | --- |
| C++ 构建 | 最终构建通过 |
| 新增 copy 测试 | 37 项通过，覆盖双向端点、非法参数、1-D/2-D/3-D、连续 copy、动态范围、忽略返回值及 chip 内存元数据 |
| 相关回归 | 首轮 2,109 项通过、2 项失败；失败项是旧 SRAM 容量为 0 的断言，更新预期后复测通过 |
| 最后一次聚焦回归 | copy 和 memory-space 合计 68 项通过，包含上述修正的容量断言 |
| `a2a3sim` 数值测试 | 2 项通过：跨 InCore DDR→SRAM→DDR 往返、子区域复制且周边数据不变 |
| `a5sim` 数值测试 | 单独运行同样 2 项，均通过 |
| 静态检查 | 本次 Python 文件的 Ruff 检查通过；C++ 修改区域完成 clang-format；相关仓库检查和 `git diff --check` 通过 |

相关回归覆盖 tensor/tile 算子、memory space、backend、parser/printer、Tensor→Tile 转换和 PTO 代码生成；没有宣称运行完整仓库全部测试。

SoC 元数据和导出已测试，但没有完成 backend 导入往返测试：仓库的 singleton backend 不允许通过 registry 创建导入实例。反序列化兼容路径已修改并通过编译。

## 7. 当前限制与后续工作

1. **地址分配仍待确定。** 当前通过全局张量地址和端点标记表达 SRAM，不提供独立 SRAM 分配、持久缓存管理或容量占用跟踪。
2. **当前是分块 MTE 实现。** 没有新增直接 DDR↔SRAM 硬件指令；未来工具链提供直接指令时，可替换 lowering，保留用户接口。
3. **不增加跨核同步协议。** 本次验证覆盖已有执行框架中的跨 InCore 传递，不证明任意跨核并发访问都安全。
4. **运行时动态边界由调用者保证。** 编译期仅检查可静态判定的非法范围；不支持重叠区域搬运语义。
5. **尚无自动 staging 策略。** 不自动选择需要搬入 SRAM 的张量，也不自动生成缓存驻留、淘汰或预取方案。
6. **性能与硬件映射尚未验证。** 已确认模拟器数值正确；真实 SRAM 地址映射、带宽收益及最佳分块参数仍需后续验证。
