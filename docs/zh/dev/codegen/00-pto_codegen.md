# PTO 代码生成 (CodeGen)

PTO 代码生成 (CodeGen) (`PTOCodegen`) 从 PyPTO 中间表示 (IR) 生成 PTO-ISA 方言的 MLIR 代码。它将高层 PyPTO 程序转换为适合加速器执行的低层 PTO 指令。

## 设计原则：严格的 1-to-1 映射

代码生成必须是从 IR 到生成代码的**严格 1-to-1 转换**。每个 IR 节点直接映射到对应的输出代码结构——代码生成层不应执行优化、分析或间接转换。

| 属于代码生成的职责 | 属于前置 Pass 的职责 |
| ------------------ | -------------------- |
| IR 节点 → 输出代码映射 | 数据流分析（如追踪返回值到参数的映射） |
| 类型/格式转换（DataType → MLIR 类型） | IR 重组或规范化 |
| 名称修饰 (Name Mangling) 和 SSA 记录 | 优化或简化 |

**原因：** 嵌入分析逻辑的代码生成会变得脆弱——它重复了 Pass 已有的逻辑，且更难以独立测试。保持代码生成为直接的转换，确保其可预测性和可维护性。

**当发现代码生成中存在分析逻辑时：** 创建跟踪 Issue，在有带宽时将其重构为专用 Pass。[#814](https://github.com/hw-native-sys/pypto/issues/814) 就是一个实例：编排代码生成中的返回值到参数追踪逻辑已重构为 [`NormalizeReturnOrder`](../passes/29-normalize_return_order.md) pass。

## 概述

### 核心特性

- **自动 MLIR 生成**: 将 PyPTO IR 转换为 PTO-ISA MLIR 方言
- **结构化代码生成 (CodeGen)**: 按顺序输出常量、张量 (Tensor) 视图和分配
- **隐式降级**: 从 `tile.load`/`tile.store` 自动生成 `pto.partition_view`
- **基于 Tile 变量的分配**: 为每个 Tile 变量生成带显式 `addr` 的 `pto.alloc_tile` 操作
- **类型 (Type) 感知转换**: 从 TileType 元数据推导 tile_buf/tensor_view 类型
- **PTOAS 类型标注**: 为所有操作生成带类型的 `ins`/`outs` 子句

### 生成顺序

代码生成按以下固定顺序生成 MLIR:

1. **常量**: 索引和浮点值的 `arith.constant`
2. **张量视图**: 所有张量参数的 `pto.make_tensor_view`
3. **分配**: 所有 Tile 变量的 `pto.alloc_tile` (按变量维度, 带 `addr` 属性)
4. **操作**: 包含加载、计算、存储操作的函数体

张量视图与分配前缀会**先**渲染到缓冲区、再定稿常量块, 因此只出现在某个 shape 或
stride 表达式里的常量 (例如复合参数维度 `M * 2` 中的 `2`) 也会在使用前被声明到常量块。

形状表达式可以保留 `tensor.dim(x, axis)`。代码生成将该轴的声明维度发射为 INDEX
操作数，支持常量、符号和复合表达式，无需先生成 `x` 的张量视图或提前折叠 IR。
只有该维度中的标量符号会成为动态维度 ABI 参数，张量操作数本身不会。
打印的类型标注在重新解析时保留这些读取，不会把参数引用作为 Python 值求值。
仅描述形状的赋值标注只有在直接维度读取临时量能够证明结构等价时才替换推导形状；否则保留
推导出的运行时形状。常量维度直接发射为 INDEX，包括无符号常量。

### 自动切分边界上的动态尺寸参数

自动切分会捕获 tensor 参数类型所需的标量符号，即使 kernel 正文没有直接读取它们。
`[B, 16]` 这样的独立维度可以通过 tensor 描述符绑定 `B`。
`[B * 2, 16]` 等支持反解的维度也保留描述符提取，因此 caller 无需单独绑定标量 `B`；
`[(B + 15) // 16 * 16 + T * 4, 16]` 这样的复合维度无法绑定 `B` 或 `T`。
切分器会保守地添加显式 `B`、`T` 参数，并传入 caller 中的值。
这些符号必须在 caller 中具有运行时绑定：标量参数、正文中的值定义，或 tensor 参数的独立维度。
仅出现在不可逆类型表达式中的符号会在切分阶段被拒绝，避免成为 orchestration 中未声明的参数。
原尺寸表达式保持不变，不会替换为新的 `ROWS` 符号。

IR 中标量参数位于引用它们的类型标注之前。运行时分发和原生 kernel 签名则按
tensor 指针、显式标量 payload、剩余描述符尺寸参数的顺序排列。
PTO codegen 和 wrapper 共享 `collect_tensor_shape_dyn_vars`，确定最后一组参数。
显式标量优先于描述符提取，并按 Var 身份去重，因此同时用于 shape 和正文的符号只传一次。

对于手写 kernel，描述符提取仍支持独立维度，以及单符号加减常量、乘非零常量。
整除没有通用逆运算：`B // 5 == 2` 不能证明 `B == 10`。
应显式传入该符号，或通过另一支持的维度暴露它，否则 wrapper 生成会报错。
现有运行时描述符和标量打包格式保持不变。

## 架构

### 显式 Buffer 输入

`GenerateBufferFunction` 在发射前验证显式构造的 Buffer IR。GM 路径支持
物理形状静态、紧密 ND 布局的二维 FP16/BF16/FP32/INT32/INT8 Tensor 参数，以及规范化的 Tensor
参数返回值。它复用现有 GM 张量视图前缀和原生“Tensor 在前、标量在后”的
ABI。Tensor 返回值保留在 IR 中供编排处理别名，不产生原生返回值。

`buffer.load` 和 `buffer.store` 将显式窗口发射为 `pto.partition_view`，
随后发射 `pto.tload` 或 `pto.tstore`。目标、偏移和 valid extent 均来自
普通操作数。`buffer.add`、`mul`、`copy` 直接使用显式 buffer 目标。
每个 `pto.alloc_tile` 对应同一作用域内的 `buffer.alloc`；只有地址操作数
存在时才发射地址，不受旧发射标志影响。GM 传输不创建 buffer、不推导
valid 状态更新，也不重建逻辑 Tile。

默认 `enable_buffer_ir=True` 时，流水线仅对选中完整 kernel 执行 `LowerTileToBuffer`。
未支持 kernel 保持 Functional Tile IR；`False` 对所有 kernel 使用 Tile 路径。
描述符、方向、动态窗口和 ABI 限制见 [Buffer 契约](../ir/02-types.md#buffer-算子契约)。
原生编译测试验证语法和操作数数据流；数值执行是单独的集成验收要求。

存储别名以显式 `buffer.subview` 和 `buffer.reshape` 操作进入代码生成，
分别发射为 `pto.subview` 和 `pto.treshape`；带 valid 元组的 subview 会发射 `valid [..]` 子句，
由其中的常量或 SSA 操作数确定原生结果每一维的类型，直接保留源句柄和结果描述符，
不分配、不复制，也不推导存储窗口。描述符边界和别名合法性在发射前完成验证。

Mat、Left、Right 和 Acc 描述符发射为 `loc=mat|left|right|acc`，并带上分形的
`blayout`/`slayout`/`fractal` 字段；它们必须由完整分形块组成，不完整的块会以与
旧路径相同的“应分配多少”诊断拒绝。`buffer.matmul` 和 `buffer.matmul_acc`
分别发射为 `pto.tmatmul` 和 `pto.tmatmul.acc`，累加形式按 PTOAS 要求把目标作为
第一个输入。`buffer.extract` 发射为带 `index` 偏移的 `pto.textract`，
Mat 到 Left/Right 的 `buffer.copy` 发射为 `pto.tmov`。

### 类结构

**头文件**: `include/pypto/codegen/pto/pto_codegen.h`

```cpp
namespace pypto::codegen {

class PTOCodegen : public CodegenBase {
 public:
  PTOCodegen();
  explicit PTOCodegen(const backend::Backend* backend);

  std::string Generate(const ir::ProgramPtr& program);

  // CodegenBase interface
  std::string GetCurrentResultTarget() const override;
  void Emit(const std::string& line) override;
  std::string GetExprAsCode(const ir::ExprPtr& expr) override;
  std::string GetTypeString(const DataType& dtype) const override;

  // PTO-specific helpers for operator codegen
  std::string NewTemp();
  std::string GetOrCreateTensorView(const ir::VarPtr& tensor);
  std::string GetOrEmitConstant(int64_t value, DataType dt);   // int/index 重载
  std::string GetOrEmitConstant(double value, DataType dt);    // float 重载
  std::string GetTensorViewTypeString(const ir::TensorType* tensor_type) const;
  std::string GetTileBufTypeString(const ir::MemRef* memref) const;
  std::string GetExprTypeAnnotation(const ir::ExprPtr& expr);
  std::string GetCurrentResultTileBufTypeString() const;
};

}  // namespace codegen
```

### 实现组件

**文件**: `src/codegen/pto/pto_codegen.cpp`

| 组件 | 用途 |
| ---- | ---- |
| `PTOCodegen` | 主访问者类 (继承 `CodegenBase`), 用于 IR 遍历 |
| `MemRefCollectorVisitor` | 收集 MemRef 对象及其关联的 TileType 用于分配 |
| 辅助函数 | `DataTypeToMLIRImpl()`, `MemorySpaceToMLIR()` |

## Python API

### 基本用法

```python
from pypto.ir import compile, OptimizationStrategy
from pypto.backend import BackendType
import pypto.language as pl

@pl.program
class MyKernel:
    @pl.function
    def vector_add(self,
                   a: pl.Tensor[[32, 32], pl.FP32],
                   b: pl.Tensor[[32, 32], pl.FP32]):
        tile_a = pl.load(a, [0, 0], [32, 32])
        tile_b = pl.load(b, [0, 0], [32, 32])
        tile_c = pl.add(tile_a, tile_b)
        pl.store(tile_c, [0, 0], a)

# Compile with PTO backend
output_dir = compile(
    MyKernel,
    strategy=OptimizationStrategy.Default,
    backend_type=BackendType.Ascend910,
)
```

`compile()` 函数会自动应用选定的优化策略, 并根据 `backend_type` 调用相应的代码生成器。
`Default` 是唯一的优化策略。

### 直接访问代码生成器

```python
from pypto.pypto_core import codegen

# After pass transformations
pto_codegen = codegen.PTOCodegen()
pto_code = pto_codegen.generate(transformed_program)
print(pto_code)
```

## 操作映射

### Tile 操作到 PTO 指令

| PyPTO 操作 | 生成的 PTO-ISA |
| ---------- | -------------- |
| `tile.load(tensor, [row, col], [h, w])` | `pto.partition_view` + `pto.tload` |
| `tile.store(tile, [row, col], tensor)` | `pto.partition_view` + `pto.tstore` |
| `tile.slice(tile, [h, w], [row, col][, valid_shape=...])` | `pto.subview`（零拷贝视图；仅在传入 `valid_shape` 时输出 `valid [...]` 子句） |
| `tile.assemble(target, source, [row, col])` | （可选）`pto.tmov target -> dst`；支持的 ND Vec 复制 / FIXPIPE 写回使用 `pto.tinsert`，其他情况使用 `pto.subview` + `pto.tmov` |
| `tile.set_validshape(tile, vr, vc)` | 发 `pto.set_validshape`；操作数是视图时报错（见下） |
| `tile.mul(lhs, rhs)` | `pto.tmul` |
| `tile.addc(src0, src1, carry)` | `pto.taddc`（`src0 + src1 + carry`） |
| `tile.subc(src0, src1, carry)` | `pto.tsubc`（`src0 - src1 + carry`） |
| `tile.addsc(src0, scalar, carry)` | `pto.taddsc`（`src0 + scalar + carry`） |
| `tile.subsc(src0, scalar, carry)` | `pto.tsubsc`（`src0 - scalar + carry`） |
| `tile.adds(tile, scalar)` | `pto.tadds` (Tile + 标量) |
| `tile.and_(lhs, rhs)` / `tile.ands(lhs, scalar)` | `pto.tand` / `pto.tands`；scalar 使用同位宽 signless `iN` |
| `tile.or_(lhs, rhs)` / `tile.ors(lhs, scalar)` | `pto.tor` / `pto.tors`；scalar 使用同位宽 signless `iN` |
| `tile.xor(lhs, rhs, tmp)` / `tile.xors(lhs, scalar, tmp)` | `pto.txor` / `pto.txors`；scalar 使用同位宽 signless `iN` |
| `tile.fillpad_expand(src, shape)` | `pto.tfillpad_expand ins(%src) outs(%dst)`（`shape` 元组仅用于类型推导；更大的 `dst` 及其 pad 来自结果类型） |

**`tile.slice` / `tile.assemble` 下沉细节。** `tile.slice` 和 `tile.assemble` 的回退路径通过 `pto.subview`
下沉，它是源 tile 的纯视图别名（不搬数据，也不会额外发 `pto.alloc_tile`）。
`pto.subview` 要求结果 `tile_buf` 与源 `tile_buf` 在 `dtype`、`memory_space`、
`blayout`、`slayout`、`fractal`、`pad` 和 `compact` 上完全一致，因此
`DeduceTileSliceType` 会将源 `TileView` 的这五个字段透传到结果，使新生成的
`TileType` 天然满足约束。后端 codegen 还会在下沉时执行 `CheckSubviewTileCompat`
做兜底校验：

- 源和结果都必须显式携带 `TileView`。
- `dtype`、`blayout`、`slayout`、`fractal`、`pad` 与 `compact` 必须严格相等。
- `pad` 必须为 `PadValue::null`——`pto.subview` 是视图而不是 fillpad；如果
  需要 zero/min/max 填充，请在切出来的子 tile 上再调用 `tile.fillpad`。

对 `tile.assemble`，前置的 `pto.tmov target → dst` 仅在缓冲复用未把
`target` 与目标缓冲合并时才会发出，用于保留写入窗口外的数据。ND（`row_major`、
`none_box`）布局下的 FP32、FP16 和 BF16 Vec-to-Vec 复制使用
`pto.tinsert`，直接传入原目标和显式偏移，保留目标行跨度，只复制源的有效区域，
其余目标元素保持不变。A2/A3 上的列插入不能使用 `subview + tmov`：PTOAS 会把
视图解析为父 tile 的物理维度，随后因源和目标形状不一致而拒绝编译（#2717）。
其他复制继续使用 `pto.subview + pto.tmov`。A2/A3 PTOAS 不接受 INT16/INT32 TINSERT，
固定版本的 ISA 也不接受有效列数为动态值的 INT8/UINT8 TINSERT；
因此保留整数类型原有的整块 TMOV 复制。
A2/A3 的插入列偏移必须满足 32 字节对齐（8 个 FP32 或 16 个 FP16/BF16 元素）。
代码生成会拒绝违反该限制的常量偏移；动态偏移也必须由调用方保证满足此条件。

**Acc 写回上的 FIXPIPE epilogue。** **发生类型转换的** `Acc → Mat` assemble，
以及**任何** `Acc → GM` store，都是 cube 的 fix-pipe 在排空 L0C——它可以在写出的
同时乘一个 FP32 scale 并施加 ReLU；两个 op 都用 `pre_quant` / `pre_relu` 承载这
件事。顺序是 **先 ReLU、再乘、最后按目标类型 clamp**——激活属于*量化前*阶段
（pto-isa 的 `ReluPreMode`），看到的是未缩放的累加器，因此这一对算出的是
`maximum(tile, 0) * scale` 而非 `maximum(tile * scale, 0)`。二者对所有
`scale > 0` 完全相同，所以只有负 scale 能把它们区分开；
`tests/st/runtime/ops/test_fixpipe_epilogue.py` 中的 `acc_to_gm_negative_scale`
在 a2a3 真机上测得了这个顺序。

scale 还**决定**走哪条下沉路径，而不只是搭个便车——`INT32` 累加器之所以能
抵达 `FP16` 的 Mat tile，正是因为它带了 scale（`DEQF16`）——因此由同一个判据
`ir::CubeMatWritebackUsesFixpipe` 同时供 emitter 与 `FixpipeEpilogueValid` 使用；
同 dtype 的 Acc-to-Mat assemble 不发生转换，仍是普通的 `pto.tmov`，**带不了任何 epilogue**。
`pto.tinsert` 与 `pto.tstore` 对打包后的 scale 操作数拼写不同，两种形式都无法从
对方推出——二者连同其出处都逐字钉在各自的发射点上。寄存器编码见
`codegen::EncodeFixpipePreQuant`，各后端的 dtype 表见 `99-verifier.md`。

PTOAS 0.65 通过为 scale 发出 `uint64_t`、为行列索引发出 `uint16_t`，解决了
[PTOAS#1570](https://github.com/hw-native-sys/PTOAS/issues/1570) 跟踪的 scaled
`pto.tinsert` 重载歧义。因此 A2/A3 与 A5 均开放已验证的
`INT32 Acc → FP16 Mat`（`DEQF16`）形式。其他带 scale 的 Acc-to-Mat dtype 组合
仍由 `FixpipeEpilogueValid` 拒绝；范围更广的各后端能力表继续独立管辖 Acc-to-GM
store。

**`tile.set_validshape` 下沉细节。** `pto.set_validshape` 修改的是操作数的
`valid_row` / `valid_col` 操作数，因此操作数必须是拥有它们的 handle：alloc、
`scf.if` 结果、跨核 pop slot。而**视图**——`tile.slice` 下沉出的 `pto.subview`，
或 `pto.treshape`——把有效范围存在自身类型里，ptoas 会拒绝对它执行这条指令，所以
PyPTO 提前报错，并在信息里指向切片。视图身份在这两个发射点被记录，而不是从渲染出
的维度推断：带运行时 `valid_shape` 的切片渲染成 `v_row=?, v_col=?`，与由 alloc
承载的 handle 完全一样。要收窄视图，请给切片传 `valid_shape=`（它会落到
`pto.subview` 的 `valid [...]` 子句，并且支持运行时范围），或在取视图之前对源
tile 调用 `set_validshape`。

### 跨核操作到 PTO 指令

| PyPTO 操作 | 生成的 PTO-ISA | 描述 |
| ---------- | -------------- | ---- |
| `tile.tpush_to_aiv(tile, split=N[, id=I])` | `pto.tpush_to_aiv ins(%tile : type) {[id = I, ]split = N}` | Cube → Vector 推送 |
| `tile.tpush_to_aic(tile, split=N[, id=I])` | `pto.tpush_to_aic ins(%tile : type) {[id = I, ]split = N}` | Vector → Cube 推送 |
| `tile.tpop_from_aic(split=N[, id=I])` | `%buf = pto.tpop_from_aic {[id = I, ]split = N} -> type` | 从 Cube 管道弹出 |
| `tile.tpop_from_aiv(split=N[, id=I])` | `%buf = pto.tpop_from_aiv {[id = I, ]split = N} -> type` | 从 Vector 管道弹出 |
| `system.tfree_to_aic(tile_from_tpop[, id=I])` | `pto.tfree_from_aic {[id = I, ]split = N}` | 将消费侧槽位释放回 Cube |
| `system.tfree_to_aiv(tile_from_tpop[, id=I])` | `pto.tfree_from_aiv {[id = I, ]split = N}` | 将消费侧槽位释放回 Vector |
| `system.aic_initialize_pipe(...)` | `pto.aic_initialize_pipe {[id = I, ]dir_mask = D, slot_size = S[, slot_num = N][, local_slot_num = L]} (c2v_consumer_buf = %ssa : i32, v2c_consumer_buf = %ssa : i32)` | Cube 管道初始化（仅在显式设置时输出 `slot_num`/`local_slot_num`，否则由 PTOAS 取默认值） |
| `system.aiv_initialize_pipe(...)` | `pto.aiv_initialize_pipe {[id = I, ]dir_mask = D, slot_size = S[, slot_num = N][, local_slot_num = L]} (c2v_consumer_buf = %ssa : i32, v2c_consumer_buf = %ssa : i32)` | Vector 管道初始化（仅在显式设置时输出 `slot_num`/`local_slot_num`，否则由 PTOAS 取默认值） |
| `system.reserve_buffer(...)` | `%name = pto.reserve_buffer {name = "N", size = S, location = #pto.address_space<loc>, auto = false, base = B} -> i32` | 预留缓冲区（`memory_planner=PTOAS` 下发射 `auto = true` 且省略 `base`） |
| `system.import_peer_buffer(...)` | `%name = pto.import_reserved_buffer {name = "N", peer_func = @F} -> i32` | 导入对等缓冲区 |
| `system.syncall(core_type=C)` | `pto.syncall() mode = #pto.sync_all_mode<hard>, core_type = #pto.sync_core_type<C>` | 跨核全员屏障（hard/FFTS 形态） |
| `system.syncall(mode="soft", core_type=C, gm_workspace=ws, used_cores=N)` | `pto.syncall(%gm_ptr[, %used] : !pto.ptr<i32>[, i32]) mode = #pto.sync_all_mode<soft>, core_type = #pto.sync_core_type<C>` | 当前 PTO-ISA 的 soft/GM 轮询屏障（部分占用即可；GM workspace 至少 64 字节；显式 `N=0` 时从设备启动寄存器推导并省略 `%used`） |

**说明：**

- Push 操作使用带类型 tile buffer 的 `ins()` 子句；前端 Pop 操作生成 SSA 结果，并带 `-> !pto.tile_buf<...>` 结果类型
- `id` 是可选属性。省略时 PTOAS 默认使用 frontend pipe id `0`。只有手写多条独立 frontend pipe 时才需要显式 `id`；自动生成的双向 mixed-kernel setup 会保持单条 `dir_mask = 3` pipe。
- 如果被 push 的 tile 通过动态 `valid_row` / `valid_col` operand 分配，或经
  `tile.set_validshape` 更新，`tpush` 会发射已经更新运行时 valid shape 的同一个
  tile handle。对于 split `tpush`，codegen 会临时使用完整物理传输 box，随后恢复
  producer tile 的逻辑 valid shape。
- `split` 就是 pto-isa 的 `TileSplitAxis`，原样打印。`0` = 不切分，`1` / `2` = 上下 /
  左右，`3` / `4` = 同样两个轴、但 extent 为**奇数**：

  | Code | pto-isa | Lane 0 | Lane 1 | Lane 1 的数据段起点 |
  | ---- | ------- | ------ | ------ | ------------------- |
  | 0 | `TILE_NO_SPLIT` | 整块 tile | （单一读者） | — |
  | 1 / 2 | `TILE_UP_DOWN` / `TILE_LEFT_RIGHT` | `e0` | `e1` | `e1 * pitch` |
  | 3 / 4 | `TILE_UP_DOWN_ODD` / `TILE_LEFT_RIGHT_ODD` | `e0` | `e1` | `(e1 + 1) * pitch` |

  `eL` 是 lane `L` 在切分轴上的**运行时** valid extent——ISA 直接从被弹出的 tile 上读取
  （`popVecTileFromGMFiFo`），因此偶数 code 要求 `e0 == e1`，奇数 code 要求
  `e0 == e1 + 1`。这些 extent 由
  [LowerAutoVectorSplit](../passes/24-lower_auto_vector_split.md) 物化，
  [ExpandMixedKernel](../passes/25-expand_mixed_kernel.md) 选择匹配的 code。
- Cube-to-Vector FIFO 搬运的是紧凑矩形：producer 以 `valid_col` 为行间距写入
  `valid_row` x `valid_col` 数据块，每个消费 lane 再以相同间距读回自己的数据段
  （`gmStrideR = valid_col`，左右切分的 code 下加倍）。因此若传输两侧的 valid shape
  不一致，pop 的 stride 就会错位——这会静默破坏*有效*区域的数据，因为 ISA 中对应的
  断言在 release 构建里被编译掉了。所以部分有效的 Acc-to-Vec 传输无论是否切分，
  TPOP 以及 TPUSH 的**列**维度都使用完整物理 box，并在传输两侧立即恢复逻辑 valid
  shape——消费侧通过纯元数据的 `pto.treshape` 恢复（前端 tpop 结果不是 PTOAS 的本地
  绑定 tile，`pto.set_validshape` 无法就地修改它）。第一个例外是切分轴上的 extent：它必须保持
  逐 lane 的值并留在 TPOP 操作数上，因为 ISA 正是靠它定位 lane 1 的数据段起点——也正因如此，
  编译期无法核验的逐 lane extent 绝不能到达那里。`pl.split` 或 `pl.split_aiv` 中切分轴 extent 为运行期
  值的边界，会让被弹出的 tile 保留完整 box（`split_axis::WithFullSplitAxisValid`），使偶数
  code 的数据段落在 box 的一半处，而把 lane 自身的 extent 交给消费者携带。
- 第二个例外是**非切分** Acc-to-Vec TPUSH 的**行**维度：它必须保持 producer 写入时
  的值。TPUSH 执行的是 L0C 上的 `TStoreAccNz2nd`，其源 pitch 对 compact tile 为
  `ceil(validRow/16)*16`，否则为 `TileData::Rows`；而 `mad` 是按 L0A 操作数的**有效**
  行数所隐含的 pitch 写出乘积的。在 push 之前把 `validRow` 撑大到物理 box，会让该
  pitch 改按 box 推导，于是 fix-pipe 以 `mad` 从未写入过的 stride 遍历 L0C——64 行的
  box 只有 16 行有效时，fractal `j` 会读到 `4j`（issue #2510）。超出 `validRow` 的行
  会在 slot 中保留旧数据，这正是窄化 `valid_shape` 对其无效区域给出的承诺，同时传输
  量也从整个 box 降到 `validRow` 行。
- 在各 lane 于 GM slot 中自行定位数据段的后端（A2/A3）上，**切分**的 Acc-to-Vec 传输对
  *compact* tile 同样保持 producer 的行数。没有任何 lane 会把
  slot 中位于 `validRow` 及之后的行当作数据读取——一个 lane 的数据段是从 `lane * S` 开始的
  `clamp(V - lane * S, 0, S)` 行，其中 extent 为静态时 `S` 是均衡的 `ceil(V / 2)`，延迟
  （deferred）时是 box 的一半——因此从来不需要 box 的其余行，而把行扩成 box 会像上面一样
  使 compact 的 L0C 读取错位（64 行 box 只有 16 行有效、跨 `pl.split(UP_DOWN)` 时，2048 个
  有效元素中有 1792 个是错的）。非 compact 的 tile 无论 `validRow` 为何都按 `Rows` 读取
  L0C，因此其切分传输仍保持完整 box。在由硬件把行拆给两个 lane 的后端上
  （`BackendHandler::SplitsCubeToVectorTransportInHardware`，即 A5 的 dual-mode TMOV），
  传输的行数*就是* lane 划分，因此切分 push 保持完整 box，两种 pitch 不同的 compact tile
  在该后端上会被**拒绝**。*运行期*行 extent 也按同样方式传输：`pl.split` 和
  `pl.split_aiv` 两种形式随后都会 pop 完整 box（见上文 popped tile 规则）；lane 1 载入它的整个
  一半，超出有效 extent 的行是旧数据，但从不被当作数据使用。
- 当 tpop 结果的 `TileView.valid_shape` 与物理 tile shape 不一致时，PTO codegen 会生成 PTOAS 前端操作数：`%buf = pto.tpop_from_*(%valid_row, %valid_col) {[id = I, ]split = N} -> !pto.tile_buf<..., v_row=?, v_col=?, ...>`。这同时覆盖动态表达式和 `[0, 0]` 这类静态非满形状；operand 携带后续计算和 store 使用的逻辑范围。对于静态形状、非空的部分 pop，上述 Cube-to-Vector 完整 box 传输优先，因为 `pto.treshape` 不带 valid-row/valid-col operand，只能恢复*静态*逻辑范围。
- 对于手写 pop 的 split consumer，`SplitVectorKernel` 会按 subblock 本地化这些动态
  tpop valid-shape operand（例如 `[16, 16]` tile 做上下切分时，全局
  `[8, 16]` 会变成 `[8, 16]` 和 `[0, 16]`）。奇数切分轴走同一条路径——`[17, 128]`
  的 tile 在 `split = 3` 下，lane 0 弹出 `[9, 128]`，lane 1 弹出 `[8, 128]`。
- `system.tfree_*` 的 `split` 来自其 tile 参数，因此前端必须释放由 `tile.tpop_*` 产生的那个确切 SSA 值，即使 PTO 指令本身并不显式接收该 tile 作为操作数
- `ExpandMixedKernel` 现在会在 split 生成的消费侧 `tile.tpop_*` 之后自动补 `system.tfree_*`，保持 `tpop -> direct users -> tfree -> next tpop`
- `reserve_buffer` 和 `import_reserved_buffer` 返回 `i32` SSA 值；`initialize_pipe` 以操作数引用这些值
- `memory_planner=PYPTO` 或 `DSA_RP` 时，`AllocateMemoryAddr` 会在 PTO 输出前
  解析 `reserve_buffer(base=AUTO)`，因此 PTO 输出
  `auto = false, base = <value>`；`memory_planner=PTOAS` 跳过该 pass，PTO 输出
  `auto = true` 且省略 `base`（ptoas 不接受两者同时出现），由 ptoas
  `PlanMemory` 放置该预留区
- `reserve_buffer` location 对于 AIC 函数为 `mat`，对于 AIV/InCore 函数为 `vec`
- `import_reserved_buffer` 使用 MLIR 符号语法（`@func_name`）表示 `peer_func`
- 缓冲区名称和 peer_func 字符串由 `CheckSafeIdentifier` 验证（仅允许字母数字和下划线）

### 参数类型处理

| PyPTO 类型 | MLIR 参数类型 | 后处理 |
| ---------- | ------------- | ------ |
| `TensorType` | `!pto.ptr<dtype>` | 生成 `pto.make_tensor_view` |
| `ScalarType` | `dtype` (如 `f32`) | 直接用作 `%argN` |
| `TileType` | 不允许作为参数 | 必须在内部计算 |

## 代码生成细节

### 张量视图生成

对于每个 `TensorType` 参数, 代码生成器会生成:

```mlir
%0 = pto.make_tensor_view %arg0,
     shape = [%c32_index, %c32_index]
     strides = [%c32_index, %c1_index]
     {layout = #pto.layout<nd>}
     : !pto.tensor_view<?x?xf32>
```

**关键要点**:

- 形状来自 `TensorType.shape_`
- 步幅按行主序计算: 二维张量为 `[dim1, 1]`
- 常量 (`%c32_index`, `%c1_index`) 自动生成, 包括只出现在复合 shape/stride 表达式里的常量
- 复合维度下沉为算术运算, 例如 `M * 2` → `arith.muli %M, %c2_index`
- 张量视图类型每个维度使用 `?` (如二维为 `?x?xf32`)

#### 二维张量的 Layout 处理

`make_tensor_view` 上的 `layout` 属性告诉 PTOAS 内存布局约定。代码生成器根据
张量的 IR 类型和形状决定 shape、strides 和 layout:

| 情况 | 输出 Shape | 输出 Strides | Layout | 说明 |
| ---- | ---------- | ------------ | ------ | ---- |
| ND `[R, C]` | `[R, C]` | `[C, 1]` | `nd` | 标准行主序 |
| DN `[R, C]` (均 > 1) | `[C, R]` | `[1, C]` | `dn` | Shape 交换以符合 PTOAS 列主序约定 |
| 列向量 `[M, 1]` | `[M, 1]` | `[1, M]` | `dn` | 自动检测, 无需 DN 标注 |

**列向量自动 DN**: 任何最后一维为编译时常量 `1` 的二维张量 (即形状 `[M, 1]`)
会自动以 `layout = dn` 和步幅 `[1, M]` 输出。这是因为 PTOAS 对于形状/步幅模式
`[M, 1] / [1, 1]` 始终推断为 DN, 使得退化的 ND 表示产生歧义。代码生成器通过始终
使用无歧义的 DN 步幅来解决此问题。用户无需在 DSL 中为 `[M, 1]` 张量标注 `pl.DN`。

列向量 `[16, 1]` 示例 (DSL 中无 DN 标注):

```mlir
%col_view = pto.make_tensor_view %arg1,
    shape = [%c16_index, %c1_index], strides = [%c1_index, %c16_index]
    {layout = #pto.layout<dn>}
    : !pto.tensor_view<?x?xf32>
```

### 分配生成

基于附加到 TileType 变量的 MemRef 对象。代码生成器从关联的 TileType 推导 Tile 维度和数据类型:

```mlir
%mi_tile = pto.alloc_tile addr = %c8320_i64 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                       v_row=16, v_col=1, blayout=col_major,
                       slayout=none_box, fractal=512, pad=0>
%mi_tile_nd = pto.alloc_tile addr = %c8320_i64 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                       v_row=1, v_col=16, blayout=row_major,
                       slayout=none_box, fractal=512, pad=0>
```

**Tile 变量到 alloc_tile 的映射**:

- 内存空间 (`TileType.memory_space_`) 映射到 `loc` 属性 (使用 PTO 地址空间名)
- Tile 数据类型和维度从每个变量自身的 TileType 元数据推导
- 每个 Tile 变量对应一次分配 (不是每个唯一 MemRef)
- `addr` 属性来自 `MemRef.addr_`，输出为 `arith.constant ... : i64`
- 共享同一 MemRef 的变量共享相同的 `addr` SSA 值

#### 由谁规划内存：`compile(memory_planner=...)`

物理 `addr` 由谁分配，通过 `memory_planner` 选项选择
（`ir.compile(..., memory_planner=passes.MemoryPlanner.PYPTO | DSA_RP | PTOAS)`，
默认 `PYPTO`）。它同时作用于 pass 流水线（经 `PassContext`）与 codegen：

| 模式 | 流水线 | `pto.alloc_tile` | `pto.reserve_buffer` | ptoas |
| ---- | ------ | ---------------- | -------------------- | ----- |
| `PYPTO`（默认） | 运行 `MaterializeSemanticAliases` + `MemoryReuse` + `AllocateMemoryAddr` | 发射 `addr = <const>`（来自 `MemRef.byte_offset_`） | `auto = false, base = <const>` | `--pto-level=level3`（信任已烘焙地址） |
| `DSA_RP` | 运行 `MaterializeSemanticAliases` + `AllocateMemoryAddr`；跳过 `MemoryReuse` | 发射进程内 canonical-greedy DSA-RP 的 `addr = <const>` | `auto = false, base = <const>` | `--pto-level=level3`（信任已烘焙地址） |
| `PTOAS` | 运行 `MaterializeSemanticAliases`；**跳过** `MemoryReuse` + `AllocateMemoryAddr` | 省略 `addr`（`PTOCodegen.generate(emit_tile_addr=False)`） | `auto = true`（不带 `base`） | `--pto-level=level2`（ptoas `PlanMemory` 做复用 + 定址） |

内存规划拆成两个 pass：**`MaterializeSemanticAliases`** 把**语义强制**的别名
（循环累加器、原地算子）归一到同一 MemRef；**`MemoryReuse`** 只做**机会性**的、
基于生命周期的独立 buffer 合并，仅供 `PYPTO` 使用。`DSA_RP` 跳过该合并，
在 `AllocateMemoryAddr` 中按容量与复用惩罚放置独立身份。
`InitMemRef` + `MaterializeSemanticAliases` 三种模式都运行，因此强制别名得以
保留；`PTOAS` 模式由 ptoas `PlanMemory`（level2 强制要求、拒绝任何 `addr`
操作数）完成生命周期复用与地址分配。

> **注意：** `PTOAS` 模式跳过了 `MemoryReuse` 里的 Ascend910 `load + tpop_from_aic`
> 原地写冒险守卫,以及 `AllocateMemoryAddr` 的 reserve-buffer 基址解析,这些交由
> ptoas 处理。`compile()` 会输出告警 —— 相关 kernel 请上机验证。

#### 多槽位声明映射为一块 ptoas 区域（`PTOAS` 模式）

多槽位的声明式分配（`pl.MemRef(slots=N)`，见
[Python 语法](../language/00-python_syntax.md#槽位)）不会被降为 N 条 `alloc_tile`，而是
对应到 ptoas 自己的多缓冲二元组：函数头声明一块区域，每个使用点选一个槽位：

```mlir
%l0c_mb = pto.alloc_multi_tile valid_row = %c64_index valid_col = %c64_index
        : !pto.multi_tile_buf<!pto.tile_buf<loc=vec, dtype=f32, rows=64, cols=64, ...>, count=2>
scf.for %i = %c0_index to %c4_index step %c1_index {
  %slot = arith.remsi %i, %c2_index : index
  %t = pto.multi_tile_get %l0c_mb[%slot]
     : !pto.multi_tile_buf<..., count=2> -> !pto.tile_buf<loc=vec, ...>
  ...
}
```

有符号整数和 `INDEX` 槽位采用[修正后的向下取模余数](../passes/31-lower_pipeline_to_slots.md#生成的-pto-ir)，保持 Python `%` 的语义。
当被除数可证明非负且除数为正常量时（如上例），该修正会被跳过，因为此时 `arith.remsi` 的结果已与向下取模一致。

有两点关键：

- **不带 `addr`。** 区域由 ptoas `PlanMemory` 放置，且它被禁止合并这些槽位——这正是把作者
  声明的隔离带进 `level2` 的方式。
- **操作数是槽位下标，而不是** `InitMemRef` 由它算出的字节偏移。修正后的余数直接传入 `multi_tile_get`，
  不会乘以槽位的字节数。ptoas 识别下标（`i % 2`）以判断哪些访问可能落在同一槽位，这才是轮转能拿到按槽位的（动态）event id 的原因
  ——第 *i* 轮的 load 由此与第 *i-1* 轮的计算重叠。

PTOAS planner 下，`PlanMultiBufferRegions` 在遍历函数体之前判定适用性；ptoas 无法描述的形态（各槽位 tile 形状
不一致、各槽位声明的 valid shape 不一致、循环内有两个槽位同时活跃、内存空间不属于
Vec / Mat / Acc、valid shape 是运行期值、某个槽位作为 phi 被带出 `if` 或循环、槽位数不在
ptoas 的 `[2, 16]` 内）会报 `ValueError` 并指明具体形态，因为回退成逐槽位
`alloc_tile` 会让 ptoas 有机会把这些槽位规划到同一块内存上。

**每轮迭代只用一个槽位。** 共活槽位被拒绝，不是因为 ptoas 无法为它*定型*，而是无法为它
*同步*。这类形态中有两种出过错：

| 形态 | ptoas 行为 | 上游 |
| ---- | ---------- | ---- |
| 同一轮迭代内填充并读取两个槽位 | ≤ 0.55 只保护**第一个** `multi_tile_get`，第二个 load 与下一轮迭代的写入竞争（0.54 真机实测算错）。0.56 起用整个区域的一个静态 event 保护循环体——正确，但没有区域形式赖以存在的逐槽位重叠 | [PTOAS#1118](https://github.com/hw-native-sys/PTOAS/issues/1118)，0.56 修复 |
| 预取：循环前填充槽位 0，之后每轮迭代填充槽位 `(i+1) % 2`、同时读取槽位 `i % 2` | ≤ 0.62 像一槽位轮转那样为两个槽位的 event 都做 prime 和 drain，在这里差一：迭代次数为偶数时算错，为奇数时设备挂死。0.63 只 prime 循环最先写入的那个槽位；0.64 和 0.65 又对两个槽位都做 prime | [PTOAS#1519](https://github.com/hw-native-sys/PTOAS/issues/1519)，0.63 修复，0.64 回退 |

`CoLiveSlotCollector` 在原有 PTOAS planner 路径上保留保守拒绝。上表记录的预取回退仍是
工具链约束；开启软件流水并不保证工具链同步正确。在 PyPTO planner 下，codegen 识别
软件流水变换写入 IR 的槽位标记；对已经 lower 的程序生成代码无需再次传入开关。
codegen 校验静态等大几何信息、region 固定基地址、
槽位步长和对齐，允许规范的 Vec 共活槽位。槽位表达式可以是常量，或非负循环变量加
非负常量后对槽位数取模，整个算术表达式必须使用 index 类型。这些槽位取模复用
标量表达式生成器生成 `arith.remui`。

在 level3 下，每次循环迭代至多使用一个存活槽位的调度可以发射
`pto.alloc_multi_tile addr = <base>`。共活预取槽位（包括静态循环），以及运行时或带条件的调度，
使用固定地址父 buffer 与 `pto.subview`，避免上文记录的 multi-tile 同步回退。
两者都保留由 PyPTO 规划的完整槽分配。原版 PTOAS 对动态 subview 的同步仍较保守，
见 [#1587](https://github.com/hw-native-sys/PTOAS/issues/1587)。
不支持的用户声明区域仍走普通固定地址 `alloc_tile`，继续保持槽位隔离。
未变换的程序保持原有路径。preload/main/drain 变换及整循环回退规则见
[LowerPipelineToSlots](../passes/31-lower_pipeline_to_slots.md)。

### 加载操作转换

**PyPTO IR**:

```python
tile_a = pl.load(tensor_a, [0, 0], [32, 32])
```

**生成的 MLIR** (两个操作):

```mlir
# 1. Create partition view
%3 = pto.partition_view %tensor_view, offsets = [%c0_index, %c0_index],
                 sizes = [%c32_index, %c32_index]
                 : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>

# 2. Load into tile buffer
pto.tload ins(%3 : !pto.partition_tensor_view<32x32xf32>)
          outs(%tile_buf : !pto.tile_buf<loc=vec, ...>)
```

**关键转换**:

- 张量参数通过 tensor_view 查找
- 偏移/大小来自 `tile.load` 参数
- 输出 tile_buf 来自变量的 MemRef, 类型从 TileType 推导

### 存储操作转换

**PyPTO IR**:

```python
pl.store(tile_c, [0, 0], tensor_out)
```

**生成的 MLIR**:

```mlir
# 1. Create partition view for output
%5 = pto.partition_view %output_view, offsets = [%c0_index, %c0_index],
                 sizes = [%c32_index, %c32_index]
                 : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>

# 2. Store from tile buffer
pto.tstore ins(%tile_buf : !pto.tile_buf<loc=vec, ...>)
           outs(%5 : !pto.partition_tensor_view<32x32xf32>)
```

### 计算操作

#### 示例: Tile 乘法

PyPTO:

```python
tile_c = pl.mul(tile_a, tile_b)
```

MLIR:

```mlir
pto.tmul ins(%tile_a_buf : !pto.tile_buf<...>,
             %tile_b_buf : !pto.tile_buf<...>)
         outs(%tile_c_buf : !pto.tile_buf<...>)
```

**结果处理**:

- 结果变量的 MemRef 决定输出 tile_buf
- 输入操作数通过变量名查找解析
- 所有 `ins`/`outs` 子句包含类型标注

### 源码位置 (`loc`)

每条生成的**操作**都会带上由 IR `Span` 构造的 MLIR 尾随位置, 例如
`pto.tadd ins(...) outs(...) loc("kernels/attn.py":41:9)`。ptoas 会原样把
`loc()` 传递到自己的诊断信息里, 因此校验失败时报告的是用户 `.py` 中的行, 而不是
生成的 `.pto` 中的行 —— 在 `@pl.jit` 下用户根本看不到后者 (该路径上 span 已由
解析器从合成的 `<jit:name>` 文本重映射回真实源文件)。

**使用哪个 span** —— 在两个层级绑定, 后者细化前者:

| 层级 | 绑定位置 | 来源 |
| ---- | -------- | ---- |
| 语句 (主) | `PTOCodegen::VisitStmt` | `Stmt::span_` |
| Call (细化) | `PTOCodegen::VisitExpr_(CallPtr)` | `Call::span_`, 仅当它嵌套在语句 span 内 |

包含性 (containment) 检查是正确性的关键。`Call::span_` 在被保留时精确到列, 但
合成 tile 算子的 pass (`ConvertTensorToTileOps`) 会用所在**函数**的 span 重建
`Call`, 同时保留 `AssignStmt` 自身的 span。这类 span 起始于语句之前, 包含性检查
不通过, 于是被丢弃并回退到语句 span —— 否则大多数操作都会指向 `def` 行。

**不带位置的内容**: 区域花括号、分隔符和基本块标签 (`loc(...)` 只在一条完整操作
的末尾合法, 因此这些行走 `EmitStructural()` 而不是 `Emit()`); 常量段中的
`arith.constant` (在所有使用点之间去重, 没有唯一正确的 span); 以及 span 未知或
没有文件名的节点。

**关闭方式** —— `Generate(program, emit_tile_addr, emit_source_loc)`、
`compile(..., emit_source_loc=...)`, 或环境变量 `PYPTO_EMIT_PTO_LOC=0`。关闭后
输出与不带位置的形式逐字节一致; 由于 ptoas 独立于 PyPTO 发布, 这是应对某个
ptoas 版本解析器拒绝尾随位置时的应急开关。

## 分块 Tile 尺寸校验

PyPTO 发射的每一条 `pto.alloc_tile` 都会按 PTOAS 将要检查的分块网格先行校验。
PTO 以「块」为单位寻址分块 tile，因此*物理*尺寸不是整数个块的 tile 根本没有地址。

该规则与 PTOAS 的 `verifyBoxedTileLayout` 完全一致：

| 布局 | 块尺寸（行 x 列） |
| ---- | ----------------- |
| fractal 1024（`Acc`） | 16 x 16 |
| fractal 512，`slayout = row_major`（`Mat` / `Left`） | 16 x (32 / sizeof(dtype)) |
| fractal 512，`slayout = col_major`（`Right`，转置对偶） | (32 / sizeof(dtype)) x 16 |
| `slayout = none_box` | 非分块，无此约束 |

并保留 PTOAS 自身的豁免：*行*方向的规则对 `Vec` 以及单行 tile（此时 NZ 映射退化）
跳过，而列方向的规则始终生效。MX scale fractal 与亚字节载体交由 PTOAS 自行诊断。

**为什么放在这里而不是交给 PTOAS。** PTOAS 会拒绝同样的形状，但它的报错只提及自身内部
概念，也不给出修复方式：

```text
'pto.alloc_tile' op expects result boxed tile rows to be a multiple of innerRows (16), but got 100
```

在发射点报错则能同时给出 tile、出问题的轴、需要达到的尺寸，以及达到它的方式：

```text
a Mat tile of physical shape [100, 128] and dtype fp16 must be a whole number of
16x16 fractal boxes, but its row extent 100 is not a multiple of 16. ...
allocate 112 on that axis and declare 100 as the tile's valid_shape ...
```

`ComputeAllocTileFields` 是所有分配的唯一收口——逐变量声明、被提升出来的
`extra_alloc_tiles`、以及控制流路径都经过它——因此校验看到的正是最终发射的内容，不会与之
漂移。张量层的 `pl.matmul` / `pl.matmul_acc` 不会因 *M 轴*触发它：M 轴已由
[`ConvertTensorToTileOps`](../passes/12-convert_tensor_to_tile_ops.md#cube-operand-m-axis-boxing)
自动对齐；仍需用户自行保证的是 `K` 与 `N`。

## 完整示例

### 输入: PyPTO 程序

```python
import pypto.language as pl

@pl.program
class MulKernel:
    @pl.function
    def mul_kernel_2d(self,
                     a: pl.Tensor[[32, 32], pl.FP32],
                     b: pl.Tensor[[32, 32], pl.FP32],
                     c: pl.Tensor[[32, 32], pl.FP32]):
        # Load tiles
        tile_a = pl.load(a, [0, 0], [32, 32])
        tile_b = pl.load(b, [0, 0], [32, 32])

        # Multiply
        tile_c = pl.mul(tile_a, tile_b)

        # Store result
        pl.store(tile_c, [0, 0], c)
```

### 输出: PTO-ISA MLIR

```mlir
module {
  func.func @mul_kernel_2d(%arg0: !pto.ptr<f32>,
                          %arg1: !pto.ptr<f32>,
                          %arg2: !pto.ptr<f32>) {
    // Constants
    %c32_index = arith.constant 32 : index
    %c1_index = arith.constant 1 : index
    %c0_index = arith.constant 0 : index

    // Tensor views
    %3 = pto.make_tensor_view %arg0, shape = [%c32_index, %c32_index]
         strides = [%c32_index, %c1_index] : !pto.tensor_view<?x?xf32>
    %4 = pto.make_tensor_view %arg1, shape = [%c32_index, %c32_index]
         strides = [%c32_index, %c1_index] : !pto.tensor_view<?x?xf32>
    %5 = pto.make_tensor_view %arg2, shape = [%c32_index, %c32_index]
         strides = [%c32_index, %c1_index] : !pto.tensor_view<?x?xf32>

    // Allocations
    %0 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, ...>
    %1 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, ...>
    %2 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, ...>

    // Load tile_a
    %6 = pto.partition_view %3, offsets = [%c0_index, %c0_index], sizes = [%c32_index, %c32_index]
         : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>
    pto.tload ins(%6 : !pto.partition_tensor_view<32x32xf32>)
              outs(%0 : !pto.tile_buf<...>)

    // Load tile_b
    %7 = pto.partition_view %4, offsets = [%c0_index, %c0_index], sizes = [%c32_index, %c32_index]
         : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>
    pto.tload ins(%7 : !pto.partition_tensor_view<32x32xf32>)
              outs(%1 : !pto.tile_buf<...>)

    // Multiply
    pto.tmul ins(%0 : !pto.tile_buf<...>, %1 : !pto.tile_buf<...>)
             outs(%2 : !pto.tile_buf<...>)

    // Store tile_c
    %8 = pto.partition_view %5, offsets = [%c0_index, %c0_index], sizes = [%c32_index, %c32_index]
         : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>
    pto.tstore ins(%2 : !pto.tile_buf<...>)
               outs(%8 : !pto.partition_tensor_view<32x32xf32>)

    return
  }
}
```

## 变量映射

### 内部跟踪

代码生成器维护多个映射来跟踪 MLIR 变量名:

| 映射 | 用途 | 示例 |
| ---- | ---- | ---- |
| `var_to_mlir_` | IR 变量到 MLIR 静态单赋值 (SSA) 名 | `"tile_a"` -> `"%0"` |
| `tensor_to_view_` | 参数到 tensor_view | `"a"` -> `"%3"` |
| `memref_to_mlir_` | MemRef 指针到 tile_buf | `memref.get()` -> `"%0"` |
| `memref_to_tile_type_` | MemRef 指针到 TileType | 用于推导 tile_buf 类型 |

**SSA 值命名**:

- 参数: `%arg0`, `%arg1`, `%arg2`, ...
- 常量: `%c0_index`, `%c1_index`, `%c32_index`, `%c0_i64`, `%cst`, ...
- 结果: `%0`, `%1`, `%2`, ...

### 基于 MemRef 的解析

对于 `tile.mul` 等操作:

```python
tile_c = pl.mul(tile_a, tile_b)
```

代码生成器:

1. 通过 `var_to_mlir_` 解析 `tile_a` -> `%0`
2. 通过 `var_to_mlir_` 解析 `tile_b` -> `%1`
3. 从 TileType 获取 `tile_c` 的 MemRef
4. 通过 `memref_to_mlir_` 映射 MemRef -> `%2`
5. 从 `memref_to_tile_type_` 获取 tile_buf 类型
6. 生成: `pto.tmul ins(%0 : !pto.tile_buf<...>, %1 : !pto.tile_buf<...>) outs(%2 : !pto.tile_buf<...>)`

## 类型转换

### 数据类型映射

| PyPTO 数据类型 | MLIR 类型 |
| -------------- | --------- |
| `DataType::FP32` | `f32` |
| `DataType::FP16` | `f16` |
| `DataType::BF16` | `bf16` |
| `DataType::INT32` | `i32` |
| `DataType::INT64` | `i64` |
| `DataType::INT8` | `i8` |
| `DataType::UINT8` | `ui8` |

### 标量布尔转换

`pl.cast(flag, dtype)` 将标量 `False` 转为数值零，将 `True` 转为数值一，
字面量与运行时值遵循相同语义。BOOL 使用 MLIR `i1`，但转换时不能将其解释为
有符号的一位整数。

| 目标类型 | 降低方式 |
| -------- | -------- |
| 有符号整数 | 用 `arith.extui` 将 `i1` 扩展到目标位宽 |
| 无符号整数 | 用 `arith.extui` 扩展到无符号性（signless）的 `iN`，再桥接到 `uiN` |
| INDEX | `arith.index_castui` |
| 浮点 | 零扩展到 `i32`，再使用 `arith.sitofp` |
| BOOL | 恒等转换 |

浮点路径使用整数中间值，因为 A3 设备编译器拒绝 PTOAS 生成的直接 `(float)bool`
转换。普通有符号整数扩宽仍使用符号扩展（sign extension）。上述规则针对标量转换；
Tensor 和 Tile 转换算子有各自的降低规则。

### 内存空间映射

| PyPTO 内存空间 | PTO 地址空间 |
| -------------- | ------------ |
| `MemorySpace::DDR` | `gm` (全局内存) |
| `MemorySpace::Vec` | `vec` (向量缓冲区) |
| `MemorySpace::Mat` | `mat` (矩阵缓冲区) |
| `MemorySpace::Left` | `left` |
| `MemorySpace::Right` | `right` |
| `MemorySpace::Acc` | `acc` (累加器) |
| `MemorySpace::Bias` | `bias` (偏置缓冲区) |

### Tile 缓冲区属性

生成的 `alloc_tile` 操作从 TileType 元数据推导数据类型和维度，并从关联的 TileView
推导布局/分形/填充/紧凑模式（如有）：

```mlir
!pto.tile_buf<
  loc=vec,             // PTO address space (from MemorySpace)
  dtype=f32,           // Element data type (from TileType)
  rows=32,             // Tile height (from TileType shape)
  cols=32,             // Tile width (from TileType shape)
  v_row=32,            // Virtual row size (= rows)
  v_col=32,            // Virtual column size (= cols)
  blayout=row_major,   // Block layout (from TileView, default: row_major)
  slayout=none_box,    // Scatter layout (from TileView, default: none_box)
  fractal=512,         // Fractal size in bytes, not elements (from TileView, default: 512)
  pad=0,               // Pad mode as int (from TileView, default: 0/null)
  compact=1            // Optional compact mode (normal=1; null=0 时省略)
>
```

**TileView 推导的属性**:

| 属性 | 来源 | 枚举值 | 默认值 |
| ---- | ---- | ------ | ------ |
| `blayout` | `TileView::blayout` | `none_box`, `row_major`, `col_major` | `row_major` |
| `slayout` | `TileView::slayout` | `none_box`, `row_major`, `col_major` | `none_box` |
| `fractal` | `TileView::fractal` | uint64 | `512` |
| `pad` | `TileView::pad` | `null(0)`, `zero(1)`, `max(2)`, `min(3)` | `null(0)` |
| `compact` | `TileView::compact` | `null(0)`, `normal(1)` | `null(0)` |

当 MemRef 没有关联 TileView 时，代码生成器使用上表中的默认值。默认的 null
`compact` 属性不会输出。有两条路径会自动设置 `normal(1)`：

- 进入 L0A/L0B 的部分 `tile.extract`，使 TEXTRACT 仅传输逻辑 `valid_shape`，
  而不会把 box 对齐填充当作数据。
- **有效行数无法证明等于物理行数的 Acc (L0C) tile**，由 `tile.matmul`、
  `tile.matmul_bias`、`tile.matmul_mx` 的类型推导产生。`mad` 始终以
  `ceil(validRow/16)*16` 的 N-fractal stride 写出乘积，其中 `validRow` 取自 **lhs**
  的有效行数；而所有 Acc 读取方在 tile 非 compact 时都按编译期物理 `Rows` 推导
  stride。缺少该标记时，运行期窄化的累加器读回时使用的 pitch 与写入时不同。只有
  **行** 维度决定这一点 —— ISA 推导的每个 Acc stride 都只是 `validRow` 的函数，
  因此仅窄化列维度时保持非 compact 形式。

  compact 只在**确立累加器布局的那一处**盖章，绝不在它的别名上重新推导。
  `tile.matmul_acc`（以及 `matmul_mx_acc`）**继承**累加器操作数的模式，因为该 op
  原地复用那块 buffer，而 codegen 只有在两者完整 tile 配置一致时才会做别名。
  `tile.set_validshape` 同样是继承：它只改元数据，且可能在 buffer **写入之后**才
  执行，因此读取方必须使用的仍是 `mad` 当初写入时的 pitch —— 若按窄化后的行数重新
  推导，就会以从未重排过的方式重新解释这些字节。

  buffer 也可以在创建时**声明**该模式：`tile.create(..., target_memory=Acc,
  compact=True)`。新建的 L0C buffer 没有既有字节可供重新解释，因此这不是别名上的重新
  推导——`AutoTileMatmulL0` 在切分 K 时正是这样声明它合成的累加器种子：
  `tile.matmul_acc` 会继承该种子的模式，种子若非 compact，就会把整条累加链以及循环之后
  的读取方一起拖回物理 pitch。声明也是唯一能存活的形式：pass 盖在 call 上的类型，会在
  任何后续 pass 重新推导时被丢弃（`InferTileMemorySpace` 就会），而 kwarg 每次都会被
  重新读取。

  `AccCompactValid`（见 [Verifier](../passes/99-verifier.md)）会校验该契约的两半：当 `mad`
  的 pitch 与累加器物理行数不同时，每个 `tile.matmul_acc` 都必须累加进 compact 的 buffer；
  且 Left/Right/Acc 之外的任何 tile 都不得携带 compact 模式。

注意：在 a2a3 与 a5 上，PTO-ISA 的 Acc → L1 读取方（`TExtractAccToMat`、
`TMovCcToCb`）都没有 `CompactMode` 分支，因此运行期窄化的累加器若经
`tile.extract` / `tile.move` 进入 L1，仍会按物理 `Rows` pitch 读取。该缺口需要
PTO-ISA 侧配套修改。

## 内核包装器生成 (PTO 后端)

PTO 后端为每个 ptoas 生成的 InCore 函数添加包装器，以供运行时派发。输出布局、运行时配置、参数解包和 SPMD 身份参数详见 [PTO 内核包装器生成](02-kernel_wrapper.md)。

## 另请参阅

- [Pass 管理器](../passes/00-pass_manager.md): 了解 Pass 流水线
- [IR 构建器 (Builder)](../ir/06-builder.md): 以编程方式构造 IR
- [操作符组织](../ir/05-operators.md): Tile 操作详情
- [PTOAS Op 状态矩阵](../ptoas-op-status.md): 每个 PTOAS op 的前端 / ST 覆盖状态
