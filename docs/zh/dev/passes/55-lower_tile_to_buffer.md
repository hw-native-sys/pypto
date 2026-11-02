# LowerTileToBuffer Pass

在最终设备表示边界，将完成存储规划的 Tile SSA 转换成显式 Buffer 操作。
PTO codegen 直接接收分配句柄和目标操作数。

## 默认整函数选择

`enable_buffer_ir` 保持 bool 接口，默认 `True`。正常流水线在 `InitMemRef` 之后、
`MaterializeSemanticAliases` 之前执行 `SelectBufferIRFunctions`，将选择记录为函数
属性 `buffer_ir_selected`。存储改写后不重新选择。仅选中的函数执行 Buffer 特有存储
合法化、存储验证和最终转换；其他函数保留 Tile。代码生成按最终函数 stage 分发，
同一 orchestration 可调用两种表示，Submit 返回值、task ID 和依赖保持原有语义。
最终结构、SSA、类型和调用验证仍覆盖整个程序。选中后出现编译、验证或执行错误
直接报告，不捕获错误后重新编译 Tile。

默认支持 a2/a3 上 PYPTO、DSA_RP、PTOAS 规划器的静态物理形状二维稠密 Vec 和普通 cube kernel。
物理行须按 32 字节对齐；valid 可以是物理 shape 内的静态部分矩形或已支持的运行时矩形。
逐元素运算的 Tile 输入与目标必须具有一致的物理 shape 和 valid 范围；仅 valid 相同不足以入选。

| 算子 | 元素类型 |
| ---- | -------- |
| 普通紧密 ND GM load/store | FP16、BF16、FP32、INT32 |
| add、mul | FP16、FP32、INT32 |
| sub、div、maximum、minimum、abs、exp、sqrt、neg、relu、log、recip | FP32 |
| adds、subs、muls、divs、maximums、minimums、lrelu、full | FP32 |

标量参数遵循已有 scalar ABI，元素标量在 lowering 中显式转换。
`full` 后接 store 的无 load kernel 也可入选。BF16 传输不包含 BF16 算术或填充。
`recip` 的独立目的地由存储合法化和验证保证；精度属性保留原生既有策略。
GM 返回别名必须解析到参数，其类型（包括 TensorView 元数据）须与声明的返回类型一致；
返回视图不匹配的 kernel 保持 Tile。

结构化 If/For/While 可携带 Tile、标量和同一 GM 参数的别名。
分支两臂的 GM 结果必须解析到同一参数；循环回边必须保持初始 GM 别名。
For 要求 INDEX 归纳变量和可证明为正的常量步长。动态/负步长、变化的 GM 别名、
区域内提前 return、无符号标量算术和 device helper 仍不在默认范围内。
Tile carry 的复制在 lowering 前规划；标量 yield 同时赋值，Torch 调试也保留交换语义。
block/subblock 查询使用已有 Buffer 配方。

具有静态或运行时偏移的 Vec slice（包括嵌套 slice）保持父视图的行距。reshape 和 reinterpret 保持物理字节数不变，
要求源行距稠密，单行源也须满足此要求。Vec assemble 要求目标不是窗口，且源与目标行距相同。发生源/目标重叠时，
在规划前显式复制完整的带类型父视图，再重建源窗口；精确原位写入或已证明不相交的静态字节范围不需要快照。
不相交证明使用继承的行距，并包含行间空隙，与验证器的保守存储范围一致。
它通过共享分配的起点和字节偏移识别连续 assemble 的结果，保留重叠源快照所需的逻辑父视图。
算术目标不能复用正在读取的窗口所属存储组。符号累计偏移在各 slice 定义点绑定一次，
重建快照时复用；静态子窗口不能让动态父窗口的存储范围重新变得精确。
动态边界仍是调用者前置条件。向量基址须按 32 字节对齐，例如 FP32 的列偏移
`block * 8` 可以保持对齐。选择器根据标量 SSA 算术与继承的行距证明每个 slice 和
assemble 目标的对齐，包括带显式 valid shape 的 slice。无法证明对齐时保留 Tile IR；
更早的 slice 规范化 pass 可以在选择前物化普通 slice。

静态稠密 Vec reshape 支持同一已规划分配的多个视图。
静态非负 MemRef 偏移必须按 32 字节对齐。lowering 验证分配容量及窗口边界；
选中的非法窗口直接报错，不回退到 Tile IR。

普通 cube 路径为 GM -> Mat -> Left/Right -> Acc -> GM，支持 `move`、`extract`、
`transpose_view`、`matmul` 和 `matmul_acc`。FP16/BF16/FP32 输入以 FP32 累加，
INT8 输入以 INT32 累加。常量或运行时 `init_cond` 选择初始化或累加写入。
较宽的 Acc valid 区域通过同一存储上的乘积视图写入。矩阵区域 carry 必须保持
同一存储别名；任意矩阵 carry 复制、MX、FIFO、槽位、缩放/量化写回和 workspace
操作仍不在默认范围。Torch 调试模拟分形布局及共享的已寻址矩阵存储，保留
转置重标记和乘积视图；部分有效乘积之外仅完整未触及的 Acc box 保证保留。

A5 在默认流水线保持 Tile，待取得运行时验收后再开放。
reduction、运行时 offset 的 relabel、窗口区域 carry、非连续 relabel、行距不兼容或目标为窗口的
assemble、槽位、helper ABI 和跨核资源在正常流水线保持 Tile。同一符号分配需要多个不同形状或
类型的窗口父视图时也保持 Tile；对已 relabel 的窗口再 slice 同样尚未开放。已有生产配方可通过显式底层 pass 流水线开发和测试，
配方存在本身不代表默认支持。

```text
默认 / True: load -> add -> mul -> store  => Buffer
默认 / True: load -> reduction -> store   => Tile（整个 kernel）
False:       两种 kernel                 => Tile
```

用 `PassContext([], enable_buffer_ir=False)` 选择旧路径。JIT 缓存包含 bool 与 Buffer
语义版本；嵌套 dump/profiling 上下文、预编译 worker 和调试重编译保持有效开关值。

## 运行时 valid 的默认选择

默认支持直线区域中的 Vec load/store、结果被赋值的 allocation-backed `set_validshape`
和上表中形状相同的逐元素配方。独立 setter 语句继续 Tile，因为元数据版本跟踪需要显式
SSA 结果。算术输入与结果的 valid 表达式须结构相等；两个 descriptor
维度都是 `-1` 不代表实际运行时值相等。前向 SSA 分析拒绝 setter 之后再次使用旧
元数据版本，包括普通别名及从该版本切出的窗口；当前版本可以有多个消费者。
带动态 valid 状态的用户控制流、窗口 setter、动态 valid 的 relabel/assemble、嵌套
动态 valid 窗口继续 Tile。已有静态控制流支持不变。

可变 allocation 使用双轴动态 descriptor，窗口保留逐轴静态/动态 descriptor。
PYPTO 仅允许动态 allocation 与兼容的、覆盖完整容量的 typed root 合并。
独立 allocation 的生命周期结束后，仍可复用其物理地址。

有效长度允许零：allocation metadata 仍更新，但 load/store/算术仅在两轴 valid
均为正时执行。动态 valid 的 subview 在受 guard 保护的消费点物化，因此空路径
不会构造空 native subview。常量空操作在创建 Buffer 调用前消除。view 不逃逸出
分支，也不产生 Buffer-valued phi。扩大 valid 不初始化数据；非零动态长度仍须
满足物理窗口边界，读取区域须已初始化。

`tests/st/runtime/ops/test_buffer_views.py::test_default_dynamic_buffer_windows`
通过公开默认 JIT 入口覆盖三种 planner，在同一编译产物内改变偏移与空/部分/完整
有效区，检查未写区 sentinel 及实际执行的 Buffer/native 产物。

## 位置与 API

迁移期间，在同一个上下文中创建并运行默认流水线：

```python
from pypto import passes
from pypto.ir.pass_manager import OptimizationStrategy, PassManager

with passes.PassContext([], enable_buffer_ir=True):
    manager = PassManager(OptimizationStrategy.Default)
    lowered = manager.run_passes(program)
```

`LowerTileToBuffer` 位于 `MaterializeValidShapeSymbols` 之后。
存储修复和 `VerifyTileStorage` 位于地址分配之前。PYPTO 与 DSA_RP 提供最终有效字节地址，
PTOAS 提供不带地址的分配身份。即使关闭自动验证，本 pass 也会重新检查存储闭合性。
地址规划器还会检查有效地址区间的重叠。

自定义流水线满足相同的存储、SSA、返回值规范化及设备/编排分离约束后，
可调用 `passes.lower_tile_to_buffer()`。还必须满足 `NoNestedCalls`：在此边界之前执行
`FlattenCallExpr`。该 pass 产生 `BufferIR` 属性，
并使描述 Tile 存储的属性失效。

## 表示示例

以下记法简写了描述符和标量元组：

```text
# Planned Tile input: lhs, rhs and total carry MemRef storage windows.
lhs = tile.load(A, (0, 0), (16, 32))
rhs = tile.load(B, (0, 0), (16, 32))
total = tile.add(lhs, rhs)
result = tile.store(total, (0, 0), Out)
return result

# Buffer output: allocs occur at the original allocation definitions.
# Each descriptor is Buffer[[16, 32], FP32, Vec].
a_buf = buffer.alloc((), address_a)
b_buf = buffer.alloc((), address_b)
r_buf = buffer.alloc((), address_r)
buffer.load(A, (0, 0), (16, 32), a_buf)
buffer.load(B, (0, 0), (16, 32), b_buf)
buffer.add(a_buf, b_buf, r_buf)
buffer.store(r_buf, (0, 0), (16, 32), Out)
return Out
```

规划器可能已经复用某个分配；示例为便于理解展示三个独立窗口。
PTOAS 省略 `buffer.alloc` 的第二个操作数。地址规划器只传入一次最终有效地址。
写入目标的 Buffer 操作返回 `VoidType`；分配操作返回句柄。

## 转换契约

索引遍历从 Tile 变量收集已规划的 MemRef。生产者调用仍保留逻辑推导类型，
不会建立额外的分配身份。第二次遍历将 Tile 使用替换为 Buffer 句柄，并改写支持的调用。
映射仅存在于 pass 内部，输出 IR 不携带转换旁表。

已有存储定义决定分配位置。对于所有引用均位于同一区域的无地址 PTOAS 累加器，
转换会将描述符声明放在首次引用处，保留生产者的局部生命周期，而不是跨循环回边延长它。
跨区域共享或由循环携带的存储仍保留在原始支配位置。
转换不新增临时空间、不分配地址、不插入隐式传输。
完全相同的自复制会被移除。Tensor 返回别名规范化为已有 GM 参数，
同时保留参数方向和编排 ABI。

转换后的 `InCore`、`AIC`、`AIV` 函数标记为 `FunctionIRStage.Buffer`，编排函数保持原样。
pass 验证输出并保持幂等；转换失败不会修改输入程序。此边界之后不应运行功能式 Tile pass。

合成分配的原始位置未知时，继承已索引 Tile 句柄的源码位置，使原生分配诊断能够定位用户源码。

## 已规划的存储视图

`tile.alloc` 的字节容量始终是分配大小的依据。当同一个根服务多个静态描述符或较小窗口时，
转换声明一个有效范围完整的 `UINT8[capacity / 32, 32]` Buffer，再创建显式
`buffer.subview` 和 `buffer.reshape` 别名。相同的描述符与窗口组合共享一个 SSA 句柄，
即使输入是不同的 Tile 变量。唯一描述符覆盖整个分配时，仍直接使用带类型的分配形式。

```text
# Planned allocation: capacity 4096, effective base address 8192.
# A dense FP32[16,32] member starts at effective address 8256.
root = buffer.alloc((), 8192) : Buffer[[128,32], UINT8, Vec]
window = buffer.subview(root, (2,0)) : Buffer[[64,32], UINT8, Vec]
value = buffer.reshape(window) : Buffer[[16,32], FP32, Vec]
```

每个别名定义放在原始分配旁边，只读取描述符元数据，不读取或复制数据。
发射器按这些已确定的原生形式逐条输出，不恢复分配大小、不选择视图指令，也不增加存储。
逻辑 `tile.reshape` 在检查类型和存储窗口保持不变后消失，其结果由索引中的带类型别名提供。

对于分配地址的规划器，必须有一个 MemRef 大小等于分配容量的成员确定最终基地址。
只有内部窗口时拒绝转换，因为它们的最小地址不能证明分配从哪里开始；
后续显式分配事实表示可解除这个限制。PTOAS 使用符号原点零。
所有视图必须位于原始容量内，字节偏移、字节大小和物理行均须静态且按 32 字节对齐。
分配处仍不支持分形布局的视图和可变视图元数据；带步长的窗口在使用处定义（见下文）。

覆盖完整容量且需要窗口父视图的成员使用该父视图的带类型分配，其他解释通过 reshape 获得。
静态 valid 矩形使用独立的零偏移子视图，因此不带地址的 PTOAS 存储也能保留部分有效的 assemble
目标，无须第二次分配。

## 窗口：slice 与 assemble {#windows-slice-and-assemble}

`tile.slice` 是源在（可能为运行时的）偏移处的窗口，保持源的行距。它的偏移只在 slice 所在位置
可用，因此窗口在那里定义，而不是在分配处；对窗口的重新标注是在同一位置对它的 `buffer.reshape`。
`tile.assemble` 原地写入：把源拷贝进结果存储上的一个窗口；如果规划器给目标和结果分配了不同存储，
先把目标拷贝到结果。

```text
# window = tile.slice(whole, [16, 32], [row, 32]); placed = tile.assemble(canvas, doubled, [16, 0])
window = buffer.subview(whole_buffer, (row, 32), (16, 32)) : Buffer[[16,32], FP32, Vec]
buffer.add(window, window, doubled_buffer)
placed_window = buffer.subview(canvas_buffer, (16, 0), (16, 32)) : Buffer[[16,32], FP32, Vec]
buffer.copy(doubled_buffer, placed_window)
```

每个 lowering 生成的窗口都带 valid 元组，PTOAS 据此确定原生每一维的类型。
常量 valid extent 必须为正且位于窗口范围内；valid extent 为零的 slice 和 assemble 窗口会被拒绝。
PTOAS 无法合法化对字节根重新标注结果的 subview。因此覆盖完整容量的窗口父视图保留带类型的根；
其他共享已寻址存储中的窗口父视图在其规划地址上使用别名分配。
源本身就是 assemble 目标窗口时，数据已原地写好，不需要拷贝。跨内存空间的 assemble（fix-pipe 的
Acc 到 Mat 形式）和降秩 slice 在对应配方完成前会被拒绝。

`BufferIR` 证明每个窗口在被读取期间与写入的目标不重叠。运行时偏移的窗口只能按整个源窗口定位；
带地址的规划器把这类目标放在该范围的一部分上时，lowering 会失败，而不是假定运行时偏移能避开它。
共享分配约束会在 PYPTO 和 DSA_RP 中阻止窗口算术发生这种复用，也覆盖显式底层流水线中的运行时偏移。

## 运行时 valid extent {#runtime-valid-extents}

Assemble 扩大有效区域时，保留原目标数据的复制会通过与原有效矩形匹配的视图写入。
子视图的字面量有效范围必须与结果的对应静态维度一致；动态结果维度必须使用非常量范围。

非常量的 valid extent（或 lane-1 的 `[0, 0]` 哨兵）会让描述符的两个维度都变为动态，
因为原生元数据更新总是同时写两维。分配以物理大小作为初始值，定义此类 tile 的每个操作先用
`buffer.set_validshape` 声明自己产生的 extent；`tile.set_validshape` 直接更新共享的句柄。
指向同一存储窗口、仅 valid extent 不同的 tile 共用一个动态句柄。窗口则在其 subview 中逐维保持
静态或动态，因为视图从不接受元数据更新。动态 tile 必须拥有自己的分配，不能共享字节视图根。

当算术操作组合具有不同有效范围描述符的窗口句柄和分配句柄时，零偏移子视图会为输入和目标
提供一致的逐维描述符。Assemble 的复制源也使用同样的转换。这些视图保留原有存储并使用当前
显式有效范围，不增加数据复制，也不更新窗口的元数据。

```text
# loaded = tile.load(x, [0, 0], [16, 64], [16, cols]); doubled = tile.add(loaded, loaded)
buffer.set_validshape(loaded_buffer, (16, cols))
buffer.load(x, (0, 0), (16, cols), loaded_buffer)
buffer.set_validshape(doubled_buffer, (16, cols))
buffer.add(loaded_buffer, loaded_buffer, doubled_buffer)
```

## 矩阵存储与 cube 转换 {#matrix-storage-and-cube-recipes}

Mat、Left、Right 和 Acc Tile 在 `BufferType` 中保留已确定的分形布局（`blayout`、
`slayout`、`fractal`、`compact`）。它们的物理范围已经是完整的分形块，而分形窗口没有
行主序的字节视图，因此这些空间不使用 `UINT8[N,32]` 根。带地址的规划器已经确定了每个
窗口，所以每个不同窗口都成为位于其最终有效地址的独立 `buffer.alloc`；同一地址上的两个
描述符正是按规划器的放置互为别名。PTOAS 分配没有地址，因此每个分配只承载一个窗口；
对该窗口的等大小重新标注成为 `buffer.reshape` 别名。

```text
# x = a @ b; y = x + a @ b; store y  (FP16 操作数，FP32 累加器)
a_mat = buffer.alloc((), 0)      : Buffer[[64,64], FP16, Mat, NZ]
b_mat = buffer.alloc((), 8192)   : Buffer[[64,64], FP16, Mat, NZ]
a_l0  = buffer.alloc((), 0)      : Buffer[[64,64], FP16, Left]
b_l0  = buffer.alloc((), 0)      : Buffer[[64,64], FP16, Right, ZN]
acc   = buffer.alloc((), 0)      : Buffer[[64,64], FP32, Acc, fractal=1024]
buffer.load(A, (0,0), (64,64), a_mat)
buffer.load(B, (0,0), (64,64), b_mat)
buffer.copy(a_mat, a_l0)
buffer.copy(b_mat, b_l0)
buffer.matmul(a_l0, b_l0, acc)
buffer.matmul_acc(a_l0, b_l0, acc)
buffer.store(acc, (0,0), (64,64), Out)
```

从 Mat 出发的 `tile.move` 转为 `buffer.copy`，`tile.extract` 转为 `buffer.extract`，
`tile.matmul` 转为 `buffer.matmul`。`tile.matmul_acc` 必须已经是原地累加（累加器与结果
共享存储），转为 `buffer.matmul_acc`。其可选的 `init_cond` 决定写入形式：字面量直接选择
一个调用，运行时条件则变成显式 `if`，两个分支分别是 `buffer.matmul` 和
`buffer.matmul_acc`。累加器的 valid 矩形比乘积更宽时，通过同一存储上与乘积同形的视图写入
（带地址时是另一个 alloc，PTOAS 下是 `buffer.reshape`），因为 PTOAS 要求原生 matmul 目标与乘积一致。
这与旧发射器的原生写入相同：乘积之外的完整分形块保持原数据，被部分覆盖的块其余部分不保留。
`tile.transpose_view` 不需要调用：存储索引阶段已经声明了重新标注的
窗口。Acc store 只使用 fix-pipe 不带 scale 的转换；`pre_quant`/`pre_relu`、原子与分阶段
store 以及 cache 策略 load 在对应传输配方完成前仍显式报错。

流水线阶段归属属性只约束存储规划，而规划结果已经记录在 MemRef 中。本 pass 会丢弃该属性
（地址规划更早地去掉它；PTOAS 下它会保留到这里）。其他调用属性仍需显式的转换契约。

## 分支

存储合法化已经为每个 Tile 分支结果选择规范目标窗口，并在各分支体内放置必要的传输。
最终转换移除这些 Tile 结果和 yield 操作数，保留标量结果的相对顺序，
因此原生 `scf.if` 只携带真实的标量 SSA 值。

```text
# Input: (chosen_tile, selected_offset) = if flag:
#          then yield (product, 16); else yield (input_tile, 0)
# Storage legalization gives chosen_tile a canonical destination.
selected_offset = if flag:
    buffer.mul(a_buf, b_buf, destination)
    yield 16
else:
    buffer.copy(a_buf, destination)
    yield 0
buffer.store(destination, (selected_offset, 0), (16, 32), Out)
```

若两个分支体的 GM 结果都解析到同一个已有参数，该结果会被移除，后续使用直接引用该参数。
不同 GM 别名需要单独的动态 GM 转换支持，当前会显式报错。
分布式 Tensor 分支结果也在此边界显式拒绝，即使两个分支体别名指向同一个参数；
它们需要单独的转换支持。
嵌套分支使用具有作用域的 yield 上下文；转换不会为修复区域而新增分配或复制。
分支与 yield 的源码注释会保留。

## 循环

For 和 While 转换在确认 Tile 初始值、iter_args、结果和回边 yield 引用同一个合法化存储后，
移除这些 Tile 循环状态。入口复制已经放在循环前，因此零次迭代仍保留初始值。
交换与扇出所需的快照是循环体内普通的 Buffer 写入；最终转换不会新增临时存储或复制。

原生控制流仅保留标量 iter_args，并保持它们原有的相对顺序。
While 条件引用重写后的标量绑定。若 GM 初始值和回边都解析到同一个参数，GM 循环状态会被移除；
变化的 GM 选择需要单独的转换支持，当前会显式报错。
嵌套循环和分支分别使用独立的 yield 上下文；每个初始值仅在绑定处遍历一次，
避免沿外层循环状态链重复遍历。
与分支结果一样，分布式 Tensor 循环状态在此边界显式拒绝，
直到其区域结果和设备 ABI 转换得到支持。
二进制往返在解码 While 条件前恢复循环状态的定义，
使条件和循环体中的共享引用都指向相同的循环状态。

```text
# Tile carries (left, row, right, column) become two scalar carries.
(row_result, column_result) = for i in range(count), (row=0, column=0):
    buffer.copy(right_buf, scratch_right)
    buffer.copy(left_buf, scratch_left)
    buffer.copy(scratch_right, left_buf)
    buffer.copy(scratch_left, right_buf)
    yield (row + 1, column + 2)
buffer.store(left_buf, (row_result, column_result), (16, 32), Out)
```

标量 SPMD 查询 `tile.get_block_idx`、`tile.get_block_num` 和
`tile.get_subblock_idx` 转换为对应的内部 `buffer.*` 查询，返回 INDEX 值且没有内存效果。
原生发射读取既有运行时传入的 kernel ABI 参数。查询保持为直接 SSA 赋值；
源程序的嵌套调用由此前的 `FlattenCallExpr` 展开。

## 首批支持的转换

当前转换支持直线程序、分支和循环、二维稠密 Vec FP16/BF16/FP32/INT32 Tile、显式存储视图、使用处窗口、静态或运行时有效范围、
普通紧密排列的 ND GM Tensor 以及默认加载/存储策略，另外还有上文的 cube 路径：Mat load、
Mat 到 Left/Right 的拷贝与 extract、matmul/matmul_acc 以及 Acc store。
它转换分配、create、load、store、move、已经合法化的别名及[带类型的逐元素配方](../ir/05-operators.md#typed-buffer-elementwise-recipes)。
标量输入在 lowering 中显式转换为目标 dtype；发射器直接消费这些类型。
`tile.full` 的形状和 dtype 由目标描述符表示，不会重复作为指令属性发射。
GM load/store 保留匹配的元素类型，不插入转换。`add`/`mul` 支持 FP16/FP32/INT32；
BF16 传输支持不代表算术支持。

辅助函数调用、其他布局、多槽位和其他操作转换由后续迁移切片补齐。
显式底层转换对未支持形式报错；正常流水线将未支持的完整 kernel 保留为 Tile。

二进制序列化保留显式表示和函数阶段。当前 Python 诊断打印器不支持 Buffer DSL 解析往返。

## 测试

`tests/ut/ir/transforms/test_lower_tile_to_buffer.py` 通过公开前端运行三种规划器的完整流水线，
检查显式分配和目标写入、转换的不可变性与幂等性、二进制持久化，并使用原生 PTOAS 编译输出。

`tests/ut/ir/transforms/test_lower_buffer_views.py` 检查分配容量、非零窗口偏移、
重复视图身份、无法证明地址时的诊断、二进制持久化，以及两个目标和三个规划器的原生编译。
它还转换静态和运行时 slice、assemble，以及来自 `tile.load` 与 `tile.set_validshape` 的运行时
valid extent，包括带地址规划器下运行时窗口部分重叠时的失败关闭。

`tests/ut/ir/transforms/test_lower_buffer_matrix.py` 针对 FP16、BF16、FP32 和 INT8 操作数
转换显式及自动分块的 cube 内核，覆盖三种 `init_cond` 形式和转置操作数，并在三种规划器、
两个目标上原生编译生成的 PTO。

数值系统测试应在公开 `@pl.jit` 入口对应的 case 上声明
`st.case(..., enable_buffer_ir=True, memory_planner=...)`。
测试框架 (Harness) 会在内联及预编译工作线程内部应用该选项；仅在测试线程外层设置
`PassContext` 不会配置工作线程。启用的 case 使用独立缓存键。框架检查编译产生的最终
设备函数阶段，并将转换后程序保存为原生构件旁的 `buffer_ir.msgpack`。

`tests/st/runtime/ops/test_buffer_ir.py` 为三种规划器提供带编排的
load/add/mul/store 数值测试，以及精确比较的 cube 用例：FP16/BF16/INT8 的 matmul 加
matmul_acc、带运行时 `init_cond` 的 K 循环、自动分块的 BF16 matmul 以及 `b_trans` matmul。
窗口和 valid extent 用例覆盖切列后 assemble 到新 tile、运行时行 slice，以及来自 `tile.load` 和
`tile.set_validshape` 的运行时 valid 列，每个构件运行四组运行时值。只运行这个目标文件：

```bash
source .claude/skills/testing/load-env.sh
python -m pytest tests/st/runtime/ops/test_buffer_ir.py --platform=a2a3 --device=0 \
    --precompile-workers "$PYPTO_TEST_JOBS" --save-kernels -v
```

预编译模式还会检查实际执行构件所保存的 Buffer 程序和 PTO 源码。
`--codegen-only` 可用于编译检查，但不构成数值验证证据。
框架测试无需设备即可覆盖内联及工作线程中的选项传递。

`tests/st/runtime/control_flow/test_buffer_ir.py` 为每种规划器增加分支、For、While、
嵌套循环和扇出数值测试。编排层从配置 Tensor 读取计数和条件，传给已编译的设备内核，
因此一个构件可执行多条运行时路径。For 和 While 覆盖 0、1、2、3 次迭代、两个分支方向、
奇偶次数交换、交错标量偏移以及同一 GM 的循环状态。独立输出区间保留未写入的哨兵值，
另一个输出保留循环后的原始输入。非对称最终表达式能够发现可交换求和掩盖的交换错误。
嵌套用例包含外层或内层零次迭代；扇出用例检查两个目标读取同一来源。

在提供 task-submit 设备队列的主机上，只运行该有限矩阵：

```bash
source .claude/skills/testing/load-env.sh
python -m pytest tests/st/runtime/control_flow/test_buffer_ir.py --platform=a2a3 \
    --precompile-workers "$PYPTO_TEST_JOBS" --execute-via-task-submit \
    --execute-batch-size=4 --task-max-time=120 --save-kernels -v
```

队列选择可用设备。其他主机应省略队列选项，并通过 `--device` 选择可用设备。
已保存构件的检查还要求原生循环结果仅包含标量，并包含预期的显式操作；
仅通过原生编译并不能证明数值正确。
