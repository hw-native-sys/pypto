# LowerTileToBuffer Pass

在最终设备表示边界，将完成存储规划的 Tile SSA 转换成显式 Buffer 操作。
PTO codegen 直接接收分配句柄和目标操作数。

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
可调用 `passes.lower_tile_to_buffer()`。该 pass 产生 `BufferIR` 属性，
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

已有存储定义决定分配位置。转换不新增临时空间、不分配地址、不插入隐式传输。
完全相同的自复制会被移除。Tensor 返回别名规范化为已有 GM 参数，
同时保留参数方向和编排 ABI。

转换后的 `InCore`、`AIC`、`AIV` 函数标记为 `FunctionIRStage.Buffer`，编排函数保持原样。
pass 验证输出并保持幂等；转换失败不会修改输入程序。此边界之后不应运行功能式 Tile pass。

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

## 首批支持的转换

当前转换支持直线程序、分支和循环、静态二维稠密 Vec FP32 Tile、每个分配一个描述符、静态有效范围、
普通紧密排列的 ND GM Tensor 以及默认加载/存储策略。
它转换分配、create、load、store、加法、乘法、move 及已经合法化的别名。

辅助函数调用、其他布局、动态元数据、多槽位和其他操作转换由后续迁移切片补齐。
暂不支持的形式会显式报错。在完整转换与运行时验收矩阵通过前，迁移选项默认关闭。

二进制序列化保留显式表示和函数阶段。当前 Python 诊断打印器不支持 Buffer DSL 解析往返。

## 测试

`tests/ut/ir/transforms/test_lower_tile_to_buffer.py` 通过公开前端运行三种规划器的完整流水线，
检查显式分配和目标写入、转换的不可变性与幂等性、二进制持久化，并使用原生 PTOAS 编译输出。

数值系统测试应在公开 `@pl.jit` 入口对应的 case 上声明
`st.case(..., enable_buffer_ir=True, memory_planner=...)`。
测试框架 (Harness) 会在内联及预编译工作线程内部应用该选项；仅在测试线程外层设置
`PassContext` 不会配置工作线程。启用的 case 使用独立缓存键。框架检查编译产生的最终
设备函数阶段，并将转换后程序保存为原生构件旁的 `buffer_ir.msgpack`。

`tests/st/runtime/ops/test_buffer_ir.py` 为三种规划器提供带编排的
load/add/mul/store 数值测试。只运行这个目标文件：

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
