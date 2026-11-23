# SIMT 接口覆盖

本矩阵跟踪任务 11.7，以固定的 PTOAS v0.77 为基准。
通用无符号类型与单个操作对 INT32 位载体的无符号解释不同。

## 第一批：32 位接口

下列 API 同时存在于 `pl.simt` 和 `pypto.ir.op.simt`。
它们通过已注册的 SimtOp schema 和既有通用 binding 暴露，无需新增 C++ 节点或原生绑定签名。

| PTOAS 操作 / 参数 | PyPTO API | 类型 / 存储 | 状态 |
| ----------------- | --------- | ----------- | ---- |
| `atomic_sub` | `atomic_sub` | INT32/FP32，GM/UB | 已实现；设备验收待完成 |
| `atomic_and/or/xor` | 同名接口 | INT32，GM/UB | 已实现；拒绝 FP32 |
| `atomic_min/max unsigned` | `atomic_min/max(..., unsigned=True)` | INT32 位载体，GM/UB | 已实现；拒绝 FP32 unsigned |
| `redux_mini/maxi unsigned` | `redux_min/max(..., unsigned=True)` | INT32 位载体 | 已实现；拒绝 FP32 unsigned |
| 四种 `shuffle_*` 的 `bound` | `shuffle_*(..., width=32, bound=None)` | INT32/FP32，width 16/32 | 已实现；默认窗口不变 |
| `get_lanemask_eq/le/lt/ge/gt` | `lanemask_eq/le/lt/ge/gt()` | INT32 位载体 | 已实现；表示物理 lane 关系 |
| `bitcast` | `bitcast(value, dtype)` | FP32/INT32，允许同类型 | 已实现；保留位模式 |

原子操作沿用 ReadWrite 参数副作用和 Simt 写通道，返回旧值；丢弃返回值时保留更新，
验证 cache 属性并参与 launch 的存储版本追踪。
源窗口规则和非活跃源限制见 [IR 契约](../ir/11-simt.md)。

2026-11-20 使用 PTOAS v0.77 验证。独立 worktree 的 C++ 构建通过。SIMT 算子、原子回归、
codegen、公共 JIT、作用域提取和存储准备共 **1,128 项测试通过**。
LLVM 导出覆盖第一批的两个验收入口，以及四种 shuffle 在 INT32/FP32、width 16/32
和两个窗口边界下的编译。

验证入口：

- `tests/ut/ir/operators/test_simt_ops.py`：schema、类型/属性错误、序列化、包装层一致性和原子读写副作用。
- `tests/ut/jit/test_simt_scope.py::test_first_batch_simt_interfaces_native_compile`：
  公共 JIT 完整降低和固定 PTOAS LLVM 导出，覆盖 GM/UB、无符号比较、窗口属性和旧值未使用的原子更新。
- `tests/ut/jit/test_simt_scope.py::test_public_warp_collectives_native_compile`：
  验证 LLVM 导出的 vote、shuffle 和 redux 调用仍在活跃分支中，并验证循环条件每轮重新计算 vote。
- `tests/st/runtime/vpto/test_simt_interface_gaps_onboard.py`：
  可选数值验收，覆盖无符号最高位边界、旧值、位模式和窗口；编译成功不代表设备执行成功。

当前主机的 NPU 探测返回 DCMI -8005。新增接口的设备证据仍待完成，
包括原子竞争、分歧中的物理 lane 掩码、FP32 减法特殊值及全部 shuffle 窗口组合。

## 第二批：标量类型与显式操作

标量存储 ABI 扩展为 FP16/BF16/FP32 和 8/16/32/64 位有符号、无符号整数。
这不表示每条指令都接受全部类型。无符号值在原生 builtin 边界使用保留位模式的
signless 载体，并显式选择无符号解释。

| 类别 | 支持的非 packed 类型 | 限制 |
| ---- | -------------------- | ---- |
| GM/UB load/store、标量捕获 | FP16/BF16/FP32；8/16/32/64 位有符号/无符号整数 | BOOL/INDEX 仅用于计算；不支持 FP8 存储 |
| `bitcast` | 同上 | 源与目标位宽相同 |
| `shuffle_*` | FP16/FP32；32/64 位有符号/无符号整数 | 不支持 BF16；lane 控制仍为 INT32 |
| `redux_*` | FP16/FP32；32 位有符号/无符号整数 | 不支持 BF16 和 64 位整数 |
| 原子 add/sub/min/max/exch | FP16/BF16/FP32；32/64 位有符号/无符号整数 | 64 位原子仅支持 GM |
| 原子 and/or/xor | 32/64 位有符号/无符号整数 | 64 位原子仅支持 GM |
| 原子 CAS | FP32；32/64 位有符号/无符号整数 | 不支持 FP16/BF16；64 位仅支持 GM |
| `ftof` | FP16/BF16/FP32 间不同类型对 | r/a/f/c/z；仅 FP32→FP16 支持 o；可选饱和 |
| `ftoi` | FP16/BF16/FP32→有符号/无符号 INT32；FP32→有符号/无符号 INT64 | r/a/f/c/z；强制饱和 |
| `itof` | 有符号/无符号 INT32→FP16/BF16/FP32；有符号/无符号 INT64→FP32 | r/a/f/c/z |
| 既有显式标量数学 | FP32；`prmt`/`mulhi` 使用 INT32 位载体 | 不统一放开半精度数学 |
| `mul_i32toi64` | INT32 输入、INT64 输出 | 可选择无符号解释 INT32 位载体 |
| `div`、`rem` | FP32 | 显式精度和截断余数；普通 `/`、`%` 不变 |
| `veccore_id` | INT32 输出 | 目标 vector core ID |
| `trap`、`assert_` | Void；assert 要求 BOOL | 依赖 runtime 的诊断，不作为安全访存保护 |

新增测试位于 `tests/ut/codegen/test_simt_codegen.py`，覆盖存储、原子/协作和转换对的
原生 LLVM 导出。公共 JIT 验收用例还覆盖 FP16/BF16/INT64/UINT32 ABI、无符号最高位、
显式除法/余数和拓宽乘法。硬件数值验收仍待完成，编译成功不等于数值验证通过。
独立 worktree C++ 构建及相关 SIMT 回归通过，包括公共 JIT 用例和要求 PTOAS v0.74 的
原生编译。设备探测仍返回 DCMI -8005，在沙箱外执行也相同。

FP32 `rem` 使用整数有效数归约，因为 A5 Bisheng 后端无法选择 `pto.remf`
生成的 LLVM `frem`。该实现无需舍入后的浮点商，保留次正规数和被除数符号
（包括零）。除数为零、被除数为无穷大或任一操作数为 NaN 时返回 NaN；
有限数除以无穷大时返回被除数。不保证 NaN 的载荷位。

JIT 回归 `test_simt_interfaces_device_compile` 将扩展接口用例和动态除数的
余数内核送入生产 PTOAS、Bisheng 辅助函数编译路径。通过
`PYPTO_RUN_VPTO_VF_VALIDATION=1` 启用，无需 NPU。主机 LLVM 回归将实际生成的
算术与 `fmod` 对比；上板用例按位检查不同符号和指数边界的结果。
上板余数验收使用精确的 IEEE-754 位模式，不依赖主机 CPU 的次正规数清零模式。
`ftof` 的位置参数和关键字形式的值操作数均按有类型表达式解析，支持 `math.inf`
等数值属性；目标 dtype 仍按数据类型解析。
主机执行和设备编译不能替代上板数值验收。

## 剩余范围

| 类别 | 当前覆盖 | 剩余工作 / 限制 |
| ---- | -------- | --------------- |
| 坐标、维度、lane ID | 已有 DSL/IR/原生降低 | 无需重复实现 |
| `get_clock32/64` | 未暴露 | 固定 PTOAS 将读取视为纯操作，合并不同采样，详见下文 |
| 运行时 trap/assert | 已有 DSL/IR/原生降低 | 设备终止和宿主错误传播尚未验证 |
| `keep/resume` | 已验证原生配对对象编译；DSL/IR 未暴露 | 需要配对 launch 表示、slot 验证和线程状态生命周期约束 |
| `addui_extended`、`mul_extended` | 无直接多结果 API | 现有算术/位运算和 `mulhi` 可表达部分能力，不宣称完整类型覆盖 |
| `ceildiv`、`select` | 支持的标量可由算术/控制流组合表达 | 缺少同名 API 不等于缺少计算能力 |
| packed 与短向量 | 任务 11.6 暂缓 | 不是 11.7 的前置条件 |

## 工具链边界

PTOAS v0.77 要求在指针运算和标量访存操作中显式标注元素偏移的类型。
PyPTO 为这些操作数发射 `index`。

### 时钟采样阻碍

PTOAS v0.74 将 `get_clock32/64` 声明为纯操作。原生探针读取两次 `get_clock32`、
相减并写入 GM，导出的 LLVM 只保留写入常量零，时钟读取完全消失。
PyPTO 保留调用无法修复下游的公共子表达式消除，因此在原生副作用契约和优化回归
修复前不暴露时钟 API；仅支持 INT64 存储不能解决这个问题。最小入口函数体：

```mlir
%a = pto.get_clock32 : i32
%b = pto.get_clock32 : i32
%elapsed = arith.subi %b, %a : i32
%zero = arith.constant 0 : index
pto.stg %elapsed, %dst[%zero] : !pto.ptr<i32, gm>, index, i32
```

放入 `pto.simt_entry` 函数，使用 PTOAS v0.77 和 `--pto-arch=a5
--pto-backend=vpto --emit-vpto-llvm-ir` 编译，输出的 store 写入零。

### 跨入口状态阻碍

原生 `keep/resume` 要求同一外层 launch 序列中的相邻 SIMT 入口。原生配对测试已通过
i32/i64/f16/bf16/f32 的 LLVM lowering、生产设备编译与最终链接。这些操作生成内联汇编，
运行时因此改走 PTOAS 原生设备对象输出，由 PTOAS 验证或修复 launch 长度，再通过链接器
别名恢复逻辑导出名。回归检查两次 launch 的长度是否匹配链接后的入口符号，也覆盖占用
两个 word 的 i64 状态 slot。无需 NPU 即可运行：

```bash
PYPTO_RUN_VPTO_VF_VALIDATION=1 python -m pytest \
  tests/ut/codegen/test_simt_codegen.py -k keep_resume --require-ptoas -v
```

这解决了编译前置条件，尚未实现公开的状态生命周期契约或数值验收。
当前可执行 PyPTO bundle 为每个 helper 生成独立模块和外层 wrapper，只传递内存副作用，
没有 helper 产物之间带类型的线程状态依赖。仅暴露指令无法验证 launch 配对和状态寿命。
后续实现需要把配对入口表示为同一 launch 序列，检查维度与 slot 类型一致，拒绝 slot
重叠（64 位值占用偶数起始的两个 word），并约束 resume 序言、keep 尾声的位置。
这类状态依赖不能建模为普通内存 fence。
