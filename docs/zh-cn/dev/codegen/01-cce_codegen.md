# PyPTO 代码生成 (CodeGen) 模块

## 概述

PyPTO 代码生成 (CodeGen) 模块将优化后的 PyPTO 中间表示 (IR) 转换为使用 pto-isa 指令集的可执行 C++ 代码。

**流水线:** `IR -> PassManager -> CCECodegen -> Compiler`

**核心设计原则:**

- **独立组件**: 不是 Pass。Pass 进行 IR->IR 转换, 代码生成进行 IR->String 转换
- **基于访问者模式**: 扩展 `IRVisitor` 进行 IR 树遍历
- **不可变**: 不修改输入 IR
- **模块化**: 关注点分离 (发射、映射、类型转换)

## 架构

### 组件结构

| 组件 | 用途 | 位置 |
| ---- | ---- | ---- |
| `CCECodegen` | 主协调器, 扩展 IRVisitor | [cce_codegen.h](../../../../include/pypto/codegen/cce_codegen.h) |
| `CodeEmitter` | 带缩进的结构化输出 | [code_emitter.h](../../../../include/pypto/codegen/code_emitter.h) |
| `CodeContext` | 变量名映射和指针跟踪 | [code_context.h](../../../../include/pypto/codegen/code_context.h) |
| `TypeConverter` | IR 类型到 pto-isa C++ 类型 | [type_converter.h](../../../../include/pypto/codegen/type_converter.h) |
| `ISAMapper` | IR 操作到 pto-isa 指令 | [isa_mapper.h](../../../../include/pypto/codegen/isa_mapper.h) |

## 核心组件

### CodeEmitter

管理带有适当缩进的结构化代码输出。

**关键方法:**

- `EmitLine(line)` - 带缩进发射行
- `IncreaseIndent()` / `DecreaseIndent()` - 管理缩进级别
- `GetCode()` - 获取累积的代码

### CodeContext

跟踪变量名映射和指针关联。

**关键特性:**

- 通过 `RegisterVar(var, cpp_name)` 将 IR 变量映射到 C++ 名称
- 通过 `SanitizeName(var)` 为 C++ 兼容性处理 IR 名称
- 通过 `RegisterPointer(tensor_var, ptr_name)` 跟踪张量 (Tensor) 到指针的映射
- 强制一次性注册 (防止重复声明)

**命名约定:**

- 函数参数: `input_a` -> `input_aGlobal` (GlobalTensor), `input_a` (原始指针)
- Tile 变量: 首次赋值时使用处理后的 IR 名称
- 普通变量: 首次赋值时使用处理后的名称

**指针跟踪:**
GlobalTensor 变量包装原始指针。对于地址运算 (如 `output + offset`), 需要原始指针名称。CodeContext 维护此映射, 并支持通过 ForStmt iter_args 和 IfStmt return_vars 进行指针继承。

### TypeConverter

将 PyPTO IR 类型转换为 pto-isa C++ 类型字符串。

**转换表:**

| PyPTO 数据类型 | C++ 类型 | PyPTO 内存空间 | 标注 |
| -------------- | -------- | -------------- | ---- |
| FP32 | `float` | DDR | `__gm__` |
| FP16 | `half` | UB/Left/Right/Acc | (无) |
| INT32 | `int32_t` | - | - |
| INT64 | `int64_t` | - | - |
| BOOL | `bool` | - | - |
| BF16 | `bfloat16` | - | - |

**形状/步幅:** 填充到 5D (前导 1), 行主序布局。

### ISAMapper

将 PyPTO IR 操作映射到 pto-isa 指令。

| IR 操作 | pto-isa | 类别 | 备注 |
| ------- | ------- | ---- | ---- |
| `block.load` | `TLOAD` | 内存 | DDR->UB |
| `block.store` | `TSTORE` | 内存 | UB->DDR |
| `block.add` / `sub` / `mul` / `div` | `TADD` / `TSUB` / `TMUL` / `TDIV` | 二元 | Tile+Tile |
| `block.adds` / `subs` / `muls` / `divs` | `TADDS` / `TSUBS` / `TMULS` / `TDIVS` | 二元 | Tile+标量 |
| `block.sqrt` | `TSQRT` | 一元 | 逐元素 |
| `block.sum` (axis=0/1) | `TCOLSUM` / `TROWSUM` | 归约 | 依赖轴 |
| `system.sync_src` | `set_flag` | 同步 | 设置标志 |
| `system.sync_dst` | `wait_flag` | 同步 | 等待标志 |
| `system.bar_v/m/all` | `pipe_barrier` | 同步 | 屏障 |

### CCECodegen

协调所有组件的主类。扩展 `IRVisitor`。

**入口点:** `std::string Generate(const FunctionPtr& func)`

## 代码生成流程

### 三阶段生成

#### 阶段 1: 序言

1. 带 `__aicore__` 和 `__attribute__((always_inline))` 的函数签名
2. 从 `int64_t* args` 数组解包参数
3. GlobalTensor 类型定义和实例
4. 带 TASSIGN 内存分配的 Tile 类型定义 (如存在 MemRef)

**TileCollector** 遍历函数体, 从 AssignStmt 节点发现 Tile 类型变量。IfStmt 的 return_vars 不会被收集; 它们在 if 语句 (Statement) 之前声明。

#### 阶段 2: 函数体

- 块操作 (TLOAD, TADD, TSTORE 等)
- 同步 (set_flag, wait_flag, pipe_barrier)
- 控制流 (循环、条件)
- 变量赋值

#### 阶段 3: 尾声

- 闭合大括号
- 可选清理

### 访问者方法

```cpp
void VisitExpr_(const CallPtr& op);         // Operations
void VisitStmt_(const AssignStmtPtr& op);   // Assignments
void VisitStmt_(const EvalStmtPtr& op);     // Sync operations
void VisitStmt_(const SeqStmtsPtr& op);     // Statement sequences
void VisitStmt_(const ForStmtPtr& op);      // Loops
void VisitStmt_(const IfStmtPtr& op);       // Conditionals
void VisitStmt_(const YieldStmtPtr& op);    // Yield values
```

## 使用示例

**Python API** (统一在 codegen 模块中: `codegen.PTOCodegen()`, `codegen.CCECodegen()`):

```python
from pypto.pypto_core import codegen
cg = codegen.CCECodegen()
cpp_code = cg.Generate(func)
```

**C++ API:**

```cpp
#include "pypto/codegen/cce_codegen.h"

FunctionPtr func = /* from IR */;
codegen::CCECodegen generator;
std::string cpp_code = generator.Generate(func);
```

**输入 IR (概念性):**

```python
def simple_add(x: Tensor([128, 64], FP32), y: Tensor([128, 64], FP32)):
    tile_x = block.load(x, [0, 0], [128, 64])
    tile_y = block.load(y, [0, 0], [128, 64])
    system.sync_src(PIPE_MTE2, PIPE_V, EVENT_ID0)
    system.sync_dst(PIPE_MTE2, PIPE_V, EVENT_ID0)
    tile_z = block.add(tile_x, tile_y)
    system.sync_src(PIPE_V, PIPE_MTE3, EVENT_ID0)
    system.sync_dst(PIPE_V, PIPE_MTE3, EVENT_ID0)
    result = block.store(tile_z, [0, 0], [128, 64], output)
```

**生成的 C++ (简化):**

```cpp
__aicore__ __attribute__((always_inline)) void runSimpleAdd(__gm__ int64_t* args) {
    // Unpack arguments
    __gm__ float* x = reinterpret_cast<__gm__ float*>(args[0]);
    __gm__ float* y = reinterpret_cast<__gm__ float*>(args[1]);
    __gm__ float* output = reinterpret_cast<__gm__ float*>(args[2]);

    // GlobalTensor declarations (types omitted for brevity)
    xGlobalType xGlobal(x);
    yGlobalType yGlobal(y);
    outputGlobalType outputGlobal(output);

    // Tile declarations
    tile_xType tile_x(128, 64);
    TASSIGN(tile_x, 0x0);
    tile_yType tile_y(128, 64);
    TASSIGN(tile_y, 0x10000);
    tile_zType tile_z(128, 64);
    TASSIGN(tile_z, 0x20000);

    // Function body
    TLOAD(tile_x, xGlobal);
    TLOAD(tile_y, yGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TADD(tile_z, tile_x, tile_y);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(outputGlobal, tile_z);
}
```

## 实现细节

### 内存地址管理

UB 内存地址来自 IR 元数据中 TileType 的 MemRef 字段:

- 转换 Pass 设置 `TileType::memref_::addr_` (ConstInt 表达式 (Expression))
- 代码生成器提取地址并格式化为十六进制 (如 `0x0`, `0x10000`)
- TASSIGN 指令将 Tile 绑定到特定 UB 地址
- 如果没有 MemRef, 则跳过 TASSIGN (未来由 AllocOp 处理分配)

**指针跟踪:** CodeContext 维护张量到指针的映射, 用于 TASSIGN 指令中的正确地址运算。支持通过控制流继承。

### 双模式表达式 (Expression) 模式

表达式访问者以两种模式运行:

#### 模式 1: 语句发射模式 (调用表达式)

- 输入: `current_target_var_` 包含赋值目标
- 行为: 发射完整的指令语句
- 输出: 清除 `current_expr_value_`
- 示例: `tile_z = block.add(tile_x, tile_y)` -> `TADD(tile_z, tile_x, tile_y);`

#### 模式 2: 值返回模式 (标量表达式)

- 输入: 表达式树
- 行为: 生成内联 C++ 代码
- 输出: 设置 `current_expr_value_` 为内联代码
- 示例: `i * 128 + j` -> `"(i * 128 + j)"`

### 同步策略

同步在 IR 中是显式的:

- 转换 Pass 插入 `system.sync_src/dst` 操作
- 代码生成器直接转换为 `set_flag/wait_flag`
- 不进行自动同步推断

**典型模式:**

```text
Load:  TLOAD -> set_flag -> wait_flag
Compute: TADD -> set_flag -> wait_flag
Store: TSTORE
```

## 控制流生成

### ForStmt (循环)

**简单循环:**

```cpp
for (int64_t i = start; i < stop; i += step) {
    // body
}
```

**带 iter_args 的循环 (循环携带值):**

```cpp
sum = init_value;  // Initialize
for (int64_t i = start; i < stop; i += step) {
    // body updates sum via yield
    sum = yielded_value;
}
// return_var registered as "sum" (no separate assignment)
```

**特性:**

- 循环变量通过自动注册确定作用域
- YieldStmt 用新值更新 iter_args
- 返回变量与最终 iter_arg 状态共享 C++ 名称

### IfStmt (条件)

**基本 if/if-else:**

```cpp
if (condition) { /* then */ }
if (condition) { /* then */ } else { /* else */ }
```

**带返回值的 if-else:**

```cpp
// Declare return variables BEFORE if statement
output_finalType output_final(128, 64);
TASSIGN(output_final, 0x20000);  // If memref present

if (has_tail) {
    // ... compute output_with_tail ...
    output_final = output_with_tail;  // Assign after then_body
} else {
    output_final = output_updated;    // Assign after else_body
}
```

**特性:**

- 返回变量在 if 语句之前声明, 包含完整类型定义
- TileType 在存在 MemRef 时包含 TASSIGN
- GlobalTensor 使用形状/步幅类型声明
- 每个分支独立赋值返回值
- GlobalTensor 赋值继承指针映射

### YieldStmt

将值从语句体传递到包含它的控制结构:

- **ForStmt**: 为下一次迭代更新 iter_args
- **IfStmt**: 在分支完成后赋值给 return_vars

实现在遍历期间将值存储在 `yield_buffer_` 中。

## 错误处理

使用 PyPTO 错误约定:

- `CHECK` 用于用户输入验证 (抛出 `pypto::ValueError`)
- `INTERNAL_CHECK` 用于内部不变量
- 不使用原生 C++ 异常

## 测试

**位置:** [tests/ut/codegen/](../../../../tests/ut/codegen/)

**测试文件:**

- `test_type_converter.py` - 数据类型、形状、步幅转换
- `test_isa_mapper.py` - 操作映射
- `test_cce_codegen.py` - 集成测试

**运行测试:**

```bash
pytest tests/ut/codegen/          # All tests
pytest tests/ut/codegen/test_type_converter.py  # Specific file
pytest -v tests/ut/codegen/       # Verbose
```

**覆盖范围:**

- 类型转换 (DataType, Shape, Stride)
- 操作映射 (20+ 操作)
- 函数生成 (签名、序言、函数体、尾声)
- GlobalTensor 和 Tile 生成
- 块操作 (TLOAD, TSTORE, TADD, TMUL 等)
- 标量操作 (TADDS, TSUBS, TMULS, TDIVS)
- 归约操作 (带轴的求和)
- 同步 (set_flag, wait_flag, 屏障)
- 控制流 (ForStmt, IfStmt, YieldStmt, 嵌套)

## 未来增强

**计划中:**

1. 动态形状 (运行时形状参数)
2. 增强的表达式处理 (嵌套表达式、常量折叠)
3. 优化 (死代码消除、CSE、指令调度)
4. 调试支持 (打印语句、性能分析、源码跟踪)

**可扩展性:**

- 添加操作: 更新 `ISAMapper::InitializeMappings()` + CCECodegen 中的可选处理
- 添加类型: 更新 `TypeConverter::ConvertDataType()`
- 添加访问者方法: 在 CCECodegen 中重写 + 测试

## 参考资料

- [IR 概述](../ir/00-overview.md)
- [IR 层次结构](../ir/01-hierarchy.md)
- [访问者模式](../../../../include/pypto/ir/transform/base/visitor.h)
- [Pass 系统](../passes/00-pass_manager.md)
- [pto-isa 文档](https://gitcode.com/cann/pto-isa)

## 总结

PyPTO 代码生成提供了一个简洁、模块化的 IR 到 C++ 代码转换系统:

**设计:**

- 独立架构 (不是 Pass)
- 基于访问者模式的遍历
- 模块化组件, 职责单一
- 可扩展 (易于添加操作/类型)

**已实现:**

- 带 `__aicore__` 属性的函数生成
- 参数解包和 GlobalTensor 定义
- Tile 类型定义和 TASSIGN 分配
- 20+ 块/同步/屏障操作
- 控制流 (循环、条件、yield)
- 变量名管理和指针跟踪
- 全面的测试覆盖 (31 个测试)

基础架构已经完善, 可以支持动态形状、优化和增强的表达式处理。
