# 编排代码生成（Orchestration Codegen）

## 概述

编排代码生成器（Orchestration Codegen）生成 PTO2 运行时 C++ 代码，用于管理昇腾硬件上的任务图执行。[PTO 代码生成](00-pto_codegen.md)产生 InCore 核函数代码（Tile 级计算），而编排代码生成器产生主机侧代码，负责：

- 将设备内存指针（通过 `TaskArg`）封装为 `Tensor` 对象
- 构建 `PTOParam` 对象，调用 `add_input`/`add_output`/`add_inout`/`add_scalar` 对参数分类
- 通过 `pto2_rt_submit_*_task` 向 AIC（CUBE）或 AIV（VECTOR）核心提交任务
- 处理控制流（循环、条件分支），使用 `PTO2_SCOPE`

**流水线：** `IR（Orchestration 函数）→ OrchestrationCodegen → C++（PTO2 运行时 API）`

**源码位置：** `src/codegen/orchestration/orchestration_codegen.cpp`

## 架构

### 组件结构

| 组件 | 职责 | 位置 |
| ---- | ---- | ---- |
| `OrchestrationInfoCollector` | IR 访问器，收集元数据（元组映射、张量赋值） | orchestration_codegen.cpp |
| `OrchestrationStmtCodegen` | 语句级 C++ 代码生成器（继承 CodegenBase） | orchestration_codegen.cpp |
| `OrchestrationOpRegistry` | 张量操作代码生成处理器的单例注册表 | orchestration_op_registry.h |
| `GenerateOrchestration()` | 主入口函数，组合所有生成阶段 | orchestration_codegen.cpp |
| `VarLineageCollector` | 通过 VarPtr 身份追踪函数体变量到函数参数的来源 | orchestration_codegen.cpp |
| `GetSSABaseName()` | 剥离 SSA/流水线后缀用于 C++ 名称生成（非身份判定） | orchestration_codegen.cpp |

### OrchestrationInfoCollector

IR 访问器，预扫描函数体以收集：

- **元组元素映射** — 追踪哪些变量来自元组解构
- **调用-元组键** — 唯一键（`_tc_N`）防止跨调用冲突
- **输出张量赋值** — 将变量名映射到其赋值语句

### OrchestrationStmtCodegen

主代码生成器。访问每条 IR 语句并生成对应的 C++：

- **AssignStmt** → 张量操作、函数调用或别名生成
- **ForStmt** → `for` 循环及迭代参数初始化和 yield 更新
- **IfStmt** → 每个分支带 `PTO2_SCOPE` 的条件块及返回变量处理
- **YieldStmt** → 循环携带值的变量重赋值

### 操作注册表

张量操作通过 `REGISTER_ORCHESTRATION_OP` 宏注册：

```cpp
REGISTER_ORCHESTRATION_OP("tensor.create", TensorCreateHandler);
REGISTER_ORCHESTRATION_OP("tensor.read", TensorReadHandler);
REGISTER_ORCHESTRATION_OP("tensor.slice", TensorSliceHandler);
```

这允许在不修改核心访问器的情况下扩展操作代码生成。

## 代码生成流程

`GenerateOrchestration()` 分 9 个阶段生成 C++：

### 阶段 1–2：模板代码

```cpp
// 阶段 1：头文件包含
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include "pto_orchestration_api.h"

// 阶段 2：辅助函数
static uint64_t float_to_u64(float f) { ... }
static inline Tensor make_tensor_external_2d_dn(...) { ... }
static inline Tensor make_tensor_2d_dn(...) { ... }
```

### 阶段 3–4：入口点

```cpp
// 阶段 3：配置函数 — 返回期望的参数数量
PTO2OrchestrationConfig aicpu_orchestration_config(TaskArg* orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{ .expected_arg_count = 3 };
}

// 阶段 4：入口函数签名
// A2/A3：
void aicpu_orchestration_entry(TaskArg* orch,
    int arg_count, int orch_thread_num, int orch_thread_index) {
// A5（Ascend950）：在参数列表前增加 PTO2Runtime* rt
// void aicpu_orchestration_entry(PTO2Runtime* rt, TaskArg* orch, ...)
```

### 阶段 5–6：张量设置

```cpp
// 阶段 5：外部张量 — ND 布局直接调用 from_task_arg()
Tensor ext_a = from_task_arg(orch[0]);
Tensor ext_b = from_task_arg(orch[1]);

// DN 布局：传入逻辑形状 — make_tensor_external_2d_dn 内部处理轴转置
uint32_t dn_shapes[2] = {orch[2].tensor.shapes[0], orch[2].tensor.shapes[1]};
Tensor ext_dn = make_tensor_external_2d_dn(orch[2].data<void>(), dn_shapes, 2, DataType::FLOAT32);

// 阶段 6：内部张量（来自 pl.create_tensor — 仅中间变量）
uint32_t tmp_shapes[2] = {16, 16};
Tensor tmp = make_tensor(tmp_shapes, 2, DataType::FLOAT32);
```

### 阶段 7–9：任务提交与控制流

所有任务提交包裹在顶层 `PTO2_SCOPE()` 中：

```cpp
// 阶段 7–9：顶层 PTO2_SCOPE 包裹所有任务提交
// A2/A3：PTO2_SCOPE()   A5：PTO2_SCOPE(rt)
PTO2_SCOPE() {
    PTOParam params_t0;
    params_t0.add_input(ext_a);
    params_t0.add_input(ext_b);
    params_t0.add_output(ext_output);
    // A2/A3：pto2_rt_submit_aiv_task(0, params_t0)
    // A5：   pto2_rt_submit_aiv_task(rt, 0, params_t0)
    pto2_rt_submit_aiv_task(0, params_t0);

    // ForStmt 示例 — 普通 for 循环，不嵌套独立的 PTO2_SCOPE
    for (int64_t i = start; i < stop; i += step) {
        // 任务提交
    }
}
```

## 核心概念

### 外部张量 vs 内部张量

| 类型 | 来源 | C++ 构造方式 | 命名 |
| ---- | ---- | ------------ | ---- |
| 外部（ND） | 函数参数（`In`/`Out`/`InOut`） | `from_task_arg(orch[N])` | `ext_<name>` |
| 外部（DN） | 函数参数，DN 布局 | `make_tensor_external_2d_dn(orch[N].data<void>(), {orch[N].tensor.shapes[0], orch[N].tensor.shapes[1]}, ...)` — 轴排序由函数内部处理 | `ext_<name>` |
| 内部（Internal） | 函数体中的 `pl.create_tensor(...)` | `make_tensor(shapes, ndims, dtype)` | `<name>`（无前缀） |

外部张量封装从主机通过 `TaskArg` 传入的设备内存指针。内部张量是运行时分配的临时工作空间。

### 参数方向

每个函数参数的 `ParamDirection` 决定其在任务提交中的表现：

| 方向 | Python 注解 | C++ 任务参数 | 语义 |
| ---- | ----------- | ------------ | ---- |
| `In` | `pl.Tensor[...]`（默认） | `params.add_input(ext_x)` | 只读 |
| `Out` | `pl.Out[pl.Tensor[...]]` | `params.add_output(ext_x)` | 只写 |
| `InOut` | `pl.InOut[pl.Tensor[...]]` | `params.add_inout(ext_x)` | 读写 |
| Scalar | `pl.Scalar[...]` | `params.add_scalar(value)` | 标量常量（所有张量之后） |

### 别名生成

当 InCore 调用的返回值名称与 `Out` 参数名称不同时，代码生成器会发出 C++ 引用别名：

```python
# Python IR
result = self.kernel_add(a, b, output)  # result ≠ output
```

```cpp
// 生成的 C++
PTOParam params_t0;
params_t0.add_output(ext_output);
pto2_rt_submit_aiv_task(0, params_t0);
Tensor& result = ext_output;  // 别名 — result 引用 ext_output
```

如果返回名称与 `Out`/`InOut` 参数名称匹配，则不需要别名。`InOut` 参数永不生成别名 — 其本身已是外部张量。

### 后端差异（A5 / Ascend950）

针对 Ascend950（`BackendType::Ascend950`）时，生成的 C++ 在三处有所不同：

| 元素 | A2/A3 | A5 |
| ---- | ----- | -- |
| 入口函数 | `aicpu_orchestration_entry(TaskArg* orch, ...)` | `aicpu_orchestration_entry(PTO2Runtime* rt, TaskArg* orch, ...)` |
| Scope 宏 | `PTO2_SCOPE()` | `PTO2_SCOPE(rt)` |
| 提交调用 | `pto2_rt_submit_aiv_task(id, params)` | `pto2_rt_submit_aiv_task(rt, id, params)` |

`rt` 参数是显式的 `PTO2Runtime*` 指针，A5 通过调用链传递，而非依赖线程局部存储。

### 核心类型推断

代码生成器根据被调用函数的 `MemorySpace` 决定提交到 AIC（CUBE）还是 AIV（VECTOR）：

| MemorySpace | 核心类型 | 提交函数 |
| ----------- | -------- | -------- |
| `Left`、`Right`、`Acc` | CUBE (AIC) | `pto2_rt_submit_aic_task` |
| `Vec`、`Mat`（默认） | VECTOR (AIV) | `pto2_rt_submit_aiv_task` |

### 元组处理

元组返回的调用使用唯一键（`_tc_N`）追踪元素：

```python
# Python IR
pij, mij, lij = self.kernel_softmax(sij, scale, pij, mij, lij)
```

```cpp
// 生成的 C++ — 先张量后标量
PTOParam params_t0;
params_t0.add_input(ext_sij);
params_t0.add_output(ext_pij);
params_t0.add_output(ext_mij);
params_t0.add_output(ext_lij);
params_t0.add_scalar(float_to_u64(scale));  // 标量在所有张量之后
pto2_rt_submit_aiv_task(0, params_t0);
```

### Group 函数（混合核）

当核函数同时使用 AIC 和 AIV 核心（混合核）时，代码生成器生成 `MixedKernels` 提交：

```cpp
// Group: mixed_kernel (AIC + AIV)
PTOParam params_t0;
// ... add_input / add_output / add_scalar 调用 ...
MixedKernels mixed_0 = {aic_id, aiv_id, INVALID_KERNEL_ID};
// A2/A3：pto2_rt_submit_task(mixed_0, params_t0)
// A5：   pto2_rt_submit_task(rt, mixed_0, params_t0)
pto2_rt_submit_task(mixed_0, params_t0);
```

## 操作映射

| IR 操作 | C++ 代码生成 | 描述 |
| ------- | ------------ | ---- |
| `tensor.create` | `make_tensor(shapes, ndims, dtype)` | 分配内部张量 |
| `tensor.read` | `*reinterpret_cast<T*>(arg_ptr + offset)` | 从主机张量读取标量 |
| `tensor.slice` | `make_tensor_external(ptr + byte_offset, ...)` | 创建现有张量的视图 |
| `tensor.dim`（静态） | `int64_t d0 = 16` | 编译时常量维度值 |
| `tensor.dim`（动态） | `int64_t d0 = (int64_t)orch[N].tensor.shapes[axis]` | 从 TaskArg 获取运行时维度 |

## 完整示例

### 输入：PyPTO 编排函数

```python
@pl.function(type=pl.FunctionType.Orchestration)
def orch_basic(
    self,
    a: pl.Tensor[[16, 16], pl.FP32],
    b: pl.Tensor[[16, 16], pl.FP32],
    d: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
) -> pl.Tensor[[16, 16], pl.FP32]:
    c: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
    c = self.kernel_add(a, b, c)       # c 是内部张量（中间变量）
    d = self.kernel_add(c, b, d)       # d 是外部张量（Out 参数）
    return d
```

### 输出：生成的 C++

```cpp
// Orchestration Function: orch_basic
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include "pto_orchestration_api.h"

static uint64_t float_to_u64(float f) { /* ... */ }

extern "C" {

PTO2OrchestrationConfig aicpu_orchestration_config(TaskArg* orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{ .expected_arg_count = 3 };
}

void aicpu_orchestration_entry(TaskArg* orch,
    int arg_count, int orch_thread_num, int orch_thread_index) {
    // 注意：A5 在参数列表前增加 PTO2Runtime* rt
    (void)arg_count;
    (void)orch_thread_num;
    (void)orch_thread_index;

    // 外部张量（来自 TaskArg）
    Tensor ext_a = from_task_arg(orch[0]);
    Tensor ext_b = from_task_arg(orch[1]);
    Tensor ext_d = from_task_arg(orch[2]);

    // 内部张量（中间变量）
    uint32_t c_shapes[2] = {16, 16};
    Tensor c = make_tensor(c_shapes, 2, DataType::FLOAT32);

    // A2/A3：PTO2_SCOPE()   A5：PTO2_SCOPE(rt)
    PTO2_SCOPE() {
        // 任务 0: kernel_add (a + b → c)
        PTOParam params_t0;
        params_t0.add_input(ext_a);
        params_t0.add_input(ext_b);
        params_t0.add_output(c);
        // A2/A3：pto2_rt_submit_aiv_task(0, params_t0)
        // A5：   pto2_rt_submit_aiv_task(rt, 0, params_t0)
        pto2_rt_submit_aiv_task(0, params_t0);

        // 任务 1: kernel_add (c + b → d)
        PTOParam params_t1;
        params_t1.add_input(c);
        params_t1.add_input(ext_b);
        params_t1.add_output(ext_d);
        pto2_rt_submit_aiv_task(1, params_t1);
    }
}

}  // extern "C"
```

## 变量命名

### 基于 VarPtr 的变量身份追踪

变量身份判定（该变量是否为参数？两个变量是否为同一张量？）使用基于
`VarPtr` 指针的身份识别，而非字符串匹配。`VarLineageCollector` 在代码生成
前遍历函数体，通过 ForStmt iter_arg/return_var 链和简单的 Var-to-Var 赋值，
将每个函数体 `Var*` 追踪回其源函数参数 `Var*`。这避免了后缀剥离导致的名称
冲突问题（例如 `out_0` → `out` 合并了不同变量）。

`GetSSABaseName()` 仍用于 C++ 代码生成（生成输出中的清晰变量名），
但不再用于身份判定。

### 命名约定

| 实体 | 模式 | 示例 |
| ---- | ---- | ---- |
| 外部张量 | `ext_<name>` | `ext_a` |
| 内部张量 | `<name>`（无前缀） | `c` |
| 任务参数 | `params_t<N>` | `params_t0` |
| TaskArg 索引 | `orch[N]`（第 N 个张量参数） | `orch[0]` |

## 控制流生成

### ForStmt

```python
# Python IR
for i in pl.range(0, 4):
    acc = self.kernel_add(a, acc, acc)
```

```cpp
// 生成的 C++（位于顶层 PTO2_SCOPE 内部）
Tensor acc = ext_acc;  // 迭代参数初始化
for (int64_t i = 0; i < 4; i += 1) {
    PTOParam params_t0;
    // ... add_input / add_output 调用 ...
    pto2_rt_submit_aiv_task(0, params_t0);
}
```

迭代参数在循环前初始化。`YieldStmt` 更新在每次迭代末尾发出。

### IfStmt

```python
# Python IR
if condition:
    c = self.kernel_a(a, b, c)
else:
    c = self.kernel_b(a, b, c)
```

```cpp
// 生成的 C++
if (condition) {
    // A2/A3：PTO2_SCOPE()   A5：PTO2_SCOPE(rt)
    PTO2_SCOPE() {
        PTOParam params_t0;
        // ... add_input / add_output 调用 ...
        pto2_rt_submit_aiv_task(0, params_t0);
    }
} else {
    PTO2_SCOPE() {
        PTOParam params_t1;
        // ... add_input / add_output 调用 ...
        pto2_rt_submit_aiv_task(1, params_t1);
    }
}
```

## Python API

```python
from pypto import codegen, backend

backend.set_backend_type(backend.BackendType.Ascend910B)
result = codegen.generate_orchestration(MyProgram, orch_func)
code = result.code

# 访问生成的编排代码
orch_code = files["orchestration/orch_func_name.cpp"]
```

编排文件在生成的文件映射中命名为 `orchestration/<func_name>.cpp`。

## 参见

- [PTO 代码生成](00-pto_codegen.md) — PTO 后端的 MLIR 生成
- [Pass 管理器](../passes/00-pass_manager.md) — 代码生成前应用的 IR 优化 Pass
