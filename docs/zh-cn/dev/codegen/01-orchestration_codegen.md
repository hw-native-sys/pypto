# 编排代码生成（Orchestration Codegen）

## 设计原则：严格的 1-to-1 映射

编排代码生成遵循与 [PTO 代码生成](00-pto_codegen.md#设计原则严格的-1-to-1-映射)相同的原则：从 IR 到生成 C++ 代码的**严格 1-to-1 转换**。代码生成不应执行优化、分析或间接转换——此类工作属于前置 Pass。

例如，返回值到参数的追踪（将被调用者返回值映射回 `Out` 参数）是分析工作，应由代码生成之前的 Pass 解决。[`NormalizeReturnOrder`](../passes/23-normalize_return_order.md) pass 现在会在代码生成之前完成此规范化，使编排代码生成可以直接将 `return[i]` 映射到 `out_indices[i]`，无需追踪 `tile.store`/yield 链。

## 概述

编排代码生成器（Orchestration Codegen）生成 PTO2 运行时 C++ 代码，用于管理昇腾硬件上的任务图执行。[PTO 代码生成](00-pto_codegen.md)产生 InCore 核函数代码（Tile 级计算），而编排代码生成器产生主机侧代码，负责：

- 将设备内存指针（通过 `ChipStorageTaskArgs`）封装为 `Tensor` 对象
- 构建 `Arg` 对象，调用 `add_input`/`add_output`/`add_inout`/`add_scalar` 对参数分类（manual scope 的依赖边通过一个 `set_dependencies` 栈数组单独发出——见 [Manual Scope 与 TaskId 降级](#manual-scope-与-taskid-降级)）
- 通过 `rt_submit_*_task` 向 AIC（CUBE）或 AIV（VECTOR）核心提交任务
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

### 阶段 1：模板代码

```cpp
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include "pto_orchestration_api.h"
```

### 阶段 2–3：入口点

```cpp
// 阶段 2：配置函数 — 返回期望的参数数量
PTO2OrchestrationConfig aicpu_orchestration_config(const ChipStorageTaskArgs& orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{ .expected_arg_count = 3 };
}

// 阶段 3：入口函数签名
void aicpu_orchestration_entry(const ChipStorageTaskArgs& orch_args) {
```

### 阶段 4–5：张量设置

```cpp
// 阶段 4：外部张量 — 所有布局统一调用 from_tensor_arg()
Tensor ext_a = from_tensor_arg(orch_args.tensor(0));
Tensor ext_b = from_tensor_arg(orch_args.tensor(1));
Tensor ext_dn = from_tensor_arg(orch_args.tensor(2));

// 阶段 5：内部张量（来自 pl.create_tensor — 仅中间变量）
// 同一 scope 中的所有 tensor.create 批量合并为一条 alloc_tensors 调用
uint32_t tmp_ci_shapes[2] = {16, 16};
TensorCreateInfo tmp_ci(tmp_ci_shapes, 2, DataType::FLOAT32);
TaskOutputTensors alloc_0 = alloc_tensors(tmp_ci);
const Tensor& tmp = alloc_0.get_ref(0);
```

### 阶段 6–8：任务提交与控制流

所有任务提交包裹在顶层 `PTO2_SCOPE()` 中。codegen 不再依据 `for` / `if` 结构
决定 scope 位置：[MaterializeRuntimeScopes](../passes/38-materialize_runtime_scopes.md)
pass 会向 IR 中插入显式的 AUTO `RuntimeScopeStmt` 节点（函数体以及每个
`for` / `if` 体），codegen 从这些节点 1:1 地 emit `PTO2_SCOPE`（manual scope
降级为 `PTO2_SCOPE(PTO2ScopeMode::MANUAL)`）：

```cpp
PTO2_SCOPE() {
    Arg params_t0;
    params_t0.add_input(ext_a);
    params_t0.add_input(ext_b);
    params_t0.add_output(tmp);               // 预分配张量使用 add_output(const Tensor&)
    rt_submit_aiv_task(0, params_t0);

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
| 外部（ND/DN） | 函数参数（`In`/`Out`/`InOut`） | `from_tensor_arg(orch_args.tensor(N))` | `ext_<name>` |
| 内部 | 函数体中的 `pl.create_tensor(...)` | `TensorCreateInfo var_ci(...)` + scope 入口处 `alloc_tensors(...)` | `<name>`（无前缀） |

外部张量封装从主机通过 `ChipStorageTaskArgs` 传入的设备内存指针。内部张量在 scope 入口处通过 `alloc_tensors()` 预分配——同一 scope（函数体、for 循环体、if 分支体）中的所有 `tensor.create` 被批量合并为一条 `alloc_tensors` 调用。预分配的张量随后通过 `add_output(const Tensor&)` (OUTPUT_EXISTING 重载) 传递给核函数。

### 参数方向

每个函数参数的 `ParamDirection` 决定其在任务提交中的表现：

| 方向 | Python 注解 | C++ 任务参数 | 语义 |
| ---- | ----------- | ------------ | ---- |
| `In` | `pl.Tensor[...]`（默认） | `params.add_input(var)` | 只读 |
| `Out`（外部） | `pl.Out[pl.Tensor[...]]`（参数） | `params.add_output(ext_x)` | 只写预分配缓冲区 |
| `Out`（内部） | `pl.Out[pl.Tensor[...]]`（tensor.create） | `params.add_output(x)` | 通过 `alloc_tensors` 预分配，使用 OUTPUT_EXISTING 重载 |
| `InOut` | `pl.InOut[pl.Tensor[...]]` | `params.add_inout(ext_x)` | 读写 |
| Scalar | `pl.Scalar[...]` | `params.add_scalar(value)` | 标量常量（独立 scalar 槽位） |

来自 `tensor.create` 的内部张量在 scope 入口通过 `alloc_tensors()` 预分配。传递给核函数时，使用 `add_output(const Tensor&)` 触发 OUTPUT_EXISTING 重载——运行时复用预分配的缓冲区，而非分配新的。

### 标量参数编码

标量参数占用 `ChipStorageTaskArgs` 的 scalar 槽位（从 0 开始独立索引，与张量槽位分离）。
浮点标量使用 `to_u64(f)` 进行位转换，其他整数/bool 标量强制转换为 `(uint64_t)`。
接收端使用联合体（union）进行类型双关，将 `uint64_t` 重新解释为目标 C 类型：

```cpp
union { uint64_t u64; float val; } scale_conv;
scale_conv.u64 = orch_args.scalar(0);
float scale = scale_conv.val;
```

### 输出别名（emit 名重映射）

kernel/submit 的输出就是它原地写入的 `Out`/`InOut` 参数——即*同一物理张量*。因此当
结果 Var 的名字与该参数不同时，代码生成器**不**再生成 `const Tensor& result =
ext_output;` 这样的重命名，而是把结果 Var 的 emit 名重映射到源，下游所有引用都直接
解析到源名。（这正是 `tensor.assemble` 采用的策略，现统一应用。）

```python
# Python IR
result = self.kernel_add(a, b, output)  # result ≠ output
consumer = self.kernel_use(result)
```

```cpp
// 生成的 C++ —— result 被重映射到 ext_output，消费者直接读取它
Arg params_t0;
params_t0.add_output(ext_output);
rt_submit_aiv_task(0, params_t0);

Arg params_t1;
params_t1.add_input(ext_output);  // result -> ext_output（无别名声明）
```

结果别名到哪个 `Out`/`InOut` 参数是查表而非启发式：流水线 IR 满足
`ReturnParamsExplicit` 属性
（[`NormalizeReturnOrder`](../passes/23-normalize_return_order.md)），
`FindReturnedParamIndex` 通过 `ir::return_lineage` 以指针同一性把每个返回值解析到参数。旧的血缘追踪（Var 到 Var 别名、循环 carry、builtin 回
写、tuple 调用的 `TupleGetItem`、Group/Spmd 包装函数）仅保留给手工解析的
IR。当追踪不到任何参数时，仅在被调用者恰好只有一个 `Out`/`InOut` 时单返回
值才回退到该唯一参数——多输出且无法追踪是内部错误，绝不猜测。

不参与重映射的情形：phi/循环 carry 的重赋值（它重新绑定外层 `if`/循环所拥有的左值）
保留 `<name> = <src>;` 形式；源在读取者的 C++ 作用域中无效的张量（manual scope 局部
的源——见下文*跨作用域张量与 `manual_scope`*）保留声明路径；绑定到
`task_<n>_outs.get_ref(k)` 的运行时分配输出同样保留其 `const Tensor&` 绑定。

### 核心类型推断

代码生成器根据被调用函数的 `MemorySpace` 决定提交到 AIC（CUBE）还是 AIV（VECTOR）：

| MemorySpace | 核心类型 | 提交函数 |
| ----------- | -------- | -------- |
| `Left`、`Right`、`Acc`、`Mat` | CUBE (AIC) | `rt_submit_aic_task` |
| `Vec`（默认） | VECTOR (AIV) | `rt_submit_aiv_task` |

### 元组处理

元组返回的调用使用唯一键（`_tc_N`）追踪元素：

```python
# Python IR
pij, mij, lij = self.kernel_softmax(sij, scale, pij, mij, lij)
```

```cpp
// 生成的 C++ — 先张量后标量
Arg params_t0;
params_t0.add_input(ext_sij);
params_t0.add_inout(ext_pij);
params_t0.add_inout(ext_mij);
params_t0.add_inout(ext_lij);
params_t0.add_scalar(to_u64(scale));  // 标量在所有张量之后
rt_submit_aiv_task(0, params_t0);
```

### Group 函数（混合核）

当核函数同时使用 AIC 和 AIV 核心（混合核）时，代码生成器生成 `MixedKernels` 提交：

```cpp
// Group: mixed_kernel (AIC + AIV)
Arg params_t0;
// ... add_input / add_inout / add_scalar 调用 ...
MixedKernels mixed_0 = {aic_id, aiv_id, INVALID_KERNEL_ID};
rt_submit_task(mixed_0, params_t0);
```

## 操作映射

| IR 操作 | C++ 代码生成 | 描述 |
| ------- | ------------ | ---- |
| `tensor.create` | `TensorCreateInfo var_ci(...)` + `alloc_tensors(...)` | scope 级批量分配；`const Tensor& var = alloc_N.get_ref(i)` |
| `tensor.read` | `*reinterpret_cast<T*>(arg_ptr + offset)` | 从主机张量读取标量 |
| `tensor.slice` | `make_tensor_external(ptr + byte_offset, ...)` | 创建现有张量的视图 |
| `tensor.transpose` | `Tensor xt = ext_x.transpose(axis1, axis2)` | 零拷贝交换两个维度的元数据（lower 到运行时 `Tensor::transpose`） |
| `tensor.dim`（静态） | `int64_t d0 = 16` | 编译时常量维度值 |
| `tensor.dim`（动态） | `int64_t d0 = (int64_t)orch_args.tensor(N).shapes[axis]` | 从 ChipStorageTaskArgs 获取运行时维度 |

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

extern "C" {

PTO2OrchestrationConfig aicpu_orchestration_config(const ChipStorageTaskArgs& orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{ .expected_arg_count = 3 };
}

void aicpu_orchestration_entry(const ChipStorageTaskArgs& orch_args) {
    // 外部张量（来自 ChipStorageTaskArgs）
    Tensor ext_a = from_tensor_arg(orch_args.tensor(0));
    Tensor ext_b = from_tensor_arg(orch_args.tensor(1));
    Tensor ext_d = from_tensor_arg(orch_args.tensor(2));

    PTO2_SCOPE() {
        // 内部张量 — 在 scope 入口通过 alloc_tensors 预分配
        uint32_t c_ci_shapes[2] = {16, 16};
        TensorCreateInfo c_ci(c_ci_shapes, 2, DataType::FLOAT32);
        TaskOutputTensors alloc_0 = alloc_tensors(c_ci);
        const Tensor& c = alloc_0.get_ref(0);

        // 任务 0: kernel_add (a + b → c)
        Arg params_t0;
        params_t0.add_input(ext_a);
        params_t0.add_input(ext_b);
        params_t0.add_output(c);
        rt_submit_aiv_task(0, params_t0);

        // 任务 1: kernel_add (c + b → d)
        Arg params_t1;
        params_t1.add_input(c);
        params_t1.add_input(ext_b);
        params_t1.add_output(ext_d);
        rt_submit_aiv_task(1, params_t1);
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
| 内部 TensorCreateInfo | `<name>_ci` | `c_ci` |
| 任务参数 | `params_t<N>` | `params_t0` |
| 分配结果 | `alloc_<N>` | `alloc_0` |
| 张量参数索引 | `orch_args.tensor(N)` | `orch_args.tensor(0)` |
| 标量参数索引 | `orch_args.scalar(N)` | `orch_args.scalar(0)` |

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
    Arg params_t0;
    // ... add_input / add_inout 调用 ...
    rt_submit_aiv_task(0, params_t0);
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
    PTO2_SCOPE() {
        Arg params_t0;
        // ... add_input / add_inout 调用 ...
        rt_submit_aiv_task(0, params_t0);
    }
} else {
    PTO2_SCOPE() {
        Arg params_t1;
        // ... add_input / add_inout 调用 ...
        rt_submit_aiv_task(1, params_t1);
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

## Manual Scope 与 TaskId 降级

`with pl.manual_scope():` 区域被降级为 `PTO2_SCOPE(PTO2ScopeMode::MANUAL)`
代码块，区域内 runtime 的 auto OverlapMap 关闭。每个 task 的 params 始终
声明为普通的 `Arg <task_var>;`。orchestration codegen 把所需的依赖边
物化为一个定长栈数组加一次 `set_dependencies` 调用：

```cpp
Arg params_t1;
params_t1.add_input(...);
// ...
PTO2TaskId params_t1_deps[K];          // K = 精确的 dep 边数
uint32_t params_t1_deps_count = 0;
if (tid.is_valid()) params_t1_deps[params_t1_deps_count++] = tid;      // 每个条目都有 is_valid() 守卫
if (carry.is_valid()) params_t1_deps[params_t1_deps_count++] = carry;
params_t1.set_dependencies(params_t1_deps, params_t1_deps_count);
```

每个 dep 槽位都被 `if (task_id.is_valid())` 包裹：任何 TaskId 都可能合法地
持有 `PTO2TaskId::invalid()` 哨兵——`None` 循环 carry 种子、循环首次迭代的
iter_arg carry，或未写入的数组槽——invalid id 绝不能进入
`set_dependencies`。对已知有效的 id，该守卫只是一个恒真的廉价分支。

不再有 `params.add_dep(...)` 调用，也没有 16 条依赖上限——runtime 的
`Arg::set_dependencies` 原语没有上限，栈数组按精确数量定长。dep 边
用户依赖来自 parser：parser 把用户的 `pl.submit(..., deps=[tid1, tid2])`
kwarg 写入类型化的 `Submit::deps_` 字段；codegen 通过临时的 `SubmitToCallView`
读取它们——该 view 把 `deps_` 合成为 `attrs["manual_dep_edges"]`（一个
`vector<VarPtr>`，每项为 `Scalar[TASK_ID]` 类型的 Var）。普通 `Call` 携带
`manual_dep_edges` 的形态已不存在——ManualDepsOnSubmitOnly 结构性属性会校验
任何跨函数 `Call` 都不携带它；只有 `system.task_dummy` barrier op 作为 fanin
契约保留该 attr。编译器推导的 manual-scope 依赖来自
[`AutoDeriveTaskDependencies`](../passes/34-auto_derive_task_dependencies.md)，
保存在 `Call.attrs["compiler_manual_dep_edges"]`（独立的 key，允许出现在普通
call 上）。Codegen 会按这个顺序合并两组列表，并按 Var identity 去重后再发出
栈数组。

### TaskId 的来源

`SubmitToCallView` 合成的 `manual_dep_edges` 或普通 call 上的
`attrs["compiler_manual_dep_edges"]` 中的每个显式依赖条目都是 TaskId
`VarPtr`（`Scalar[TASK_ID]` 或 `Array[N, TASK_ID]`）。每个条目在 codegen 时通过
`manual_task_id_map_` 解析为以下来源之一：

| Producer 种类 | codegen 发出的 C++ |
| ------------- | ------------------ |
| `pl.submit` 的 producer TaskId（增广 Call 的 TaskId tuple 元素） | `PTO2TaskId <tid_name> = task_<n>_outs.task_id();`，其中 `task_<n>_outs` 是 submit 捕获的 `TaskOutputTensors` |
| `None` 种子（`deps=[None]` 条目中的字面量，或 TaskId iter_arg init） | `PTO2TaskId::invalid()` |
| 循环 carry iter_arg（穿行循环的 TaskId 配套） | for 循环中穿行的命名变量——标量或数组，见下 |
| 数组槽读取（`prev = tids[k]`——对 `Array[TASK_ID]` 的 `array.get_element`） | `PTO2TaskId <name> = <arr>[k];`——一个标量快照局部变量；dep 引用该局部变量而非重新读取槽位，因此之后的 `tids[k] = ...` 覆写不会改变它 |

`pl.submit` call 的 kernel-result tuple 元素与普通多输出 kernel call 一样，
直接 alias kernel 的 `Out`/`InOut` 参数。

每个 dep 数组填充条目都会被 `if (<task_id>.is_valid())` 包裹——包括直接来自
`pl.submit` 的 producer TaskId。`EmitManualDeps` 对所有标量（string 形式）
TaskId 统一加守卫，因为任何 TaskId 都可能持有 `PTO2TaskId::invalid()` 哨兵
（首轮迭代的 iter_arg carry、未写入的数组槽、数组槽读取，或 `None` 种子）。
array-carry iter_arg 则按元素逐槽生成带守卫的填充。

**词法作用域生命周期。** TaskId 绑定命名的是在其产生所在的 `PTO2_SCOPE { ... }`
块内声明的 C++ 局部变量（`PTO2TaskId tid = ...`）。每个 `PTO2_SCOPE`（AUTO 或
MANUAL）在进入时快照 `manual_task_id_map_` 与 `array_carry_vars_`、退出时恢复，
因此在某作用域内产生的绑定不会泄漏到外层作用域（否则其标识符会超出 C++ 作用域）。
循环 / 分支的 carry 在其 body 的 `PTO2_SCOPE` *之前*声明，因此能正确地在块结束后存活。

对 `MANUAL` 作用域的 `array_carry_vars_` 恢复有一个例外：在该作用域 *内部* 注册、
但其底层数组声明于 *外层* 作用域的 array carry 必须在恢复后存活。这就是将
`manual_scope` 产生的 TaskId loop-carry 进 `Array[TASK_ID]` 的场景（issue #1811）——
例如从外层 `pl.range` 循环的底层存储穿入的 `pl.parallel` array carry，每次迭代写入
一个槽位（`carry[n] = prod_tid`）。外层循环的 `YieldStmt` 在 `PTO2_SCOPE(MANUAL)`
块 *之后* 发出并引用该 carry；若将其抹除会丢失被 loop-carry 的 TaskId，并触发
*scalar yield to array carry* 的 `INTERNAL_CHECK`。因此退出时 codegen 会保留底层存储
为 enclosing-scope-valid（由不在该作用域 local 集合中的标识符命名）的 array carry，
仅回退作用域内局部的那些。

当后续 sibling scope 或父 scope 通过 `compiler_manual_dep_edges` 引用在更早的嵌套
scope 或 sibling scope 中创建的 producer TaskId 时，codegen 只提升这个 TaskId 绑定：
先在 producer 的 `PTO2_SCOPE` 前声明
`PTO2TaskId <name> = PTO2TaskId::invalid();`，在块内执行
`<name> = task_<n>_outs.task_id();`，再在外层 C++ 作用域里生成后续带 guard 的
`set_dependencies(...)` 条目。普通 scope-local TaskId 仍然保持局部。

**跨作用域张量与 `manual_scope`。** `manual_scope` 是一个*调度*区域，而非存储/取值
作用域：它所触及的张量会透明地流向 `PTO2_SCOPE(MANUAL) { ... }` 块*之后*的 task。
因此块后读取者所命名的任何标识符都不能是 manual scope 内的局部 C++ 标识符——否则它会
在右花括号处失效，读取者的 `add_input(...)` 将引用一个超出作用域的名字（`.cpp` 随即
无法通过 C++ 编译，issue #1697）。两种机制保证这一点，二者都以名字是否*在外层作用域
有效*（在块之前保留，或为已提升的块内缓冲——即非作用域局部）为判据：

- **输出重映射（remap）。** 一个由调用方分配、且别名为外层作用域源的 kernel/submit
  输出，*不*单独生成 `const Tensor&` 声明——其 emit 名被重映射到源，于是所有引用
  （块内与块后）都直接解析到外层名字。这正是 `tensor.assemble` 已采用的策略；由于该
  输出与其源是同一物理张量（原地写），共享名字恰好正确。phi/循环 carry 的重赋值被排除
  ——它重新绑定的是外层 `if`/循环所拥有的左值。

- **分配提升（allocation hoisting）。** 在块*内部*创建的缓冲（`pl.create_tensor`
  → `alloc_tensors`）只是存储预留、没有调度依赖，因此其声明被提升到外层作用域。codegen
  会缓冲每个 `PTO2_SCOPE(MANUAL)` 块体，并把提升后的 `alloc_tensors` 声明刷写到块头
  之前。该批次按构造即在外层作用域有效（形状引用了作用域局部值的 create 会被排除并保持
  原位）。

二者结合后，无论张量在块之前还是块内部创建、并在块后被读取，都会解析到外层作用域中唯一的
`const Tensor& buf = ...;`——块后 task 只需 `add_input(buf)`，不再产生任何按 SSA 版本
的别名。

### `pl.parallel` TaskId iter_arg 的 array carry

承载 TaskId 配套的 `pl.parallel(N)` ForStmt 被降级为**大小为 N 的数组 carry**，
而不是"last-write wins"的标量。pass 在 IR 中保留 iter_arg 为
`Scalar[TASK_ID]`；codegen 识别这一形态（Parallel + TaskId iter_arg）后：

1. 在 iter_arg 声明处分配定长数组：`PTO2TaskId arr[N];`，初始化时
   广播标量 init（init 为标量时）或按槽位拷贝（init 本身是数组时——
   例如 case1 内层 `pl.parallel` 的 init 来自外层 `pl.range` 的数组 carry）。
2. 每个 parallel iter 体内把新产生的 task id 写入一个槽位：
   `arr[(loop_var - start) / step] = <task_id>;`。当 `start == 0 && step == 1`
   （常见形式）时，槽位表达式被 peephole 化简为 `arr[loop_var]`。
3. 对每个 `deps_` 引用该 iter_arg 的下游消费者 `Submit`，把 N 个带守卫
   的槽位填入该 task 的 dep 栈数组，每个槽位一条：

   ```cpp
   for each k in [0..N):
       if (arr[k].is_valid()) { params_deps[params_deps_count++] = arr[k]; }
   ```

`pl.range`（Sequential）循环，若其 yield 是内层 `pl.parallel` array carry
的 rv，会继承同一个 N：它自身的 iter_arg 也成为大小 N 的数组 carry，
外层 yield 时按槽位拷贝。这种结构上的传播就是 case1（外层 SEQ × 内层 PARALLEL）
等拓扑中"多 iter fence 语义"的来源。

### Phase-fence dummy barrier

`DeriveCallDirections` 之后，`ExpandManualPhaseFence` pass 可能压缩有收益且稳定的完整数组
manual dependency：它把选中的 consumer `Submit` 从 `deps_=[tids]` 改写为
`deps_=[barrier_tid]`。该 pass 会插入一个带标记的 `system.task_dummy`
call；这个 dummy call 自己的 `manual_dep_edges` attr 仍然引用原始 TaskId 数组
（这是该 attr 唯一被允许的 op-call 载体——普通跨函数 `Call` 永不携带它）。Orchestration
codegen 会把带标记的 call 降低为 `rt_submit_dummy_task(...)`，随后对被改写的 consumer
继续使用普通标量 dependency lowering。

这会保留 phase boundary，同时避免重复 all-to-all fanout：

```text
tids[N] -> dummy barrier -> consumers[M]
```

如果形状、安全性或收益不够明确，则继续走原有直接 `Submit::deps_` lowering 路径。
尤其在 `manual_scope` 中，用户显式写出的 deps 是权威约束：如果 `pl.parallel`
body 读取 `deps=[tids]`，随后又更新 `tids[branch]`，这表示 same-carrier
dependency chain，而不是可在 loop 前压缩的 snapshot source。若用户需要
layer-parallel snapshot 语义，应写入单独的 `tids_next` carrier，并通过
loop-carried `init_values` / `pl.yield_` 在 parallel body 之后传回。这里不写成
普通的 `tids = tids_next`，因为当前 codegen 路径暂不支持 `ArrayType` 的普通
`AssignStmt`。

**codegen 入口检查的约束（带用户友好 CHECK 消息）：**

- `pl.parallel` 的 trip count 必须是 Python 字面量（编译期常量）。
  动态 trip count 在 codegen 时被拒绝，提示 "statically-known trip count"。

dep 栈数组按精确依赖数定长（对数组 carry 为 `N` 个槽），不再对 trip count
超过 16 设上限——runtime 原语 `Arg::set_dependencies(ptr, count)` 同样
没有上限。

### 示例

源 DSL（case1 形态）：

```python
with pl.manual_scope():
    prev_tid = None
    for phase in pl.range(N_PHASES):
        for branch in pl.parallel(N_BRANCHES):
            row = (phase * N_BRANCHES + branch) * TILE_M
            out, prev_tid = pl.submit(self.kernel_stripe, data, row, 1.0, out, deps=[prev_tid])
```

生成 C++（骨架）：

```cpp
PTO2_SCOPE(PTO2ScopeMode::MANUAL) {
    PTO2TaskId out__rv_v2__tid[N_BRANCHES];                    // 外层 rv = 数组
    for (int64_t i = 0; i < N_BRANCHES; ++i)
        out__rv_v2__tid[i] = PTO2TaskId::invalid();            // 广播 None 种子
    for (int64_t phase = 0; phase < N_PHASES; phase += 1) {
        PTO2TaskId out__rv_v4__tid[N_BRANCHES];                // 内层 rv = 数组
        for (int64_t i = 0; i < N_BRANCHES; ++i)
            out__rv_v4__tid[i] = out__rv_v2__tid[i];           // 按槽位拷贝
        for (int64_t branch = 0; branch < N_BRANCHES; branch += 1) {
            int64_t row = ...;
            Arg params_t0; /* ... */
            PTO2TaskId params_t0_deps[N_BRANCHES];             // 按数组 carry N 定长
            uint32_t params_t0_deps_count = 0;
            for (int64_t k = 0; k < N_BRANCHES; ++k) {         // 多依赖 fanout
                if (out__rv_v2__tid[k].is_valid())
                    params_t0_deps[params_t0_deps_count++] = out__rv_v2__tid[k];
            }
            params_t0.set_dependencies(params_t0_deps, params_t0_deps_count);
            TaskOutputTensors task_0_outs = rt_submit_aiv_task(0, params_t0);
            PTO2TaskId out__ssa_v5__tid = task_0_outs.task_id();
            out__rv_v4__tid[branch] = out__ssa_v5__tid;        // 槽位 yield
        }
        for (int64_t i = 0; i < N_BRANCHES; ++i)
            out__rv_v2__tid[i] = out__rv_v4__tid[i];           // 外层 yield (拷贝)
    }
}
```

phase `N+1` 中的每个 task 都会等待 phase `N` 的**全部** `N_BRANCHES` 个 task。

## 参见

- [PTO 代码生成](00-pto_codegen.md) — PTO 后端的 MLIR 生成
- [Pass 管理器](../passes/00-pass_manager.md) — 代码生成前应用的 IR 优化 Pass
- [Python syntax: 手工依赖原语](../language/00-python_syntax.md#手工依赖原语) — 表层语法及语义
