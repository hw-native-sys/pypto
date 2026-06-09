# ConvertToSSA Pass

将非 SSA IR 转换为静态单赋值 (SSA) 形式，包含变量重命名、phi 节点和 iter_args。

## 概述

此 Pass 将具有对同一变量多次赋值的中间表示 (IR) 转换为 SSA 形式，使每个变量恰好被赋值一次。它处理以下情况：

- **直线代码**：对同一变量的多次赋值
- **If 语句 (Statement)**：在一个或两个分支中修改的变量
- **For 循环**：在循环体内修改的变量
- **混合 SSA/非 SSA**：保留现有的 SSA 结构，同时转换非 SSA 部分
- **幂等性**：在已 SSA 化的 IR 上再次运行本 Pass 仍产生 SSA 等价的 IR，且不会引入 use-before-def 违例 —— 这对嵌入在动态 Tile/Tensor 类型注解中的 Var 引用（shape、valid_shape、stride）同样成立。所有从已有 Var 中读取类型的构造点都会通过 `SubstType` 转译类型，使得被重命名的 Var 引用始终保持一致。

**需要**：TypeChecked 属性 (Property)（需在运行本 Pass 之前已建立，可通过属性验证/`VerificationInstrument` 等机制检查）。

**使用时机**：在任何需要 SSA 形式的优化或分析之前运行此 Pass（如 OutlineIncoreScopes、内存优化 Pass）。

**Pipeline 位置**：默认的张量优化策略调用 `ConvertToSSA` 两次：

1. **第 4 个 Pass** — 对用户编写的 DSL 进行初始 SSA 转换（在所有 outline 之前）。
2. **紧接 `OutlineClusterScopes`** （入口名：`ConvertToSSA_post_outline`） — 对 outlined function 重新跑 SSA。Outlining 将 `tile.store` 的结果从 `EvalStmt` 形式提升为循环内的 `AssignStmt`；第二次 Pass 通过 **逃逸变量提升**（见算法步骤 4）将这些定义通过 `IterArg` / `YieldStmt` 串到函数作用域，使 outlined function 的 `ReturnStmt` 引用合法的 SSA Var。此机制依赖本 Pass 的 **幂等性**（见上文「混合 SSA/非 SSA」/「幂等性」条款）：第一次 Pass 的产物在第二次 Pass 中保持不变。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::ConvertToSSA()` | `passes.convert_to_ssa()` | 函数级 |

**工厂函数**：

```cpp
Pass ConvertToSSA();
```

**Python 用法**：

```python
from pypto.pypto_core import passes

ssa_pass = passes.convert_to_ssa()
program_ssa = ssa_pass(program)
```

## 算法

1. **变量重命名**：为每次赋值添加 SSA 后缀（x -> x__ssa_v0, x__ssa_v1, x__ssa_v2）
2. **If 的 Phi 节点**：为在 if 分支中修改的变量添加 phi 节点（return_vars + YieldStmt），包括在两个分支中独立定义的变量
3. **循环的 Iter_args**：将循环中修改的变量转换为 iter_args + return_vars 模式，带有 YieldStmt
4. **逃逸变量提升**：通过前向使用分析，将首次在循环体内定义但在循环后使用的变量提升为 iter_args + return_vars
5. **作用域跟踪**：跨嵌套作用域跟踪变量定义
6. **跨作用域的 escaping 防护**：除 `RuntimeScopeStmt` 外的 `ScopeStmt` 子类（`HierarchyScopeStmt` / `InCoreScopeStmt` / `AutoInCoreScopeStmt` / `ClusterScopeStmt` / `SpmdScopeStmt`），`ConvertScope` 在进入 body 前把 `future_needs_` 裁剪到仅包含 scope 之前已经存在的变量。这样可以阻止 scope body 内的嵌套循环把 **scope-local 新变量**根据 scope 之后的引用错误地提升成 `init_values=(foo__FREE_VAR,)`。scope body 内首次定义、然后在 scope 之外引用的变量仍然能正常替换（`InterchangeChunkLoops` 等后续 pass 依赖这一行为）。`RuntimeScopeStmt`（`pl.scope()`）只是 codegen 包装节点，保持完全透传。
7. **保留**：保持现有 SSA 构造不变

**关键变换**：

- `x = 1; x = 2` -> `x__ssa_v0 = 1; x__ssa_v1 = 2`
- 具有分歧赋值的 If -> 在两个分支中添加 return_vars 和 YieldStmt
- 具有循环携带依赖的 For 循环 -> 添加 iter_args/return_vars/YieldStmt
- 循环逃逸变量 -> 提升为 iter_args，以匹配的 Out 参数作为初始值

## 示例

### 直线代码

**变换前**：

```python
x = 1
y = x + 2
x = 3  # Multiple assignment
z = x + 4
```

**变换后**：

```python
x__ssa_v0 = 1
y = x__ssa_v0 + 2
x__ssa_v1 = 3
z = x__ssa_v1 + 4
```

### If 语句

**变换前**：

```python
x = 1
if condition:
    x = 2  # Modified in then branch
z = x + 3  # Uses x after if
```

**变换后**：

```python
x__ssa_v0 = 1
if condition:
    x__ssa_v1 = 2
    yield (x__ssa_v1,)  # Yield modified variable
else:
    yield (x__ssa_v0,)  # Yield original variable
return_vars = (x__phi_v2,)  # Phi node
z = x__phi_v2 + 3
```

### For 循环

**变换前**：

```python
sum = 0
for i in range(10):
    sum = sum + i  # Loop-carried dependency
```

**变换后**：

```python
sum__ssa_v0 = 0
for i in range(10):
    iter_args = (sum__iter_v1,)
    init_values = (sum__ssa_v0,)
    # Loop body
    sum__ssa_v2 = sum__iter_v1 + i
    yield (sum__ssa_v2,)
return_vars = (sum__rv_v3,)
```

### 循环逃逸变量

**变换前**（变量 `out` 首次在循环内定义，在循环后使用）：

```python
for i in range(4):
    tile_c = tile.add(tile_a, tile_b)
    out = tile.store(tile_c, [offset, 0], c)  # first assignment inside loop
return out  # used after loop
```

**变换后**（提升为 iter_arg + return_var）：

```python
for i in range(4):
    iter_args = (out__iter_v0,)
    init_values = (c__ssa_v0,)  # Out parameter as initial value
    tile_c__ssa_v0 = tile.add(tile_a__ssa_v0, tile_b__ssa_v0)
    out__ssa_v1 = tile.store(tile_c__ssa_v0, [offset__ssa_v0, 0], out__iter_v0)
    yield (out__ssa_v1,)
return_vars = (out__rv_v2,)
return out__rv_v2
```

## 实现

**头文件**：`include/pypto/ir/transforms/passes.h`

```cpp
Pass ConvertToSSA();
```

**实现文件**：`src/ir/transforms/convert_to_ssa_pass.cpp`

- 使用手动语句分派和显式作用域管理
- 维护变量重命名的版本映射
- 插入 YieldStmt 并管理 return_vars/iter_args

**Python 绑定**：`python/bindings/modules/passes.cpp`

```cpp
passes.def("convert_to_ssa", &pass::ConvertToSSA, "Convert to SSA form");
```

**测试**：`tests/ut/ir/transforms/test_convert_to_ssa_pass.py`

- 测试直线代码重命名
- 测试 if 语句 phi 节点
- 测试 for 循环 iter_args
- 测试嵌套作用域
- 测试混合 SSA/非 SSA
