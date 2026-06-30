# OutlineIncoreScopes Pass

将 InCore 作用域提取为独立函数。

## 概述

该 Pass 将 `InCoreScopeStmt` 节点变换为独立的 `Function(InCore)` 定义，并将原作用域替换为对提取函数的调用。

**前置条件**：

- 输入 IR 必须为静态单赋值 (SSA) 形式（需先运行 ConvertToSSA）；该 Pass 保持（产生）SSAForm
- 仅处理 Opaque 函数（InCore 函数保持不变）

**使用时机**：在 ConvertToSSA 之后运行，当需要将 InCore 计算区域提取为独立的可调用函数时使用。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::OutlineIncoreScopes()` | `passes.outline_incore_scopes()` | 程序级 |

**工厂函数**：

```cpp
Pass OutlineIncoreScopes();
```

**Python 用法**：

```python
from pypto.pypto_core import passes

outline_pass = passes.outline_incore_scopes()
program_outlined = outline_pass(program)
```

## 算法

1. **扫描 InCore 作用域**：在 Opaque 函数中查找所有 `InCoreScopeStmt` 节点
2. **分析输入**：确定外部变量引用（在作用域外定义、在作用域内使用的变量）
3. **分析输出**：确定在作用域之后仍被使用的内部定义（在作用域内定义、在作用域外使用的变量）
4. **创建函数**：将作用域体提取为新的 `Function(scope_type=InCore)`，其中：
   - 参数 = 输入变量
   - 返回值 = 输出变量
   - 函数体 = 作用域体
5. **替换作用域**：将 `InCoreScopeStmt` 替换为：
   - 带有输入参数的提取函数调用
   - 每个输出变量对应一个 AssignStmt
6. **添加到程序**：将提取的函数添加到程序的函数列表中

**参数化显式返回**：只要某个 tensor 输出是经由参数回写
的，外提函数就返回自身的参数而非 SSA 结果变量——store 目标输出直接返回对应
参数，其余输出通过共享的 `return_lineage` 工具追踪。kernel 内部分配的输出
保留其 SSA 值。这使编排代码生成只需按指针同一性查表即可建立返回值到参数的
映射（`ReturnParamsExplicit` 不变量）。

**命名规则**：

- 默认：`{原函数名}_incore_{计数器}`（如 `main_incore_0`、`main_incore_1`）
- 用户自定义：当 `InCoreScopeStmt.name_hint` 非空时，直接使用该名称
  - `with pl.at(level=pl.Level.CORE_GROUP, name_hint="fused_add"):` → 函数名为 `fused_add`

**命名冲突**（`name_hint` 是“提示”而非唯一标识——所有外提函数共享同一个程序级
命名空间，因此冲突会自动消解）：

- **函数内冲突**——同一函数内两个 scope 共用一个 `name_hint` 时，追加数字后缀：
  `my_kernel`、`my_kernel_0`。
- **跨函数冲突**——两个*不同*函数外提出同名 `name_hint` 的 scope（常见于把复用的
  `@pl.jit.inline` helper 组合进同一个 host 程序）时，按来源函数对冲突方做命名空间
  化。先出现的函数保留原始提示名（稳定，与其单独编译时一致），后出现的加前缀：
  - `single_a` → `dup_scope`，`single_b` → `single_b_dup_scope`

  这样无需手动重命名共享 helper 的内部 `name_hint`，即可把可独立运行的子 kernel
  组合进一个 `@pl.jit.host` 程序。同一规则也适用于共用外提工具的兄弟 pass
  `OutlineHierarchyScopes` 与 `OutlineClusterScopes`。

## 示例

### 基本提取

**之前**：

```python
@pl.program
class Before:
    @pl.function  # Opaque function
    def main(self, x: Tensor[[64], FP32]) -> Tensor[[64], FP32]:
        y = x + 1

        with pl.at(level=pl.Level.CORE_GROUP):  # InCore scope
            tile = pl.load(y, [0], [64])
            tile_sq = pl.mul(tile, tile)
            result_tile = tile_sq + 1
            result = pl.store(result_tile, [0], x)

        z = result + 2
        return z
```

**之后**：

```python
@pl.program
class After:
    @pl.function  # Opaque function
    def main(self, x: Tensor[[64], FP32]) -> Tensor[[64], FP32]:
        y = x + 1

        # Scope replaced with call + assignments
        result = self.main_incore_0(y, x)  # Call outlined function

        z = result + 2
        return z

    @pl.function(scope_type=InCore)  # Outlined InCore function
    def main_incore_0(self, y: Tensor[[64], FP32], x: Tensor[[64], FP32]) -> Tensor[[64], FP32]:
        # Scope body moved here
        tile = pl.load(y, [0], [64])
        tile_sq = pl.mul(tile, tile)
        result_tile = tile_sq + 1
        result = pl.store(result_tile, [0], x)
        return x  # store target: returns the param, not `result`
```

### 多输出

**之前**：

```python
with pl.at(level=pl.Level.CORE_GROUP):
    a_tile = pl.load(a, [0], [64])
    b_tile = pl.load(b, [0], [64])
    c_tile = pl.add(a_tile, b_tile)
    out_a = pl.store(c_tile, [0], out)
    out_b = pl.mul(c_tile, 2.0)
# Both out_a and out_b used after scope
x = out_a + out_b
```

**之后**：

```python
out_a, out_b = self.main_incore_0(a, b, out)  # Multiple outputs
x = out_a + out_b

# Outlined function:
def main_incore_0(self, a, b, out):
    a_tile = pl.load(a, [0], [64])
    b_tile = pl.load(b, [0], [64])
    c_tile = pl.add(a_tile, b_tile)
    out_a = pl.store(c_tile, [0], out)
    out_b = pl.mul(c_tile, 2.0)
    return (out, out_b)  # out_a → param `out`; out_b is kernel-local, kept as-is
```

## 实现

**头文件**：`include/pypto/ir/transforms/passes.h`

```cpp
Pass OutlineIncoreScopes();
```

**实现文件**：`src/ir/transforms/outline_incore_scopes.cpp`

- 使用 SSA 分析确定输入/输出
- 创建带有 InCore 作用域类型的新 Function 节点
- 将 InCoreScopeStmt 替换为 Call + AssignStmt
- 管理函数命名和计数器

**Python 绑定**：`python/bindings/modules/passes.cpp`

```cpp
passes.def("outline_incore_scopes", &pass::OutlineIncoreScopes, "Outline InCore scopes");
```

**测试**：`tests/ut/ir/transforms/test_outline_incore_scopes.py`

- 测试基本作用域提取
- 测试输入/输出分析
- 测试同一函数中的多个作用域
- 测试嵌套作用域
- 测试 SSA 保持

## 前置条件

**需要 SSA 形式**：该 Pass 依赖 SSA 属性 (Property)：

- 单赋值确保清晰的输入/输出分析
- 无变量遮蔽简化了作用域分析
- 控制流中的 YieldStmt 被正确处理

如果 IR 不是 SSA 形式，**请先运行 ConvertToSSA**。
