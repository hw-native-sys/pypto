# OutlineIncoreScopes Pass

将 InCore 作用域提取为独立函数。

## 概述

该 Pass 将 `InCoreScopeStmt` 节点变换为独立的 `Function(InCore)` 定义，并将原作用域替换为对提取函数的调用。

**前置条件**：

- 输入 IR 必须为静态单赋值 (SSA) 形式（需先运行 ConvertToSSA）；该 Pass 保持（产生）SSAForm
- 处理 Opaque 与 Orchestration 函数（InCore 函数保持不变）。当解析器将
  `for i in pl.spmd(...)` 这类高层构造展开时，Orchestration 函数同样会携带
  InCore 作用域；至少提取出一个作用域的 Opaque 父函数会被提升为 Orchestration

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

1. **扫描 InCore 作用域**：在 Opaque 与 Orchestration 函数中查找所有
   `InCoreScopeStmt` 节点
2. **分析输入**：收集作用域的 *live-in*（活跃入口）集合——作用域体在（重新）定义
   某变量之前就读取它，说明该变量的入口值来自调用方
3. **分析输出**：确定在作用域之后仍被使用的内部定义（在作用域内定义、在作用域外使用的变量）
4. **创建函数**：将作用域体提取为新的 `Function(scope_type=InCore)`，其中：
   - 参数 = 输入变量
   - 返回值 = 输出变量
   - 函数体 = 作用域体
5. **替换作用域**：将 `InCoreScopeStmt` 替换为：
   - 带有输入参数的提取函数调用
   - 每个输出变量对应一个 AssignStmt
6. **添加到程序**：将提取的函数添加到程序的函数列表中
7. **提升父函数**：至少提取出一个作用域的 Opaque 父函数将变为 `Orchestration`——
   并在此之前先折叠其参数动态维度读取（见下）

**参数动态维度读取在提升时折叠**：tensor 声明的 extent *就是*它的运行期
extent，因此对以 `pl.dynamic` 符号为某一轴的参数调用 `pl.tensor.dim(a, 0)`，会
为同一个量再造出**第二个** IR 名字，由该副本构造的 shape 也就不再与由符号构造
的 shape 结构相等。DSL 解析器会把该读取折叠到符号上
（`ASTParser._fold_tensor_dim`），但仅限 Orchestration 函数体——只有在那里
Orchestration codegen 才会从参数的 task-arg 描述符定义该符号，折叠才是可靠的。
写成 `Opaque` 的函数体会保留该读取，因此本 pass 在提升该函数的那一刻、**在提取
之前**完成折叠：

```python
# 写法为 Opaque 的父函数                      # 提升之后
m = pl.tensor.dim(a, 0)                    # （绑定语句已折叠消失）
with pl.spmd(m // 16):                     with pl.spmd(M_DYN // 16):
    ...                                        ...
```

在提取器运行前折叠，意味着被提升的函数体进入提取器时，与解析器交给一个本就是
Orchestration 的函数的形态完全一致，两条路径产出相同的 IR。若不折叠，本 pass 产
出的 IR 将无法再解析回自身（打印出的 `tensor.dim` 绑定在重新解析时消失），从而
破坏 print→parse 往返验证。解析器不会折叠的读取同样保持原样：常量 extent、运行期
轴，或并非该签名所声明的符号。

**使用 live-in 而非 `uses \ defs`**：输入集合按流敏感方式计算
（`UpwardExposedUseCollector`）。对于「先读取、再以同名重新绑定」的被捕获
tensor，简单的集合差是错误的——这正是 `ConvertToSSA` 拆分之前，解析器为
`pl.Out` 参数生成的形态：

```python
with pl.at(level=pl.Level.CORE_GROUP):
    c = pl.store(t, [0, 0], c)   # 同一个 Var：既作为 store 目标被读取，又被重新绑定
```

`c` 同时出现在 `var_uses` 与 `var_defs` 中，集合差会把它从参数表中剔除，导致
外提函数体内的读取变成自由变量。将其视为 live-in 后，它成为 `InOut` 参数，而
`tile.store` 的结果绑定到一个独立的 Var（`c__store`），因此外提函数体永远不会
重新绑定自身的参数：

```python
def main_incore_0(a: Tensor[[128, 128], FP32], c: InOut[Tensor[[128, 128], FP32]]):
    c__store = pl.tile.store(..., c)
    return c                       # 返回参数——store 就地经由它写回
```

在 SSA 输入（即本 pass 声明的 `IRProperty::SSAForm` 前置条件）下，live-in 与
`uses \ defs` 完全一致，因此该行为差异只出现在前置条件未满足就进入本 pass 的
IR 上。若被捕获变量是被 `tile.store` 之外的方式重新绑定，则无法在不做真正 SSA
构造的前提下表达，此时会抛出内部错误并提示先运行 `ConvertToSSA`。

「绑定到独立 Var」是 InCore / Cluster / Spmd 的行为。Hierarchy 作用域会完全跳过
store 目标导出（该缓冲区已经通过 InOut 参数对调用方可见），因此其函数体保留原有的
重新绑定——被捕获变量仍然会成为参数，而这正是此前出问题的部分。

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
    @pl.function(type=pl.FunctionType.Orchestration)  # promoted from Opaque
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

**互斥的 AIV 拆分机制**：函数级 AUTO split（`optimizations=[pl.split(mode)]`，
承载于作用域自身的 `split_`）与显式 `pl.split_aiv` 区域（`SplitAivScopeStmt`）不能在同一
作用域共存（outliner 会把单个区域的模式桥接为函数级代表 `split`，从而与用户的
`pl.split` 静默冲突）。幸存机制如何下降见
[`LowerAutoVectorSplit`](20-lower_auto_vector_split.md)。

**任何** `pl.split(...)` 都会被拒绝，包括 `SplitMode.NONE`（RFC #1820）。NONE 本身不
携带拆分，但把它写在同时持有区域的作用域上，读起来仍像"在一个作用域里混用了自动与手动
拆分"。此前之所以对它豁免，只是因为跨核槽位数除了 `pl.split(..., slot_num=N)` 之外没有
别的承载方式；现在它有了自己的条目——`optimizations=[pl.cross_core_slot(slot_num=N)]`，
与拆分正交，可自由地与区域共存。

**拒绝发生在哪一层**：`InCoreScopeStmt::split_` 对"不切分"只有一种编码
（`SplitMode::None`），因此字面写出的 `pl.split(pl.SplitMode.NONE)` 对本 Pass 不可见
——它与完全不写 `pl.split` 无法区分。于是该拒绝由 **parser** 负责：只有它能看到用户写下的
字面量，且它会拒绝所有模式（含 NONE）。本 Pass 保留 `split_ != SplitMode::None` 的检查，
作为未经 parser 的 IR（反序列化的 `.pto`、以编程方式构造的作用域）的兜底。

三种标注由此语义分明：

| 标注 | 含义 |
| ---- | ---- |
| `optimizations=[pl.split(MODE)]` | AUTO 拆分——由编译器划分向量计算 |
| `for aiv_id in pl.split_aiv(2, mode=...)` | 手动拆分——由作者按区域划分 |
| `optimizations=[pl.cross_core_slot(slot_num=N)]` | 都不是——仅决定跨核 pipe 的大小 |

**函数级 `split` 属性对"不切分"只有一种编码：不存在该键。** 当被外提的函数体中含有
`pl.split_aiv` 区域时，本 Pass 仅在所有区域模式一致 **且** 该模式是真实切分时，才把它
提升为函数级属性：

| 函数体中的区域 | 外提函数上标记的 attrs |
| -------------- | ---------------------- |
| 全部 `mode=UP_DOWN`（或全部 `LEFT_RIGHT`） | `{"split_aiv": True, "split": pl.SplitMode.UP_DOWN}` |
| 全部 `mode=NONE` | `{"split_aiv": True}`——不带 `split` 键 |
| 模式不一致 | `{"split_aiv": True}`——没有代表性模式 |

`Function::GetSplitMode()` 把存储的 `0` 与缺失的键同样映射为 `nullopt`，因此
`split=SplitMode.NONE` 这一项对所有消费方都不可见；而 parser 会在回读时丢弃它，导致
print → parse 有损（`Kwargs size mismatch`）。权威的逐区域模式始终承载于
`SplitAivScopeStmt::split_`，由 [`LowerAutoVectorSplit`](20-lower_auto_vector_split.md)
消费。printer 以同一规则兜底：省略取值为 `SplitMode.NONE` 的 `split` 属性，使绕过本 Pass
的 IR（此前写出的 `.pto`、以编程方式构造的 `Function`）依然以规范、可重新解析的形式打印。
