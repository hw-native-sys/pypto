# OutlineGraphScopes Pass

将 `pl.graph` 区域提取为 `FunctionType::Graph` 函数，使作用域形式与
`@pl.jit.graph` 汇聚为同一种表示。

## 概述

该 Pass 将 `GraphScopeStmt` 节点（由 `with pl.graph("name"):` 生成）变换为
`Function(Graph)` 定义，并将原作用域替换为对提取函数的调用。

它存在的全部意义就是**汇聚 (convergence)**。该 Pass 之后，用户就地标记的区域与用户用
`@pl.jit.graph` 装饰的函数具有相同的形态 —— 一个带编排元数据的 `Function(Graph)` 加一个
`Call` —— 因此 `LegalizeGraphBoundary`、Graph 验证器与编排代码生成都无需知道用户写的是
哪一种形式。两种形式并存是出于书写习惯而非语义差异：装饰器适合本身已是独立函数的
layer；作用域适合用户不愿拆分出去的、较大编排函数体中的一段。

唯一**不会**对齐的是形参**顺序**：outliner 按捕获顺序追加形参，而装饰器形式用的是用户
声明的签名。两者的边界互为排列，而下游不依赖这一点 —— 所有消费方都通过
`param_directions_` 而非位置来读取形参。

**前置条件**：

- 输入 IR 必须为静态单赋值 (SSA) 形式（需先运行 `ConvertToSSA`）；该 Pass
  保持（产生）SSAForm
- 必须已运行 `InlineFunctions`（`InlineFunctionsEliminated`）。解析器有意允许
  `pl.graph` 出现在 `Inline` 函数体中，前提是该函数体会在本 Pass 之前被内联进
  它的编排调用者；若流水线把两者的顺序颠倒，本 Pass 会直接报错，而不是静默地
  让该区域不被提取
- 处理 `Opaque` 与编排类 (`Orchestration` / `Graph`) 函数；设备侧 kernel
  类型不携带可提取的 Graph 区域
- 在 `OutlineIncoreScopes` **紧邻之前**运行

**为什么是这个位置**。被标记区域内部的 InCore 作用域必须在该区域*成为函数之后*
才被提取，这样 `OutlineIncoreScopes` 看到的输入与它处理手写
`@pl.jit.graph` 函数时看到的输入完全一致 —— 一个携带 `pl.at` 作用域的编排函数体
—— 从而产生一致的输出。反过来做、或者放到流水线更靠后的位置，都会让
`GraphScopeStmt` 存活穿过数十个 Pass，而每个 Pass 都要学会处理它；RFC (#2399)
明确指出这正是作用域形态载体的主要代价，而提前提取正是规避该代价的手段。

**父函数类型保持不变**。与 `OutlineIncoreScopes` 不同，该 Pass **不会**把
`Opaque` 父函数提升为 `Orchestration`。携带一个 Graph 区域并不能说明外层函数
是什么；而一旦提升，任何恰好包含 Graph 区域的 Opaque 辅助函数都可能被选作编译
入口 —— 后端取的是**第一个** Orchestration 函数，于是一次无关的改动就可能静默
改变程序编译出的入口。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::OutlineGraphScopes()` | `passes.outline_graph_scopes()` | 程序级 |

**工厂函数**：

```cpp
Pass OutlineGraphScopes();
```

**Python 用法**：

```python
from pypto.pypto_core import passes

outline_pass = passes.outline_graph_scopes()
program_outlined = outline_pass(program)
```

## 算法

1. **对每个函数做一次扫描**，查找 Graph 区域，并拒绝 `GraphScopeStmt` 内再出现
   `GraphScopeStmt`（编译期错误，见下文"嵌套"）。
2. **不含 Graph 区域的函数原样输出**。绝大多数程序根本不含 `pl.graph` 区域，
   这类程序只需付出一次线性扫描。

   这条快速路径只是省下开销，并不是复杂度的保证。`ScopeOutliner` 计算某个位置
   的 used-after 集合时，会遍历其后每条语句的整棵子树；若在所有位置都这么做，
   在含 M 条语句的块上就是 O(M²)。实际上它只在「本身是、或内部含有」待提取
   scope 的位置才计算，因为只有这些位置会读取该结果。因此散布在普通语句之间的
   区域是线性的：对**真正使用** `pl.graph` 的程序，本 Pass 同样落在
   `.claude/rules/pass-complexity.md` 规定的 O(N log N) 界内，而不是仅仅对不使用
   它的程序成立。
3. **对本 Pass 不提取的函数中残留的 Graph 区域直接报错**。跳过它会让
   `GraphScopeStmt` 原地留存，而本 Pass 仍然宣称产生了 `GraphOutlined`；由于
   `required` 仅在开启校验时才被检查，这个错误属性在关闭校验时会一路流到代码
   生成而无人察觉。
4. **提取**：以 `ScopeKind::Graph` / `FunctionType::Graph` / 后缀 `_graph_`
   运行共享的 `outline_utils::ScopeOutliner`。被捕获的值成为形参，区域之后仍被
   使用的值成为返回值，作用域本身被替换为一个 Call。

提取出的函数不需要额外补 level/role：`Function` 的构造函数会为任何编排类类型推导出
`Level::CHIP` + `Role::Orchestrator`，因此这里产生的 Graph 天然带着与解析器给
`@pl.jit.graph` 函数相同的元数据。

## 命名与 graph key

区域名是**必填**的 —— `pl.graph()` 不带参数是解析错误，`IRBuilder::EndScope`
也会拒绝该 kind 的空 `name_hint`。其他所有作用域 kind 都把 `name_hint` 当作可选
提示，因此这一点值得明说：

`name` 成为提取函数的名字；代码生成据此推导出 C++ 符号；运行时以该符号的地址
为键缓存 `GraphDefinition`。若名字自动生成，那么在文件靠前处新增一个无关区域，
就会改变已录制图的身份。名字由用户掌握，正是因为这份稳定性由用户掌握。

**同名区域是消歧而非合并**。两个请求同一名字的区域会通过程序级保留名集合
(#1711) 得到互不相同的后缀名。这是安全的方向：共用一个名字会让两套不同的拓扑
共享一份 Definition，第二次调用将回放第一次录制的图。

## 嵌套

Graph 区域内再嵌套 Graph 区域，会在两个层面被拒绝：

| 层面 | 捕获对象 | 诊断质量 |
| ---- | -------- | -------- |
| 解析器 (`_parse_graph_scope`) | 文本上嵌套的 `with pl.graph(...)` | 定位到出错的源码行 |
| Pass (`NestedGraphScopeChecker`) | 任何来源构造的嵌套 `GraphScopeStmt` | 指出两个区域名与所在函数 |

运行时把录制过程中出现的 `graph_begin` 视为不支持，并将整个区域回退为普通
submit。该回退是**静默的** —— 程序依然算出正确结果，只是收益归零 —— 因此编译期
报错是用户唯一能发现它的途径。Pass 层的检查是不变量；解析器检查存在的意义，是
在源码层嵌套关系尚可见时给出更好的诊断。

解析器还会拒绝嵌套在 `pl.at`、`pl.cluster`、`pl.spmd` 内部的 Graph 区域：这些
构造会变成单个设备任务，而 Graph 区域录制的是任务的*拓扑*，因此它必须包住这些
分发，而不是待在其中之一里面。

## 示例

**变换前**（编排函数体中的 `with pl.graph(...)`）：

```python
@pl.jit
def entry(w: pl.Tensor, acc: pl.InOut[pl.Tensor]):
    for i in pl.range(LAYERS):
        with pl.graph("accumulate_band"):
            base = i * ROWS
            with pl.at(level=pl.Level.CORE_GROUP):
                band = pl.load(w, [base, 0], [ROWS, COLS])
                cur = pl.load(acc, [0, 0], [ROWS, COLS])
                pl.store(pl.add(cur, band), [0, 0], acc)
    return acc
```

**变换后**（区域被提升；那个 Call 就是运行时录制一次、其后回放的对象）：

```python
@pl.function(type=pl.FunctionType.Graph, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
def accumulate_band(self, i, w, acc, base):
    with pl.at(level=pl.Level.CORE_GROUP):
        band = pl.load(w, [base, 0], [ROWS, COLS])
        cur = pl.load(acc, [0, 0], [ROWS, COLS])
        pl.store(pl.add(cur, band), [0, 0], acc)
    return acc


@pl.function(type=pl.FunctionType.Orchestration)
def entry(self, w, acc):
    for i in pl.range(LAYERS):
        acc = self.accumulate_band(i, w, acc, i * ROWS)
    return acc
```

`base` 出现在调用点是 `LegalizeGraphBoundary` 的 Step A，而非该 Pass 所为：区域
内部*派生*出来的标量没有实参槽位，运行时会把首次调用的值固化进录制结果。该 Pass
只负责提升区域，边界契约由下游强制执行。

## 属性

| 属性 | 作用 |
| ---- | ---- |
| `SSAForm` | 前置且产生 |
| `InlineFunctionsEliminated` | 前置 |
| `GraphOutlined` | 产生 |

`GraphOutlined` 断言**没有任何**函数还保留 `GraphScopeStmt` —— 包括 Graph 函数
本身，因为嵌套区域被直接拒绝。

## 相关文档

- [09-outline_incore_scopes.md](09-outline_incore_scopes.md) —— 紧接着运行；在提取
  Graph 函数体内的 InCore 作用域时保持其 `Graph` 类型
- [46-legalize_graph_boundary.md](47-legalize_graph_boundary.md) —— 对提取出的函数
  强制执行运行时边界契约
- [00-pass_manager.md](00-pass_manager.md) —— 属性表中的 `GraphOutlined` IRProperty
