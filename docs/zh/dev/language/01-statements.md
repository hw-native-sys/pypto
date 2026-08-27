# 语句与控制流

parser 接受的语句形式，以及 `pl.yield_` 背后的 SSA phi 节点语义。类型系统与表达式
文法参见 [Python IR 语法规范](00-python_syntax.md)。

## 语句 (Statement)

### 赋值

```python
x: pl.INT64 = expr
y: pl.Tensor[[4], pl.FP32] = tensor_op(a)
```

### If 语句 (SSA 风格)

```python
# If with both branches
if condition:
    y1 = pl.yield_(value1)
else:
    y1 = pl.yield_(value2)

# Multiple return values (no inline type annotations)
if condition:
    y1, y2 = pl.yield_(value1, value2)
else:
    y1, y2 = pl.yield_(value3, value4)
```

**要点:**

- `pl.yield_()` 赋值给 SSA phi 节点
- yield 中定义的变量在 if 之后可访问
- 两个分支必须 yield 相同的变量
- 元组解包时不能使用内联类型标注

### For 循环 (带 iter_args 的 SSA 风格)

```python
# 简单循环 (1-3 个位置参数，类似 Python 的 range())
for i in pl.range(stop):                    # start=0, step=1
for i in pl.range(start, stop):             # step=1
for i in pl.range(start, stop, step):       # 完整形式

# 带 iter_args 的循环 (循环携带值)
sum_init: pl.INT64 = 0
for i, (sum,) in pl.range(n, init_values=(sum_init,)):
    sum = pl.yield_(sum + i)
sum_final = sum

# 并行 for 循环 (同样支持 1-3 个参数)
for i in pl.parallel(stop):
for i in pl.parallel(start, stop, step):
    body_statements
```

**要点:** 循环携带值使用 `pl.range()` 或 `pl.parallel()` 的 `init_values`, 元组解包 `(sum,)` 声明 iter_args, `pl.yield_()` 为下一次迭代更新值, 循环结束后 iter_args 包含最终值。`pl.parallel()` 生成 `ForKind.Parallel` 循环, `pl.range()` 生成 `ForKind.Sequential` (默认)。

### While 循环 (带 iter_args 的 SSA 风格)

```python
# 自然 while：条件作为 while 头部表达式
i: pl.Scalar[pl.INT64] = 0
while i < n:
    i = i + 1

# 带 init_values 的 SSA 形式：头部元组 = iter_args，第一条语句是 pl.cond()。
# yield-LHS 名字成为循环外的绑定名（与 pl.range 一致）。
x_init: pl.Scalar[pl.INT64] = 0
for (x,) in pl.while_(init_values=(x_init,)):
    pl.cond(x < n)
    x_next = pl.yield_(x + 1)
# 此处 `x_next` 已由 yield-LHS 绑定；`x` 仅在循环 body 内可见。

# Pre-SSA：body 中完全没有 pl.yield_，由 ConvertToSSA 后续补出。
for (x,) in pl.while_(init_values=(x_init,)):
    pl.cond(x < n)
    x = x + 1

# ❌ init_values 非空时不允许裸 pl.yield_(...)，parser 直接报错：
#    for (x,) in pl.while_(init_values=(x_init,)):
#        pl.cond(x < n)
#        pl.yield_(x + 1)             # ParserSyntaxError: requires assignment-form pl.yield_
```

**要点:** `pl.while_(init_values=(...,))` 复用 `for ... in` 头部，用于 SSA 风格循环；body 的第一条语句必须是 `pl.cond(<bool>)`。循环外的绑定名来自 **yield-LHS**（上面的 `x_next`），而不是头部元组——头部元组中的名字只在循环 body 内可见。这一约定与 `pl.range` **保持一致**：当 `init_values` 非空且 body 中确实出现 `pl.yield_(...)` 调用时，必须使用 assignment 形式。Pre-SSA 形式的循环（body 中完全没有 yield，如最后一种写法）仍然合法。

### 作用域上下文管理器 (Scope Context Managers)

| 形式 | Scope 类型 | 说明 |
| ---- | ---------- | ---- |
| `pl.at(level=pl.Level.CORE_GROUP)` | `InCore` | CORE_GROUP 级固定边界 outline |
| `pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.split(MODE)])` | `InCore` | InCore + 跨核 split 提示 |
| `pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.cross_core_slot(slot_num=N)])` | `InCore` | InCore + 跨核 pipe 槽位数 |
| `pl.at(level=pl.Level.HOST)`（或任意非 `CORE_GROUP` 级别） | `Hierarchy` | 分布式层级作用域 |
| `pl.cluster()` | `Cluster` | AIC+AIV 协同调度组 |
| `with pl.spmd(N)` / `for i in pl.spmd(N)` | `Spmd`（for-form 内嵌 `InCore`） | SPMD 多 block 派发——见 [pl.spmd](#plspmd-多-block-派发) |
| `pl.spmd(N, optimizations=[pl.split(MODE)])` | `Spmd(InCore(split=MODE))` | split 提示作用于内层 InCore（两种形式均适用） |
| `pl.spmd(N, optimizations=[pl.cross_core_slot(slot_num=N)])` | `Spmd(InCore(slot_num=N))` | 槽位数作用于内层 InCore（两种形式均适用），可与 `pl.split(MODE)` 组合 |
| `pl.scope(mode=pl.ScopeMode.MANUAL)` / `pl.manual_scope()` | `Runtime(manual=true)` | orchestrator 的 MANUAL scope——由用户管理任务排序。两种 `auto_scope` 模式下都可用（它是依赖语义选择）。见[手工依赖原语](02-manual_dependencies.md#手工依赖原语) |
| `pl.scope()` | `Runtime(manual=false)` | orchestrator 的 AUTO scope（`SIMPLER_SCOPE()`）。手写它需要 `@pl.function(auto_scope=False)`（默认 `auto_scope=True` 下由编译器决定 AUTO 放置）。见 [MaterializeRuntimeScopes](../passes/46-materialize_runtime_scopes.md) |

#### `pl.spmd` 多 block 派发

`pl.spmd(N)` 把一个 kernel 派发到 `N` 个 block。形式：

- `with pl.spmd(N): ...` —— body **既可以**是调用已声明 InCore kernel 的*派发型* body（`SpmdScopeStmt(body=<stmts>)`，无内层 InCore 包裹），**也可以**是一段*内联*块，自动外包成一段隐式 InCore 区域（与 for-form 相同，只是不自动绑定索引）。区分依据是语义而非语句数量：body 若读取 `pl.tile.get_block_idx()`，即为内联 body 并被包裹；否则即为派发型 body，无论包含多少条语句都不加包裹。若 body 既不读取索引、也不派发 `self.<kernel>(...)` 调用，则会被拒绝。不捕获 producer TaskId。
  - 当 body 唯一的语句是显式的 `with pl.at(<CORE_GROUP level>, ...):` 时，该嵌套 scope 本身*就是* InCore 载体：它会被当作普通嵌套 scope 解析，而不会被二次包裹（`level` 可用位置或关键字形式，也可带 `as tid` / `name_hint=`）。printer 正是以这种形式输出 `Spmd(InCore(...))`，因此这也是该 IR 能够 round-trip 的原因。当 body 已提供载体时，`optimizations=` 必须写在该 `pl.at(...)` 上；写在 `pl.spmd(...)` 行会被拒绝，无论载体自身是否也带有该项。
  - 派发型 body 只能启动**一个** kernel。它经由 `FindFirstInnerCall` 下降，而后者在第一个调用处即停止，因此第二个派发不会被启动而是被静默丢弃；parser 会直接拒绝这种写法。提升出的临时变量与 tuple 投影不算派发，不计入数量。
- `for i in pl.spmd(N): ...` —— 循环变量绑定到每个 block 的索引（`pl.tile.get_block_idx()`）；body 自动外包成一段隐式 InCore 区域。
- `with pl.spmd(N, deps=[...]) as tid: ...` —— **捕获形式**：与 `with pl.at(...) as tid:` 对称。body 形态与上面的普通形式相同，并额外在 `tid` 中捕获该分发的 grid 级 producer `pl.Scalar[pl.TASK_ID]`（可用作 `deps=` 边、存入 `pl.array.create(N, pl.TASK_ID)`、或跨入 `pl.manual_scope`）。TaskId 捕获与内联 body 正交——这是该形式相比普通形式唯一多出来的能力。lower 成一个 `ir.Submit`，其尾部 tuple 元素即 grid TaskId；`core_num` / `sync_start` 记录在该 `Submit` 自身的字段上（launch spec 属于启动点，而非外包出的被调函数）。参见下文“手动依赖原语”小节。
- `out, tid = pl.spmd_submit(kernel, *args, core_num=N)` —— **submit 形式**：将 kernel 在 `N` 个 block 上分发，同时捕获该分发的 producer `pl.Scalar[pl.TASK_ID]`（针对已声明 kernel 的 `pl.submit` 版本）。参见下文“手动依赖原语”小节。

以上三种形式也都接受 `allow_early_resolve=True`（布尔字面量；与 `pl.submit` / `pl.at` 相同的 early-dispatch 选项）。即使不写 `as tid` 也会强制走 `ir.Submit` 形态，并 lower 为 `Arg::set_allow_early_resolve(true)`。在嵌套于 `pl.cluster()` 内的 `pl.spmd` 上会被拒绝（此类 scope 会被 unwrap 进 Group 函数、永远不会产生 Submit，提示会丢失）。

可选 `optimizations=[...]`。各条目彼此正交，可在同一列表中组合
（例如 `[pl.split(MODE), pl.cross_core_slot(slot_num=4)]`）：

| 条目 | 适用形式 | 作用 |
| ---- | -------- | ---- |
| `pl.split(MODE)` | 两种均适用 | 给内层 InCore 设置 `split_` 字段（跨核数据搬运提示，由 `ExpandMixedKernel` / `MemoryReuse` 消费）。with-form 会在原 call 外多包一层 `InCoreScopeStmt` 来承载该字段。 |
| `pl.cross_core_slot(slot_num=N)` | 两种均适用 | 给内层 InCore 设置 `slot_num` 属性——自动跨核 pipe 的槽位数（环深），由 `ExpandMixedKernel` 消费。它只决定数据通道大小，**不**划分计算，因此可与 `pl.split_aiv` 区域共存（而 `pl.split(...)` 不能）。省略时沿用默认深度：每个活跃方向 2 个槽位。 |

> `pl.split(MODE, slot_num=N)` 是该槽位数的已废弃别名，会发出警告——参见
> [ExpandMixedKernel](../passes/22-expand_mixed_kernel.md#覆盖槽位数slot_num)。

示例参见 [作用域与放置](../../user/language/04-scopes.md)。

### Yield 语句

```python
yield            # No values
yield x          # Single value
yield x, y       # Multiple values
```

### Break 和 Continue

```python
break              # 退出最内层循环
continue           # 跳到下一次迭代
```

**限制:** 仅当**最内层**封闭循环为顺序循环 (`pl.range`) 或 `while` 时有效。当最内层循环为 `pl.parallel()` 或 `pl.unroll()` 时不支持。在外层 `pl.parallel` 循环内嵌套的内层 `pl.range` 循环中使用 `break` 是合法的。**注意:** 代码生成后端对 `break`/`continue` 的支持跟踪在 [#448](https://github.com/hw-native-sys/pypto/issues/448) 中。

### 编译期调试 (Compile-Time Debugging)

`pl.static_print()` 和 `pl.static_assert()` 是仅在解析期执行的构造，用于在解析过程中检查 IR 状态和断言条件。它们**不生成任何 IR**。

```python
@pl.function
def func(x: pl.Tensor[[128, 64], pl.FP16]) -> pl.Tensor[[128, 64], pl.FP16]:
    pl.static_print("input:", x)          # → static_print [file:line]: input: x: pl.Tensor[[128, 64], pl.FP16]
    pl.static_print(f"input: {x}")        # → static_print [file:line]: input: x: pl.Tensor[[128, 64], pl.FP16]
    pl.static_assert(True)                # 静默通过
    pl.static_assert(N > 32, "N too small")  # 在解析期检查闭包变量 N
    return x
```

| 函数 | 用途 | 失败时 |
| ---- | ---- | ------ |
| `pl.static_print(*args)` | 将变量类型/值打印到 stdout | 需要 ≥1 个参数 |
| `pl.static_assert(cond, msg="")` | 断言编译期条件 | 抛出 `ParserError` |
| `pl.dump_tag(tensor)` | 把某个张量标记为运行期选择性 dump 的目标 —— 声明式逐张量标记（在 Orchestration 作用域，或被 orch 内联的 Inline helper 中均可使用 —— 见 [运行期 DFX](../03-runtime-dfx.md#选择性张量-dump)） | 在非 Orchestration / Inline 函数中使用、或参数不是裸变量名时抛出 `ParserSyntaxError` |

**要点：**

- 三者均为语句级构造（不能用在表达式中）
- `static_print` 接受变量、常量、字符串标签（原样打印）和 f-string 的简单 `{expr}` 占位符（格式化为 IR）。不支持转换标志（`!r`、`!s`、`!a`）和格式说明符（`:...`）。
- `static_assert` 支持闭包变量表达式（如 `N > 32`）和 IR 常量
- `static_assert` 的消息参数必须是字符串字面量
- `dump_tag` 接受单个绑定在外层 Orchestration（或 Inline）作用域内的张量变量名，在解析期被消耗，并自始至终以 Var 身份（而非名字）跟踪到 codegen。在显式 `self.kernel(...)` 调用点，它把该张量记录到每个后续消费它的 Call 的 `dump_vars` 上；在 `@pl.jit` / `with pl.at(level=...)` 风格（派发由 outline pass 合成）下，它改为写入所在 scope 的 `dump_vars`，再由 outliner 映射到合成派发的实参上（见 [运行期 DFX](../03-runtime-dfx.md#选择性张量-dump)）。若需要在单次 task 启动处显式列出 dump 目标，请用 `pl.submit(...)` / `pl.at(...)` 上的 `dumps=[...]` kwarg（与 `deps=` 对称）
- 即使后续解析失败，输出仍会显示——适用于调试解析错误

### 语句序列

```python
stmt1            # Natural Python sequencing
stmt2
stmt3
```

## SSA 风格控制流

`pl.yield_()` 为 if/for 语句创建 SSA phi 节点:

```python
# If: phi node at merge point
if condition:
    y1 = pl.yield_(x + 1)
else:
    y1 = pl.yield_(x + 2)
# y1 = phi(x + 1, x + 2)

# For: loop-carried values via iter_args
sum_init: pl.INT64 = 0
for i, (sum,) in pl.range(10, init_values=(sum_init,)):
    sum = pl.yield_(sum + i)
sum_final: pl.INT64 = sum  # captures final value
```

## 参考资料

- [Python IR 语法规范](00-python_syntax.md) —— 类型与表达式
- [手工依赖原语](02-manual_dependencies.md) —— 退出自动依赖跟踪
- [函数与程序结构](03-functions.md) —— 函数类型与参数方向
