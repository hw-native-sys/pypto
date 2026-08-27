# 手工依赖原语

默认情况下 runtime 通过缓冲区读写重叠（`OverlapMap`）自动推导任务间依赖。
DSL 暴露**两套正交的机制**，用户可任意组合：

> **两套机制相互独立。** 把某个 buffer / 区域 / arg 从自动跟踪中"摘出来"
> 并**不要求**你同时声明显式边；声明显式边也**不要求**你同时关掉自动
> 跟踪。最终 task 的 fanin 是 **`自动跟踪 deps ∪ 显式 deps`**——它们
> 是相加而非互相替代。

## 机制 A——退出自动依赖跟踪（3 种粒度）

三种粒度彼此独立。按需选择最小的单位，必要时叠加。

| 表层语法 | 粒度 | 作用 |
| -------- | ---- | ---- |
| `with pl.manual_scope():` | per-region | 下沉为 `SIMPLER_SCOPE(ScopeMode::MANUAL)`。区域内 runtime 不做自动跟踪；用户需要的排序边必须通过机制 B 显式声明。 |
| `pl.create_tensor([...], dtype=..., manual_dep=True)` | per-tensor 生命周期 | 任何读 / 写该 tensor 的 task 都**整生命周期**跳过 `OverlapMap` 的 lookup 和 insert，不受 scope 影响。适合那种"完全交给显式边管理"的 scratch buffer。 |
| `pl.no_dep(arg)` | per-call 参数 | kernel 调用点上，被包装的参数其 `ArgDirection` 变为 `NoDep`——**仅本次提交**对该槽位不进入自动跟踪。不论 callee 把该槽位声明为 `In`、`Out` 还是 `InOut` 都合法：用户在带外（out-of-band）承诺该槽位不存在 RaW / WaW / WaR 冲突——例如 paged-attention 那种"写偏移是数据相关、但按分配协议保证不相交"的场景。在 `pl.manual_scope` 内没有意义（scope 已经全员退出）。 |
| `with pl.at(..., no_dep_args=[t1, t2]):` | per-arg, 作用于 `pl.at`-块 | `pl.no_dep(arg)` 在 `pl.at`-块上的对应物。outliner 把列出的 tensor 作为合成 kernel call 的实参；`DeriveCallDirections` 随后把这些实参槽位标为 `NoDep`——和在显式 call 站点用 `pl.no_dep(...)` 等效。每一项必须是外层 scope 可见的张量名。In / Out / InOut 的适用范围与 `pl.no_dep(arg)` 相同：如果 scope 体里用 `pl.assemble` 写过这个 capture，outliner 会把合成 kernel 上该形参推断成 `InOut`，`no_dep_args=` 仍然把它覆盖为 `NoDep`（和覆盖 `In` 一样）。注意：`no_dep_args=` 接收**张量**，`deps=` 接收 **TaskId**——同一个 "dep"，作用在不同层。 |

## 机制 B——显式声明 task 间的边（`deps=`）

这些表面都会下沉为 `set_dependencies` codegen；按 producer 形态选择：
单个 kernel 调用、outlined `pl.at` 区域，或 dependency-only fan-in。

| 表层语法 | producer 形态 | 备注 |
| -------- | ------------- | ---- |
| `result, tid = pl.submit(kernel, *args, deps=[...], allow_early_resolve=False)` | 单个 kernel 调用 | 尾部 `tid` 是 producer `pl.Scalar[pl.TASK_ID]`。它是 parser construct（类似 `pl.range`），不是 runtime 函数。`allow_early_resolve=True` 将该 task 标记为推测式 early-dispatch producer（让调度器提前预置其 consumer；lower 为 `Arg::set_allow_early_resolve(true)`）。同样接受 `predicate=(t[i] > 0)` —— 调度器在 dispatch 点求值的调度谓词（参见[调度谓词](#调度谓词predicate)）。 |
| `result, tid = pl.spmd_submit(kernel, *args, core_num=N, sync_start=False, deps=[...])` | 单个 SPMD task launch | `pl.submit` 的 SPMD 版本：将 kernel 在 `N` 个 block 上分发（一个 orchestration task → 一个 `tid`）。`core_num` 是必填关键字参数（正整数表达式）；`sync_start=True` 强制所有 block 原子启动。callee 可以是 InCore / AIC / AIV / Group。launch spec 记录在 `Submit.core_num` / `Submit.sync_start` 上。同样接受 `allow_early_resolve=True`（与 `pl.submit` 相同的 early-dispatch 选项）和 `predicate=(t[i] > 0)`（参见[调度谓词](#调度谓词predicate)）。 |
| `with pl.at(level=pl.Level.CORE_GROUP, deps=[...]) as tid:` | outlined `pl.at`-块 | 整块被 outline 成 InCore kernel + `Submit`；`tid` 捕获被合成的 Submit 的 TaskId，可作为后续 `pl.submit` / `pl.at` 的 dep。不写 `as tid` 时 outliner 会合成一个未使用的 TaskId Var——deps 始终走 `Submit::deps_`。同样接受 `allow_early_resolve=True`（与 `pl.submit` 相同的 early-dispatch 选项）；即使不写 `as tid` 也会强制走 `Submit` 形态，并 lower 为 `Arg::set_allow_early_resolve(true)`。 |
| `with pl.spmd(N, deps=[...]) as tid:` | outlined SPMD 分发 | `pl.at ... as tid` 形式的 SPMD 版本。内联 body 自动外包成 InCore kernel 并在 `N` 个 block 上分发；`tid` 捕获 grid 级 producer TaskId。`deps=` 仅在带 `as tid` 时可用。`core_num` / `sync_start` 记录在 lower 出的 `Submit` 自身的 `core_num` / `sync_start` 字段上（launch spec 属于启动点，而非外包出的被调函数）；codegen 直接从那里读取。同样接受 `allow_early_resolve=True`（与 `pl.submit` / `pl.at` 相同的 early-dispatch 选项；`pl.spmd` 三种形式均可用，即使不写 `as tid` 也会强制走 `Submit` 形态）和 `predicate=(t[i] > 0)`（参见[调度谓词](#调度谓词predicate)；同样三种形式均可用，同样强制走 `Submit` 形态）。不能嵌套在 `pl.cluster()` 内。 |
| `barrier = pl.system.task_dummy(deps=[...])` | dependency-only barrier | 不提交 kernel。返回的 TaskId 是一个紧凑的 fan-in 点，可供后续 `deps=[barrier]` 使用。 |
| `None`（Python 字面量） | 种子 / dep 条目 | "暂无 producer" 的哨兵。`prev_tid = None` 用作 TaskId 循环 iter_arg 的种子；`deps=[None]` 中的 `None` 被丢弃（不贡献任何边）。下沉为 `system.task_invalid` → `TaskId::invalid()`。 |

**这些表面都不依赖机制 A 的状态。** 显式 deps 可用于普通自动跟踪、
`pl.manual_scope()` 内或 `manual_dep=True` tensor 上，并总是在自动跟踪结果
**之上**追加；早期"`deps=` 只在 `pl.manual_scope` 内有效"的限制已经解除。

普通的 `out = self.kernel(...)` 是 **fire-and-forget**：它不返回 task id，
并且在它上面写 `deps=` 会被拒绝（parser 报错，提示 "use `pl.submit`"）。
每个 `deps=[...]` 条目必须是 TaskId 值：先前 `pl.submit(...)` /
`pl.at(..., deps=) as tid` 绑定的 `tid`、`pl.system.task_dummy(deps=[...])`
的返回值、TaskId 循环 iter_arg carry、从 TaskId 数组槽读出的
`Scalar[TASK_ID]`（`prev = tids[k]`）、来自
`pl.array.create(N, pl.TASK_ID)` 的 `Array[N, TASK_ID]`，或字面量 `None`。
`deps=[...]` 不接受 tensor。

```python
# 示例 1——两套机制同用：scope-wide 退出 + 显式边。
@pl.function(type=pl.FunctionType.Orchestration)
def main(self, x: pl.Tensor[[64], pl.FP32],
         scratch: pl.Out[pl.Tensor[[64], pl.FP32]],
         out: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
    with pl.manual_scope():                                           # 机制 A: scope-wide
        scratch, stage1_tid = pl.submit(self.stage1, x, scratch)
        out, _ = pl.submit(self.stage2, scratch, out, deps=[stage1_tid])  # 机制 B
    return out
```

```python
# 示例 2——只用机制 B，**不**进 manual_scope。其他 buffer 仍然走自动跟踪；
# 显式边是在自动跟踪结果**之上**追加。注意没有 `with pl.manual_scope():`。
@pl.function(type=pl.FunctionType.Orchestration)
def main(self, x: pl.Tensor[[64], pl.FP32],
         out: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
    tmp, prep_tid = pl.submit(self.preprocess, x)
    out, _ = pl.submit(self.consume, tmp, out, deps=[prep_tid])
    return out
```

```python
# 示例 3——以 pl.at-块作为 producer，给下游 pl.at-块加显式边。
# `as tid` 捕获被合成 outlined Call 的 TaskId。
@pl.function(type=pl.FunctionType.Orchestration)
def main(self, x: pl.Tensor[[64], pl.FP32],
         out: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
    with pl.at(level=pl.Level.CORE_GROUP) as tid_a:
        # 块体被 outline 成 InCore kernel
        ...
    with pl.at(level=pl.Level.CORE_GROUP, deps=[tid_a]) as tid_b:
        # 显式边——严格在 tid_a 块之后运行
        ...
    return out
```

```python
# 示例 4——机制 A 的 tensor-lifetime 形态：scratch buffer 整生命周期退出
# 自动跟踪；ordering 完全交给显式边管理。
scratch = pl.create_tensor([N], dtype=pl.FP32, manual_dep=True)
scratch, prod_tid = pl.submit(self.fill, x, scratch)
out, _ = pl.submit(self.consume, scratch, out, deps=[prod_tid])
```

`pl.submit` 脱糖为单个 `ir.Submit`，其返回类型是扁平的增广
`TupleType([*<kernel return types>, ScalarType(TASK_ID)])` ——
元素 `0..N-1` 是 kernel 结果，元素 `N` 是 producer TaskId。parser 把每个
`deps=[...]` 列表直接写入类型化的 `Submit::deps_` 字段（普通 `Call` 永不
携带 `manual_dep_edges`——ManualDepsOnSubmitOnly 不变式）。`pl.at(..., deps=)`
走相同的路径：outliner 读 `ScopeStmt` 上的 `attrs["task_id_var"]` +
`attrs["manual_dep_edges"]`，把它们一起搬到合成的 `Submit` 上（带 deps 但
没写 `as tid` 的 scope 会得到一个合成的未使用 TaskId Var，使派发仍是 Submit）。codegen 填充一个按精确依赖数定长的栈数组，
并对每个 task 发出一次 `params.set_dependencies(arr, count);` 调用。
runtime 的 `Arg::set_dependencies(ptr, count)` 直接接收调用者持有的任意
长度数组，所以单 call 的依赖边数没有硬上限。显式 fan-in 可写成
`barrier = pl.system.task_dummy(deps=[tids])`，再让 consumer `deps=[barrier]`；
它复用同一套 dependency parser，lowering 成 `rt_submit_dummy_task(...)`，
在 dep 全 invalid 时返回 invalid 且跳过 dummy submit，并可与自动
`ExpandManualPhaseFence` barrier 共存。

`pl.no_dep(arg)` 是 auto scope 原语；在 `pl.manual_scope` 内不起作用
（整个 scope 已经退出自动跟踪了）。

## 调度谓词（`predicate=`）

`pl.submit` / `pl.spmd_submit` 接受可选的
`predicate=(tensor[indices] <op> target)`。调度器在 **dispatch 点**
求值该比较 —— 此时该 task 的依赖已满足，值一定是最新的，
无需 orchestration 阶段 `wait_for_tensor_ready` 的阻塞。当比较结果为 **假** 时，该 task
被 **inline 退休**（根本不下发到 core），但仍结算 fanin/fanout，下游 consumer 照常解锁
—— 它不会从 task 图中消失。为 **真** 时正常下发。

典型用途是 MoE“跳过空专家”：所有专家静态 submit，每个带 `predicate=(row_count[e] > 0)`
并依赖 gather producer —— 调度器只下发非空专家，且无需阻塞 orchestration 去读每个专家的
行数。

> **该比较按普通表达式解析，但永不求值。** `rc[0, 0]` 就是 `pl.read` 的常规语法糖，
> 因此该 kwarg 下降为普通 IR —— `Gt(Cast(tensor.read(rc, [0, 0])), 0)` —— 复用 IR 已有的
> 比较节点，而非任何私有编码。它存放在 `Submit.predicate` 上，**不在语句位置**，所以这个
> `tensor.read` **不会**在 orchestration 中执行：执行它就会阻塞在 `wait_for_tensor_ready`，
> 正是谓词要消灭的事情。orchestration codegen 负责把该 Expr 分解成运行时的
> `operand OP target` 三元组，因此只接受下述形状。

| 组成 | 含义 | 约束 |
| ---- | ---- | ---- |
| `tensor` | dispatch 点读取的操作数张量 | 必须是具名 tensor（函数参数，或绑定到 tensor 的变量），且下标定位到单个元素 |
| `indices` | 定位 `tensor` 中一个元素的下标 | 每个下标是整数标量（`ConstInt` 或 int/index `Var`）；每个维度一个下标 |
| `<op>` | 比较算子 | 取 `==` `!=` `>` `<` `>=` `<=` 之一（单个、非链式比较） |
| `target` | 右侧比较值 | **整数字面量**（可负） |

镜像写法也被接受 —— `0 < rc[e]` 与 `rc[e] > 0` 含义相同。IR 按书写原样保留该比较；
orchestration codegen 会翻转算子，使 tensor 始终是运行时的操作数。

在 orchestration codegen 中 lower 为运行时 `CoreTaskPredicate` + `Arg::set_predicate(...)`
（operand → 其 `ext_<name>` 引用，`op` → `PredicateOp::*`，`target` 原样；`elem_size`
由运行时从张量 dtype 推导）。

**契约：** 谓词操作数张量的 producer **必须**是该 submit 的 `deps=` 之一，这样 dispatch
点读到的才是最新值。若遗漏，调度器可能在 producer 尚未写入张量前就求值谓词，从而基于过期
数据做出 dispatch 决策。

parser 只做**尽力而为的抽查**，不是保证:它记录 `pl.submit(...)` 通过元组解包绑定的结果变量，
当谓词操作数是其中之一、而对应 producer TaskId 未出现在 `deps=` 时报错。解析通过应理解为
"未发现明显错误"，而非"已证明正确"。

以下情况它**看不穿**，因而会静默接受:

| 未覆盖 | 原因 |
| ------ | ---- |
| `rc2 = rc` 后使用 `rc2[0, 0]` | 别名是新变量，没有记录的 producer |
| 张量作 `pl.Out` 实参传入、结果绑到新名字 | 只跟踪返回绑定，不跟踪实参别名 |
| `rc3 = self.helper(rc)` | 任何中间调用都会洗掉关联 |
| `res = pl.spmd_submit(...)` 单左值形式 | 该路径根本不记录 |
| `deps=` 中含 `Array[N, TASK_ID]` 条目——包括常见的 `deps=[tids[i]]` 写法 | 数组条目未逐个列出 producer，该 submit 的检查整体跳过 |
| producer 写在源码**后面**，例如循环携带的 `rc` 由上一轮迭代写入 | 查表发生在解析谓词的那一刻，其后的 producer 尚未记录 |

因此 `deps=` 写对仍然是作者的责任。

**表达力**固定为 `tensor[indices] OP const` —— 单个比较，与运行时单比较的
`DispatchPredicate` 对齐。链式比较（`0 < t[i] < 8`）、算术（`t[i] % 8 == 0`）、布尔组合
（`a[0] > 0 and b[0] > 0`）、以及非字面量的右侧（`t[i] > u[i]`）都会在解析期被拒绝；
请在前序 kernel 里把它们归约成一个 gate 值，再对该值做谓词。

```python
with pl.manual_scope():
    rc, g_tid = pl.spmd_submit(self.gate, rc, core_num=1)       # rc 的 producer
    out, _ = pl.spmd_submit(
        self.expert, x, out, core_num=1,
        deps=[g_tid],                                           # producer 是依赖
        predicate=(rc[0, 0] > 0),
    )
```

**范围：** `predicate=` 可用于 `pl.submit` / `pl.spmd_submit`（直接产 `Submit` 的形式），
以及 `with pl.spmd(...)` 作用域形式的全部三种写法（普通 `with`、`with ... as tid`、
`for i in pl.spmd(...)`）。`pl.at(...)` 不接受该参数。

### 作用域形式

作用域形式使用相同的表达式与相同的校验，区别只在于谓词进入 IR 的路径：它先挂在
`SpmdScopeStmt.attrs` 上，直到该作用域被 outline 时才移动到 `Submit.predicate`。
因此 lowering、codegen 产物与契约都完全一致。

```python
with pl.spmd(1) as g_tid:                                        # rc 的 producer
    rc = self.gate(rc)

with pl.spmd(4, deps=[g_tid], predicate=(rc[0, 0] > 0)) as tid:  # producer 是依赖
    out = self.expert(x, out)
```

由这条路径引出两点：

- **`deps=` 需要 `as tid` 形式。** `deps=` 只在 `with pl.spmd(...) as tid:` 上被接受。
  因此，若谓词读取的张量由同一函数内的其他任务产出，就必须用该形式；普通 `with` 与
  `for` 形式只能对没有函数内 producer 的张量（通常是函数参数）加谓词——这种情况契约
  检查会放行。
- **其余情况不要求 `as tid`。** 与 `allow_early_resolve=True` 一样，谓词会强制该作用域
  lower 为 `Submit`；当作用域没有 `as tid` 时，outliner 会合成一个未被使用的 TaskId Var。

嵌套在 `pl.cluster()` 内的 `pl.spmd` 会被展开进 Group 函数、永远不会产生 `Submit`，
因此 `predicate=`（与 `allow_early_resolve=` 一样）会在解析期被拒绝，而不是被静默丢弃。

契约检查同样覆盖作用域 producer：在 `with pl.spmd(...) as tid:` 体内被赋值的张量会被记录
为该作用域的产物，所以后续 `deps=` 漏写它会被拒绝。上表中列出的 best-effort 限制
（别名、中间调用、`Array[N, TASK_ID]` 依赖）依然适用。

## Manual scope 下的 `pl.parallel`：array-carry fence

当 manual-dep 边穿过一个 `pl.parallel` 循环（即循环 iter_arg 承载被依赖的
TaskId）时，orchestration codegen 把对应的 TaskId iter_arg 视作**大小等于
parallel 循环 trip count 的数组**。每次 parallel 迭代写入自己的槽位；
下游消费者依赖**每一个**槽位（不是只依赖"最后被发射"的那个
task）。这就保证用户声明的 fence 语义即便在迭代乱序完成时也是正确的。

走 array-carry 路径的前提：

- `pl.parallel` 的 trip count 必须是 Python 字面量（编译期常量）。
  trip count 是动态值的情况下 codegen 会拒绝，提示 "statically-known
  trip count"。

```python
with pl.manual_scope():
    prev_tid = None                                      # 种子：还没有 producer
    for phase in pl.range(N_PHASES):
        for branch in pl.parallel(N_BRANCHES):           # 编译期常量
            row = (phase * N_BRANCHES + branch) * TILE_M
            out, prev_tid = pl.submit(self.kernel_stripe, data, row, 1.0, out, deps=[prev_tid])
```

`prev_tid` 在 `pl.parallel` 内被重新绑定，所以 codegen 把 carry 下沉为
`TaskId[N_BRANCHES]` 数组。phase `N+1` 中的每个 task 都会等待
phase `N` 的全部 `N_BRANCHES` 个 task，而非只等最后那个。

## 参考资料

- [语句与控制流](01-statements.md) —— 这些原语所依赖的作用域上下文管理器
- [编排代码生成](../codegen/01-orchestration_codegen.md) —— 它们如何下降
- [AutoDeriveTaskDependencies](../passes/39-auto_derive_task_dependencies.md) —— 消费这些信息的 pass
