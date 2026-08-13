# 声明一条边

给一次派发起个名字，好让后面的任务等它。

> **前置**：[依赖模型](00-model.md) 与 [运行时作用域](01-scopes.md)。

## Concept

普通的 kernel 调用也会成为任务，但你拿不到它的句柄。要声明一条推导够不着的边，你需要一个 **TaskId** —— 指代某一次派发的句柄 —— 以及把它作为依赖交给后续任务的办法。

有三种写法，用哪些取决于函数是怎么写的，而不是偏好：

| 写法 | 可用于 | 命名的是 |
| ---- | ------ | -------- |
| `with pl.at(level=..., deps=[...]) as tid:` | `@pl.jit` 与 `@pl.function` | 一个内联区域 |
| `with pl.spmd(n, deps=[...]) as tid:` | `@pl.jit` 与 `@pl.function` | 一个内联的多 block 区域 |
| `result, tid = pl.submit(self.kernel, ...)` | 仅 `@pl.program` 类 | 一个预先声明的 kernel |

两种作用域写法用 `as` 绑定 TaskId；`pl.submit` 则是返回它。`pl.submit` 要求被调方写成 `self.<kernel>` —— 外层 `@pl.program` 类的一个方法 —— 所以在 `@pl.jit` 函数里够不着它，因为那里的 kernel 是普通的模块级函数。三者喂给的是同一套 `deps=` 机制。

**显式边与手动作用域无关。** `deps=` 与自动跟踪是叠加的 —— 最终等待集合是两者的并集 —— 所以显式边待在普通的 auto 作用域里完全正常。在那里把它当作精修工具，补上推导够不着的那一条边；只有当你想要整张图时才动用 [`pl.manual_scope`](01-scopes.md)。

## Quickstart：auto 作用域里的一条显式边

```python
import pypto.language as pl

@pl.jit
def two_stage(
    x: pl.Tensor[[256, 128], pl.FP32],
    scratch: pl.Out[pl.Tensor[[256, 128], pl.FP32]],
    out: pl.Out[pl.Tensor[[256, 128], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP) as first:
        scratch[:] = pl.add(x, x)
    with pl.at(level=pl.Level.CORE_GROUP, deps=[first]) as second:
        out[:] = pl.add(scratch, scratch)
    return scratch, out
```

| 元素 | 含义 |
| ---- | ---- |
| `as first` | 绑定该区域的 TaskId —— 后续任务点名用的句柄 |
| `deps=[first]` | 本区域等待那个生产者，无论 OverlapMap 推没推出这条边 |

注意这里没有 `manual_scope`。`deps=` 是加在正常跟踪之上的，不是取而代之。

## Mechanics

### 把 `pl.at` 当作任务边界

每个 `pl.at` 区域就是一次派发，所以用 `as tid` 绑定它就拿到了它的 TaskId。依赖相关的关键字有：

| 关键字 | 用途 |
| ------ | ---- |
| `deps=[...]` | 本区域必须等待的 TaskId 标量／数组 |
| `no_dep_args=[...]` | 退出跟踪的张量 —— 见 [精修依赖图](03-tuning.md) |
| `dumps=[...]` | 标记为选择性 dump 的张量 |
| `allow_early_resolve=True` | 调度提示 —— 见 [精修依赖图](03-tuning.md) |

`pl.at` 没有 `predicate=`。需要带谓词的区域请用 `pl.spmd` 或某种 submit 写法。

### 把 `pl.spmd` 当作任务边界

`pl.spmd(n)` 作为放置构造在 [作用域与放置](../language/04-scopes.md) 里已有说明；它同样是一次派发，所以也能用同样的方式命名一个任务：

```python
with pl.spmd(4, name_hint="stage1") as first:
    ...
with pl.spmd(4, name_hint="stage2", deps=[first]) as second:
    ...
```

它接受 `deps=`、`predicate=` 与 `allow_early_resolve=`，这使它成为唯一带派发谓词的内联写法。

**`deps=` 只有 `as` 形式接受。** 用 `as` 绑定才拿得到 TaskId；裸 `with pl.spmd(4):` 与 `for i in pl.spmd(4):` 照样执行同样的工作但不给它命名，向这两种形式传 `deps=` 会被拒绝并报 `pl.spmd() does not accept 'deps=' here`。

### `pl.submit`

用于写成 `@pl.program` 类、kernel 预先声明的程序：

```text
result, tid = pl.submit(self.kernel, *kernel_args, deps=[...], dumps=[...],
                        allow_early_resolve=False, timing_slot=N, predicate=(...))
```

被调方之后的位置槽是 kernel 自己的实参；其余都是可选关键字。被调方**必须**写成 `self.<kernel>`，其他任何表达式都是解析错误。

可用的解包形状：

```python
a, tid = pl.submit(self.k1, x)          # 结果与 TaskId
res    = pl.submit(self.k1, x)          # 整个扁平元组绑到一个名字
```

对那个扁平元组做下标取到的 TaskId **不能**喂给 `deps=` —— 依赖必须是 TaskId 变量或 TASK_ID 数组元素，所以要按名字绑定。

### `pl.spmd_submit`

SPMD 版本：一个编排任务，由运行时在 `core_num` 个逻辑 block 上展开，每个 kernel 通过 `pl.tile.get_block_idx()` 读自己的下标。它仍然只返回一个生产者 TaskId，因此整个展开可以作为一条依赖被点名。

```python
a, tid = pl.spmd_submit(self.k1, x, core_num=8)
```

`core_num` 是**必需的关键字** —— 位置槽属于 kernel。`sync_start`（默认 `False`）要求所有 block 原子启动。`deps=`、`allow_early_resolve=`、`timing_slot=` 与 `predicate=` 的行为与 `pl.submit` 完全一致。

### Device timing slot

`timing_slot=N`（`N` 必须是 `0` 到 `15` 的整数文本）把一个 task 标记为 device timing 的成员。runtime 对每个 slot 输出一个 span：同 slot 所有 task 的最早 dispatch 到最晚 completion。把 all-gather 与 GEMM 标到同一个 slot，就可以测量二者合成的 device 区间；warmup 与对齐 barrier 不标记即可排除：

```python
_, barrier_tid = pl.submit(self.all_rank_barrier, signal)
gathered, _ = pl.spmd_submit(self.gather, x, core_num=N, deps=[barrier_tid], timing_slot=0)
out, _ = pl.spmd_submit(self.gemm, gathered, w, core_num=N, deps=[barrier_tid], timing_slot=0)
```

产生的 trace span 名称为 `simpler_run.runner_run.device_wall.task_slot_0`。slot 只在单个 device 时钟域内有效；L3 汇总应取各 rank duration 的最大值，不能相减不同 device 的时间戳。

### 用 TaskId 数组做扇入

一个 TaskId 只指代一个任务。要等待**一组**任务 —— 比如一个循环产出的全部生产者 —— 把它们收进一个 `pl.TASK_ID` 的 `pl.array`，再把数组本身作为依赖传入：

```python
tids = pl.array.create(branches, pl.TASK_ID)
for branch in pl.parallel(branches):
    out, tid = pl.submit(self.producer, data, branch, out)
    tids[branch] = tid
out, _ = pl.submit(self.consumer, data, out, deps=[tids])
```

`deps=` 既接受标量也接受数组，所以消费者会等待循环创建的每一个生产者。记住数组更新是重绑定 —— 在循环内数组和其他携带值一样。见 [控制流](../language/02-control-flow.md)。

## 边界情况

> **致命陷阱：** `deps=` 给你的**只有**这条显式边。它不蕴含手动作用域，手动作用域也不蕴含任何边。把这两件事混为一谈 —— 以为 `manual_scope` 会给里面的语句定序 —— 得到的是一个什么顺序都没有的区域。

| 症状 | 可能原因 | 修复 |
| ---- | -------- | ---- |
| **`pl.submit(...) first argument must be a self.<kernel> method reference`** | 在 `@pl.jit` 函数里用了 `pl.submit` | 改用 `with pl.at(..., deps=[...]) as tid:` —— 多 block 派发则用 `with pl.spmd(n, deps=[...]) as tid:` |
| **`pl.submit is a DSL parser construct and cannot be called directly`** | 在被装饰函数体外使用 | 移进被装饰的函数体内 |
| **`unpacks 1 result value(s) but kernel returns 0`** | kernel 写的是 `Out` 参数且没声明返回类型 | 只解包 kernel 真正返回的东西，或给它加上返回类型 |
| **`deps= entries must be a TaskId variable`** | TaskId 是对扁平 submit 结果做下标得到的 | 把 TaskId 绑定到独立的名字再传 |
| **`pl.spmd() does not accept 'deps=' here`** | 向裸 `with` 或 `for` 形式传了 `deps=` | 用 `as tid:` 绑定该区域 —— 只有这种形式接受 `deps=` |
| **`core_num` 缺失** | 它是 `pl.spmd_submit` 的必需关键字 | 传 `core_num=N`；位置槽是 kernel 的实参 |
| **消费者只等到了循环的最后一个生产者** | 复用了一个 TaskId 而没有收集 | 收进 `pl.TASK_ID` 的 `pl.array` 并把数组传入 |
| **auto 作用域里显式边似乎被忽略** | 并没有 —— 等待集合是并集 | 去别处找**缺失**的边，而不是被丢弃的边 |

## See Also

- [运行时作用域](01-scopes.md) —— 自动跟踪在哪里开、在哪里关。
- [精修依赖图](03-tuning.md) —— `predicate=` 与 `allow_early_resolve=`。
- [控制流](../language/02-control-flow.md) —— 携带值，TaskId 数组正是其中一种。
- [编译期指令](../language/05-directives.md) —— `dumps=` 所喂给的 dump 标记。
