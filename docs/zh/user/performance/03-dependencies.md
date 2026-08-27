# 管理任务依赖

运行时为什么有时会把本可同时跑的活排成一队，以及该怎么办。

> **前置**：[任务与定序](../tasks/index.md) —— 本页假定你已经知道 `deps=`、`manual_scope`、`no_dep_args` **是什么**。这里把它们当作调优决策来讲。

## 运行时到底推断了什么

任务依赖是在 submit 时从每个任务携带的 tensor 参数推出来的。把这条规则说准值得花这一段，因为本章里每一条假边都出自它：

| 步骤 | 作用于 | 效果 |
| ---- | ------ | ---- |
| **创建者保持** | *每一个* tensor 参数，任意方向 | 连一条边到创建该 tensor 的任务 |
| **生产者查找** | 仅 `INPUT` / `INOUT` | 对任何**区域重叠**的当前已注册生产者连一条边 |
| **生产者注册** | `INOUT` / `OUTPUT_EXISTING` | 本任务成为该 buffer 的已注册生产者 |

于是你得到两个经典冒险，外加一个缺口：

- **RAW** —— 读者查到当前写者并连边。**跟踪**。
- **WAW** —— 新写者对前一个写者连边，然后取而代之。**跟踪**。
- **WAR** —— 写者覆盖一块可能仍有纯读者在读的 buffer。**不跟踪**。写者得找出每一个在飞的读者，那是热路径上一次遍历读者集合的开销。需要这个定序，就得你自己拥有它。

循环种类叠在这之上：`pl.range` 是串行，`pl.parallel` 断言各次迭代互相独立。`pl.parallel` 是**断言而不是请求** —— 它不会移除上面那些边，它承诺的是你没有制造出会起作用的边。

下面这些 kernel 每次 CI 都会被执行，所以它们是真货而不是草图。它们共用这段准备：

<!-- doctest: setup -->
```python
import pypto.language as pl
import torch
from pypto.runtime import RunConfig

N, TILE_ROWS, COLS = 4, 64, 128
ROWS = N * TILE_ROWS
CFG = RunConfig(platform="__PLATFORM__")

torch.manual_seed(0)
A = torch.randn(ROWS, COLS, dtype=torch.float32)


def check(kernel):
    out = torch.zeros(ROWS, COLS, dtype=torch.float32)
    kernel(A, out, config=CFG)
    torch.testing.assert_close(out, A * 2.0, rtol=1e-4, atol=1e-4)
```

## 你没要的串行化

### 累加链

最常见的一种。一个串行循环，各次迭代写同一块 buffer，就产生一条 WAW 链，每次迭代一条边 —— 而这是**正确的**：这些写确实落在同一个地方。

<!-- doctest: run -->
```python
@pl.jit
def serialized(a: pl.Tensor, out: pl.Out[pl.Tensor]):
    for i in pl.range(N):
        with pl.at(level=pl.Level.CORE_GROUP):   # writes `out` every iteration
            t = pl.load(a, [i * TILE_ROWS, 0], [TILE_ROWS, COLS])
            pl.store(pl.mul(t, 2.0), [i * TILE_ROWS, 0], out)   # -> WAW edge on i-1
    return out


check(serialized)
```

它成为性能问题，是在各次迭代写的是那块 buffer 的**互不相交区域**、只是看上去撞在一起的时候。生产者查找是对 buffer 地址做**重叠**判定；一个它无法证明不相交的区域，会被当作重叠。

这也是为什么 `pl.range` 外层套 `pl.parallel` 内层经常令人失望：内层各次迭代之间也许确实可以重叠，但外层循环那块共享输出 buffer 仍然把各次迭代串成一条链，你在里面声明的并行度根本没有机会表现出来。

**修法是把编译器证明不了的事情说出来**，并且用能表达它的最窄作用域：

| 断言的范围 | 构造 |
| ---------- | ---- |
| 一个 tensor，一个任务 | `pl.at(..., no_dep_args=[t])` |
| 一个 tensor，其整个生命期 | `pl.create_tensor(..., manual_dep=True)` |
| 一个区域内的每个任务 | `with pl.manual_scope():` |

最窄的两种，作用在同一份工作上：

<!-- doctest: run -->
```python
@pl.jit
def narrow_claim(a: pl.Tensor, out: pl.Out[pl.Tensor]):
    for i in pl.range(N):
        with pl.at(level=pl.Level.CORE_GROUP, no_dep_args=[out]):
            t = pl.load(a, [i * TILE_ROWS, 0], [TILE_ROWS, COLS])
            pl.store(pl.mul(t, 2.0), [i * TILE_ROWS, 0], out)
    return out


@pl.jit
def region_claim(a: pl.Tensor, out: pl.Out[pl.Tensor]):
    with pl.manual_scope():                       # nothing is inferred in here
        for i in pl.range(N):
            with pl.at(level=pl.Level.CORE_GROUP):
                t = pl.load(a, [i * TILE_ROWS, 0], [TILE_ROWS, COLS])
                pl.store(pl.mul(t, 2.0), [i * TILE_ROWS, 0], out)
    return out


check(narrow_claim)
check(region_claim)
```

`manual_dep=True` 是中间那一档，它有一处锋利的地方值得完整看一遍：

<!-- doctest: run -->
```python
@pl.jit
def tensor_claim(a: pl.Tensor, out: pl.Out[pl.Tensor]):
    scratch = pl.create_tensor([ROWS, COLS], pl.FP32, manual_dep=True)
    writers = pl.array.create(N, pl.TASK_ID)
    for i in pl.range(N):
        with pl.at(level=pl.Level.CORE_GROUP) as tid:
            t = pl.load(a, [i * TILE_ROWS, 0], [TILE_ROWS, COLS])
            pl.store(pl.mul(t, 2.0), [i * TILE_ROWS, 0], scratch)
        writers[i] = tid
    # deps= is REQUIRED here: manual_dep dropped the consumer's RAW edges too,
    # so without it this task reads bands that have not been written yet.
    with pl.at(level=pl.Level.CORE_GROUP, deps=[writers]):
        for i in pl.range(N):
            t = pl.load(scratch, [i * TILE_ROWS, 0], [TILE_ROWS, COLS])
            pl.store(t, [i * TILE_ROWS, 0], out)
    return out


check(tensor_claim)
```

> **`manual_dep` 会把你想要的那些边一并抹掉。** 它覆盖该张量的整个生命期，所以之后的消费者也会失去它对写者的 RAW 边 —— 这正是上面那个 `deps=` 的原因。去掉那一行不会响亮地失败，而是返回一个**部分**正确的结果，也就是本节警告的那种偶发形态。

优先选能用的最窄那个。每一个都是编译器**无法检验**的断言 —— 如果那些区域其实是相交的，你修掉的不是串行化，而是造出了一个在别人机器上才复现的偶发竞态。

> **`manual_dep` 会把你想要的那些边一并抹掉。** 它作用于该张量的整个生命期，所以之后读这个张量的消费者也会失去它对那些写者的 RAW 边 —— 示例里的 `tensor_claim` 模式必须通过 `deps=` 把写者的 TaskId `pl.array` 交给消费者。不加的话，跑出来的结果是**部分**正确的，正是本节警告的那种偶发形态。

**还有第四种办法，而且它是唯一不需要断言的。** 在编排里把输出切开，把每一片分别传给对应的 InCore 函数，让这些任务干脆不再共享一块 buffer：

```python
for i in pl.range(N):
    part = pl.slice(out, [TILE, COLS], [i * TILE, 0])   # 每次迭代一个互不相同的区域
    with pl.at(level=pl.Level.CORE_GROUP):
        ...                                             # 写 `part`，不是 `out`
```

这样区域**在构造上**就是不相交的，运行时是推导出这一点，而不是被你告知。**代价：** 多出来的编排级张量本身也是活 —— 要注册的参数更多、要走的条目更多 —— 于是每个任务的解依赖时间变长。在一张本来就受派发限制的图上（[01](01-task-granularity.md)），这可能比它消掉的串行化还贵。两头都要测。

### 互相把对方串起来的读者

反方向的情形，而且它通常是以一个好意的修复登场的。因为 WAR 不被跟踪，一个必须在后续覆盖之前读完的读者身上没有边保护它。诱人的做法是把这个读者从 `INPUT` 提升成 `INOUT`，那确实建立了边 —— `INOUT` 会注册为写者，于是覆盖方对它连一条 WAW 边。

**而它把这块 buffer 的其他每一个读者都串了起来。** 每个 `INOUT` 读者依次成为已注册生产者，于是第二个对第一个连一条 WAW 边。一块被多个任务并发读的 tensor 会彻底失去这份并发，只为换一条反依赖。

改成显式声明那条边 —— 在写者上写 `deps=[reader_tid]` —— 并让读者保持 `INPUT`。

**两种修法怎么确认：** `enable_dep_gen=True` 后对比图 —— 被去掉的边应当消失，且其他什么都不该动。然后看泳道图，因为图上扇出并不能证明任务真的重叠了；某个环饱和了仍然会串行执行。两个都要看。

## 必须你自己建的细粒度边

有些依赖根本不以 buffer 重叠的形式存在，再怎么推断也找不到。pypto-lib 里的 `models/qwen3_14b/decode_fwd.py` 是这件事在规模上的参考 —— 一个几乎完全手工连线的 decode layer。

值得从中拿走的模式：**把 TaskId 数组上提到编排作用域**，这样一个在 `manual_scope` **之后**运行的消费者，仍然能对在它**里面**创建的任务设门。

```python
# 声明在 manual scope 之前 —— 这样一个更晚的、在外面的消费者才读得到
down_tids = pl.array.create(DOWN_ON * K_SPLITS, pl.TASK_ID)

with pl.manual_scope():
    # ... 循环在 submit 每个 down_proj 任务时填 down_tids[k]
    ...

# 出了作用域之后：让合并写者对那些生产者设门
with pl.at(level=pl.Level.CORE_GROUP,
           deps=[down_tids[k] for k in range(DOWN_ON * K_SPLITS)]):
    ...
```

在 `pl.manual_scope()` 里，运行时对该区域**整个跳过**扇入计算 —— 不只是生产者注册，创建者保持与生产者查找也一并跳过 —— 所以该区域内每一条边都是你写的。这正是它的意义所在：在一条推断出来的边大多是假的路径上，把真的那些声明出来，比把错的那些去掉更省事。

> 该模型用的是上面这种逐索引形式，而不是传整个数组。两种写法都存在（[用 TaskId 数组做扇入](../tasks/02-submit.md#用-taskid-数组做扇入)）；如果在这种跨作用域上提的场景里，整数组 `deps=` 没有产生你预期的边，逐索引列表就是那个模型所依赖的形式。

**代价：** 每一条边现在都归你负责，包括那些原本免费就正确的。manual scope 里少一条边是一个竞态，不是一条报错。

**怎么确认：** `enable_dep_gen=True`，然后拿图对着你的意图读 —— 这是唯一一种「该读整张图而不是读它的 diff」的情形。

## 你其实不需要的那些边

手工加过边之后，反过来的问题也值得一问：其中哪些本来就已经被蕴含了？当 `v` 能从 `u` 经由别的路径到达时，边 `(u, v)` 就是冗余的 —— 去掉它不会改变执行顺序，只会减少调度器要维护的簿记。

```bash
DEPS_JSON="outputs/<run>/deps.json"
python -m simpler_setup.tools.deps_viewer "$DEPS_JSON" --edge-mode reduced
python -m simpler_setup.tools.deps_viewer "$DEPS_JSON" --edge-mode reduced_dataflow
```

> **绝不要只看 `reduced` 就下结论。** 边带有 `source`，而 `creator` 边 —— 那些为了让「拥有某个消费者仍在引用的张量」的任务保持存活而存在的边 —— 会无条件地免于结构化归约，因为它们编码的根本不是顺序。这种保护是按 pair 生效的，所以一条 creator 标注就能护住整条边。在一份实测的图上（5120 条 `creator` 加 1008 条 `tensormap`），**全部** 2032 个冗余 pair 都带着 creator 标注：`reduced` 报 `0`，而 `reduced_dataflow` 去掉了 992 条。`reduced` 报零，是关于这个模式的证据，不是关于你这张图的。

`reduced_dataflow` 让 creator 边变得可去除，但只在该 pair 上每一条 creator 标注都是确切已知的 `INOUT` 区域、且每个字节都可证明是从更早的 `Output` 流向同一 creator 拥有的更晚的 `INOUT` 时才去。stride 元数据含糊或过于复杂会保留该边，`OUTPUT_EXISTING` 边也会 —— 它开启的是一个新的复用世代。

另外两件看着像答案、其实不是的事：

- **深度为 1 的图根本不可能有冗余边** —— 没有两跳路径，也就没有什么能蕴含一条边。先看深度；那里的 `0` 意味着审计到此为止，而不是说明这张图已经最简。
- **存在环会让归约失效，但不会让命令失败。** 工具会在 stderr 上告警、输出完整图，并且**仍然以 0 退出**。要读 stderr；退出码为 0 不能证明归约真的跑过。

审计本身只消耗 `deps.json` —— 不需要时序产物，也不需要设备。加上 `--func-names` 会多读一个文件，即本次运行的 `name_map*.json`，但值得：它会让打印出的边列表显示 kernel 名而不是数字 id。

## 怎么判断

```text
任务被串起来了，但本不该
├─ 它们确实写同一块 buffer          → 不是 bug；合并它们或改结构
├─ 它们写一块 buffer 的不相交区域    → no_dep_args / manual_dep，从最窄的开始
├─ 有读者为了定序被提成了 INOUT      → 撤回；改在写者上用 deps=
└─ 这条边根本与 buffer 无关          → manual_scope + 显式 deps
```

## 参见

- [任务与定序 § 依赖模型](../tasks/00-model.md) —— 完整的推导。
- [调度调优](../tutorials/05-scheduling-tuning.md) —— 同样的推理，在一个 kernel 上一步步走一遍。
