# GM 缓存访问策略

一种**声明式**策略，用于描述 kernel 如何从全局内存（GM，global memory）读取某个
张量。`CachePolicy.BYPASS` 表示"流式读取该张量 —— 不要为它花费缓存"；
`CachePolicy.DEFAULT` 则是当前所有读取默认采用的普通缓存访问。

该策略是*作者声明的契约（contract）*，绝不是编译器推断出来的提示（hint）。因此它必须
显式书写，粒度二选一，并且从 DSL 一路原样传递到代码生成（codegen）。

> **当前状态**：PTOAS 尚未提供 L2 bypass 通路
> （[PTOAS#1356](https://github.com/hw-native-sys/PTOAS/issues/1356)），因此
> `BYPASS` 声明目前会**告警，并按普通缓存访问编译**。生成代码在有无该声明时逐字节
> 一致。参见[当前状态](#当前状态)。

## 两个书写面

| 书写面 | 粒度 | 写法 | 适用场景 |
| ------ | ---- | ---- | -------- |
| `pl.set_cache_policy(t, policy)` | 所在作用域（scope）内对 `t` 的每一次读取 | `pl.at(...)` / `pl.spmd(...)` body 顶层的一条独立语句 | 张量编程 —— GM 读取是隐式的（`pl.matmul`、`pl.assemble`、切片），没有 load 调用可供标注 |
| `pl.load(..., cache=policy)` | 单次读取 | load 上的一个 kwarg | Tile 编程 —— 该次访问本来就由你显式写出 |

`pl.slice` / `tensor.slice` 刻意**不**提供 `cache=` kwarg：slice 计算的是地址描述符
（address descriptor），并不搬运数据。策略属于真正发起 GM 读取的那个 op。

### 作用域声明

```python
@pl.program
class Demo:
    @pl.function
    def main(self, a: pl.Tensor[[256, 128], pl.FP32], b: pl.Tensor[[128, 256], pl.FP32],
             out: pl.Out[pl.Tensor[[256, 256], pl.FP32]]) -> pl.Tensor[[256, 256], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="mm"):
            pl.set_cache_policy(b, pl.CachePolicy.BYPASS)     # every read of b streams
            c: pl.Tensor[[256, 256], pl.FP32] = pl.matmul(a, b, out_dtype=pl.FP32)
            out = pl.assemble(out, c, [0, 0])
        return out
```

### 单次访问的 kwarg

```python
tile = pl.load(x, [0, 0], [32, 32], cache=pl.CachePolicy.BYPASS)
```

## 契约

`CachePolicy.BYPASS` 对该张量断言了**两件**事：

| 断言 | 由谁保证 | 编译器是否检查？ |
| ---- | -------- | ---------------- |
| kernel 只是流式读取这些字节 —— 不存在值得缓存的复用 | 作者 | 否（这是性能层面的断言） |
| kernel 运行期间没有任何一方写这些字节 | 作者 | 部分检查 —— 见下文 |

对同一批字节混用带缓存的写与 bypass 的读，是一个**一致性（coherency）缺陷**：两条通路
可能看到不同的数据。编译器无法跨任务、跨 rank 或跨 host 证明"不存在并发写者"，因此
一致性是作者的契约。这也正是该策略绝不作为默认值、也绝不由优化 pass 推断出来的原因。

编译器*能*检查的那一部分，它确实做了：对**本作用域自身会写**的张量声明 `BYPASS`，
会在外提（outlining）阶段被拒绝（见[错误](#错误)）。作用域自身的参数方向
（param direction）已经说明了它是否写该张量，所以这类自伤式的一致性缺陷会被直接拦下，
而不是被信任放过。

## 优先级

一次 GM 读取的生效策略，按以下顺序取第一个命中的：

1. 它自己显式写出的 `cache=` kwarg，否则
2. 它所读参数对应的作用域声明，否则
3. `CachePolicy.DEFAULT`。

显式写法在**两个方向上**都优先 —— `cache=pl.CachePolicy.DEFAULT` 可以在一个 bypass
作用域内，把某一次访问单独放回缓存：

```python
with pl.at(level=pl.Level.CORE_GROUP):
    pl.set_cache_policy(b, pl.CachePolicy.BYPASS)
    hot = pl.load(b, [0, 0], [32, 32], cache=pl.CachePolicy.DEFAULT)  # cached anyway
    rest = pl.load(b, [32, 0], [32, 32])                              # BYPASS (declared)
```

body 中已经存在的 load（手写的，或由更早的 pass 产生的）与编译器合成的 load 适用同一
条规则：除非该 load 自己写了 `cache=`，否则声明对它生效。

## 声明可以写在哪里

`pl.set_cache_policy` 挂到*所在作用域*上，而 `ScopeStmt` 的 attrs（属性）在作用域开始
时就已固定。因此 parser 会预扫描作用域 body 的**顶层**语句，在解析 body 之前先把这些
标记（marker）提升（hoist）到作用域上。

| 位置 | 是否接受 | 原因 |
| ---- | -------- | ---- |
| `with pl.at(level=pl.Level.CORE_GROUP, ...):` 顶层 | 是 | 该作用域会成为设备 kernel，声明正是针对它的参数做解析 |
| `for i in pl.spmd(N):` / 内联 `with pl.spmd(N):` body 的顶层 | 是 | 挂到 body 被外提进入的 InCore 载体上 |
| `with pl.at(<非 CORE_GROUP level>):` 顶层（Hierarchy） | 语法上接受 | 会解析到该外提函数上；但目前没有任何 pass 把它下沉到 load（见[限制](#限制)） |
| *dispatch* 形态的 `with pl.spmd(N):` body（调用预先定义好的 kernel） | 否 | GM 读取发生在被调用者内部 —— 请在那里声明 |
| `pl.cluster(...)` / `pl.manual_scope()` / `pl.scope()` body | 否 | 这些作用域只做协同调度或选择依赖语义，本身不发起 GM 读取 |
| 作用域内嵌套在 `if` / `for` 里 | 否 | 有条件执行的声明是编译器无法校验的承诺 |
| 函数 body，但不在任何作用域内 | 否 | 没有可挂载的作用域 |

parser 还强制另外两条规则：

- **按 Var 身份跟踪，绝不按名字。** 声明指的是*作用域处*活跃的那个绑定。之后重新绑定
  同名变量（`b = self.foo(b)`）得到的是一个新值，声明不覆盖它。
- **必须是裸变量名、张量类型、且已绑定。** 属性 / 下标 / 调用表达式并不命名任何绑定；
  非张量绑定没有 GM 读取可管辖；在 body *内部*创建的张量不会被作用域捕获（capture）。

对同一个绑定重复声明属于冗余而非错误 —— 保留第一条，因此该 attr 始终是一组互不相同
的张量。

## 错误

| 消息（节选） | 抛出方 | 原因 |
| ------------ | ------ | ---- |
| `pl.set_cache_policy() must be a standalone statement directly inside a pl.at(...) / pl.spmd(...) scope body` | Parser（`ParserSyntaxError`） | 写在作用域之外，或嵌套在 `if` / `for` 内 |
| `pl.set_cache_policy() has nothing to attach to on this <Kind> scope` | Parser（`ParserSyntaxError`） | Spmd dispatch body、`pl.cluster` 或运行时作用域 |
| `pl.set_cache_policy() takes exactly two positional arguments (no keywords)` | Parser（`ParserSyntaxError`） | 元数不对，或使用了关键字形式 |
| `pl.set_cache_policy() first argument must be a bare variable name` | Parser（`ParserSyntaxError`） | `t.field`、`t[0]`、`f(t)` —— 没有可跟踪的绑定 |
| `pl.set_cache_policy() argument '<n>' is not defined at this point` | Parser（`ParserSyntaxError`） | 作用域起始处该名字尚未绑定 |
| `pl.set_cache_policy() argument '<n>' is not a tensor` | Parser（`ParserTypeError`） | 只有 GM 张量读取才有缓存策略 |
| `pl.set_cache_policy(...) references tensor '<n>', which is not captured by the scope body` | `OutlineIncoreScopes`（`CHECK_SPAN` → `ValueError`） | 作用域 body 既不读也不写该张量，因此它未被捕获，没有参数承载该策略 |
| `pl.set_cache_policy(<n>, CachePolicy.BYPASS) is not allowed on a tensor this scope writes (<dir>)` | `OutlineIncoreScopes`（`CHECK_SPAN` → `ValueError`） | 对同一 kernel 自己会写的字节做 bypass 读取，是一致性缺陷 |

外提阶段这两条拒绝属于**用户错误**，而非编译器 bug —— 所以用 `CHECK_SPAN`，它会附带
IR 源码位置。

## 载体链路

声明在下沉过程中三次更换载体（carrier）。每一跳都有其存在理由，彼此不可互换。

```text
pl.set_cache_policy(b, BYPASS)                 statement, consumed at parse
  -> ScopeStmt.attrs_["cache_policy_vars"]     parse .. pass 8   (Var identity)
  -> Function attr "cache_policy"              pass 8 .. pass 10 (param INDICES)
  -> tile.load kwarg "cache"                   pass 10 .. codegen
  -> codegen: warn, emit an ordinary cached access
```

| 跳 | 载体 | 负载类型 | 写入方 | 消费方 |
| -- | ---- | -------- | ------ | ------ |
| 1 | `ScopeStmt.attrs_[kAttrCachePolicyVars]` | `vector<pair<VarPtr, int>>` | DSL parser | [`OutlineIncoreScopes`](../passes/08-outline_incore_scopes.md)（pass 8） |
| 2 | `Function.attrs_[kAttrCachePolicyParams]` | `vector<pair<int32_t, int>>`，按索引排序 | pass 8 | [`ConvertTensorToTileOps`](../passes/10-convert_tensor_to_tile_ops.md)（pass 10） |
| 3 | `tile.load` 的 `cache` kwarg | `int`（`ir::CachePolicy`） | pass 10 | PTO codegen |

保证这条链路不出错的几点设计考量：

- **不放在 `TensorView` 的字段上。** 普通 kernel 参数根本没有 `tensor_view_`，在那里
  打策略会强行造出一个 TensorView —— 从而牵入严格的 `TensorViewCanonical` verifier；
  而且 [`MaterializeTensorStrides`](../passes/31-materialize_tensor_strides.md) 会通过
  一个位置参数构造函数重建该 view，会静默丢掉这个字段。
- **参数索引只在 pass 8..10 之间有效。** 二者之间只夹着 `OutlineClusterScopes`，它不会
  改动已外提 InCore 函数的参数列表。而下游的 pass *会*改：
  [`InjectGMPipeBuffer`](../passes/23-inject_gm_pipe_buffer.md) 与
  [`MaterializeDistTensorCtx`](../passes/44-materialize_dist_tensor_ctx.md) 追加参数，
  [`MaterializeValidShapeSymbols`](../passes/49-materialize_valid_shape_symbols.md)
  则在*前面插入*。这正是 pass 10 转换完成后必须擦除该 attr 的原因。
- **kwarg 是 `int` 而不是枚举。** 它沿用 `tile.store` 的 `atomic` kwarg 做法，因此
  序列化器、反序列化器、`structural_hash` 与 `structural_equal` 都无需新增枚举分支。
  出于同样的理由，`pl.CachePolicy` 绑定为可转 int（`nb::is_arithmetic`），DSL 直接
  传 `int(cache)`。
- **`cache` kwarg 能存活到流水线末端**，理由与 `target_memory` 相同 —— 它挂在 op 的
  kwargs 上，后续没有任何 pass 会重写它。

## 打印与往返

作用域 attr 打印为**标记语句**，而不是像 `no_dep_args=` / `dumps=` 那样的头部 kwarg，
因为语句才是 parser 接受的书写面：

```python
with pl.at(level=pl.Level.CORE_GROUP, name_hint="mm"):
    pl.set_cache_policy(b, pl.CachePolicy.BYPASS)
    ...
```

| 性质 | 行为 |
| ---- | ---- |
| 顺序 | 位置归一化 —— 无论作者原本把它们写在哪里，标记总是打印在最前面；parser 也会从 body 的任意位置把它们提升上来 |
| Spmd 内联形态 | 从嵌套的 InCore 载体打印 —— 它的 `pl.at(...)` 头部被 Spmd printer 内联掉了 |
| 其余为空的作用域 | 只含一条声明的作用域打印该标记，而不是 `pass` |
| 函数 attr（`cache_policy`） | 打印为 `(索引, 策略)` 元组列表，因此在 pass 8 与 pass 10 之间 —— 它唯一存在的窗口 —— 抓取的 pass dump 可以重新解析 |

## 当前状态

PTOAS 尚未提供 L2 bypass 通路
（[PTOAS#1356](https://github.com/hw-native-sys/PTOAS/issues/1356)）。因此 codegen 会把
该请求一路携带下来，然后按普通缓存访问编译，并按"每 kernel 每张量一次"告警（而不是
每条发射出的 load 一次 —— 展开后的循环会发射同一条 load 很多次）：

```text
[warning] [CacheBypassUnsupported] tensor 'b' requests CachePolicy.BYPASS, but PTOAS
has no L2-bypass path yet (https://github.com/hw-native-sys/PTOAS/issues/1356);
compiling as an ordinary cached access at <file>:<line>
```

生成的 MLIR 在有无该声明时**逐字节一致**。今天就写上它的意义在于：等 PTOAS 侧落地
后，kernel 可以零成本地享受到 bypass —— 届时告警点会被原地替换为一个以 bypass 为根的
tensor view，codegen 之上的一切都无需改动。

### 限制

- 只有 **InCore** kernel 的 load 会接收该声明：把 GM 读取变成 `tile.load` 的是
  `ConvertTensorToTileOps`，而它变换的是 InCore 函数。请把策略声明在会成为设备 kernel
  的那个作用域上（`CORE_GROUP` 的 `pl.at`，或 `pl.spmd` 内联 body）。
- 该策略管辖的是**读**。没有对应的 store 侧机制；对被写张量声明 `BYPASS` 会被拒绝，
  而不会被重新解释为别的含义。

## 实现位置索引

| 层次 | 文件 |
| ---- | ---- |
| 枚举、attr key | `include/pypto/ir/expr.h`（`CachePolicy`、`kAttrCachePolicyVars`、`kAttrCachePolicyParams`） |
| Op 注册 | `src/ir/op/tile_ops/memory.cpp`（`tile.load` 的 `.set_attr<int>("cache")`） |
| DSL | `python/pypto/language/op/tensor_ops.py`（`set_cache_policy`）、`python/pypto/language/op/tile_ops.py`（`load(cache=...)`） |
| Parser | `python/pypto/language/parser/ast_parser.py`（标记提升 + 各项拒绝） |
| 外提 | `src/ir/transforms/utils/scope_outline_utils.cpp` |
| 下沉 | `src/ir/transforms/convert_tensor_to_tile_ops_pass.cpp` |
| Printer | `src/ir/transforms/python_printer.cpp`（`PrintScopeCachePolicyStmts`） |
| Codegen | `src/backend/common/pto_ops_memory.cpp`（`MakeTileLoadCodegenPTO`） |

## 另请参阅

- [语句与控制流](01-statements.md) —— 作用域形态，以及其它解析期标记
  （`pl.dump_tag`、`pl.static_assert`）。
- [OutlineIncoreScopes](../passes/08-outline_incore_scopes.md) —— 第 1 跳 → 第 2 跳。
- [ConvertTensorToTileOps](../passes/10-convert_tensor_to_tile_ops.md) —— 第 2 跳 → 第 3 跳。
