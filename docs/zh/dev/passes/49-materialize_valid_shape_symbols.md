# MaterializeValidShapeSymbols Pass

将设备 kernel 中无法绑定的 `valid_shape` 符号转换为前置的 `Scalar[INDEX]`
参数，并在每个调用点传入调用方的实际有效范围（valid extent）。

## 概述

在参数的 `pl.TensorView(valid_shape=...)` 中使用的 `pl.dynamic()` 符号，在预编译
kernel 内部没有取值：

```python
VALID = pl.dynamic("VALID")

@pl.function(type=pl.FunctionType.InCore)
def softmax_prepare(
    sij: pl.Tensor[[Q, BLK], pl.FP32,
                   pl.TensorView(valid_shape=[Q, VALID], layout=pl.TensorLayout.ND)],
    out: pl.Out[pl.Tensor[[Q, BLK], pl.FP32]],
): ...
```

`VALID` 既不是物理张量维度（那类符号由 kernel wrapper 从运行时 tensor 的
`shapes[]` 还原），也不是标量参数。运行时 `ChipTensor` 并不携带有效范围信息（参见
`runtime/src/common/task_interface/tensor.h`），因此该值必须以参数形式传入。在本
pass 出现之前，PTO codegen 会打印 `Variable VALID not found in MLIR mapping` 并继续
执行，生成缺少操作数的 `%0 = arith.minsi , %c128_index : index`，最终在很靠后的阶段
表现为难以定位的 ptoas `error: expected SSA operand`。

本 pass 将程序改写为复用既有的标量参数通路，端到端传递该值：

1. 对每个设备 kernel（`InCore` / `AIC` / `AIV` / `Spmd`），找出参数声明的
   `valid_shape` 中读取、且未被任何张量参数的物理 shape 绑定、也不是已有标量参数的
   符号。
2. 将这些符号以 `ParamDirection::In` 插入签名**最前面**。符号 Var 本身即成为参数
   —— `DynVar.unwrap()` 已将其构造为 `Scalar[INDEX]` Var，并由所有引用它的注解共享，
   因此一次插入即可绑定全部出现位置，无需重写类型。
3. 对该 kernel 的每个 `Call` / `Submit`，从实参在对应声明位置的 `valid_shape` 中读取
   取值并前置到参数列表。
4. 若调用点的 `arg_directions` 已解析，则同步前置对应的 `ArgDirection::Scalar`。

该 pass 在 `Default` 策略中最后运行：它只扩展签名与调用实参列表，而此时两者均已定型，
因此后续 pass 无需感知新增参数。

## 为何参数置于最前

符号正是被命名它的那个参数注解所读取。文本形式按从左到右声明参数，而 Python 在外层
作用域中求值注解，因此追加到末尾会打印出「先使用 `VALID`、后声明 `VALID`」的签名，
无法重新解析：

```python
# 无法重新解析：def 时抛出 NameError
def kernel(a: pl.Tensor[..., pl.TensorView(valid_shape=[M, VALID])],
           VALID: pl.Scalar[pl.INDEX]): ...
```

前置放置解决了顺序问题。除此之外签名顺序是自由的：PTOParam 按
`[tensors..., scalars...]` 分发实参，与签名顺序无关（参见
`PTOCodegen::GenerateFunction`）。

另有两条配套规则保证打印结果可往返（round-trip）：

- Python printer 会为「被参数 **valid_shape** 读取的**参数**」保留 `pl.dynamic()`
  声明，使注解在 def 时可解析。而 valid_shape 中的函数体局部变量、以及被当作**物理**
  维度读取的参数，仍不生成声明（issue #854）。
- Parser 在参数声明后立即将该 `DynVar` 重新指向该参数，使后续注解读取到参数的 Var，
  而不是另一个同名且未绑定的 Var。

## 绑定规则及其边界

符号按**位置**绑定：声明位置必须单独命名该符号，调用点从实参 `valid_shape` 的同一
位置读取取值。

| 声明 | 实参 | 结果 |
| ---- | ---- | ---- |
| `valid_shape=[Q, VALID]` | `valid_shape=[16, valid_len]` | `VALID := valid_len` |
| `valid_shape=[Q, VALID * 2]` | `valid_shape=[16, n]` | 拒绝 —— 无法求逆 |
| `VALID` 出现在两个参数中且实参不一致 | — | 拒绝 —— 一个符号两个取值 |

复合表达式选择拒绝而非求逆：错误的有效范围会静默地读写错误区域。解决办法是在某个参数的
`valid_shape` 中单独命名该符号，或将其作为 `pl.Scalar[pl.INDEX]` 参数传入并在
`pl.load(..., valid_shape=[...])` 中使用。

## 结果

```mlir
func.func @softmax_prepare(%arg0: !pto.ptr<f32>, %arg1: !pto.ptr<f32>,
                           %arg2: index, %arg3: index, %arg4: index) {
  %0 = arith.minsi %arg2, %c128_index : index      // %arg2 == VALID
  %t = pto.alloc_tile addr = %c0_i64 valid_row = %c16_index valid_col = %0 : ...
```

orchestration 在下发任务时传入调用方的取值：

```cpp
params_t0.add_scalar(valid_len);
```

## 兜底检查

若仍有符号未绑定就到达 codegen（例如自定义 pass 列表省略了本 pass），
`PTOCodegen::GetVarName` 会抛出可操作的 `ValueError`，指明符号名及其来源参数，
绝不会输出空操作数。

## 参见

- [44-materialize_dist_tensor_ctx.md](44-materialize_dist_tensor_ctx.md) ——
  针对 `CommCtxType` 的同类「签名 + 调用点」改写
- [00-pass_manager.md](00-pass_manager.md) —— pass 顺序
