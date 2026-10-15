# 结构比较

`structural_equal`、`structural_hash` 和 `assert_structural_equal` 使用唯一的精确结构语义，
不提供比较模式。`Type.__eq__` 和 `Type.__hash__` 使用同一实现。

```python
from pypto import DataType, ir

span = ir.Span.unknown()
x = ir.Var("x", ir.ScalarType(DataType.INDEX), span)
y = ir.Var("y", ir.ScalarType(DataType.INDEX), span)
assert not ir.structural_equal(x, y)
assert ir.structural_equal(x, y, enable_auto_mapping=True)
assert ir.structural_hash(x, True) == ir.structural_hash(y, True)
ir.assert_structural_equal(x, y, True)
```

## 相等契约

| 字段 | 比较方式 |
| ---- | -------- |
| 源码位置、注释、函数名、绑定变量名 | 通过 `IgnoreField` 忽略 |
| 定义与引用 | 一致地重命名绑定变量，保留别名关系 |
| 自由变量 | 保留身份，除非设置 `enable_auto_mapping=True` |
| 标量 dtype | 精确比较原始编码，区分 `INDEX` 与 `INT64` |
| 浮点数 | 比较存储位，包括正负零和 NaN 的载荷、符号 |
| 算子 | 比较种类和名称，不依赖指针相同 |
| 属性 | 唯一键排序，保留值类型和向量顺序 |
| 类型与视图 | 比较所有反射字段，包括附带存储与可空偏移 |
| 函数元信息 | 方向、返回类型、属性、种类、阶段和运行时绑定标志 |
| Call 与 Submit | 种类不同，所有反射调度字段参与比较 |

`f(x) = x + x` 与 `g(y) = y + y` 相等；将第二个 `y` 改为另一个参数后不相等。
共享纯表达式子树不改变结构相等性。

```python
a = ir.ConstFloat(0.0, DataType.FP32, span)
b = ir.ConstFloat(-0.0, DataType.FP32, span)
assert not ir.structural_equal(a, b)
assert not ir.structural_equal(
    ir.ConstInt(1, DataType.INDEX, span),
    ir.ConstInt(1, DataType.INT64, span),
)
```

Function 在遍历签名前预留全部参数绑定，因此较早参数的形状可以引用较晚的标量参数。
物理参数形状中的符号和裸 `valid_shape` 槽位符号也属于绑定。
分类逻辑与 `MaterializeValidShapeSymbols` 共用，并遍历 tuple 构造与取项表达式。
其他开放引用保留自由变量语义。Program 中各函数的上下文独立；映射键和被调用函数名仍参与比较。

## 存储与未完成的 IR

支持 `MemRef`、`WindowBuffer`、指针和资源标记类型以及附带存储。
存储描述符保留分配基址引用、几何信息、槽位表达式与暂存标志。
不能仅因尺寸相同就合并不同的分配引用关系。UnknownType 可以与 UnknownType 比较，
但相等并不证明类型推断已经完成。

结构比较不验证 SSA、作用域合法性、闭包、副作用或某项优化的候选资格。
非法 IR 与自身比较可能相等。优化 pass 应先验证自己的前提条件。

```python
ir.verify_structural_bindings(function)
fingerprint = ir.structural_hash(function)
```

`verify_structural_bindings` 检查重复定义、定义前使用和局部变量越出词法作用域，
包括类型与属性内的引用。赋值在右侧求值后建立绑定，循环初值使用外层作用域，
区域结果在区域结束后绑定。分支和循环局部变量不能逃逸；直线 scope 定义可流出。
允许开放捕获。该验证器还检查签名符号资格并拒绝共享语句对象，
不能替代完整编译器验证或优化的副作用检查。

## 诊断与不支持的载荷

`assert_structural_equal` 在不相等时抛出含首个差异字段路径的 `ValueError`。
未知属性载荷类型、重复属性键、环或超过 512 层的节点/类型嵌套，
会抛出带 `reason` 和 `field_path` 的 `StructuralComparisonUnsupported`（TypeError 子类）。
无法表示与不相等是两种不同结果。绑定验证使用相同错误格式；
`BindingNotVisible` 指向实际非法使用，包括共享表达式或类型的后续越界使用。

## 实现与复杂度

反射描述符决定比较字段：`IgnoreField` 跳过元信息，`UsualField` 遍历结构，
`DefField` 建立绑定。`src/ir/transforms/structural_comparison.cpp` 中的统一投影
提供相等、哈希和诊断：先预留绑定身份，再创建包含精确标量和子记录编号的不可变记录。
记录按完整有序键驻留。相等判断比较共享 arena 中的编号，不比较哈希值。
哈希组合子节点哈希而非编号，因此独立调用结果一致。

缓存区分定义和引用，并属于单个函数上下文。表达式和类型 DAG 不展开为树。
设 N 为访问节点数加字段和字符串数据规模，投影时间复杂度为 O(N log N)，空间为 O(N)。
绑定验证通过 DAG 边传播使用区间，避免重复遍历共享子树。

哈希是进程内指纹，不是持久缓存键。哈希与相等必须采用一致的 `enable_auto_mapping` 策略，
合并候选前必须对哈希匹配结果再进行相等检查。

## 从宽松比较迁移

结构比较不再将整数语法 dtype 视为可互换、不再按数值比较浮点数，也不再忽略附带内存引用。
需要类型兼容判断的调用方应明确哪些字段与该操作无关。
打印器和解析器应保留精确 dtype，或由调用方在比较前显式规范化语法。
不要在结构比较器中加入兼容例外。元组注解会递归校验元素的类型种类、形状和数据类型。
省略的视图和内存空间从推导类型继承，带注解的循环结果也遵循此规则。
分配注解保持精确，包括未指定分配的情况。

## 无损文本往返

打印器保留非 INDEX 形状维度的数据类型及带类型的偏移表达式。带存储信息的调用使用
私有语法 `pl._dump.call` 和 `pl._dump.submit`，以独立于算子或被调用函数的类型推导，
保留已记录的结果类型。多结果区域使用 `pl._dump.yield_(..., types=[...])`。
这些是解析器标记，不属于内核编写 API。往返检查显式映射开放引用，因为重建会创建
新的对象；定义关系与存储别名仍然参与精确比较。

函数内仅通过存储描述符引用的分配基址属于隐式资源绑定。MemRef 的身份由基址及描述符
字段决定，而非描述符对象的地址。WindowBuffer 保留资源绑定及全部存储字段。因此，
重命名或序列化后的函数可以相等，同时仍保持分配与窗口的别名关系。
