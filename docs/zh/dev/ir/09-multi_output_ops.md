# 多输出算子 (Multi-Output Operators)

## 概述

有些硬件指令会产生多个结果。PTOAS `TGATHER` 的 compare 形式同时写出收集到的索引和每行的匹配计数，`pto.tgather` 把它们写作两个 `outs(...)` 操作数。包装这类指令的算子就是**多输出算子 (multi-output operator)**，PyPTO 用 `TupleType` 表达它的结果，绝不用目的参数 (destination argument)。

**规则**：注册时只写*输入*。每个输出都是 `f_deduce_type` 返回的 `TupleType` 中的一个元素。

参见 [算子系统](05-operators.md) 了解通用注册 API，[类型与示例](02-types.md#tupletype) 了解 `TupleType` 本身。

## 为什么目的地不能是参数

包装目的传递风格 (destination-passing-style, DPS) 指令时，最省事的写法是照抄硬件签名，把目的地声明成参数。这样错在两处：

```text
tile.gather_compare(src, kvalue, tmp, dst, cdst)   ← 泄漏
  → 调用方必须自己分配 dst 和 cdst
     但 tile 分配属于 InitMemRef，所有 tile buffer 都归它管
  → 这个算子看起来像 5 输入算子
     于是方向推导 (direction inference) 把 dst/cdst 读成消费者，写操作凭空消失
```

第二条后果才是危险的：没有分类的参数默认为 `ArgEffect::Read`，于是不会对读取该目的地的一方发出 RAW 边，故障会以设备上的陈旧数据形式出现，而不是在编译期报错。参见[算子系统](05-operators.md)的"参数效应"一节。

改用 `TupleType` 表达后，结果就是普通的 SSA 值：`InitMemRef` 像分配任何其他 tile 一样分配每个元素，`MemoryReuse` 把它们当作独立的复用候选，硬件的 DPS 形状完全不会传到用户面前。

## 注册

用 `set_output_arity(N)` 声明输出个数，并从 `f_deduce_type` 返回 N 元 `TupleType`：

```cpp
// ❌ 错误 —— DPS 目的地泄漏进了参数列表
REGISTER_OP("tile.gather_compare")
    .add_argument("src", "...")
    .add_argument("kvalue", "...")
    .add_argument("tmp", "...")
    .add_argument("dst", "...")     // 泄漏
    .add_argument("cdst", "...");   // 泄漏

// ✅ 正确 —— 只有输入；输出由推导出的 TupleType 承载
REGISTER_OP("tile.gather_compare")
    .add_argument("src", "Source tile (FP16/FP32/INT16/INT32, 2D)")
    .add_argument("kvalue", "Scalar threshold")
    .add_argument("tmp", "Workspace tile (UINT8)")
    .set_output_arity(2)
    .set_arg_effect(0, ArgEffect::Read)
    .set_arg_effect(1, ArgEffect::Read)
    .set_arg_effect(2, ArgEffect::Write)
    .set_workspace_arg(2)
    .f_deduce_type([](const auto& args, const auto& kwargs) {
      return std::make_shared<TupleType>(std::vector<TypePtr>{
          DeduceDstType(args, kwargs),
          DeduceCdstType(args, kwargs),
      });
    });
```

| 方法 | 用途 |
| ---- | ---- |
| `set_output_arity(N)` | 声明产生 N 个值。`N > 1` 表示推导结果是恰好 N 个元素的 `TupleType` |
| `set_workspace_arg(i)` | 声明参数 `i` 是编译器提供的暂存空间 (scratch)——硬件会写它，但不承载任何人读取的结果 |

`set_output_memory(space)` 会作用于 `TupleType` 内的**每一个** `TileType` 元素。若某算子的输出位于不同内存空间，必须在 `f_deduce_type` 内部设置 `memory_space_`，而不能依赖这一回退路径。

### 暂存空间与目的地

被写入的参数只可能是两者之一，注册必须说清是哪一个：

| 种类 | 示例 | 声明方式 |
| ---- | ---- | -------- |
| **暂存空间 (workspace)**——硬件 scratch，不承载任何人读取的结果 | `tile.gather_compare` 的 `tmp`，由 `ConvertTensorToTileOps` 合成 | `set_arg_effect(i, ArgEffect::Write)` + `set_workspace_arg(i)` |
| **目的地 (destination)**——调用方要读取的结果 | `dst`、`cdst` | 根本不是参数——它是 `TupleType` 的元素 |

这个区分不是修辞。暂存空间由合成它的 pass 分配且从不被读取，因此不需要 SSA 结果；目的地则是程序后续要使用的值。

## 注册表强制的规则

两道检查，都会大声失败，而不是让泄漏溜到设备上。

**import 期**——`OpRegistry::ValidateMultiOutputOps()` 遍历所有 arity > 1 的算子，拒绝三种形状：

| 被拒绝的情况 | 原因 |
| ------------ | ---- |
| 存在未声明效应的参数 | 默认的 `Read` 与藏起来的目的地无法区分 |
| 被写入却未声明为 workspace 的参数 | 它要么是必须自报家门的 scratch，要么是本该放进 `TupleType` 的目的地 |
| `set_workspace_arg(i)` 指向不存在的参数 | 越界的下标是一个笔误，它什么也保护不了 |
| 声明为 workspace 却从不被写入的参数 | scratch 按定义就是硬件会写的。把它声明成 `Read`——或者经由 `no_arg_writes()` 落到 `Read`——依然是那条被丢掉的依赖边，只是外面套了一个说法相反的标记 |
| `set_output_reuses_input(N)` | 有多个结果时，"输出复用输入 N"说不清是哪一个输出 |

这项检查是从参数列表出发的，所以它是靠目的地留下的痕迹来抓它——一处没人声明为 scratch 的写，或者一个根本没人分类的槽位。若某个目的地被声明成 `Read` 且从未标记 workspace，它不留下任何痕迹，因而能通过；拦住这一种的是上面的约定，不是注册表。

**建 call 时**——`OpRegistry::Create` 双向交叉校验声明的 arity 与推导出的类型：声明 N 就必须推导出 N 元 `TupleType`，推导出 `TupleType` 就必须先声明过。第二个方向同样重要，因为 codegen 从注册表读取 arity；没有声明过的 tuple 结果就没有 arity 可用来解析它的元素。

## 多输出调用在流水线中的流转

```text
DSL 包装器            dst, cdst = pl.tile.gather_compare(src, kvalue, tmp, ...)
                     返回 (Tile(TupleGetItemExpr(call, 0)),
                           Tile(TupleGetItemExpr(call, 1)))
        ↓
解析器脱糖            _tuple_tmp = tile.gather_compare(src, kvalue, tmp)
                     dst  = _tuple_tmp[0]
                     cdst = _tuple_tmp[1]
        ↓
InitMemRef           dst 和 cdst 是普通 TileType 变量，各自拿到自己的 MemRef；
                     _tuple_tmp 是 TupleType，不拿 MemRef
        ↓
MemoryReuse          tuple 临时变量把该 call 的 no-alias 输入传递到每个元素上；
                     每个元素都是独立的复用候选
        ↓
PTO codegen          PrepareTupleOutputs(op) 从 `<var> = _tuple_tmp[i]` 绑定中
                     找回元素变量并为它们分配空间
```

DSL 包装器为每个输出返回一个指向*同一个* call 的 `TupleGetItemExpr`；`python/pypto/language/parser/_dsl_invoker.py` 中的 `_unwrap_result` 通用地识别这一形状，并把裸 `Call` 交还给解析器重新绑定。

## 代码生成

多输出算子的 `Call` 不在 `args_` 里携带它的目的地——解析器把它们放进了单独的 `AssignStmt`——所以 emitter 需要去查：

```cpp
static std::string MakeGatherCompareCodegenPTO(const CallPtr& op,
                                               codegen::CodegenBase& codegen_base) {
  auto& codegen = AsPto(codegen_base);
  const auto outs = codegen.PrepareTupleOutputs(op);   // 解析 + 分配

  std::ostringstream oss;
  oss << "pto.tgather ins(" << /* ... 输入 ... */ ")"
      << " outs(" << outs[0].name << ", " << outs[1].name
      << " : " << outs[0].type_str << ", " << outs[1].type_str << ")";
  codegen.Emit(oss.str());
  return "";
}
```

`PrepareTupleOutputs` 从注册表读取 arity，解析出每个元素变量，检查它带有 `InitMemRef` 赋予的 MemRef，并**提前**发出它的 `alloc_tile`——指令会在那些本该负责分配的 `<var> = tuple[i]` `AssignStmt` 之前就写这些 buffer。发射是幂等的，因此那些语句随后会跳过重复发射。

元素绑定在进入函数时一次性建立索引 (`fs_.tuple_element_index`)，所以解析它们是一次 map 查表，而不是每个 call 都重扫一遍函数体。

## 新增一个多输出算子

1. 只注册输入；加上 `set_output_arity(N)`。
2. 用 `set_arg_effect` 对**每一个**参数分类；若该算子不通过任何参数写入，则用 `no_arg_writes()`。import 期检查要求每个参数都有明确结论。
3. 给任何被写入的参数标记 `set_workspace_arg(i)`——如果它不是 scratch，那它就是目的地，不该出现在参数列表里。
4. 从 `f_deduce_type` 返回 N 元 `TupleType`。
5. DSL 包装器返回一组指向同一个 call 的 `TupleGetItemExpr(call, i)`——解析器的解包路径认这个形状。
6. codegen emitter 使用 `PrepareTupleOutputs(op)` 编写。
7. 测试注册契约（`tests/ut/ir/operators/test_op_registry.py` 会自动发现任何新的 `set_output_arity(N > 1)`）以及端到端的 lowering。

## 参见

- [算子系统](05-operators.md) —— 通用注册 API 与参数效应
- [类型与示例](02-types.md) —— `TupleType` 及其余类型系统
- [参数方向](08-param-directions.md) —— 一处未声明的写是如何丢掉它的依赖边的
- [InitMemRef](../passes/34-init_memref.md) —— 拥有 tile 分配职责的 pass
- [MemoryReuse](../passes/36-memory_reuse.md) —— 跨 tuple 元素的生命周期复用
