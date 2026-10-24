# 寄存器操作

寄存器操作在 A5（Ascend950）的非嵌套 `pl.reg_scope()` 中表达逻辑向量与掩码。
前端允许尚未特化的 InCore 函数；`ExpandMixedKernel` 后保留的区域必须属于 AIV
函数。显式配置非 A5 目标时拒绝；尚未选择目标时，目标检查延迟到编译配置阶段。

Opaque、Orchestration 或 Graph 函数中的显式 InCore 作用域在 outlining 前也提供
设备上下文，包括 graph 作用域作为祖先的情况。直接位于调度代码中的 RegScope
仍不合法，RegScope 内也禁止包含调度作用域；已有 InCore 作用域规则保持不变。

```python
@pl.function(type=pl.FunctionType.InCore)
def kernel(
    src: pl.Tile[[128], pl.FP32, pl.MemorySpace.Vec],
    dst: pl.Tile[[128], pl.FP32, pl.MemorySpace.Vec],
):
    with pl.reg_scope():
        x = pl.reg.load(src, offset=0, lanes=64)
        mask = pl.reg.active_mask(32, lanes=64)
        y = pl.reg.add(x, x, mask=mask, pmode="zero")
        out = pl.reg.store(dst, offset=0, value=y, mask=mask)
    z = pl.tile.add(out, out)
```

区域没有 `captures` 参数、结果列表或 yield。外部 Tile 和只读 Scalar 是隐式输入。
将局部值包装到元组 (tuple) 中也不能逃逸；函数边界检查同样递归检查嵌套元组类型。
Reg 与局部 Scalar 归属于定义它们的区域，禁止通过区域外使用、函数返回、循环结果
或向外部变量赋值而逃逸。Tile 输入必须源于区域外；内部只有 `reg.store` 能产生 Tile 结果。

每个寄存器操作使用 `Call` 和 `AssignStmt`，裸 store 被拒绝。
store 操作数顺序为 `(tile, offset, value, mask?)`，结果类型与目标 Tile 完全一致，
复用 operand 0 存储。结果是别名，不是快照或新分配，也不扩大 `valid_shape`。

## P0 schema

| 操作 | 输入 | 结果 |
| ---- | ---- | ---- |
| `load` | Vec Tile、元素 offset；`lanes` | VecReg |
| `store` | Vec Tile、元素 offset、VecReg、可选 MaskReg | 目标 Tile |
| `broadcast`、`index_gen` | Scalar；`lanes`、`dtype` | VecReg |
| `full_mask` | `lanes` | MaskReg |
| `active_mask` | 静态 active lane 数；`lanes` | MaskReg |
| `add/sub/mul/div/minimum/maximum` | 同型 VecReg、可选 MaskReg；`pmode` | VecReg |
| `convert` | VecReg；`dtype`、可选 `rounding`、`saturate` | VecReg |
| `compare` | 同型 VecReg、seed MaskReg；`predicate` | MaskReg |
| `select` | MaskReg、同型 true/false VecReg | VecReg |
| `mask_and/or/xor/not` | 同 lane MaskReg | MaskReg |

lane 数必须是静态正整数。Vec 输入的 dtype、signedness 和 lane 数必须完全一致，
不做隐式类型提升或 lane 广播。Mask 使用 BOOL，并与受控数据具有相同 lane 数。
算术 `pmode` 仅支持 `"zero"`，拒绝 `"merge"`；`index_gen` 顺序为 `"ascending"`。
比较 predicate 为 `"eq"`、`"ne"`、`"lt"`、`"le"`、`"gt"`、`"ge"`。

通用 dtype 集合为 FP16/BF16/FP32，以及有符号和无符号 8/16/32 位整数。
`mul` 不支持 8 位整数；`div` 仅支持 FP16/FP32；`compare` 支持
FP16/FP32/INT32/UINT32；`select` 不支持 BF16。已有 Scalar SSA 的 dtype 必须匹配
生成器要求。DSL 字面量按目标 dtype 归一化；原始 IR 和 Builder 字面量必须已具有目标 dtype。

转换白名单：FP16→FP32、FP32→FP16、FP32→INT32、INT32→FP32、INT8→FP16、
INT8→INT16、INT16→INT32、UINT8→UINT16、UINT16→UINT32、INT32→INT8。
其余转换拒绝。同 dtype 的 DSL 转换在没有属性时返回原值。
FP32→FP16 必须显式指定 `saturate`，舍入支持 R/A/H/Z，默认 R；
FP32→INT32 必须显式指定 `saturate`，舍入支持 R/A/F/C/Z，默认 Z。
INT32→INT8 要求 `saturate=False`，不允许舍入属性。其余转换均不接受这两个属性。

## 内存检查与效应

offset 以元素计，类型为 INDEX。静态 offset 必须非负，整个
`[offset, offset + lanes)` 区间必须位于 Tile 容量内，masked store 也不放宽容量要求。
load 还要求整个区间位于已初始化的 `valid_shape` 内；后续操作的 mask 不能保护
未初始化读取。一维 Tile 可直接使用；多维 Tile 必须具有显式连续行优先 View，
拒绝带行 padding、boxed 或 compact 布局。静态 View offset 计入绝对效应区间。

`ir.get_reg_op_spec(name)` 返回 scope 要求及未来 VMI emitter 名称。
`ir.get_reg_memory_effect(call)` 返回目标、已知时的根分配、Read/Write、绝对元素区间
以及可选 mask。动态 offset 产生未知端点，不会被当作空效应；它不能证明边界，
发射前需要后续分析。动态 offset load 要求完整 valid-shape 覆盖。
保留 mask 不代表整个缓冲区被初始化，也不证明区间互不重叠。

PR2 仅提供 schema 效应与前端验证。SSA 传播、跨语句初始化跟踪、别名调度与 VMI
发射属于 PR3。无论是否安装 PTOAS，当前 PTO codegen 都会对 RegScope 报告稳定的
后端不支持错误。

## Builder

Python 提供 `with ib.reg_scope():`、`ib.emit_reg(name, op_name, args, kwargs)` 和
`ib.emit_reg_store(name, tile, value, offset=0, mask=None)`。
C++ 提供 `BeginRegScope`、`EmitReg`、`EmitRegStore`、`EndRegScope`。
两者与 DSL 共享 schema 推导，完成函数时检查区域归属。
`RegScopeValid` 属性通过标准 verifier 覆盖手工 IR；
`MixedKernelExpanded` 检查最终 AIV placement。

Python 文本语法不接受区域元数据。打印带非空 `name_hint` 或 `attrs` 的手工 RegScope
会报错，避免静默丢失字段；二进制序列化保留这些字段。空区域打印为 `pass` 块。
