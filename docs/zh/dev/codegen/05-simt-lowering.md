# SIMT lowering 与执行

公开的 `simt_scope`/JIT 入口先提取（outline）为 `FunctionType.SIMT` helper
和 `SimtLaunchStmt`，再接入正式 A5 编译器及持久运行时（persistent runtime）。
详见[规范 SIMT IR](../ir/11-simt.md)。

## 可执行子集

- A5 AIV 调用方，由 `SpecializeSimtHelpers` 特化常量三维 block 大小，最多 2,048 线程。
- 借用的 GM Tensor 和 UB Vec Tile，元素类型为 FP32 或 INT32。Tile 需要仿射、
  非分形布局。寻址使用物理 stride 和 Tile 起始偏移，逻辑 `valid_shape` 不替代物理 stride。
  借用的 `tile.slice`/reshape 视图句柄会被拒绝，因为其类型可能缺少父存储的物理行跨度。
  Launch 前须通过 `tile.extract` 将窗口物化到独立存储。视图来源会沿别名、条件分支结果
  和循环携带值传播，包括循环回边；内存规划插入的 `tile.move` 不能证明存储独立且稠密。
  原地 `tile.assemble` 的结果继承目标存储的来源：更新稠密存储仍受支持，更新视图仍会被拒绝。
  受支持存储的显式 stride 优先；
  稠密 DN Tensor 的后备 stride 仅交换末尾两个轴，批次轴保持在最外层。
- FP32、INT32、INDEX、BOOL 标量及结构化 `if`、`for`、`while`。
  有符号整数整除和取模保留向下取整语义；FP32 不等比较将 NaN 视为不等；
  数值转 BOOL 通过与零比较实现。
- 线程、block、grid 坐标和维度，lane ID，以及标量读写。
  GM 缓存策略原样传递到原生 PTOAS 操作。

无符号运算、其他存储类型、标量幂运算（`**`）、浮点向下整除/取模、提前 return、break/continue、
分形 Tile 布局和分组内核执行目前会给出明确诊断。CPU 仿真（`a5sim`）和其他
设备后端不能执行这些原生 helper。与规范 IR 一致，调用方必须保证索引不越界，
并发写入没有数据竞争。

## ABI 与同步

Codegen 拆分编译单元时保留可达 helper，不将其注册为独立运行时任务。
每个 helper 产物包含原生 SIMT 入口和导出的 AIV launch 包装函数。标量按值传递，
Tensor 使用 GM 指针，Tile 使用 UB 指针。特化得到的线程/寄存器限制写入
`pto.simt_max_threads` 和 `pto.simt_max_regs`。

运行时 Tensor 指针已包含视图起点。Tile 指针指向分配基址，因此原生函数体仅添加
一次 `TileView.start_offset`。特化捕获的动态元数据仍作为标量 ABI 参数。
相同特化 helper 和调用方类型的重复 launch 共享产物。

调用方保留带完整存储 read/readwrite 效果标注的 `simd` 适配函数。部分写入仍为
readwrite，使未写元素及此前生产者保持依赖。MTE/V 依赖和分配复用同步由 PTOAS
InsertSync 负责，PyPTO 不在 launch 周围添加固定屏障。Launch 结果别名指向
`MaterializeSimtOutputs` 选择的可写句柄；原 Tile 值保持独立，GM 结果保留 Tensor 身份。

## 原生编译与产物

复用 VF 产物链路，在清单中以 `kind: simt` 标识 helper。产物包含相对 helper 路径，
经过 PTOAS LLVM 导出、Bisheng 设备编译，以 `kernel_entry` 为首进行链接、验证 ELF，
再将 `.text` 交给持久运行时。Helper 内容和编译器身份参与二进制缓存，目录迁移
不依赖原 helper 路径。

LLVM 导出绕过 PTOAS 对象级 SIMT 大小修复。当前子集使用内建指令（intrinsics）；
包含内联汇编的 SIMT 模块在设备编译前被拒绝。验收测试比较链接后 `VF_SIMT`
大小与被调用符号大小。PyPTO 不修改设备指令字节。

## 定向验收

`tests/st/runtime/vpto/test_simt_ir_onboard.py` 现通过 `.compile()` 编译包含
`pl.simt_scope` 的公共 `@pl.jit.aiv` 函数，不替换生成的源码，取代此前 P3 仅使用
规范 IR 编写测试的例外。

- GM copy 覆盖非零起点 Tensor 子视图、保护区域、非活动元素，以及消费 SIMT
  结果的后续 Tile load。
- UB gather 覆盖二维线程、INT32 索引、向量生产者、部分写入和原 Tile 后续读取。
- 两者均检查冷编译、热缓存和目录迁移。每阶段七种长度、三组输入，共 126 个运行用例。

公共 JIT 版本已在 Ascend950DT_9581、PTOAS v0.72、CANN 9.3.0 和固定运行时
`391b2fbcd4dd053204a677662c39b6b7636e8043` 上通过全部 126 项数值检查。
所有产物在当前 checkout 构建；执行前通过主机设备队列预留设备：

```bash
PYPTO_RUN_VPTO_ONBOARD=1 PYPTO_VPTO_DEVICE_ID="$DEVICE_ID" \
PYPTO_VPTO_TIMEOUT=600 python -m pytest \
  tests/st/runtime/vpto/test_simt_ir_onboard.py --platform=a5 -k persistent -v
```

`build` 变体使用 `PYPTO_RUN_VPTO_VF_VALIDATION=1`，不执行设备任务。
若主机 pytest 在仅编译模式下也要求可见设备节点，可在已有输出目录上直接调用构建 worker：

```bash
python tests/st/runtime/vpto/test_simt_ir_onboard.py --worker build/simt-ir-runtime gm
python tests/st/runtime/vpto/test_simt_ir_onboard.py --worker build/simt-ir-runtime ub
```

## Warp 协作扩展

支持活跃 lane 的 vote/ballot、INT32/FP32 shuffle（width 为 16 或 32），以及
INT32/FP32 的 add/min/max redux；整数 min/max 显式使用 signed 比较。
参见[协作操作契约](../ir/11-simt.md#warp-内协作操作)。

Warp 验收与上述 GM/UB 历史验收分开记录。
`tests/st/runtime/vpto/test_simt_collectives_onboard.py` 使用相同环境开关提供编译和
设备执行模式；JIT 单元测试还会在无需设备的情况下将所有协作形式经原生 PTOAS 导出为 LLVM。

协作操作验收已在 Ascend950DT_9581、PTOAS v0.72、CANN 9.3.0 和运行时
`391b2fbcd4dd053204a677662c39b6b7636e8043` 上通过。覆盖两个 warp 中的全部
七种活跃 lane 数量（0、1、15、16、17、31、32），检查全部 11 个 API、
INT32/FP32 载荷以及 16/32 宽度，包括分支中的参与集合和非活跃输出保持不变。

## 原子与同步验收

原生映射使用 `pto.atomic_add/min/max/exch/cas`、`pto.syncthreads`、
`pto.threadfence` 和 `pto.threadfence_block`。整数 min/max 显式携带 `signed`。
原子沿用标量访存的仿射元素偏移计算，再通过 `pto.addptr` 寻址；不暴露无类型字节
地址或绕过分配的路径。所有原子目标均为 ReadWrite，即使未使用旧标量返回值。
同步调用保持有序副作用，`no_arg_writes` 不表示纯函数。

| 能力 | 原生编译 | A5 数值覆盖 |
| ---- | -------- | ----------- |
| 五种原子 × INT32/FP32 × GM/UB | 全部 20 个组合，使用或丢弃返回值 | 竞争、旧值、CAS 成功/失败、未写元素 |
| INT32 Add/Min/Max/CAS 边界 | GM 与 UB | 回绕、有符号极值、条件替换 |
| FP32 Add/Min/Max/CAS 边界 | GM 与 UB | NaN、无穷、正负零、按位 CAS |
| `syncthreads` 与两种 fence | LLVM workitem 同步/fence intrinsic | 四个 warp 的 UB/GM 发布与读取，以及后续 MTE 输出 |

此矩阵已在 Ascend950DT_9581、PTOAS v0.72、CANN 9.3.0 及 runtime
`391b2fbcd4dd053204a677662c39b6b7636e8043` 上验证。原子位置为自然对齐的 4 字节
元素；UINT32/INT64、half/bfloat16、packed 值、可选顺序/scope 及跨 AIV 同步不属于
当前前端契约。详见 [API 语义与限制](../ir/11-simt.md#原子操作与同步)。
发布测试使用了会合，不能证明单独 fence 构成同步协议，也不承诺系统级可见性。

`tests/st/runtime/vpto/test_simt_atomics_onboard.py` 提供竞争和屏障用例（15 项检查），
`test_simt_atomic_edges_onboard.py` 提供边界用例（9 项检查）。两者均提供与其他
SIMT 测试相同环境开关控制的 `build`/`persistent` pytest 变体，及
`--worker OUTPUT [DEVICE]` 入口。JIT 单测在无设备时编译全部四个入口并导出原生 LLVM。
结构回归拒绝只读或未初始化的原子捕获、嵌套屏障以及绕过屏障的退出。
未验证无锁前进性保证。

`tests/st/runtime/vpto/test_simt_atomic_regressions_onboard.py` 额外验证五种原子对
`[2, 3, 32]` UB Tile 的三维坐标 `(1, 2, 7)`（物理元素 167）的更新、未写元素，
以及算术、分支和 while 条件中的原子求值次数。三项 A5 检查与原生 LLVM 地址和
调用保留断言配合；codegen 会拒绝与展平后存储秩不一致的索引数量。

条件回归覆盖被选元素前后的未选原子 tuple 元素，以及嵌套 tuple 投影。对于只执行一次循环体的循环，每个原子调用执行两次：条件为真时一次，最终条件为假时一次。

JIT 循环归纳变量 (induction variable) 在循环体中遮蔽之前的标量别名和已折叠的 shape 值。循环边界使用进入循环前的绑定，嵌套循环退出后恢复外层绑定。A5 回归将 while 计数器名称复用为后续复制循环的索引，以验证这一作用域区别。

`pl.while_` 使用 `for` 语法，但其目标全部是循环携带值 (loop-carried values)，没有归纳变量。赋值形式的 `pl.yield_` 左值定义循环后的输出，包括零次迭代的结果。循环体中遮蔽进入前的别名和已折叠 shape；仅当名称不是该循环的输出时，才恢复外层绑定。JIT 和 parser 共用 yield 归属规则：嵌套条件分支或真实循环中的 yield 属于对应结构。运行时作用域（`pl.scope` / `pl.manual_scope`）和 `pl.split_aiv` 是透明包装。A5 回归验证条件分支的 yield 不会替换外层标量别名。

## 标量数学与转换验证

显式标量接口降低为原生 `pto.absf/minf/maxf`、数学操作、`pto.prmt`、`pto.mulhi`
与 `pto.ftoi/itof`。转换显式发射 signedness 和 `round(r/a/f/c/z)`，`ftoi` 始终
发射 `sat`。首批矩阵覆盖 FP32 数学与 INT32 位载体，无需将工具链版本提升至 v0.72 以上。

`tests/st/runtime/vpto/test_simt_math_onboard.py` 通过公开 JIT 覆盖全部 18 个 API，
包括两种符号解释、五种舍入模式、饱和、NaN/Inf、负数整除/取模、融合算术和部分
活跃线程块。build/persistent 变体沿用上述环境开关与设备排队要求。
JIT 单元测试 `test_public_math_native_compile` 覆盖原生 LLVM 导出。
编译与设备执行属于不同的验证阶段。

LLVM 文本设备编译在调用 CANN Bisheng 前移除其不支持的 LLVM 内存效果属性，
与 PTOAS 对象文件生成路径一致；保留指令、同步和其他属性。
