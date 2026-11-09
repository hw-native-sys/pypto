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
