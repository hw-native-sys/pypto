# Torch 代码生成（Torch Codegen）

## 概述

`torch_codegen` 将 PyPTO IR 直接生成为可 `exec()` 的 Python/PyTorch 脚本，用于调试与数值一致性验证。

与生产代码生成（PTO/Orchestration）不同，`torch_codegen` 的核心目标是：

- 快速复现 IR 语义
- 在 Python 环境中观察中间行为
- 为 Pass 调试和系统用例提供可执行参考

**源码位置：** `python/pypto/debug/torch_codegen.py`

**入口 API：**

```python
torch_codegen(node: _ir.Program | _ir.Function, check_shapes: bool = False) -> str
```

## 设计目标与边界

### 设计目标

- 保持 IR 到 Python 表达式/语句的可读映射
- 尽量覆盖 tensor/tile 常见算子
- 支持 cross-core `tpush/tpop` 的并发语义模拟
- 对可疑输入提供可定位的错误信息（超时、split 维度不合法等）

### 非目标

- 不追求执行性能
- 不模拟 Ascend 真实硬件时序和内存模型
- 不作为训练/推理生产后端

## 整体架构

生成结果由三部分拼接而成：

1. **运行时前导（`_PREAMBLE`）**
   提供 helper 函数、tile/tensor 边界处理、cross-core 运行时和并发调度器。
2. **Group 元信息注入（`_GROUP_META.update(...)`）**
   仅当输入是 `Program` 且存在 Group/AIC/AIV 配对函数时注入。
3. **函数体代码**
   `TorchCodegen(IRVisitor)` 逐个函数、逐条语句发射 Python 代码。

数据流：

`Program/Function IR -> (可选 _build_group_meta) -> TorchCodegen 访存发射 -> 拼接成字符串脚本`

## 表达式与语句生成模型

### 表达式层

- `_OP_MAP` 将 `op_name` 映射到 `OpHandler(args, kwargs) -> str`
- `_visit_expr_str()` 强制嵌套 `Call` 走 Python 侧 `visit_call`，避免底层 C++ 访问器在嵌套调用时丢失表达式
- 对二元/一元 IR 节点通过 `_BINARY_OP_STR` 与统一 visitor 适配方法生成

### 语句层

- `AssignStmt`：`var = expr`
- `EvalStmt`：发射表达式本体（用于副作用调用）
- `ReturnStmt`：支持单返回与多返回 tuple
- `ScopeStmt`：透明展开
- `ForStmt`/`WhileStmt`：将 SSA `iter_args + yield` 降为可变变量更新
- `IfStmt`：分支内通过 `yield` 写回 `return_vars`

## 命名与函数隔离

变量命名通过 `_unique_name()` 做三类规整：

- 非标识符字符替换为 `_`
- 连续下划线折叠
- 关键字与数字开头规避

`visit_function()` 会重置 `_var_names/_name_counter/_yield_targets`。这是为了避免 Program 多函数场景下对象 id 复用导致的名字串扰。

## 算子映射系统（`_OP_MAP`）

算子映射按类别注册：

- Tensor/Tile 通用逐元素、广播、规约、逻辑和位运算
- `matmul/matmul_acc` 及其 tile 变体
- `create/full/cast/slice/read/write/assemble/fillpad`
- cross-core 管道操作
- `system.*` 操作（调试场景下按 no-op 处理）

### cross-core 相关映射

- `tile.tpush_to_aiv` -> `_cross_core_rt.push_to_aiv(tile, split)`
- `tile.tpush_to_aic` -> `_cross_core_rt.push_to_aic(tile, split)`
- `tile.tpop_from_aic` -> `_cross_core_rt.pop_from_aic(split)`
- `tile.tpop_from_aiv` -> `_cross_core_rt.pop_from_aiv(split)`
- `tile.get_subblock_idx` -> `_get_subblock_idx()`

`split` 参数统一通过 `_split_mode_to_int()` 归一化为整数：

- `0`: NONE
- `1`: UP_DOWN
- `2`: LEFT_RIGHT
- 非法或不可解析取值：抛出 `ValueError`（fail fast）

## Tile/Tensor 边界与有效区语义

`_PREAMBLE` 中的 helper 负责调试执行时的边界行为：

- `_tile_load`：越界时按请求形状补零，并记录有效区
- `_tile_store`：仅回写有效区
- `_tensor_slice`：越界时物化到请求形状
- `_fillpad`：按 `zero/min/max` 填充无效区域
- `_assemble`：按偏移将 source 的有效区拼回 target

有效区通过两个动态属性传播：

- `_pypto_valid_shape`
- `_pypto_full_shape`

Cross-core 的入队/切分/合并路径同样会保留这两个属性，确保边界 tile
在 `tpush/tpop` 过程中不丢失有效区语义。

## Cross-Core 运行时设计

### 结构

`_CrossCoreRuntime` 基于 `threading.Condition` 实现阻塞队列语义，维护：

- 常规通道：
  - 发往 AIV：`_to_aiv` 与 `_to_aiv_split[split][lane]`
  - 发往 AIC：`_to_aic` 与 `_to_aic_split[split][lane]`
- no-split 双分发通道：
  - 发往 AIV：`_to_aiv_dual_nosplit[lane]`
  - 发往 AIC：`_to_aic_dual_nosplit[lane]`

同时提供：

- `reset(no_split_dual_aiv_dispatch=False)`：每次 Group 调度前清空通道，并配置 no-split 双分发模式
- `snapshot()`：输出当前队列深度，用于超时诊断，包含 no-split 双分发模式与 dual-nosplit 队列深度

### push/pop 语义

### `push_to_aiv(tile, split)` / `pop_from_aic(split)`

- `split=0` 且 `no_split_dual_aiv_dispatch=False`：单队列传输
- `split=0` 且 `no_split_dual_aiv_dispatch=True`：
  - `push_to_aiv` 会把同一 tile 复制后广播到 lane0/lane1 两路队列
  - `pop_from_aic` 按当前 lane (`lane in {0,1}`) 从对应队列消费
- `split=1/2`：push 时按 split 维度切成 lane0/lane1 两块；pop 按当前 lane 消费

### `push_to_aic(tile, split)` / `pop_from_aiv(split)`

- `split=0` 且 `no_split_dual_aiv_dispatch=False`：单队列传输
- `split=0` 且 `no_split_dual_aiv_dispatch=True`：
  - `push_to_aic` 按当前 lane (`lane in {0,1}`) 分路入队
  - `pop_from_aiv` 等待两路都就绪后成对消费，并返回 lane0 的 payload
- `split=1/2`：push 按当前 lane 入队；pop 端等待两路齐备后 merge 返回

split 维度定义：

- `split=1`（UP_DOWN）：按 `dim=0` 切分/拼接
- `split=2`（LEFT_RIGHT）：按 `dim=1` 切分/拼接

### 同步与超时

- 单次 `pop` 等待超时：`_PIPE_WAIT_TIMEOUT_SEC`（默认 10s）
- Group 混合核整体等待超时：`_MIXED_KERNEL_TIMEOUT_SEC`（默认 30s）
- Group 超时/失败时，运行时会发出取消信号并唤醒全部等待线程，
  避免阻塞线程污染后续调度

超时异常会包含：

- 操作名、split、lane
- 活跃线程名（Group 超时）
- 当前 pipe 快照（队列深度）

## Group 混合核并发调度

### 元信息构建（`_build_group_meta`）

对 Program 扫描 `FunctionType.Group` 函数，并按命名约定配对：

- `<group>_aic`
- `<group>_aiv`

元信息字段：

- `aic` / `aiv`：对应函数名
- `split`：优先取 Group 的 split；若为 0，则回退 AIV 的 split
- `dual_aiv_dispatch`：从 AIV attrs 读取；当 `split==0` 时，该标记会强制 debug 运行时按双 AIV lane 调度

### 调度入口

`visit_call()` 遇到 `GlobalVar` 调用时：

- 若命中 `_group_meta`，改写为 `_run_group_call(group_name, *args)`
- 否则保持普通函数调用

`_run_group_call()` 在有 meta 时走 `_run_mixed_kernels()`。

### 线程模型

`_run_mixed_kernels(group_name, meta, *args)` 行为：

- 固定 1 个 AIC 线程
- AIV 线程数：
  - `split in (1,2)` -> 2 lane
  - `split == 0 and dual_aiv_dispatch == True` -> 2 lane
  - `split == 0 and dual_aiv_dispatch == False` -> 1 lane
- 每线程写入 thread-local `subblock_idx`（AIC 设为 0，AIV 为 lane id）
- 运行时模式切换：
  - 调度器调用 `reset(no_split_dual_aiv_dispatch=(split == 0 and dual_aiv_dispatch))`
  - 仅在 group meta 指定时启用 no-split 双 lane pipe 语义
- 返回值约定：仅采集 `aiv lane0` 的返回作为 Group 返回值；
  若其他 lane/role 产生非 `None` 返回，会抛出返回契约错误

## 形状/类型检查（`check_shapes`）

开启 `check_shapes=True` 时：

- 对函数参数：只检查 dtype，不检查 shape
  - 原因：InCore 参数在边界 tile 场景可能是部分数据
- 对赋值目标：检查 tensor 类型与 dtype；shape 按静态/动态维分别检查

动态维策略：

- 若并非全静态 `ConstInt`，则检查 `ndim` + 静态维位置值

## 错误处理与可观测性

主要错误类型：

- `TypeError`：入口 node 类型错误
- `ValueError`：不支持的 op、非法 split、lane 不合法、split 维度不可二分
- `RuntimeError`：pipe 等待超时、混合核线程失败/超时

并发失败时保留首个线程 traceback，便于快速定位具体函数和 lane。

## 当前限制

- `system.*` 操作按 no-op 处理，仅保留语义占位
- Group 配对依赖命名约定 `<group>_aic/_aiv`
- `split=1/2` 依赖对应切分维度可被 2 整除
- Group 返回值固定取 AIV lane0
- 该运行时为单机 Python 语义模拟，不等价于硬件流水线时序

## 测试建议与覆盖点

建议至少覆盖以下维度：

- 三类 split：`NONE/UP_DOWN/LEFT_RIGHT`
- 三类方向：`V->C`、`C->V`、双向 `V<->C`
- `tile.get_subblock_idx` lane 语义
- `split=0 + dual_aiv_dispatch` 下的双 lane 行为（`lane0` + `lane1`）
- Group 调度路径是否改写为 `_run_group_call`
- 超时和异常路径（必要时通过构造不配对 push/pop）

可参考：

- `tests/ut/debug/test_torch_codegen.py`
- `tests/st/codegen/test_torch_codegen_cross_core.py`
- `tests/st/codegen/test_torch_codegen_qwen3_decode_scope3_mixed.py`

## 与其它代码生成文档的关系

- 本文档：Python 调试代码生成（`torch_codegen`）
- [00-pto_codegen.md](00-pto_codegen.md)：PTO 内核代码生成
- [01-orchestration_codegen.md](01-orchestration_codegen.md)：编排侧 C++ 代码生成

三者职责互补：`torch_codegen` 负责“可执行语义对照”，PTO/Orchestration 负责“生产链路生成”。
