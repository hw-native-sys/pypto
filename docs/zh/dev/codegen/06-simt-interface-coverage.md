# SIMT 接口覆盖

本矩阵跟踪任务 11.7，以固定的 PTOAS v0.74
（`be4c7a51f66ceadda6eb8a001d7bddccb3507700`）为基准。
接口声明不等于设备执行证据。通用无符号类型与单个操作对 INT32 位载体的无符号解释不同。

## 第一批：32 位接口

下列 API 同时存在于 `pl.simt` 和 `pypto.ir.op.simt`。
它们通过已注册的 SimtOp schema 和既有通用 binding 暴露，无需新增 C++ 节点或原生绑定签名。

| PTOAS 操作 / 参数 | PyPTO API | 类型 / 存储 | 状态 |
| ----------------- | --------- | ----------- | ---- |
| `atomic_sub` | `atomic_sub` | INT32/FP32，GM/UB | 已实现；设备验收待完成 |
| `atomic_and/or/xor` | 同名接口 | INT32，GM/UB | 已实现；拒绝 FP32 |
| `atomic_min/max unsigned` | `atomic_min/max(..., unsigned=True)` | INT32 位载体，GM/UB | 已实现；拒绝 FP32 unsigned |
| `redux_mini/maxi unsigned` | `redux_min/max(..., unsigned=True)` | INT32 位载体 | 已实现；拒绝 FP32 unsigned |
| 四种 `shuffle_*` 的 `bound` | `shuffle_*(..., width=32, bound=None)` | INT32/FP32，width 16/32 | 已实现；默认窗口不变 |
| `get_lanemask_eq/le/lt/ge/gt` | `lanemask_eq/le/lt/ge/gt()` | INT32 位载体 | 已实现；表示物理 lane 关系 |
| `bitcast` | `bitcast(value, dtype)` | FP32/INT32，允许同类型 | 已实现；保留位模式 |

原子操作沿用 ReadWrite 参数副作用和 Simt 写通道，返回旧值；丢弃返回值时保留更新，
验证 cache 属性并参与 launch 的存储版本追踪。
源窗口规则和非活跃源限制见 [IR 契约](../ir/11-simt.md)。

2026-11-19 验证：独立 worktree 的 C++ 构建通过；SIMT 算子、原子回归、codegen、
公共 JIT、作用域提取和存储准备共 **1,124 项测试通过**。测试要求 PTOAS v0.74，
包括两个新增验收入口的 LLVM 导出，以及四种 shuffle 在 INT32/FP32、width 16/32
和两个窗口边界下的编译。Python lint、实现类型检查、文档双语与导航检查通过。
不宣称设备执行验收通过。

验证入口：

- `tests/ut/ir/operators/test_simt_ops.py`：schema、类型/属性错误、序列化、包装层一致性和原子读写副作用。
- `tests/ut/jit/test_simt_scope.py::test_first_batch_simt_interfaces_native_compile`：
  公共 JIT 完整降低和固定 PTOAS LLVM 导出，覆盖 GM/UB、无符号比较、窗口属性和旧值未使用的原子更新。
- `tests/st/runtime/vpto/test_simt_interface_gaps_onboard.py`：
  可选数值验收，覆盖无符号最高位边界、旧值、位模式和窗口；编译成功不代表设备执行成功。

当前主机的 NPU 探测返回 DCMI -8005。新增接口的设备证据仍待完成，
包括原子竞争、分歧中的物理 lane 掩码、FP32 减法特殊值及全部 shuffle 窗口组合。

## 剩余范围

| 类别 | 当前覆盖 | 剩余工作 / 限制 |
| ---- | -------- | --------------- |
| 坐标、维度、lane ID | 已有 DSL/IR/原生降低 | 无需重复实现 |
| `get_veccoreid`、`get_clock32/64` | 缺失 | 保留时钟采样的优化语义；clock64 依赖 INT64 |
| FP32 算术/数学 | 已有算术、数学和显式转换 | 显式高精度 `divf`、截断余数 `remf` |
| 非 packed FP16/BF16、INT64、无符号类型 | 尚非通用可执行类型 | 逐操作存储/计算/协作/原子矩阵；UB i64 原子仍禁止 |
| `ftof` 和其他 `ftoi/itof` 组合 | 已有 FP32↔INT32 转换 | 每个新增转换对的舍入/饱和验证 |
| `mul_i32toi64` | 缺失 | INT64 结果及有符号/无符号操作数解释 |
| 运行时 trap/assert | 缺失 | 目标/runtime 终止与宿主错误报告 |
| `keep/resume` | 缺失 | slot、序言/尾声、相邻 launch 和线程状态生命周期 |
| `addui_extended`、`mul_extended` | 无直接多结果 API | 现有算术/位运算和 `mulhi` 可表达部分能力，不宣称完整类型覆盖 |
| `ceildiv`、`select` | 支持的标量可由算术/控制流组合表达 | 缺少同名 API 不等于缺少计算能力 |
| packed 与短向量 | 任务 11.6 暂缓 | 不是 11.7 的前置条件 |

## 工具链边界

保持 PTOAS v0.74。升级 v0.77 时需要同步适配内存 offset 类型，
以及分歧控制流下 vote/shuffle/redux 的活跃 lane 副作用回归，不能只修改版本号。
