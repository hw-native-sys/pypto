# DeduplicateOutlinedKernels

在 tensor pipeline 的 `OutlineClusterScopes` 后、`ConvertTensorToTileOps` 前执行。
它在同一 Program 内复用函数定义，保留每个 Call 和 Submit：两个等价 kernel 合并为
一个定义，但仍执行两次。

```python
# Before
x1 = self.step(x0)
x2 = self.step_copy(x1)
# After
x1 = self.step(x0)
x2 = self.step(x1)
```

## 候选与等价关系

候选是有引用的闭合 InCore 函数或独占 Group 单元。无引用函数可能是暴露入口，始终
保留。Orchestration、Graph 和 Spmd 保留身份。先索引所有 Group 成员，成员被外部
调用或跨 Group 共享时，整个单元及其成员都不合并。单元包含完整 Group 参数桥接和
按首次使用顺序排列的直接 InCore/AIC/AIV 成员。内部 callee 和已知
`system.import_peer_buffer(peer_func=...)` 字段一起规范化。

使用唯一的精确结构 hash/equality，`enable_auto_mapping=False`。绑定参数和签名
符号按角色映射；语义 attrs、方向、返回关系、dtype、layout、常数位模式保持有效。
合并另要求根函数及每个 Group 成员的签名类型精确相等，保留 shape/view 符号身份。
这项限制独立于结构等价：tensor-to-tile lowering 直接用 callee 签名分配输出，
不会将其中符号映射到每个调用的符号空间。因此 `[N]` 和 `[M]` kernel 即使
alpha 等价也不合并；共享相同符号的 kernel 仍可合并。仅保留原调用的返回类型
不足以保证后续输出分配正确。
绑定验证与闭合性检查覆盖 type、shape、view 和 attrs。自由捕获、unknown type、
显式存储或通信资源、external/builtin kernel、runtime binding、未支持的 helper 图
保守跳过。普通内存副作用允许存在，因为每次执行及实参都保留。若未知 payload
使完整引用索引无法建立，则保留整个 Program；debug 日志记录投影诊断。

## 重写与复杂度

Call 返回类型保留调用者的符号空间。Submit 保留 deps、TaskId、core_num、同步、
predicate 和 early-resolve。Group 单元原子重写，最后检查不存在悬空引用。

按函数名稳定遍历。每个 hash 桶最多保存八个代表元，碰撞仍须精确比较。每次比较先
从新单元八倍权重的预算扣除双方投影权重；预算不足则保留原函数。私有投影检查复用
结构比较的 DAG memoization 和反射字段，权重包括 payload 字节。总时间 O(N log N)、
空间 O(N)，允许保守漏合并。

要求并保持 `SSAForm`、`SplitIncoreOrch`、`ClusterOutlined`、
`NormalizedStmtStructure`，不新增“所有函数唯一”的 property。

## 代码与注册身份

`OrchestrationResult.kernel_registrations` 取代旧的 `func_name_to_id`、
`func_name_to_core_type` 和 `func_name_to_signature` maps。每条 `KernelRegistration`
包含 `func_id`、`code_name`、`core_type`、`signature`。最终 task params 排序完成后，
按代码、core type 和 tensor signature 分配 ID。Scalar 不进入 tensor signature；
NoDep 映射为 runtime INOUT，但该 task 仍使用 `add_no_dep`。相同 signature 共用 ID，
不同 signature 可使用不同 ID 引用同一份代码。

配置中的每条 KERNELS 对应一个注册。运行时按 generated code 和编译契约只编译一次，
然后按各自 signature 构建 callable。External、builtin template 和旧配置保留独立编译身份。
Generated cache key 包含源码及头文件内容，binary context 失效检查仍生效。
Prebuilt schema 1 允许多个注册引用同一校验过的 binary 文件，重复
func_id 仍非法。二进制复用不表示设备加载地址共享。
