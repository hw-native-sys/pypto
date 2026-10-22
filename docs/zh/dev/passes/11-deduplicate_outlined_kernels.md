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
普通 InCore kernel 的 physical shape 符号身份不同时，若每个受影响的 Call/Submit
都能一致地绑定规范函数签名，并恢复原调用者坐标中的结果类型，则可以合并。
支持 Spmd 包装函数内的调用、同时作为维度的 scalar 参数、重复符号，以及
`N * 128` 等同构表达式。绑定只来自已提供的实参，不能由尚未分配的输出反推。
形状符号身份不同的候选必须提供所有参数，包括 Submit 输出；运行时合成输出仍直接读取
callee 签名，尚未使用调用点的形状绑定。签名身份相同的合并保留已有的省略输出行为。
不进行代数反解，也不增加运行时形状检查；返回别名和 void 函数的内存副作用保持不变。

首阶段支持 scalar 参数和不带存储的普通 Tensor，可携带仅描述 layout 的 TensorView
（包括 NZ 权重）。stride、valid_shape 元数据、分布式存储、未支持的表达式及
Group 单元仍要求精确的签名符号身份。新增证明将参数数、返回数及不同形状表达式
节点数分别限制为 256；更大的签名保留原路径。任一调用证明失败，保留整个候选，
不会只删除部分调用所引用的定义。

绑定验证与闭合性检查覆盖 type、shape、view 和 attrs。自由捕获、未确定的值类型、
显式存储或通信资源、external/builtin kernel、runtime binding、未支持的 helper 图
保守跳过。普通内存副作用允许存在，因为每次执行及实参都保留。若未知 payload
使完整引用索引无法建立，则保留整个 Program；debug 日志记录投影诊断。

以下两类例外经过严格检查后可以参与去重：

- 旧版无返回值调用仅在直接作为 `EvalStmt` 结果时允许携带 `UnknownType`。
  白名单为 `pld.tensor.put`、`pld.tile.remote_store`、`pld.tensor.remote_store`、
  `pld.system.notify`、`pld.system.wait`、`pld.system.defer_wait`、
  `system.set_ffts`、`system.sync_set` 和 `system.sync_wait`。
  签名、实参、返回值或 metadata 中的 unknown 值仍不准入。
- Prefetch context、event 和 session 必须在函数内部由对应的 `prefetch`
  操作产生和使用；wait 必须使用 event 原始 context 的 session。
  句柄不能通过签名、metadata、别名或任意消费者逃逸。
  每次调用仍独立创建 context。

准入检查覆盖表达式的每条入边（包括共享 DAG 节点）以及嵌套类型。
这些例外不放宽结构等价或适用的签名证明。

## 重写与复杂度

Call 返回类型保留调用者的符号空间。Submit 保留 deps、TaskId、core_num、同步、
predicate 和 early-resolve。Group 单元原子重写，最后检查不存在悬空引用。

按函数名稳定遍历。每个 hash 桶最多保存八个代表元，碰撞仍须精确比较。每次比较先
从新单元八倍权重的预算扣除双方投影权重；预算不足则保留原函数。私有投影检查复用
结构比较的 DAG memoization 和反射字段，权重包括 payload 字节。总时间 O(N log N)、
空间 O(N)，允许保守漏合并。

要求并保持 `SSAForm`、`SplitIncoreOrch`、`ClusterOutlined`、
`NormalizedStmtStructure`，不新增“所有函数唯一”的 property。

调用形状证明另受全局预算约束，预算等于 Program 的 inspection weight；每个调用点
最多执行 65,536 个匹配步骤，并缓存共享表达式对。预算耗尽时保留候选定义。

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
