# 手动 Buffer-ID 同步（仅 A5）

属于[算子系统](05-operators.md)参考文档。

这是 [#3268](https://github.com/hw-native-sys/pypto/issues/3268) 的手动 Buffer-ID 子集。

`pl.system.get_buf(buf_id, *, pipe, mode=0)` 和
`pl.system.rls_buf(buf_id, *, pipe, mode=0)` 为**同一核内**不同 pipe 对本地
buffer 的访问建立顺序。两者均为有副作用的无返回值语句（PyPTO IR 中为
`UnknownType`）。支持具体 pipe `MTE1`、`MTE2`、`MTE3`、`M`、`V`、`S`、`FIX`，
拒绝 `ALL`；应使用所属核上可用的 pipe。拒绝 A2/A3 目标和所有非零 mode。

Python 整数 ID 存为静态 attribute，范围为 `[0, 31]`，不接受 bool。INDEX
表达式（包括 INDEX 常量）存为动态 SSA operand，不接受其他标量 dtype。
kernel 作者须保证动态 ID 在运行期处于 `[0, 31]`、所有控制流路径上的 get/rls
配对，以及覆盖共享 buffer 的全部访问；编译器不插入运行期 ID 检查或协议验证。

```python
# Interface fragment only: surround the actual accesses with matching calls.
pl.system.get_buf(0, pipe=pl.PipeType.MTE2)
# Load into the protected local buffer here.
pl.system.rls_buf(0, pipe=pl.PipeType.MTE2)
pl.system.get_buf(buf_id, pipe=pl.PipeType.V, mode=0)  # buf_id: Scalar[INDEX]
# Consume the buffer here, then release it on the same pipe and ID.
pl.system.rls_buf(buf_id, pipe=pl.PipeType.V)
```

仅使用显式 `FunctionType.AIC` / `FunctionType.AIV` kernel。由于 MTE2 等 pipe
不能唯一确定所属核，InCore 和混合核归属会在展开前报错。默认 pipeline 保留
循环中的调用，并继续启用自动 event 同步。T1 不新增自动同步策略选择。
PTOAS BufidSync 遇到任意静态或动态手写 Buffer-ID 操作时跳过整个函数，
不合并手动与自动 Buffer-ID 分配。本接口不提供核间 mode-4 同步或完整硬件
Mutex API。

自动分块（Auto-tiling）也保留这些边界。如果完整结果无法放入 L0C，跨过
Buffer-ID 操作的延迟 M/N matmul/store 折叠会被拒绝。应将 store 放进受保护
区域或显式分块；较小结果可以在原位置采用仅 K 维分块方案。

这些 API 需要 [PTOAS !462](https://gitcode.com/cann/pto-as/merge_requests/462) 中的后端修复。
通用最低 PTOAS 版本不能证明具备该能力。当生成的 kernel 包含手动 Buffer-ID
操作时，PyPTO 会探测所选 assembler 的具体 pipe EmitC lowering 和动态手动同步
BufidSync 跳过行为；缺少任一能力时报告依赖错误。应将 `PTOAS_ROOT` 指向包含
这些修复的构建。普通 kernel 仍沿用基础版本要求；成功的能力探测按可执行文件缓存。

CPU 仿真不属于本次改动已验证的支持范围，并按任务验收要求跳过。
`tests/st/runtime/ops/test_buffer_sync.py` 的设备验收用例
覆盖三个 pipe、静态复用和动态物理 ping-pong 槽位，使用独立 golden 数据并重复
启动。用例仅在测试框架中关闭自动 event 插入，公开的默认编译策略保持不变。
