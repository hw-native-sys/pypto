# 异步传输校验（Async Transfer Validation）

`pld.tensor.put_async` 和 `pld.tensor.get_async` 在 A5 后端下降为按 dtype 后缀命名的
外部 helper 调用（`pypto_urma_{put,get}_async_<dtype>`）。helper 会同时校验两端张量的
范围与通信窗口，通过 IR `engines` 字段申请 URMA workspace，并复用现有运行时入口提交；
它从不轮询。完成由所属任务拥有：提交时会注册该请求的完成条件，因此后续以目标为
输入的任务要等传输完成后才会被释放。
两端都必须是窗口绑定、稠密行主序（dense row-major）的 `DistributedTensor`，且位于同一通信域：
被物化到不同窗口作用域的两端可以编译，但会在运行时被拒绝。
结果与 `dst` 共用存储；算子返回不代表传输完成。

adapter 符号与 C++ 元素类型复用公共 dtype 转换。

- Python builder 要求 `engine=pld.CommEngine.URMA`（也是默认值）。
  整数、浮点数、布尔值、字符串及其他枚举在整数转换前即被拒绝。
  原始 IR 将 engine 存储为整数 `1`。
- `peer` 必须是整数/index 标量。负的常量 rank 会被拒绝；
  该检查不保证通信域内运行时 rank 的边界合法。
- `dst_offsets`、`src_offsets` 和 `shape` 必须一起提供。区域在两端都必须连续：
  跳过传输形状开头已知等于一的维度，再要求第一个非单元素维度之后的各维取满。
  动态长度可能大于一，因此不能跳过。
  所以 `[16, 64]` 中的 `[1, 32]` 和 `[2, 4, 64]` 中的 `[1, 2, 64]` 连续，
  而对应的 `[2, 32]` 和 `[2, 1, 64]` 不连续。

## 支持的异步传输

| 维度 | 当前支持 | 尚未支持 |
| ---- | -------- | -------- |
| 算子 | `pld.tensor.put_async` / `get_async` | PTOAS 原生异步算子 |
| 引擎 | 仅 URMA（`pld.CommEngine.URMA`）；SDMA/RDMA 按名称拒绝 | URMA / SDMA / RDMA 选择 |
| 目标 | A5（Ascend 950）InCore/AIV body | helper ABI 移植后的其他目标 |
| 载荷 | 每次调用一个连续区域；运行时在内部拆分超过 256 MiB 的传输 | 多 SGE / scatter-gather 列表 |
| 完成 | 所属 AIV task（正常完成 + 已注册的异步条件）；producer `InOut` → consumer `In` | 显式 event、`wait`/`test`、按请求释放 |
| 注册 | 两端均为已存在的、窗口绑定且位于同一通信域的 `DistributedTensor` | 自动 MR 注册 / 隐式暂存 |
| 源生命周期 | 调用方在传输 task 退役前不得覆盖 `src` | 运行时管理生命周期 |
| 远端可见性 | 接收方仍需现有的跨 rank notify/wait 协议 | 完成序的远端消费 |

**使用模式。** 传输必须在拥有目的地的 InCore body 中执行；在 host orchestration
body 中调用会被 `OrchestrationReferencesResolved` 拒绝。目的地是 producer 的
`InOut` 输出，后续 task 以 `In` 消费：

```python
data = pl.store(row, [my_rank, 0], data)
for peer in pl.range(nranks):
    if peer != my_rank:
        pld.tensor.put_async(
            data,
            peer=peer,
            src=data,
            dst_offsets=[my_rank, 0],
            src_offsets=[my_rank, 0],
            shape=[1, SIZE],
            engine=pld.CommEngine.URMA,
        )
return data
```

算子返回**不**代表字节已经移动。消费者对 `data` 的 `In` 依赖才是排序点；
在该 task 退役前不得覆盖 `src`。生产者任务可以在提交前后执行其他计算，但在传输未完成前不得覆盖 `src`。远端消费者仍需要现有的跨 rank notify/wait
协议：本地 task 完成只排序发起方，不排序对端的读取。

**ABI 说明。** 为 workspace engine 字段，`CommContext` 从 1056 字节增长到
1088 字节。该变化是全局的（包括 A2/A3 构建），codegen 与 runtime 必须一起
升级。异步 helper 通过 `comm_workspace<CommWorkspaceKind::URMA>(*ctx)` 读取
URMA workspace，并要求通信域申请 `workspace_engines=("urma",)`（目前为程序级
保守策略）。

**失败诊断。** 容量按 `分配 >= 窗口视图 >= 传输区域` 传递地约束。窗口具化时，
其连续布局字节数（`product(shape) * dtype_bytes`）必须不超过分配的逻辑大小；即使程序只访问前缀，
过大的视图也会被拒绝。槽位的物理 cache-line 对齐填充不能通过视图访问。verifier
还拒绝非窗口端点、非连续区域、缺少 offset/shape 参数、负的常量 peer、非法 engine，
以及不满足其端点视图的传输区域。符号化的视图或槽位大小在 host 执行 window 声明时检查，早于其消费者。
生成的 helper 在运行时拒绝非法 peer（`peer < 0`、`peer >= nranks` 或
`peer == rankId`）、越界的端点或窗口字节范围、为零/未对齐的字节数以及缺失的
workspace。所有失败都是显式的；不存在静默的 SDMA/RDMA 回退。

**兼容性。** 返回类型（`dst`）与完成保证保持稳定。容量校验不改变生成 kernel 的
ABI；符号化的视图/槽位大小在 host orchestration 中检查。静态窗口声明现在会在编译期
校验，因此窗口视图超出其后备分配的现有程序会被拒绝，而不是继续编译。对 session/event、
`wait`/`test`、SDMA/RDMA 引擎、PTOAS 原生下降、按请求释放和多 SGE 的支持将以
增量方式加入。
