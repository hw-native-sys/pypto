# Gloo 与对称内存窗口

> 收尾范围仅包含性能覆盖、最终审查与回归、交付材料，不再继续优化。
> 下文早期“下一步”记录属于历史，不代表尚待执行的优化任务。
> 当前验收状态以末尾“新基线完整模型验收”章节为准。

Path B 在单机上按每 rank 一个进程运行。Gloo 负责 CPU 协调，torch_npu
对称内存提供可访问的对端窗口，PyPTO chip kernel 执行 TSTORE、TPUT，以及
用 `remote_load` 加 `add` 实现的多 rank 求和。
入口为 `examples/distributed/08_gloo_shmem_collectives.py`。

## 窗口与上下文约定

`pypto.runtime.shmem_gloo` 按 64 字节边界切分槽位，与分布式 codegen 一致。
首槽 `_comm_ctx` 存储 1088 字节的 runtime `CommContext`：

| 字段 | 字节偏移 |
| ---- | -------- |
| rankId | 0 |
| rankNum | 4 |
| winSize | 8 |
| windowsIn[64] | 16 |
| windowsOut[64] | 528 |
| workspaces[3] | 1040 |

每个 workspace 是由两个 uint64 组成的 `(address, size)`。此路径的三个
workspace 都为零，不支持依赖 workspace 的 SDMA/URMA/RDMA 操作。
上下文与数据一起存放在对称堆中，必须存活至所有使用其指针的 kernel 完成。

`pld.import_window_buffer`（也可写为 `pld.tensor.import_window_buffer`）
使用与 `alloc_window_buffer` 相同的 IR 标记声明命名槽位。赋值左侧名称是
运行时绑定名，IR 不嵌入主机地址。这个别名本身不会改变分配行为；生成的
HOST 代码仍需要显式 provider。`RunConfig.domain_provider` 将 provider
传给生成的 HOST 入口，不进入编译选项。持久化分布式 worker 拒绝外部
provider，因为其域保留协议负责管理生命周期。

`make_shmem_domain_provider` 使用当前 simpler 的
`ChipDomainContext.buffers` 接口。传入已初始化的**直接 L2** simpler
worker，通过公开的 `Buffer.wrap` / `device_location` 注册外部存储。
释放域只撤销注册，不释放 SHMEM 堆。域必须存活到同步 dispatch 完成。
provider 检查 rank 映射、槽位容量、workspace 请求、worker 选择与进程身份，
不能为 L3 子进程授予父进程 SHMEM 映射的访问权限。

## 运行双 rank 示例

需要相互匹配的 PyPTO、其固定版本的 simpler runtime、torch/torch_npu、CANN、
SHMEM，以及两张可用 A2/A3 卡。加载 SHMEM 环境后执行：

```bash
torchrun --standalone --nproc_per_node=2 \
  examples/distributed/08_gloo_shmem_collectives.py -p a2a3
```

共享机器上应通过设备队列提交，并在启动 torchrun 前将
`ASCEND_RT_VISIBLE_DEVICES` 设置为分配到的设备列表；local rank 对应这些可见设备。

示例使用直接 `ChipWorker` dispatch，不运行生成的 HOST/L3 入口。
构造时设置 `auto_init=False`，再调用 `init(borrowed_context_generation=1)`，
借用 torch 已激活的上下文。token 标识本进程唯一的上下文代次；重置或重建
上下文后必须换用新 token。关闭 worker 不会重置 torch 的设备。

数据参数使用由 `Buffer.wrap` 支持的无地址 `TaskArgs`，物化后的 CommContext
指针放入 uint64 标量槽位。输入输出保持为 NPU tensor。每次 launch 前先同步
torch 工作，再将当前调用者 stream 传入 `_run_chip`；这个串行示例不承诺与
框架队列重叠执行。同步运行成功后，撤销注册前已完成引用退休。Gloo barrier
分隔跨 rank 发布与读取，也防止对端尚在读取时复用槽位。golden 来自 Gloo
收集的真实 rank 输入。成功时两端均打印 `GLOO_SHMEM_COLLECTIVES_PASS`。

## 历史发现与限制

源实验为 `nalinaly/pypto` 的 `f3b6e7f0`。其调试记录报告：父进程 SHMEM GVA
未映射到 L3 fork 子进程，TSTORE 出现 MTE `0x800000`；进程内 L2 dispatch
避免了地址空间不一致。历史实验结果不代表当前 runtime 已验证通过。

同一记录报告，两个进程独立 launch builtin allreduce 时出现 UB 越界。
本示例使用 `remote_load + add`，不 launch 该 builtin，也不假定 NPUSHMEM
支持 barrier/put_signal。集合通信示例支持 2/4/8 rank；Qwen runner 根据 `WORLD_SIZE` 特化分片。
不包含 vLLM 集成或多机传输。

## 移植验证状态

M1 已于 2026-11-12 通过双 rank 硬件验收。验证环境为 PyPTO `88efefc33`
加本地移植、固定版本 simpler `355ce301`、CANN 9.3.0、torch 2.10.0+cpu、
torch_npu 2.10.0.post7、cann-shmem 1.6.0（`backends/910`）及 PTOAS 0.72。
经设备队列在 Ascend A3 的 0、1 号卡上运行，两个 rank 的 TSTORE、TPUT、
allreduce 均通过基于实际输入的 golden 校验，并均输出
`GLOO_SHMEM_COLLECTIVES_PASS`。任务退出码为 0，两张卡的锁均已释放。
校验完成后的退出阶段有 SHMEM peer-disconnect 警告。
这些结果仅覆盖当前双 rank 配置；M2 随后通过了下文限定范围内的整网 FP32 参考验收。

原 torch_npu 2.10.0 环境可以加载动态库，但 SHMEM 初始化失败。补充
`SHMEM_INSTANCE_PORT_RANGE` 后暴露出异常且各 rank 不同的 `instance_id`，
未解决问题。独立的 2.10.0.post7 环境无需该设置即可通过分配和 rendezvous。
不能仅凭动态库成功加载就认定运行兼容。

硬件执行还暴露了导出 `ChipCallable` 签名遗漏 scalar 的问题：入口传入了
CommContext scalar，但注册签名仅声明 tensor，借用上下文执行因此以 `-1004`
拒绝准备。现已修复 codegen，使其依次输出 tensor 方向和 scalar 条目；
CoreCallable 的纯 tensor 签名保持原有语义。五个示例回归用例在修复前均失败，
修复后均通过。定向签名及 orchestration 回归共通过 50 项，扩展 orchestration
回归通过 359 项（`build/pathb-orchestration-regression.log`）。此前移植回归通过
645 项、4 项因可选 runtime 环境而跳过；11 项纯 CPU 窗口测试使用
`--noconftest` 通过。CPU 结果本身不能证明硬件正确性。Native bindings 和
runtime 二进制均在本工作区编译；runtime 版本校验和五个内核的原生编译通过。

本地日志：`build/pathb-example-onboard-signature.log`（硬件验收）、
`build/pathb-shmem-probe-post7.log`（环境探测）、
`build/pathb-signature-red.log` 和 `build/pathb-signature-green.log`
（修复前后回归），以及 `build/pathb-example-native-compile.log`。
本地环境为 `.venv-npu`；加载 CANN 后，`build/pathb-shmem-env.sh` 提供动态库
搜索路径。启动时将本工作区的 `python`、`runtime/python`、`runtime` 加入
`PYTHONPATH`，并将每个 rank 的原生编译并发限制为 1，以遵守机器总并发 2 的限制。

## M2 独立 Qwen3 TP2

`examples/models/qwen3_tp/runner.py` 使用真实 Qwen3-14B safetensors 权重、
40 层、batch one、最多 32 个 prefill token 和 128 个位置的 KV cache。
Embedding、RMSNorm、Q/K 归一化、RoPE、attention、MLP 和 LM head 均由
PyPTO kernel 计算。每个 rank 只加载自身的行/列矩阵分片，embedding、LM head
与归一化权重复制。Python 调度同步 L2 执行。o_proj 和 down_proj 都依次执行
FP32 SHMEM 发布、Gloo barrier、remote-load/add 和防止过早复用槽位的第二次 barrier。

计算 kernel 改编自 Apache-2.0 参考实现 `nalinaly/vllm-ascend` 的提交
`e7cb166290dfcbf2f997aa67f01b323be643fe0e`；源码署名和原始许可证保留在同目录。
该 runner 不依赖 vLLM 运行时，也不使用设备端信号协议。

在 M1 的路径之外，将仓库根目录加入 `PYTHONPATH`，通过硬件任务队列启动，
沿用同样的原生编译并发限制：

```bash
torchrun --standalone --nproc_per_node=2 -m examples.models.qwen3_tp.runner \
    --model-dir /path/to/Qwen3-14B --steps 2 --output-dir build/qwen3-output
```

`QWEN3_TP_RANK_AGREEMENT_PASS` 只说明两 rank 的 logits 有限且一致，
不是 M2 数值验收通过标记。独立 CPU FP32 Hugging Face 校验读取 prefill 和 decode
保存的 logits：

```bash
PYPTO_QWEN3_MODEL=/path/to/Qwen3-14B \
PYPTO_QWEN3_OUTPUT=build/qwen3-output/outputs.pt \
python -m pytest --noconftest tests/st/distributed/test_qwen3_tp_reference.py -v
```

参考校验要求贪心 token 一致、相对均方根误差低于 3%、logits 余弦相似度高于
0.999。Safetensors 和 Transformers 是示例/参考校验的可选依赖。CPU 编译测试
已通过 28 个 Qwen3 入口和七个 M1/共享调度检查；九个权重布局/配置测试通过
（共 44 项，`build/pathb-m2-regression.log`）。
CPU 测试本身不能证明整网硬件正确性。

首次整网冒烟任务在 A3 设备 2、3 上使用真实权重，完成了全部 40 层的 prefill
和一次 decode。对于 `The capital of France is`，两个 rank 都生成 `" Paris."`，
通过 logits 有限性与跨 rank 一致性检查，任务退出码为 0。
日志为 `build/pathb-m2-onboard/run.log`。保存输出的第二次任务在设备 0、1 上也以 0 退出并释放了设备锁。
独立 CPU Hugging Face FP32 验收通过，使用 Transformers 5.5.3：

| 阶段 | 相对均方根误差 | 余弦相似度 | 贪心 token（两者一致） |
| ---- | -------------- | ---------- | ---------------------- |
| Prefill | 0.005979 | 0.999992 | 12095 (`" Paris"`) |
| Decode | 0.011355 | 0.999941 | 13 (`.`) |

证据位于 `build/pathb-m2-acceptance/run.log`、`outputs.pt` 和
`reference-fp32.log`（一个参考校验测试通过）。结合 44 项 CPU 回归检查，
M2 对当前 prompt 和一次 decode 的首个真实权重 TP2 验收完成。
这不代表长上下文、一般 batch、4/8 rank 正确性或生产性能已验证；M3 集合通信泛化见下文。

**数值约定：**权重和激活存储为 BF16，移植的 kernel 在归一化、RoPE、SwiGLU
与投影归约之间保留 FP32 中间值。因此验收采用高精度 FP32 eager 参考，
相对误差 3% 和余弦相似度 0.999 的阈值保持不变。
最初与 eager **BF16** 的比较得到 prefill/decode 相对误差 2.26%/3.97%；
虽然贪心 token 一致，decode 未通过 3% 阈值。该结果保留在
`build/pathb-m2-acceptance/reference.log`。
此实现不承诺逐算子 BF16 等价；需要该约定的应用须另行对齐舍入位置。

## 逐层诊断与独立序列

runner 支持 `--prompts-file prompts.json`（prompt 字符串的 JSON 数组），
在 case 之间重置 hidden state 和两份 KV cache，保留权重与已编译程序。
每个 case 保存为 `case_N/outputs.pt`。`--trace-layers` 需要 `--output-dir`，
在 rank zero 捕获全部 40 层、十一个阶段的最后一个有效 token。
捕获会同步设备拷贝，仅用于数值诊断，不用于性能测量。

CPU 参考校验支持单份输出或包含多个 case 的目录。
`PYPTO_QWEN3_REFERENCE_DTYPE=bf16` 选择 eager BF16，默认仍为 FP32。
两种模式使用相同的数值断言。`PYPTO_QWEN3_REPORT=report.json` 即使在数值断言
失败时也保存完整 logits 和逐阶段指标。局部 RMSNorm、RoPE、SwiGLU 校验
固定实际 kernel 输入，比较 FP32 融合与 eager BF16 的舍入方式，以区分局部运算
和层输入已经累积的偏差；这些诊断不改变模型执行。

## 扩展数值验证

诊断矩阵覆盖 1、5、16、32 token 的 prompt，每项包含一次 prefill 和连续七次
decode。原 kernel 在两个 rank 上完成全部 32 步，logits 有限且一致，覆盖了
超过位置 32 的 cache 读取及独立序列之间的重置。未修改的 HF FP32 参考有
1/32 步失败：`Hello` prefill 相对均方根误差为 12.414%。HF BF16 有 6/32 步
失败，其中三步贪心 token 不一致。失败结果保留在
`build/pathb-m2-trace/{fp32,bf16}.{log,json}`；此前单 prompt 的验收不能推广到此矩阵。

原 Paris decode 与 BF16 的层输出误差在第 0 层已为 0.883%，第 2 层首次超过
1%（1.606%），第 39 层为 1.501%，最终归一化和 head 后的 logits 误差为
3.966%。这是累积偏差，并非最后某一层突然发生异常。

固定实际设备输入后，发现额外的数值问题：RMSNorm 与 FP32 融合 CPU 计算平均
相差 0.208%，Q/RoPE 相差 0.233%，而 SwiGLU 仅相差 0.000211%。原实现使用了
`pl.rsqrt` 默认的 A2/A3 近似 `vrsqrt` 路径。三处归一化现已指定
`high_precision=True`，选择带 scratch 的 sqrt/divide 实现。四个 lowering 入口
检查要求输出高精度 RSQRT，连同三个快照/重置测试，共七项定向检查通过。
更新后的双 rank 硬件运行完成全部 32 步，两个 rank 输出有限且一致。
固定输入的 RMSNorm 平均误差降至 0.000409%，Q/RoPE 降至 0.000119%。
这验证了局部精度提升，但不代表全模型等价：未修改的 HF FP32 仍有 3/32 步
失败（一步 token 不一致），HF BF16 有 8/32 步失败（两步 token 不一致）。
结果保留在 `build/pathb-m2-rsqrt/{fp32,bf16}.{log,json}`。

`PYPTO_QWEN3_REFERENCE_DTYPE=fused` 是独立的 CPU 诊断模式，使用 HF 完整矩阵
并采用 PyPTO 的存储舍入边界。它不是未经修改的 eager 参考，不能替代 FP32/BF16
验收。CPU 参考加载禁用 Transformers 额外的异步加载池，以遵守双 worker 限制。

更新后的 fused 诊断中，全部 32 步均满足均方根误差与余弦相似度限制，但贪心
token 仅 31/32 步一致。16-token 样本的第 2 步，设备选择 `" human"`（19.210857），
参考选择的 `" the"` 在设备上的分数为 19.205627，两者相差 0.005230。
该 token 检查仍判失败，未放宽阈值。报告保留在
`build/pathb-m2-rsqrt/{fused,fused_remaining}.{log,json}`。

设备中点精确测试（`tests/st/distributed/test_bf16_rounding.py`）确认：显式
`rint` 在全部 32 个正负 BF16 中点上与 Torch 一致，默认 `round` 有 16/32 个
不同。这证实了转换约定的差异，但尚不能单凭此解释模型剩余的 token 不一致。
模型全部九处 BF16 cast 已显式指定 `rint`。28 个 chip 入口编译检查全部通过，
包含生成的 BF16 转换必须使用 RINT 的检查。更新后的设备运行完成全部 32 步，
两个 rank 输出有限且一致。fused CPU 参考全部 32 步及贪心 token 均通过：最大
相对均方根误差 1.9412%，最小余弦相似度 0.999861。此前 fused token 不一致已
消除。未修改的 HF FP32 全部 token 一致，但 `Hello` prefill 均方根误差为
9.7618%，仍不达标；其余 31 步通过。未修改的 HF BF16 有 6/32 步失败，其中三步贪心 token
不一致，最大相对均方根误差为 4.5996%。报告位于
`build/pathb-m2-rint-full/`。扩展 eager 参考验收仍未通过，M3 传输验证见下文。

参考 decode 使用保存的实际前一个 token 进行 teacher forcing，确保比较相同前缀。
token 一致性是该前缀下的条件比较，并不保证两套实现独立自由生成的结果完全一致。

## M3 集合通信泛化

本次采用 BF16 存储、FP32 融合计算作为 M2 契约，上述全部 32 步 fused 参考已
通过；不宣称与 eager BF16 等价。`08` 示例接受 2/4/8 个进程。
TSTORE/TPUT 校验来自环上前一 rank 的输入；allreduce 用 FP32 累加本地及
所有其他 rank，再以中点取偶数方式转为 FP16。这是 O(N) 读取的正确性基线，
并非带宽优化的 ring 算法。
发布和窗口复用的 Gloo barrier 均保留。`--iterations 5` 每轮改变发布数据，
所有 rank 均对照独立收集的 CPU 输入校验。跨 rank 串行编译以遵守主机资源限制。
七项 CPU 检查及 2/4/8 rank 硬件运行全部通过（每个 rank 的 TSTORE、TPUT、
五轮求和）。所有任务退出码为零并释放设备。证据位于
`build/pathb-m3-codegen.log` 和 `build/pathb-m3-{2,4,8}rank.log`。Qwen runner 现根据 `WORLD_SIZE` 支持 2/4/8 路切分，在产物中保存该大小，并
按固定 rank 顺序进行 FP32 归约。TP8 使用 128 宽 K tile，完整覆盖 640/2176
宽行分片，避免遗漏尾块。每种规模的全部 28 个 chip 入口编译通过，16 项权重/
runner 测试通过。TP4 与 TP8 均完成全部 32 步设备矩阵，各 rank 输出有限且一致。
TP4 全部 32 步 fused 参考及 token 验收通过（最大相对均方根误差 2.2947%，
最小余弦相似度 0.999802）。TP8 全部 32 步及 token 也通过（最大相对均方根
误差 1.9481%，最小余弦相似度 0.999810）。在约定的 fused 契约下，已完成本次
2/4/8 rank 模型扩展验收。移除 barrier、融合
集合通信和 stream 重叠仍待完成。证据位于 `build/pathb-m3-tp{4,8}/`。

## 性能测量

`examples/models/qwen3_tp/benchmark.py` 在每次队列分配中测量一个固定 prompt。
先配置仓库/runtime 路径及支持的 PTOAS，在 CPU 上准备产物：

```bash
WORLD_SIZE=4 python -m examples.models.qwen3_tp.benchmark --prepare \
    --artifacts build/pathb-perf/tp4/artifacts
```

随后通过四卡 TaskQueue 任务，以 `torchrun --nproc_per_node=4` 运行该模块，
传入 `--artifacts`、`--model-dir`、`--prompt`、`--output` 和
`--rounds 5 --steps 8`。TP2、TP8 分别提交独立任务。提交时捕获已准备好的
CANN/SHMEM 环境，架构预检查在分配内执行。native 产物可能延迟加载，使用
一个完整预热序列排除其影响；不在一个分配内混跑不同模型规模。

每个测量序列重置 KV cache 并使用同一 prompt。每步在设备同步和 Gloo 对齐
barrier 后开始计时。墙钟时间包含同步 L2 调用、模型集合通信 barrier、最终
logits 回拷 CPU；不包含 rank 对比、token 选择、报告和下一步的对齐 barrier。
关闭 trace 和逐层日志，以最慢 rank 的时间作为该步样本。保留五个 prefill 和
35 个 decode 样本，计算中位数、插值 p95 和均值。decode 速率是 batch=1 的
平均 forward 延迟倒数，不代表服务吞吐。allreduce 总耗时包含发布、两次
barrier 和归约调用，不等同于纯网络或纯设备耗时。计时外校验所有 rank 的
logits 有限且一致，以及预热/测量的 token 可重复性。这是共享主机上的短测量
基线，不是性能 SLA。

2026-11-12 在 Ascend A3 上测量，batch=1、32-token prompt、每个序列七步
decode，一轮预热加五轮测量：

| TP | Prefill 中位数 / p95（ms） | Decode 中位数 / p95（ms） | Forward tokens/s | Decode 集合通信平均耗时（ms） |
| -- | -------------------------- | ------------------------- | ---------------- | ----------------------------- |
| 2 | 1045.5 / 1317.5 | 1038.1 / 1054.0 | 0.954 | 177.4 |
| 4 | 952.6 / 1248.8 | 943.4 / 959.7 | 1.045 | 210.4 |
| 8 | 946.7 / 1499.0 | 945.5 / 978.6 | 1.034 | 263.0 |

所有运行均通过 rank 一致性和 token 可重复性检查。TP4 相对 TP2 的 decode
中位延迟降低约 9%；本次样本中 TP8 未继续获益。集合通信总耗时占各 rank
累计 decode 墙钟时间的 16.9% / 22.0% / 27.2%。剩余部分包含计算和 host
调用开销，本测量尚未拆分两者。Prefill p95 仅基于五个样本且波动明显。
三次任务在共享主机上顺序运行，分别使用 1–2、0–3、0–7 号卡，并非同时进行
的严格受控实验。下一步先拆分计算与 host 调用开销，再选择融合或 stream
优化，保留同步正确性基线。
逐 rank 原始样本位于 `build/pathb-perf/tp{2,4,8}/timings.json`，日志在同目录。
`build/pathb-perf/summary.json` 保存任务 ID、设备、源码哈希和仓库版本；
工作区仍未提交。

## TP4 调用开销拆分

单独以 `--profile-dispatch --rounds 2 --steps 3` 运行相同 32-token prompt，
排除一轮预热，采集 native TIMING span。日志直接写入各 rank 的
`native-rankN.log`。首次共享管道日志触发 native 队列溢出，已拒绝使用其
拆分结果，保留在 `build/pathb-profile/tp4/`。有效重跑位于
`build/pathb-profile/tp4-direct/`。释放设备后分析：

```bash
python -m examples.models.qwen3_tp.profile \
    --timings build/pathb-profile/tp4-direct/timings.json \
    --log build/pathb-profile/tp4-direct \
    --output build/pathb-profile/tp4-direct/breakdown.json
```

分析器按 PID 和单调时钟区间排除预热及校验。native `chip.run` 和设备执行
窗口的数量都必须与 Python dispatch 数量一致。每步每 rank 共 723 次调用，
其中包含 80 次发布及 80 次归约。以下 decode 均值覆盖 16 个 rank 样本
（四 rank × 两轮测量 × 两步 decode）：

| 互不重叠的耗时分类 | 每 rank 每步均值（ms） |
| ------------------ | ---------------------- |
| 设备执行窗口，含设备编排/调度 | 211.2 |
| native `chip.run` 中设备窗口以外的时间 | 221.3 |
| dispatch helper 中 native `chip.run` 以外的时间 | 557.1 |
| forward 中 dispatch helper 以外的时间 | 78.4 |
| 开启 trace 的 forward 总耗时 | 1068.0 |

第三行包含 tensor/Buffer 准备、Torch 同步、配置/ABI 绑定和清理，不等于
纯 Python 解释器时间。host/device 时长相减是归因估算，并非同步时钟上的
关键路径追踪。第四行约 69.8 ms 来自集合通信 barrier 及周边 Python。
集合通信总耗时 235.3 ms 与上表分类重叠，不可再次相加。设备窗口也不等于
纯 AI Core 计算时间。trace 运行比无 trace 的 TP4 均值 956.6 ms 更慢，且
覆盖的 decode 位置更少，不能替代原基线。
下一步优先检查重复的调用准备和每步 723 次 launch，再考虑兼容的 kernel
融合。在替代可见性/窗口复用协议得到独立正确性证明和验证前，保留两次
集合通信 barrier。

## 调用准备信息复用实验

`gloo_shmem_utils.DispatchPlan` 为一个不变的编译产物保留 callable、tensor
方向元数据及默认调用配置，不保留 tensor 地址或 Buffer 注册。每次调用仍
重新创建 TaskArgs、填入 scalar、注册 Buffer、获取当前 stream、同步 Torch、
同步执行并关闭注册。使用其他编译产物的 plan 会报错；修改配置后应重建 plan。
最初仅在 benchmark 中通过 `--cache-dispatch-plan` 开启；模型 runner 集成
验证见下文，默认仍不缓存。

新一轮顺序 TP4 对照采用相同 32-token prompt、五轮测量、每轮七步 decode，
关闭 native trace：

| 模式 | Prefill 中位数（ms） | Decode 中位数 / p95（ms） | Forward tokens/s |
| ---- | -------------------- | ------------------------- | ---------------- |
| 新测无缓存基线 | 945.5 | 951.9 / 1001.0 | 1.041 |
| 复用 dispatch plan | 833.8 | 834.3 / 864.5 | 1.185 |

本次观察到 decode 中位延迟降低 12.3%，prefill 降低 11.8%。两次任务均使用
0–3 号卡，退出码为零，rank/重复性检查全部通过，两种模式生成的 token 完全
相同。这是共享主机上的一次成对测量，并非重复交叉实验；本次仅复用元数据，
未重跑独立 HF 数值验收。定向测试覆盖地址/scalar 变化、逐次清理及产物不匹配。
证据位于 `build/pathb-cache/{baseline,cached}/timings.json` 与
`build/pathb-cache/comparison.json`。
未移除 barrier 或减少 launch。下文完成模型 runner 集成验证后，下一步
单独测量剩余 TaskArgs/Buffer 准备开销；若考虑持久注册，需要额外的生命周期
管理约定。

## 模型 runner 的 dispatch-plan 集成

模型 runner 现支持 `--cache-dispatch-plan`，每个编译入口保留一个 plan，
入口对象被替换时重建。计算、发布及归约均走同一 dispatch 路径。benchmark
现在直接测量此 runner 实现，不再维护另一份缓存包装。开关仍为 opt-in，
省略或显式传入 `--no-cache-dispatch-plan` 时使用无缓存路径。`--artifacts` 可复用 CPU 准备的 manifest，在模型/设备
设置前校验 TP 大小和 kernel 源码哈希；延迟 native 加载仍属于启动阶段。
runner 测试覆盖复用、scalar 变化、程序替换和无缓存路径。TP2/TP4/TP8 设备
运行及独立 fused 参考验证全部通过，各覆盖 32 步，贪心 token 均 32/32 一致。
最大相对均方根误差分别为 1.9412% / 2.2947% / 1.9481%，与此前通过的报告一致。
产物位于 `build/pathb-integrated/tp{2,4,8}/`。全部 96 个保存的 logits 向量还与对应已通过验收的无缓存 TP2/TP4/TP8 运行
逐位一致。集成后 TP4 五轮测量通过：prefill 中位数 867.8 ms，decode 中位数/
p95 为 845.8/861.7 ms，forward 速率 1.161 tokens/s。相对新测无缓存基线
951.9 ms，decode 中位延迟降低 11.1%。共享主机顺序对照说明集成后收益仍在，
不将其视为严格受控的加速比。证据位于
`build/pathb-integrated/perf-tp4/{timings.json,run.log}`。

## 开启缓存后的 dispatch 准备开销

`--profile-setup --cache-dispatch-plan` 在共享 helper 中加入不重叠的阶段计时，
不启用 native TIMING 日志。计时器记账及区间之外的 helper 析构时间保留为
未归因项。相同 TP4 32-token 工作负载（一轮预热、五轮测量、每轮七步 decode）
通过全部 rank 一致性和 token 重复性检查。释放设备后分析：

```bash
python -m examples.models.qwen3_tp.profile --setup-only \
    --timings build/pathb-setup/tp4/timings.json \
    --output build/pathb-setup/tp4/breakdown.json
```

140 个 rank decode 样本的均值（每 rank 每步仍为 723 次 dispatch）：

| 不重叠阶段 | 每 rank 每步均值（ms） |
| ---------- | ---------------------- |
| Plan 校验 / TaskArgs 初始化 | 0.5 / 2.9 |
| Tensor 检查 / Buffer 注册 | 21.6 / 110.3 |
| Tensor 绑定 / scalar 绑定 | 28.8 / 0.9 |
| Torch 同步 / 当前 stream 查询 | 35.1 / 28.1 |
| `_run_chip` 调用总耗时 | 552.8 |
| Buffer 释放 | 9.0 |
| dispatch 未归因时间 | 11.7 |
| forward 中 dispatch 以外时间 | 78.2 |
| forward 总耗时 | 880.0 |

调用行包含 Python 绑定、native runtime 和设备执行，与此前 native `chip.run`
span 的边界不同，也不等于纯计算时间。集合通信总耗时 190.4 ms 与上表重叠。
计时本身及共享主机条件变化意味着本次是诊断结果，不能据此声称新的受控加速。
使用的设备为调度器允许的同组 1、3、4、5 号卡。
TaskArgs 初始化仅占 forward 约 0.3%，tensor 绑定另占 3.3%；Buffer 注册加释放
约占 13.5%。下一步优先研究相邻兼容 kernel 融合，减少每步 723 次 launch，
从而有机会同时减少调用和注册开销。持久 Buffer 注册是独立备选方案，需明确
内存所有权、失效条件、失败后的使用结束及清理测试；本次未引入此缓存，也未
移除 barrier。

## SwiGLU + Down launch 合并实验

Runner 和 benchmark 新增可选参数 `--fuse-swiglu-down`（默认关闭）。
一个 chip 入口先调用现有 SwiGLU AIV 任务，再调用 Down AIC 任务。
保留 BF16 中间结果、RINT 转换、collective barrier 和 dispatch 所有权。
SwiGLU trace 快照调整到 Down 之后采集；Down 只读取这个中间结果。
两个输出仍由调用者持有。

生成的编排将 `act` 标为第一个任务的 InOut、第二个任务的输入，
保留生产者与消费者依赖。TP2/4/8 的 CPU lowering 检查通过。
TP4 setup-profile 实测的全部 16 个 rank 样本中，每个 rank 每步均为
**683 次 host dispatch**，其中 40 次为融合调用。原路径为 723 次，
因此减少 40 次调用（5.53%）。

设备验证覆盖 TP2/4/8，每种配置分别测试 prompt 长度 1/5/16/32，
每个 prompt 八步。所有 rank 一致，全部生成 token ID 与已验收非融合运行一致，
**96 个保存的 logits 张量全部逐位一致**，基线为
`build/pathb-integrated/tp{2,4,8}/case_*/outputs.pt`。
这保持了此前验收的 BF16 存储、FP32 算术契约；
没有重新与未经修改的 Hugging Face BF16 执行进行比较。

TP4 性能测量在同一组分配设备（1、4、5、6）上顺序执行：
开启 dispatch-plan 缓存，使用一个 32-token prompt，
一次预热、五次测量，每次包括一次 prefill 和七步 decode。
先跑非融合，再跑融合；这两次运行均关闭 native tracing 和 setup 计时。

| 指标 | 非融合 | 融合 |
| ---- | ------ | ---- |
| Prefill 中位数 / p95（ms） | 868.5 / 1225.2 | 839.6 / 1032.3 |
| Decode 中位数 / p95（ms） | 869.6 / 928.5 | 845.1 / 924.9 |
| Decode 均值（ms） | 883.2 | 854.3 |
| Decode 均值倒数速率（tokens/s） | 1.132 | 1.171 |

观察到 decode 中位数降低 **2.82%**（24.5 ms）。
这是共享主机上的顺序测量，并非隔离环境中的因果加速保证。
Collective 耗时也有波动，不能把全部变化归因于 launch 开销。
另外执行的短时插桩运行用于确认调用次数，不混入上述性能数据。

证据位于 `build/pathb-fusion/`：

- `tp{2,4,8}/correctness/case_*/outputs.pt` 和 `comparison.json`。
- `tp4/{baseline,fused}/timings.json`、`performance-comparison.json`、
  `performance.log` 和 `count/{timings,breakdown}.json`。
- CPU 预编译产物位于 `tp{2,4,8}/artifacts`；native wrapper 的编译和加载
  仍在设备任务首次使用时发生。
- 原 TP4 任务 `task_20261113_200904_391595127625` 等待 600 秒后超时取消，
  未获取设备。成功重试任务为 `task_20261113_203215_47228218169`，
  日志为 `tp4/correctness-retry.log`。后续提交使用 `--timeout 0` 等待队列。

## 完整 FFN launch 合并

Runner 和 benchmark 新增 `--fuse-ffn`，与 `--fuse-swiglu-down` 互斥，
两个选项默认均关闭。一个 chip 入口编排现有的 Residual/RMS、Gate/Up、
SwiGLU、Down 四个任务。保留调用者持有的中间缓冲区、BF16 舍入边界
及两处 collective barrier。诊断快照在 FFN 入口完成后采集，
下游任务只读取这些中间结果。

TP2/4/8 编译检查覆盖生产者与消费者的张量依赖、RINT 转换及高精度 RMSNorm。
设备验证覆盖四种 prompt 长度（1/5/16/32），各八步。
96 个 logits 张量全部与已验收非融合输出逐位一致，生成 token 一致，
各 rank 检查通过。证据：`build/pathb-ffn/tp{2,4,8}/{correctness,comparison.json}`。

TP4 性能测量使用设备 0–3，开启 dispatch-plan 缓存，
一个 32-token prompt，一次预热、五次测量，每次包含 prefill 和七步 decode。
以下三种配置按表中顺序执行，均未开启计时插桩：

| 指标 | 非融合 | SwiGLU + Down | 完整 FFN |
| ---- | ------ | ------------- | -------- |
| Prefill 中位数 / p95（ms） | 832.0 / 1133.1 | 827.4 / 993.2 | 765.7 / 1064.8 |
| Decode 中位数 / p95（ms） | 837.8 / 1166.1 | 822.4 / 845.5 | 769.6 / 804.0 |
| Decode 均值（ms） | 915.4 | 828.9 | 773.1 |
| Decode 均值倒数速率（tokens/s） | 1.092 | 1.206 | 1.293 |

完整 FFN 的 decode 中位数相对非融合降低 **8.13%**，
相对 SwiGLU + Down 降低 **6.42%**。三种运行的生成 token 一致。
这仍是共享主机上的顺序比较：基线尾延迟及 collective 耗时存在波动，
不能将均值速率变化全部归因于 launch 开销降低。

独立的 setup-profile 运行确认：全部 16 个 rank 样本中，
**每 rank 每步均为 603 次 dispatch**，其中 40 次为完整 FFN 调用。
相对非融合减少 120 次（16.60%），相对 SwiGLU + Down 减少 80 次。
每个 FFN 入口内部仍有四个设备任务；未删除算术 kernel 或 collective。

性能证据：`build/pathb-ffn/tp4/perf/{baseline,swiglu_down,ffn}/timings.json`、
`perf/comparison.json`、`perf/count/{timings,breakdown}.json`
及 `performance-retry.log`。
此前 `performance.log` 对应任务因与我们的 TP8 验证重叠被主动停止
（exit 130），其计时数据均未采用。成功重试在 TP8 完成并释放设备后启动。

完整 FFN 仍默认关闭。要推广为普遍性能结论，还需要更广泛工作负载
及重复、交错顺序的性能测量。

## 平衡执行顺序的完整 FFN 性能测量

TP4 的 A–B–B–A 复测已在设备 2–5 上完成，A 为非融合，B 为完整 FFN。
每组一次预热、五次测量，使用相同的 32-token prompt、七步 decode
和 dispatch-plan 缓存。

| 分组 | A1 | B1 | B2 | A2 |
| ---- | -- | -- | -- | -- |
| Decode 中位数（ms） | 844.6 | 774.3 | 765.9 | 850.6 |

两组中位数降幅分别为 8.32%（A1/B1）和 9.95%（A2/B2）。
每种配置合并 70 个 decode 样本后，中位数为 848.8 与 768.1 ms，
降幅 9.50%。四组生成 token 一致。这支持当前工作负载下收益可重复；
序列内的 step 样本存在相关性，不能当作相互独立的统计重复实验。

证据：`build/pathb-ffn-abba/tp4/{a1,b1,b2,a2}/timings.json`、
`comparison.json` 和 `run.log`。TP2/8 分别申请设备，因此跨 TP 的绝对
延迟不能解释为受控的扩展性测试。

TP2 和 TP8 已按相同 A–B–B–A 协议完成，两个测量任务顺序执行，
彼此没有重叠。TP2 使用设备 6/7；TP8 使用 0/2/3/4/5/6/7/8。
每组仍包含五个 prefill 和 35 个 decode 测量样本。
所有 rank 一致性、预热与测量 token 可重复性及跨组 token 检查通过。
两个任务均以零退出码完成并释放设备。

| TP | A1 decode 中位数 | B1 | B2 | A2 | 两组降幅 |
| -- | ---------------- | -- | -- | -- | -------- |
| 2 | 959.9 ms | 851.8 ms | 876.0 ms | 923.0 ms | 11.27% / 5.10% |
| 8 | 866.1 ms | 780.6 ms | 798.7 ms | 935.2 ms | 9.87% / 14.60% |

每种 TP 内，每个方案合并 70 个 decode 样本：

| TP | 非融合中位数 | 完整 FFN 中位数 | 观察到的降幅 |
| -- | ------------ | --------------- | ------------ |
| 2 | 941.2 ms | 866.5 ms | 7.94% |
| 4 | 848.8 ms | 768.1 ms | 9.50% |
| 8 | 903.8 ms | 790.0 ms | 12.59% |

当前工作负载下，六组配对比较均支持融合改善性能。
但配对降幅的差异及基线漂移仍然存在，不能据此断言普遍加速，
也不能证明 TP4 天然最优。
证据：`build/pathb-ffn-abba/tp{2,8}/{a1,b1,b2,a2}/timings.json`、
各 TP 的 `comparison.json`、`run.log` 以及汇总 `summary.json`。
完整 FFN 仍默认关闭。后续优化不属于本次收尾范围。

## 收尾审查与交付

范围冻结为现有的可选 dispatch 缓存和 FFN launch 融合。
本次收尾不再安排 profiling、attention 融合、持久 Buffer 缓存或 barrier 删除。
未 commit、未 push、未开 PR。

**代码审查：**在支持的单机/direct-L2 流程内未发现新的阻塞性正确性问题。
检查覆盖 C++/绑定/stub 的 scalar pool ABI、provider 转发及 fork/L3/workspace
防护、借用 context 的 generation 和关闭重建流程、注册所有权与失败清理、
TP 权重切分、固定形状与 cache 边界、FFN 依赖和舍入点、
benchmark 的计时边界与 token 检查。

审查发现失败路径测试缺口及新测试的类型问题。
收尾期间只修改测试：验证先 teardown 再释放 Buffer、teardown 失败时不释放，
并明确动态 import、AST、mock module 的类型。
未修改生产 runtime 或 kernel 行为。

**验证：**

- Worktree CMake 构建通过，使用两个 worker。
- 相关回归：964 passed，268 warnings。
- TP2/4/8 全部模型入口编译：各 30 passed，共 90 项。
- 修复测试类型问题后：90 项受影响测试通过，76 warnings。
- Ruff 检查及格式：全部 36 个变更 Python/stub 文件通过。
- Pyright 指定 `--pythonpath .venv/bin/python`：零错误、零警告。
- 文档：213 对中英文文件及 213 项导航检查通过。
- `git diff --check` 通过；测试日志保留弃用警告。

**扩大性能覆盖：**TP4，启用缓存，A–B–B–A 顺序，
每组一次预热、三次测量。一个 token 的 “Hello” prompt 加七步 decode
已通过全部 rank/token 检查。A1/B1/B2/A2 的 decode 中位数分别为
845.7/740.7/782.8/850.8 ms；合并中位数为 850.4 → 756.2 ms，
降低 11.08%。32-token prompt 加 31 步 decode 的任务
`task_20261113_225819_369013932697` 已完成（exit 0，设备 2–5 已释放）。
A1/B1/B2/A2 decode 中位数分别为 830.5/767.0/774.3/855.1 ms，
两组配对降幅为 7.65% 和 9.45%。每种方案合并 186 个 decode 样本后，
中位数为 842.5 → 771.0 ms，降低 8.48%。四组生成 token 全部一致。
三项收尾工作现已**全部完成**：扩大性能覆盖、最终审查与回归、
更新审阅交付包。不再安排后续优化。共享主机波动仍限制性能结论的推广。
证据位于 `build/pathb-closeout/{short,longdecode}/`。

新增 benchmark 检查有限 logits、rank 一致性及 token 可重复性，
没有对较长 decode 的每个位置新增独立 CPU 参考验证。
数值验收证据仍是此前 TP2/4/8 共 96 个 logits 的融合前后逐位一致，
以及已接受的 BF16 存储/FP32 算术参考契约。

**审阅包：**`build/pathb-closeout/pathb-review.tar.gz` 包含完整 patch
（包括未跟踪的新文件）、源码快照、文件哈希与状态、基线/runtime revision、
复现命令、回归日志和性能 JSON。不包含模型权重、编译二进制或大量 native trace。
Manifest 记录已完成的收尾检查。只应向记录基线上的干净 checkout 应用 patch，
并使用记录的 runtime revision；依赖和 CPU 预编译设备产物需在本地准备或重建。

支持边界保持为 Qwen3-14B、batch 1、TP2/4/8、prefill 1..32、cache 128、
每 rank 一个进程、Gloo 控制面加 SHMEM 和 simpler direct L2。
不宣称与未经修改的 HF eager BF16 等价，不支持 L3 SHMEM 映射、
需要 workspace 的通信引擎、跨机器执行，也不提供服务吞吐/SLA 承诺。
保留 kernel 原始来源和许可证；不包含 vLLM runtime 集成或 simpler 子模块修改。

## 最新主线上的 PR 准备

2026-11-16，补丁已无文本冲突地应用到 PyPTO `7d2c5bf3889c`。
该基线固定 simpler `c1c234667d54`，要求 PTOAS v0.74。
上面的原始真机结果来自 PyPTO `88efefc33` 和 simpler `355ce3015e3e`；
这些结果保留为历史证据，不能视为更新后依赖组合的真机验收结果。

新基线构建成功。六项 C++ 测试、在新编译 native runtime 环境中的 964 项相关回归、
以及 11 项 SHMEM ABI/窗口布局检查均通过。TP4 和 TP8 的编译入口检查均通过，
各覆盖 30 个入口。新的双 rank 真机冒烟测试在两个 rank 上均通过 TSTORE、TPUT
以及三轮每轮改变输入的 allreduce；内核在 CPU 侧预先准备，设备由任务队列分配。

这些检查覆盖集成和双 rank 通信；上面的完整模型 TP2/4/8 数值与性能结果仍属历史证据。
功能范围仍为单机直接 L2 执行，以及已接受的 BF16 存储/FP32 算术契约，
不包含新的性能优化。

BF16 中点系统测试使用受设备锁保护的标准 compiled-call 路径和 fixture 分配的设备。
其 32 个数值的舍入断言已在队列分配的单卡上通过，九项设备锁回归检查也全部通过。

## 新基线完整模型验收

2026-11-16，使用真实 Qwen3-14B 权重，在 PyPTO `7d2c5bf3889c`、
simpler `c1c234667d54` 和 PTOAS v0.74 上通过完整模型数值验收。
每个 TP 配置覆盖 1/5/16/32-token prompt，各执行八步（一次 prefill 和七步 decode）。
独立 CPU 参考实现遵循已接受的 BF16 存储/FP32 算术契约。
验收阈值保持不变：相对 RMS 误差小于 3%、cosine 大于 0.999、greedy token 完全一致。

- TP2：token 一致 32/32；最大相对 RMS 误差 1.9412%；最小 cosine 0.999860644。
- TP4：token 一致 32/32；最大相对 RMS 误差 2.2947%；最小 cosine 0.999801934。
- TP8：token 一致 32/32；最大相对 RMS 误差 1.9481%；最小 cosine 0.999809921。

六次真机任务均通过各 rank 一致性检查并释放设备。默认未融合/缓存关闭配置
与 FFN 融合/缓存开启配置的 96 对 logits 逐位一致，生成 token 一致。
90 个 native 内核产物均在分配设备前准备完成。
本次验收未修改生产代码，也未放宽阈值。

证据位于 `build/pathb-acceptance-pr/summary.json`、`hardware-summary.json`、
`fusion-results.json`、`tp{2,4,8}/reference.json`，以及各任务的命令、日志和输出张量。
这是上述存储契约下的数值验收，不声称与未经修改的 HF eager BF16 等价。
本次未复测性能，早期性能数字仍属历史证据。
