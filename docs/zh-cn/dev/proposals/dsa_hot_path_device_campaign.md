# 设备任务：解释并扩展 DSA-RP 热路径效应

## 目标

完成原有 20 个内核的 DSA-RP 研究，用受控固定布局解释每个可重复的加速或减速，并在更大的内核集合上前瞻性验证完成前沿（completion frontier）模型。

工作猜想是：

> 当跨缓冲区地址复用引入一个较晚、执行频率较高的前驱，并延后下一消费者的异步完成前沿时，该复用代价较高。

同步数量仅用于诊断；单内核设备延迟才是性能结果。本任务不调优 DSA 算法、不拟合统一的周期权重、不引入负权，也不改变非负成对 DSA-RP 问题。

## 1. 精确版本

必须通过 HTTPS 获取以下精确版本；不可用分支顶端代替。

| 组件 | 仓库 | 版本 |
| --- | --- | --- |
| PyPTO | `https://github.com/tonibohnlein/pypto.git` | `63413940e1db791556fd1830c255554cf930d7e9` |
| dsa-solver | `https://github.com/tonibohnlein/dsa-solver.git` | `553b9ce933711e8d78363475c81a9e1ca3b44466` |
| PyPTO-Lib | `https://github.com/hw-native-sys/pypto-lib.git` | `6e897cd99c28767b22e05f209da3e041f15c3dfc` |
| PTOAS | `https://github.com/tonibohnlein/PTOAS.git` | `007f2d637059d907a08faece045e6d3d82943d4b` |
| runtime | PyPTO 子模块 | `8cdb306cb9a81ad1a0561325021105c676a69c1e` |
| pto-isa | `runtime/pto_isa.pin` | `83d01313d9bfc247c4b7c8bcf969d1019f0d106f` |

统一使用 `/opt/dsa-rp-hot-path-expansion` 作为全新制品根目录，并在任务前后记录版本和干净工作树状态。

## 2. 资源与卫生要求

- 构建、测试、编译和分析最多使用两个 worker。
- 设置 `PYPTO_CODEGEN_MAX_WORKERS=1`。
- 每个 NPU 最多一个进程，全局最多两个设备进程。
- 每个端点使用全新输出目录。
- 不得编辑导出的 DSA 问题、原始候选记录、解或 PTO。
- 驱动与生成结果放在制品根目录，不能放入源码检出。
- 设备命令必须使用 `timeout --kill-after=30s` 限时。
- 运行前后记录设备健康状态和进程。
- 所有失败端点都必须有终态分类。

## 3. 构建和主机预检

以 Release、测试开启、MiniMalloc 基线关闭的配置构建 dsa-solver；构建和 CTest 都最多使用两个 worker，并要求所有测试通过。随后在全新环境中安装指定 runtime 和启用 DSA 的 PyPTO，验证：

```text
is_dsa_solver_available() == True
MemoryPlanner.DSA 存在
DsaReusePenaltyRecognizer.QUADRATIC 存在
PassContext 可往返 DSA 导出、回放、识别器和参考布局字段
```

运行：

```bash
python -m pytest tests/ut/tools -n 2 --maxprocesses 2 -q
```

从精确版本构建 PTOAS，记录二进制 SHA-256，并要求支持：

```text
--enable-insert-sync
--pto-insert-sync-summary
--pto-insert-sync-debug=3
```

## 4. 端点定义与强制编译预检

每个内核正常编译两个求解器端点：

```text
compact: DSA + DISABLED recognizer + DEFAULT placement
rp:      DSA + QUADRATIC recognizer + DEFAULT placement
```

两者都必须满足容量并通过独立解验证；求解器相关的语义指纹必须一致。任何设备运行前都执行：

```bash
python .claude/skills/incore-profiling/preflight_standalone_comparison.py \
  --baseline-build <compact-build> \
  --candidate-build <rp-or-ablation-build> \
  --function <target-function> \
  --invocation-profile <kernel-invocation.json> \
  --ptoas-bin <exact-ptoas> \
  --ptoas-root <PTOAS-checkout> \
  --output-root <kernel-preflight> \
  --build-npu --pto-isa-root <pto-isa> \
  --soc-version Ascend910B2
```

`preflight.json` 必须证明 post-InsertSync 源码已生成、两个 NPU 单内核程序均构建成功、ABI/block_dim/标量/指针输入一致且至少指定一个输出。缺少旧 `.cpp` 不是跳过理由；必须从持久化 level-3 `.pto` 重新生成。

## 5. 完成原有 20 个内核

从以下先前制品恢复精确列表：

```text
/opt/pypto/dsa-rp-top20-sync.tar.gz
SHA-256 75c1d8395a5c74c5b9cd3e5ee2751cdf6b8e3a775061489d0d2063a450b394dd
```

使用前验证哈希。制品中的 `results/targets.tsv` 是 20 行选择的权威来源，不能根据旧报告中的名称重新拼接。生成唯一的 `results/top20-ledger.tsv`。先前有 14 个内核完成计时，六个受阻：

| 内核 | 数量 | 原阻塞 | 必须修复 |
| --- | ---: | --- | --- |
| `kv_proj`、`gate_up_proj`、`mtp_projection_linear_aic` | 3 | 合成标量/索引越界，507015 | 使用由源码证明、边界有效的 invocation profile |
| 两个 Qwen3-32B 程序中的 `out_proj`、`rope_kv_cache` | 3 | 旧工具未保存 `.cpp` | 从持久化 PTO 定位并重新生成 |

另外恢复旧汇总 TSV 漏掉的 `rope_kv_cache` 和 `rmsnorm_rope_cache_write` 报告。

每个 invocation profile 必须显式给出所有标量、保证最后一个 tile 不越界、为非零索引/控制数组提供 `pointer_fills` 或精确文件，并注明值来自源码还是实际调度。两端使用同一 profile；单内核计时前，全模型 compact 和 RP 都要通过 golden。

20 行必须分别终止为：

```text
TIMED
NO_PLACEMENT_CHANGE
CORRECTNESS_BLOCKED
COMPILE_BLOCKED
INPUT_PROFILE_BLOCKED
```

## 6. 重现已有性能效应

在两个安静设备上，以真正包含 InsertSync 的二进制重新测量所有有效端点。重点重新分类：

- `mtp_projection_quant`：先前校正后的两设备结果约快 17.8%，大部分由热循环携带的 load-destination handoff 解释；
- `qk_norm`、`mtp_projection_norm`、`rope_cs`、`rmsnorm_rope_cache_write`：先前分别约快 27.5%、16.8%、11.2%、4.7%，尚未解释；
- `prefill_hca_c128_rmsnorm_rope`：先前约慢 9%，尚未解释；
- `mtp_projection_rms`：后续研究约快 19%，目前仅知道五个同步组的组合；
- `down_dual_proj`：约快 0.5%，L1 handoff 尚未单独隔离。

这些数字只是历史观察，不是期望值。

## 7. 解释每个可重复效应

仅对 compact 与 RP 在两个设备都有可重复实质差异的内核继续分析：

1. 计算物理重叠集合的对称差。
2. 通过逻辑缓冲区身份、范围、route、循环和访问位置关联 raw-v4 候选。
3. 将候选映射到按函数区分的同步后 PTOAS 程序。
4. 找出释放消费者的最终前驱。
5. 从循环结构估算动态次数；动态界无法确定时保留符号表达。
6. 按完成时刻与动态频率排序。
7. 编译消融前冻结预测。

构造可独立放置候选的单边 on/off、RP 多边时的 leave-one-out、按预测优先级累积的 ladder、保持完整重叠图的平移地址控制，以及已有更晚 release 的 covered-frontier 控制。

每个端点必须通过：精确 overlap XOR、无意外重叠变化、容量和硬约束一致、PyPTO 回放验证、InsertSync 前 PTO 只有地址变化、函数级同步归因、独立构建预检以及输出逐位一致。

只有当两设备符号都符合预测、累积消融解释至少 80% 的差异（或置信区间覆盖完整差异）、地址控制重现拓扑和延迟分类且不需要未建模拓扑变化时，才标记 `EXPLAINED`；否则使用 `PARTIALLY_EXPLAINED` 或 `UNEXPLAINED`。同步与重叠拓扑不变而延迟稳定变化时，应记录为独立的地址布局机制。

## 8. 前瞻性扩展

扫描完整 PyPTO/PyPTO-Lib 库存，从至少 10 个程序选择至少 40 个新内核：

| 分层 | 数量 |
| --- | ---: |
| 预测会暴露完成前沿扩展 | 24 |
| route 匹配但已被 drain 覆盖的控制 | 8 |
| 低频或 distance-zero 控制 | 8 |

内存空间至少覆盖 20 个 UB/Vec、5 个 L1/Mat，以及合计 5 个 L0A/L0B/L0C。若无法构造合法布局或没有有效输入，按库存数量报告 `COVERAGE_BLOCKED`，不得人为制造重叠。

候选按以下事实排序：跨缓冲区而非自递归、完整范围与写证据、重叠可机械移除、消费者完成前沿确实延后、前驱完成时刻、动态频率以及替代布局的容量余量。不得按复用对数或同步组数排序。

在 PTOAS 之前冻结：内核/程序、内存空间、逻辑缓冲区对与子范围、WAR/WAW、源/目的 route、消费者、预测最终前驱、循环距离和动态频率、exposed/covered 分类及延迟方向。

## 9. 计时协议

仅测量独立单内核或规范的 mixed group；不得拆开 mixed AIC/AIV。两设备初筛均采用：

```text
ABBA 顺序
每进程 10 次 warmup
8 个 quartet
每 block 100 次测量
每个比较、每个设备共 3200 个样本
```

效应或边界案例提升至 24 个 quartet。使用配对 bootstrap 置信区间并保留原始 `samples.tsv` 与 `report.json`。

```text
BENEFICIAL：两设备 CI 都在加速方向排除 0
HARMFUL：两设备 CI 都在减速方向排除 0
EQUIVALENT：通过 ±max(1%, 0.5 us) 的 TOST
INCONCLUSIVE：其他情况
```

## 10. 必需输出

生成：

```text
results/top20-ledger.tsv
results/kernel-selection.tsv
results/overlap-deltas.tsv
results/handoff-map.tsv
results/frozen-predictions.tsv
results/sync-attribution.tsv
results/timing.tsv
results/explanation-status.tsv
REPORT.md
HANDOFF.md
```

每个端点保留 DSA 问题/解、语义指纹、invocation profile、`preflight.json`、InsertSync 前后 PTO/C++、函数级 summary/debug、正确性哈希和原始计时样本。归档只包含证据及再生成元数据；排除构建树和大型可再生成张量，并记录其哈希和生成方法。

## 11. 报告问题和终态

报告必须回答：20 个旧内核是否都有终态；六个基础设施阻塞是否消除；哪些历史效应在两设备重现；各效应解释程度；完成前沿扩展能否前瞻性预测符号；完成 lateness 能否排序同一消费者内的代价；动态频率能否区分实质与免费 handoff；哪些案例因已有 drain 而免费；是否存在稳定的地址布局效应；哪些可机械识别的成对类别足以支持非负正权。

最终 verdict 只能是：

```text
HOT_PATH_MODEL_SUPPORTED
HOT_PATH_MODEL_REFINED
HOT_PATH_MODEL_REFUTED
INFRASTRUCTURE_BLOCKED
PARTIAL
```
