# HBG eager Host 测量

`tests/benchmarks/hbg_csa_stability.py` 测量真实 CSA 单层权重与合成历史的
HC pre → norm → CSA → HC post 路径。它需要额外的 vllm-ascend CSA fixture、
与 fixture 匹配的已构建 vllm 原生扩展、DeepSeek 权重以及 NPU；这些资产不随
PyPTO 分发，因此这不是在干净 checkout 上即可运行的自包含 benchmark。

## 前置条件

- 在当前 checkout 构建 PyPTO 与可选 Torch NPU adapter，安装匹配的 torch_npu/CANN。
- 准备外部 vllm-ascend 源码，其中 `tests/pypto_test` 包含
  `dsv4_csa_single_layer.py`、`dsv4_csa_env.py` 及其依赖，且包含
  `vllm_ascend.ops.pypto.deepseek_v4_flash_csa` 包。
- `--native` 指向包含 `src/vllm` 和已构建
  `src/vllm-ascend/vllm_ascend/vllm_ascend_C` 扩展的目录；`--checkpoint` 指向权重。
- 显式设置 `PYTHONNOUSERSITE=1`、`PTOAS_ROOT`、`PTO_ISA_ROOT`、`ASCEND_HOME_PATH`。
  确认 pypto、simpler 和 native 扩展加载路径。硬件任务使用服务器任务调度器，
  遵循当前机器的物理卡访问限制；`--device` 是可见设备中的逻辑编号。
- 每个 queue 配置和重跑使用独立进程；关闭逐次 TIMING/STRACE、预加载插桩和 graph。
  首次 enqueue 的 Identity bootstrap 与编译放在预热阶段，冷启动需另测。

## 运行与口径

以下变量由运行者提供；输出目录必须不存在。

```bash
TASK_QUEUE_ENABLE=0 python tests/benchmarks/hbg_csa_stability.py \
  --source "$CSA_SOURCE" --native "$CSA_NATIVE" --checkpoint "$CSA_WEIGHTS" \
  --output "$RESULT" --device 0 --history 8192 --mode paired \
  --blocks 6 --samples 32 --warmup 5 --cpu-samples 8 --seed 1
```

`paired` 在同一进程内交替 Native/PTO，每个 block 反转顺序；双方计时内均生成
compact metadata。包含 PyPTO adapter 的每次构造成本。输入恢复、输出毒化和设备同步
在 Host 计时外；同一 arm 连续调用保留前次租约，arm 边界 drain。两个实现分别与
自身 reference 做完整状态校验，不将已知 Native/PTO 零容差差异当作数值验收通过。
固定已编译 artifact，计时期间禁止编译。

将 `--mode` 改为 `paired-diagnostic` 可单独采集 Native Torch dispatcher 算子次数和
PyPTO 调用 spans。此模式有插桩开销，不能作为性能数字。Torch dispatcher 名称也不能
证明底层必然使用某个 ACLNN API；需结合匹配版本的扩展实现或 profiler 确认。

历史三版对照使用 `--mode bench --suite current`：每个进程测一版 PyPTO，两个
A/A 标签校准噪声；此模式 PyPTO **复用** metadata，而 Native 保留原生生成路径，
且 Native 先行测一次。因此其中 Native/PTO 比值不能作为同口径性能验收。

## 固定版本与聚合

`hbg_csa_version.py` 面向 scikit-build editable 环境：同一已构建 native/runtime，
在导入前把完整 Python source 映射到独立 Git worktree。先用 `--preflight` 验证。
不支持任意两版 ABI；运行者必须先确认 native/runtime 相容，保留真实 `.git` 和
runtime pin，不能复制 `.so` 或混合单函数替换。输出 manifest 记录版本和源码哈希。

```bash
CSA_VERSION_ROOT="$VERSION_WORKTREE" CSA_VERSION_NAME=before \
  python tests/benchmarks/hbg_csa_version.py --output "$RESULT" --preflight
```

正式运行时用上节 benchmark 参数替代 `--preflight`。历史版本标签分别对应
`before=d5ee5fd`、`pre-unify=12460f9`、`current=6850dc0`；harness 来自修复后的
测量 checkout，基线无需自带 harness。每版使用完整 Python worktree。

原始证据目录按 `h8192-q0-r1-before` 等命名，包含 report、identity、manifest、
Host/CPU 样本、状态检查以及同名 `.log`。聚合仅读取文件，不使用 NPU：

```bash
python tests/benchmarks/hbg_csa_aggregate.py --repo "$REPO" \
  --evidence "$EVIDENCE" --device "$RECORDED_PHYSICAL_DEVICE" \
  --repeats 5 --output "$SUMMARY"
```

聚合验证版本源码哈希、runtime/native 身份、输出状态、384 个 Host 样本及 CPU
生命周期样本，再按进程配对计算。每阶段取中位数，A/A 取均值后对 block 取中位数。
噪声取 A/A 最大绝对差与跨进程极差的最大值；使用 50,000 次进程级 bootstrap，
要求所有轮次节省为正且 CI 下界超过噪声。不同物理卡分别报告，不混合作为同卡重跑。
历史 136→11 ms 是作者环境观测；原始外部 fixture 和采集数据未随本仓库发布，
单凭此源码不能独立复核该数字，也不能视作最新 main 或 ACLNN 的验收结果。
