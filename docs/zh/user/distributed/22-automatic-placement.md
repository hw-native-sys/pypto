# 按 rank 分配自动选择 L3 / L4

lib 的同一个 `@pl.jit.host` 入口可以使用两种拓扑：所有 rank 在同一主机时编译为 L3；rank 分布在多台主机时，编译器生成 L4 控制器和 L3 转发入口，使用现有 Direct MPI 执行链路。CHIP 计算函数无需改写。

## 接入配置

rank 是本次算子的全局 NPU 编号。`placements` 的下标就是全局 rank；每项填写资源分配系统已经给出的主机和物理设备编号。PyPTO 不探测机器或申请卡。

```python
from pypto.ir import RankConfig, RankPlacement
from pypto.runtime import RunConfig

ranks = RankConfig(
    world_size=4,
    placements=[
        RankPlacement("node-a", 0),  # global rank 0
        RankPlacement("node-a", 1),  # global rank 1
        RankPlacement("node-b", 0),  # global rank 2
        RankPlacement("node-b", 1),  # global rank 3
    ],
    python_executable="/opt/venv/bin/python",  # 各节点上的相同路径
)
compiled = lib_host_entry.compile(config=RunConfig(rank_config=ranks))
# 运行方式和现有 HOST/CLUSTER 产物一致。
```

示例中的设备号用于说明配置结构，实际必须填写调度系统分配的设备。算子代码中的 `worker` 始终使用逻辑 rank。若四个 rank 全在 `node-a`，相同源码将走 L3；单机配置要求调用方在该主机上执行。

已解析的 Program 也可使用 `ir.compile(program, rank_config=ranks)`。`rank_config` 与显式 `distributed_config` / `cluster_config` 二选一。

## 编译逻辑

1. 校验 rank 数、设备唯一性和主机布局，生成已有的 `DistributedConfig` 或 `ClusterConfig`。
2. 单机保留 HOST 入口。多机把 HOST 入口提升为 CLUSTER 控制器，按全局 rank 分发到目标节点；节点内的 HOST 转发入口调用对应的本地 CHIP worker。
3. 全局 rank 参与的输入切片仍在控制器侧计算；通信窗口沿用现有 L4 的物化和传递流程。转发入口按被调函数和本地 worker 复用。
4. 拓扑快照进入 JIT 缓存键；改变主机、设备或 rank 顺序会触发独立编译。产物继续使用现有元数据冷加载。

## 当前支持范围

多机沿用现有 L4 ABI：每台主机卡数相同，同一主机的全局 rank 连续排列。HOST 到 CHIP 的 worker 必须能静态求出；支持静态 rank 循环和嵌套的静态轮次循环。带循环携带值、运行时选择 worker 或显式 Submit 依赖的自动跨层转换暂不支持。手写的 L3/L4 入口和配置保持原有用法。

## 验证

```bash
source .claude/skills/testing/load-env.sh
PYTHONPATH="$PWD/python:$PYTHONPATH" python -m pytest \
    tests/ut/ir/test_rank_placement.py tests/ut/jit/test_auto_rank.py \
    -n "$PYPTO_TEST_JOBS" -q
```

该测试覆盖配置选择、rank 顺序、缓存隔离、源码编译、共享窗口和冷加载；硬件执行需另行使用已分配的 NPU 进行验证。
