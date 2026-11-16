# 原生 SIMT runtime 验证

P0 探针通过 PyPTO 现有的 VPTO 设备编译器和持久化 runtime，验证原生 A5 SIMT
GM copy。它独立于 [`SimtLaunchStmt` lowering](05-simt-lowering.md)，
后者现已实现 P3 可执行子集。

## 探针执行内容

`tests/ut/backend/vpto_vf_validation/simt_gm_copy.pto` 导出一个启动 128 个 SIMT
线程的 AIV 函数。活跃线程直接在 GM 指针间复制 FP32 元素，`valid` 是 INT32
运行时参数。AIV 导出函数末尾采用 PTOAS SIMT GM 访问示例中的原生 `PIPE_ALL`
排空操作。它提供本独立探针的 kernel 退出边界，并未实现 SIMT 与 Tile 操作间的
自动同步。

测试通过生产 kernel wrapper 生成器解包 Tensor 描述符和标量参数。每个指针只
计入一次 Tensor 子视图原点。测试在私有生成产物中将被调度的 copy kernel 替换
为原生 copy，并以 `vpto_sources` 附加原生模块。这是 ABI 验证夹具，不代表
PyPTO SIMT IR 已能 lowering 为可执行代码。

未修改的生产路径完成以下步骤：

1. PTOAS 导出 VPTO LLVM，并检查外部符号。
2. 毕昇将 helper 和 C++ wrapper 编译成设备对象。
3. 设备链接将 `kernel_entry` 排在首位，随后校验 ELF。
4. 提取 `.text`、注册 `CoreCallable`，通过持久化 A5 runtime 执行。

探针检查 GM 地址空间、线程/寄存器注解、无 UB 中转的直接 `LDG`/`STG`、SIMT
入口保留，以及链接后的 callee 大小与 `VF_SIMT` 长度操作数是否一致。LLVM 导出
绕过了 PTOAS 对象生成阶段的长度修正器。本仅使用 intrinsic 的夹具必须在不修补
二进制的情况下正确编译。通用 SIMT 函数，尤其是需要长度修正的内联汇编，必须先
建立受支持的对象生成契约，才能使用这条路径。P3 明确拒绝内联汇编。

## 聚焦验收

在当前 checkout 构建 PyPTO 及固定版本的 runtime。配置固定的 PTOAS v0.74 和
A5 CANN/毕昇工具链，并保持 runtime pin 检查开启。
本测试复用 [VF 测试](03-vpto-vf-integration.md) 的显式启用开关。

首次硬件验收在 Ascend950DT、PTOAS v0.72、CANN 9.3.0 和 runtime 版本
`31f9e72ee6832cd198b77df18c1c305e7fd83676` 上通过。

仅验证编译、缓存和产物迁移，不打开设备：

```bash
PYPTO_RUN_VPTO_VF_VALIDATION=1 python -m pytest \
  tests/st/runtime/vpto/test_simt_gm_copy_onboard.py --platform=a5 -k build -v
```

通过主机设备队列预留 A5 设备后：

```bash
PYPTO_RUN_VPTO_ONBOARD=1 PYPTO_VPTO_DEVICE_ID="$DEVICE_ID" \
PYPTO_VPTO_TIMEOUT=600 python -m pytest \
  tests/st/runtime/vpto/test_simt_gm_copy_onboard.py --platform=a5 -k persistent -v
```

同一个持久化 worker 执行冷加载、缓存命中和迁移后的产物。每阶段检查
`valid = 0, 1, 31, 32, 33, 127, 128`，改变输入三次，覆盖 32 元素的 Tensor
子视图偏移、前后保护区和未激活元素不变，共 63 次精确数值比较。
缓存加载和迁移均不得重新编译 helper。测试会移动原始源码目录，避免旧绝对路径
意外掩盖迁移问题。

`PYPTO_VPTO_RUNTIME_OUTPUT` 指定证据根目录，默认 `build/simt-runtime`。
每个子进程创建新的 `simt-*` 目录，保留实际编译命令、输入、LLVM、对象、链接后的
ELF 和反汇编。`result.json` 区分仅编译验证与硬件执行；`build-result.json`
记录 runtime 版本和编译次数。pytest 父进程为包括设备清理在内的整个子进程设置超时。

本里程碑仅覆盖 FP32 连续 GM copy。带跨度或动态布局、UB gather、混合流水线
依赖、多维 launch、lane 操作、原子和通用 IR lowering 需要各自的验收测试。
