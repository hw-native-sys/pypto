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

## 短向量值的编译器能力门禁

`tests/ut/backend/vpto_vf_validation/test_simt_packed.py` 在公开 packed-value API
之前，为任务 11.6 建立原生编译器能力边界。这些夹具仅用于编译：所有线程访问相同地址，
UB 入口参数也不是可运行的主机 ABI。不要在设备上执行这些内核。

矩阵针对 PTOAS v0.74、A5（`dav-c310-vec`）及 CANN 9.3.0 Bisheng
（Clang 15.0.5）。依次执行 PTOAS LLVM 导出、PyPTO 生产路径的
`prepare_llvm_for_bisheng` 兼容处理，以及使用生产 `DEVICE_FLAGS` 的编译。
原始与处理后的 LLVM 均保留；LLVM 19 的 `memory(...)` 注解需要已有的兼容处理。

| 操作 | FP16×2 | BF16×2 | FP32×2 |
| --- | --- | --- | --- |
| GM、UB 读写及加法、乘法 | 可编译 | 可编译 | 可编译 |
| `pto.absf` | 可编译 | 可编译 | 可编译 |
| `pto.sqrt` | 可编译 | PTOAS 拒绝类型 | PTOAS 拒绝类型 |
| `shuffle_idx/up/down/bfly` | 可编译 | PTOAS 拒绝类型 | PTOAS 拒绝类型 |
| GM/UB 原子 add/min/max/exch/cas，丢弃旧值 | 可编译 | 可编译 | 未探测 |
| 同上，但使用旧值 | PTOAS 拒绝使用 | PTOAS 拒绝使用 | 未探测 |
| `vector.extract` + `vector.from_elements` | 方言不可用 | 方言不可用 | 方言不可用 |

FP16×2 ↔ FP32×2、BF16×2 ↔ FP32×2 转换的原生舍入选择器 `r`、`z`、`a`、`f`
均可编译。这只证明编译成功，数值舍入行为仍需设备验收。四元素 FP16/BF16/FP32
拷贝被 GM `pto.ldg` 拒绝，但 UB `pto.load/store` 可编译；这不证明四元素算术
受支持，也不证明会生成单条 packed 硬件指令。

104 个用例要求生成 65 个包含 SIMT 入口及非空机器码的设备对象，并得到 39 个特定的
PTOAS 拒绝诊断。崩溃、超时、诊断不符或负向用例变为接受，都会使门禁失败。
正向用例还检查 LLVM 的 packed 元素寻址：偏移 1 跨过整个向量。UB 加载带有自然
对齐属性（半精度双元素为 4 字节，FP32 双元素为 8 字节；四元素为 8/16 字节）。
这些是编译器生成的假设，并非运行时对齐检查。

将完整矩阵及工具链版本保存在新的证据目录中：

```bash
source .claude/skills/testing/load-env.sh
python tests/ut/backend/vpto_vf_validation/test_simt_packed.py \
  --probe build/simt-packed --jobs "$PYPTO_TEST_JOBS"
```

也可通过 pytest 运行参数化门禁：

```bash
PYPTO_RUN_VPTO_VF_VALIDATION=1 python -m pytest \
  tests/ut/backend/vpto_vf_validation/test_simt_packed.py \
  -n "$PYPTO_TEST_JOBS" -v
```

先配置 `PTOAS_ROOT` 和 `ASCEND_HOME_PATH`。普通单元测试会跳过这些原生用例。
本阶段不增加 PyPTO packed 类型或操作，不验证链接和启动大小，也不证明硬件正确性或
性能。公开 API 前仍需找到受支持的构造与拆分 lowering。对齐检查、奇数尾部、子视图
偏移、转换数值、原子竞争及标量版本对比仍需独立验收。packed 值应保持为线程私有值，
与 Tile 存储及 VMI `RegType` 区分。
