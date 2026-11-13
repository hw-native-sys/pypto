# EmitC 与 VPTO VF 执行接入

显式 `pl.reg_scope()` 可以通过常驻运行时（persistent runtime）在 A5 AIV
kernel 内执行。普通 Tile 操作使用 EmitC，每个寄存器作用域生成一个独立编译的
VPTO helper。本功能扩展 typed VMI lowering，不改变 DSL，也不自动选择向量作用域。

## 产物与执行

```mermaid
flowchart TD
  IR[完成内存规划的 A5 AIV 函数] --> Bundle[PTOCodegen.generate_bundle]
  Bundle --> Caller[Caller PTO: Tiles 与带注解 adapter]
  Bundle --> Helper[Helper PTO: 公共 UB 基址与静态偏移]
  Caller --> EmitC[PTOAS EmitC 与 caller 同步]
  EmitC --> Wrapper[C++ kernel_entry wrapper]
  Wrapper --> CallerObj[kernel.o]
  Helper --> LLVM[PTOAS VPTO LLVM 与 VF 同步]
  LLVM --> Symbols[检查 helper 外部符号]
  Symbols --> HelperObj[Bisheng 设备 helper 对象]
  CallerObj --> Link[ld.lld: kernel_entry 位于首部]
  HelperObj --> Link
  Link --> ELF[验证链接后的设备 ELF]
  ELF --> Text[提取 .text 字节]
  Text --> Core[CoreCallable child]
  Core --> Chip[包含 orchestration 的 ChipCallable]
  Chip --> Worker[A5 常驻 ChipWorker]
```

`PTOCodegen.generate()` 保留组合的 typed VMI 表示，供诊断使用。执行路径使用
`generate_bundle()`，把 helper 写入 `vpto/<kernel>/`，并生成记录符号、参数类型、
effects 与偏移的 manifest。`kernel_config.py` 通过 `vpto_sources` 关联这些路径。
多个显式作用域生成多个 helper，最终链接进同一个 kernel。

Helper 发射复用 `PTOCodegen` 的寄存器操作 lowering。UB 指针对应内存规划得到的
MemRef 基址；VMI 发射器为每次访问加上 Tile view 的元素起点，且只加一次。
不同静态分配 view 的捕获项通过一个公共 UB 基址和常量位移重建，使独立编译后的
helper 仍保留访问重叠关系。

Tile 参数及其派生 view 会被拒绝：它们的运行时 UB 指针关系不能通过局部
分配的规划偏移确定。

## 两层同步边界

| 边界 | PyPTO 提供的信息 | PTOAS 完成的工作 |
| ---- | ---------------- | ---------------- |
| Caller | SIMD adapter 上的 Tile 参数与 `pto.tileop.effects` | 为搬运和调用插入 MTE2 → V、V → MTE3 事件 |
| VF body | 公共 UB 指针基址、常量偏移与 VMI 内存操作 | 为重叠的 WR、RW、WW 访问插入屏障 |

Caller adapter 在依赖分析期间保留 Tile，只有 adapter 函数体才通过
`pto.tile_buf_addr` 取出 UB 指针并调用 helper。Tile 捕获项标记为 `read`；只要有
store 就标记为 `readwrite`；标量标记为 `none`。Effects 覆盖整个捕获 Tile。
Masked store 使用 `readwrite`，表示未激活 lane 的内容需要保留，不表示硬件实际
读取了这些 lane。

PyPTO 不在调用前后添加固定的 `pipe_barrier(PIPE_ALL)`，正常的 kernel 尾部同步仍
保留。调用边界契约依赖 A5 VF 完成语义，因此当前限定为 AIV helper。

PTOAS 负责 VF 依赖分析与屏障插入，包括通过 SSA 数据依赖证明的执行顺序。
PyPTO 将生成的 LLVM 交给 Bisheng，不再重新推导访存依赖或施加 LLVM 指令白名单。
PTOAS/Bisheng 的错误会传递给调用者。下述独立的外部符号与设备镜像检查负责
保证 helper 的打包契约。

索引 gather/scatter 依赖当前 checkout 固定的 PTOAS 版本，对同一 vecscope 内
未知访存范围的 RAW/WAR/WAW 依赖进行保守同步。该版本包含
[PTOAS #1730](https://github.com/hw-native-sys/PTOAS/issues/1730) 的修复，
PyPTO 不再为每次索引访问添加 `VV_ALL` 屏障。Caller/helper 边界与 GM 同步
继续使用已有机制。

## 设备代码与 ABI

LLVM 的 `ptr addrspace(6)`、PTO 的 `!pto.ptr<f32, ub>` 与 caller 的
`__ubuf__ float*` 必须描述相同的 UB 地址空间和传参 ABI。Helper 编译保留
`--cce-generic-addrspace=off`，使用 A5 vector 设备目标，生成设备对象而非 host fat
object。LLVM 导出与 Bisheng 编译分步进行，以便链接前检查 helper 外部符号；相关编译选项
全部参与二进制缓存标识。

运行时接收原始代码字节。本路径只提取 `.text`，因此链接后的镜像必须满足：

- 非空 `.text`，可执行的小端 AI Core ELF64。
- `kernel_entry` 位于 `.text` 开头，运行时从这里开始执行。
- 没有未解析符号或剩余重定位。
- 没有其他非空的已分配 section，包括 `.rodata`、`.data` 和 `.bss`。

符号表和调试表可以留在 ELF 中，但不会装载。需要其他数据存储或 Bridge/外部函数
的 helper 会被拒绝。结构检查不能证明任意手写机器码都与装载位置无关；helper
仍必须遵守设备调用 ABI。

## 当前可执行子集

| 属性 | 当前支持 |
| ---- | -------- |
| 目标 | A5、AIV、ONBOARD、PyPTO Tile 内存规划器 |
| 作用域 | 寄存器操作和最多四层嵌套的规范硬件循环 |
| 内存 | 前端与 PTOAS 支持的寄存器读写和索引访存；不同 Tile dtype 共用 UB 基址 |
| 别名 | 公共 UB 基址、静态 view 位移、store SSA 别名 |
| Mask 与标量 | 运行时 mask 与受支持的标量捕获 |
| 打包 | 源码 artifact 迁移、包含 helper 标识的二进制缓存 |

Grouped/peer 函数、设备对象的 CPU 模拟执行、Bridge 调用及一般 helper 控制流
仍不在当前可执行前端的支持范围内。访存模式与对齐约束由前端和工具链检查，支持的组合需要编译与硬件测试覆盖。
具体 dtype、mask 与索引访存契约见[寄存器操作参考](../language/06-register-operations.md)。

## Required hardware loops

A5 PTO codegen 支持 `pl.reg_scope()` 内的规范 `pl.hardware_loop(stop)`，
最多嵌套四层。VF helper 可在迭代间携带 VecReg 和 MaskReg；不支持携带 Scalar、Tile
和 Tensor。`start` 固定为 0，`step` 固定为 1。
整数 stop 可以是静态或动态值。标量捕获 ABI 支持有符号和无符号的 8、16、32、64 位整数以及 index 值。
对于非空循环，只有在静态算出的 stop 超过 65535 时才拒绝。静态求值在能够证明结果时遵循结果 dtype 的语义，
包括窄位宽无符号整数的回绕。无法证明范围的 stop 继续放行；
调用方需要保证计数和内存访问处于有效范围。
运行时分支和 while 会被拒绝。

指数为非负常量的整数幂（如 `n ** 2`）降低为标量乘法，不增加循环。
标量 PTO emitter 当前不支持动态或负指数；这是算子实现限制，不是 hardware loop 的范围要求。

Codegen 在 `pto.vecscope` 内生成使用 `index` 归纳变量的 `scf.for`。
运行时 `i32` stop 作为标量参数捕获，通过 `arith.index_cast` 转换。
资格检查和回归验证截止于生成 PTO 的边界；PyPTO 不添加硬件循环标记、不改写
LLVM 计数器、不插入内存屏障，也不增加链接后指令校验。后续降低（lowering）、
同步和优化由 PTOAS 负责。

## 运行完整示例

在当前 checkout 构建 PyPTO 及其固定版本 runtime，选择 `toolchain/versions.env` 固定的 PTOAS 版本，
配置匹配的 CANN/Bisheng 工具链，保持 runtime pin 检查开启。先预约空闲 A5 设备，
再运行以下命令：

```bash
python examples/runtime/vf_emitc_vpto.py \
  --device "$DEVICE_ID" --timeout 600 --output build/vf-add-example
python examples/runtime/vf_adapter_ub_overlap.py \
  --device "$DEVICE_ID" --timeout 900 --output build/vf-overlap-example
```

输出目录必须是新目录。父进程的超时涵盖编译、设备初始化、执行和退出。两个示例
都保存真实编译产物、命令日志和机器可读结果。记录 hook 观察生产编译器的结果，
不会替换代码字节。

加法示例验证 `n = 0, 1, 63, 64, 65, 127, 128, 1023, 1024`、32 元素 Tensor
offset、哨兵和输入变化。它复用一个常驻 Worker，依次检查冷编译、缓存命中、把
helper 加法改成减法后的失效与数值变化，以及删除原目录后的 artifact 迁移执行。

重叠示例加载一个 640 元素 Tensor slice，建立重叠 view，保存 masked 写入前后的
快照。它同时检查 caller 搬运依赖和 VF 内部 WR/RW/WW 顺序，覆盖前缀/后缀 mask
以及不应变化的哨兵。

## 定向验证

单元测试覆盖 bundle 接口、view 起点、拒绝情况、ELF 检查、helper 缓存和 artifact
路径。`tests/st/runtime/vpto/` 下的硬件测试需要显式开启、预约的 device ID 和超时，
应串行运行：

```bash
PYPTO_RUN_VPTO_ONBOARD=1 PYPTO_VPTO_DEVICE_ID="$DEVICE_ID" \
PYPTO_VPTO_TIMEOUT=900 python -m pytest \
  tests/st/runtime/vpto --platform=a5 -k persistent -v
```

每个 PR 的 `onboard-tests-a5` job 通过 `vf-tests-a5` 步骤运行此套件，
显式开启真卡执行，并使用 `task-submit` 预约的卡号。套件通过有超时限制的子进程
使用 ACL 核验所选设备型号，不依赖 `torch_npu`。验收 Worker 显式调用
`check_runtime_pin(checkout_root=...)` 指定源码 checkout，因此通过 wheel 安装的
CI 环境仍会比较扩展编译标记、runtime HEAD 和 checkout 的 gitlink。版本未知或
不匹配都会导致验收失败。每次测试调用分配全新的输出目录，
串行执行并限制子进程超时。源码/缓存迁移检查的直接调用也会创建独立的 `run-*`
目录，根目录的 `build-result.json` 指向最近一次运行。其他平台会取消选择
这些仅支持 A5 的测试。`vf-a5-results` CI artifact 在失败时也会保留 JUnit
结果、设备信息、命令日志和编译产物。

仅编译检查、独立 ACL launch 和常驻 Worker 执行属于不同证据。模拟器或 ACL launch
成功不能证明生产 `.text` 已被 Worker 接收并调度。每次运行都应记录工具链版本、
runtime pin、卡号、命令和日志。
