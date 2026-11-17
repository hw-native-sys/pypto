# A5 RegScope 寄存器溢出策略

## 目的

A5 `reg_scope` helper 需要 VF stack 保存寄存器溢出数据时，应给出清晰的编译失败信息。默认策略是在设备端 Bisheng 编译时加入 `--cce-disable-vf-stack-reserved-ubuf`，将 VF stack 大小设为零；需要 spill 的 kernel 会在编译阶段失败。用户可以显式允许 spill，以恢复原有行为。

该策略只作用于 A5 上由 `reg_scope` 生成的 VPTO helper，不改变 A2/A3、非 RegScope kernel 或独立 SIMT helper。

## 已复现的编译行为

归档 `build/register-spill-case-20261116` 使用 PTOAS v0.74 和 `--vpto-scheduler=off` 生成 LLVM，再用以下 Bisheng 参数编译：

```text
--cce-long-scbz=true
-mllvm -cce-dyn-kernel-stack-size=true
-mllvm --cce-res-usage
```

不加 guard 时编译成功，并报告：

```text
Stack size: 672 bytes, VReg number: 13, MaskReg(PReg) number: 7
```

反汇编中可见 `PSTI`/`PLDI` predicate spill/reload。加入 `--cce-disable-vf-stack-reserved-ubuf` 后 driver 返回 1（内部 frontend 报告退出码 70），并报告：

```text
The total stack object size (704) exceeded vf stack size (0).
Total Spilled Byte Size: 704
```

该用例说明，源码中的 mask 数量不是压力指标；真正触发 spill 的结构是 predicate 通过循环 PHI 携带，并在循环中使用 `pand` 形成较长 live range。

## API 与参数传递

在编译配置中增加 `allow_reg_spill: bool = False`。目标为 A5 且生成单元含 `reg_scope` 的 `vf` helper 时，默认加入该 Bisheng 参数；只有 `allow_reg_spill=True` 时才移除。

有效策略和最终 flags 必须参与 native build cache key，并覆盖普通编译、replay 和 debug rebuild。guard 编译失败后不得自动用允许 spill 的配置重试。

## 错误信息约定

只有启用 guard 且 Bisheng 输出包含 `exceeded vf stack size (0)`（允许空白差异）时，才分类为 `RegisterSpillError`。其他 Bisheng 错误保持普通编译错误。

摘要应包含 kernel、helper、可用的源位置、编译阶段和处理建议：

```text
RegisterSpillError [PYPTO-REG-SPILL]
A5 reg_scope compilation requires register spilling, but register spilling is
disabled by default.

Helper: <artifact>/vpto/<unit>/<helper>.pto
Stage: Bisheng device compilation

Bisheng diagnostic:
fatal error: error in backend: The total stack object size (704) exceeded vf stack size (0). The spilled objects are:

Full compiler diagnostics: <artifact>/vpto/<unit>/report/<helper>-spill.txt

Reduce simultaneously live vector/predicate values or split the reg_scope.
To allow spilling, set allow_reg_spill=True in RunConfig or CompileOptions.
```

完整命令与编译器原始错误（包含其实际输出的版本、spill 字节数和资源诊断）写入持久化报告。没有源 span 时只显示 helper 和 `.pto` 路径，不编造 Python 行号。

## 实现与验证细节

后端对生成的 VF helper 写入 `// pypto.reg_spill: deny` 或 `// pypto.reg_spill: allow` 源码注释，SIMT 不写该标记。运行时只在 A5 且存在 deny 标记时添加 guard。源文件参与已有缓存摘要，JIT 缓存也显式包含 allow_reg_spill，因此迁移产物、replay 和不同策略不会串用缓存。没有标记的外部 helper 保持原有行为。manifest 的 hash 包含完整注释。

终端仅展示 VF stack 超限的关键诊断；完整命令和编译错误保存在 helper 旁的 `report/<helper>-spill.txt`，LLVM 输入保存在同名 `.ll` 文件。没有可靠 Python 源位置时显示真实 helper 路径。日志写入失败则异常保留全部诊断。`Predicates: 0` 的分类语义尚未确认，不能解释成 spill 数量。

回归验证覆盖布尔参数校验、配置传递、JIT/native 缓存隔离、VF/SIMT 混合标记、manifest hash、Bisheng 命令、错误分类、持久日志和普通编译失败。编译验证使用循环携带 mask 的归档用例，无需 NPU 执行。

真实编译验收用例位于 `tests/ut/backend/vpto_vf_validation/{mask_spill,no_spill}.pto`。配置 PTOAS 的 PATH 或 `PTOAS_ROOT` 后，将 `PYPTO_SPILL_TEST_BISHENG` 设置为 Bisheng 可执行文件路径，执行 `pytest tests/ut/runtime/test_kernel_compiler.py -k reg_spill_real_bisheng`。这些用例使用空 UB 基地址，仅能编译，不能执行。
