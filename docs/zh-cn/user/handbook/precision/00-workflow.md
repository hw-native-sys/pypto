# 精度定位：工作流

**症状：** 你的 kernel 能编译能运行，但输出与参考（torch 黄金值、已知正确的构建或
手算）发散。本页是把你导向正确定位工具的决策树。

## 核心思路

PyPTO 经由一条 pass 流水线下降你的程序。错误结果是在这条路径*某处*引入的 —— 要么
在你写的程序里，要么在某个变换它的具体 pass 里。精度定位就是**二分出发散首次出现
的位置**，然后只读那个阶段。

三种工具覆盖整条路径：

| 工具 | 回答的问题 | 章节 |
| ---- | ---------- | ---- |
| torch 黄金对比 | “是源程序错，还是后续阶段错？” | [01](01-torch-golden.md) |
| Pass-IR 二分 | “哪个 pass 第一个破坏了结果？” | [02](02-pass-ir-bisection.md) |
| 选择性张量 dump | “这个张量在设备上到底是什么值？” | [03](03-selective-dump.md) |

**两条 codegen 路径，在此扮演不同角色。** *torch codegen* 把 IR 重新表达为可运行的
PyTorch 并在 host 上执行 —— 一种无需设备的快速数值**仿真**。**步骤 1 与 2 都用
torch codegen**（分别作用于 pass 前 IR 与 pass 后 IR），因此定位的是你的程序或某个
pass 引入的发散。*pto codegen* 才是真正的设备路径（`.pto` → AICore）；**步骤 3 用
pto codegen 在设备上**捕获那些只在真实 codegen / 硬件执行中出现、而 torch 仿真里
不出现的发散。

## 决策树

```text
输出错误
│
├─ 1. 对你的程序 IR 做 TORCH codegen，与黄金值对比
│      （torch_codegen(program) → 对比）                                → 章节 01
│      │
│      ├─ 已经发散？   → bug 在你的程序 / op 用法里。
│      │                 修源程序；不涉及任何 pass。
│      │
│      └─ 与黄金一致？ → 是某个 pass 引入了发散。转 2。
│
├─ 2. 经过 PassManager(Default) 后做 TORCH codegen，与黄金值对比
│      （torch_codegen(PassManager…run_passes(program)) → 对比）        → 章节 01
│      │
│      ├─ 与黄金一致？ → 发散在 torch codegen 之下 —— 只在
│      │                 pto codegen / 设备侧出现。转 3。
│      │
│      └─ 发散？       → 二分 pass dump，定位第一个出错的 pass          → 章节 02
│                        （每个阶段都用 TORCH codegen 校验）。
│                        validate_pass_ir_codegen_results(...)
│                        或 compiled.validate_ir(...)
│
└─ 3. PTO codegen 设备侧 —— 需要某个可疑张量的真实值？                  → 章节 03
       用 pl.dump_tag 打标，打开 runtime-DFX 选择性 dump，
       （L2 swimlane 双跑做板上对比），然后 diff。
```

## 逐步操作

1. **建立黄金值。** 构造输入与参考输出（如一个 torch 函数）。
   [章节 01](01-torch-golden.md#build-inputs-and-golden-output) 有一份完整的
   `build_tensors` / `golden` 示例。

2. **对程序 IR 做 torch codegen** 并对比
   （[01 › 模式 1](01-torch-golden.md#1-codegen-directly-from-program-ir)）。若此处
   已发散，问题在你的源程序 —— 到此为止并修复它。

3. **经过默认 pass 流水线后做 torch codegen** 并对比
   （[01 › 模式 2](01-torch-golden.md#2-codegen-after-passmanagerdefault)）。若模式 1
   正确但此处发散，则某个 pass 是元凶。

4. **二分 pass dump** 定位*第一个*产生错误结果的 pass —— 每个 dump 阶段都用 torch
   codegen 校验（[章节 02](02-pass-ir-bisection.md)）。有 `CompiledProgram` 时只需
   一次调用：

   ```python
   compiled = ir.compile(MyProgram)          # dump_passes=True 为默认
   compiled.validate_ir(tensors, expected)   # 逐 pass 数值校验（torch codegen）
   ```

   报告会指出第一个发散的 pass（如 `19_after_ExpandMixedKernel`）；阅读该 pass 在
   [`dev/passes/`](../../../dev/passes/00-pass_manager.md) 的开发文档以理解其变换。

5. **用 pto codegen 在设备侧确认**（torch 仿真正确、板上错误，即发散在 torch codegen
   之下）：用[选择性 dump](03-selective-dump.md) 捕获可疑张量并与黄金值 diff。这种
   “torch 里正确、设备上错误”的特征，一个常见成因是**丢失的 WAR 依赖** —— 深入排查
   前先看下方的陷阱说明。

## 已知陷阱：丢失的 WAR 依赖（循环携带缓冲）

有一类调度隐患会产生**静默、间歇**的精度错误，而步骤 1–2 *无法*捕获：torch codegen
顺序执行，竞态在仿真中永不复现 —— 只有在设备上（步骤 3）才暴露。

**成因。** 在 AUTO 运行时作用域中，运行时调度器目前对“分配一次、跨循环迭代携带”的
缓冲会遗漏 **WAR（write-after-read，读后写）** 反依赖。对一个每次迭代先写后读的
缓冲：

- `writer(N)`（inout）→ `reader(N)`（纯 `Input`）：RAW 边已发出 ✓
- `writer(N)` → `writer(N+1)`（循环携带）：WAW 边已发出 ✓
- `reader(N)` → `writer(N+1)`：**WAR 边缺失 ✗**

因为 `reader(N)` 不产生缓冲的新版本，`writer(N+1)` 只依赖 `writer(N)` —— 从不依赖
`reader(N)`。于是调度器可能让 `writer(N+1)` 与 `reader(N)` 并发运行，在读取过程中
覆盖缓冲 → 数据竞态与结果污染（实测约 10–35% 数值错误，例如跨 MLP band 共享的
gate/累加器缓冲发生竞态）。

**症状。** torch codegen（步骤 1–2）正确；设备上错误且*不确定*。

**当前修法：手动建立 WAR 边**，用 `pl.submit(..., deps=[...])` 让 `reader(N)` 先于
`writer(N+1)` 覆盖缓冲完成：

```python
# reader(N) 必须在 writer(N+1) 覆盖 `buf` 之前完成
_, tid_read  = pl.submit(self.reader, buf, ...)
_, tid_write = pl.submit(self.writer, buf, ..., deps=[tid_read])
```

自动检测 / 防护由 **issue #2058** 跟踪。在其落地前，PyPTO 在 AUTO 作用域下既不告警
也不自动修复此情形 —— 你必须自己加这条边。见
[性能 › 依赖与分发](../perf/03-dependency-dispatch.md)。

## 参见

- [Torch Codegen 调试](01-torch-golden.md) —— 黄金对比配方
- [Pass-IR 二分](02-pass-ir-bisection.md) —— `--dump-passes` / `validate_ir`
- [选择性张量 dump](03-selective-dump.md) —— `pl.dump_tag` + runtime-DFX
- 开发者参考：[`dev/debug/00-torch_codegen.md`](../../../dev/debug/00-torch_codegen.md)
