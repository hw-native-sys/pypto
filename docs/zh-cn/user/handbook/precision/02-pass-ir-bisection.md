# 精度定位：Pass-IR 二分

> **状态：** 草稿骨架。定位*第一个*使 IR 偏离预期结果的 pass。

## 症状

直接从程序 IR codegen 是正确的，但经过 `PassManager(Default)` 后 codegen 错误
—— 因此是某个 pass 引入了发散。

## 工具

- `--dump-passes` + `PassDumpLevel` —— 在每个 pass 之后 dump IR（含当前分支新增的
  显式 layout dump 级别）。
- `CompiledProgram.validate_ir` —— 将每个 dump 出的 pass IR 与黄金值校验。

*TODO —— 列出各 `PassDumpLevel` 取值及其捕获内容。*

## 步骤

*TODO：*

1. 以合适的 `PassDumpLevel` 运行 `--dump-passes`。
2. 对每个 dump 阶段做 codegen + 校验。
3. 定位第一个校验失败的 pass。
4. 阅读该 pass 的开发文档（`docs/zh-cn/dev/passes/NN-*.md`）以理解其变换。

## 如何读输出

*TODO —— dump 目录布局、文件命名，以及如何 diff 相邻两个阶段。*

## 参见

- [Torch Codegen 调试](01-torch-golden.md)
- 开发者参考：[`dev/debug/00-torch_codegen.md`](../../../dev/debug/00-torch_codegen.md)、[`dev/passes/00-pass_manager.md`](../../../dev/passes/00-pass_manager.md)
