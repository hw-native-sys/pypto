# 调试

用于缩小"已编译程序从哪里开始与参考实现不符"范围的工具。

| 页面 | 内容 |
| ---- | ---- |
| [Torch 代码生成](00-torch_codegen.md) | 把 PyPTO IR lower 成可执行的 Python/PyTorch 脚本用于数值校验 |

## 另请参阅

- [Torch Codegen 调试指南](../../user/tools/01-torch-codegen.md) —— 从用户视角看同一个工具。
- [错误处理](../02-error-handling.md) —— 异常类型与失败信息中的 IR 源码位置。
- [日志](../03-logging.md) —— 两套日志子系统及其详细级别调整方式。
- [运行时 DFX 开关](../03-runtime-dfx.md) —— 运行时侧诊断，含选择性张量 dump。
- [IR 验证器](../passes/99-verifier.md) —— 在 pass 之间捕获非法 IR。
