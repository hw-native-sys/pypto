# 产物身份基础

内部模块 `pypto._identity` 为 [RFC #2653](https://github.com/hw-native-sys/pypto/issues/2653)
提出的持久 JIT 缓存提供内容哈希和确定性记录编码。
目前未接入 JIT 调用或产物查找，现有进程内缓存行为及每次调用的开销保持不变。

这是身份阶段的第一部分。实际编译器、SDK 资源和动态依赖的自动清单仍需后续实现。
运行时已有的编译器版本 token 不是完整内容身份，不能直接作为持久缓存键。

## 带类型的记录

`encode_record()` 使用带版本和类型标签的编码；`digest_record()` 返回完整 SHA-256 摘要。
支持 `None`、布尔值、整数、浮点数、字符串、字节串、列表、元组和字符串键字典。
字典插入顺序不影响结果，序列顺序影响结果。

```python
from pypto._identity import digest_record

assert digest_record(True) != digest_record(1)
assert digest_record(1) != digest_record(1.0)
assert digest_record(0.0) != digest_record(-0.0)
assert digest_record({"rows": 32, "cols": 64}) == digest_record({"cols": 64, "rows": 32})
```

浮点数保留 IEEE-754 位表示，包括有符号零和 NaN payload。
字符串值和字典键保留 Python 码点，在 JSON 序列化前区分非 BMP 字符和显式代理对。
不支持的对象、非字符串字典键及循环引用会报错，不使用 `str()`/`repr()` 兜底。
适配层必须显式转换枚举、路径和有效配置，并保留语义类型；修改编码时需提升身份 schema 版本。

## 文件和目录输入

`ContentRoot` 在构造时捕获绝对路径，保留 `..`，让文件系统正确解析前面的符号链接。
`fingerprint_content()` 读取文件原始字节，
并按排序后的路径递归枚举目录，保留输入根顺序及边界。
在编译器提供稳定的源码位置和 include 路径映射之前，路径仍参与身份；
相同内容位于不同路径时可能不命中。

目录清单排除 `.git`、`__pycache__`、`.pyc` 和 `.pyo` 元数据，其他资源全部参与。
`fingerprint_extra_sources()` 对目录只纳入 Python 源码；直接指定的文件不受扩展名限制。
额外源码列表可以为空，但必需的安装清单为空时身份不可用。

符号链接（symlink）计入实际目标路径及目标内容。断链、目录循环、非普通文件、
不可读输入或检测到的读取期间变更，都会返回不可用摘要和原因，不能静默视为空文件或省略依赖。
文件元数据只用于发现竞争，不能替代内容身份。这不是原子的文件系统快照：
后续接线必须在发布前重新验证可变源码，进程内的安装输入必须保持不可变。

应用额外源码在每次请求重新读取。应用提供的额外指纹只能补充输入，
不能替代缺失文件或工具链证据。

## 安装清单与缺失证据

`ToolchainInputs` 包含五个必需分量：PyPTO、runtime、PTO-ISA、ptoas，
以及设备/编排工具链。`ComponentInputs` 默认不可用，即使已经知道部分文件路径。
具备依赖发现能力的适配层必须确认清单完整后，才能清除 `unavailable_reason`。

适配层必须涵盖实际导入代码、原生库、编译器资源、动态依赖、头文件、SDK/sysroot 和链接输入。
工具解析必须与实际编译路径共用，包括 ptoas 启动脚本及有效编译器选择。
文件哈希只证明该文件的身份，不能证明它代表完整依赖集合。

`InstallationIdentityCache.capture()` 报告所有不可用分量。
任一清单不完整或不可读时，结果为 `usable=False`、`digest=None`，没有共享的 `UNKNOWN` 键。
应用额外指纹不能把这个结果变成可用的工具链身份。

成功的内容读取按完整解析清单记忆化，并在线程间同步。工具选择改变时必须形成新清单；
读取失败会在下次重试，不永久缓存失败结果。在相同安装路径替换代码、库或工具需要重启进程。
应用源码刷新不使用这种记忆化。

## 环境变量分类

`python/pypto/_environment.json` 登记环境输入及理由：

| 类别 | 接线时必须完成的处理 |
| ---- | -------------------- |
| `semantic` | 按实际优先级解析有效值，纳入编译输入。 |
| `tool_resolution` | 识别实际工具及依赖内容，不能只哈希搜索路径字符串。 |
| `fresh_request` | 查缓存前满足重新编译、检查或输出请求。 |
| `nonsemantic` | 仅按已记录的理由排除，例如终端格式或日志。 |

登记表是审计清单，不是无差别哈希全部环境，也不代表上述策略已经接线。
已有诊断旁路行为见 [JIT 函数](language/03-functions.md#编译选项与诊断请求)。

`tests/lint/check_environment_inputs.py` 在 pre-commit 中运行，不加载原生扩展。
它检查 `python/pypto`、`python/bindings`、`src` 和 `include` 中的 Python 读取，
以及 C++ `getenv`/`secure_getenv` 调用。识别的 Python 写法包括导入、别名、
模块字符串常量、映射读取和批量读取。别名按词法作用域解析；
存在歧义的模块常量赋值（包括控制流内的写入）仍按动态读取处理。
该 hook 使用 Python 3.10。动态读取必须有精确文件/函数及理由的例外；
该例外不能隐藏新出现的字面量变量，未使用的例外也会导致检查失败。

静态检查不能证明下游工具或任意 Python 反射代码的依赖完整性。
登记表还列出 `PATH`、编译器 include/library 搜索变量、加载器注入和 locale 等非 `PYPTO_*` 输入；
工具链适配层必须覆盖这些输入后，才能声称身份完整。
后续接入持久查找时，不支持的依赖发现必须保持不可用。
