# 产物身份基础

内部模块 `pypto._identity` 为 [RFC #2653](https://github.com/hw-native-sys/pypto/issues/2653)
提出的持久 JIT 缓存提供内容哈希和确定性记录编码。
显式启用的 [JIT 集成](10-jit-cache.md) 为支持的 Linux 工具链提供自动清单。
不支持的依赖发现仍报告不可用。运行时编译器版本 token 不是内容身份，不能替代依赖内容。

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
JIT 适配器在发布前重新验证可变源码，进程内的安装输入必须保持不可变。

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

### 外部已验证的版本 {#externally-verified-revisions}

`ComponentInputs.verified_revision` 是"读取内容"这一规则的唯一例外。适配器只有在
**同一次解析中**已由其他机制证明该组件的字节**就是**该版本时，才可以设置它。安装
自报的版本不构成这种证明，路径里带版本号同样不构成。

目前只有 PTO-ISA 满足条件。`ensure_pto_isa_root()` 只在证明 checkout 干净且位于
pin 提交之后才返回，否则从 pin 重新 clone；git 对象按内容寻址，因此"干净且
`HEAD == pin`"的工作树与 pin 逐字节相同。再哈一遍这棵树，是在重复证明该解析已经
确立的事实。

该解析用 `git status --porcelain` 判定干净与否，而它不报告 checkout 所忽略的
路径——若有构建把生成文件写进 ISA 树，它不会被惊动。因此适配器会带 `--ignored`
再问一次，同时也重新覆盖了被跟踪状态：**凡是 git 无法交代的东西**（被忽略的、
未跟踪的、已修改的）都会让该组件退回内容清单。这次检查约 18 毫秒，而内容读取是
0.71 秒。

适配器读不到版本、或拿不到关于该树的可用答复时，同样回退；`unavailable_reason`
的优先级仍高于版本，因此不完整的组件依旧不可用。

已验证的版本与组件名一并记录，因此一个组件的版本不可能与另一个组件的版本、或与
任何内容摘要产生相同的摘要。

### 自报版本 {#self-reported-versions}

`ComponentInputs.reported_version` 是有意更弱的一档，名字就是它的实质：安装**自己
声称**的版本，没有任何东西验证它。它能区分"自称不同"的安装；它看不见版本没动而
字节变了的情况，因此原地重编译或就地打补丁对它是不可见的。

ptoas 用的就是这一档。没有任何东西证明该安装的字节：它是由 `PTOAS_ROOT` 选中的
外部目录树，其发布物不带安装器会校验的清单，而 `toolchain/versions.env` 里的
sha256 指的是下载的 wheel 文件，不是解包后目录树里可达的任何东西。因此这个身份
建立在一条部署性质上——ptoas 是以未经修改的官方发布形态到达的——而不是建立在
PyPTO 能检验的证据上。

身份取的是 `--version` 的**完整输出**，不是从中解析出的数字。版本准入检查只保留
数字段，因此 dev 构建的后缀——恰恰是它与所源出的正式版之间唯一的区分标记——会被
丢弃，两者就会共用同一个身份。

`reported_version` 与 `verified_revision` 以不同的标签记录，因此同一个字符串放在
两者之下会产生不同的摘要：自报的声明永远无法冒充证明。探测失败时回退到内容清单；
`unavailable_reason` 的优先级仍高于两者。

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
持久查找期间，不支持的依赖发现必须保持不可用。
