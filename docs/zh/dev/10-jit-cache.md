# JIT 持久缓存

持久缓存（persistent cache）默认关闭，可跨进程复用生成代码和完整二进制。
每个 JIT 函数还保留进程内编译对象。缓存包含可执行代码，因此必须信任缓存写入方。

```python
from pathlib import Path

import pypto
from pypto.runtime import RunConfig
from my_kernels import decode

pypto.configure_cache(pypto.CacheConfig(
    enabled=True,
    root=Path("kernel-cache"),
    extra_source_paths=(Path("kernels"),),
    extra_fingerprint="model-config-v1",
))
prepared = decode.warmup(config=RunConfig(platform="a2a3"))
print(pypto.cache_stats())
```

这里 `decode` 有完整张量注解。也可以按 `compile()` 的参数规则传入样例张量和标量。预热（warmup）构建所有必要二进制，不初始化 NPU、不执行 kernel；
构建机器仍需要目标编译器、SDK 和主机运行时。

## 请求流程

1. 捕获特化、源码命名空间、有效编译选项和缓存策略。诊断及显式输出请求重新编译。
2. 建立源码、额外应用输入和工具链的完整内容标识。无法验证的输入报告旁路（bypass）。
3. 查询兼容的进程内对象；持久对象还需要匹配捕获的缓存根目录及只读策略。
4. 校验 GENERATED 元数据并查询完整 READY 契约，优先复用二进制，尚未发布二进制则恢复生成代码。
5. 未命中时，在按 key 去重的事务中私有构建、打包 extern、重新检查可变源码，发布不可变 GENERATED。
6. 普通执行或 warmup 完成二进制事务并发布 READY。后续进程加载 READY 时不运行编译阶段。

`compile()` 不执行 kernel，也不承诺二进制完整。普通执行自动发布缺失二进制，调用方
无需单独 load/store。`specialize()` 和 `lower()` 始终直接生成 IR。新编译对象保留
`.program`；从缓存恢复的对象 `.program is None`，包括其他进程赢得构建竞争的情况。
需要 `compile()` 返回 IR 时关闭持久缓存；此时单独选择兼容的私有对象。

可写缓存中的构建跨协作进程去重。私有回退只合并同一进程内重叠请求；无效或不可写缓存、
只读未命中不提供跨进程私有构建去重。编译错误向调用方传播，允许重试。不支持的 extern
打包或构建期间变化的应用源码保留为私有结果。

## Program 构建职责

`pypto.runtime.kernel_compiler.KernelCompiler` 负责调用编译器、链接、临时输出目录
和二进制校验。它查询已安装 Simpler SDK 的元数据，不再继承 SDK 编译器或调用其
构建方法。这些行为适用于现有 runtime pin，不引入 kernel 执行，也不改变 program 调用语义。

| 原来继承的职责 | 当前负责方 |
| -------------- | ---------- |
| SDK 根目录、工具选择、目标参数、runtime 头文件和辅助源码 | Simpler 元数据查询，由 PyPTO 消费 |
| AICore 编译和 `kernel_entry` 链接 | PyPTO `KernelCompiler.compile_incore` |
| 模拟器 kernel 共享库 | PyPTO `KernelCompiler.compile_incore` |
| Orchestration 共享库、Build-ID 和 Host 线程参数 | PyPTO `KernelCompiler.compile_orchestration` |
| 成功或失败后的临时输出校验与清理 | PyPTO；可选 `build_dir` 指定临时目录的父目录，不保留中间文件 |
| Callable 组装、二进制发布与恢复 | 现有 PyPTO device runner、prebuilt loader 和 artifact store |

HBG orchestration 使用 Host 编译器；TRB 在模拟器上使用 Host 编译器，在真实设备目标上
使用 AArch64 编译器。SDK 声明的辅助源码必须存在，缺失时在调用编译器前报错。
编译命令保留 SDK 的相对路径形式和工作目录，生成的输出均放在 PyPTO 管理的临时目录
或产物目录中。构建和恢复不初始化 Worker，也不执行业务逻辑。

每次编译器或链接器调用的超时时间默认为 900 秒。创建编译器前可通过
`PYPTO_COMPILER_TIMEOUT` 设置正的有限秒数来覆盖默认值。该限制针对单次调用，
不是整个 program 构建的总时限。超时抛出标明构建阶段的 `RuntimeError`，
清理临时输出，失败的构建不会发布缓存有效标记。

可变 program 输出目录使用 binary-context schema 2。成功的事务记录构建上下文和可复用
二进制文件的 SHA-256 摘要。下一次事务保留校验通过的文件，删除内容变化或未记录的文件，
按需重建缺失文件。旧格式需要重建一次。组装前删除有效标记，成功后才重新发布，因此失败
或中断的事务不能让部分输出被视为有效产物。

持久 GENERATED 条目仍只证明源码身份，晋升时在私有目录中构建二进制，不信任继承来的
可变二进制作为 READY 依据。完整 READY 条目沿用已有清单校验，恢复时不调用编译器，
也不写缓存，包括只读恢复。目录锁和 key 锁保留现有并发契约，不引入第二套缓存存储。

## 配置

`CacheConfig` 不可变。完整的每次调用 `RunConfig.cache_config` 优先于
`configure_cache()` 进程默认值，再优先于环境配置。不同层级的字段不会部分合并。

| 字段 | 默认值 | 含义 |
| ---- | ------ | ---- |
| `enabled` | `False` | 启用持久查找和发布。 |
| `root` | `None` | 使用 `~/.cache/pypto/jit`；显式相对路径在捕获请求时解析。 |
| `readonly` | `False` | 缓存根目录内禁止写入、锁和字节码；私有构建及运行输出位于根目录外。 |
| `extra_source_paths` | `()` | 每次请求按内容哈希文件，或递归哈希目录中的 Python 源码。缺失输入旁路复用。 |
| `extra_fingerprint` | `None` | 额外应用版本或配置标记，不能替代缺失的工具链证据。 |

环境配置使用 `PYPTO_CACHE`、`PYPTO_CACHE_DIR`、`PYPTO_CACHE_READONLY`。
布尔值仅接受 `0` 或 `1`，非法值抛出 `ValueError`。`configure_cache(None)` 恢复
环境及默认配置优先级。进行中的请求保留已捕获的策略；配置变化不清零统计、不删除文件。

```python
readonly = RunConfig(cache_config=pypto.CacheConfig(
    enabled=True,
    root=Path("kernel-cache"),
    readonly=True,
    extra_source_paths=(Path("kernels"),),
    extra_fingerprint="model-config-v1",
))
compiled = decode.compile(config=readonly)
with_ir = decode.compile(config=RunConfig(cache_config=pypto.CacheConfig(enabled=False)))
```

切换存储策略时重复提供应用标识输入。只读未命中可以私有构建，因此 warmup 成功不代表
共享发布成功。运行目录及私有构建树随编译对象保持有效；没有在线清理或淘汰 API。
离线删除缓存条目或锁文件前须停止所有消费者。

缓存策略由 JIT 在选择编译对象前消费，不传入编译器选项或逐次启动选项。私有输出通常
位于工作目录的 `build_output`；若它位于缓存根内，则使用不可预测、权限为 0700 的临时
父目录隔离私有构建和运行输出，不对 `TMPDIR` 执行写探测。此回退父目录可能在缓存命中
时创建，并随编译对象保留，不自动清理。

## 工具链支持与开销

默认的 `PYPTO_CACHE_IDENTITY=build` 标识本次编译实际选择的工具，并与缓存构建时的
身份匹配；不会另外增加 expected 版本或 pin 一致性审计。

| 组件 | 身份依据 |
| ---- | -------- |
| PyPTO | Python 包源码与打包的代码生成模板内容、实际导入扩展的完整 GNU ELF Build-ID、Python 版本及 ABI。 |
| Runtime | 干净的源码 checkout 版本或安装构建版本、实际原生扩展 Build-ID、Python 源码及可用的 runtime/PTO-ISA 构建元数据；源码 checkout 有未提交更改时绕过持久缓存。 |
| PTO-ISA | runtime 的 `pto_isa.pin` 选定的版本；仅编译未命中时获取并校验 checkout。 |
| PTOAS | 标准 wheel 启动器使用其解释器实际选中的包元数据、NumPy wheel 记录、原生编译器 Build-ID、启动 `.pth` 文件与非标准库启动模块；缺少启动或 wheel 证据、不支持的启动器会绕过持久缓存。独立 ELF 使用 Build-ID。 |
| 设备及编排工具链 | 选中的编译器路径/版本、调用入口及实际执行的 GCC 驱动的 Build-ID、GCC 辅助程序 Build-ID、CANN 安装构建版本及链接器 Build-ID；无法识别编译器 wrapper 或缺少 CANN 构建版本时绕过持久缓存。 |

原生文件缺少可用 Build-ID 时回退到内容哈希。Build-ID 从小型 ELF note 读取，不读取
整个动态库。wheel PTOAS 探测不导入其编译器包；选中解释器的启动搜索路径和钩子文件
也参与 key。身份不可用时绕过缓存，不生成共用的 `UNKNOWN` key。现有 runtime ABI 及 PTOAS
最低版本检查仍在原有路径执行。
带有 `_online` 构建目录的 PTOAS 包会绕过持久缓存，因为本地扩展重编可能不会改变
报告的版本。

Python 根目录和原生扩展位置来自实际导入结果，支持 `pip install`、`pip install -e`
及 `PYTHONPATH`，包括源码 Python 配合 editable 安装扩展的布局。不单独依赖包分发
元数据来判断实际导入的 PyPTO 编译器。

此策略信任发布的原生构建/版本标识，不扫描系统头文件、Python 标准库或传递动态库
依赖。源码 runtime checkout 必须没有已跟踪或未跟踪的 Git 更改。若修改已安装的输入而
没有更新版本标识，应设置新的 `PYPTO_CACHE_EPOCH` 或清空缓存。进程内安装文件应保持
不变，替换后重启进程。
应用的额外源码仍逐请求重新读取。有效选择输入（路径、环境覆盖等）参与 key，移动
安装位置可能导致未命中。

`PYPTO_CACHE_IDENTITY=content` 保留之前的 Linux 依赖清单：ELF 依赖闭包、编译器资源、
隐式 include、链接输入和 Python/native runtime 内容。它仍保留原有 PTOAS/CANN
报告版本及经验证 PTO-ISA revision 的快捷路径，不代表对所有厂商文件逐字节审计。
不支持的启动器、sanitizer 构建以及 `CPATH`、`LD_PRELOAD` 等未建模隐式覆盖会绕过
缓存。两种策略均通过 `cache_stats().last_bypass_reason` 报告原因，身份相互隔离。

缓存产物仍进行完整 manifest/内容校验。新打包的产物携带 `kernel_config.json`，READY
发现无需执行 Python 配置。恢复复用 lookup 已验证的 manifest，直接构造原生 callable，
无需导入 Worker/通信初始化模块。
若 GENERATED 槽缺失或损坏，查找仅扫描同一 artifact key 下最多 32 个 READY 阶段目录。
每个候选目录自身的 JSON 必须推导出与目录一致的 spec 摘要，且通过完整 manifest/产物
校验；多个有效候选会被拒绝。

用独立进程测量首次命中：

```bash
PYTHONPATH=python python tests/benchmarks/jit_cache_latency.py \
  --cache-root /tmp/pypto-jit-benchmark --runs 5 --output build/jit-latency.json
```

验证 wheel 时使用安装环境的 Python 并去掉 `PYTHONPATH`。第一个子进程填充缓存，
其余每个测量进程都必须 READY 命中且无构建、无 bypass。计时包含首次身份检查、首次
warmup 及 callable 恢复，不包含进程启动、初始 import、张量创建和设备执行。慢速探测
回退、文件系统冷页或大型产物可能超过 100 ms；这是测量目标，不是延迟保证。
没有使用基于文件时间戳的跨进程身份 memo。

## 统计与 CLI

`cache_stats()` 返回不可变、线程安全的进程内快照，不扫描磁盘、不清零。
计数包括 `requests`、`object_hits`、`ready_hits`、`generated_hits`、`misses`、
`invalid_entries`、`storage_errors`、`generation_builds`、`binary_builds`，
耗时累计为 `lookup_ns` 和 `build_ns`。以下字段区分未使用持久缓存的原因：

| 字段 | 含义 |
| ---- | ---- |
| `disabled_requests` | 未开启持久缓存的请求，包括私有对象命中。 |
| `forced_rebuilds` | 诊断或显式输出要求强制编译的请求，与是否开启持久缓存无关。 |
| `bypasses` | 已开启持久缓存，但标识或打包不可用、源码变化导致的回退。 |
| `last_bypass_reason` | 最近一次已开启缓存的旁路原因；尚未发生时为 `None`。 |

未开启缓存和强制重编不增加 `bypasses`。这些计数并非互斥：未开启持久缓存的诊断请求
同时增加 `disabled_requests` 和 `forced_rebuilds`。初始查找每请求计数一次，锁内复查
不增加请求数。未命中后遇到不支持的打包输入，还会增加一次旁路计数。无效条目与存储
错误为额外事件；存储错误根据结构化状态统计，不依赖诊断文字。构建计数反映实际阶段，
耗时不包含设备执行。比较数值字段的快照差值得到区间统计；`last_bypass_reason` 是累计上下文。

```bash
python -m pypto.jit warm --module my_kernels --config warmup.json
python -m pypto.jit stat --root kernel-cache
```

```json
{
  "schema_version": 1,
  "cache": {"enabled": true, "root": "kernel-cache"},
  "requests": [
    {
      "kernel": "decode",
      "run_config": {"platform": "a2a3"},
      "tensors": {"x": {"shape": [1, 128], "dtype": "FP16"}},
      "scalars": {"block_size": 128}
    }
  ]
}
```

CLI 导入指定的可信模块，只解析显式列出的模块级 JIT 函数，在任何构建前校验完整列表。
路径相对于配置文件；张量元数据使用无需数据分配的 meta tensor，注解完整时可以省略。
支持 `FP16`、`BF16`、`FP32`、`INT8`、`INT16`、`INT32`、`INT64`、`BOOL`。
标量为有限 JSON 数值或布尔值；它们只用于完成绑定，不再选择产物——标量参数是运行期值。
除非请求同时给出样例张量（此时绑定按位置进行），否则可以省略标量。

可序列化的 `run_config` 字段为 `platform`、`strategy`、`memory_planner`、
`distributed_config`、`dump_passes`、`dump_ptoas_passes`、`save_kernels`、
`save_kernels_dir`；枚举使用 Python 成员名。分布式设置支持 `device_ids`、
`num_sub_workers`、`runtime`、`aicpu_thread_num`。预热结果区分共享及私有准备，失败
返回非零退出码。`stat` 仅读取 JSON 清单及声明的负载大小，不导入缓存代码、不获取
写锁、不修复条目。

内部契约参见[产物标识](08-artifact-identity.md)与[不可变存储及运行时协议](09-artifact-store.md)。
