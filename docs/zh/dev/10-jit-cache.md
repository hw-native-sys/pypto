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

这里 `decode` 有完整张量注解及标量默认值。也可以按 `compile()` 的参数规则传入
样例张量和标量。预热（warmup）构建所有必要二进制，不初始化 NPU、不执行 kernel；
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

首个适配器支持 Linux ELF GCC 工具链、CANN BiSheng 布局、独立 ELF PTOAS，以及
打包 CPython PTOAS 的启动脚本语法及带 NumPy 依赖的 PTOAS wheel 标准 pip/uv 入口脚本。
wheel 清单覆盖实际虚拟环境与解释器、全部安装包资源、启动输入及原生依赖；拒绝导入重定向。
标识覆盖安装内容、编译器子程序与资源、隐式包含
目录、链接输入、Python/native 运行时文件以及 ELF 动态依赖。未知启动脚本、在线构建的
PTOAS 扩展、不支持的编译器布局、sanitizer 构建，以及 `CPATH`、`LD_PRELOAD` 等隐式
依赖覆盖会旁路持久缓存。最新原因可通过 `cache_stats().last_bypass_reason` 获取，
无需开启 INFO 日志。每次旁路也由 `pypto.jit._persistent` 以 INFO 级别记录。

隐式链接脚本支持绝对路径的 `INPUT`/`GROUP` 依赖、嵌套 `AS_NEEDED` 以及
`OUTPUT_FORMAT`/`OUTPUT_ARCH` 声明，按选定的 sysroot 递归追踪依赖。
相对路径、`-l` 名称、`SEARCH_DIR`、`INCLUDE` 和未知语法需要完整的链接器搜索上下文，
当前会旁路持久缓存，继续普通私有编译。

成功的安装标识在每个进程内按工具选择和解析后的组件清单记忆化（memoization）。
工作目录变化会重新发现工具，因为相对搜索路径可能选择不同工具；清单未变的组件直接
复用内容摘要。进程存活期间安装文件须保持不变，替换
后重启进程。额外应用源码每次请求重新读取。当前路径也参与标识，移动安装可能未命中。

首次内容清单读取较保守，可能耗时较长。一套本地 CANN/PTOAS 安装的首次清单约需
12 秒；该测量不代表通用性能结论。当前每个独立新进程都需要支付这项成本。仅依赖路径、
大小、mtime 和 inode 的磁盘记忆不能证明内容未变，因此不采用。统计包含标识计算和
校验时间。没有可验证本地工具链的
部署需要另一套协议，本 API 尚不支持。

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
标量为有限 JSON 数值或布尔值，遵循普通标量默认值规则。

可序列化的 `run_config` 字段为 `platform`、`strategy`、`memory_planner`、
`distributed_config`、`dump_passes`、`dump_ptoas_passes`、`save_kernels`、
`save_kernels_dir`；枚举使用 Python 成员名。分布式设置支持 `device_ids`、
`num_sub_workers`、`runtime`、`aicpu_thread_num`。预热结果区分共享及私有准备，失败
返回非零退出码。`stat` 仅读取 JSON 清单及声明的负载大小，不导入缓存代码、不获取
写锁、不修复条目。

内部契约参见[产物标识](08-artifact-identity.md)与[不可变存储及运行时协议](09-artifact-store.md)。
