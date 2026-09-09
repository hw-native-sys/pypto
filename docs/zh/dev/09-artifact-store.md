# 不可变产物存储

内部模块 `pypto.jit.artifact_cache` 实现了
[RFC #2653](https://github.com/hw-native-sys/pypto/issues/2653) 的存储阶段，提供
manifest 校验、按 key 加锁、不可变发布和私有构建回退。目前尚未接入 JIT 分发。
[产物身份基础](08-artifact-identity.md) 中完整工具链清单的自动发现、公共缓存配置和预热
仍属于独立的后续阶段。下文的显式运行时适配器实现设备阶段晋级和只读加载，但不会
自动启用持久化缓存。

## 适配器契约

`ArtifactKey` 要求可用且使用当前身份 schema 的 `ToolchainIdentity`，以及完整的
SHA-256 源码和特化摘要。其记录保留每个环境组件摘要和两个请求摘要，完整 key 摘要还包含
产物 schema。身份缺失、摘要截断或格式错误会抛出 `ValueError`。此处只验证表示形式，
不证明适配器提供的源码或工具链清单完整。

`ArtifactSpec` 指定 `GENERATED` 或 `BINARY_READY`、单芯片或分布式构建类型，
以及非空且无重复的必需文件相对路径列表。存储层将阶段、构建类型和排序后的必需文件列表
共同计算出的摘要纳入 slot 路径。因此 spec 变化会选择新 slot 并正常未命中，
不依赖适配器的特化摘要约定；原 spec 仍可复用。
生成阶段必须列出全部必需源码和配置文件；二进制就绪阶段必须列出全部二进制及完整加载元数据。
仅将目录标记为 `BINARY_READY` 并不证明运行时已经可以使用它。

`ArtifactStore.get_or_build(key, spec, builder)` 仅在无法获得有效缓存命中时调用
`builder(private_directory)`。构建器返回适配器持有的值，且必须在返回前完成所有文件写入。
编译异常（包括 `OSError`）和缺失或无效的构建输出会向上传播。
构建器不得递归获取同一个 key 的锁。

`ArtifactBuild` 报告 `HIT`、`PUBLISHED` 或 `PRIVATE`。新构建始终保留构建器返回值和
`private_directory`，成功发布后也一样，因为返回值可能仍引用私有文件。
适配器负责路径重绑定和后续清理。缓存命中返回 `ArtifactHandle`，没有构建器返回值或私有目录。
查找返回 `HIT`、`MISS`、`INVALID` 或 `STORAGE_ERROR`，并为无效条目或不可用存储提供诊断原因。

同一进程中，根目录、私有根目录、只读策略、key 和 spec 均相同的重叠调用共用一个正在执行的操作，
即使调用来自不同的 store 实例。这包括 `INVALID`、`STORAGE_ERROR`、只读未命中及加锁或发布失败。
私有结果的构建器返回值和目录会共享给等待者，因此匹配的构建器必须可互换，返回值必须适合共享。
编译错误（包括取消）会唤醒所有等待者并向上传播。
操作结束后，协调器删除对应结果或错误；后续独立的私有构建请求仍会重新构建。
这不是私有对象缓存。

## 布局与校验

```text
<root>/
  locks/<key>.lock
  artifacts/<environment-digest>/<key>/<spec-digest>/
    <state>/artifact_manifest.json
    .tmp.<random>/
```

阶段为 `generated` 或 `ready`，各自有独立的 spec 摘要。
每个已发布阶段的目录包含实际文件及 `artifact_manifest.json`。
完成标记包含 schema、完整 key 及其组件、阶段、构建类型、必需文件列表，
以及所有实际文件的有序清单：相对路径、字节数、SHA-256 摘要和所有者可执行标志。标记大小上限为 16 MiB。
读取方根据当前请求验证完整清单，时间戳不替代内容摘要。
额外文件、重复 JSON 字段、元数据变化、文件缺失和标记格式错误均使条目失效。
空目录不携带产物语义。

校验不会根据 manifest 中的路径打开文件，而是枚举实际目录树，再将其规范记录与标记比较。
绝对路径、非规范路径、父目录穿越和反斜杠路径均被拒绝。实际文件中的链接、特殊文件，
以及缓存根目录下通过符号链接连接的路径也被拒绝。
显式配置的根目录在初始化时解析为规范路径。只有所有者的执行位属于产物契约。
读写位及组和其他用户的执行位属于访问策略，其变化不会使内容未变的文件失效。
文件副本对所有者可读写，并保留所有者执行位；
不会把组和其他用户的权限位，以及 setuid、setgid 或 sticky 位传播到发布文件。

根目录的写入方必须可信：摘要能检测损坏，不能防止攻击者同时替换可执行代码及匹配的 manifest。
写入方不得修改已发布条目，也不得在读取期间删除它们。
此协议不是原子文件系统快照，也不防御恶意缓存所有者。

## 发布与恢复

1. 查找并校验所请求的阶段，不执行任何写入。
2. 可写模式下，所有非命中状态均在锁可用时获取持久 key 锁文件的 `flock`，然后重新查找。
   两个阶段及所有 spec 共用此锁；不同 key 可以并发构建。
3. 在缓存根目录外构建，验证所有必需的私有输出文件。
4. 将实际文件复制到最终目录旁的唯一暂存目录，使用独立文件而非硬链接。
   重新校验副本，同步实际文件和目录项，最后写入完成标记并同步。
5. 使用 Linux `renameat2(RENAME_NOREPLACE)` 发布，然后同步父目录。
   即使目标是一个已存在的空目录，也不会替换它。

暂存、加锁或发布失败时，返回可用的私有构建及原因。失败的暂存目录会尽力删除。
进程崩溃可能留下私有目录或 `.tmp.*` 目录，两者都不会命中缓存。
内核会释放已退出进程的锁。锁文件始终不被删除，因此等待进程继续在同一个 inode 上同步。

无效的最终目录不会被在线修复或覆盖；离线清理移除无效目录前，请求均在私有目录构建。
跨进程时，文件锁只有在首个进程能够发布可复用产物的情况下才会消除重复编译。
锁可用时会串行执行私有构建，但不会共享私有构建结果；
存储或锁持续不可用时，不同进程仍独立构建。
跨进程复用私有结果需要额外的共享回退产物发布协议，目前未实现。
如果发布成功但最后的父目录同步失败，私有结果仍被保留，有效的已发布目录也保持不变。
不支持禁止覆盖的原子重命名或没有可写缓存存储时，同样返回私有输出。
写入需要支持 `flock` 和禁止覆盖原子重命名语义的 Linux 文件系统；
网络文件系统必须先确认具备这些语义。

## 共享权限

默认只有发布方 UID 可以复用产物。已发布阶段目录继承暂存目录的 `0700`，
普通实际文件使用 `0600`，需要所有者执行权限的文件使用 `0700`。
中间目录和完成标记遵循进程 umask，但不能使外层 `0700` 阶段目录对其他 UID 可遍历。
目前没有 mode 配置项，也不能通过 umask 覆盖阶段目录和实际文件的固定权限。
不支持不同 UID 并发写入。

若要将预热产物提供给其他 UID 的只读消费者，须停止全部写入方和消费者，
将 `CACHE_ROOT` 设为目标缓存根目录，再显式授予读取权限。
以下策略向所有本地用户开放读取；若范围过大，应由管理员配置受限的用户组或 ACL 策略：

```bash
chmod a+rx,go-w -- "$CACHE_ROOT"
chmod -R a+rX,a-w -- "$CACHE_ROOT/artifacts"
```

根目录的所有祖先目录也必须允许目标读取者遍历。消费者必须使用 `readonly=True`；
这不会授予访问写入锁或发布产物的权限。上述命令保留所有者执行位。
通过 `a+rX` 或 `go-x` 增删组和其他用户的执行位也不会改变产物身份；
移除所有者执行位则会改变。

## 只读使用与阶段升级

`ArtifactStore(..., readonly=True)` 不在缓存根目录内写入任何内容，包括锁、索引和暂存目录。
命中时只读取 manifest 和实际文件。未命中或条目无效时，在缓存根目录外的
显式指定的 `private_root` 中构建。未指定时，缓存命中仍可使用，但需要构建的请求抛出 `OSError`。
存储层不探测临时目录候选位置，因为候选位置本身可能位于缓存根目录内。
若所选私有目录不可写，文件系统错误向上传播。
存储层从不导入或执行缓存中的 Python 文件；后续加载器仍须独立避免写入字节码。

升级生成产物时，二进制构建器通过
`generated_handle.materialize(private_directory)` 将已校验的实际文件复制到整个共享缓存根目录外的空目录，
即使目标通过符号链接访问，也会检查这一边界。
复制排除旧的完成标记，且不使用硬链接。即使缓存文件只读，副本也对所有者可写。
二进制编译可以自由修改私有文件；
发布在对应 spec 摘要下生成独立的 `ready/` 目录，保持 `generated/` 不变。
运行时适配器负责路径重绑定，以及二进制和元数据的完整覆盖。

此层没有在线垃圾回收、诊断索引、全局缓存统计或自动阶段优先级。
清理必须在全部消费者停止后离线执行。后续集成必须优先选择 ready，再选择 generated，
并确保对象整个生命周期内所引用的路径始终有效。

## 显式运行时适配器

内部模块 `pypto.runtime._artifact_runtime` 连接已校验的句柄（handle）和编译程序。
它实现 RFC #2653 的运行时阶段，不是公共缓存配置 API。调用方必须提供完整的
`ArtifactKey`、匹配的输入快照，以及列出全部必需生成文件的 `ArtifactSpec`。
生产环境不能使用占位身份摘要。

发布 `GENERATED` 前，构建函数调用 `pypto.runtime._artifact_sources` 中的
`package_generated_sources(private_directory, build_kind)`，规范化生成配置中的路径，
并将支持的 extern 依赖打包进私有目录。存储层随后校验并发布这个自包含目录。

```python
from pypto.runtime._artifact_runtime import bind_artifact, restore_artifact

# generated_handle is a validated ArtifactHandle for this compiled input snapshot.
# run_directory is a Path outside store.root, owned by this runtime session.
bind_artifact(compiled, store, generated_handle, run_directory)
compiled.load()  # Single-chip: promotes if needed, then assembles live callables.
ready_handle = compiled._artifact_runtime.handle

# A different process may look up the ready key/spec in a read-only ArtifactStore.
restored = restore_artifact(readonly_store, ready_handle, another_run_directory)
restored.load()  # Validates metadata and bytes; does not compile or execute.
```

分布式程序采用相同的绑定和恢复方式，由现有 runner 或 worker 延迟请求全部子 callable。
上述 `compiled.load()` 是单芯片接口。目前支持单个单芯片构建，或包含芯片子构建的分布式
父产物。不支持单芯片多 orchestration 父产物，因为现有 `from_dir` 协议无法在没有实时 IR
的情况下重建该父对象；应分别持久化受支持的子构建。

### 阶段晋级与加载

1. 执行任何生成配置前，按精确 key 和 spec 校验句柄。计算 ready spec，列出父标记、
   每个子标记、orchestration 二进制和每个 kernel 二进制。Generated spec 必须声明
   所有必需的芯片配置；声明的文件缺失时存储校验失败。不含 `kernel_config.py` 的
   辅助目录会被跳过，与普通分布式重放保持一致。
2. 通过 `ArtifactStore.get_or_build` 查找 `BINARY_READY`。未命中时，将 generated
   句柄物化到私有目录。加锁顺序是产物 key 锁，再到私有运行时构建锁。
3. 使用现有运行时编译器完成全部芯片构建。即使旧上下文标记匹配，也跳过继承的可变
   二进制缓存和源码旁的二进制文件：generated 身份不能证明这些字节有效。
   记录传给 `CoreCallable.build` 的最终 kernel
   字节和传给 `ChipCallable.build` 的 orchestration 字节。编译和组装不会在设备上执行；
   每个私有编译锁释放后，删除其 `cache/` 目录（包括上下文标记和锁文件）及生成源码旁
   的 `.o`/`.so` 输出。保留源码、配置、extern 输入、二进制 manifest，以及 `prebuilt/`
   下每份最终二进制的唯一副本。继承的缓存和旁置二进制文件不再纳入 ready spec。
   只有全部子构建成功后才能发布 ready 产物。
4. 读取带版本的 `binary_manifest.json`，在构造任何 callable 前校验所有子项。
   记录包含相对二进制路径、大小、SHA-256 摘要、平台、运行时配置、函数 ID、签名和
   诊断名称。分布式父标记列出完整的芯片子构建集合。
5. 从字节重建 callable。Ready 加载不解析 PTO-ISA、不构造编译器、不获取可写缓存锁、
   不改写头文件、不执行 `kernel_config.py`，也不写二进制上下文标记或 Python 字节码。
   仍需要与所提供 key 对应的兼容运行时库。

完整身份和负载清单仍以外层存储 manifest 为准。仅有二进制标记不足以绑定句柄。
绑定或恢复时仅对整个负载计算一次哈希、恢复一次元数据。加载复用已验证的清单来核对
内层二进制大小和摘要，不再重复计算相同字节的哈希。直接构造的内部 `ArtifactRuntime`
在首次加载时校验。晋级过程在生成文件复制、新 ready 负载各自的边界上校验；私有回退
加载直接校验二进制摘要。存储查找校验与绑定校验相互独立。
校验结果仅供绑定对象的生命周期内复用，不是进程级缓存；所有使用者释放前，发布的文件
必须始终存在且不变。新的绑定会重新校验。缺失或损坏的句柄在上述边界、执行之前失败。
适配器不会捕获设备执行错误并重试操作。

存储或发布失败时，适配器在其生命周期内保留可用的私有目录和实时 callable。
此时 `handle` 仍是 generated 句柄，`directory` 指向私有二进制输出。重叠的晋级请求
共享存储层的进程内操作，包括私有回退；跨进程私有结果仍按前文说明各自独立。
私有目录不会自动删除，所有者必须在所有使用者释放后清理。

### 路径、诊断与 extern 输入

绑定保留新编译对象的实时 IR；从持久化元数据恢复时 `program is None`。绑定在注册到
worker 前仅重设一次源码路径，之后晋级只更新运行时二进制句柄。编译对象的路径和哈希
保持稳定，适用于 worker 注册表。未绑定的普通 `from_dir` 重放保留原有可变编译行为。

DFX 输出、依赖捕获和泳道转换写入独立的 `run_directory`。并发运行时会话必须各自选择
独立目录。名称来自二进制记录，生成的 host orchestration 加载不使用 Python 字节码缓存。
编译对象或 worker 持有引用期间，已发布产物必须保持存在且不变。

Extern 打包支持递归解析的字面量本地 include，保留相对 include 拓扑和显式 include
目录顺序。源码目录与显式 include 目录的公共根仅解析一次，支持 workspace 或 home
上游的符号链接；公共根下的符号链接（包括 `..` 遍历之前经过的链接）要求私有编译。
扫描器移除注释、连接续行，并根据字面量 `#if 0`/`#if 1` 及其 `#elif`/`#else` 结构
忽略确定不会生效的分支。条件未知时保守扫描所有可能分支，不实现完整 C 预处理器。
可能生效的宏 include、绝对路径 include 和未解析的引号 include 要求私有编译。
非 UTF-8 源码按原始字节复制，仅在 include 扫描时使用替换解码。未解析的尖括号
include 由单独标识的 SDK/工具链提供。空 include 目录可以在发布时消失，对应缺失的
`-I` 路径仍然有效；`extra_include_dirs=None` 会规范化为空列表。
其他引用文件的预处理器或汇编构造不在支持范围内。
调用方必须在发布前建立完整输入身份；打包器不会发现工具链清单，也不会让任意 C++ 构建
自动成为封闭构建。`pypto.runtime._artifact_sources` 提供专用的 `ValueError` 子类
`UnsupportedArtifactInput`，用于表示上述打包限制。适配器可以仅捕获此异常，跳过持久化
发布并对原始输入进行普通私有编译。打包会修改私有暂存树，回退时应丢弃该树。
文件系统失败、损坏的输入或配置仍作为不同的错误传播，不能把所有 `ValueError` 都当作
缓存旁路信号。当前显式适配器不会在拒绝打包后自动调用普通编译器。
受支持产物一旦 ready，即可迁移并
在原始 extern 源码目录不存在时加载。
