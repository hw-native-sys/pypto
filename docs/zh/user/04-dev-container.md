# 开发容器

拉取一个已经装好 CANN、PyPTO 和运行时的镜像，不用自己拼工具链，直接在真实昇腾卡上跑内核。

本指南开头的命令和镜像信息描述旧版 ARM A2/A3 `a3-dist` 镜像（CANN 9.0.0）。
CANN 9.2 的 ARM A2/A3、ARM A5 和 x86 A5 镜像，请使用
[在 CI 中维护发布镜像](#在-ci-中维护发布镜像)中的标签、驱动和验证用例。
这些标签在首次成功发布后才可拉取。

## 概念

[安装](01-installation.md)从源码构建 PyPTO，得到的是**编译器前端** —— 足够写内核、读它降级出的
IR。真要跑起来还需要更多：配套的 CANN、针对它构建的 `simpler` 运行时、ptoas，以及一张卡。开发
容器就是这整套栈的预构建版本。

动手之前最该搞清楚的是镜像与宿主机的分工：

| 层 | 由谁提供 | 说明 |
| -- | -------- | ---- |
| NPU 硬件 | 宿主机 | 昇腾 A2 / A3，至少一张空闲卡 |
| 内核驱动 | **宿主机** | 不在镜像里，而且不是可以随意选的一项 —— 见下 |
| Ascend Docker Runtime | 宿主机 | 把驱动库和设备节点注入容器 |
| CANN 用户态 | 镜像 | 9.0.0 |
| CPython 3.10、torch、ptoas | 镜像 | `/opt/pypto/venv`、`/opt/pypto/ptoas` |
| 已构建的 PyPTO + `simpler` | 镜像 | `/workspace/pypto`，已编译完成，可通过 `$PYPTO_SRC` 访问 |
| `pypto-lib` 模型与示例 | 镜像 | `/workspace/pypto-lib`，可通过 `$PYPTO_LIB_SRC` 访问 |

**宿主机驱动版本不能随意选。** 镜像带的是 CANN 用户态，不带驱动。AICPU 的 device 侧包由容器内的
CANN 在运行时经 TSD 下发给 device，驱动会在接收前做一次版本校验。这道校验只出现在 AICPU 路径
上 —— 所以驱动不配套时，编译、显存分配、`aclrtSetDevice` 全部正常，直到第一次 AICPU 初始化才
以 `507018` 失败。受支持的组合是 `driver 26.0.rc1 + CANN 9.0.0`，与运行时在
`runtime/docs/install.md` 中声明的受支持环境一致；这也是本镜像唯一验证过的组合。命中之后怎么办
见[排查 `507018`](#排查-507018)。

### 镜像信息

```text
swr.cn-east-3.myhuaweicloud.com/cloud-pypto/pypto-dev:a3-dist
```

| 属性 | 取值 |
| ---- | ---- |
| 架构 | ARM64（`linux/arm64`）—— 不能在 x86_64 宿主机上运行 |
| 目标设备 | 昇腾 A2 / A3（PyPTO 平台名 `a2a3`） |
| 用户态 | Ubuntu 22.04，CANN 9.0.0 |
| ptoas | 0.61 —— 即镜像自带 pypto 提交所 pin 的版本；`main` 此后已升到更高版本 |
| 未压缩大小 | 约 16.7 GB |
| 工作目录 | `/workspace` |
| 默认入口 | 已激活 PyPTO 环境的交互式 bash |
| 已验证摘要 | `sha256:ffa6bef8ed34f565a2331a66369902303ff957e832476fa1abaad2be435b82b5` |

**镜像里的 ptoas 落后于 `main`。** 它与构建镜像所用 pypto 提交（即下面 `dist-manifest` 的
`pypto` 一行）中的 `toolchain/versions.env` 一致，而不是与仓库当前的 pin 一致。codegen 会拒绝
低于源码 pin 的 ptoas，因此一旦容器里的源码 pin 的 ptoas 比镜像自带的新 —— 拉取了更新，或把当前
checkout 挂载到 `/workspace` —— 每次 codegen 都会失败，直到重装 ptoas；见[更新源码并重建](#更新源码并重建)。
对比两者：

```bash
grep '^PTOAS_VERSION=' "$PYPTO_SRC/toolchain/versions.env"
/opt/pypto/ptoas/bin/ptoas --version
```

`/opt/pypto/dist-manifest` 记录了这个镜像到底是用什么构建的 —— pypto、runtime、pto-isa、
pypto-lib 四个提交，以及构建时间。反馈问题时请附上它：

```bash
cat /opt/pypto/dist-manifest
```

```text
pypto     6fdd989fdf45b10011c4bd6b0291a7e6c18fd7e4
runtime   39ce891dbb3f665e72e99b4a1387b47012fb18bd
pto-isa   5a4f74cbf627d4aac2e0ce10d5e0d8b118343265
pypto-lib c3f0dea274f55d9648920f17c968e8564ae9fcdc
built     2026-09-14T09:19:43Z
```

容器内 `PATH` 上有三个来自 `/opt/pypto/bin` 的命令：

| 命令 | 作用 |
| ---- | ---- |
| `pypto-doctor` | 检查工具链、驱动、设备访问、源码树和构建产物，并对每个失败项给出具体做法 |
| `pypto-build` | 用原生 CMake 重建 `$PYPTO_SRC` 下的源码，然后安装 `./runtime`（`simpler`） |
| `pypto-activate.sh` | 环境本身 —— 既作为 shell 的 rcfile 也作为 `BASH_ENV`，因此 `docker exec … bash -c '…'` 看到的环境与登录 shell 一致 |

仓库现在维护更新器、安装器及这些辅助脚本。重新构建的镜像把它们安装到 `PATH` 上；
上面已验证的旧镜像仍可能只在源码树中提供更新器。
见[更新源码并重建](#更新源码并重建)。

镜像还内置了与所 pin 运行时匹配的 `pto-isa` checkout，首次运行不会联网克隆。

## 快速上手

以下假设宿主机已经备齐 NPU 驱动、Docker Engine 和 Ascend Docker Runtime。三者的核对清单
见[宿主机前置条件](#宿主机前置条件)，缺哪一样也在那里装。

在宿主机上找一张空闲卡：

```bash
npu-smi info
```

假设物理卡 1 空闲：

```bash
export PYPTO_IMAGE=swr.cn-east-3.myhuaweicloud.com/cloud-pypto/pypto-dev:a3-dist
docker pull "$PYPTO_IMAGE"

docker run --rm -it \
  --runtime=ascend \
  -e ASCEND_VISIBLE_DEVICES=1 \
  --shm-size=16g \
  "$PYPTO_IMAGE"
```

进入容器后，先做环境自检再跑别的：

```bash
pypto-doctor
```

关键的几项是：

```text
ok    device access: aclrtSetDevice(0) ok, 62748MB HBM visible
ok    import pypto -> /workspace/pypto/python/pypto/__init__.py
ok    simpler binding matches its source tree
environment looks good.
```

HBM 数值随卡而变。另有两行是 `warn`，在 `--runtime=ascend` 下属于预期，都不代表环境有问题：

```text
warn  driver libs present but version.info is not exposed (normal under --runtime=ascend)
warn  npu-smi: npu get board type failed. ret is -9005 — expected under --runtime=ascend
```

驱动的 `version.info` 只在宿主机上读得到，容器里读不到；`npu-smi` 失败是因为它要枚举整机所有板卡，
而容器里只有你暴露的那几张。`device access: aclrtSetDevice(0) ok` 确认设备访问正常，
但不能证明驱动兼容 AICPU 初始化。缺少 CANN 解释器或 ACL 探测无法执行时，doctor 会报告阻塞性失败。
如果可以读取 `version.info`，doctor 会校验运行时支持的组合：CANN `9.0.0` 配驱动 `26.0.rc1`，
或 CANN `9.1.T500` 配驱动 `25.6.rc1.b108`。CANN 9.2 还接受下文 CI 镜像表中按架构区分的
部署配置，但仍须通过上板验证。未知组合无法通过校验。
可以用 `ASCEND_DRIVER_ROOT` 覆盖默认驱动目录 `/usr/local/Ascend/driver`。

然后在卡上跑一个内核：

```bash
python $PYPTO_SRC/examples/beginner/01_hello_world.py
```

末行是：

```text
OK
```

这一行覆盖了整条链路：解析器构建出 IR、pass 流水线跑通、ptoas 汇编出内核、运行时完成下发、结果
与 torch 对齐。

`--rm` 会在退出时删除容器的可写层。镜像会保留，你在容器里改的东西不会。要留住改动见
[保留你的源码](#保留你的源码)。

## 机制

### 宿主机前置条件

| 要求 | 期望值 | 核验命令 |
| ---- | ------ | -------- |
| 架构 | `aarch64` | `uname -m` |
| NPU 驱动 | `Version=26.0.rc1` | `cat /usr/local/Ascend/driver/version.info` |
| 设备节点 | `davinci0`、`davinci_manager`、`devmm_svm`、`hisi_hdc` | `ls /dev/davinci*` |
| 设备健康状态 | `OK` | `npu-smi info` |
| Docker Engine | 已验证 27.2.0；其他版本需同时满足驱动与 Ascend Docker Runtime 的要求 | `docker version` |
| Ascend Docker Runtime | 输出中含 `ascend` | `docker info --format '{{json .Runtimes}}'` |
| 空闲磁盘 | ≥ 30 GB，用于镜像层、容器和构建缓存 | `df -h /var/lib/docker` |

`version.info` 里 `Version=` 是公开驱动版本，`compatible_version=` 列出
`V100R001C10` 等内部版本标识，并非公开的 CANN 版本号。
受支持的公开驱动/CANN 组合以 `runtime/docs/install.md` 为准。

缺什么按下面的顺序装 —— 每一步都依赖前一步：

| 缺失项 | 安装方式 |
| ------ | -------- |
| NPU 驱动 | 按服务器型号和操作系统安装配套的昇腾驱动，装完重启。它是镜像唯一无法提供的部件，且版本被镜像内的 CANN 9.0.0 钉死在 `26.0.rc1` |
| Docker Engine | 按 [Docker Engine 官方安装文档](https://docs.docker.com/engine/install/)安装适合当前发行版的版本 |
| Ascend Docker Runtime | 见下面的[安装 Ascend Docker Runtime](#安装-ascend-docker-runtime) |

### 安装 Ascend Docker Runtime

Ascend Docker Runtime 是宿主机上的 OCI Runtime 插件，作用类似 NVIDIA Container Runtime：把选定的
NPU、驱动库和设备节点注入容器。

从昇腾渠道下载与宿主机架构、驱动匹配的 `.run`，以 root 安装：

```bash
RUNTIME_RUN=Ascend-docker-runtime_<在此填入版本号>_linux-aarch64.run

chmod +x "$RUNTIME_RUN"
sudo "./$RUNTIME_RUN" --check
sudo "./$RUNTIME_RUN" --install
sudo systemctl daemon-reload
sudo systemctl restart docker
```

先把版本号填进第一行再跑后面几条 —— 不加引号的 `<` 和 `>` 对 shell 来说是重定向，直接粘贴占位符
会把文件名截断，而且不会报出你以为的那个错。

重启 Docker 会中断该宿主机上所有正在运行的容器 —— 请在维护窗口执行。之后确认注册成功：

```bash
docker info --format '{{json .Runtimes}}'
```

输出中必须出现 `ascend`，否则 `--runtime=ascend` 用不了。参见厂商的
[手动安装指南](https://www.hiascend.com/document/detail/zh/mindcluster/72rc1/clustersched/dlug/dlug_installation_017.html)
和 [Docker 客户端使用说明](https://www.hiascend.com/document/detail/zh/mindx-dl/500/dockerruntime/dockerruntimeug/dlruntime_ug_013.html)。

### 拉取镜像

```bash
export PYPTO_IMAGE=swr.cn-east-3.myhuaweicloud.com/cloud-pypto/pypto-dev:a3-dist
docker pull "$PYPTO_IMAGE"
```

该仓库允许匿名拉取，不需要 `docker login`。这是本页最慢的一步：镜像在磁盘上展开约 16 GB，
前置条件里要求的 30 GB 空闲空间就是为它准备的。每台宿主机只付一次 —— 之后每一次
`docker run`、每一个新建的容器，用的都是本地这一份。

标签 `a3-dist` 中，`a3` 是芯片系列，`dist` 是完整构建的变体，也就是本页所讲的这一个。

确认拉到的东西：

```bash
docker image inspect "$PYPTO_IMAGE" --format 'arch={{.Architecture}} os={{.Os}} size={{.Size}}'
docker image inspect "$PYPTO_IMAGE" --format '{{index .RepoDigests 0}}'
```

要环境可复现，就按摘要而不是按标签来指定镜像 —— 标签是可以被移动的。做法是重新绑定
`PYPTO_IMAGE`，而不是单独拉一次摘要，这样本页后面每一条 `docker run` 用的也都是它：

```bash
export PYPTO_IMAGE=swr.cn-east-3.myhuaweicloud.com/cloud-pypto/pypto-dev@sha256:ffa6bef8ed34f565a2331a66369902303ff957e832476fa1abaad2be435b82b5
docker pull "$PYPTO_IMAGE"
```

### 选卡

`ASCEND_VISIBLE_DEVICES` 指的是**宿主机物理卡**。Ascend Docker Runtime 会在容器内重新编号，且总是
从 0 开始：

| 宿主机 | `ASCEND_VISIBLE_DEVICES` | 容器内 | PyPTO 参数 |
| ------ | ------------------------ | ------ | ---------- |
| 卡 1 | `1` | 卡 0 | `-d 0` |
| 卡 2、3 | `2,3` | 卡 0、1 | `-d 0,1` |
| 卡 4-7 | `4,5,6,7` | 卡 0-3 | `-d 0,1,2,3` |

所以不管暴露的是哪几张物理卡，PyPTO 命令里的编号永远从 0 数起。

绝不要把同一张物理 NPU 交给两个容器。启动前用 `npu-smi info` 和 `docker ps` 确认。

### 多卡

暴露两张空闲卡，跑一个最小的集合通信：

```bash
docker run --rm -it \
  --runtime=ascend \
  -e ASCEND_VISIBLE_DEVICES=2,3 \
  --shm-size=16g \
  "$PYPTO_IMAGE"
```

```bash
python $PYPTO_SRC/examples/distributed/08_allreduce_mesh.py -p a2a3 -d 0,1
```

末行是 `OK`。这条用例覆盖了 HCCL 和跨 rank 路径，单卡跑不到。

某些 HCCL / 驱动 / Ascend Docker Runtime 组合在容器内跑多卡程序时会出问题。`driver 26.0.rc1` 配本
镜像是验证过的；若换成别的组合出现静默退出，就按项目 CI 的做法改到宿主机上跑，并核对驱动 / CANN /
HCCL 的兼容性矩阵。

### 保留你的源码

镜像里已经有构建好的源码，所以快速上手那一段完全不需要挂载 —— 但你改的一切都会随容器一起消失。
两种留住的办法：

**命名卷。** 第一次挂载空卷时，Docker 会把镜像里的内容复制进去；此后即使容器被删，源码和缓存仍在。

```bash
docker volume create pypto-a3-src
docker volume create pypto-a3-lib-src
docker volume create pypto-a3-cache

docker run -it \
  --name pypto-a3-dev \
  --runtime=ascend \
  -e ASCEND_VISIBLE_DEVICES=1 \
  --shm-size=16g \
  -v pypto-a3-src:/workspace/pypto \
  -v pypto-a3-lib-src:/workspace/pypto-lib \
  -v pypto-a3-cache:/opt/pypto/cache \
  "$PYPTO_IMAGE"
```

用 `docker start -ai pypto-a3-dev` 重新进入同一个容器。即使删掉容器，用同样这三个卷创建的新容器也能
接着上次的进度。

拉取新镜像不会刷新已有卷里的内容：**环境**跟着镜像走，**源码**跟着卷内的 `git pull` 走。

**绑定挂载。** 要在宿主机上的 checkout 里开发，宿主机目录必须已经是这个结构：

```text
/path/to/workspace/
├── pypto/
└── pypto-lib/
```

```bash
export PYPTO_HOST_WORKSPACE=/path/to/workspace

docker run --rm -it \
  --runtime=ascend \
  -e ASCEND_VISIBLE_DEVICES=1 \
  --shm-size=16g \
  -v "$PYPTO_HOST_WORKSPACE:/workspace" \
  "$PYPTO_IMAGE"
```

挂载的是*包含* `pypto/` 和 `pypto-lib/` 的那一层，而不是它的上一层 —— 容器内必须直接存在
`/workspace/pypto` 和 `/workspace/pypto-lib`。绑定挂载会隐藏镜像原有的 `/workspace`，所以你挂进去
的源码就是唯一的源码，而它几乎肯定不是 `simpler` 当初构建时用的那份。先跑 `pypto-build`，再做别的。
启动提示始终引导你检查当前 checkout；镜像的 `dist-manifest` 仅记录交付时的版本，
不能证明绑定挂载的源码已经构建完成。

模型示例通过 `PYPTO_PLATFORM` 选择平台：启动器在 `run a3` 时设置 `a2a3`，
在 `run a5` 时设置 `a5`；直接运行镜像时则根据镜像的 CANN 标签推导。
自定义环境应显式设置 `PYPTO_PLATFORM=a2a3` 或 `PYPTO_PLATFORM=a5`。
无法识别标签时，启动提示会要求设置平台，不会猜测模型的编译目标。

### 更新源码并重建

维护中的辅助脚本位于 [`docker/scripts/`](../../../docker/scripts/)。用该 Dockerfile
重新构建的镜像会把脚本安装到 `/opt/pypto/bin`，绑定挂载覆盖 `/workspace` 后仍然可用。
已发布的旧镜像可能仍只在自带 checkout 中提供旧更新器；这时使用下面的仓库版本。

如果已有容器的 checkout 仍将辅助脚本直接放在 `.github/docker/` 或 `docker/`，先执行
`git -C "$PYPTO_SRC" pull --ff-only`，再运行下面的命令。旧版已安装更新器拉取目录迁移后
若提示找不到辅助脚本，也可这样恢复。仓库中的更新器会刷新 `/opt/pypto/bin`，使其适配新目录。

对于已经设置跟踪分支且没有本地改动的 checkout：

```bash
bash "$PYPTO_SRC/docker/scripts/pypto-update.sh"
```

它在各个已有仓库的当前跟踪分支上做快进更新，更新所 pin 的子模块，按
`toolchain/versions.env` 安装所需的 PTOAS，然后重建 PyPTO 和 simpler，最后运行
`pypto-doctor`。它不会切换分支或 reset 提交。分离 HEAD、分支分叉或本地改动需要先自行处理，
也可以使用 `--no-pull`。
拉取完成后（使用 `--no-pull` 时则直接），更新器会使用 `$PYPTO_SRC/docker/scripts` 中的辅助脚本，
从 `/opt/pypto/bin` 启动时也一样。所选 checkout 必须包含这些脚本；缺少时直接报错，不回退到镜像中的旧副本。
每次更新都会原子替换已安装的辅助脚本和工具链配置，不带 `--reinstall-deps` 时也一样。
当前 shell 需执行 `source "$PYPTO_HOME/bin/pypto-activate.sh"` 或重新进入，以加载新的激活逻辑。
`pypto-setup.sh --helpers-only` 只刷新这些文件，不安装依赖。

修改绑定挂载的 checkout 后，保留源码并重建：

```bash
bash "$PYPTO_SRC/docker/scripts/pypto-update.sh" --no-pull --jobs 2
```

| 参数 | 作用 |
| ---- | ---- |
| `--no-pull` | 不改变源码和子模块；对齐 PTOAS 并重建 |
| `--reinstall-deps` | 使用共享 setup 脚本刷新 Python 依赖 |
| `--proxy http://host:port` | 为本次运行设置 HTTP 和 HTTPS 代理 |
| `--remote-submodules` | 支持反复跟踪子模块远程；允许提交偏离 pin，但拒绝本地文件改动 |
| `--jobs N` | 正整数构建并行度，受机器限制约束；否则使用仓库资源限制加载器 |
| `--skip-doctor` | 没有暴露设备时跳过收尾检查 |

只修复过旧的 PTOAS，不更新源码或重建：

```bash
bash "$PYPTO_SRC/docker/scripts/pypto-install-ptoas.sh"
source "$PYPTO_SRC/docker/scripts/pypto-activate.sh"
"$PTOAS_ROOT/bin/ptoas" --version
```

安装器读取 checkout 中的版本及架构对应的 SHA256，在改变安装前校验 release wheel，
安装到 `${PTOAS_ROOT:-/opt/pypto/ptoas}`。`--force` 可重装同一版本。
两个专用环境都必须包含可运行的 CPython 3.10、`pyvenv.cfg` 和 `bin/activate`。
PTOAS 环境不兼容或不完整时，安装器会先验证下载 wheel 的校验和，再重建该环境。
激活后 PyPTO venv 在 PATH 中优先，固定版本的 PTOAS 则优先于 CANN 自带的汇编器；
runtime 包始终显式安装到 `$PYPTO_HOME/venv`。
使用旧镜像时，每个新 shell 使用 PTOAS 前都要 source checkout 中的激活脚本；只选择可执行文件
仍可能加载 CANN 的 Python 包，覆盖固定版本的 wheel。安装器会自动隔离自身版本探测的导入路径。
环境需要已有 uv 和 CPython 3.10，已发布镜像满足这个条件。在其他 CANN 容器中配置环境时，
从完整 checkout 运行 `pypto-setup.sh`；它安装同一份辅助脚本，并通过
`build-constraints.txt` 约束构建工具。
Setup 的专用 `$PYPTO_HOME/venv` 必须使用 CPython 3.10，并包含可读取的
`Python.h`、`pyconfig.h` 和 Python 库以支持原生构建。不兼容或不完整的环境会在安装依赖前
使用 uv 管理的 Python 重建；已有完整构建环境会保留。PTOAS 的独立运行时环境不要求开发文件。
Setup 完成后（包括 `--helpers-only`），将输出的激活命令复制到当前 shell 执行。
该命令会先导出选定的安装目录和 checkout 路径，再 source 激活脚本，
支持通过 `--pypto-home` 和 `--pypto-src` 指定的路径。更新器输出的重新加载命令也会保留这些路径。
Setup 将发现的 uv 可执行文件复制到 `$PYPTO_HOME/bin`，使 checkout 所有者执行的构建
也能使用它，即使原始文件位于 root 的私有主目录下。
如需固定 uv 版本，传入 `--uv-version TAG` 或设置 `UV_VERSION=TAG`；指定具体发布标签时，
即使已经安装 uv，也会下载指定版本。命令行选项优先于环境变量。默认的 `latest` 设置
会复用已有 uv，仅在未安装 uv 时下载最新版本。

`pypto-build` 使用原生 CMake，安装与 checkout 匹配的 simpler，并遵循其中的 PTO-ISA pin。
每次运行都会重新构建并安装 runtime wheel，包括未提交的源码改动；
原生构建和 ccache 仍可复用编译结果。
它刻意避免 `pip install -e .`：scikit-build 配置过的 `build/` 可能把扩展放到源码导入路径之外。
并行度通过 `.claude/skills/testing/load-env.sh` 加载，未分类的机器默认使用两个任务。
显式指定的任务数量可以降低这个上限，不能提高它。
已有原生构建保留其 CMake generator，新构建使用 Ninja。
如果路径、环境或工具链不同，会同时丢弃缓存和目标文件，包括挂载到相同路径的宿主机构建缓存。
辅助脚本在每次成功构建后记录构建状态；状态不变时继续增量构建，
未记录的构建或在辅助脚本之外修改过的产物会重新构建。记录不包含容器主机名、路径列表顺序、
所有权和文件系统 change time，因此重建等价容器时仍可复用增量构建。
`pypto-doctor` 检查已安装 simpler 的构建版本是否与挂载的 `runtime/` 匹配。
如果 `runtime/` 有未提交改动，会给出警告，因为版本一致无法证明已安装 binding 包含这些改动。
即使设置了 `PYPTO_SKIP_RUNTIME_PIN_CHECK=1`，已知版本不匹配或使用引入版本标记之前的旧 binding
仍会导致检查失败。构建版本标记为空或无法读取 checkout 版本时，会明确报告无法验证，而不会声称版本匹配。

### 构建开发镜像

在目标架构机器上，从仓库根目录构建。`deps` 提供环境，`sim` 增加 GCC/G++ 15，
`dist` 还会克隆并构建 PyPTO 和 pypto-lib：

```bash
docker buildx build --load --target dist \
  --file docker/pypto_dev.Dockerfile \
  --build-arg DIST_BUILD_JOBS=2 \
  --build-arg PYPTO_REF=<pypto-commit> \
  --build-arg PYPTO_LIB_REF=<pypto-lib-commit> \
  --tag pypto-dev:a3-dist .
```

默认的 `main` 引用保留跟踪分支，因此 `pypto-update.sh` 可以拉取更新。
标签和提交引用使用分离 HEAD；这类固定版本构建应使用 `--no-pull`。
选定的 PyPTO ref 必须包含开发容器辅助脚本、`toolchain/versions.env` 和
`build-constraints.txt`。`dist` 阶段执行该 checkout 的 setup 和 build 脚本，
根据其中的 pin 和约束刷新依赖，并安装该版本的辅助脚本供后续会话使用。
如果选定 ref 缺少这些文件，构建会失败，不会回退到 Docker 构建上下文中的辅助脚本。

构建 AMD64 镜像时，使用 x86_64 宿主机及支持 `linux/amd64` 的 CANN 基础镜像标签；
构建前先确认该标签支持目标架构。先发布独立架构标签，在对应 NPU 宿主机上通过
`pypto-doctor` 和模型验证后，再组合多架构 manifest。无 NPU 时构建成功只验证镜像构建过程。

本地环境镜像可以用 `docker/scripts/pypto-dev.sh build a3` 构建，
用 `docker/scripts/pypto-dev.sh run a3 0` 创建或重新进入容器。
启动器支持 CANN 标签、镜像、缓存卷和代理覆盖，具体见源码。
代理值通过环境传递，不在命令跟踪中打印凭据，也不作为命令参数传入。重新进入容器时保留容器原有环境。
复用容器前，启动器会检查 checkout 路径、镜像（包括同一标签重新构建的情况）、缓存卷和设备配置。
配置不匹配或旧容器缺少配置元数据时，启动器会停止并提示重建方法；
显式删除容器前，请先保存仅存在于容器内的改动。
使用 Ascend Docker Runtime 时，选中的物理卡在容器内对应逻辑设备 `0`，因此启动器导出
`PYPTO_DEVICE=0`。privileged 模式保留物理卡编号；`pypto-doctor` 探测同一个选定设备。
挂载关联工作树（linked worktree）时，启动器保留 checkout 和共享 Git 元数据的原始路径，
相应设置 `PYPTO_SRC`/`PYPTO_LIB_SRC`，并从 `PYPTO_SRC` 启动容器；
这样无需改写宿主机元数据，就能保持子模块的相对 gitdir 路径和工作树反向引用有效。
普通 checkout 仍使用 `/workspace/pypto` 和 `/workspace/pypto-lib`。
启动器为挂载的 checkout 及其已初始化的子模块（包括嵌套子模块）传递限定路径的 Git
`safe.directory` 配置，使容器 root 可以访问宿主机用户拥有的工作树。
新增已初始化的子模块会改变容器配置，需要重建容器。
Git 更新使用各 checkout 所有者的数字 UID/GID；原生编译和 runtime wheel 构建使用 PyPTO
checkout 的所有者身份，使新源码、Git 元数据和构建产物在宿主机上仍可写。
rootful 容器需要 util-linux 提供的 `setpriv`，无需在 `/etc/passwd` 中创建同名账户。
写入源码的命令使用独立的 `$PYPTO_HOME/users/<uid>` HOME 和
`$PYPTO_HOME/cache/ccache/<uid>` 缓存目录。私有仓库认证需在该 HOME 配置，或使用 SSH
agent；不会复制 root 的凭据。完成的 wheel 由调用者以原权限安装，保持已有 Python 环境的
所有权不变。脚本不会递归修改已有宿主机文件或共享 worktree 元数据的所有权。
交互 shell 和设备诊断保持原有权限，checkout 诊断使用源码所有者身份；激活脚本禁用 Python
字节码写入，以及只读 Git 检查时可选的索引刷新。
如果 PyPTO 主 checkout 包含 `.claude/skills/testing/testing.env`，启动器也会将该文件
只读挂载到原始路径，以保留机器的构建并行度限制。

### 在 CI 中维护发布镜像

[`docker/images.json`](../../../docker/images.json) 是镜像维护入口。
`Maintain Development Images` workflow 每天在 UTC 19:37（北京时间 03:37）运行；
`main` 上的 `docker/`、工具链版本、构建约束或该 workflow 变化时也会触发，
还可通过 **Run workflow** 手动运行。修改上述路径的 PR 在 GitHub 托管 runner 上运行配置检查和
离线脚本测试。仅当 PR 来源仓库为 `hw-native-sys/pypto` 或明确认可的可信 fork `luohuan19/pypto` 时，
才在指定 NPU runner 上构建镜像，执行 doctor、单卡 decode 和双卡分布式用例验证；其他 fork 跳过该 job。
允许名单位于 workflow 的 `image.if` 条件中；新增仓库意味着允许其代码在持久化宿主机上执行，
必须由维护者审核其可信性。
PR 可以修改 workflow YAML，因此该名单本身不是安全边界。启用共享 runner 的 PR 构建前，
管理员必须在 **Settings → Actions → General** 中设置 **Require approval for all external contributors**，
并确保维护者只批准可信来源和经过审查的代码，包括 workflow 修改。这是仓库级策略，会影响所有外部 PR workflow；
等待批准时，配置检查也不会运行。该设置不会隔离已经批准执行的代码。
参见 [GitHub fork workflow 审批设置](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/enabling-features-for-your-repository/managing-github-actions-settings-for-a-repository#controlling-changes-from-forks-to-workflows-in-public-repositories)。
定时任务在 workflow 合并到默认分支 `main` 后生效。匹配路径的改动合并到 `main` 时，
也会立即触发构建发布；请在合并前配置 SWR Secrets。
PR 不发布镜像，也不获取 SWR 凭据。
硬件构建只在 `hw-native-sys/pypto` 仓库中执行。手动运行默认仅验证；发布还要求分支为 `main`。

当前启用三个原生镜像，均使用 CANN 9.2.0：

| 镜像 | Runner 标签 | 宿主机驱动（`Version=`） | 发布标签 |
| ---- | ----------- | ------------------------ | -------- |
| `arm-a2a3` | `self-hosted`、`linux`、`ARM64`、`npu-xp` | `26.0.rc1` | `arm-a2a3-cann9.2-dist` |
| `arm-a5` | `self-hosted`、`Linux`、`ARM64`、`npu-a5` | `25.6.rc2.b023` | `arm-a5-cann9.2-dist` |
| `x86-a5` | `self-hosted`、`Linux`、`npu-a5`、`X64`、`pr` | `25.7.rc1.12` | `x86-a5-cann9.2-dist` |

A5 基础镜像为 `9.2.0-950-ubuntu22.04-py3.11-devel`，分别固定到 ARM64 和 AMD64 manifest。
架构标签确保 A5 任务调度到匹配的原生 runner。doctor 将 CANN 9.2 A5 的驱动组合限定到对应宿主机架构。
`pr` 标签用于选择 x86 runner，不限制 workflow 的触发事件类型。
原计划中尚缺 `x86-a2a3`：需要原生 X64 A2/A3 runner、AMD64 CANN 基础镜像 digest、
经过验证的驱动组合及单卡、双卡验证。
A2/A3 预检还要求 runner 名称为 `227-npu-a2a3-XP-1`；若该标签池扩大，
应为宿主机分配专用标签并更新 `runners`。
每个 runner 需要提供 Python 3.10 的 `pypto-toolchain` 工具链包、Git、Docker、
Ascend Docker Runtime，以及支持 `--find`/`--kill` 的 `task-submit`；不使用宿主机的 PyPTO 安装。
宿主机维护命令会显式进入该工具链环境，不依赖可能较旧的系统 `python3`。
工具链检查打印的是宿主机 CANN 版本；镜像版本由 `cann_tag`/`cann_digest` 指定，
并由容器内的 strict doctor 核实。
workflow 在独立 Docker 配置中安装 Buildx v0.35.0，使用宿主机的默认 Docker builder，保留共享构建缓存。

首次发布前，在仓库设置中配置：

| 类型 | 名称 | 内容 |
| ---- | ---- | ---- |
| Secret | `SWR_USERNAME` | SWR 长期登录指令中的用户名 |
| Secret | `SWR_PASSWORD` | 该指令中的密码 |

凭据仅提供给发布步骤，通过 `--password-stdin` 登录。每次执行使用独立的临时 Docker 配置目录，
由 `always()` 步骤删除。不要提交临时登录 token。
workflow 设置 `PYPTO_DOCKER_PROXY=http://127.0.0.1:7892`，使用 CI runner 上的本地代理，
无需配置 GitHub Actions 代理 Variable。代理传给 Git 和 Docker build arguments，不写入最终镜像环境。
Buildx 安装和 GitHub artifact 上传也使用该代理。
手动调用脚本时，将 `PYPTO_DOCKER_PROXY` 环境变量设置为当前机器可访问的代理。
首期配置设置了 `proxy_required: true`，缺少该环境变量时预检直接失败。
`--network=host` 作用于构建步骤；拉取基础镜像仍依赖 Docker daemon 的网络和代理。
SWR 认证与 manifest 操作直接连接，不使用代理。登录和推送子进程也会清除代理变量；
如果 Docker daemon 自身配置了代理，其 `NO_PROXY` 必须包含 SWR 主机名。

首次构建前，确认配置中的 CANN 基础镜像可访问。默认 `cann_repository` 为 `quay.io/ascend/cann`；
若 daemon 无法访问，将对应架构的准确镜像同步到允许匿名读取的 SWR 仓库，再修改该字段。
每种架构均固定镜像 digest；切换仓库时，应将验证过的镜像仓库 digest 填入 `cann_digest`。
值为 null 时，每次拉取指定 tag 并解析 digest。
两种模式都会使用解析后的 digest 构建并记录到产物中；旧缓存不能掩盖拉取失败。
升级 CANN 时，应一起修改 tag、digest、驱动要求和发布标签。

每次构建前冻结 PyPTO 和 pypto-lib 的提交。PyPTO 的 `ref: main` 使用 workflow checkout 的提交，
保证 Docker 配方与辅助脚本属于同一次事件快照。PR 使用准确的 head 提交，从 PR 来源仓库
（包括 fork）克隆，不使用临时合并提交。显式指定的 PyPTO ref 仍使用配置中的来源仓库。
其他 ref 和 pypto-lib ref 解析为完整提交 SHA。
runtime 始终跟随 PyPTO 的 gitlink；PTO-ISA 跟随该 runtime 的 `pto_isa.pin`；
PTOAS 跟随所选 PyPTO 的工具链版本。使用这些 SHA 使源码构建缓存失效，而不是传入可变的 `main`。
编译并发同时受 `build_jobs` 和资源限制加载器约束。
CI 选择 simpler 的 `build_package_<validation_platform>` 和 `build_package_sim` 目标：
每个镜像预编译自身平台的上板 runtime 和两种 CPU 模拟器。
`DIST_RUNTIME_TARGETS` 将该选择保存为 `PYPTO_RUNTIME_BUILD_TARGETS`，供后续 `pypto-build`
继续使用；空值使用 simpler 的默认目标。该选项只影响 runtime wheel 的构建。
所选 PyPTO ref 必须提供 `pypto-doctor --strict`；验证会拒绝那些静默忽略严格选项的旧辅助脚本。

`cann92-a2a3-aicpu-class-memaccess` 和 `cann92-a5-aicpu-class-memaccess` 补丁
只增加对应平台在 CANN 9.2 GNU 编译器下所需的 onboard AICPU 警告豁免，每个镜像应用自身平台的补丁。
编译前应用补丁；若相同修改已经进入上游则跳过；上下文变化导致无法应用时直接失败。
成功构建后，`/opt/pypto/dist-state.json` 记录 runtime SHA、差异哈希、补丁哈希及安装的扩展哈希。
`pypto-doctor --strict` 仅接受该记录对应的补丁；其他修改、缺少编译版本标记或无法验证的绑定都会失败。
表中的驱动组合属于部署验证配置，发布仍须通过 strict doctor 和两个上板测试，
不代表对其他宿主机的通用兼容保证。

验证使用全新容器，不挂载源码或缓存卷，并设置 `PYPTO_BENCH=0` 执行正确性检查。
登录和构建成功后的提示会为 `a5` 选择 DeepSeek SWA，为 `a2a3` 选择 Qwen3 decode。
每个镜像配置通过 `decode` 选择单卡脚本，通过 `distributed` 选择双卡脚本；
可选的 `distributed_args` 列表按独立参数传入。A2/A3 使用 Qwen3-14B decode 和 pypto-lib 的 all-reduce 示例。
A5 使用 `models/deepseek_v4_1_flash/decode_swa.py` 验证 SWA、KV cache 和 MXFP8，
使用 `models/deepseek_v4_1_flash/decode_moe.py --ep 2` 验证专家计算和跨卡分发、合并。
两个 A5 用例自行生成输入，并与各自的 golden 参考结果比较。
一次单卡 `task-submit` 借卡先运行严格 doctor，再运行 decode；另一次借卡运行双卡用例。
物理卡来自 `TASK_DEVICE`，模型参数使用容器内编号 `0` 或 `0,1`。
宿主机驱动元数据以只读方式挂载。ARM A5 容器还传入 `/dev/ummu` 和 `/dev/uburma`，用于 URMA 通信。
两个模型都必须退出码为零且输出 `[RUN] PASS`；
缺少 PASS、超时或测试失败都禁止发布。`task_timeout_seconds` 限制排队加执行的总等待时间，
`task_max_seconds` 单独限制设备执行时间。总超时必须大于执行上限；首期每个用例允许总计 3600 秒，
其中设备执行最多 1800 秒。清理步骤约束孤儿任务，
只删除本次运行的容器和任务，不清理共享构建缓存。

上传前及切换正式标签前，发布步骤都会确认构建配方提交仍为上游最新 `main`。
旧运行重跑或无法查询上游时直接失败，不修改最新与前一个标签。
检查使用配方提交，因此仍支持显式固定 PyPTO 源码版本。
重试应构建最新 `main`；回滚需要单独显式执行。

验证通过后，先将测试过的准确 image ID 推送到临时标签
`arm-a2a3-cann9.2-dist-<run_id>-<attempt>`，核对 manifest 和 config digest 与测试镜像一致后才发布。
先将现有 `arm-a2a3-cann9.2-dist` 移到 `arm-a2a3-cann9.2-dist-previous`，再更新为新镜像。
对相同 digest 重试不会覆盖前一个版本。首次发布仅创建最新标签；第二次不同镜像发布才创建前一个标签。
临时标签按名称删除，绝不删除共用 digest。成功发布后，每个配置的镜像只保留这两个标签。
配置检查会跨镜像预留最新标签及生成的 `-previous` 标签，在调度原生 job 前拒绝名称冲突。

workflow 不会更新或删除 `a3-dist`、`a3-dist-20260918` 等旧标签。
等两个新构建均通过上板验证、完成上传，且最新与前一个标签均可用后，再单独清理旧标签。
未核实内容前，不要把旧镜像改名为 CANN 9.2。

Actions Summary 和保留 14 天的 artifacts 包含提交版本、digest、构建与验证日志，以及发布前的 manifest，
供失败恢复使用。将最新标签重新指向前一个 digest 即可回滚，无需重建。
发布跨越多个仓库请求：若中途失败，重试前应检查两个标签及保存的 manifest。
临时标签删除失败会使 job 失败，可能需要手动删除该次运行的具体标签；不得删除其 digest 或共享层。

镜像内的源码是 detached 快照，`pypto-update.sh` 不会直接拉取更新，受管补丁也会使 runtime 工作树变脏。
获取新快照时，拉取新镜像并创建新容器或源码卷。已有 named volume 会保留原来的源码。
希望跟踪分支的开发者，需要显式处理 checkout 和本地补丁，然后再使用更新脚本。

后续添加 `x86-a2a3` 或其他 CANN 版本的镜像时，新增配置项并指定匹配的原生 runner、CANN 基础镜像、
驱动和硬件验证路径；环境准备好前保持禁用。各组合独立发布标签；Docker 架构 manifest 无法按宿主机 NPU 选择镜像。

### 不装 Ascend Docker Runtime

未注册 Ascend runtime 时，启动器默认的 `auto` 模式会选择 `privileged`。
这种回退需要运行 privileged Docker 容器的权限，并会暴露宿主机的所有设备和 NPU 卡。
启动器会在创建或重新进入容器之前说明这一行为。允许这类访问时，可以显式选择：

```bash
PYPTO_DEVICE_ACCESS=privileged docker/scripts/pypto-dev.sh run a3 1
```

启动器挂载宿主机驱动目录和管理工具，并设置 `PYPTO_DEVICE=1`，将物理卡 1 作为默认计算设备。
这个设置不会限制容器访问其他卡或宿主机设备。
A5 请在适配的宿主机上使用 `run a5`；启动器还会传入所需的 URMA 设备。

只传入四个设备节点和驱动挂载并不是可移植的替代方案：设备枚举可能报
`drvGetDevNum`/`507899`，包括在驱动 `26.0.rc1` 上。因此，本指南不再声称这种受限方式
可以在该驱动上工作。需要设备隔离时，应使用已注册的 `--runtime=ascend` 路径；
它会注入驱动，并只暴露指定的卡。如果不允许 privileged 容器，请联系宿主机管理员配置该 runtime。

## 边界情况

> **致命陷阱：** 宿主机驱动必须是 `26.0.rc1`，以配套镜像内的 CANN 9.0.0。版本不匹配不会在启动时报
> 错 —— 它会在第一次 AICPU 初始化时失败，此时编译和 `aclrtSetDevice` 早已成功，看起来非常像内核 bug。

| 症状 | 可能原因 | 处理 |
| ---- | -------- | ---- |
| `unknown or invalid runtime name: ascend` | Ascend Docker Runtime 未安装、未注册，或安装后没重启 Docker | `sudo systemctl daemon-reload && sudo systemctl restart docker`，再查 `docker info --format '{{json .Runtimes}}'`；或改用 [privileged 回退方案](#不装-ascend-docker-runtime) |
| `exec format error` | 在 x86_64 宿主机上跑 ARM64 镜像 | 换到 `aarch64` 宿主机 —— 没有绕过办法 |
| `aclrtSetDevice` 返回 `507899` 或 `Resource_Busy` | 同一张物理卡被另一个容器或进程占用 | 用 `npu-smi info` 和 `docker ps` 找到并停掉它，或换一张卡。之后 ACL 上下文可能已进入 sticky error 状态 —— 重建容器再测 |
| `simpler_init failed with code 507018` | 几乎总是驱动 / CANN 不匹配 | 见[排查 `507018`](#排查-507018) |
| `npu-smi` 报 `-9005`，或 `DrvMngGetConsoleLogLevel failed (ret=4)` | 只暴露了部分卡，而 `npu-smi` 试图枚举整机 | 无害。以 `pypto-doctor` 的 `device access: aclrtSetDevice(0) ok` 为准 |
| `/workspace/pypto` 不存在 | 绑定挂载高了一层 | 挂载包含 `pypto/` 的那一层：用 `-v "$PWD/workspace:/workspace"`，而不是 `-v "$PWD:/workspace"` |
| `pypto-doctor` 报 simpler binding 与源码不匹配 | 挂进来的源码与镜像构建时用的不是同一份 | `pypto-build && pypto-doctor` |
| codegen 报 `ptoas at '...' is version X, but PyPTO requires PTOAS >= vY` | PTOAS 低于 checkout 要求的版本 | 按[更新源码并重建](#更新源码并重建)运行仓库中的安装器；绑定挂载和旧镜像也适用 |
| `pto-isa` 试图联网克隆 | 挂载的源码 bump 了 `runtime/pto_isa.pin`，或 managed checkout 被改动过 | pin 变了之后属于预期行为。若 GitHub HTTP/2 不稳定，执行 `git config --global http.version HTTP/1.1` —— resolver 会重试 GitHub，失败后回退到 GitCode 镜像 |

### 排查 `507018`

失败长这样 —— 编译、显存分配、打开设备全部成功，只有运行时阶段失败：

```text
[RUN] runtime ...
[ERROR] ensure_aicpu_init_launched: [device_runner_base.cpp:532]
        ensure_aicpu_init_launched: stream sync failed: 507018 (device_id=0)
RuntimeError: simpler_init failed with code 507018
```

`507018` 是 `ACL_ERROR_RT_AICPU_EXCEPTION`，而且它是一个**通用码**：多种互不相关的 device 侧机制都会
汇聚到这一个数字上，所以仅凭它无法断定是死锁、显存不足还是某个算子有问题。完整分类见运行时自己的
`runtime/docs/troubleshooting/device-error-codes.md`。

不过调用栈把范围缩得很小。`ensure_aicpu_init_launched` 属于 AICPU **初始化**，发生在任何用户内核执行
之前：此时 PyPTO 刚把 AICPU runtime SO 下发到 device 并第一次拉起它。失败点在环境，不在你跑的那个
模型。

**第一，核对宿主机驱动版本。** 这是容器场景下最常见的原因。要在**宿主机**上执行 —— `--runtime=ascend`
的容器内不暴露 `version.info`：

```bash
cat /usr/local/Ascend/driver/version.info
```

对于 A3 镜像的 CANN 9.0.0，`Version=` 必须是 `26.0.rc1`。
`compatible_version=` 使用内部标识，不应在其中搜索 `9.0.0`。
其他公开版本组合不在该镜像的支持范围内（运行时在 `runtime/docs/install.md` 中列出了支持的组合）；
先把驱动和固件升级到配套版本再复测。

**第二，确认芯片系列与 `-p` 一致。** 用错 arch 同样会以 `507018` 的形式出现，而且看起来非常像编译器
bug。`a2a3` 覆盖 `Ascend910*`（A2）和 `Ascend910_93*`（A3），`a5` 覆盖 `Ascend950*`：

```bash
npu-smi info -t board -i 0 -c 0 | grep -iE 'Chip Name|NPU Name'
```

**第三，确认没有两个容器共用同一颗芯片的一对 die。** 在 a2a3 上，`npu-smi` 中相邻的 Phy-ID 是同一颗
`Ascend910` 的 die0 和 die1，它们共享 device 侧用于暂存 AICPU SO 的目录。并发引导曾导致该文件损坏并
触发 AICPU 异常。分配卡时按整颗芯片分配。

**第四，抓 device 侧日志。** host 日志只能看到级联结果，真正的原因在 CANN 的 device slog 里：

```bash
CARD_ID=0   # 一张空闲物理卡

docker run --rm \
  --runtime=ascend -e ASCEND_VISIBLE_DEVICES="$CARD_ID" \
  --shm-size=16g \
  -e ASCEND_SLOG_PRINT_TO_STDOUT=1 \
  -e ASCEND_GLOBAL_LOG_LEVEL=1 \
  "$PYPTO_IMAGE" \
  -c 'python $PYPTO_SRC/examples/beginner/01_hello_world.py' \
  > slog.txt 2>&1
```

在 `slog.txt` 中找**第一条** `PrintAicpuErrorInfo` 或 `ProcessStarsAicpuErrorInfo`，记下它的 `soName`、
`funcName`、`errorCode` 和 `chipId/dieId`；其后的每一条 `507899`、`507901` 都是级联噪声。同时检查提到
`version_verify` 和 `package_process_config` 的行，版本校验失败或 AICPU 包下发失败会在那里留下记录。

反馈问题时，请附上宿主机的 `version.info`、`npu-smi info`、完整的 `docker run` 命令、`pypto-doctor`
的完整输出，以及 `slog.txt`。

## 参见

- [安装](01-installation.md) —— 从源码安装的路径，以及一次安装能给你什么、不能给你什么。
- [快速上手](02-quickstart.md) —— 环境就绪之后的第一个内核。
- [PTO 项目生态](../dev/00-ecosystem.md) —— PyPTO、PTOAS、pto-isa 和运行时之间的关系，以及它们的版本
  为什么必须一起走。
- [运行时文档](https://hw-native-sys.github.io/simpler/) —— 安装和操作执行已编译程序的运行时。
