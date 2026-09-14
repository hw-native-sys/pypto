# 开发容器

拉取一个已经装好 CANN、PyPTO 和运行时的镜像，不用自己拼工具链，直接在真实昇腾卡上跑内核。

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
| 已构建的 PyPTO + `simpler` | 镜像 | `/workspace/pypto`，已编译完成 |
| `pypto-lib` 模型与示例 | 镜像 | `/workspace/pypto-lib`，可通过 `$PYPTO_LIB_SRC` 访问 |

**宿主机驱动版本不是自由项。** 镜像带的是 CANN 用户态，不带驱动。AICPU 的 device 侧包由容器内的
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
| ptoas | 0.61 —— 与 `toolchain/versions.env` 所 pin 的版本一致 |
| 未压缩大小 | 约 16.7 GB |
| 工作目录 | `/workspace` |
| 默认入口 | 已激活 PyPTO 环境的交互式 bash |
| 已验证摘要 | `sha256:ffa6bef8ed34f565a2331a66369902303ff957e832476fa1abaad2be435b82b5` |

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

第四个脚本 `pypto-update.sh` 不在 `PATH` 上，而是放在源码树里 ——
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
而容器里只有你暴露的那几张。判断卡能不能用，以 `device access: aclrtSetDevice(0) ok` 那行为准。

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

`version.info` 里 `Version=` 是驱动版本，`compatible_version=` 是它接受的 CANN 版本范围。所有版本
争议都以这两行为准。

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

### 更新源码并重建

拉取最新的 main、改了卷里的 C++、或挂了另一份 checkout 进来 —— 三种情形最后都要重建。镜像里带了
一个脚本，把整套流程一次做完：

```bash
/workspace/pypto/.github/docker/pypto-update.sh
```

它会拉取 pypto（含子模块）和 pypto-lib，在 `toolchain/versions.env` 变动时重装 ptoas，跑
`pypto-build`，最后以 `pypto-doctor` 收尾。和前面三个命令不同，它**不在** `PATH` 上 —— 用上面的
全路径调用。

| 参数 | 作用 |
| ---- | ---- |
| `--no-pull` | 只重建，保留本地改动 |
| `--reinstall-deps` | 连 pip 依赖一起重装（torch、numpy、clang-tidy…） |
| `--proxy http://host:port` | git 拉取和 ptoas 下载走的代理 |
| `--remote-submodules` | 子模块追各自远程最新，而不是跟随所 pin 的提交 |
| `--jobs N` | 构建并行度 |
| `--skip-doctor` | 跳过收尾的 `pypto-doctor` |

> **它会丢弃本地工作，已提交的也算。** 拉取用的是 `git reset --hard origin/main`，它移动的是分支
> 本身 —— 先 commit 救不了你，那个提交只是不再被分支引用而已。要么用 `--no-pull`，要么在跑之前
> 留一个引用：
>
> ```bash
> git -C /workspace/pypto branch backup/before-update
> ```
>
> 未跟踪的文件不受影响，这也是它不会把自己删掉的原因：它在那份 checkout 里本身就是未跟踪的。

**三种情形里只有两种跑得了它。** 这个脚本住在镜像自带的源码树里，既不在仓库中也不在 `PATH` 上，
所以一旦用绑定挂载盖住 `/workspace`，它会和镜像在那儿的其他东西一起被藏起来。命名卷没问题 ——
空卷会从镜像预填充，脚本一并被复制进去。用绑定挂载的 checkout 时，手工重建：

```bash
git -C /workspace/pypto pull
git -C /workspace/pypto submodule update --init --recursive
git -C /workspace/pypto-lib pull
pypto-build
pypto-doctor
```

这几步是**部分**替代，不是等价物。脚本做而这里没做的有两件：`toolchain/versions.env` 的 pin 变动时
重装 ptoas，以及把代理和并行度设置一路传给两者。pin 变动之后手工比对汇编器：

```bash
grep '^PTOAS_VERSION=' /workspace/pypto/toolchain/versions.env
/opt/pypto/ptoas/bin/ptoas --version
```

`pypto-build` 用原生 CMake 重建 PyPTO，安装与当前 `runtime/` 源码匹配的 `simpler`，并解析该源码指定
的 `pto-isa` pin。它刻意不用 `pip install -e .`：被 scikit-build 配置过的 `build/` 会把扩展放进
`build/python/bindings/` 而不是 `python/pypto/`，此后每次构建都"成功"，而 Python 一直导入过期的
`.so` —— C++ 改动看起来完全没生效。

`pypto-build` **不做**的事情是更新 ptoas —— 这一步只有 `pypto-update.sh` 会做。

### 不装 Ascend Docker Runtime

装不了 Runtime（没有 root、不能重启 Docker、维护窗口未到）时，自己把设备节点和驱动传进去：

```bash
docker run --rm -it \
  --device /dev/davinci1 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  --shm-size=16g \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  "$PYPTO_IMAGE"
```

`davinci1` 是你要用的卡；另外三个是驱动管理节点，必须和它一起传入。`driver` 挂载提供驱动库。
`dcmi`、`npu-smi`、`ascend_install.info` 三个挂载只决定容器内 `npu-smi` 能不能用，可以省略。

**这种方式下卡号不重映射。** 容器内看到的仍是 `/dev/davinci1`，但 ACL 只枚举到这一张卡，所以 PyPTO
参数仍是 `-d 0`；传入 `davinci2`、`davinci3` 时对应 `-d 0,1`。若希望在多卡可见时按物理卡号选择，用
CANN 自己的变量，逻辑编号按列出顺序从 0 开始：

```bash
-e ASCEND_RT_VISIBLE_DEVICES=2,3
```

该方案已在 `driver 26.0.rc1` 配本镜像上验证：`pypto-doctor` 全项通过，单卡和双卡用例均通过。它不需要
`--privileged`，而且 `--privileged` 不是等价替代 —— 那会把整机所有卡都交给容器。只有在上面的做法仍然
报 `507899` 时，才把它当最后手段。

`--runtime=ascend` 依然是推荐方式：它自动注入驱动与设备节点、重新编号卡、并且只暴露你指定的那几张。

## 边界情况

> **致命陷阱：** 宿主机驱动必须是 `26.0.rc1`，以配套镜像内的 CANN 9.0.0。版本不匹配不会在启动时报
> 错 —— 它会在第一次 AICPU 初始化时失败，此时编译和 `aclrtSetDevice` 早已成功，看起来非常像内核 bug。

| 症状 | 可能原因 | 处理 |
| ---- | -------- | ---- |
| `unknown or invalid runtime name: ascend` | Ascend Docker Runtime 未安装、未注册，或安装后没重启 Docker | `sudo systemctl daemon-reload && sudo systemctl restart docker`，再查 `docker info --format '{{json .Runtimes}}'`；或改用 [`--device` 回退方案](#不装-ascend-docker-runtime) |
| `exec format error` | 在 x86_64 宿主机上跑 ARM64 镜像 | 换到 `aarch64` 宿主机 —— 没有绕过办法 |
| `aclrtSetDevice` 返回 `507899` 或 `Resource_Busy` | 同一张物理卡被另一个容器或进程占用 | 用 `npu-smi info` 和 `docker ps` 找到并停掉它，或换一张卡。之后 ACL 上下文可能已进入 sticky error 状态 —— 重建容器再测 |
| `simpler_init failed with code 507018` | 几乎总是驱动 / CANN 不匹配 | 见[排查 `507018`](#排查-507018) |
| `npu-smi` 报 `-9005`，或 `DrvMngGetConsoleLogLevel failed (ret=4)` | 只暴露了部分卡，而 `npu-smi` 试图枚举整机 | 无害。以 `pypto-doctor` 的 `device access: aclrtSetDevice(0) ok` 为准 |
| `/workspace/pypto` 不存在 | 绑定挂载高了一层 | 挂载包含 `pypto/` 的那一层：用 `-v "$PWD/workspace:/workspace"`，而不是 `-v "$PWD:/workspace"` |
| `pypto-doctor` 报 simpler binding 与源码不匹配 | 挂进来的源码与镜像构建时用的不是同一份 | `pypto-build && pypto-doctor` |
| ptoas 报错说生成代码里某个指令形式不接受 | 手工更新源码之后，汇编器落在了 `toolchain/versions.env` 后面 | 跑 `/workspace/pypto/.github/docker/pypto-update.sh` —— 只有它会重装 ptoas。镜像发布时二者是对齐的（0.61） |
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

`Version=` 必须是 `26.0.rc1`，`compatible_version=` 必须包含镜像内的 CANN 9.0.0。其他组合都属于未支持
的环境（运行时在 `runtime/docs/install.md` 里列出了它支持什么）；先把驱动和固件升级到配套版本再复测。

**第二，确认芯片系列与 `-p` 一致。** 用错 arch 同样会以 `507018` 的形式出现，而且看起来非常像编译器
bug。`a2a3` 覆盖 `Ascend910B*`（A2）和 `Ascend910_93*`（A3），`a5` 覆盖 `Ascend950*`：

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
