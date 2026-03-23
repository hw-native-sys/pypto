# TPUSH/TPOP 指令（TPUSH/TPOP Instructions）

## 概述

`TPUSH` 和 `TPOP` 是在同一集群内 Cube 和 Vector 核心上协同调度的 InCore 内核之间移动 Tile 的主要数据通信指令。它们在多槽环形缓冲区上实现基于标签（Tag）的双通道 FIFO 协议。

参见 [集群架构](00-cluster_architecture.md) 了解硬件背景。

## 动机

当混合 InCore 函数被分解为协同调度的内核（如 Vector 上的数据搬运、Cube 上的计算）时，这些内核需要高效的同步数据通道。带环形缓冲区流控的 TPUSH/TPOP 正是为此设计。

## 生产者 / 消费者角色

角色是概念性的，不绑定到特定核心类型：

| 场景 | 生产者 | 消费者 |
| ---- | ------ | ------ |
| 矩阵乘法输出 → 后处理 | Cube | Vector |
| 数据加载 → 计算 | Vector | Cube |
| 双向 | 双方（相反方向） | 双方（相反方向） |

## 环形缓冲区结构

每个环形缓冲区是一个**单向**通道。槽数量（`SLOT_NUM`）取决于通信模式：

| 模式 | SLOT_NUM | 使用的标志位 |
| ---- | -------- | ------------ |
| 单向 | 8 | 每方向全部 8 个标志位 |
| 双向（每方向） | 4 | 每个环形缓冲区 4 个标志位 |

```text
Unidirectional (SLOT_NUM=8):
    Cube (producer)  ──── slot[0..7], flags 0..7 ────▶  Vector (consumer)

Bidirectional (SLOT_NUM=4 per direction):
    Cube  ──── Ring Buffer A (slot[0..3], flags 0..3) ────▶  Vector
    Cube  ◀──── Ring Buffer B (slot[0..3], flags 4..7) ────  Vector
```

每个槽容纳一个 Tile，由**标签**（0 .. SLOT_NUM-1）标识。两个信号通道承载按标签的通知：

| 信号 | 发送方 | 接收方 | 含义 |
| ---- | ------ | ------ | ---- |
| `SET P2C: tag` | 生产者 | 消费者 | `slot[tag]` 中的数据已就绪 |
| `SET C2P: tag` | 消费者 | 生产者 | `slot[tag]` 可复用 |
| `WAIT P2C: tag` | 消费者 | — | 阻塞直到 `slot[tag]` 就绪 |
| `WAIT C2P: tag` | 生产者 | — | 阻塞直到 `slot[tag]` 可用 |

## 常量与参数

### 平台（Platform）

```cpp
enum PlatformID : uint8_t {
    PLATFORM_A2A3 = 0,   // Ring buffer in Global Memory
    PLATFORM_A5   = 1,   // Ring buffer in consumer's on-chip SRAM
};
```

`PLATFORM_ID` 是嵌入内核二进制中的编译时常量。

| PLATFORM_ID | 环形缓冲区位置 | Push 行为 | Pop 行为 |
| ----------- | -------------- | --------- | -------- |
| `PLATFORM_A2A3` | GM | DMA tile → GM 槽 | DMA GM 槽 → 本地 tile |
| `PLATFORM_A5` | 消费者 SRAM | DMA tile → 消费者 SRAM 槽 | 零拷贝：直接引用本地 SRAM |

### 方向（Direction）

```cpp
enum Direction : uint8_t {
    DIR_C2V = 1,   // Cube → Vector  (0b01)
    DIR_V2C = 2,   // Vector → Cube  (0b10)
};
```

`DIR_MASK` 是活跃方向的位掩码：

| DIR_MASK | 值 | SLOT_NUM |
| -------- | -- | -------- |
| `DIR_C2V` | `0b01` | 8 |
| `DIR_V2C` | `0b10` | 8 |
| `DIR_C2V \| DIR_V2C` | `0b11` | 每方向 4 |

## 初始化 API

### `aic_initialize_pipe` / `aiv_initialize_pipe`

分别在 Cube (AIC) 和 Vector (AIV) 核心启动时调用。两者共享相同的函数签名：

| 参数 | 类型 | 说明 |
| ---- | ---- | ---- |
| `DIR_MASK` | `uint8_t` | 活跃方向位掩码 |
| `SLOT_SIZE` | `uint32_t` | 每槽字节数（= Tile 大小） |
| `GM_SLOT_BUFFER` | `__gm__ void*` | GM 缓冲区（A2A3）；A5 上为 `nullptr` |
| `C2V_CONSUMER_BUF` | `uint32_t` | C2V 方向消费者 SRAM 基址（仅 A5；A2A3 上为 `0`） |
| `V2C_CONSUMER_BUF` | `uint32_t` | V2C 方向消费者 SRAM 基址（仅 A5；A2A3 上为 `0`） |

**行为：**

1. 根据 `DIR_MASK` 计算 `SLOT_NUM`（单向为 8，双向为 4）
2. 根据平台和方向绑定环形缓冲区到后端内存
3. 对于本核心作为**消费者**的方向，预先信号通知所有槽为可用状态

各平台的环形缓冲区绑定：

| 平台 | C2V 缓冲区 | V2C 缓冲区 |
| ---- | ---------- | ---------- |
| A2A3 | `GM_SLOT_BUFFER`（偏移 0） | `GM_SLOT_BUFFER` + `SLOT_NUM * SLOT_SIZE` |
| A5 | `C2V_CONSUMER_BUF`（Vector 的 UB） | `V2C_CONSUMER_BUF`（Cube 的 L1） |

**伪代码**（以 `aic_initialize_pipe` 为例；`aiv_initialize_pipe` 对称，消费者/生产者角色互换）：

```text
function aic_initialize_pipe(DIR_MASK, SLOT_SIZE, GM_SLOT_BUFFER,
                             C2V_CONSUMER_BUF, V2C_CONSUMER_BUF):
    SLOT_NUM = 4 if (DIR_MASK == (DIR_C2V | DIR_V2C)) else 8

    if DIR_MASK & DIR_C2V:        // Cube is PRODUCER
        c2v_ring_buf = GM_SLOT_BUFFER if A2A3 else C2V_CONSUMER_BUF
        c2v_target_tag = 0

    if DIR_MASK & DIR_V2C:        // Cube is CONSUMER
        // On A2A3 bidirectional, V2C buffer follows C2V buffer in GM
        v2c_offset = SLOT_NUM * SLOT_SIZE if (A2A3 and DIR_MASK & DIR_C2V) else 0
        v2c_ring_buf = GM_SLOT_BUFFER + v2c_offset if A2A3 else V2C_CONSUMER_BUF
        v2c_target_tag = 0
        for i in 0..SLOT_NUM-1:   // Pre-signal all slots as free
            SET flag_V2C_free: i
```

## 数据传输指令

四条方向特定的指令 — 方向编码在操作码中，无需运行时 `DIR` 参数：

| 指令 | 核心 | 角色 | 方向 |
| ---- | ---- | ---- | ---- |
| `tpush_to_aiv(TILE, SPLIT)` | Cube | 生产者 | C2V |
| `tpush_to_aic(TILE, SPLIT)` | Vector | 生产者 | V2C |
| `tpop_from_aic(TILE, SPLIT)` | Vector | 消费者 | C2V |
| `tpop_from_aiv(TILE, SPLIT)` | Cube | 消费者 | V2C |

**参数（tpush）：**

| 参数 | 类型 | 说明 |
| ---- | ---- | ---- |
| `TILE` | `Tile&` | 要推送的源 tile |
| `SPLIT` | `int` | 分割模式（0=无，1=上下，2=左右） |

**参数（tpop）：**

| 参数 | 类型 | 说明 |
| ---- | ---- | ---- |
| `TILE` | `Tile&` | 接收数据的目标 tile |
| `SPLIT` | `int` | 分割模式（0=无，1=上下，2=左右） |

### Push 协议（`tpush_to_aiv` / `tpush_to_aic`）

```text
function tpush_*(TILE, SPLIT):
    // 1. Wait for slot to be free (consumer has released it)
    WAIT flag_free[dir]: target_tag

    // 2. DMA tile data into ring buffer slot
    MTE_copy(src=TILE.data, dst=ring_buf + target_tag * SLOT_SIZE, size=SLOT_SIZE)
    SET mte_flag; WAIT mte_flag           // ensure DMA complete

    // 3. Signal consumer: data in slot is ready
    SET flag_ready[dir]: target_tag

    // 4. Advance to next slot
    target_tag = (target_tag + 1) % SLOT_NUM
```

### Pop 协议（`tpop_from_aic` / `tpop_from_aiv`）

```text
function tpop_*(TILE, SPLIT):
    // 1. Wait for data to be ready (producer has filled the slot)
    WAIT flag_ready[dir]: target_tag

    // 2. Load tile data from ring buffer slot
    if PLATFORM_A2A3:
        MTE_copy(src=ring_buf + target_tag * SLOT_SIZE, dst=TILE.data, size=SLOT_SIZE)
        SET mte_flag; WAIT mte_flag       // DMA slot → local tile
    else:  // PLATFORM_A5
        TILE.data = ring_buf + target_tag * SLOT_SIZE   // zero-copy

    // 3. Signal producer: slot is free for reuse
    SET flag_free[dir]: target_tag

    // 4. Advance to next slot
    target_tag = (target_tag + 1) % SLOT_NUM
```

## 标志位分配（Flag Assignment）

```text
Unidirectional (SLOT_NUM=8):
    flag_ready[dir] : P2C channel, flags 0..7   (producer SETs, consumer WAITs)
    flag_free [dir] : C2P channel, flags 0..7   (consumer SETs, producer WAITs)
    Note: ready and free use opposite hardware channels (P2C vs C2P),
          so the same tag indices refer to different physical flags.

Bidirectional (SLOT_NUM=4):
    C2V: flags 0..3   (ready on P2C channel, free on C2P channel)
    V2C: flags 4..7   (ready on P2C channel, free on C2P channel)
```

## 时序图（C2V，SLOT_NUM=4）

```text
          iter 0           iter 1           iter 2           iter 3
tag:        0                1                2                3

Cube:     tpush_to_aiv   tpush_to_aiv   tpush_to_aiv   tpush_to_aiv
          WAIT f:0        WAIT f:1        WAIT f:2        WAIT f:3
          MTE → slot[0]   MTE → slot[1]   MTE → slot[2]   MTE → slot[3]
          SET r:0         SET r:1         SET r:2         SET r:3
              │ ready         │ ready         │ ready         │ ready
              ▼               ▼               ▼               ▼
Vector:       tpop_from_aic  tpop_from_aic  tpop_from_aic  tpop_from_aic
              WAIT r:0       WAIT r:1       WAIT r:2       WAIT r:3
              use [0]        use [1]        use [2]        use [3]
              SET f:0        SET f:1        SET f:2        SET f:3

Legend: r = flag_ready, f = flag_free
        Consumer init pre-SETs f:0..3 so producer does not block initially
```

## 时序图（双向，SLOT_NUM=4）

```text
          iter 0              iter 1              iter 2              iter 3

AIC (Cube):
  C2V:    tpush_to_aiv      tpush_to_aiv      tpush_to_aiv      tpush_to_aiv
          tag=0               tag=1               tag=2               tag=3
  V2C:         tpop_from_aiv      tpop_from_aiv      tpop_from_aiv      tpop_from_aiv
               tag=0               tag=1               tag=2               tag=3

AIV (Vector):
  V2C:    tpush_to_aic       tpush_to_aic       tpush_to_aic       tpush_to_aic
          tag=0               tag=1               tag=2               tag=3
  C2V:         tpop_from_aic       tpop_from_aic       tpop_from_aic       tpop_from_aic
               tag=0               tag=1               tag=2               tag=3

Flag usage (per AIV peer):
  flags 0..3 : C2V direction (ready + free)
  flags 4..7 : V2C direction (ready + free)
```

## 关键特性

1. **无死锁** — 消费者在主循环前预先信号通知所有槽为可用
2. **背压（Backpressure）** — 生产者在所有槽满时阻塞；消费者在无数据时阻塞
3. **FIFO 顺序** — 严格的轮询 `(tag + 1) % SLOT_NUM`
4. **解耦 DMA** — 异步 MTE 传输，显式标志位等待
5. **伙伴核心选择** — 由初始化处理，非逐指令指定
6. **静态方向** — 方向编码在操作码中，编译时验证

## API 总结

| API | 核心 | 角色 | 说明 |
| --- | ---- | ---- | ---- |
| `aic_initialize_pipe(...)` | Cube | 初始化 | 绑定环形缓冲区，初始化标签，预信号 V2C 空闲槽 |
| `aiv_initialize_pipe(...)` | Vector | 初始化 | 绑定环形缓冲区，初始化标签，预信号 C2V 空闲槽 |
| `tpush_to_aiv(TILE, SPLIT)` | Cube | 生产者 | 等待空闲 → DMA tile → 信号就绪 |
| `tpush_to_aic(TILE, SPLIT)` | Vector | 生产者 | 等待空闲 → DMA tile → 信号就绪 |
| `tpop_from_aic(TILE, SPLIT)` | Vector | 消费者 | 等待就绪 → 加载 tile → 信号空闲 |
| `tpop_from_aiv(TILE, SPLIT)` | Cube | 消费者 | 等待就绪 → 加载 tile → 信号空闲 |
