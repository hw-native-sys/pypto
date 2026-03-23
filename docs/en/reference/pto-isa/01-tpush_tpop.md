# TPUSH/TPOP Instructions

## Overview

`TPUSH` and `TPOP` are the primary data communication instructions for moving tiles between InCore kernels co-scheduled on Cube and Vector cores within the same cluster. They implement a tag-based dual-channel FIFO protocol over a multi-slot ring buffer.

See [Cluster Architecture](00-cluster_architecture.md) for hardware background.

## Motivation

When a mixed InCore function is decomposed into co-scheduled kernels (e.g., data-movement on Vector, compute on Cube), these kernels need an efficient, synchronized data channel. TPUSH/TPOP with ring buffer flow control provides this.

## Producer / Consumer Roles

Roles are conceptual and not bound to a specific core type:

| Scenario | Producer | Consumer |
| -------- | -------- | -------- |
| Matmul output → post-processing | Cube | Vector |
| Data loading → compute | Vector | Cube |
| Bidirectional | Both (opposite directions) | Both (opposite directions) |

## Ring Buffer Structure

Each ring buffer is a **unidirectional** channel. Slot count (`SLOT_NUM`) depends on the communication pattern:

| Pattern | SLOT_NUM | Flags Used |
| ------- | -------- | ---------- |
| Unidirectional | 8 | All 8 flags per direction |
| Bidirectional (per direction) | 4 | 4 flags per ring buffer |

```text
Unidirectional (SLOT_NUM=8):
    Cube (producer)  ──── slot[0..7], flags 0..7 ────▶  Vector (consumer)

Bidirectional (SLOT_NUM=4 per direction):
    Cube  ──── Ring Buffer A (slot[0..3], flags 0..3) ────▶  Vector
    Cube  ◀──── Ring Buffer B (slot[0..3], flags 4..7) ────  Vector
```

Each slot holds one Tile, identified by a **tag** (0 .. SLOT_NUM-1). Two signal channels carry per-tag notifications:

| Signal | Sender | Receiver | Meaning |
| ------ | ------ | -------- | ------- |
| `SET P2C: tag` | Producer | Consumer | Data in `slot[tag]` is ready |
| `SET C2P: tag` | Consumer | Producer | `slot[tag]` is free for reuse |
| `WAIT P2C: tag` | Consumer | — | Block until `slot[tag]` is ready |
| `WAIT C2P: tag` | Producer | — | Block until `slot[tag]` is free |

## Constants and Parameters

### Platform

```cpp
enum PlatformID : uint8_t {
    PLATFORM_A2A3 = 0,   // Ring buffer in Global Memory
    PLATFORM_A5   = 1,   // Ring buffer in consumer's on-chip SRAM
};
```

`PLATFORM_ID` is a compile-time constant embedded in the kernel binary.

| PLATFORM_ID | Ring Buffer Location | Push Behavior | Pop Behavior |
| ----------- | -------------------- | ------------- | ------------ |
| `PLATFORM_A2A3` | GM | DMA tile → GM slot | DMA GM slot → local tile |
| `PLATFORM_A5` | Consumer's SRAM | DMA tile → consumer's SRAM slot | Zero-copy: reference local SRAM directly |

### Direction

```cpp
enum Direction : uint8_t {
    DIR_C2V = 1,   // Cube → Vector  (0b01)
    DIR_V2C = 2,   // Vector → Cube  (0b10)
};
```

`DIR_MASK` is a bitmask of active directions:

| DIR_MASK | Value | SLOT_NUM |
| -------- | ----- | -------- |
| `DIR_C2V` | `0b01` | 8 |
| `DIR_V2C` | `0b10` | 8 |
| `DIR_C2V \| DIR_V2C` | `0b11` | 4 per direction |

## Initialization APIs

### `aic_initialize_pipe` / `aiv_initialize_pipe`

Called at kernel startup on Cube (AIC) and Vector (AIV) respectively. Both share the same signature:

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| `DIR_MASK` | `uint8_t` | Active directions bitmask |
| `SLOT_SIZE` | `uint32_t` | Bytes per slot (= Tile size) |
| `GM_SLOT_BUFFER` | `__gm__ void*` | GM buffer (A2A3); `nullptr` on A5 |
| `C2V_CONSUMER_BUF` | `uint32_t` | Consumer SRAM base for C2V (A5 only; `0` on A2A3) |
| `V2C_CONSUMER_BUF` | `uint32_t` | Consumer SRAM base for V2C (A5 only; `0` on A2A3) |

**Behavior:**

1. Compute `SLOT_NUM` from `DIR_MASK` (8 if unidirectional, 4 if bidirectional)
2. Bind ring buffer to backing memory based on platform and direction
3. For directions where this core is the **consumer**, pre-signal all slots as free

Ring buffer binding per platform:

| Platform | C2V buffer | V2C buffer |
| -------- | ---------- | ---------- |
| A2A3 | `GM_SLOT_BUFFER` (offset 0) | `GM_SLOT_BUFFER` + `SLOT_NUM * SLOT_SIZE` |
| A5 | `C2V_CONSUMER_BUF` (Vector's UB) | `V2C_CONSUMER_BUF` (Cube's L1) |

**Pseudocode** (showing `aic_initialize_pipe`; `aiv_initialize_pipe` is symmetric with consumer/producer roles swapped):

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

## Transfer Instructions

Four direction-specific instructions — direction is encoded in the opcode, no runtime `DIR` argument:

| Instruction | Core | Role | Direction |
| ----------- | ---- | ---- | --------- |
| `tpush_to_aiv(TILE, SPLIT)` | Cube | Producer | C2V |
| `tpush_to_aic(TILE, SPLIT)` | Vector | Producer | V2C |
| `tpop_from_aic(TILE, SPLIT)` | Vector | Consumer | C2V |
| `tpop_from_aiv(TILE, SPLIT)` | Cube | Consumer | V2C |

**Parameters (tpush):**

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| `TILE` | `Tile&` | Source tile to push |
| `SPLIT` | `int` | Split mode (0=none, 1=up-down, 2=left-right) |

**Parameters (tpop):**

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| `TILE` | `Tile&` | Destination tile to receive into |
| `SPLIT` | `int` | Split mode (0=none, 1=up-down, 2=left-right) |

### Push Protocol (`tpush_to_aiv` / `tpush_to_aic`)

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

### Pop Protocol (`tpop_from_aic` / `tpop_from_aiv`)

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

## Flag Assignment

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

## Timing Diagram (C2V, SLOT_NUM=4)

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

## Timing Diagram (Bidirectional, SLOT_NUM=4)

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

## Key Properties

1. **No deadlock** — consumer pre-signals all slots free before the main loop
2. **Backpressure** — producer blocks when all slots full; consumer blocks when empty
3. **FIFO order** — strict round-robin `(tag + 1) % SLOT_NUM`
4. **Decoupled DMA** — async MTE transfer with explicit flag wait
5. **Buddy core selection** — handled by initialization, not per-instruction
6. **Static direction** — direction encoded in opcode, verified at compile time

## API Summary

| API | Core | Role | Description |
| --- | ---- | ---- | ----------- |
| `aic_initialize_pipe(...)` | Cube | Setup | Bind ring buffer, init tags, pre-signal free slots for V2C |
| `aiv_initialize_pipe(...)` | Vector | Setup | Bind ring buffer, init tags, pre-signal free slots for C2V |
| `tpush_to_aiv(TILE, SPLIT)` | Cube | Producer | Wait free → DMA tile → signal ready |
| `tpush_to_aic(TILE, SPLIT)` | Vector | Producer | Wait free → DMA tile → signal ready |
| `tpop_from_aic(TILE, SPLIT)` | Vector | Consumer | Wait ready → load tile → signal free |
| `tpop_from_aiv(TILE, SPLIT)` | Cube | Consumer | Wait ready → load tile → signal free |
