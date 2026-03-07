# Cluster Architecture

## Overview

Each cluster contains **1 Cube core** and **2 buddy Vector cores** that share a hardware flag-based synchronization mechanism. This architecture enables efficient intra-cluster data communication for co-scheduled InCore kernels.

## Cluster Topology

```text
┌─────────────────────── Cluster ───────────────────────┐
│                                                       │
│  ┌──────────┐    flags (8 per dir)     ┌──────────┐   │
│  │ Vector 0 │◄════════════════════════►│          │   │
│  └──────────┘    SET/WAIT V→C, C→V     │   Cube   │   │
│                                        │          │   │
│  ┌──────────┐    flags (8 per dir)     │          │   │
│  │ Vector 1 │◄════════════════════════►│          │   │
│  └──────────┘    SET/WAIT V→C, C→V     └──────────┘   │
│                                                       │
└───────────────────────────────────────────────────────┘
```

## Hardware Flags

Cross-core synchronization uses hardware flags with SET/WAIT semantics:

| Property | Value |
| -------- | ----- |
| Flags per direction per peer | 8 |
| Directions per peer | 2 (Vector→Cube, Cube→Vector) |
| Flags per Vector-Cube pair | 16 |
| Total flags per cluster | 32 (2 peers × 16) |

A Vector core can **SET** a flag that the Cube core **WAITs** on, and vice versa.

## Ring Buffer Data Channel

Data flows between producer and consumer kernels through a **multi-slot ring buffer** with flow control. Each slot holds one fixed-size **Tile**. The ring buffer location is platform-dependent:

| Platform | Location | Description |
| -------- | -------- | ----------- |
| **A2/A3** | Global Memory (GM) | Off-chip DDR/HBM, accessible by all cores |
| **A5** | Consumer's on-chip SRAM | UB (Vector consumer) or L1 (Cube consumer) |

```text
A2/A3 Platform:                           A5 Platform:

Producer          GM         Consumer    Producer                    Consumer
┌──────┐   ┌─────────────┐   ┌──────┐    ┌──────┐                ┌──────────────┐
│      │──▶│ slot[0..N-1]│──▶│      │    │      │───────────────▶│ UB / L1      │
│ Cube │   │ (off-chip)  │   │ Vec  │    │ Cube │  DMA to local  │ slot[0..N-1] │
│ /Vec │   └─────────────┘   │ /Cube│    │ /Vec │                │ (on-chip)    │
└──────┘                     └──────┘    └──────┘                └──────────────┘
```

The A5 placement in consumer-local SRAM eliminates the round-trip to GM, enabling lower-latency data handoff. On A5, the consumer can directly operate on tile data in its local buffer without an explicit load from GM.

## Communication Capabilities

The TPUSH/TPOP mechanism enables:

- **Tile-level data flow** between Cube and buddy Vector cores (producer → ring buffer → consumer)
- **Cross-core synchronization** via hardware SET/WAIT flags
- **Pipelined execution** through multi-slot ring buffers (8 slots unidirectional, 4 slots per direction bidirectional)
- **Platform-adaptive buffer placement** — GM on A2/A3, consumer SRAM on A5

See [TPUSH/TPOP Instructions](01-tpush_tpop.md) for the instruction specification and [Buffer Management](02-buffer_management.md) for platform-specific buffer placement details.
