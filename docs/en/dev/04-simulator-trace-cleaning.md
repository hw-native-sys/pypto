# Simulator Trace Cleaning

The operator simulator emits, per kernel run, an `OPPROF_*/simulator/`
directory containing `trace.json` (a Chrome Trace Event JSON) and
`visualize_data.bin` (a binary container for MindStudio Insight). Opening
`trace.json` directly in the Perfetto UI is hard to read: `SET_FLAG` /
`WAIT_FLAG` synchronization slices and scalar address-arithmetic instructions
bury the actual AI-core pipeline.

`clean_sim_trace` rebuilds the dump into a de-cluttered, Chrome-viewable
pipeline trace.

## Usage

```bash
python -m pypto.tools.clean_sim_trace <path> [-o OUTPUT_DIR] [--keep-scalar] [--raw-metrics]
```

`<path>` is a `visualize_data.bin` file or an `OPPROF_*` directory (the tool
locates `simulator/visualize_data.bin` inside it). Two files are written next
to the input, or to `-o` if given:

| File | Content |
| ---- | ------- |
| `trace.clean.json` | Rebuilt Chrome Trace Event JSON — opens in `chrome://tracing` and the Perfetto UI |
| `instr_metrics.json` | Per-core instruction metrics from the `API_INSTR` block |

| Flag | Effect |
| ---- | ------ |
| `--keep-scalar` | Keep the `SCALAR` setup lane (dropped by default) |
| `--raw-metrics` | Dump the `API_INSTR` block verbatim instead of reshaping it |

## The `visualize_data.bin` format

The file is a flat sequence of length-prefixed blocks. Each block has a
12-byte, 4-byte-aligned header:

| Offset | Size | Field | Meaning |
| ------ | ---- | ----- | ------- |
| 0 | `uint64` LE | `contentSize` | payload length, including trailing padding |
| 8 | `uint8` | `type` | block type (`2` = TRACE, `4` = API_INSTR, `1` = SOURCE, ...) |
| 9 | `uint8` | `paddingLength` | trailing zero bytes for 4-byte alignment |
| 10 | `uint8` | `instrVersion` | API_INSTR version marker |
| 11 | `uint8` | `reserve` | always `0x5a` (binary-format magic) |

Each block payload is plain JSON. `SOURCE` blocks prefix the payload with a
fixed 4096-byte file path. The tool consumes the `TRACE` and `API_INSTR`
blocks; other block types are skipped.

## Rebuild rules

1. **Lane selection** — keep the pipeline lanes (`MTE2`, `MTE1`, `CUBE`,
   `VECTOR`, `FIXPIPE`, `MTE3`); drop `CACHEMISS`, `FLOWCTRL`, `ALL`. The
   `SCALAR` lane is dropped by default (`--keep-scalar` restores it).
2. **Event filtering** — keep `X` (complete) instruction events; drop
   `SET_FLAG` / `WAIT_FLAG` / `BAR` slices.
3. **Lane ordering** — emit `process_*` / `thread_*` metadata so cores and
   pipeline lanes render in dataflow order (load -> compute -> store).
4. **Sync as arrows** — each `SET_FLAG` -> `WAIT_FLAG` pair becomes one flow
   arrow, re-anchored from the producing instruction to the consuming one.
5. **Coloring** — slices are recolored per pipeline lane.
6. **Timestamps** — kept verbatim, so the cleaned trace lines up with the raw
   `trace.json`.

## Metrics sidecar

`instr_metrics.json` reshapes the `API_INSTR` block: each metric, stored in the
raw block as an array indexed by core, is flattened into a per-core list of
instruction records (`address`, `pipe`, `cycles`, `vector_utilization_percentage`,
...). The original `Instructions Dtype` map is preserved under `column_types`.
