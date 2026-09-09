# Developer Documentation

How PyPTO is built: the IR, the pass pipeline, code generation, and the
infrastructure around them.

This is documentation for people working *on* the compiler. If you are writing
PyPTO programs, start with the [User Manual](../user/index.md).

## Sub-chapters

| Chapter | What it covers |
| ------- | -------------- |
| [IR](ir/index.md) | Node hierarchy, type system, operators, builder, parser, serialization, structural comparison |
| [Passes](passes/index.md) | The pass framework and every pass in the default pipeline, numbered in execution order |
| [Language](language/index.md) | The Python DSL syntax specification and external C++ kernel integration |
| [Code Generation](codegen/index.md) | Lowering IR to PTO-ISA dialect MLIR and to orchestration C++ |
| [Backend](backend/index.md) | Per-architecture dispatch through `BackendHandler` |
| [Debug](debug/index.md) | Lowering IR to an executable PyTorch script for numerical validation |

## Top-level topics

| Page | What it covers |
| ---- | -------------- |
| [PTO Project Ecosystem](00-ecosystem.md) | The multi-repo toolchain — PyPTO, PTOAS, pto-isa, simpler, pypto-lib — and how they fit together |
| [Compile Profiling](01-compile-profiling.md) | Built-in wall-clock timing of the compilation pipeline |
| [Error Handling](02-error-handling.md) | `CHECK` vs `INTERNAL_CHECK`, PyPTO exception types, IR source locations in failures |
| [Logging](03-logging.md) | The two independent logging subsystems and which one a message came from |
| [Runtime DFX Flags](03-runtime-dfx.md) | The five runtime diagnostic sub-features exposed through `RunConfig` |
| [Replaying an Existing `build_output`](03-runtime-replay.md) | Re-run, edit, and re-measure a compiled build directory without recompiling |
| [Simulator Trace Cleaning](04-simulator-trace-cleaning.md) | Converting MindStudio Insight binary dumps into readable traces |
| [Per-Task Ring Sizing](05-runtime-ring-sizing.md) | The three ring-size overrides on `RunConfig` and when to tune them |
| [Persistent L3 execution](06-persistent-l3.md) | Reusing one worker across prepared distributed programs |
| [Memory Map](07-memory-map.md) | Rendering a pass dump into an interactive HTML map of on-chip memory |
| [Compile and Execution Entry Points](08-entry-points.md) | Every compile and execution entry point, the layer it belongs to, and when to reach for it |
| [Distributed Operators](distributed_ops.md) | The N6 distributed op family — typed DSL access to collectives and primitives |
| [PTOAS Op Status Matrix](ptoas-op-status.md) | Which public and compatibility PTOAS ops the compiler currently emits |

## See Also

- [PTO ISA reference](../reference/index.md) — the hardware model the backend targets.
- [Runtime documentation](https://www.pypto.ai/simpler/) — the scheduler that executes compiled programs.

## Documentation site

The site at <https://www.pypto.ai/pypto/> inherits its layout, colors, project
switcher, and theme preferences from
[`hw-native-sys.github.io`](https://github.com/hw-native-sys/hw-native-sys.github.io).
This repository owns its pages, navigation, translations, and API plugins.
`docs/theme-revision.txt` pins the shared theme to a
full Git commit SHA. CI and local builds use that same revision; the generated
site includes its own copy of the theme assets.

### Preview locally

Run from the PyPTO repository root with Python 3.10 or newer. Fetch the theme
before installing the documentation dependencies:

```bash
set -euo pipefail
git init .site-theme
git -C .site-theme fetch --depth 1 \
  https://github.com/hw-native-sys/hw-native-sys.github.io.git \
  "$(cat docs/theme-revision.txt)"
git -C .site-theme checkout --detach FETCH_HEAD

python -m venv .venv
source .venv/bin/activate
python -m pip install -r docs/requirements.txt
python tests/lint/check_docs_nav.py
python tests/lint/check_docs_en_zh_parity.py
mkdocs build --strict
mkdocs serve
```

The preview runs at <http://127.0.0.1:8000/>. `.site-theme/` and the virtual
environment are ignored by Git. The documentation toolchain parses API source
statically, so it does not require a PyPTO build or an Ascend device.

### Update or roll back the theme

Replace the SHA in `docs/theme-revision.txt` with the desired commit from the
shared repository, then rerun the fetch, checkout, dependency installation, and
checks above. Review the English and Chinese pages, API reference, narrow-screen
navigation, and light/dark modes before submitting the revision change.

`mkdocs.yml` inherits `.site-theme/docs-theme/base.yml`; local values override
shared values, and lists replace inherited lists. Keep project navigation and
plugins here, and make common appearance changes in the shared repository.
Updating the shared repository alone does not change this deployed site: merge
the revision update here to publish it. To roll back, restore the previous SHA
and rebuild through the Docs workflow.
