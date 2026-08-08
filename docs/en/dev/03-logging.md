# Logging

<!-- Copyright (c) PyPTO Contributors. See LICENSE for terms. -->

PyPTO ships **two independent logging subsystems**. Knowing which one a
message comes from — and which knob controls it — is essential for
day-to-day debugging.

| Subsystem | Source | Sink | Threshold knob |
| --------- | ------ | ---- | -------------- |
| PyPTO C++ logger | Compiler core (`src/`, passes, codegen, diagnostics) | stderr | `pypto.set_log_level()` / `pypto.get_log_level()` / `PYPTO_LOG_LEVEL` |
| PyPTO runtime logger | On-device runtime (`runtime/`, simpler Python + C++) | stderr via Python `logging` | `pypto.runtime.configure_log()` / `PYPTO_RUNTIME_LOG` |

They are deliberately separate: the compile-time logger and the run-time
logger have different audiences (kernel author vs. integrator) and run in
different processes (host compiler vs. simpler worker). A change to one
does **not** silence the other unless you opt in via `sync_pypto=True`.

## 1. PyPTO C++ logger (compile-time)

Used by every `LOG_INFO` / `LOG_WARN` / `LOG_ERROR` call inside the
compiler, including diagnostics (`Warning`, `PerfHint`) — see
[passes/92-diagnostics.md](passes/92-diagnostics.md) for what fires on
each level.

### Levels

`LogLevel` is a coarse enum exposed in
[python/pypto/pypto_core/logging.pyi](../../../python/pypto/pypto_core/logging.pyi):

| Value | Name | Use |
| ----- | ---- | --- |
| 0 | `DEBUG` | Verbose pass tracing, IR dumps |
| 1 | `INFO` | Per-hint perf summaries, pipeline status |
| 2 | `WARN` | Likely-mistake diagnostics |
| 3 | `ERROR` | Recoverable compile errors |
| 4 | `FATAL` | Unrecoverable, abort imminent |
| 5 | `EVENT` | Structured timing events |
| 6 | `NONE` | Silence everything |

### Setting the threshold

**Programmatic** (preferred in tests and library code):

```python
from pypto import LogLevel, get_log_level, set_log_level

set_log_level(LogLevel.WARN)   # mute INFO/DEBUG
```

The threshold is **process-global**, so a library or test that lowers it
must put it back. `get_log_level()` reads the global back for exactly that
save-restore pair:

```python
saved = get_log_level()
set_log_level(LogLevel.ERROR)
try:
    compile_something_noisy()
finally:
    set_log_level(saved)
```

Leaving it lowered does not just mute chatter — it silences `Warning` and
`PerfHint` diagnostics for everything that runs afterwards in the same
process, which reads downstream as "the compiler had nothing to say".
Unit tests get this for free: `tests/ut/conftest.py::_reset_log_level`
pins the level around every test.

**Environment variable** (`PYPTO_LOG_LEVEL`) — case-insensitive, accepts
the names above. Read once at C++ logger init:

```bash
export PYPTO_LOG_LEVEL=warn      # release default: info
python my_program.py
```

Override order: explicit `set_log_level()` wins over `PYPTO_LOG_LEVEL`,
which wins over the build-time default (`info` in release, `debug`
otherwise).

## 2. PyPTO runtime logger (run-time)

Drives the Python logger named `"simpler"` *and* (via a one-shot snapshot
inside `Worker.init()`) simpler's C++ runtime. Everything emitted while
launching kernels, waiting on tasks, or tearing down the worker flows
through here.

The user-facing entry point lives in
[python/pypto/runtime/log_config.py](../../../python/pypto/runtime/log_config.py):

```python
from pypto.runtime import configure_log, log_level

configure_log("timing")            # stable performance markers, but no INFO chatter
print(log_level())                 # → 25  (Python logging int)
```

### Levels

The runtime logger has a dedicated timing tier that PyPTO's C++ enum does not.
The canonical table lives in
[runtime/simpler_setup/log_config.py](../../../runtime/simpler_setup/log_config.py)
and `pypto.runtime.configure_log()` delegates parsing there:

| Name(s) | Python `logging` int | Notes |
| ------- | -------------------- | ----- |
| `debug` | 10 | full verbosity |
| `info` | 20 | lifecycle and summary messages |
| `timing` | 25 | runtime default; stable performance markers such as `[STRACE]` |
| `warn` / `warning` | 30 | |
| `error` | 40 | |
| `null` | 60 | silence everything |

At the default `timing` threshold, timing markers, warnings, and errors are
visible while ordinary INFO and DEBUG traffic stays silent.

### `configure_log(level, *, sync_pypto=False)`

| Argument | Type | Effect |
| -------- | ---- | ------ |
| `level` | `int` or `str` | Python logger int (e.g. `20`) or any name from the table above. Case-insensitive. |
| `sync_pypto` | `bool` (default `False`) | When `True`, also push the closest `LogLevel` band onto PyPTO's C++ logger — useful when you want a single knob to cover both subsystems. |

The band mapping used by `sync_pypto=True`
([log_config.py](../../../python/pypto/runtime/log_config.py)):

| runtime threshold | PyPTO `LogLevel` |
| ----------------- | ---------------- |
| ≤10 (`debug`) | `DEBUG` |
| 11..20 (`info`) | `INFO` |
| 21..30 (`timing` / `warn`) | `WARN` |
| 31..40 (`error`) | `ERROR` |
| ≥41 (`null`) | `NONE` |

`timing` maps to `WARN` because PyPTO has no timing-only level; mapping it to
`INFO` would also enable ordinary compiler information messages.

Read back the effective threshold with
`pypto.runtime.log_level()` (re-exported from `current_level()`).

### Environment variables

`pypto.runtime` runs an idempotent bootstrap (`_ensure_configured`) once
at import time, so you can drive logging from the shell without touching
Python:

| Env var | Effect |
| ------- | ------ |
| `PYPTO_RUNTIME_LOG` | Same string accepted by `configure_log(level=...)`. Unset = keep the runtime logger at its `timing` default. |
| `PYPTO_RUNTIME_LOG_SYNC` | When `=1`, flips the default of `sync_pypto` to `True` for the env-var bootstrap. Ignored when `PYPTO_RUNTIME_LOG` is unset. |

```bash
# Runtime lifecycle logs, leave PyPTO C++ untouched
PYPTO_RUNTIME_LOG=info python -m my_test

# One knob for both subsystems
PYPTO_RUNTIME_LOG=debug PYPTO_RUNTIME_LOG_SYNC=1 python -m my_test
```

An explicit `configure_log(...)` call after import overrides whatever the
env bootstrap chose.

## 3. pytest options (`tests/st/`)

The integration-test harness exposes both subsystems as CLI options
([tests/st/conftest.py](../../../tests/st/conftest.py)).

| Option | Default | Drives |
| ------ | ------- | ------ |
| `--pypto-log-level` | `ERROR` | PyPTO C++ logger, as a **thread-local** override applied per ST item |
| `--runtime-log-level` | unset (keeps `timing`) | PyPTO runtime logger via `configure_log(level)` — note this **does not** pass `sync_pypto=True` |

`--pypto-log-level` deliberately does *not* call `set_log_level()`. That
threshold is process-global, and `tests/st/conftest.py` is loaded in mixed
ST/UT sessions too, so setting it in `pytest_configure` muted the
diagnostics that later unit tests assert on. It is instead installed via
`_set_thread_log_level` in a `pytest_runtest_protocol` wrapper around each
ST item and cleared in a `finally` — forked runtime workers still inherit
it, and nothing survives the item. Compile pools take the same level
through their `ThreadPoolExecutor(initializer=...)`.

```bash
# Quiet PyPTO compile chatter, verbose runtime logs
pytest tests/st/ --pypto-log-level=ERROR --runtime-log-level=info

# Debug both
pytest tests/st/ --pypto-log-level=DEBUG --runtime-log-level=debug
```

## 4. Decision guide

| You want to … | Use |
| ------------- | --- |
| Mute compiler warnings during a long compile | `set_log_level(LogLevel.ERROR)` or `PYPTO_LOG_LEVEL=error` |
| See pass-by-pass tracing | `set_log_level(LogLevel.DEBUG)` |
| Read perf hints on stderr | leave PyPTO at default `INFO` (or `PYPTO_LOG_LEVEL=info`) — see [passes/92-diagnostics.md](passes/92-diagnostics.md) |
| Trace a hang at execute time | `configure_log("debug")` or `PYPTO_RUNTIME_LOG=debug` |
| Show runtime lifecycle messages without DEBUG traffic | `configure_log("info")` or `PYPTO_RUNTIME_LOG=info` |
| One env var to silence everything | `PYPTO_RUNTIME_LOG=null PYPTO_RUNTIME_LOG_SYNC=1 PYPTO_LOG_LEVEL=none` |

## 5. Common pitfalls

- **`PYPTO_LOG_LEVEL` does not affect runtime logs**, and
  `PYPTO_RUNTIME_LOG` does not affect compiler logs. Use `sync_pypto=True`
  (or set both env vars) if you want one knob.
- **`configure_log()` re-imports simpler lazily.** In environments where
  simpler is not installed (codegen-only flows), avoid calling it — the
  `pypto.runtime` import itself is safe because the env-var bootstrap
  short-circuits when `PYPTO_RUNTIME_LOG` is unset.
- **`--runtime-log-level` in `tests/st/` does not sync PyPTO.** Pair it
  with `--pypto-log-level` if you want both subsystems aligned.
- **The runtime C++ side reads its threshold once** at `Worker.init()`.
  Calling `configure_log()` *after* the worker is up changes only the
  Python side until the next worker init.

## See also

- [passes/92-diagnostics.md](passes/92-diagnostics.md) — what each
  diagnostic phase emits at INFO / WARN, and the perf-hint file output
- [python/pypto/runtime/log_config.py](../../../python/pypto/runtime/log_config.py) — the canonical implementation
- [runtime/simpler_setup/log_config.py](../../../runtime/simpler_setup/log_config.py) — simpler's level table
