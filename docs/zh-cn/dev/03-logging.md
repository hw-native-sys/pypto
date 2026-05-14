# 日志（Logging）

<!-- Copyright (c) PyPTO Contributors. See LICENSE for terms. -->

PyPTO 内部存在 **两套相互独立的日志子系统**，调试时务必先分清当前看到的
日志来自哪一套、由哪个开关控制。

| 子系统 | 来源 | 输出 | 阈值开关 |
| ------ | ---- | ---- | -------- |
| PyPTO C++ 日志 | 编译器核心（`src/`、Pass、Codegen、Diagnostics） | stderr | `pypto.set_log_level()` / `PYPTO_LOG_LEVEL` |
| PyPTO runtime 日志 | 运行时（`runtime/`、simpler 的 Python 与 C++） | 经 Python `logging` 输出到 stderr | `pypto.runtime.configure_log()` / `PYPTO_RUNTIME_LOG` |

两者刻意保持独立：编译期日志的读者是 kernel 作者，运行期日志的读者是
集成方，且分别属于不同进程（host 编译器 vs. simpler worker）。除非显式
传入 `sync_pypto=True`，对其中一个的修改 **不会** 影响另一个。

## 1. PyPTO C++ 日志（编译期）

编译器内部所有 `LOG_INFO` / `LOG_WARN` / `LOG_ERROR` 都走它，包括
Diagnostics（`Warning`、`PerfHint`）；各级别在何处触发请参考
[passes/92-diagnostics.md](passes/92-diagnostics.md)。

### 级别

`LogLevel` 是一组较粗的枚举，定义见
[python/pypto/pypto_core/logging.pyi:18](../../../python/pypto/pypto_core/logging.pyi#L18)：

| 值 | 名称 | 用途 |
| -- | ---- | ---- |
| 0 | `DEBUG` | 详细 Pass 跟踪、IR dump |
| 1 | `INFO` | 性能提示汇总、流水线状态 |
| 2 | `WARN` | 疑似错误的诊断 |
| 3 | `ERROR` | 可恢复的编译错误 |
| 4 | `FATAL` | 不可恢复，即将 abort |
| 5 | `EVENT` | 结构化时间点事件 |
| 6 | `NONE` | 全部静音 |

### 设置阈值

**编程方式**（推荐用于测试与库代码）：

```python
from pypto import LogLevel, set_log_level

set_log_level(LogLevel.WARN)   # 屏蔽 INFO / DEBUG
```

**环境变量** `PYPTO_LOG_LEVEL`（大小写不敏感，取值同上表）。在 C++
日志器初始化时一次性读取：

```bash
export PYPTO_LOG_LEVEL=warn      # release 默认值是 info
python my_program.py
```

优先级：显式 `set_log_level()` > `PYPTO_LOG_LEVEL` > 编译期默认
（release 为 `info`，否则为 `debug`）。

## 2. PyPTO runtime 日志（运行期）

驱动名为 `"simpler"` 的 Python logger，并在 `Worker.init()` 中以一次性
快照的方式同步到 simpler 的 C++ 运行时。启动 kernel、等待 task、销毁
worker 阶段的所有日志都经此输出。

用户入口位于
[python/pypto/runtime/log_config.py:38](../../../python/pypto/runtime/log_config.py#L38)：

```python
from pypto.runtime import configure_log, log_level

configure_log("v7")                # 比 INFO 细，比 DEBUG 粗
print(log_level())                 # → 22  (Python logging 整数)
```

### 级别

runtime 日志使用比 PyPTO C++ 更细的级别表，定义在
[runtime/simpler_setup/log_config.py](../../../runtime/simpler_setup/log_config.py)，
`pypto.runtime.configure_log()` 直接复用其 `parse_level`：

| 名称 | Python `logging` 整数 | 说明 |
| ---- | --------------------- | ---- |
| `debug` | 10 | 全量详细 |
| `v0` .. `v9` | 15..24 | INFO 子档；`v5` == `info` (20) |
| `info` | 20 | runtime 默认 |
| `warn` / `warning` | 30 | |
| `error` | 40 | |
| `null` | 60 | 全部静音 |

`v0..v9` 是与 PyPTO C++ 日志的最大差异：在 INFO 内部细分 10 档，可在
不切到 `debug` 的前提下单独放大某些噪声较多的子系统。

### `configure_log(level, *, sync_pypto=False)`

| 参数 | 类型 | 作用 |
| ---- | ---- | ---- |
| `level` | `int` 或 `str` | Python logger 整数（如 `20`）或上表名称，大小写不敏感 |
| `sync_pypto` | `bool`，默认 `False` | 为 `True` 时同步把对应 band 的 `LogLevel` 推送给 PyPTO C++ 日志器，便于一个开关同时控制两套子系统 |

`sync_pypto=True` 使用的 band 映射见
[log_config.py:63-81](../../../python/pypto/runtime/log_config.py#L63-L81)：

| runtime 阈值 | PyPTO `LogLevel` |
| ------------ | ---------------- |
| ≤14 | `DEBUG` |
| 15..24 (`v0..v9`) | `INFO` |
| 25..39 (`warn`) | `WARN` |
| 40..59 (`error`) | `ERROR` |
| ≥60 (`null`) | `NONE` |

可用 `pypto.runtime.log_level()`（即 `current_level()` 的别名）回读
当前生效阈值。

### 环境变量

`pypto.runtime` 在导入时会执行一次幂等的 `_ensure_configured`
bootstrap，因此可以不写 Python，仅靠 shell 环境变量驱动：

| 环境变量 | 作用 |
| -------- | ---- |
| `PYPTO_RUNTIME_LOG` | 接受与 `configure_log(level=...)` 相同的字符串；未设置则保持 runtime 日志的 `v5` 默认 |
| `PYPTO_RUNTIME_LOG_SYNC` | 设为 `=1` 时把 bootstrap 阶段 `sync_pypto` 的默认值翻成 `True`；当 `PYPTO_RUNTIME_LOG` 未设置时被忽略 |

```bash
# runtime 详细日志，不影响 PyPTO C++ 日志
PYPTO_RUNTIME_LOG=v7 python -m my_test

# 一个开关同时管两套
PYPTO_RUNTIME_LOG=debug PYPTO_RUNTIME_LOG_SYNC=1 python -m my_test
```

导入后再显式调用 `configure_log(...)` 会覆盖 bootstrap 的选择。

## 3. pytest 选项（`tests/st/`）

集成测试 harness 把两套子系统都暴露成命令行参数，定义见
[tests/st/conftest.py:157-170](../../../tests/st/conftest.py#L157-L170)，
并在 `pytest_configure` 中提前应用，因此 collection 阶段的日志也会
受影响。

| 选项 | 默认值 | 作用 |
| ---- | ------ | ---- |
| `--pypto-log-level` | `ERROR` | 通过 `set_log_level(LogLevel[name])` 控制 PyPTO C++ 日志 |
| `--runtime-log-level` | 未设置（保留 `v5`） | 通过 `configure_log(level)` 控制 PyPTO runtime 日志；**不会** 同时传 `sync_pypto=True` |

```bash
# 抑制编译噪声，放大 runtime 日志
pytest tests/st/ --pypto-log-level=ERROR --runtime-log-level=v8

# 两侧都开 debug
pytest tests/st/ --pypto-log-level=DEBUG --runtime-log-level=debug
```

## 4. 决策表

| 场景 | 用法 |
| ---- | ---- |
| 长编译过程屏蔽 warning | `set_log_level(LogLevel.ERROR)` 或 `PYPTO_LOG_LEVEL=error` |
| 查看 Pass 级跟踪 | `set_log_level(LogLevel.DEBUG)` |
| 在 stderr 看到性能提示 | 保持 PyPTO 默认 `INFO`（或 `PYPTO_LOG_LEVEL=info`），详见 [passes/92-diagnostics.md](passes/92-diagnostics.md) |
| 排查执行期 hang | `configure_log("debug")` 或 `PYPTO_RUNTIME_LOG=debug` |
| 仅放大 runtime 的某个噪声子系统 | `configure_log("v7")`（V0..V9 比 `info`/`debug` 粒度更细） |
| 一条环境变量全部静音 | `PYPTO_RUNTIME_LOG=null PYPTO_RUNTIME_LOG_SYNC=1 PYPTO_LOG_LEVEL=none` |

## 5. 常见坑

- **`PYPTO_LOG_LEVEL` 不会影响 runtime 日志**，反之
  `PYPTO_RUNTIME_LOG` 也不会影响编译器日志。需要一个开关同时管两侧时，
  请使用 `sync_pypto=True` 或两个环境变量都设。
- **`configure_log()` 内部惰性导入 simpler。** 在没有安装 simpler 的
  纯 codegen 环境里不要调用它；`import pypto.runtime` 本身仍然安全，因为
  当 `PYPTO_RUNTIME_LOG` 未设置时 bootstrap 会直接短路。
- **`tests/st/` 里的 `--runtime-log-level` 不会自动同步 PyPTO。** 想让
  两套子系统对齐，请显式同时设 `--pypto-log-level`。
- **runtime C++ 端只在 `Worker.init()` 时读一次阈值。** 在 worker 起来
  之后再调 `configure_log()` 仅改 Python 侧，下一次 worker init 才会
  把新阈值带到 C++。

## 参考

- [passes/92-diagnostics.md](passes/92-diagnostics.md) —— 各诊断阶段在
  INFO / WARN 上的输出，以及 perf-hint 文件
- [python/pypto/runtime/log_config.py](../../../python/pypto/runtime/log_config.py) —— 实现
- [runtime/simpler_setup/log_config.py](../../../runtime/simpler_setup/log_config.py) —— simpler 级别表
