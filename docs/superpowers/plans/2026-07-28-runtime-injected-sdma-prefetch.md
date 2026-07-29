# Runtime-Injected SDMA Prefetch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make PR #2089 use simpler's runtime-injected SDMA workspace instead of a nonexistent host workspace getter.

**Architecture:** `pl.prefetch.make_context()` becomes parameterless. PTOCodegen adds a hidden workspace-pointer parameter to prefetch kernels, and the generated wrapper fills it with `get_dma_workspace(args, DMA_WORKSPACE_SDMA)`. Generated runtime metadata propagates `enable_sdma=True` through L2 and L3 worker creation while preserving the default for ordinary programs.

**Tech Stack:** C++17 PyPTO IR/PTOCodegen, Python 3.10+ DSL/backend/runtime, PTOAS, simpler runtime, pytest, CMake.

## Global Constraints

- Do not update the `runtime` submodule; pinned runtime commit `9922afdb` already contains simpler PR #1406.
- Do not expose a host API that returns a runtime-owned DMA workspace address.
- Keep non-prefetch workers at `enable_sdma=False` so they allocate no SDMA streams.
- Keep user, orchestration, and runtime tensor signatures free of the hidden workspace.
- Runtime initialization must fail on platforms without an SDMA provider; do not allocate or simulate a fallback workspace.
- English documentation under `docs/en/dev/` is authoritative; update the matching Chinese document in the same change.
- Run onboard a2a3 work only through `task-submit`, after `runtime/.claude/skills/onboard-arch-precheck/check.sh a2a3`.

---

### Task 1: Make the prefetch context API parameterless

**Files:**
- Modify: `tests/ut/ir/operators/test_prefetch_ops.py`
- Modify: `python/pypto/ir/op/prefetch_ops.py`
- Modify: `python/pypto/language/op/prefetch_ops.py`
- Modify: `src/ir/op/prefetch/prefetch_async.cpp`
- Modify: `include/pypto/ir/type.h`

**Interfaces:**
- Produces: `pypto.ir.op.prefetch_ops.make_context(span: Span | None = None) -> Call`
- Produces: `pypto.language.prefetch.make_context() -> PrefetchAsyncContext`
- Consumes: existing `PrefetchAsyncContextType` and `prefetch.make_context` registry name

- [ ] **Step 1: Write the failing zero-argument API tests**

Change every valid program in `test_prefetch_ops.py` from:

```python
ctx = pl.prefetch.make_context(ws)
```

to:

```python
ctx = pl.prefetch.make_context()
```

Remove workspace parameters that are no longer otherwise used. Replace the INT8-workspace verifier tests with an old-API rejection test:

```python
def test_make_context_rejects_workspace_argument(self):
    with pytest.raises(TypeError, match="make_context.*positional"):
        pl.prefetch.make_context(object())
```

Keep the existing source-shape and handle-type verifier tests so only workspace ownership changes.

- [ ] **Step 2: Run the focused IR tests and verify RED**

Run:

```bash
PYTHONPATH=$PWD/python:/data/linyifan/.conda/envs/lyf/lib/python3.10/site-packages \
python -S -m pytest tests/ut/ir/operators/test_prefetch_ops.py -q
```

Expected: valid programs fail because `make_context` still requires `workspace`.

- [ ] **Step 3: Implement the zero-argument API**

Make the low-level wrapper create an argument-free op:

```python
def make_context(span: Span | None = None) -> Call:
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("prefetch.make_context", [], {}, actual_span)
```

Make the language wrapper call it without unwrapping a tensor:

```python
def make_context() -> PrefetchAsyncContext:
    return PrefetchAsyncContext(expr=_ir_prefetch.make_context())
```

Change the C++ registry verifier to `CheckArity(..., "no arguments", 0, ...)`, return the singleton handle type directly, and remove workspace dtype/type checks. Update comments and the type docstring to describe a runtime-injected workspace.

- [ ] **Step 4: Rebuild and verify GREEN**

Run:

```bash
cmake --build build --parallel
PYTHONPATH=$PWD/python:/data/linyifan/.conda/envs/lyf/lib/python3.10/site-packages \
python -S -m pytest tests/ut/ir/operators/test_prefetch_ops.py -q
```

- [ ] **Step 5: Commit the API change**

```bash
git add tests/ut/ir/operators/test_prefetch_ops.py \
  python/pypto/ir/op/prefetch_ops.py \
  python/pypto/language/op/prefetch_ops.py \
  src/ir/op/prefetch/prefetch_async.cpp include/pypto/ir/type.h
git commit -m "fix(ir): Use runtime-owned prefetch context"
```

### Task 2: Inject the hidden SDMA pointer through PTOCodegen and the kernel wrapper

**Files:**
- Modify: `tests/ut/codegen/test_prefetch_codegen.py`
- Modify: `include/pypto/codegen/pto/pto_codegen.h`
- Modify: `src/codegen/pto/pto_codegen.cpp`
- Modify: `src/backend/common/pto_ops_prefetch.cpp`
- Modify: `python/pypto/backend/pto_backend.py`

**Interfaces:**
- Produces: `PTOCodegen::GetSdmaWorkspaceArgSSA() const -> std::string`
- Produces: `_uses_sdma_workspace(func: Function) -> bool`
- Produces: `_generate_config_file(..., enable_sdma: bool = False) -> str`
- Consumes: runtime `get_dma_workspace(args, DMA_WORKSPACE_SDMA)` from `intrinsic.h`

- [ ] **Step 1: Write failing codegen and manifest tests**

Change `PrefetchProgram` and `MixedPrefetchProgram` to use no workspace parameter. Assert the PTO function contains a hidden pointer and still has only the original user-visible tensors in generated runtime signatures:

```python
assert re.search(r"func\.func @main\([^)]*!pto\.ptr<i8>[^)]*\)", mlir)
assert re.search(
    r"pto\.make_prefetch_async_context\(%\w+ : !pto\.ptr<i8>\)", mlir
)
```

Generate the full backend artifact in a temporary directory and assert:

```python
wrapper = result["kernels/aiv/main.cpp"]
assert "get_dma_workspace(args, DMA_WORKSPACE_SDMA)" in wrapper
assert "main(x, out, __pypto_sdma_workspace" in wrapper
assert '"enable_sdma": True' in result["kernel_config.py"]
assert "workspace" not in result["orchestration/main.cpp"]
```

Add a non-prefetch program assertion that its wrapper has no DMA intrinsic and its config does not enable SDMA.

- [ ] **Step 2: Run codegen tests and verify RED**

Run:

```bash
PYTHONPATH=$PWD/python:/data/linyifan/.conda/envs/lyf/lib/python3.10/site-packages \
python -S -m pytest tests/ut/codegen/test_prefetch_codegen.py -q
```

Expected: `prefetch.make_context` still expects a workspace SSA, and wrappers/configs contain no runtime injection.

- [ ] **Step 3: Add the synthetic PTO parameter**

Extend the existing per-function pre-scan in `pto_codegen.cpp` to detect `prefetch.make_context`. Store a function-state field such as:

```cpp
std::string sdma_workspace_arg_ssa;
```

After user tensor/scalar parameters and before SPMD synthetic `i32` parameters, append:

```text
%argN: !pto.ptr<i8>
```

Expose the bound name through `GetSdmaWorkspaceArgSSA()`. Update the prefetch emitter to require zero IR operands and emit `pto.make_prefetch_async_context` from that synthetic SSA. Raise `INTERNAL_CHECK_SPAN` if the pre-scan and emitter disagree.

- [ ] **Step 4: Add wrapper injection and manifest metadata**

Add `_uses_sdma_workspace(func)` using `_function_uses_ops` and include `intrinsic.h` whenever it returns true. In `_generate_kernel_wrapper`, place the hidden pointer after user arguments and before SPMD values:

```python
sdma_setup = (
    "    __gm__ int8_t* __pypto_sdma_workspace = "
    "reinterpret_cast<__gm__ int8_t*>("
    "get_dma_workspace(args, DMA_WORKSPACE_SDMA));\n\n"
)
call_args_list.append("__pypto_sdma_workspace")
```

Pass `enable_sdma=any(_uses_sdma_workspace(func) for func in emitted_incore_funcs)` to `_generate_config_file`, which writes `"enable_sdma": True` into `RUNTIME_CONFIG` only when required.

- [ ] **Step 5: Rebuild and verify GREEN plus ordering**

Run:

```bash
cmake --build build --parallel
PYTHONPATH=$PWD/python:/data/linyifan/.conda/envs/lyf/lib/python3.10/site-packages \
python -S -m pytest tests/ut/codegen/test_prefetch_codegen.py \
  tests/ut/codegen/test_pto_codegen.py -q
```

- [ ] **Step 6: Commit codegen injection**

```bash
git add tests/ut/codegen/test_prefetch_codegen.py \
  include/pypto/codegen/pto/pto_codegen.h \
  src/codegen/pto/pto_codegen.cpp src/backend/common/pto_ops_prefetch.cpp \
  python/pypto/backend/pto_backend.py
git commit -m "fix(codegen): Inject runtime SDMA workspace"
```

### Task 3: Propagate the SDMA capability through L2 execution

**Files:**
- Modify: `tests/ut/runtime/test_worker_reuse.py`
- Modify: `tests/ut/runtime/test_run_config.py`
- Modify: `python/pypto/runtime/worker.py`
- Modify: `python/pypto/runtime/device_runner.py`
- Modify: `python/pypto/runtime/runner.py`
- Modify: `python/pypto/runtime/_dep_gen_capture.py`
- Modify: `python/pypto/runtime/execute_artifact.py`
- Modify: `tests/st/harness/core/test_runner.py`

**Interfaces:**
- Produces: `ChipWorker(..., enable_sdma: bool = False)`
- Produces: `execute_on_device(..., enable_sdma: bool = False) -> None`
- Produces: `ChipWorker.current(..., require_sdma: bool = False) -> ChipWorker | None`
- Consumes: generated `runtime_config.get("enable_sdma", False)`

- [ ] **Step 1: Write failing worker-construction tests**

Use the existing fake simpler worker fixtures to assert:

```python
ChipWorker(config=RunConfig(platform="a2a3"), enable_sdma=True)
fake_worker_cls.assert_called_once_with(
    level=2,
    device_id=0,
    platform="a2a3",
    runtime="tensormap_and_ringbuffer",
    enable_sdma=True,
)
```

Add one-shot assertions that `execute_on_device(..., enable_sdma=True)` constructs a simpler worker with the flag and that the default path passes false or omits the key consistently.

Add a reuse test where an active ordinary `ChipWorker` receives an SDMA-required dispatch and raises:

```text
active ChipWorker was created without enable_sdma=True
```

Also assert an SDMA-enabled active worker can run an ordinary dispatch.

- [ ] **Step 2: Run runtime tests and verify RED**

Run:

```bash
PYTHONPATH=$PWD/python:/data/linyifan/.conda/envs/lyf/lib/python3.10/site-packages \
python -S -m pytest tests/ut/runtime/test_worker_reuse.py \
  tests/ut/runtime/test_run_config.py -q
```

Expected: constructors and execution helpers reject the unknown `enable_sdma` keyword or fail the new assertions.

- [ ] **Step 3: Implement explicit worker capability and reuse checks**

Store `self._enable_sdma = bool(enable_sdma)` and pass it to simpler's `Worker` constructor. Extend `ChipWorker.current` with `require_sdma`; among workers matching level/platform/device/runtime, return the newest worker when it is capable, otherwise raise the targeted error instead of returning `None` and opening a second device worker.

- [ ] **Step 4: Thread artifact metadata through all L2 entry points**

Add `enable_sdma` to `device_runner.execute_on_device` and pass it both to `ChipWorker.current(require_sdma=...)` and one-shot simpler `Worker(...)`. At every `compile_and_assemble` caller, read:

```python
enable_sdma = bool(runtime_config.get("enable_sdma", False))
```

and forward it through `runner._execute_on_device`, normal compiled execution, dependency-capture subprocesses, artifact replay, and the ST harness artifact record. Preserve default `False` for hand-built callables with no manifest.

- [ ] **Step 5: Verify GREEN and default compatibility**

Run:

```bash
PYTHONPATH=$PWD/python:/data/linyifan/.conda/envs/lyf/lib/python3.10/site-packages \
python -S -m pytest tests/ut/runtime/test_worker_reuse.py \
  tests/ut/runtime/test_run_config.py \
  tests/ut/runtime/test_execute_artifact.py \
  tests/ut/runtime/test_task_submit_dispatch.py -q
```

- [ ] **Step 6: Commit L2 propagation**

```bash
git add python/pypto/runtime/worker.py python/pypto/runtime/device_runner.py \
  python/pypto/runtime/runner.py python/pypto/runtime/_dep_gen_capture.py \
  python/pypto/runtime/execute_artifact.py tests/st/harness/core/test_runner.py \
  tests/ut/runtime/test_worker_reuse.py tests/ut/runtime/test_run_config.py \
  tests/ut/runtime/test_execute_artifact.py tests/ut/runtime/test_task_submit_dispatch.py
git commit -m "fix(runtime): Enable SDMA for prefetch artifacts"
```

### Task 4: Propagate the SDMA capability through distributed execution

**Files:**
- Modify: `tests/ut/runtime/test_distributed_worker.py`
- Modify: `python/pypto/runtime/distributed_runner.py`

**Interfaces:**
- Produces: `_assemble_chip_callables(...) -> tuple[dict[str, Any], str, bool]`
- Produces: `_construct_worker(..., enable_sdma: bool = False) -> Any`
- Consumes: every chip artifact's `RUNTIME_CONFIG["enable_sdma"]`

- [ ] **Step 1: Write failing L3 aggregation tests**

Extend the stubbed `compile_and_assemble` results to include runtime configs. Assert `_assemble_chip_callables` ORs the capability across chip artifacts:

```python
chip_callables, runtime_name, enable_sdma = _assemble_chip_callables(compiled)
assert enable_sdma is True
```

Assert one-shot and reusable distributed workers pass `enable_sdma=True` to `_construct_worker`, and multi-program preparation enables it when any program requires it.

- [ ] **Step 2: Run distributed tests and verify RED**

Run:

```bash
PYTHONPATH=$PWD/python:/data/linyifan/.conda/envs/lyf/lib/python3.10/site-packages \
python -S -m pytest tests/ut/runtime/test_distributed_worker.py -q
```

- [ ] **Step 3: Implement L3 aggregation**

Have `_assemble_chip_callables` accumulate:

```python
enable_sdma = enable_sdma or bool(chip_runtime_config.get("enable_sdma", False))
```

Add the flag to `_construct_worker`, one-shot execution, and reusable/multi-program preparation. For a multi-program worker, OR all prepared programs before constructing the one shared simpler L3 worker.

- [ ] **Step 4: Verify GREEN**

Run the full distributed-worker unit file again and confirm all tests pass.

- [ ] **Step 5: Commit L3 propagation**

```bash
git add python/pypto/runtime/distributed_runner.py \
  tests/ut/runtime/test_distributed_worker.py
git commit -m "fix(runtime): Propagate SDMA to distributed workers"
```

### Task 5: Replace the ST and synchronize user documentation

**Files:**
- Modify: `tests/st/runtime/ops/test_prefetch_async.py`
- Modify: `docs/en/dev/ir/05-operators.md`
- Modify: `docs/zh-cn/dev/ir/05-operators.md`
- Modify: `docs/zh-cn/dev/ptoas-op-status.md`
- Modify: prefetch docstrings in `python/pypto/ir/op/prefetch_ops.py`, `python/pypto/language/op/prefetch_ops.py`, and `src/ir/op/prefetch/prefetch_async.cpp`

**Interfaces:**
- Consumes: parameterless `pl.prefetch.make_context()`
- Consumes: artifact-driven one-shot `enable_sdma=True`
- Produces: onboard a2a3 test with only `(a, out)` user tensors

- [ ] **Step 1: Rewrite the system test before implementation is complete**

Remove `WORKSPACE_BYTES`, the workspace kernel/orchestration parameter, `ChipWorker`, and `DeviceTensor`. The kernel begins with:

```python
ctx = pl.prefetch.make_context()
```

The test executes normally so generated metadata selects the SDMA worker:

```python
compiled(a, out, config=test_config)
assert torch.equal(out, a)
```

Mark/guard the test for onboard a2a3 using the repository's existing platform marker conventions; do not skip after worker initialization.

- [ ] **Step 2: Run the ST locally in codegen/sim mode and verify the expected boundary**

Run its collection and codegen path without claiming SDMA execution on sim. Expected: collection/codegen succeeds, while an actual sim execution fails at runtime initialization because sim has no SDMA provider.

- [ ] **Step 3: Update English and Chinese docs**

Replace every `make_context(ws)` example with `make_context()`. Explain runtime ownership, automatic one-shot enablement, explicit `ChipWorker(enable_sdma=True)` reuse, and current onboard a2a3 support. Remove workspace sizing, getter, `DeviceTensor`, malformed-buffer, and unsupported-target no-op claims. Update the PTOAS status note to distinguish PTOAS op availability from runtime-provisioned execution coverage.

- [ ] **Step 4: Run doc/API searches**

Run:

```bash
rg -n "make_context\(ws\)|sdma_prefetch_workspace_addr|WORKSPACE_BYTES" \
  python src include tests docs
```

- [ ] **Step 5: Commit ST and documentation**

```bash
git add tests/st/runtime/ops/test_prefetch_async.py \
  docs/en/dev/ir/05-operators.md docs/zh-cn/dev/ir/05-operators.md \
  docs/zh-cn/dev/ptoas-op-status.md python/pypto/ir/op/prefetch_ops.py \
  python/pypto/language/op/prefetch_ops.py src/ir/op/prefetch/prefetch_async.cpp
git commit -m "test(runtime): Cover injected SDMA prefetch"
```

### Task 6: Full verification, review, hardware ST, and PR update

**Files:**
- Review: all files changed since `7b3aeff7`
- Inspect: `/data/linyifan/pypto/KNOWN_ISSUES.md`

**Interfaces:**
- Consumes: completed implementation and test suite
- Produces: reviewed commits pushed to PR #2089 with verified CI restart

- [ ] **Step 1: Build from the worktree**

```bash
cmake --build build --parallel
```

Expected: exit 0 with no new warnings.

- [ ] **Step 2: Run focused and surrounding unit suites**

```bash
PYTHONPATH=$PWD/python:/data/linyifan/.conda/envs/lyf/lib/python3.10/site-packages \
python -S -m pytest \
  tests/ut/ir/operators/test_prefetch_ops.py \
  tests/ut/codegen/test_prefetch_codegen.py \
  tests/ut/codegen/test_pto_codegen.py \
  tests/ut/runtime/test_worker_reuse.py \
  tests/ut/runtime/test_run_config.py \
  tests/ut/runtime/test_execute_artifact.py \
  tests/ut/runtime/test_task_submit_dispatch.py \
  tests/ut/runtime/test_distributed_worker.py -q
```

- [ ] **Step 3: Run formatting, lint, and clang-tidy for the diff**

```bash
ruff check .
ruff format --check .
python tests/lint/clang_tidy.py --diff-base upstream/main
git diff --check upstream/main...HEAD
```

Classify any known main-branch clang-tidy findings using `KNOWN_ISSUES.md`; fix every finding on changed lines.

- [ ] **Step 4: Run the required architecture precheck and hardware ST**

```bash
source /usr/local/Ascend/cann/set_env.sh
runtime/.claude/skills/onboard-arch-precheck/check.sh a2a3
task-submit --list
```

Then submit the focused test with the worktree-local PyPTO/runtime Python paths and per-run `ASCEND_PROCESS_LOG_PATH`:

```bash
task-submit --device auto --timeout 3600 --max-time 1800 \
  --env ASCEND_PROCESS_LOG_PATH=/data/linyifan/pypto/.claude/worktrees/pr-2089/build/st-prefetch-logs/ascend \
  --run 'cd /data/linyifan/pypto/.claude/worktrees/pr-2089 && source /usr/local/Ascend/cann/set_env.sh && \
    PYTHONPATH=/data/linyifan/pypto/.claude/worktrees/pr-2089/python:/data/linyifan/pypto/.claude/worktrees/pr-2089/runtime:/data/linyifan/pypto/.claude/worktrees/pr-2089/runtime/python:/data/linyifan/.conda/envs/lyf/lib/python3.10/site-packages \
    runtime/.venv/bin/python -S -m pytest \
    tests/st/runtime/ops/test_prefetch_async.py -v --forked \
    --platform=a2a3 --device=$TASK_DEVICE'
```

Expected: one test passes, output is bit-exact, and the device log contains the SDMA stream provisioning message.

- [ ] **Step 5: Perform project code review and final verification**

Use `.claude/skills/code-review/SKILL.md` against `upstream/main...HEAD`, then use `superpowers:verification-before-completion`. Confirm the worktree is clean and the branch remains based on current `upstream/main`.

- [ ] **Step 6: Push and re-check PR #2089**

```bash
git push lyf issue-2075-prefetch-async
gh pr view 2089 --json headRefOid,mergeStateStatus,statusCheckRollup,url
gh pr checks 2089
```

Expected: PR head matches local HEAD and the new CI run is queued or running. Continue the `fix-pr` loop for actionable review comments and completed failing checks, up to five iterations.
