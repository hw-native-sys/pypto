# 显式 Tile Buffer Slot 实现计划

> **供智能体执行者使用：** 必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`，逐任务执行本计划。步骤使用 checkbox（`- [ ]`）跟踪。

**目标：** 增加一等公民、支持动态索引的 tile buffer 集合，让用户能在 PyPTO 和 PTOAS 两种内存规划器下分别轮换 L1、L0B 和 L0C destination。

**架构：** 新增存储类型 `TileBufferSetType`，并通过 tuple 风格的 Python `TileBufferSet` 暴露。`tile.create_buffer_set` 持有一个同构 allocation group，`tile.buffer_slot` 在运行时选择 slot 并返回普通 `TileType`，destination-form producer 直接写入该 slot。内存和依赖 pass 保留 allocation-set identity，使用 `(set identity, normalized slot index)` 做保守 alias 分析。

**技术栈：** C++17 IR 与 pass、nanobind Python binding、Python 3.10+ DSL、pytest、PTO MLIR codegen。

## 全局约束

- `count` 是处于 `[2, 16]` 的编译期整数。
- slot shape 非空、静态且所有维度为正数。
- memory space 仅支持 `Vec`、`Mat`、`Left`、`Right`、`Acc` 和 `Bias`。
- 未提供 `out=` 的现有 producer 保持原有签名和行为。
- 英文文档是规范源，中文文档与其同步。
- 每项生产行为都必须先由聚焦测试以预期原因失败，之后才能实现。

---

### 任务 1：一等公民 `TileBufferSetType`

**文件：**

- 修改：`include/pypto/ir/core.h`、`include/pypto/ir/kind_traits.h`
- 修改：`include/pypto/ir/type.h`、`src/ir/type.cpp`
- 修改：`src/ir/transforms/structural_equal.cpp`、`src/ir/transforms/structural_hash.cpp`
- 修改：`src/ir/serialization/serializer.cpp`、`src/ir/serialization/deserializer.cpp`
- 修改：`src/ir/transforms/python_printer.cpp`
- 修改：`python/bindings/modules/ir.cpp`、`python/pypto/pypto_core/ir.pyi`
- 新建：`tests/ut/ir/core/test_tile_buffer_set_type.py`
- 修改：`tests/ut/ir/transforms/test_serialization.py`、`tests/ut/ir/transforms/test_equality.py`

**接口：**

- 产出：`TileBufferSetType(shape, dtype, count, memref=None, tile_view=None, memory_space=None)`。
- 字段：`shape_`、`dtype_`、`count_`、`memref_`、`tile_view_`、`memory_space_`。

- [ ] **步骤 1：先写类型行为失败测试**

```python
def test_tile_buffer_set_type_exposes_homogeneous_slot_contract():
    ty = ir.TileBufferSetType([16, 128], DataType.FP32, 2, None, None, ir.MemorySpace.Acc)
    assert ty.count == 2
    assert [dim.value for dim in ty.shape] == [16, 128]
    assert ty.memory_space == ir.MemorySpace.Acc

@pytest.mark.parametrize("count", [1, 17])
def test_tile_buffer_set_type_rejects_invalid_count(count):
    with pytest.raises(ValueError, match=r"count.*\[2, 16\]"):
        ir.TileBufferSetType([16, 128], DataType.FP32, count)
```

- [ ] **步骤 2：运行 `python -m pytest tests/ut/ir/core/test_tile_buffer_set_type.py -v`，确认因类型不存在而失败。**
- [ ] **步骤 3：增加 C++ type、ObjectKind、reflection、binding 和 stub，并校验 count、静态正 shape 与片上 memory space。**
- [ ] **步骤 4：增加 count 参与 structural equality/hash，以及携带该类型的 Program serialization round trip 失败测试。**
- [ ] **步骤 5：实现 printer、equality、hash、serialization，运行类型、equality 和 serialization 测试直至通过。**
- [ ] **步骤 6：提交 `feat(ir): add tile buffer set type`。**

### 任务 2：Buffer-set IR op 与校验

**文件：**

- 修改：`src/ir/op/tile_ops/memory.cpp`
- 修改：`python/pypto/ir/op/tile_ops.py`、`python/pypto/ir/op/__init__.py`
- 新建：`tests/ut/ir/operators/test_tile_buffer_set_ops.py`

**接口：**

- `tile.create_buffer_set(shape, dtype, target_memory, count) -> TileBufferSetType`
- `tile.buffer_slot(buffer_set, index) -> TileType`
- `tile.release(slot) -> ScalarType(BOOL)`，作为 codegen 前删除的生命周期 marker。

- [ ] **步骤 1：为动态整数 index、常量越界、非整数 index 和错误 source 写失败测试。**

```python
buffers = tile_ops.create_buffer_set([16, 128], DataType.FP32, ir.MemorySpace.Acc, 2)
index = ir.Var("index", ir.ScalarType(DataType.INDEX), ir.Span.unknown())
slot = tile_ops.buffer_slot(buffers, index)
assert isinstance(slot.type, ir.TileType)
```

- [ ] **步骤 2：运行 `python -m pytest tests/ut/ir/operators/test_tile_buffer_set_ops.py -v`，确认三个 op 尚未注册。**
- [ ] **步骤 3：注册三个 op；slot 继承集合的 shape、dtype、view 和 memory space，但不创建独立 allocation。**
- [ ] **步骤 4：运行新测试与 `tests/ut/ir/operators/test_tile_ops.py`，确认通过。**
- [ ] **步骤 5：提交 `feat(ir): add tile buffer set operations`。**

### 任务 3：Tuple 风格 Python DSL API

**文件：**

- 新建：`python/pypto/language/typing/tile_buffer_set.py`
- 修改：`python/pypto/language/typing/__init__.py`
- 修改：`python/pypto/language/op/unified_ops.py`、`python/pypto/language/op/tile_ops.py`
- 修改：`python/pypto/language/op/__init__.py`、`python/pypto/language/__init__.py`
- 修改：`python/pypto/language/parser/_dsl_invoker.py`、`python/pypto/language/parser/ast_parser.py`
- 修改：`python/pypto/language/parser/type_resolver.py`
- 新建：`tests/ut/language/parser/test_tile_buffer_set.py`

**接口：**

- `create_tile_buffers(count, shape, dtype, memory_space) -> TileBufferSet`
- `TileBufferSet.__len__() -> int`
- `TileBufferSet.__getitem__(index: int | Scalar) -> Tile`
- `pl.tile.release(slot: Tile) -> None`
- printer round trip annotation：`pl.TileBufferSet[[16, 128], pl.FP32, 2, pl.MemorySpace.Acc]`

- [ ] **步骤 1：写包含 `buffers[i % 2]` 的 DSL parse 失败测试，检查一个 create op、动态 slot op 和 release marker。**
- [ ] **步骤 2：运行该测试，确认因 `create_tile_buffers`/`TileBufferSet` 缺失而失败。**
- [ ] **步骤 3：实现 wrapper、export、parser invocation 与 type resolver；`__getitem__` 必须调用 `tile.buffer_slot`，不能使用 `TupleGetItemExpr`。**
- [ ] **步骤 4：运行新测试及 subscript/tuple 语法回归测试。**
- [ ] **步骤 5：提交 `feat(language): expose explicit tile buffer sets`。**

### 任务 4：Group MemRef 规划

**文件：**

- 修改：`include/pypto/ir/transforms/utils/memref_utils.h`
- 修改：`src/ir/transforms/init_memref.cpp`、`src/ir/transforms/memory_reuse_pass.cpp`
- 修改：`src/ir/transforms/allocate_memory_addr_pass.cpp`、`src/ir/transforms/mutator.cpp`
- 修改：`tests/ut/ir/transforms/test_init_memref.py`
- 修改：`tests/ut/ir/transforms/test_memory_reuse.py`
- 修改：`tests/ut/ir/transforms/test_allocate_memory_addr_pass.py`

**接口：**

- set 获得一个大小为 `count * aligned_slot_size` 的 group `MemRef`。
- slot `TileType` 共享 group base；index 作为符号化 slot 选择保留，不形成独立 allocation root。

- [ ] **步骤 1：先写测试：两个 `[16, 128]` FP32 Acc slot 只产生一个 16384-byte group root，所有选中 slot 共享该 root。**
- [ ] **步骤 2：运行 `test_init_memref.py -k tile_buffer_set`，确认 pass 尚不识别该存储类型。**
- [ ] **步骤 3：扩展 type remap 与 InitMemRef，保留 count/view/memory space 并物化单一 group root。**
- [ ] **步骤 4：先写测试：不同显式 group 不被合并，普通 tile 不得复用显式 group，AllocateMemoryAddr 为每组分配单一对齐区间。**
- [ ] **步骤 5：按 `tile.create_buffer_set` lineage 排除 reuse candidate，并把 group 总大小纳入容量诊断。**
- [ ] **步骤 6：运行 InitMemRef、MemoryReuse、AllocateMemoryAddr 三组测试。**
- [ ] **步骤 7：提交 `feat(passes): plan explicit tile buffer groups`。**

### 任务 5：Destination-form producer 与 PTO codegen

**文件：**

- 修改：`src/ir/op/tile_ops/memory.cpp`、`transform.cpp`、`matmul.cpp`
- 修改：`python/pypto/ir/op/tile_ops.py`
- 修改：`python/pypto/language/op/unified_ops.py`、`tile_ops.py`
- 修改：`src/backend/common/pto_ops_memory.cpp`、`pto_ops_datamove.cpp`、`pto_ops_elementwise.cpp`
- 修改：`src/backend/common/pto_ops_shared.cpp`
- 修改：`include/pypto/codegen/pto/pto_codegen.h`、`src/codegen/pto/pto_codegen.cpp`
- 新建：`tests/ut/ir/operators/test_tile_destination_ops.py`
- 新建：`tests/ut/codegen/test_explicit_tile_buffers.py`

**接口：**

- 内部 op：`tile.load_into`、`tile.extract_into`、`tile.move_into`、`tile.matmul_into`、`tile.matmul_acc_into`。
- destination 是最后一个参数，结果 `TileType` alias destination `MemRef`。
- 现有公开函数增加 keyword-only `out: Tile | None = None`；`out is None` 时仍生成原 op。

- [ ] **步骤 1：先写成功 matmul destination，以及 shape/dtype/memory-space/valid-shape/layout 分别不匹配的失败测试。**
- [ ] **步骤 2：运行测试，确认 destination op 和 `out=` 尚不存在。**
- [ ] **步骤 3：复用现有 type deduction 注册 destination op，对推断结果与 destination contract 做完整比较，并返回 destination lineage。**
- [ ] **步骤 4：先写双 planner codegen 测试：PyPTO alloc 带 `addr`，PTOAS 不带；两者都有动态 get，producer `outs` 指向选中 slot，且没有单 slot alloc 或 Acc-to-Acc TMOV。**
- [ ] **步骤 5：实现 multi allocation/get、release 删除与五类 destination emitter。**
- [ ] **步骤 6：运行 destination、explicit codegen 和 planner switch 测试。**
- [ ] **步骤 7：提交 `feat(codegen): bind producers to explicit tile slots`。**

### 任务 6：Slot 生命周期、依赖与流水调度

**文件：**

- 修改：`include/pypto/ir/transforms/utils/stmt_dependency_analysis.h`
- 修改：`src/ir/transforms/utils/stmt_dependency_analysis.cpp`
- 修改：`src/ir/transforms/lower_pipeline_loops_pass.cpp`
- 修改：`src/ir/transforms/canonicalize_io_order_pass.cpp`、`src/ir/transforms/simplify_pass.cpp`
- 修改：`include/pypto/ir/verifier/verifier.h`、`CMakeLists.txt`
- 新建：`src/ir/verifier/verify_tile_buffer_lifetime.cpp`
- 修改：`src/ir/verifier/property_verifier_registry.cpp`
- 修改：`tests/ut/ir/transforms/test_lower_pipeline_loops.py`
- 修改：`tests/ut/ir/transforms/test_canonicalize_io_order.py`
- 新建：`tests/ut/ir/verifier/test_tile_buffer_lifetime.py`

**接口：**

- 依赖 key：`(buffer-set allocation identity, normalized slot index)`。
- verifier：`VerifyTileBufferLifetime(const ProgramPtr&)`。

- [ ] **步骤 1：先写 release 后使用、release 普通 tile、无排序边的重叠写入失败测试，并断言源代码诊断。**
- [ ] **步骤 2：运行 verifier 测试，确认 verifier 缺失。**
- [ ] **步骤 3：跟踪 slot 与 destination alias；lease 在显式 release 或最后 SSA use 结束，并在 codegen 前删除 marker。**
- [ ] **步骤 4：先写嵌套流水测试：L1 使用 `stack % 2`，Right/Acc 使用 `col % 2`；断言两 extract、两 matmul、两 store 排序及下次 slot 0 复用依赖。**
- [ ] **步骤 5：lowering clone 保留 set identity；简化 modulo residue；可证明 residue 不同时独立，否则保守 alias。**
- [ ] **步骤 6：运行 lifetime、LowerPipelineLoops 和 CanonicalizeIOOrder 测试。**
- [ ] **步骤 7：提交 `feat(passes): schedule explicit tile slot leases`。**

### 任务 7：文档与最终回归

**文件：**

- 修改：中英文 `docs/*/dev/ir/02-types.md`、`05-operators.md`
- 修改：中英文 `docs/*/dev/language/00-python_syntax.md`
- 修改：中英文 `docs/*/dev/passes/25-lower_pipeline_loops.md`、`28-init_memref.md`、`30-memory_reuse.md`
- 新建：`tests/ut/codegen/test_issue_2131_explicit_buffers.py`

- [ ] **步骤 1：增加 issue #2131 两级 L1/L0 流水端到端测试，精确检查 multi alloc/get、extract/matmul/store 数量和无冗余 allocation/move。**
- [ ] **步骤 2：同步更新中英文 API、校验、lifetime、planner 差异和嵌套流水示例。**
- [ ] **步骤 3：运行全部 feature-focused 测试，要求零失败。**
- [ ] **步骤 4：执行 `cmake --build build --parallel`，再运行受影响的 type、op、memory pass、pipeline 和 codegen suites。**
- [ ] **步骤 5：执行 `ruff check .`、`ruff format --check .`、`pyright` 和 `pre-commit run --all-files`。用户已确认可忽略与本功能无关的基线失败，但聚焦验证不能省略。**
- [ ] **步骤 6：检查 `git diff --check` 与完整 diff，提交 `docs(ir): document explicit tile buffer slots`。**
- [ ] **步骤 7：从已提交 HEAD 重新运行聚焦测试，记录精确通过/失败数与干净工作区状态后再声明完成。**
