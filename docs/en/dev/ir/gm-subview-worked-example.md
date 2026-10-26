# GM Subview RFC: A Complete Worked Example

This accompanies the [design RFC](gm-subview-rfc.md). It follows one case through
API, IR, passes and PTO codegen. The implementation tests now exercise this
chain and compile actual emitted PTO. The standalone fixture remains an
independent backend probe; device numerical and physical DMA evidence is separate.

## 1. Inputs, output and interface differences

Use an ordinary packed ND BF16 Tensor of shape [8,512], select rows 1 and 4,
and copy them into an ordinary packed output Tensor of shape [2,512].

| Operation on the same source | Meaning | Result / InCore behavior |
| ---------------------------- | ------- | ------------------------ |
| tensor.subview(cache, [2,512], [1,0], steps=[3,1]) | Sample a strictly smaller parent-coordinate window without loading | GM Tensor, origin 512, stride [1536,1] |
| tensor.view(cache, shape=[2,2048]) | Reinterpret the same 4096 elements using a canonical shape/layout | GM Tensor; not the two selected rows |
| tensor.view(cache, shape=[2,512]) | Tries to reinterpret 4096 elements as 1024 | Existing static element-count check rejects |
| tensor.slice(cache, [2,512], [1,0]) | Select consecutive rows 1 and 2 | GM-input slice becomes a Tile load in InCore |
| load(cache, [1,0], [2,512]) | Transfer consecutive rows 1 and 2 explicitly | Fresh Tile |
| load(subview, [0,0], [2,512]) | Transfer sampled rows 1 and 4 explicitly | Fresh Tile; descriptor stays GM |

Subview changes the starting coordinate and sampling stride while preserving
the parent's address map. View reinterprets shape/layout over existing storage;
it has no offsets/steps parameters and is not a bounded sampling operation.
Slice keeps its current materialization semantics. This RFC adds no implicit
slice-to-subview rewrite.

The complete user-facing program is:

~~~python
import torch
import pypto.language as pl
from pypto.runtime import RunConfig


@pl.jit
def copy_selected_rows(cache: pl.Tensor, output: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        v = pl.tensor.subview(cache, [2, 512], [1, 0], steps=[3, 1])
        pair = pl.load(v, [0, 0], [2, 512])
        pl.store(pair, [0, 0], output)
    return output


if __name__ == "__main__":
    rows = torch.arange(8).reshape(8, 1)
    cols = torch.arange(512).reshape(1, 512)
    cache = (16 * rows + cols % 16).to(torch.bfloat16)
    output = torch.zeros((2, 512), dtype=torch.bfloat16)
    copy_selected_rows(cache, output, config=RunConfig())
    torch.testing.assert_close(output, cache[[1, 4], :], rtol=0, atol=0)
~~~

Run this only after implementing the API and executable pipeline, on a supported
runtime/device. All values are BF16-exact integers; row/column encoding detects
the wrong sampled row. The test must compile and execute, not stop at lower().

## 2. How the API constructs IR

The DSL wrapper unwraps the source and integer Scalar operands, calls the Python
IR builder, then wraps its Call result as a Tensor. The public signature is
defined in RFC section 4.1; omitted steps become rank-many ones.
The intended IR-builder body reuses the existing tuple helper and generic op
construction:

~~~python
from pypto.pypto_core import ir as core
from pypto.ir.utils import _to_make_tuple


def subview_ir(source, shape, offsets, steps, span):
    return core.create_op_call(
        "tensor.subview",
        [
            source,
            _to_make_tuple(shape, span),
            _to_make_tuple(offsets, span),
            _to_make_tuple(steps, span),
        ],
        {},
        span,
    )
~~~

This low-level builder illustrates the four IR operands; the public wrapper
also normalizes integer scalars to INDEX before arithmetic. The C++ operator
is registered by the implementation. No arbitrary Python callback or extra custom IR node is involved.
DeduceTensorSubviewType computes each result stride from the immediate parent's
stride and step. Origin remains in the defining Call rather than a new type
offset field.

The complete InCore body, omitting only the enclosing printed-program class,
has the following expected operation sequence (IR notation, not printer syntax):

~~~text
func copy_selected_rows_incore(cache: Tensor<[8,512],BF16>,
                              output: Tensor<[2,512],BF16>):
  v    = tensor.subview(cache, (2,512), (1,0), (3,1))
  pair = tile.load(v, (0,0), (2,512), (2,512))
  out  = tile.store(pair, (0,0), output)
  return out
~~~

The actual initial/default target-memory annotation may remain unset until
InferTileMemorySpace; the transfer Call is already tile.load. JIT specialization,
SSA naming and the surrounding orchestration are ordinary existing machinery.

| Value | IR type / metadata | Address / allocation |
| ----- | ------------------ | -------------------- |
| cache | TensorType([8,512], BF16), canonical ND stride [512,1] | Caller GM pointer |
| v | TensorType([2,512], BF16), TensorView stride [1536,1], layout ND, addressing kind AffineSubview | No MemRef, no new payload allocation |
| pair | TileType([2,512], BF16), eventually Vec | Fresh result Tile allocation |
| out | Store result for output | Ordinary output GM alias |

Required operator-test assertions are exact: Call op name tensor.subview,
four operands in source/shape/offsets/steps order, ordinary TensorType result,
the type metadata above, and read-only alias root cache.

The bounds proof is concrete:

~~~text
Row:    shape=2 > 0; step=3 > 0; offset=1 >= 0;
        1 + (2-1)*3 = 4 < 8.
Column: shape=512 > 0; step=1 > 0; offset=0 >= 0;
        0 + (512-1)*1 = 511 < 512.
Strict reduction: 2 < 8.
Address(v[r,c]) = cache_pointer + (512 + r*1536 + c) BF16 elements.
~~~

Changing row offset to 5 makes the last row 8: reject. Using an unconstrained
device-loaded offset/step gives unknown: reject. An identity [8,512] subview also
fails the requested strict-reduction rule. Dynamic guarded inputs require the
additional scoped proof in RFC section 5.1.

## 3. Required pass handling, with before/after contracts

| Pass / component | Before -> after for this case | Required adjustment |
| ---------------- | ----------------------------- | ------------------- |
| SSA / outlining | Local descriptor and scalar geometry stay inside the consuming kernel | Do not outline a descriptor into a new function parameter or lose a required guard |
| Bounds verification | Pending geometry -> proven row/column containment and strict reduction | New shared contextual visitor before conversion and again before codegen; false/unknown rejects |
| ConvertTensorToTileOps | tensor.subview + tile.load -> tensor.subview + tile.load | Explicit metadata passthrough; prediction/preload analysis preserves GM identity; validate rewritten load source |
| ConsumerSpaceCollector | Vec demand at load -> demand propagated to appropriate consumer | No global descriptor-to-Tile replacement |
| MaterializeTensorStrides / canonical verifier | Explicit [1536,1], AffineSubview -> same | Preserve affine kind/stride; distinguish canonical and affine rules |
| InitMemRef / memory planning | Borrowed GM descriptor -> no MemRef; result Tile -> normal placement | Skip descriptor allocation and inherited-offset MemRef synthesis |
| Composite / ND/layout passes | Rank-2 ND geometry -> same coordinate map | No packed-stride reconstruction; diagnose unsupported rewrites |
| Final backend check | Proven geometry -> proven target-legal load | Validate dtype, layout, pitch, alignment and integer widths independently |

For the pass unit test, construct Before and Expected programs with the operation
sequence in section 2 and compare using ir.assert_structural_equal. Before
memory-space inference the explicit target may remain unset; after inference
Expected places pair in Vec. Additionally assert that v remains an ordinary
Tensor, only one tile.load exists, and it consumes v. These are separate
checkpoints, not one assumed final transformation.

The metadata/memory test must assert v.memref is absent after InitMemRef and
the only input-side payload allocation is pair. Tile physical size follows the
selected Tile/Buffer memory planner. A zero-sized descriptor MemRef is not an
acceptable substitute.

Additional operator paths require their own before/after tests:

| Path | Required behavior / artifact |
| ---- | ---------------------------- |
| subview -> nested subview -> load | Compose parent stride and origin exactly once |
| subview -> slice -> compute | Slice becomes a load from v; it must use v's sampled pitch |
| subview -> set_validshape -> load | Keep narrowed GM descriptor; no tile.set_validshape or preload |
| subview -> view -> load | Prove the existing packed-view restriction; this [1536,1] example cannot flatten away the row gap |
| slice -> subview / slice -> load in InCore | Do not reinterpret the existing slice; diagnose rewritten Tile input |

## 4. Complete codegen output and a runnable PTO fixture

The backend handler for tensor.subview belongs with the memory handlers in
src/backend/common/pto_ops_memory.cpp. It must:

1. Resolve the ordinary source Tensor's current-origin pointer and real strides.
2. Form origin = 1*512 + 0*1 = 512 elements; form result stride [3*512,1].
3. Emit pto.addptr and pto.make_tensor_view in this InCore function.
4. Bind v's current-origin pointer with RegisterBasePtr and its descriptor with
   RegisterTensorView / RegisterVarToMlir, as existing view handlers do.
5. Let existing tile.load emit partition_view(v,[0,0],[2,512]) and tload.
   Do not apply origin a second time. Let existing tile.store handle output.

The **complete, standalone module**, including constants, result Tile allocation,
output descriptor/store and return, is
[gm_subview_static.pto](../../../../tests/fixtures/gm_subview_static.pto).
It is an independent handwritten backend probe. Implementation tests separately
compile the PTO emitted by the real API and default passes.
It contains no undefined SSA values or tile-buffer-type placeholders.

Compile it from the repository root:

~~~bash
/usr/local/ptoas/0.67/bin/ptoas tests/fixtures/gm_subview_static.pto \
  --pto-arch=a2 --pto-level=level2 --enable-insert-sync \
  -o /tmp/gm_subview_static.cpp
~~~

Use a3 instead of a2 for an A3 compiler check. The module's a2a3 family attribute
is distinct from PTOAS's accepted a2/a3 command-line targets. Level2 delegates
result Tile placement to PTOAS; it does not allocate a GM descriptor buffer.

On PTOAS 0.67 the A2 output has one BF16 Vec Tile declaration with dimensions
2 by 512, one TLOAD and one TSTORE. Its global-memory descriptor starts at
cache + 512 elements and has row stride 1536 elements. The payload is 2048
logical bytes; the selected address envelope is 4096 bytes. No 4096-byte
intermediate Tile or envelope transfer is permitted.

An automated fixture check is provided in
[test_gm_subview_rfc_fixture.py](../../../../tests/ut/codegen/test_gm_subview_rfc_fixture.py).
After building the worktree and loading the testing environment, run:

~~~bash
source .claude/skills/testing/load-env.sh
cmake --build build --parallel "$PYPTO_BUILD_JOBS"
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest tests/ut/codegen/test_gm_subview_rfc_fixture.py -v
~~~

This fixture checks compiler acceptance and the TLOAD/TSTORE path. It does not
prove final machine DMA count or device correctness. The eventual codegen test
must compile the actual transformed API program, compare its address/stride and
allocation semantics with this module, then invoke PTOAS on that emitted module;
matching a handwritten module alone cannot establish API-to-backend correctness.

## 5. Independent data/address check and evidence

The complete CPU reference below needs only Python's standard library. It checks
every selected address against direct row indexing without calling subview:

~~~python
source = [[16 * r + c % 16 for c in range(512)] for r in range(8)]
flat = [value for row in source for value in row]
selected = [[flat[512 + r * 1536 + c] for c in range(512)] for r in range(2)]
assert selected == [source[1], source[4]]
assert len(selected) * len(selected[0]) * 2 == 2048
assert (512 + 1536 + 511 - 512 + 1) * 2 == 4096
~~~

Evidence updated on 2026-10-26:

| Check | State |
| ----- | ----- |
| Static bounds / CPU address mapping | Verified against every selected element |
| Complete PTO fixture, PTOAS 0.67 A2/A3 | Assembles successfully; one TLOAD, one TSTORE |
| New API / default passes / actual emitted PTO | Static and guarded dynamic BF16 compile for A2/A3; one result Tile, one TLOAD/TSTORE |
| Worktree build / pytest | Rebased build and 67 focused operator/codegen/backend-inventory checks pass; C++ tests 5/5 pass |
| Runtime case, codegen-only | Passes, including orchestration and InCore generation |
| Simulator numerical execution | Blocked by RuntimePinMismatch: installed runtime differs from the pinned source |
| Whole UT suite | On 935ed13c: 19,873 pass, 94 fail (69 mixed-shape constructor, 23 runtime dependency/import, 2 printing); on ced0e5ab, build/focused/C++/codegen-only checks pass; whole suite was not repeated |
| Device result / single hardware DMA | Not run; no claim |

Implementation acceptance requires the user program, operator assertions,
pass checkpoints, emitted-PTO compile and runtime data comparison all to agree.

Run the implementation chain with an already built worktree:

~~~bash
source .claude/skills/testing/load-env.sh
export PYTHONPATH="$PWD/python:$PYTHONPATH"
export PTOAS_ROOT=/usr/local/ptoas/0.67
python -m pytest tests/ut/ir/operators/test_tensor_subview.py \
  tests/ut/codegen/test_tensor_subview.py tests/ut/codegen/test_gm_subview_rfc_fixture.py \
  tests/ut/backend/test_buffer_migration_inventory.py -n "$PYPTO_TEST_JOBS" -q
~~~

Generated default-pipeline PTO uses `--pto-level=level3` for explicitly assigned
Tile addresses. The handwritten fixture above uses level2 without explicit addresses.
The targeted runtime case is `tests/st/runtime/ops/test_tensor_subview.py`; it
requires the simulator/device runtime and is not evidence of a physical single DMA.
