# Structural Comparison

`structural_equal`, `structural_hash`, and `assert_structural_equal` use one exact
structural semantics. There is no comparison mode. `Type.__eq__` and
`Type.__hash__` use the same implementation.

```python
from pypto import DataType, ir

span = ir.Span.unknown()
x = ir.Var("x", ir.ScalarType(DataType.INDEX), span)
y = ir.Var("y", ir.ScalarType(DataType.INDEX), span)
assert not ir.structural_equal(x, y)
assert ir.structural_equal(x, y, enable_auto_mapping=True)
assert ir.structural_hash(x, True) == ir.structural_hash(y, True)
ir.assert_structural_equal(x, y, True)
```

## Equality Contract

| Field | Comparison |
| ----- | ---------- |
| Source spans, comments, function and bound-variable names | Ignored through `IgnoreField` |
| Definitions and uses | Consistently alpha-renamed; alias patterns preserved |
| Free variables | Identity preserved unless `enable_auto_mapping=True` |
| Scalar dtypes | Exact raw codes, including `INDEX` versus `INT64` |
| Floating values | Exact stored bits, including signed zero and NaN payload/sign |
| Operators | Kind and name, independent of pointer identity |
| Attributes | Unique keys sorted; value types and vector order retained |
| Types and views | All reflected fields, including attached storage and nullable offsets |
| Function metadata | Directions, return types, attrs, kind, stage and runtime-binding flag |
| Call and Submit | Distinct kinds; reflected scheduling fields participate |

`f(x) = x + x` and `g(y) = y + y` compare equal. Replacing the second `y` with
another parameter changes the reference pattern and compares unequal. Sharing a
pure expression subtree does not change equality.

```python
a = ir.ConstFloat(0.0, DataType.FP32, span)
b = ir.ConstFloat(-0.0, DataType.FP32, span)
assert not ir.structural_equal(a, b)
assert not ir.structural_equal(
    ir.ConstInt(1, DataType.INDEX, span),
    ir.ConstInt(1, DataType.INT64, span),
)
```

A Function reserves all parameters before traversing its signature, so an earlier
tensor shape may reference a later scalar parameter. Symbols in physical parameter
shapes and bare `valid_shape` slots are additional bindings. The classification is
shared with `MaterializeValidShapeSymbols` and traverses tuple constructors and
item expressions. Other open references retain free-variable semantics.
Program functions have independent contexts; program map keys and callee names
remain significant.

## Storage and Partial IR

`MemRef`, `WindowBuffer`, pointer/resource marker types, and attached storage are
supported. Storage descriptors retain their base allocation references, geometry,
slot expressions, and staging flags. Different allocation-reference patterns must
not collapse merely because their sizes match. Unknown types can be compared as
unknown types; their presence does not certify that type inference is complete.

Structure comparison does not verify SSA, scope validity, closure, side effects,
or eligibility for a particular optimization. Comparing invalid IR to itself can
succeed. Optimization passes must verify their own preconditions first.

```python
ir.verify_structural_bindings(function)
fingerprint = ir.structural_hash(function)
```

`verify_structural_bindings` checks repeated definitions and local uses before
definition or outside lexical scope, including uses inside types and attributes.
Assignments bind after their RHS, loop initializers use the outer scope, and region
results bind after the region. Branch/loop locals cannot escape; straight-line
scope definitions can flow out. Open captures are allowed. The verifier also checks
signature-symbol eligibility and rejects shared statement objects. It is not a
replacement for the complete compiler verifier or an optimization's effect checks.

## Diagnostics and Unsupported Payloads

`assert_structural_equal` raises `ValueError` with the first differing field path.
Unrepresentable structures raise `StructuralComparisonUnsupported` (a `TypeError`)
with `reason` and `field_path`: unknown attribute payload types, duplicate attribute
keys, cycles, and node/type nesting beyond 512 levels. These errors are distinct
from inequality. Binding verification uses the same structured error reporting;
`BindingNotVisible` identifies the actual invalid use, including later uses of a
shared expression or type.

## Implementation and Complexity

The reflection descriptors determine the compared fields: `IgnoreField` skips
metadata, `UsualField` follows structure, and `DefField` establishes a binding.
One projection in `src/ir/transforms/structural_comparison.cpp` supplies equality,
hashing, and diagnostics. It first reserves binding identities, then builds immutable
records containing exact scalar values and child record IDs. Records are interned
by full ordered keys. Equality compares IDs in a shared arena, never hash values.
Hashes combine child hashes rather than arena IDs, so independent invocations agree.

Memoization distinguishes definitions from references and belongs to one function
context. Expression/type DAGs are traversed without expansion into trees. For N
visited nodes plus field/string data, projection costs O(N log N) time and O(N)
space. Binding verification propagates use intervals through DAG edges to check all
use sites without repeatedly traversing shared subtrees.

Hashes are process-local fingerprints, not persistent cache keys. Use the same
`enable_auto_mapping` policy for hashing and equality, and always confirm hash
matches with equality before merging candidates.

## Migration from Relaxed Comparison

Structural comparison no longer treats integer syntax dtypes as interchangeable,
compares floating values numerically, or ignores attached memory references.
Consumers needing type compatibility must state which fields are irrelevant to
that operation. Printers/parsers must preserve exact dtypes, or their callers must
explicitly normalize syntax before comparison. Do not add compatibility exceptions
to the structural comparator. Tuple annotations validate element kinds, shapes, and
dtypes recursively. Omitted views and memory spaces are inherited from inference, including for
annotated loop results. Allocation annotations, including their absence, remain exact.

## Lossless Text Roundtrips

The printer retains non-INDEX shape dtypes and typed offset expressions. Storage
calls use private `pl._dump.call` and `pl._dump.submit` spellings to preserve their
recorded result types independently of operator or callee inference. Multi-result
regions use `pl._dump.yield_(..., types=[...])`. These spellings are parser markers,
not a kernel authoring API. Roundtrip instruments explicitly map open references
because reconstruction creates fresh objects; definitions and storage aliases
still participate in exact comparison.

Within a function, allocation bases referenced only by storage descriptors are
implicit resource bindings. A MemRef's identity is its base plus descriptor fields,
not the descriptor object's address. WindowBuffer retains its resource binding and
all storage fields. This permits renamed and serialized functions to compare equal
while preserving allocation and window alias patterns.
