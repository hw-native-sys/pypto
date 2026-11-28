# LowerGraphForRuntime Pass

The first pass of the default pipeline reads the target runtime from
`PassContext`. On `host_build_graph` (HBG), Graph functions and scopes proceed
to the existing outlining, legalization, and recording path. On
`tensormap_and_ringbuffer` (TMR), the pass prepares them for ordinary task
submission.

## TMR lowering

`FunctionType::Graph` functions become `FunctionType::Inline`. The existing
`InlineFunctions` pass substitutes arguments, renames locals, and binds return
values at each call site. A `GraphScopeStmt` expands to its body in the same
position in the enclosing function. The transformation preserves the original
program so it can also be compiled for HBG.

SSA conversion, task dependency analysis, and memory planning see the expanded
orchestration. Loops, branches, kernel calls, and explicit task dependencies
inside the fragment remain in the IR. The generated orchestration submits the
individual tasks on each execution.

Graph boundary dump tags are forwarded through the same path as Inline dump
tags. An optional `calendar_hash` PGO bucket hint is dropped when TMR inlines a
Graph call or scope. An explicit `Submit` targeting a Graph function is a compile error on
TMR. Other metadata attached to a Graph call or scope also produces a diagnostic,
including outer dependencies and a captured completion TaskId on a Graph scope.
Those dependencies and IDs refer to the outer scheduling boundary removed by
TMR lowering.

The HBG recorder starts with a separate automatic runtime scope. Inlining
a Graph into an enclosing manual scope on TMR would change task dependency
tracking. TMR disallows Graph calls and regions inside a manual scope, including
through Inline helpers. Manual scopes authored inside an ordinarily invoked
Graph stay in its body.

The pass has no required or produced IR properties. It invalidates
`InlineFunctionsEliminated` because it can introduce Inline functions on TMR.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::LowerGraphForRuntime()` | `passes.lower_graph_for_runtime()` | Program-level |

## See also

- [InlineFunctions](02-inline_functions.md)
- [OutlineGraphScopes](09-outline_graph_scopes.md)
- [LegalizeGraphBoundary](53-legalize_graph_boundary.md)
