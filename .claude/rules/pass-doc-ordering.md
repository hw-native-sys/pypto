# Pass Documentation Ordering

## Rule

Keep pass documentation in compilation order. Before adding, removing, or
reordering a pass, read the pass documentation index in `docs/en/dev/passes/`
and follow its links to the pass manager overview and relevant pass pages.
Use `python/pypto/ir/pass_manager.py` to verify the actual execution order,
including repeated utility passes and optional or backend-dependent passes.

The documentation index describes the numbering convention and current pass
inventory. Documentation numbers are not necessarily execution-slot numbers.

## When Changing the Pipeline

1. Locate the affected passes by name in the documentation and pass manager.
2. Follow the documented numbering convention; renumber subsequent pages when
   needed, using temporary names to avoid rename collisions.
3. Keep the English and Chinese pass pages and indexes synchronized.
4. Update the pass manager documentation, site navigation, and cross-references
   affected by added, removed, or renamed pages.
5. Verify that documented order and optional-pass conditions match the code.

## Keep Policy Independent of the Pass Inventory

Keep current pass lists, numeric positions, and numbered per-pass page links in
`docs/`, not in agent rules. In rules, describe semantic ordering constraints by
pass name and direct readers to the documentation for current details.

Adding, removing, or reordering a pass should not require an agent-policy edit
solely to maintain an index or numeric reference. Update policy only when the
development rules themselves change.
