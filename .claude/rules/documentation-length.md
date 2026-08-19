# Documentation and File Length Guidelines

## Length Limits

| File class | Limit | Why |
| ---------- | ----- | --- |
| `docs/**.md` | ≤1000 lines | Read by humans on demand; scannability |
| `.claude/rules/*.md` | ≤200 lines **per file** and **≤2500 lines total** across the directory | Always-on context |
| `.claude/skills/*/SKILL.md` | ≤200 lines | Entry point — must be actionable at a glance |
| `.claude/skills/**` supporting files (`reference.md`, templates, scripts) | ≤500 lines | Loaded on demand, but read end-to-end mid-task |

**The aggregate budget is the binding constraint for rules.** Every file in
`.claude/rules/` is injected into the system prompt of *every* session, so the
directory total — not any single file — is what costs tokens and dilutes
attention. Nineteen 200-line files would each "comply" and still be unusable.
Check it before adding a rule file:

```bash
wc -l .claude/rules/*.md | tail -1   # keep the total ≤ 2500
```

When the total approaches the budget, **merge or delete a rule** rather than
splitting an oversized one into two always-loaded files — a split lowers the
per-file count while leaving the aggregate unchanged.

**Skills are load-on-demand, so only the entry point is tightly capped.** Keep
`SKILL.md` under 200 lines and push detail (full API tables, long worked
examples, step-by-step recipes) into sibling reference files that the skill
links to — those carry their own ≤500 limit (tighter than `docs/`, since a
reference file is read start-to-finish mid-task). A `SKILL.md` over the cap is a
signal to move content into a reference file, not to request an exception. When
the reference file has no headroom either, add a second one split by topic.

## When to Split vs Condense

### Split Files (>1000 lines — over the limit)

**For very large files, split into focused components:**

```text
# Example: Pass documentation split into topic folders
docs/en/dev/passes/
├── 00-pass_manager.md      (~295 lines) - Pass system overview
├── 02-unroll_loops.md      (~100 lines) - Loop unrolling
├── 04-convert_to_ssa.md    (~150 lines) - SSA conversion
└── ...                     - Individual pass docs
```

**Splitting criteria:**

- File has multiple distinct topics
- Each section could standalone
- >1000 lines even after condensing
- Natural breaking points exist

### Condense Files (800-1000 lines — approaching the limit)

**For moderately large files, condense content:**

**Apply techniques:**

- Tables over prose
- Consolidate similar examples
- Remove verbose explanations
- Cross-reference instead of repeating

## Condensing Techniques

### 1. Tables Over Prose

Replace paragraph descriptions with comparison tables.

### 2. Consolidate Examples

Show pattern once, not 5-10 times. One representative example per concept.

### 3. Remove Verbose "Why"

Keep "what" and "how", reduce "why" explanations.

### 4. Cross-Reference Instead of Repeating

Link to other docs instead of duplicating content.

### 5. Eliminate Redundancy

Combine similar sections that repeat the same pattern.

## File Organization Principles

### For Documentation

**Structure for scannability:**

- Clear headings (##, ###)
- Code blocks with language tags
- Tables for comparisons
- Bullet points over paragraphs
- Examples after concepts (not interleaved)

### For AI Rules/Skills

**Essential content only:**

- Core principles and patterns
- Key decision criteria
- 1-2 examples per concept
- Reference other files instead of duplicating
- Use numbered/bulleted lists

For skills specifically: `SKILL.md` states *what to do and when*; a reference
file holds *everything you need to look up while doing it*.

## Quality Checklist

Before finalizing, verify:

- [ ] File ≤ its target length (see "Length Limits")
- [ ] For a new/grown rule file: `.claude/rules/` total still ≤2500 lines
- [ ] All examples work and are necessary
- [ ] No redundant explanations
- [ ] Tables used for comparisons
- [ ] Cross-references accurate
- [ ] Technical accuracy maintained
- [ ] Scannability (can understand in 2 minutes)

## Enforcement

**Code review process checks:**

- New documentation files must comply
- Modified files should move toward compliance
- Files exceeding limits trigger review warnings
- Large PRs may require splitting documentation

## Exceptions

**Request user approval for:**

- Critical reference material (API specs, grammar definitions)
- Complex algorithms requiring detailed explanation
- Files with many necessary examples
- Migration guides with step-by-step instructions

**In all cases, try condensing first before requesting exception.**
