# Documentation and File Length Guidelines

## Length Limits

**Strict limits for maintainability and readability:**

- **Documentation files** (`docs/`): ≤500 lines
- **AI rules** (`.claude/rules/`): ≤200 lines
- **AI skills** (`.claude/skills/`): ≤200 lines
- **AI agents** (`.claude/agents/`): ≤200 lines

## When to Split vs Condense

### Split Files (>700 lines)

**For very large files, split into focused components:**

```text
# Example: Pass documentation split into topic folders
docs/dev/passes/
├── 00-pass_manager.md      (~295 lines) - Pass system overview
├── 01-verifier.md          (~498 lines) - IR verification
├── 02-convert_to_ssa.md    (~150 lines) - SSA conversion
└── ...                     - Individual pass docs
```

**Splitting criteria:**

- File has multiple distinct topics
- Each section could standalone
- >700 lines even after condensing
- Natural breaking points exist

### Condense Files (500-700 lines)

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

### For AI Rules/Skills/Agents

**Essential content only:**

- Core principles and patterns
- Key decision criteria
- 1-2 examples per concept
- Reference other files instead of duplicating
- Use numbered/bulleted lists

## Quality Checklist

Before finalizing, verify:

- [ ] File ≤ target length (500 for docs, 200 for AI files)
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
