# Reply Language

## Core Principle

**Reply in the language the user wrote in. Never default to Japanese.**

This rule complements `language-policy.md` (which governs intermediate vs. final
output language). This rule fixes one specific recurring failure: replying in
Japanese when the user did not write in Japanese.

## The Rule

1. **Match the user's language in the final reply.** Detect the language of the
   user's most recent message and respond in that same language.
   - User writes Chinese → reply in Chinese
   - User writes English → reply in English
   - User writes Japanese → reply in Japanese (only then)

2. **Never use Japanese unless the user's message is in Japanese.** Japanese is
   not a default, a fallback, or a "neutral" choice. If the user has never
   written Japanese in the conversation, do not produce Japanese output.

3. **When the user mixes languages**, follow the dominant language of their
   latest message. If genuinely ambiguous (e.g. a one-word message, a bare code
   snippet, or only a file path), default to **English**, never Japanese.

4. **Intermediate work stays English** per `language-policy.md` — reasoning,
   narration between tool calls, commit messages, code comments. Only the final
   user-facing reply matches the user's language.

## Quick Check Before Replying

```text
What language was the user's latest message in?
├─ Chinese  → reply in Chinese
├─ English  → reply in English
├─ Japanese → reply in Japanese
└─ Ambiguous / code-only → reply in English (NOT Japanese)
```

## Why

The user works primarily in Chinese and English. Unprompted Japanese replies are
unreadable to them and break the conversation. Matching the user's language is a
baseline requirement for clear communication.

## See Also

- `language-policy.md` — full intermediate vs. final output language policy
