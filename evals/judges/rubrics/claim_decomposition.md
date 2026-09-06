Split the answer below into atomic factual claims for grounding checks.

An atomic claim asserts exactly one checkable thing. The test: could this
sentence be supported by the passages while another part of the original
sentence is not? If yes, split it.

## Rules

1. **Split compound assertions.** "Revenue was $4,218 million and gross
   margin was 42.1%" is two claims — retrieval may well support one and not
   the other, and a single claim would hide that.
2. **Keep the subject, period, and entity in each claim.** Each claim is
   checked in isolation, so "It rose 11%" must become "Net revenue rose 11%
   in fiscal 2024". Resolve every pronoun.
3. **Drop pure discourse.** "Here is a summary", "Based on the filing",
   "I hope this helps" assert nothing checkable. Omit them.
4. **Keep hedges attached.** "Revenue may have exceeded $4 billion" stays
   one claim with its hedge intact; do not strip qualifiers to make the
   claim easier to check.
5. **Do not add, infer, or correct.** Reproduce what the answer asserts,
   even if it looks wrong. You are segmenting, not grading.
6. **Preserve figures exactly** as written, including units and currency.
7. If the answer is a refusal or contains no factual assertion, return an
   empty list.

## Examples

**Answer:** "Northwind's fiscal 2024 net revenue was $4,218 million, up 11%
from the prior year. Gross margin was 42.1%. I hope this helps."

```
["Northwind's fiscal 2024 net revenue was $4,218 million.",
 "Northwind's net revenue in fiscal 2024 rose 11% from the prior year.",
 "Northwind's gross margin was 42.1%."]
```

**Answer:** "The filing does not disclose segment-level headcount."

```
["The filing does not disclose segment-level headcount."]
```

**Answer:** "I could not find that information in the provided documents."

```
[]
```

## Output

Return JSON only:

- `claims`: an array of strings, each one atomic claim in the order it
  appears in the answer.
