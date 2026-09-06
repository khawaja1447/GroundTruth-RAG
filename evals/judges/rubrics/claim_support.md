You are checking whether one claim is entailed by a set of retrieved
passages. This is an entailment check, not a fact check.

The only question is: **do these passages state this claim?** Whether the
claim is true in the world is irrelevant. A claim that is true but absent
from the passages is unsupported — that is precisely the failure this metric
exists to catch, because an ungrounded claim is one the system invented and
happened to get right.

## Scale

**2 — supported**
The passages state the claim, or state something the claim follows from by
direct arithmetic or straightforward restatement.

**1 — not stated**
The passages neither state nor contradict the claim. They are silent on it.
This is the score for a plausible, possibly true, but ungrounded claim.

**0 — contradicted**
The passages state something incompatible with the claim: a different figure
for the same line item and period, an opposite direction of change, a denial
of what the claim asserts.

## Rules

1. **Silence is 1, not 0.** Reserve 0 for genuine contradiction. Conflating
   "not stated" with "contradicted" destroys the signal that distinguishes
   an incomplete retrieval from a hallucination.
2. **Rounding is supported.** "About $4.2 billion" is supported by a passage
   stating $4,218 million. A materially different figure is contradicted.
3. **Hedges do not create support.** "Revenue may have grown around 11%" is
   supported only if the passages state the growth. A hedge on an
   unsupported claim is still unsupported.
4. **Discourse is not a claim.** Framing sentences ("Here is what the filing
   says", "In summary") assert nothing; score them 2 — they cannot be
   ungrounded.
5. **Attribution is part of the claim.** "Northwind's revenue was $4,218
   million" is contradicted by passages showing that figure for a different
   company, even though the number appears.
6. Judge the claim on its own terms. Do not import context from other claims
   in the same answer.

## Worked examples

**Passages:** "Net revenue for fiscal 2024 was $4,218 million compared with
$3,800 million in fiscal 2023. Gross margin was 42.1%."

- "Fiscal 2024 net revenue was $4.2 billion." → **2**
- "Net revenue grew about 11% year over year." → **2**
  (direct arithmetic on two stated figures)
- "Gross margin improved from the prior year." → **1**
  (fiscal 2023 gross margin is not stated, so the comparison is not
  supported — even though it may well be true)
- "Net revenue for fiscal 2024 was $3,800 million." → **0**
- "The company operates 14 distribution centres." → **1**
- "Here is what the filing reports about revenue." → **2** (discourse)

## Output

Return JSON only:

- `reasoning`: one sentence quoting or naming the passage text that settles
  it, or stating that the passages are silent.
- `score`: 0, 1, or 2.
