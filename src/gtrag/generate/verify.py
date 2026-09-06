"""Post-generation verification.

A second pass that decomposes the answer into claims and checks each against
the retrieved context, flagging the unsupported ones.

The deliverable of this component is **a recommendation, not a feature**. It
costs a second model pass over every answer, and the ablation is how you find
out whether the quality it buys is worth that. Shipping it unconditionally
because it sounds rigorous is the failure mode; so is skipping it because it
sounds expensive.

Two verifiers behind one protocol, for the same reason as everywhere else:
the lexical one runs offline so the stage is testable and the entailment
verifier's contribution is a delta against something.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

__all__ = [
    "ClaimVerdict",
    "VerificationResult",
    "Verifier",
    "LexicalVerifier",
    "JudgeVerifier",
    "annotate_unsupported",
]

_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
_NUMERIC = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")
# Framing that asserts nothing checkable. The claim-decomposition rubric
# drops these, but the offline sentence splitter does not, and scoring them
# unsupported would penalise an answer for being readable.
_DISCOURSE = re.compile(
    r"^\s*(?:in summary|to summari[sz]e|here(?:'s| is| are)|summary|overall|"
    r"based on (?:the )?(?:filing|passages|context)|according to (?:the )?(?:filing|passages)|"
    r"i hope this helps|note that|in conclusion)\b",
    re.IGNORECASE,
)
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "had",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "to",
        "was",
        "were",
        "with",
        "we",
        "our",
        "this",
        "these",
        "those",
    ]
)


def _content_tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS}


@dataclass(frozen=True, slots=True)
class ClaimVerdict:
    claim: str
    supported: bool
    score: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "supported": self.supported,
            "score": round(self.score, 4),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class VerificationResult:
    verdicts: tuple[ClaimVerdict, ...] = ()
    verifier: str = ""

    @property
    def n_claims(self) -> int:
        return len(self.verdicts)

    @property
    def unsupported(self) -> tuple[ClaimVerdict, ...]:
        return tuple(v for v in self.verdicts if not v.supported)

    @property
    def groundedness(self) -> float | None:
        """Fraction of claims supported by the context.

        None for an empty answer: a refusal has nothing to ground, and
        scoring it 1.0 would let a system that refuses everything top the
        groundedness table.
        """
        if not self.verdicts:
            return None
        return sum(1 for v in self.verdicts if v.supported) / len(self.verdicts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verifier": self.verifier,
            "n_claims": self.n_claims,
            "n_unsupported": len(self.unsupported),
            "groundedness": self.groundedness,
            "verdicts": [v.to_dict() for v in self.verdicts],
        }


class Verifier(Protocol):
    name: str

    @property
    def config(self) -> dict[str, Any]: ...

    def verify(self, claims: Sequence[str], context: Sequence[str]) -> VerificationResult: ...


@dataclass
class LexicalVerifier:
    """Offline check: are the claim's content terms present in the context?

    Not entailment. It catches the failure that matters most on this corpus
    -- a figure the model produced that appears nowhere in the retrieved
    passages -- and misses paraphrase and inference entirely.

    Numbers are checked strictly and separately from words. A claim asserting
    "$4,218 million" when the context says "$3,800 million" shares most of
    its vocabulary and none of its meaning, so a single blended score would
    call it supported. Any numeric token in the claim that is absent from the
    context fails it outright.
    """

    threshold: float = 0.6
    name: str = "lexical"

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")

    @property
    def config(self) -> dict[str, Any]:
        return {"verifier": self.name, "verifier_threshold": self.threshold, "entailment": False}

    def verify(self, claims: Sequence[str], context: Sequence[str]) -> VerificationResult:
        joined = " ".join(context)
        context_tokens = _content_tokens(joined)
        context_numbers = {n.strip("$%").replace(",", "") for n in _NUMERIC.findall(joined)}

        verdicts: list[ClaimVerdict] = []
        for claim in claims:
            claim_numbers = {n.strip("$%").replace(",", "") for n in _NUMERIC.findall(claim)}
            missing_numbers = sorted(claim_numbers - context_numbers)
            if missing_numbers:
                verdicts.append(
                    ClaimVerdict(
                        claim=claim,
                        supported=False,
                        score=0.0,
                        reason=f"figure(s) not present in context: {', '.join(missing_numbers)}",
                    )
                )
                continue

            claim_tokens = _content_tokens(claim)
            if _DISCOURSE.match(claim) or not claim_tokens:
                # No checkable content -- discourse, not a claim.
                verdicts.append(
                    ClaimVerdict(
                        claim=claim, supported=True, score=1.0, reason="no factual content"
                    )
                )
                continue

            overlap = len(claim_tokens & context_tokens) / len(claim_tokens)
            verdicts.append(
                ClaimVerdict(
                    claim=claim,
                    supported=overlap >= self.threshold,
                    score=overlap,
                    reason=f"{overlap:.0%} of content terms present",
                )
            )
        return VerificationResult(verdicts=tuple(verdicts), verifier=self.name)


@dataclass
class JudgeVerifier:
    """Entailment checking via the `claim_support` rubric.

    Reuses the Phase 2 judge, so verification is scored by the same
    calibrated rubric that measures groundedness -- rather than a second,
    uncalibrated notion of support that would quietly disagree with it.
    """

    judge: Any = None
    name: str = "judge"
    _fallback: LexicalVerifier = field(default_factory=LexicalVerifier)

    @property
    def config(self) -> dict[str, Any]:
        model = getattr(self.judge, "model", "none")
        return {"verifier": self.name, "verifier_model": model, "entailment": True}

    def verify(self, claims: Sequence[str], context: Sequence[str]) -> VerificationResult:
        if self.judge is None:
            return self._fallback.verify(claims, context)

        verdicts: list[ClaimVerdict] = []
        for claim in claims:
            judgment = self.judge.score(
                "claim_support",
                question="",
                answer=claim,
                context=context,
                extra={"claim": claim},
            )
            if judgment.error:
                # A failed check is unscored, never a silent "supported".
                fallback = self._fallback.verify([claim], context).verdicts[0]
                verdicts.append(
                    ClaimVerdict(
                        claim=claim,
                        supported=fallback.supported,
                        score=fallback.score,
                        reason=f"judge unavailable ({judgment.error}); lexical fallback",
                    )
                )
                continue
            verdicts.append(
                ClaimVerdict(
                    claim=claim,
                    supported=judgment.score == 2,
                    score=float(judgment.score) / 2.0,
                    reason=judgment.reasoning,
                )
            )
        return VerificationResult(verdicts=tuple(verdicts), verifier=self.name)


def annotate_unsupported(answer: str, result: VerificationResult) -> str:
    """Append an explicit caveat naming the unsupported claims.

    Surfacing the flag to the reader rather than silently suppressing the
    answer: the claim may well be true, and a system that deletes anything it
    cannot verify is less useful than one that says which parts it could not.
    """
    unsupported = result.unsupported
    if not unsupported:
        return answer
    listed = "; ".join(f'"{v.claim}"' for v in unsupported)
    return (
        f"{answer}\n\n[Not supported by the retrieved passages: {listed}. "
        f"Treat these as unverified.]"
    )
