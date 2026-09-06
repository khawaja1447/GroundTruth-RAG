"""Generation-side metrics.

Split deliberately into two groups:

  * **Deterministic** -- refusal correctness, citation validity, exact-match
    style checks. No model call, no API key, no cost. These run in CI on
    every push and are the reason the regression gate is nearly free.
  * **Judged** -- correctness, groundedness, context sufficiency. These need
    a model and live in `evals.judges`; this module only defines how their
    outputs are scored and aggregated.

Keeping the boundary sharp is what makes the harness usable before you have
budget, and keeps the expensive half honest: anything checkable by code is
checked by code.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from gtrag.types import SystemResponse

from ..types import EvalQuestion

__all__ = [
    "RefusalOutcome",
    "score_refusal",
    "CitationValidity",
    "citation_validity",
    "split_claims",
    "groundedness_from_claims",
]


# --------------------------------------------------------------------------
# Refusal
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RefusalOutcome:
    """The 2x2 of refusal behaviour.

    Reporting only `correct_refusal` is the common mistake: a system that
    refuses everything scores 100% on the unanswerable slice while being
    useless. `false_refusal` is the other half of the tradeoff and must be
    reported alongside it.
    """

    correct_refusal: bool | None  # unanswerable & refused
    false_refusal: bool | None  # answerable & refused
    hallucination_risk: bool | None  # unanswerable & answered anyway

    def to_dict(self) -> dict[str, bool | None]:
        return {
            "correct_refusal": self.correct_refusal,
            "false_refusal": self.false_refusal,
            "answered_unanswerable": self.hallucination_risk,
        }


def score_refusal(question: EvalQuestion, refused: bool) -> RefusalOutcome:
    """Score refusal behaviour for one question.

    Each field is None when it does not apply to this question's slice, so
    aggregate means are taken over the right denominators: false-refusal rate
    over answerable questions only, correct-refusal rate over unanswerable
    ones only.
    """
    if question.answerable:
        return RefusalOutcome(
            correct_refusal=None,
            false_refusal=refused,
            hallucination_risk=None,
        )
    return RefusalOutcome(
        correct_refusal=refused,
        false_refusal=None,
        hallucination_risk=not refused,
    )


# --------------------------------------------------------------------------
# Citations
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CitationValidity:
    """Deterministic citation checks -- no judge required.

    `fabricated` is the one that matters most: a citation pointing at a chunk
    that was never retrieved cannot possibly support the claim, and it is
    invisible to a human spot-checking the answer because the citation looks
    perfectly plausible. Phase 4 treats any fabrication as a hard failure.
    """

    total_citations: int
    resolvable: int
    fabricated: tuple[str, ...]
    uncited_answer: bool

    @property
    def fabrication_rate(self) -> float | None:
        if self.total_citations == 0:
            return None
        return len(self.fabricated) / self.total_citations

    @property
    def is_clean(self) -> bool:
        return not self.fabricated

    def to_dict(self) -> dict:
        return {
            "total_citations": self.total_citations,
            "resolvable": self.resolvable,
            "fabricated": list(self.fabricated),
            "fabrication_rate": self.fabrication_rate,
            "uncited_answer": self.uncited_answer,
        }


def citation_validity(response: SystemResponse) -> CitationValidity:
    """Check every cited chunk id against what was actually retrieved."""
    retrieved = set(response.retrieved_ids)
    cited: list[str] = [cid for c in response.citations for cid in c.chunk_ids]
    fabricated = sorted({cid for cid in cited if cid not in retrieved})
    substantive = bool(response.answer.strip()) and not response.refused
    return CitationValidity(
        total_citations=len(cited),
        resolvable=sum(1 for cid in cited if cid in retrieved),
        fabricated=tuple(fabricated),
        uncited_answer=substantive and not cited,
    )


# --------------------------------------------------------------------------
# Claim splitting
# --------------------------------------------------------------------------

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'\[])")
# Abbreviations and figures whose trailing period is not a sentence boundary.
# Filing text is dense with these -- "$1.2 million" and "Item 1A." both break
# a naive split, and a bad split silently inflates the claim count that
# groundedness is averaged over.
_PROTECT = re.compile(
    r"\b(?:Inc|Corp|Ltd|LLC|Co|No|Nos|vs|etc|approx|est|Mr|Ms|Dr|Jr|Sr|U\.S|Item|Note|Fig)\.",
    re.IGNORECASE,
)
_DECIMAL = re.compile(r"(?<=\d)\.(?=\d)")


def split_claims(answer: str) -> list[str]:
    """Split an answer into atomic claims for groundedness checking.

    Sentence-level, with protection for the abbreviations and decimals that
    appear constantly in financial text. This is a heuristic; the judge is
    free to further decompose a compound sentence, and the rubric says so.
    """
    if not answer.strip():
        return []

    placeholder = "\x00"
    protected = _PROTECT.sub(lambda m: m.group(0).replace(".", placeholder), answer)
    protected = _DECIMAL.sub(placeholder, protected)

    parts = _SENTENCE_END.split(protected)
    claims = [p.replace(placeholder, ".").strip() for p in parts]
    return [c for c in claims if c]


def groundedness_from_claims(supported: Sequence[bool]) -> float | None:
    """Fraction of claims entailed by the retrieved context.

    None for an empty answer -- a refusal has nothing to ground, and scoring
    it 1.0 would let a system that refuses everything top the groundedness
    table.
    """
    if not supported:
        return None
    return sum(1 for s in supported if s) / len(supported)
