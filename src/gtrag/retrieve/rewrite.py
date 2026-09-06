"""Multi-turn query rewriting.

"And what drove that increase?" retrieves nothing on its own: it contains no
company, no period, and no subject. The retriever sees four stopwords and a
noun. Rewriting it against the conversation into a standalone question is the
difference between that query working and not working at all.

Two implementations behind one protocol:

  * `HeuristicRewriter` -- rule-based, offline, deterministic. Detects that a
    query is context-dependent and splices in the missing entities from the
    conversation. It is not clever, and it is honest about what it does.
  * `LLMRewriter` -- a model call, which handles the cases rules cannot.

The heuristic exists so multi-turn is testable and measurable without a
model, and so the LLM version's contribution can be reported as a delta
against something rather than against nothing.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

__all__ = ["QueryRewriter", "HeuristicRewriter", "LLMRewriter", "NullRewriter", "is_dependent"]

# Markers that a query cannot stand alone. Anaphora ("that", "it"),
# continuation ("and what about"), and ellipsis ("the year before?").
_ANAPHORA = re.compile(
    r"\b(?:that|those|these|this|it|its|they|them|their|he|she|him|her|his|there)\b",
    re.IGNORECASE,
)
_CONTINUATION = re.compile(
    r"^\s*(?:and|but|so|then|also|what about|how about|why|and what|and how)\b", re.IGNORECASE
)
_ELLIPTICAL = re.compile(
    r"\b(?:the (?:year|quarter|period) (?:before|after|prior)|prior year|same for|too|as well)\b",
    re.IGNORECASE,
)
_ENTITY = re.compile(r"\b[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]+)*\b")
_YEAR = re.compile(r"\b(?:fiscal\s+(?:year\s+)?|FY\s*)?((?:19|20)\d{2})\b", re.IGNORECASE)


def is_dependent(query: str) -> bool:
    """Whether a query appears to depend on prior turns.

    Deliberately over-inclusive. Rewriting a query that did not need it is
    usually harmless -- the added entities were already implied -- whereas
    failing to rewrite one that did means retrieving on four stopwords.
    """
    stripped = query.strip()
    if not stripped:
        return False
    if _CONTINUATION.search(stripped) or _ELLIPTICAL.search(stripped):
        return True
    if _ANAPHORA.search(stripped) and not _ENTITY.search(stripped[1:]):
        return True
    # Very short queries with no named entity cannot stand alone.
    return len(stripped.split()) <= 6 and not _ENTITY.search(stripped[1:])


class QueryRewriter(Protocol):
    name: str

    @property
    def config(self) -> dict[str, Any]: ...

    def rewrite(self, query: str, history: Sequence[tuple[str, str]]) -> str: ...


@dataclass
class NullRewriter:
    """No rewriting. The control, so the ablation has a baseline."""

    name: str = "none"

    @property
    def config(self) -> dict[str, Any]:
        return {"rewriter": self.name}

    def rewrite(self, query: str, history: Sequence[tuple[str, str]]) -> str:
        return query


@dataclass
class HeuristicRewriter:
    """Splice entities and periods from the conversation into the query.

    Takes the most recent turn's named entities and fiscal years and prefixes
    any that the query lacks. Crude, but it converts "and what drove that
    increase?" into something with a company and a year in it, which is the
    whole difficulty.

    It does not attempt to resolve *which* entity a pronoun refers to when
    several are in play -- that is where `LLMRewriter` earns its cost, and
    the ablation is how you find out whether it is worth paying.
    """

    lookback: int = 2
    name: str = "heuristic"

    @property
    def config(self) -> dict[str, Any]:
        return {"rewriter": self.name, "lookback": self.lookback}

    def rewrite(self, query: str, history: Sequence[tuple[str, str]]) -> str:
        if not history or not is_dependent(query):
            return query

        recent = list(history)[-self.lookback :]
        context = " ".join(f"{q} {a}" for q, a in recent)

        # Entities and years already in the query need no splicing. Skip the
        # first character when scanning the query so a capitalised sentence
        # opener ("What...") is not mistaken for a named entity.
        have_entities = {e.lower() for e in _ENTITY.findall(query[1:])}
        have_years = {m.group(1) for m in _YEAR.finditer(query)}

        entities: list[str] = []
        for candidate in _ENTITY.findall(context):
            lowered = candidate.lower()
            if lowered in have_entities or lowered in {e.lower() for e in entities}:
                continue
            if len(candidate) > 3:
                entities.append(candidate)

        years = [m.group(1) for m in _YEAR.finditer(context) if m.group(1) not in have_years]

        prefix_parts = entities[:2] + ([f"fiscal {years[0]}"] if years else [])
        if not prefix_parts:
            return query
        return f"{' '.join(prefix_parts)}: {query.strip()}"


@dataclass
class LLMRewriter:
    """Model-based rewriting into a standalone question.

    Falls back to the heuristic on any failure, so a multi-turn eval degrades
    rather than dying when the model is unavailable.
    """

    model: str = "claude-opus-5"
    name: str = "llm"
    max_tokens: int = 512
    client: Any = None
    _fallback: HeuristicRewriter | None = None

    SYSTEM = (
        "Rewrite the user's latest question into a standalone search query, "
        "resolving every pronoun and ellipsis against the conversation. Name the "
        "company, the fiscal period, and the metric explicitly. Do not answer the "
        "question and do not add facts that are not implied by the conversation. "
        "Return only the rewritten query."
    )

    def __post_init__(self) -> None:
        self._fallback = HeuristicRewriter()
        if self.client is None:
            try:
                import anthropic

                self.client = anthropic.Anthropic()
            except ImportError:
                self.client = None

    @property
    def config(self) -> dict[str, Any]:
        return {"rewriter": self.name, "rewriter_model": self.model}

    def rewrite(self, query: str, history: Sequence[tuple[str, str]]) -> str:
        assert self._fallback is not None
        if not history or not is_dependent(query):
            return query
        if self.client is None:
            return self._fallback.rewrite(query, history)

        turns: list[dict[str, Any]] = []
        for prior_q, prior_a in history:
            turns.append({"role": "user", "content": prior_q})
            turns.append({"role": "assistant", "content": prior_a})
        turns.append({"role": "user", "content": query})

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.SYSTEM,
                messages=turns,
                output_config={"effort": "low"},
            )
        except Exception:  # noqa: BLE001 - degrade, never fail the question
            return self._fallback.rewrite(query, history)

        if getattr(response, "stop_reason", None) == "refusal":
            return self._fallback.rewrite(query, history)

        text = next((b.text for b in response.content if b.type == "text"), "").strip()
        return text or self._fallback.rewrite(query, history)
