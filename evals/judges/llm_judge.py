"""LLM judge backed by the Anthropic API.

Design constraints, in order of importance:

1. **Deterministic.** Structured outputs constrain the response to a schema
   derived from the declared scale, so a score can never arrive as "2/3" or
   "mostly correct". Every call is cached on a hash that includes the rubric
   text, so a repeated eval returns byte-identical judgments.
2. **Auditable.** The judge's reasoning is stored with every score. When
   calibration shows disagreement, you read why the judge said what it said
   rather than guessing.
3. **Never silently wrong.** A malformed or errored response produces a
   `Judgment` with `error` set, which the runner records as "not judged".
   It never becomes a zero that gets averaged into a published number.

The judge model is `claude-opus-5` by default. That is the deliberate
choice: judge quality caps the trustworthiness of every judged metric in the
report, and a cheap judge that disagrees with you is worse than no judge.
Once calibration shows a smaller model holds its kappa on your rubrics, swap
it in with `--judge-model` and re-run the calibration to prove it — that
comparison is itself a good line in the eval report.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Any

from ..cache import ResponseCache, cache_key
from .base import SCALES, Judgment, Scale, load_rubric, rubric_version

__all__ = ["AnthropicJudge", "build_judge", "DEFAULT_JUDGE_MODEL"]

DEFAULT_JUDGE_MODEL = "claude-opus-5"
MAX_CONTEXT_PASSAGES = 20
MAX_PASSAGE_CHARS = 4000

_SYSTEM = (
    "You are a meticulous evaluation judge for a retrieval-augmented question "
    "answering system. You apply the rubric you are given exactly as written, "
    "including its edge cases, and you do not substitute your own standards. "
    "You return only the JSON object the rubric asks for."
)


def _score_schema(scale: Scale) -> dict[str, Any]:
    """JSON schema derived from the scale, so prompt and parser cannot drift."""
    return {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "One or two sentences justifying the score.",
            },
            "score": {
                "type": "integer",
                "enum": list(scale.categories),
                "description": "; ".join(
                    f"{cat} = {scale.labels[cat]}" for cat in scale.categories
                ),
            },
        },
        "required": ["reasoning", "score"],
        "additionalProperties": False,
    }


_CLAIMS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Atomic factual claims, in order of appearance.",
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}


def _format_context(context: Sequence[str]) -> str:
    """Render passages with stable delimiters and indices.

    Truncation is explicit and marked. A silently truncated passage would let
    the judge score "not stated" on a fact that was retrieved, which reads as
    a retrieval failure in the report and is not one.
    """
    if not context:
        return "(no passages were retrieved)"
    blocks = []
    for i, passage in enumerate(context[:MAX_CONTEXT_PASSAGES], start=1):
        text = passage.strip()
        if len(text) > MAX_PASSAGE_CHARS:
            text = text[:MAX_PASSAGE_CHARS] + "\n[... passage truncated for judging ...]"
        blocks.append(f'<passage index="{i}">\n{text}\n</passage>')
    if len(context) > MAX_CONTEXT_PASSAGES:
        blocks.append(
            f"<note>{len(context) - MAX_CONTEXT_PASSAGES} further passages omitted</note>"
        )
    return "\n".join(blocks)


class AnthropicJudge:
    """Thread-safe judge. The SDK client is safe to share across threads."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_JUDGE_MODEL,
        cache: ResponseCache | None = None,
        effort: str = "medium",
        max_tokens: int = 2000,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.cache = cache or ResponseCache(enabled=False)

        if client is not None:
            self._client = client
        else:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise ImportError(
                    "The judge needs the Anthropic SDK. Install it with:\n"
                    "    pip install -e '.[judge]'\n"
                    "Deterministic metrics (retrieval, refusal, citation validity) "
                    "run without it."
                ) from exc
            self._client = anthropic.Anthropic()

    # -- internals --------------------------------------------------------

    def _call(
        self, namespace: str, prompt: str, schema: dict[str, Any], version: str
    ) -> tuple[dict[str, Any], bool, str | None]:
        """Return (parsed_json, was_cached, error)."""
        key = cache_key(
            namespace,
            {
                "model": self.model,
                "effort": self.effort,
                "prompt": prompt,
                "schema": schema,
                "rubric_version": version,
            },
        )
        hit = self.cache.get(key)
        if hit is not None:
            return hit, True, None

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": schema},
                },
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as a per-item error
            return {}, False, f"{type(exc).__name__}: {exc}"

        if getattr(response, "stop_reason", None) == "refusal":
            return {}, False, "judge model refused the request"

        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            return {}, False, f"judge returned unparseable JSON: {exc}"

        self.cache.put(key, namespace, parsed)
        return parsed, False, None

    # -- Judge protocol ---------------------------------------------------

    def score(
        self,
        metric: str,
        *,
        question: str,
        answer: str,
        context: Sequence[str],
        gold_answer: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Judgment:
        if metric not in SCALES:
            raise KeyError(f"no scale declared for metric {metric!r}; add it to SCALES")
        scale = SCALES[metric]
        rubric = load_rubric(metric)
        version = rubric_version(metric)

        sections = [
            rubric,
            "\n---\n",
            f"<question>\n{question.strip()}\n</question>",
        ]
        if gold_answer is not None:
            sections.append(f"<reference_answer>\n{gold_answer.strip()}\n</reference_answer>")
        if extra and extra.get("claim"):
            sections.append(f"<claim>\n{str(extra['claim']).strip()}\n</claim>")

        sections.append(f"<retrieved_passages>\n{_format_context(context)}\n</retrieved_passages>")
        # claim_support judges one claim against the passages; showing it the
        # full answer invites it to grade the answer instead of the claim.
        if metric != "claim_support":
            sections.append(f"<system_answer>\n{answer.strip()}\n</system_answer>")

        prompt = "\n\n".join(sections)
        parsed, cached, error = self._call(f"judge.{metric}", prompt, _score_schema(scale), version)

        if error is not None:
            return Judgment(
                metric=metric,
                score=scale.categories[0],
                rubric_version=version,
                model=self.model,
                error=error,
            )

        try:
            score = scale.validate(int(parsed["score"]))
        except (KeyError, TypeError, ValueError) as exc:
            return Judgment(
                metric=metric,
                score=scale.categories[0],
                rubric_version=version,
                model=self.model,
                raw=parsed,
                error=f"invalid score in judge response: {exc}",
            )

        return Judgment(
            metric=metric,
            score=score,
            reasoning=str(parsed.get("reasoning", "")),
            rubric_version=version,
            model=self.model,
            cached=cached,
            raw=parsed,
        )

    def decompose_claims(self, answer: str) -> list[str]:
        """Split an answer into atomic claims.

        Falls back to the deterministic sentence splitter when the model call
        fails, so a groundedness run degrades rather than dying.
        """
        if not answer.strip():
            return []

        rubric = load_rubric("claim_decomposition")
        version = rubric_version("claim_decomposition")
        prompt = f"{rubric}\n\n---\n\n<answer>\n{answer.strip()}\n</answer>"

        parsed, _, error = self._call("judge.claim_decomposition", prompt, _CLAIMS_SCHEMA, version)
        if error is not None or "claims" not in parsed:
            from ..metrics.generation import split_claims

            return split_claims(answer)

        return [str(c).strip() for c in parsed["claims"] if str(c).strip()]


def _has_credentials() -> bool:
    """Whether the Anthropic SDK is likely to find a credential.

    An unset ANTHROPIC_API_KEY does not mean there are no credentials -- the
    SDK also resolves a profile written by `ant auth login`. Checking only the
    env var would send anyone using a profile down the no-judge path with no
    explanation.
    """
    from pathlib import Path

    if any(
        os.environ.get(var)
        for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE")
    ):
        return True

    config_home = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return (Path(config_home) / "anthropic").exists()


def build_judge(
    *,
    model: str = DEFAULT_JUDGE_MODEL,
    cache: ResponseCache | None = None,
    effort: str = "medium",
    enabled: bool = True,
) -> Any:
    """Construct a judge, or a `NullJudge` when judging is off or unavailable.

    Returning a NullJudge rather than raising keeps the deterministic half of
    the harness usable with no credentials -- which is the normal state in CI.
    """
    from .base import NullJudge

    if not enabled:
        return NullJudge()

    if not _has_credentials():
        return NullJudge()

    try:
        return AnthropicJudge(model=model, cache=cache, effort=effort)
    except ImportError:
        return NullJudge()
