"""Permission-aware retrieval.

The problem every company deploying RAG internally hits in week one: the
corpus contains documents not everyone may read, and the retriever does not
know that.

**Filtering must happen before scoring, never after.** Post-filtering is the
common implementation and it is wrong twice over:

  1. **It silently degrades quality.** The top-k was chosen from the full
     corpus; removing forbidden chunks afterwards leaves fewer than k, and
     the ones the user *was* allowed to see never got a chance to rank.
  2. **It is not a security boundary.** Anything that reaches the scorer has
     already been read. A post-filter narrows what is *returned*, which does
     nothing about a cache keyed on the unfiltered result, a log line
     carrying the chunk text, or a reranker that saw it.

The pre-filter here is a predicate handed to the retriever, so a forbidden
chunk is never scored, never cached, never logged, and never reaches the
model's context.

Deny-by-default: a document with no ACL is treated as restricted rather than
public. The opposite default fails open, and a mis-tagged document then leaks
silently.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..types import Chunk

__all__ = [
    "AccessPolicy",
    "Principal",
    "AccessDecision",
    "PUBLIC",
    "acl_of",
]

# The one label that means "anyone may read this". Documents carrying it are
# public; everything else requires an explicit role match.
PUBLIC = "public"

# Metadata key holding the roles permitted to read a document.
ACL_KEY = "allowed_roles"


@dataclass(frozen=True, slots=True)
class Principal:
    """Who is asking."""

    subject: str = "anonymous"
    roles: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def of(cls, subject: str = "anonymous", roles: Iterable[str] = ()) -> Principal:
        return cls(subject=subject, roles=frozenset(roles))

    @property
    def is_anonymous(self) -> bool:
        return not self.roles

    def to_dict(self) -> dict[str, Any]:
        return {"subject": self.subject, "roles": sorted(self.roles)}


def acl_of(chunk: Chunk) -> frozenset[str]:
    """Roles permitted to read this chunk.

    An absent or empty ACL yields the empty set, which grants nobody access.
    Treating "untagged" as "public" is the failure mode this exists to avoid:
    one document ingested before the tagging pipeline existed would be
    readable by everyone, and nothing would report it.
    """
    raw = chunk.metadata.get(ACL_KEY)
    if raw is None:
        return frozenset()
    if isinstance(raw, str):
        return frozenset({raw})
    return frozenset(str(r) for r in raw)


@dataclass(frozen=True, slots=True)
class AccessDecision:
    allowed: bool
    reason: str
    chunk_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"chunk_id": self.chunk_id, "allowed": self.allowed, "reason": self.reason}


@dataclass
class AccessPolicy:
    """Decides whether a principal may read a chunk, and builds pre-filters.

    `strict` is the deny-by-default switch. It is on by default and turning
    it off is a deliberate act with a comment attached, because the failure
    mode of the permissive setting is a silent leak rather than an error.
    """

    strict: bool = True
    name: str = "role_acl"

    @property
    def config(self) -> dict[str, Any]:
        return {"access_policy": self.name, "access_strict": self.strict}

    def decide(self, principal: Principal, chunk: Chunk) -> AccessDecision:
        acl = acl_of(chunk)

        if PUBLIC in acl:
            return AccessDecision(True, "public document", chunk.chunk_id)

        if not acl:
            if self.strict:
                return AccessDecision(
                    False,
                    "no ACL on the document; denied under deny-by-default",
                    chunk.chunk_id,
                )
            return AccessDecision(True, "no ACL and strict mode disabled", chunk.chunk_id)

        shared = principal.roles & acl
        if shared:
            return AccessDecision(True, f"role match: {sorted(shared)}", chunk.chunk_id)
        return AccessDecision(
            False,
            f"principal holds {sorted(principal.roles)}, document requires one of {sorted(acl)}",
            chunk.chunk_id,
        )

    def allows(self, principal: Principal, chunk: Chunk) -> bool:
        return self.decide(principal, chunk).allowed

    def predicate(self, principal: Principal) -> Callable[[Chunk], bool]:
        """A pre-filter for the retriever.

        Handed to `retrieve(where=...)` so forbidden chunks are excluded
        before scoring -- never scored, never cached, never logged, never in
        the model's context.
        """

        def allowed(chunk: Chunk) -> bool:
            return self.allows(principal, chunk)

        return allowed

    def audit(self, principal: Principal, chunks: Sequence[Chunk]) -> list[AccessDecision]:
        """Explain the decision for every chunk. For tests and incident review."""
        return [self.decide(principal, c) for c in chunks]

    def partition(
        self, principal: Principal, chunks: Sequence[Chunk]
    ) -> tuple[list[Chunk], list[Chunk]]:
        """Split into (readable, withheld)."""
        readable: list[Chunk] = []
        withheld: list[Chunk] = []
        for chunk in chunks:
            (readable if self.allows(principal, chunk) else withheld).append(chunk)
        return readable, withheld
