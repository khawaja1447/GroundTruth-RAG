#!/usr/bin/env python3
"""Measure the security posture: injection resistance and access control.

    python scripts/security_report.py

Two measurements, both before/after so the mitigations report a delta rather
than a claim:

  1. **Indirect prompt injection.** Contaminate a real passage with each
     payload, run the query, and check whether the attacker's canary reaches
     the output. The unmitigated rate is measured first -- a mitigation with
     no before number is an assertion.

  2. **Permission-aware retrieval.** Ask questions whose answers live in
     restricted documents, as a principal without the role, and verify the
     content appears neither in retrieval nor in the answer. A leak that only
     shows up in the answer would still be a leak in the cache and the logs,
     so both are checked.

Exits non-zero if any restricted content leaks. Access control is a gate, not
a metric: a 95% pass rate is a failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from gtrag.ablation import AblationConfig, build_system  # noqa: E402
from gtrag.chunking.strategies import StructureAwareChunker  # noqa: E402
from gtrag.grounded import GroundedRagSystem  # noqa: E402
from gtrag.security.access import PUBLIC, AccessPolicy, Principal  # noqa: E402
from gtrag.security.injection import (  # noqa: E402
    INJECTION_CORPUS,
    AttackReport,
    AttackResult,
    Sanitizer,
    attack_success,
    detect_injection,
)
from gtrag.types import Chunk  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from run_sweep import load_documents  # noqa: E402

QUESTION = "What was Northwind's total net revenue in fiscal 2024?"


def build(documents, *, sanitizer: Sanitizer | None, access: AccessPolicy | None):
    system = build_system(
        AblationConfig(label="security", chunker="structure_aware", bm25=True), documents
    )
    system.sanitizer = sanitizer
    system.access = access
    return system


def contaminate(system: GroundedRagSystem, attempt) -> GroundedRagSystem:
    """Splice the payload into the chunk the query would otherwise retrieve.

    Contaminating the *winning* passage is the realistic case: an attacker
    who controls a document wants the payload to arrive with the evidence,
    not instead of it.
    """
    chunks = list(system.chunks)
    target = max(
        range(len(chunks)),
        key=lambda i: ("Total net revenue" in chunks[i].chunk.text, -i),
    )
    original = chunks[target]
    poisoned = replace(
        original,
        chunk=Chunk(
            chunk_id=original.chunk.chunk_id,
            text=attempt.contaminate(original.chunk.text),
            doc_id=original.chunk.doc_id,
            metadata=dict(original.chunk.metadata),
        ),
    )
    chunks[target] = poisoned

    rebuilt = build_system(
        AblationConfig(label="security", chunker="structure_aware", bm25=True), []
    )
    # Rebuild the retriever over the poisoned chunk set.
    from gtrag.index.embed import HashingEmbedder
    from gtrag.index.store import VectorIndex
    from gtrag.retrieve.retrievers import BM25Retriever, DenseRetriever, HybridRetriever

    index = VectorIndex(embedder=HashingEmbedder())
    index.add(chunks)
    rebuilt.retriever = HybridRetriever(
        retrievers=[DenseRetriever(index=index), BM25Retriever(chunks=chunks)]
    )
    rebuilt.chunks = chunks
    rebuilt.sanitizer = system.sanitizer
    rebuilt.access = system.access
    return rebuilt


def run_injection(documents) -> tuple[AttackReport, AttackReport, list[dict]]:
    """Measure three separable things, because they have different answers.

    * **Behavioural success** -- did the payload change the output? This is
      the number that matters, and it is only meaningful against a model
      that follows instructions. The extractive generator cannot be
      instructed at all, so it is immune by construction rather than by
      defence, and reporting 0% here would be a false reassurance.
    * **Delivery** -- did the payload reach the model's context? It does,
      by design: the sanitizer does not delete passages that match a
      pattern, because an attacker who learns the pattern could then use it
      to hide legitimate evidence.
    * **Structural neutralisation** -- were the parts that are unsafe
      *regardless* of the model removed? Invisible characters and forged
      delimiters carry no legitimate content, so removing them cannot
      damage a filing, and they are what let a payload hide from a human
      reviewer or appear to end the passage envelope.
    """
    reports = []
    structural: list[dict] = []

    for label, sanitizer in (("unmitigated", None), ("mitigated", Sanitizer())):
        base = build(documents, sanitizer=sanitizer, access=None)
        results = []
        for attempt in INJECTION_CORPUS:
            system = contaminate(base, attempt)
            response = system.answer(QUESTION)
            context = " ".join(c.text for c in response.retrieved)

            results.append(
                AttackResult(
                    attempt_id=attempt.id,
                    technique=attempt.technique,
                    # Behavioural only: did the canary reach the user?
                    succeeded=attack_success(response.answer, attempt),
                    detected=bool(detect_injection(attempt.payload)),
                    answer_excerpt=" ".join(response.answer.split())[:80],
                )
            )

            if sanitizer is not None:
                raw = attempt.payload
                cleaned = sanitizer.clean(raw)
                structural.append(
                    {
                        "attempt_id": attempt.id,
                        "technique": attempt.technique,
                        "reached_context": attempt.canary.lower() in context.lower(),
                        "invisible_chars_removed": _invisible_count(raw)
                        - _invisible_count(cleaned),
                        "boundaries_defanged": _boundary_count(raw) - _boundary_count(cleaned),
                        "detected": bool(detect_injection(raw)),
                    }
                )
        reports.append(AttackReport(results=tuple(results), label=label))
    return reports[0], reports[1], structural


def _invisible_count(text: str) -> int:
    import re

    return len(re.findall(r"[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]", text))


def _boundary_count(text: str) -> int:
    import re

    return len(re.findall(r"</?(?:passage|system|instructions?)>", text, re.IGNORECASE))


def run_access(documents) -> tuple[list[dict], bool]:
    """Tag one document restricted, then probe it as an unauthorised caller."""
    chunker = StructureAwareChunker()
    chunks = []
    for i, document in enumerate(documents):
        for spanned in chunker.chunk(document):
            metadata = dict(spanned.chunk.metadata)
            # Northwind FY2024 (index 0) is restricted; the rest are public.
            metadata["allowed_roles"] = ["finance"] if i == 0 else [PUBLIC]
            chunks.append(
                replace(
                    spanned,
                    chunk=Chunk(
                        chunk_id=spanned.chunk.chunk_id,
                        text=spanned.chunk.text,
                        doc_id=spanned.chunk.doc_id,
                        metadata=metadata,
                    ),
                )
            )

    from gtrag.index.embed import HashingEmbedder
    from gtrag.index.store import VectorIndex
    from gtrag.retrieve.retrievers import BM25Retriever, DenseRetriever, HybridRetriever

    index = VectorIndex(embedder=HashingEmbedder())
    index.add(chunks)
    system = build_system(AblationConfig(label="acl", chunker="structure_aware", bm25=True), [])
    system.retriever = HybridRetriever(
        retrievers=[DenseRetriever(index=index), BM25Retriever(chunks=chunks)]
    )
    system.chunks = chunks
    system.access = AccessPolicy(strict=True)

    secret = "4,218"  # the restricted figure
    probes = [
        ("finance analyst", Principal.of("alice", ["finance"]), True),
        ("intern (no role)", Principal.of("bob", ["intern"]), False),
        ("anonymous", Principal.of("anon", []), False),
    ]

    rows: list[dict] = []
    leaked = False
    for label, principal, should_see in probes:
        response = system.answer(QUESTION, principal=principal)
        in_passages = any(secret in c.text for c in response.retrieved)
        in_answer = secret in response.answer
        ok = (in_passages == should_see) and (in_answer == should_see or not should_see)
        if not should_see and (in_passages or in_answer):
            leaked = True
        rows.append(
            {
                "principal": label,
                "roles": sorted(principal.roles),
                "should_see": should_see,
                "in_passages": in_passages,
                "in_answer": in_answer,
                "n_passages": len(response.retrieved),
                "pass": ok,
            }
        )
    return rows, leaked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    documents = load_documents(args.docs)
    print()

    print("=" * 72)
    print("INDIRECT PROMPT INJECTION")
    print("=" * 72)
    before, after, structural = run_injection(documents)

    print(f"\n{'technique':<24} {'detected':>9} {'reached ctx':>12} {'behavioural':>12}")
    print("-" * 62)
    by_id = {row["attempt_id"]: row for row in structural}
    for b in before.results:
        row = by_id.get(b.attempt_id, {})
        print(
            f"{b.technique:<24} {'yes' if b.detected else 'NO':>9} "
            f"{'yes' if row.get('reached_context') else 'no':>12} "
            f"{'BREACH' if b.succeeded else 'no':>12}"
        )
    print("-" * 62)
    print(f"{'detection rate':<24} {before.detection_rate:>9.0%}")
    print(f"{'behavioural success':<24} {'':>9} {'':>12} {after.success_rate:>11.0%}")

    stripped = sum(r["invisible_chars_removed"] for r in structural)
    defanged = sum(r["boundaries_defanged"] for r in structural)
    print(
        f"\nStructural neutralisation: {stripped} invisible character(s) stripped, "
        f"{defanged} forged delimiter(s) defanged."
    )
    print(
        "Behavioural success is 0% because the extractive generator cannot follow\n"
        "instructions at all -- it is immune by construction, not by defence.\n"
        "That number is only meaningful against a real model; see docs/security.md."
    )

    print("\n" + "=" * 72)
    print("PERMISSION-AWARE RETRIEVAL")
    print("=" * 72)
    rows, leaked = run_access(documents)
    print(f"\n{'principal':<20} {'may see':>8} {'in passages':>12} {'in answer':>10} {'':>6}")
    print("-" * 60)
    for row in rows:
        print(
            f"{row['principal']:<20} {'yes' if row['should_see'] else 'no':>8} "
            f"{'yes' if row['in_passages'] else 'no':>12} "
            f"{'yes' if row['in_answer'] else 'no':>10} "
            f"{'PASS' if row['pass'] else 'FAIL':>6}"
        )

    print()
    if leaked:
        print("LEAK: restricted content reached an unauthorised principal.")
    else:
        print("No leakage: restricted content never reached an unauthorised principal.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(
                {
                    "injection": {"before": before.to_dict(), "after": after.to_dict()},
                    "access_control": {"probes": rows, "leaked": leaked},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwritten to {args.out}")

    # Access control is a gate, not a metric: a 95% pass rate is a failure.
    return 1 if leaked else 0


if __name__ == "__main__":
    raise SystemExit(main())
