"""Corpus CLI: `python -m gtrag.cli <command>`.

    ingest    fetch and parse filings from EDGAR into a document store
    index     chunk and embed the document store into a vector index
    query     ask the baseline system a question
    inspect   show what was parsed out of one document

`ingest` is the only command that touches the network. It is separate from
`index` on purpose: you will re-chunk and re-embed the same corpus many times
during Phase 3, and each of those must not re-download anything.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from .baseline import build_baseline
from .chunking.base import FixedTokenChunker
from .generate.generator import build_generator
from .index.embed import HashingEmbedder, SentenceTransformerEmbedder
from .index.store import VectorIndex
from .ingest.document import Document
from .ingest.edgar import CachingFetcher, EdgarClient, EdgarError, HttpFetcher
from .ingest.parse import parse_filing

DEFAULT_DOCS = "corpus/documents"
DEFAULT_INDEX = "corpus/index.jsonl"
DEFAULT_HTTP_CACHE = "corpus/.http-cache"

# A default corpus: mixed sectors so vocabulary varies and near-duplicate
# language across peers becomes a real retrieval problem rather than a
# theoretical one.
DEFAULT_CIKS: tuple[tuple[int, str], ...] = (
    (320193, "Apple"),
    (789019, "Microsoft"),
    (1018724, "Amazon"),
    (104169, "Walmart"),
    (34088, "Exxon Mobil"),
    (19617, "JPMorgan Chase"),
    (93410, "Chevron"),
    (66740, "3M"),
    (1467858, "General Motors"),
    (97745, "Thermo Fisher"),
)


def _embedder(name: str):
    if name == "hashing":
        return HashingEmbedder()
    if name == "sentence-transformers":
        return SentenceTransformerEmbedder()
    raise ValueError(f"unknown embedder {name!r} (have: hashing, sentence-transformers)")


# --------------------------------------------------------------------------


def cmd_ingest(args: argparse.Namespace) -> int:
    user_agent = os.environ.get("GTRAG_SEC_USER_AGENT", "")
    if not user_agent:
        print(
            "GTRAG_SEC_USER_AGENT is not set.\n\n"
            "SEC requires a User-Agent identifying you, including a contact email:\n"
            '    export GTRAG_SEC_USER_AGENT="groundtruth-rag research you@example.com"\n\n'
            "Requests without one are refused with a 403, and repeat offenders are\n"
            "IP-banned. This is a hard requirement, not a courtesy.",
            file=sys.stderr,
        )
        return 1

    try:
        fetcher = CachingFetcher(
            inner=HttpFetcher(user_agent=user_agent), cache_dir=Path(args.http_cache)
        )
    except EdgarError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    client = EdgarClient(fetcher)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ciks = [(int(c), "") for c in args.cik] if args.cik else list(DEFAULT_CIKS)[: args.companies]

    written = 0
    failed: list[str] = []
    for cik, label in ciks:
        try:
            filings = client.list_filings(cik, form_types=tuple(args.forms), limit=args.years)
        except EdgarError as exc:
            failed.append(f"CIK {cik}: {exc}")
            continue

        for ref in filings:
            target = out_dir / f"{ref.cik}-{ref.accession}.json"
            if target.exists() and not args.force:
                continue
            try:
                html = client.fetch_document(ref)
            except EdgarError as exc:
                failed.append(f"{ref.accession}: {exc}")
                continue

            document = parse_filing(
                html,
                metadata={
                    "company": ref.company or label,
                    "cik": ref.cik,
                    "ticker": "",
                    "form_type": ref.form_type,
                    "fiscal_year": ref.fiscal_year,
                    "filing_date": ref.filing_date,
                    "accession": ref.accession,
                },
                source_url=ref.url,
            )
            target.write_text(json.dumps(document.to_dict(), ensure_ascii=False), encoding="utf-8")
            written += 1
            if not args.quiet:
                print(
                    f"  {ref.company or label} {ref.form_type} FY{ref.fiscal_year}: "
                    f"{len(document.text):,} chars, {len(document.sections)} sections, "
                    f"{len(document.tables)} tables"
                )

    print(f"\ningested {written} filing(s) into {out_dir}")
    print(f"http cache: {fetcher.hits} hits, {fetcher.misses} fetches")
    if failed:
        print(f"\n{len(failed)} failure(s):", file=sys.stderr)
        for message in failed[:10]:
            print(f"  - {message}", file=sys.stderr)
    return 0 if written or not failed else 1


def cmd_index(args: argparse.Namespace) -> int:
    docs_dir = Path(args.docs)
    files = sorted(docs_dir.glob("*.json"))
    if not files:
        print(f"no documents in {docs_dir} -- run `make ingest` first", file=sys.stderr)
        return 1

    documents = [Document.from_dict(json.loads(f.read_text(encoding="utf-8"))) for f in files]
    chunker = FixedTokenChunker(chunk_tokens=args.chunk_tokens, overlap_tokens=args.overlap_tokens)
    index = VectorIndex(embedder=_embedder(args.embedder))

    total = 0
    for document in documents:
        chunks = chunker.chunk(document)
        total += index.add(chunks)
        if not args.quiet:
            company = document.metadata.get("company", document.doc_id)
            year = document.metadata.get("fiscal_year", "")
            print(f"  {company} FY{year}: {len(chunks)} chunks")

    path = index.save(args.out)
    print(f"\nindexed {total} chunks from {len(documents)} document(s) -> {path}")
    print(f"chunker:  {chunker.config}")
    print(f"embedder: {index.embedder.config}")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    index = VectorIndex.load(args.index, _embedder(args.embedder))
    system = build_baseline(
        index=index,
        generator=build_generator(prefer_model=not args.no_model),
        top_k=args.top_k,
    )
    response = system.answer(args.question)

    print(f"Q: {args.question}\n")
    if response.refused:
        print("A: [refused] the retrieved passages do not support an answer\n")
    else:
        print(f"A: {response.answer}\n")
    if response.error:
        print(f"error: {response.error}\n", file=sys.stderr)

    print("retrieved:")
    for chunk in sorted(response.retrieved, key=lambda c: c.rank):
        company = chunk.metadata.get("company", "?")
        year = chunk.metadata.get("fiscal_year", "?")
        section = chunk.metadata.get("section", "")
        preview = " ".join(chunk.text.split())[:110]
        print(f"  {chunk.rank}. [{chunk.score:.4f}] {company} FY{year} {section}")
        print(f"     {preview}...")

    timings = ", ".join(f"{k}={v:.1f}ms" for k, v in response.timings.items())
    print(f"\ntimings: {timings}")
    if response.usage.cost_usd:
        print(f"cost: ${response.usage.cost_usd:.5f}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    document = Document.from_dict(json.loads(Path(args.path).read_text(encoding="utf-8")))
    print(f"doc_id:   {document.doc_id}")
    print(f"source:   {document.source_url}")
    print(f"metadata: {document.metadata}")
    print(f"length:   {len(document.text):,} chars")
    print(f"\nsections ({len(document.sections)}):")
    for section in document.sections:
        body = document.slice(section.span)
        print(
            f"  {section.name:<36} {len(body):>8,} chars  [{section.span.start}:{section.span.end}]"
        )
    print(f"\ntables ({len(document.tables)}):")
    for i, table in enumerate(document.tables[: args.max_tables]):
        print(
            f"  {i + 1}. {table.n_rows} rows x {table.n_cols} cols  [{table.span.start}:{table.span.end}]"
        )
    return 0


# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gtrag", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    i = sub.add_parser("ingest", help="fetch and parse filings from EDGAR")
    i.add_argument("--out", default=DEFAULT_DOCS)
    i.add_argument("--http-cache", default=DEFAULT_HTTP_CACHE)
    i.add_argument("--cik", nargs="*", default=None, help="explicit CIKs (default: a built-in set)")
    i.add_argument("--companies", type=int, default=10)
    i.add_argument("--years", type=int, default=3, help="filings per company, most recent first")
    i.add_argument("--forms", nargs="*", default=["10-K"])
    i.add_argument("--force", action="store_true", help="re-parse filings already on disk")
    i.add_argument("--quiet", action="store_true")
    i.set_defaults(func=cmd_ingest)

    x = sub.add_parser("index", help="chunk and embed the document store")
    x.add_argument("--docs", default=DEFAULT_DOCS)
    x.add_argument("--out", default=DEFAULT_INDEX)
    x.add_argument("--chunk-tokens", type=int, default=512)
    x.add_argument("--overlap-tokens", type=int, default=50)
    x.add_argument("--embedder", default="hashing", choices=["hashing", "sentence-transformers"])
    x.add_argument("--quiet", action="store_true")
    x.set_defaults(func=cmd_index)

    q = sub.add_parser("query", help="ask the baseline system a question")
    q.add_argument("question")
    q.add_argument("--index", default=DEFAULT_INDEX)
    q.add_argument("--embedder", default="hashing", choices=["hashing", "sentence-transformers"])
    q.add_argument("--top-k", type=int, default=5)
    q.add_argument("--no-model", action="store_true", help="extractive generation only")
    q.set_defaults(func=cmd_query)

    n = sub.add_parser("inspect", help="show what was parsed out of one document")
    n.add_argument("path")
    n.add_argument("--max-tables", type=int, default=10)
    n.set_defaults(func=cmd_inspect)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (EdgarError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
