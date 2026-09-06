"""EDGAR client.

Two compliance details that are not optional and that most scrapers get
wrong -- SEC blocks on both:

  * **User-Agent must identify you** with a real contact address. A default
    `python-requests/2.x` gets a 403, and repeated offenders get the IP
    banned. `GTRAG_SEC_USER_AGENT` is required rather than defaulted, so the
    failure is a clear error at startup instead of a confusing 403 later.
  * **Rate limit is 10 requests/second.** Enforced here with a token bucket
    that is shared across threads, because a ThreadPoolExecutor over a
    per-request `sleep` does not actually limit anything.

Network access is injected through the `Fetcher` protocol. Tests use a
recorded fetcher over on-disk fixtures, so the whole parsing pipeline is
exercised with no network -- which is also what makes CI deterministic.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

__all__ = [
    "Fetcher",
    "HttpFetcher",
    "CachingFetcher",
    "RecordedFetcher",
    "RateLimiter",
    "EdgarClient",
    "FilingRef",
    "EdgarError",
]

SEC_RATE_LIMIT_PER_SEC = 10.0
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{document}"


class EdgarError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------


class RateLimiter:
    """Thread-safe token bucket.

    A per-call `sleep` does not bound concurrent throughput -- N threads each
    sleeping 0.1s still issue N requests at once. Serialising the *grant* of
    permission is what actually enforces the limit.
    """

    def __init__(self, rate_per_sec: float = SEC_RATE_LIMIT_PER_SEC) -> None:
        if rate_per_sec <= 0:
            raise ValueError("rate_per_sec must be positive")
        self._min_interval = 1.0 / rate_per_sec
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self) -> float:
        """Block until a request may be issued. Returns the seconds waited."""
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next_allowed - now)
            self._next_allowed = max(now, self._next_allowed) + self._min_interval
        if wait > 0:
            time.sleep(wait)
        return wait


# --------------------------------------------------------------------------
# Fetchers
# --------------------------------------------------------------------------


class Fetcher(Protocol):
    def get(self, url: str) -> bytes: ...


@dataclass
class HttpFetcher:
    """Real network access, rate-limited and retrying on transient failures."""

    user_agent: str
    limiter: RateLimiter | None = None
    timeout: float = 30.0
    max_retries: int = 4

    def __post_init__(self) -> None:
        if not self.user_agent or "@" not in self.user_agent:
            raise EdgarError(
                "SEC requires a User-Agent identifying you, including a contact "
                "email address, e.g. 'groundtruth-rag research you@example.com'. "
                "Set GTRAG_SEC_USER_AGENT. Requests without one are refused with "
                "a 403 and repeat offenders are IP-banned."
            )
        if self.limiter is None:
            self.limiter = RateLimiter()

    def get(self, url: str) -> bytes:
        assert self.limiter is not None
        last: Exception | None = None
        for attempt in range(self.max_retries):
            self.limiter.acquire()
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": self.user_agent,
                    "Accept-Encoding": "gzip, deflate",
                    "Host": urllib.parse.urlsplit(url).netloc,
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = response.read()
                    if response.headers.get("Content-Encoding") == "gzip":
                        import gzip

                        body = gzip.decompress(body)
                    return body
            except urllib.error.HTTPError as exc:
                # 4xx other than 429 will not succeed on retry.
                if exc.code == 429 or exc.code >= 500:
                    last = exc
                else:
                    raise EdgarError(f"{exc.code} fetching {url}: {exc.reason}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                last = exc

            backoff = 2.0**attempt
            time.sleep(backoff)

        raise EdgarError(f"failed to fetch {url} after {self.max_retries} attempts: {last}")


@dataclass
class CachingFetcher:
    """Disk cache in front of another fetcher.

    Re-ingestion is a normal operation -- you will re-parse the same filings
    many times while developing the section parser -- and re-downloading each
    time is both slow and rude to SEC's servers.
    """

    inner: Fetcher
    cache_dir: Path

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _path(self, url: str) -> Path:
        import hashlib

        return self.cache_dir / f"{hashlib.sha256(url.encode()).hexdigest()[:24]}.bin"

    def get(self, url: str) -> bytes:
        path = self._path(url)
        if path.exists():
            self.hits += 1
            return path.read_bytes()
        body = self.inner.get(url)
        path.write_bytes(body)
        self.misses += 1
        return body


@dataclass
class RecordedFetcher:
    """Serves responses from an on-disk manifest. No network.

    Used by the test suite so the parser is exercised against real filing
    HTML deterministically. `strict` makes an unrecorded URL an error rather
    than a silent empty response -- a test that quietly parses nothing passes
    for the wrong reason.
    """

    responses: dict[str, bytes]
    strict: bool = True

    def get(self, url: str) -> bytes:
        if url in self.responses:
            return self.responses[url]
        if self.strict:
            raise EdgarError(
                f"no recorded response for {url}\nrecorded: {sorted(self.responses)[:5]}"
            )
        return b""

    @classmethod
    def from_dir(cls, directory: str | Path, strict: bool = True) -> RecordedFetcher:
        """Load a directory of `<name>.url` / `<name>.body` pairs."""
        d = Path(directory)
        responses: dict[str, bytes] = {}
        for url_file in sorted(d.glob("*.url")):
            body_file = url_file.with_suffix(".body")
            if body_file.exists():
                responses[url_file.read_text(encoding="utf-8").strip()] = body_file.read_bytes()
        return cls(responses=responses, strict=strict)


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FilingRef:
    """A pointer to one filing, before its document is fetched."""

    cik: int
    company: str
    form_type: str
    accession: str
    filing_date: str
    report_date: str
    primary_document: str

    @property
    def fiscal_year(self) -> int:
        """Year of the reporting period, not the filing date.

        These differ for almost every 10-K -- a fiscal 2024 report is filed
        in late 2024 or in 2025 -- and using the filing date silently
        mislabels the year on a corpus whose whole point includes temporal
        disambiguation.
        """
        source = self.report_date or self.filing_date
        return int(source[:4]) if source else 0

    @property
    def url(self) -> str:
        return ARCHIVE_URL.format(
            cik=self.cik,
            accession_nodash=self.accession.replace("-", ""),
            document=self.primary_document,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cik": self.cik,
            "company": self.company,
            "form_type": self.form_type,
            "accession": self.accession,
            "filing_date": self.filing_date,
            "report_date": self.report_date,
            "fiscal_year": self.fiscal_year,
            "url": self.url,
        }


class EdgarClient:
    """Lists and fetches filings."""

    def __init__(self, fetcher: Fetcher) -> None:
        self.fetcher = fetcher

    def list_filings(
        self,
        cik: int,
        *,
        form_types: tuple[str, ...] = ("10-K",),
        limit: int | None = None,
    ) -> list[FilingRef]:
        """List a company's filings, most recent first.

        Parses the column-oriented `recent` block of the submissions JSON,
        where each field is a parallel array rather than a list of records.
        """
        raw = self.fetcher.get(SUBMISSIONS_URL.format(cik=cik))
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise EdgarError(f"submissions JSON for CIK {cik} is not valid JSON: {exc}") from exc

        company = data.get("name", "")
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        if not forms:
            return []

        wanted = {f.upper() for f in form_types}
        accessions = recent.get("accessionNumber", [])
        filing_dates = recent.get("filingDate", [])
        report_dates = recent.get("reportDate", [])
        documents = recent.get("primaryDocument", [])

        out: list[FilingRef] = []
        for i, form in enumerate(forms):
            if form.upper() not in wanted:
                continue
            out.append(
                FilingRef(
                    cik=cik,
                    company=company,
                    form_type=form,
                    accession=accessions[i] if i < len(accessions) else "",
                    filing_date=filing_dates[i] if i < len(filing_dates) else "",
                    report_date=report_dates[i] if i < len(report_dates) else "",
                    primary_document=documents[i] if i < len(documents) else "",
                )
            )
            if limit is not None and len(out) >= limit:
                break
        return out

    def fetch_document(self, ref: FilingRef) -> str:
        """Fetch a filing's primary document as text."""
        if not ref.primary_document:
            raise EdgarError(f"filing {ref.accession} has no primary document listed")
        body = self.fetcher.get(ref.url)
        # Filings are largely latin-1/windows-1252 in practice; replace rather
        # than raise, since one stray byte should not lose an entire filing.
        return body.decode("utf-8", errors="replace")
