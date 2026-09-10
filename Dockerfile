# syntax=docker/dockerfile:1

# Multi-stage so the runtime image carries the package and its serving
# dependencies but not the build toolchain or the test suite.
FROM python:3.11-slim AS build

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
COPY evals ./evals

# Install into a virtualenv we can copy wholesale into the runtime stage.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir '.[serve]'


FROM python:3.11-slim AS runtime

# Run as a non-root user. The service reads a corpus and answers questions;
# it has no reason to hold root in a container that is exposed to a network.
RUN useradd --create-home --uid 10001 gtrag

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GTRAG_DOCS=/data/documents

COPY --from=build /opt/venv /opt/venv

WORKDIR /app
COPY --chown=gtrag:gtrag src ./src
COPY --chown=gtrag:gtrag evals ./evals
COPY --chown=gtrag:gtrag tests/fixtures ./tests/fixtures

# The corpus is a mounted volume, not baked into the image: it is large,
# regenerable, and changes on a different cadence from the code.
RUN mkdir -p /data/documents && chown -R gtrag:gtrag /data

USER gtrag
EXPOSE 8000

# Readiness, not liveness: the process can be up while the corpus is still
# loading, and routing traffic to it then produces empty answers.
HEALTHCHECK --interval=15s --timeout=3s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8000/readyz', timeout=2).status==200 else 1)"

# One worker per container. The pipeline is CPU-bound Python, so in-process
# threads contend on the GIL -- scale with replicas, not with --workers.
CMD ["uvicorn", "gtrag.serve.app:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "65"]
