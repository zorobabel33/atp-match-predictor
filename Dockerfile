FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

WORKDIR /app

# Cache serving dependencies separately from application changes. Never fetch a model at build time.
COPY app/requirements.txt ./app/requirements.txt
RUN python -m pip install --no-compile -r app/requirements.txt \
    && useradd --create-home --user-group --uid 10001 app \
    && chown app:app /app

COPY app/ ./app/
COPY ml/__init__.py ml/contract.py ml/model.py ml/registry.py ml/drift.py ml/policy.py ./ml/

USER 10001:10001
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.getenv('PORT', '8000') + '/ready', timeout=3)"

# app.main reads PORT from the environment and runs uvicorn on 0.0.0.0.
CMD ["python", "-m", "app.main"]
