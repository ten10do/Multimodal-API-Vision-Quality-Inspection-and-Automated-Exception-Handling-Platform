FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY pyproject.toml ./
COPY app ./app
RUN pip install .
COPY alembic.ini ./
COPY alembic ./alembic

RUN useradd --create-home --uid 10001 appuser && \
    mkdir -p /data/uploads && chown -R appuser:appuser /app /data
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
