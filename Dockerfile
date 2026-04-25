# =============================================================================
# Universal Data Ingestion Layer — Dockerfile
# =============================================================================
FROM python:3.11-slim

# System dependencies (for psycopg2-binary)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app/ ./app/
COPY sql/ ./sql/
COPY data/ ./data/

# Run as non-root user
RUN useradd -m appuser
USER appuser

CMD ["python", "-m", "app.main", "--help"]
