# ResearchOS application image: the FastAPI API + the LangGraph researcher.
# It can run the MCP toolbelt over stdio itself, or connect to the standalone
# MCP service (see docker-compose.yml) over streamable HTTP.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# System deps kept minimal; psycopg[binary] is not used, so libpq is needed.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq5 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8000

# Migrations are applied by the compose entrypoint before the API starts.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
