# Small benchmark client image (runners, summarize, mlflow_import). No model, no GPU.
# Runtime configuration (conf/app.conf, .env) is mounted at run time, never baked in.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/

RUN useradd --create-home --uid 1000 bench
USER bench

ENTRYPOINT ["python", "-m"]
CMD ["src.runner.summarize", "--help"]
