# flypaint: a fruit fly brain that listens to music and paints.
#
#   docker compose up          # first run downloads the 1.1 GB connectome into ./data
#   open http://localhost:8000
#
# The connectome files and every painting live in mounted volumes (./data, ./runs),
# so rebuilding or updating the image never re-downloads or loses anything.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FLYPAINT_DATA=/data \
    OMP_NUM_THREADS=4

# libsndfile decodes wav/flac/ogg/mp3; ffmpeg is a fallback decoder for exotic formats
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY flypaint ./flypaint
RUN pip install '.[web]'

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

VOLUME ["/data", "/app/runs"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s \
    CMD curl -fs http://127.0.0.1:8000/api/status || exit 1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["serve"]
