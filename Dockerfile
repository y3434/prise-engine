# PRISE engine web app — container image
FROM python:3.13-slim

RUN useradd -m -u 1000 user
WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code
COPY --chown=user engine/ ./engine/
COPY --chown=user web/    ./web/

# Derived M0..M16 artifacts only. The attribution-gated CPAD bulk
# (data/cpad_raw, data/snapshots, data/pdb_cache, data/*.zip) is excluded
# here and in .dockerignore.
COPY --chown=user data/processed/ ./data/processed/

USER user
# PORT is injected by the host at runtime; 8000 is only a local fallback.
ENV PRISE_HOST=0.0.0.0 \
    PORT=10000 \
    PYTHONUNBUFFERED=1

EXPOSE 10000
CMD ["python", "web/server.py"]
