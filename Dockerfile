# Serves the trained captioning model as a JSON HTTP API (api.py).
# The Streamlit demo (app.py) has its own deploy paths -- see
# DEPLOY_STREAMLIT.md / DEPLOY_SPACES.md -- because Streamlit Community
# Cloud and Hugging Face Spaces both build from source directly and don't
# need a container; this Dockerfile is for running the API anywhere a
# generic container is expected (a VM, ECS/Cloud Run, a colleague's
# laptop) rather than a platform-specific deploy.
#
# Build:  docker build -t caption-generator-api .
# Run:    docker run -p 8000:8000 caption-generator-api
# The checkpoint (~380MB) is not baked into the image -- api.py downloads
# it on first request if CAPTION_CHECKPOINT_PATH doesn't already exist,
# same as app.py does for the Streamlit deploy. Mount a volume with the
# checkpoint pre-placed to skip that download on every fresh container.

FROM python:3.11-slim

WORKDIR /app

# torchvision needs libjpeg/libpng at runtime for image decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir -e .

COPY api.py ./

EXPOSE 8000

# 1 worker: the model is loaded once into process memory at startup
# (see api.py's lifespan handler); multiple workers would each load
# their own copy, multiplying memory use for no throughput benefit on
# CPU-bound single-image requests at this model's size.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
