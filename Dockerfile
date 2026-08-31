# ContextLens — production worker image (Flask API + SaaS dashboard).
#
# NOTE: the vision/audio models (YOLO, DINOv2, Whisper, BEATs) are heavy and
# excluded from the build context (.dockerignore). Mount them at runtime:
#   -v ./weights:/app/weights  -v ./BEATs_iter3_plus_AS2M.pt:/app/BEATs_iter3_plus_AS2M.pt
#
# Build:  docker build -t contextlens .
# Run:    docker run -it --rm -p 5000:5000 -v .... contextlens

FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    ADSCENE_PORT=5000

WORKDIR /app

# System libs required by OpenCV, audio (librosa/torchaudio) and ultralytics.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# gunicorn is the production WSGI server (add if not already pinned in reqs).
RUN pip install "gunicorn>=21.0.0"

COPY . .

EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5000/api/health', timeout=5).status==200 else 1)"

CMD ["gunicorn", "wsgi:application", "-c", "gunicorn.conf.py"]
