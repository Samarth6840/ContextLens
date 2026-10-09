# ContextLens — production worker image (Flask API + SaaS dashboard).
#
# NOTE: the vision/audio models (YOLO, DINOv2, Whisper, BEATs) are heavy and
# excluded from the build context (.dockerignore). Mount them at runtime:
#   -v ./weights:/app/weights \
#   -v ./BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt:/app/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
# The root *.pt files (yolov8x.pt, yolov8s-worldv2.pt) are also dropped by
# .dockerignore; either mount them too or let ultralytics auto-download them on
# first use (needs network + a writable working dir at runtime).
#
# Build:  docker build -t contextlens .
# Run:    docker run -it --rm -p 5000:5000 -v .... contextlens

# 3.12, not 3.13: paddlepaddle and openai-whisper's numba dependency lag on
# cp313 wheels. requirements-dev.txt is not installed here (runtime only).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CONTEXTLENS_PORT=5000

WORKDIR /app

# System libs required by OpenCV, audio (librosa/torchaudio) and ultralytics.
# git is required by requirements.txt's `clip@git+https://github.com/openai/CLIP.git`;
# python:3.13-slim ships without it, so the build fails without this.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# gunicorn is the production WSGI server (flask/gunicorn are pinned in
# requirements.txt; no separate install needed).

COPY . .

# Drop privileges: the API never needs root. Own /app so the app can write
# var/ (job store, logs) and brand_memory.json at runtime.
RUN useradd -m app && chown -R app:app /app
USER app

EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import os,urllib.request,sys; p=os.environ.get('PORT') or os.environ.get('CONTEXTLENS_PORT','5000'); sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{p}/api/health', timeout=5).status==200 else 1)"

CMD ["gunicorn", "wsgi:application", "-c", "gunicorn.conf.py"]
