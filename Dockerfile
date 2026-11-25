# Aplikasi SignBridge AI untuk hosting (Render, dll.). Versi Python sama dengan yang Anda pakai (3.12).
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# Library sistem yang dibutuhkan OpenCV/MediaPipe (tanpa ini muncul error "libGL.so.1 not found")
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgles2 libegl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-deploy.txt .
RUN pip install -r requirements-deploy.txt

COPY . .

# 1 worker: model & MediaPipe dimuat sekali (hemat memori). Render mengisi $PORT otomatis.
CMD gunicorn app:app --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 2 --timeout 120
