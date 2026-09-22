# syntax=docker/dockerfile:1
# Road Pothole Detection & Severity Analysis - production image.
# Runs the classical detector by default (no torch needed, image stays small).
# To enable the YOLOv8 backend, build with:
#   docker build --build-arg WITH_TORCH=1 .
FROM python:3.12-slim

ARG WITH_TORCH=0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000

WORKDIR /app

# Install lightweight deps first for layer caching
COPY requirements-render.txt ./
RUN pip install --no-cache-dir -r requirements-render.txt

# Optional: add the YOLOv8 backend (large download, needs CPU build)
RUN if [ "$WITH_TORCH" = "1" ]; then \
        pip install --no-cache-dir torch ultralytics \
        --index-url https://download.pytorch.org/whl/cpu; \
    fi

COPY . .

EXPOSE 5000

# waitress is cross-platform (Linux/macOS/Windows) and production-safe
CMD ["sh", "-c", "waitress-serve --host=0.0.0.0 --port=${PORT:-5000} api.app:app"]