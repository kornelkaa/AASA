FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU-only torch keeps the image much smaller than the default CUDA build
RUN pip install --no-cache-dir torch==2.6.0+cpu torchvision==0.21.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "aasa.server:app", "--host", "0.0.0.0", "--port", "8000"]