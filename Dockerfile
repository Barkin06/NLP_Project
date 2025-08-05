FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# HF cache için
ENV TRANSFORMERS_CACHE=/app/cache

# Başlatma scripti
RUN chmod +x start.sh

CMD ["bash", "start.sh"]
