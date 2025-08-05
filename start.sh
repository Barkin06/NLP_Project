#!/bin/bash
set -e

MODEL_DIR="./deberta_intent_model"

if [ ! -d "$MODEL_DIR" ]; then
    echo "=== Model bulunamadı, training başlatılıyor ==="
    python train.py
else
    echo "=== Model mevcut, training atlandı ==="
fi

echo "=== Hibrit Sistem Çalışıyor (CLI Mode) ==="
python hybrid_system.py
