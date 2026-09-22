#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# Experiment settings.
# Supported datasets: celeba, ffhq
# Supported target models: IR152, IRSE50, FaceNet, MobileFace
DATASET="${DATASET:-celeba}"
MODEL="${MODEL:-MobileFace}"

NUM="${NUM:-92}"
T="${T:-50}"
S="${S:-300}"

SEEDS=(4 5 6 7 8)

mkdir -p results logs

for SEED in "${SEEDS[@]}"; do
    SAVE_DIR="results/${DATASET}_${MODEL}_seed${SEED}"
    LOG_FILE="logs/${DATASET}_${MODEL}_seed${SEED}.log"

    mkdir -p "$SAVE_DIR"

    echo "========================================"
    echo "Dataset: ${DATASET}"
    echo "Held-out model: ${MODEL}"
    echo "Seed: ${SEED}"
    echo "T: ${T}"
    echo "S: ${S}"
    echo "Output: ${SAVE_DIR}"
    echo "========================================"

    if ! python -u eval.py \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --num "$NUM" \
        --t "$T" \
        --s "$S" \
        --seed "$SEED" \
        --save "$SAVE_DIR" \
        > "$LOG_FILE" 2>&1; then

        echo "Seed ${SEED} failed. See ${LOG_FILE}" >&2
        exit 1
    fi

    echo "Seed ${SEED} finished."
done

echo "========================================"
echo "All five seeds finished."
echo "========================================"
