#!/usr/bin/env bash
set -e

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${PROJECT}/ablation_ffhq_mobile_seed4"
cd "$PROJECT"
mkdir -p "$ROOT"

run () {
    name=$1
    t=$2
    s=$3

    echo "======================================"
    echo "$name : T=$t S=$s"
    echo "======================================"

    python -u eval.py \
      --dataset ffhq \
      --model MobileFace \
      --num 92 \
      --seed 4 \
      --t "$t" \
      --s "$s" \
      --save "$ROOT/$name" \
      2>&1 | tee "$ROOT/${name}.log"
}

run T35 35 300
run DEFAULT_T50_S300 50 300
run T65 65 300
run S200 50 200
run S400 50 400

echo "ALL T/S ABLATIONS FINISHED"
