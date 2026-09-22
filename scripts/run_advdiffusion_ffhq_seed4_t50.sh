#!/usr/bin/env bash
# Adv-Diffusion: FFHQ, seed 4, four held-out models, t=50, s=300.
# Uses the current eval_source.py (retain the MobileFace CPU evaluation fix).
# Does not change ddim_steps. Every launch creates a fresh output directory.
set -Eeuo pipefail

PROJECT_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
NUM=92

for required in eval_source.py utils.py configs/stable-diffusion/v2-inpainting-inference.yaml; do
    if [[ ! -f "$required" ]]; then
        echo "Missing file: $PWD/$required" >&2
        echo "Run from your project directory, or pass its path as the first argument." >&2
        exit 1
    fi
done
for dataset_dir in ffhq_sample; do
    for split in src target; do
        if [[ ! -d "$dataset_dir/$split" ]]; then
            echo "Missing dataset directory: $PWD/$dataset_dir/$split" >&2
            exit 1
        fi
    done
done

# mktemp guarantees a fresh output directory on every invocation.
mkdir -p res_source
RUN_ROOT="$(mktemp -d "$PWD/res_source/advdiffusion_ffhq_seed4_t50_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
SUMMARY="$RUN_ROOT/summary.csv"
printf 'seed,dataset,blackbox_model,num,status,result_path\n' > "$SUMMARY"
echo "Output directory: $RUN_ROOT"
echo "4 runs; 92 image pairs per run. Each --model is the held-out black-box model."

completed=0
failed=0
for seed in 4; do
    for dataset in ffhq; do
        for model in IR152 IRSE50 FaceNet MobileFace; do
            out_dir="$RUN_ROOT/seed${seed}/${dataset}_${model}"
            mkdir -p "$out_dir"
            echo "[$((completed + failed + 1))/4] seed=$seed dataset=$dataset model=$model"
            if "$PYTHON_BIN" -u eval_source.py \
                --dataset "$dataset" \
                --model "$model" \
                --num "$NUM" \
                --seed "$seed" \
                --t 50 \
                --s 300 \
                --save "$out_dir" 2>&1 | tee "$out_dir/run.log"; then
                completed=$((completed + 1))
                printf '%s,%s,%s,%s,completed,%s\n' "$seed" "$dataset" "$model" "$NUM" "$out_dir/result.json" >> "$SUMMARY"
            else
                failed=$((failed + 1))
                printf '%s,%s,%s,%s,failed,%s\n' "$seed" "$dataset" "$model" "$NUM" "$out_dir/result.json" >> "$SUMMARY"
                echo "Run failed; see $out_dir/run.log. Continuing with the next run." >&2
            fi
        done
    done
done

# Aggregate only complete runs; partial ASRs must not be treated as final results.
"$PYTHON_BIN" - "$RUN_ROOT" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
with (root / 'summary.csv').open(newline='') as f:
    runs = list(csv.DictReader(f))
fields = ['seed', 'dataset', 'blackbox_model', 'num', 'ASR@0.1', 'ASR@0.01', 'ASR@0.001']
with (root / 'asr_summary.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for run in runs:
        if run['status'] != 'completed':
            continue
        result_path = Path(run['result_path'])
        progress = json.loads(result_path.with_name('progress.json').read_text())
        if not progress['finished'] or progress['completed'] != int(run['num']):
            raise RuntimeError(f'Incomplete result: {result_path}')
        scores = json.loads(result_path.read_text())[run['blackbox_model']]
        writer.writerow({**{key: run[key] for key in fields[:4]},
                         **{key: scores[key] for key in fields[4:]}})
print(f'ASR summary (fractions, 0–1): {root / "asr_summary.csv"}')
PY

echo "Finished: $completed succeeded, $failed failed."
echo "Outputs: $RUN_ROOT"
if (( failed > 0 )); then
    exit 1
fi
