#!/usr/bin/env bash
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to the external split JSONL root}"
: "${MODEL_ROOT:?Set MODEL_ROOT to the external model-weight root}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to an external experiment output root}"

release_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
runner="${release_root}/scripts/baselines/run_baselines.py"
config="${release_root}/configs/baselines.json"

models=(qwen_2.5_7b_instruct ministral_8b_instruct mistral_7b_instruct llama3.1_8b_chat)

for model in "${models[@]}"; do
  for seed in 0 42 100; do
    python "${runner}" --method ni_avg_state --model-key "${model}" \
      --data-root "${DATA_ROOT}" --model-root "${MODEL_ROOT}" \
      --output-root "${OUTPUT_ROOT}" --config "${config}" --seed "${seed}"
  done
  for seed in 0 1 2; do
    python "${runner}" --method prism_saplma --model-key "${model}" \
      --data-root "${DATA_ROOT}" --model-root "${MODEL_ROOT}" \
      --output-root "${OUTPUT_ROOT}" --config "${config}" --seed "${seed}"
  done
done
