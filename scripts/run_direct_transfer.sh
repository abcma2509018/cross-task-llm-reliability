#!/usr/bin/env bash
set -euo pipefail

# Final no-scalar, raw-activation Direct Transfer protocol.
RELEASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE="${CODE:-$RELEASE_ROOT/src/code/triviaqa_mlp_trajectory.py}"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the labelled split root}"
MODEL_ROOT="${MODEL_ROOT:?Set MODEL_ROOT to the local Hugging Face model root}"
OUT_ROOT="${OUT_ROOT:-$RELEASE_ROOT/outputs/direct_transfer}"
SEEDS=(13 21 42 87 100)
MODELS=(qwen_2.5_7b_instruct ministral_8b_instruct mistral_7b_instruct llama3.1_8b_chat)
TARGETS=(notable_people cities_10k math_operations_6k medals_9k gsm8k)

# Aligned-v2 GSM8K used 1024. The retained standard pipeline used 128 for the
# other transfer targets; the formal TriviaQA command used 64.
declare -A MAX_NEW_TOKENS=(
  [trivia_qa_2_60k]=64 [notable_people]=128 [cities_10k]=128
  [math_operations_6k]=128 [medals_9k]=128 [gsm8k]=1024
)

for model in "${MODELS[@]}"; do
  neurons="$RELEASE_ROOT/selected_nodes/$model/topk_mlp_neurons_k128_trivia10k.json"
  run="$OUT_ROOT/$model"
  model_path="$MODEL_ROOT/$model"

  python "$CODE" extract-prefill-generation-trajectories \
    --model-key "$model" --model-path "$model_path" --dataset trivia_qa_2_60k \
    --splits train dev test --neurons-json "$neurons" \
    --train-path "$DATA_ROOT/$model/trivia_qa_2_60k/train.jsonl" \
    --dev-path "$DATA_ROOT/$model/trivia_qa_2_60k/dev.jsonl" \
    --test-path "$DATA_ROOT/$model/trivia_qa_2_60k/test.jsonl" \
    --out-dir "$run/features/trivia_qa_2_60k" \
    --checkpoint-dir "$run/feature_checkpoints/trivia_qa_2_60k" \
    --batch-size 1 --max-input-tokens 1024 \
    --max-new-tokens "${MAX_NEW_TOKENS[trivia_qa_2_60k]}" --resume

  for dataset in "${TARGETS[@]}"; do
    extract_batch=1
    [[ "$dataset" == gsm8k ]] && extract_batch=8
    python "$CODE" extract-prefill-generation-trajectories \
      --model-key "$model" --model-path "$model_path" --dataset "$dataset" \
      --splits test --neurons-json "$neurons" \
      --test-path "$DATA_ROOT/$model/$dataset/test.jsonl" \
      --out-dir "$run/features/$dataset" \
      --checkpoint-dir "$run/feature_checkpoints/$dataset" \
      --batch-size "$extract_batch" --max-input-tokens 1024 \
      --max-new-tokens "${MAX_NEW_TOKENS[$dataset]}" --resume
  done

  for seed in "${SEEDS[@]}"; do
    python "$CODE" train-gru --feature-mode activation \
      --train-pt "$run/features/trivia_qa_2_60k/train.pt" \
      --dev-pt "$run/features/trivia_qa_2_60k/dev.pt" \
      --out-dir "$run/models/seed$seed" --hidden-size 256 --num-layers 1 \
      --dropout 0.2 --epochs 40 --batch-size 64 --lr 0.001 \
      --weight-decay 0.0001 --grad-clip 1.0 --seed "$seed"
  done

  predictions=(); checkpoints=()
  for seed in "${SEEDS[@]}"; do
    predictions+=("$run/models/seed$seed/dev_predictions.csv")
    checkpoints+=("$run/models/seed$seed/best_gru.pt")
  done
  python "$CODE" ensemble-dev --prediction-files "${predictions[@]}" \
    --out-csv "$run/dev_ensemble.csv" --out-json "$run/dev_ensemble.json"
  threshold="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["threshold"])' "$run/dev_ensemble.json")"

  for dataset in trivia_qa_2_60k "${TARGETS[@]}"; do
    python "$CODE" ensemble-test --test-pt "$run/features/$dataset/test.pt" \
      --checkpoints "${checkpoints[@]}" --threshold "$threshold" --batch-size 64 \
      --out-csv "$run/results/$dataset.csv" --out-json "$run/results/$dataset.json"
  done
done
