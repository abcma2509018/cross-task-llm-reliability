#!/usr/bin/env bash
set -euo pipefail

# The external artifact root supplies trajectories/checkpoints deliberately
# excluded from the public package. This entry invokes the retained formal code.
RELEASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_ROOT="${ARTIFACT_ROOT:?Set ARTIFACT_ROOT to the formal experiment artifact root}"
ABLATION_OUT_ROOT="${ABLATION_OUT_ROOT:-$PWD/ablation_reproduction}"
phase="${1:-all}"
models=(qwen_2.5_7b_instruct ministral_8b_instruct mistral_7b_instruct llama3.1_8b_chat)
datasets=(trivia_qa_2_60k gsm8k)
seeds=(13 21 42 87 100)

run_nonrandom() {
  local variant="$1" model dataset seed
  for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
      for seed in "${seeds[@]}"; do
        CLOUD1_ARTIFACT_ROOT="$ARTIFACT_ROOT" CLOUD1_ABLATION_OUT="$ABLATION_OUT_ROOT/nonrandom" \
          python "$RELEASE_ROOT/scripts/ablation/run_minimal_ablation.py" train-group \
          --variant "$variant" --model "$model" --dataset "$dataset" \
          --epochs 40 --seeds "$seed"
      done
      CLOUD1_ARTIFACT_ROOT="$ARTIFACT_ROOT" CLOUD1_ABLATION_OUT="$ABLATION_OUT_ROOT/nonrandom" \
        python "$RELEASE_ROOT/scripts/ablation/run_minimal_ablation.py" ensemble \
        --variant "$variant" --model "$model" --dataset "$dataset"
    done
  done
}

run_full() {
  local model dataset
  for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
      CLOUD1_ARTIFACT_ROOT="$ARTIFACT_ROOT" CLOUD1_ABLATION_OUT="$ABLATION_OUT_ROOT/nonrandom" \
        python "$RELEASE_ROOT/scripts/ablation/run_minimal_ablation.py" collect-full \
        --model "$model" --dataset "$dataset"
    done
  done
}

run_random128() {
  local cmd=(python "$RELEASE_ROOT/scripts/ablation/random128_ablation.py")
  export CLOUD1_ARTIFACT_ROOT="$ARTIFACT_ROOT"
  export CLOUD1_RANDOM128_OUT="$ABLATION_OUT_ROOT/random128"
  export CLOUD1_SELECTED_NODES_ROOT="$RELEASE_ROOT/selected_nodes"
  export CLOUD1_TRAIN_SCRIPT="$RELEASE_ROOT/src/code/triviaqa_mlp_trajectory.py"
  "${cmd[@]}" preflight
  "${cmd[@]}" sample-nodes
  for model in "${models[@]}"; do "${cmd[@]}" extract --model "$model"; done
  "${cmd[@]}" integrity
  for model in "${models[@]}"; do
    "${cmd[@]}" train --model "$model"
    "${cmd[@]}" ensemble --model "$model"
  done
  "${cmd[@]}" aggregate
}

case "$phase" in
  full) run_full ;;
  prefill-only) run_nonrandom prefill_only ;;
  meanpool-mlp) run_nonrandom meanpool_mlp ;;
  random-128) run_random128 ;;
  all) run_full; run_nonrandom prefill_only; run_nonrandom meanpool_mlp; run_random128 ;;
  *) echo "usage: $0 {full|prefill-only|meanpool-mlp|random-128|all}" >&2; exit 2 ;;
esac
