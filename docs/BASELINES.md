# Baseline methods release

This release contains paper-faithful reproduction code for Ni-Avg-State and
PRISM-SAPLMA across four language models and six datasets. It does not include
models, datasets, generated answers, hidden-state features, predictions,
checkpoints, or caches.

## External inputs

Set these environment variables when using the all-run launcher, or pass the
equivalent `--data-root`, `--model-root`, and `--output-root` arguments to the
Python runner:

```text
DATA_ROOT/<model_key>/<dataset>/{train,dev,test}.jsonl
MODEL_ROOT/<model_key>/
OUTPUT_ROOT/
```

Every JSONL row must provide a sample ID, prompt, generated answer, binary
correctness label, and split. Accepted field aliases are defined by `read_rows`
in `scripts/baselines/run_baselines.py`. The generated answers, partitions, and
labels are external project inputs and are not distributed here.

Supported model keys are `qwen_2.5_7b_instruct`, `ministral_8b_instruct`,
`mistral_7b_instruct`, and `llama3.1_8b_chat`. Supported datasets are
`trivia_qa_2_60k`, `notable_people`, `cities_10k`, `math_operations_6k`,
`medals_9k`, and `gsm8k`.

## Ni-Avg-State

The base model is frozen and generation is greedy. The runner records an
intermediate-layer hidden state at every valid generation step and averages the
states into a fixed-length representation. A single-hidden-layer MLP is trained
on exactly 1,000 correct and 1,000 incorrect TriviaQA source examples. The
source development split selects the best checkpoint. The selected classifier
is frozen for all target tests, with decision threshold 0.5 and no target
retraining.

The included source manifests contain only sample IDs and binary labels. When a
matching seed manifest exists, it fixes the source sample set; otherwise the
runner reconstructs a balanced set deterministically from the seed. Regenerated
answers are checked against the supplied generated-answer text.

## PRISM-SAPLMA

The judgment query is exactly:

```text
Does the statement <generated_response> accurately reflect the truth?
```

The response is inserted without quotation marks, labels, or prompt variants.
The representation is the last layer hidden state at the final valid token. A
deterministic 80:20 split of TriviaQA source-train data supplies classifier
training and development partitions. Development selects the best MLP
checkpoint. The classifier is frozen for target evaluation, with threshold 0.5
and no target retraining.

## Running

Run one method/model combination:

```bash
python scripts/baselines/run_baselines.py \
  --method ni_avg_state \
  --model-key qwen_2.5_7b_instruct \
  --data-root "$DATA_ROOT" \
  --model-root "$MODEL_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --config configs/baselines.json \
  --seed 42
```

Use `prism_saplma` for the second baseline. To run every configured model and
historical seed, set the three environment variables and run:

```bash
bash scripts/baselines/run_all_baselines.sh
```

Runtime tensors and checkpoints are written under `OUTPUT_ROOT`, not into this
release.

## Parameter provenance

Paper-specified protocol fields and historical implementation details are
separated in `configs/baselines.json` and
`metadata/implementation_parameters.csv`. Epoch count, hidden size, learning
rate, optimizer details, seeds, layer indices, generation limits, and the
checkpoint-selection metric must not be described as paper-specified unless
the paper states them.

## Archived result summary

`results/baselines/table7_baselines_mean_std.csv` is an unchanged, small
archived paper-result summary. It was produced by the historical experiment
workflow. The corrected release runner has not been rerun in the current
environment, so this CSV must not be presented as output newly generated or
validated by this release.
