# Reproducibility notes

## External inputs

The release intentionally requires locally supplied Hugging Face model folders
and labelled JSONL splits. `DATA_ROOT` must contain
`<model>/<dataset>/{train,dev,test}.jsonl`; `MODEL_ROOT` must contain one folder
per model key. These inputs are not distributed.

## Generation limits

The shell entry points retain protocol-specific limits instead of applying one
value everywhere: TriviaQA uses 64 new tokens; notable_people, cities_10k,
math_operations_6k, and medals_9k use 128; the final GSM8K aligned-v2 protocol
uses 1024 with extraction batch size 8. The scripts call
`extract-prefill-generation-trajectories`, not the with-logits command.

## Predictor protocol

Both main entry points pass `--feature-mode activation` explicitly. This selects
the 128 raw activation coordinates without scalar/logit fusion. The source
implementation trains mini-batches using `DataLoader(batch_size=64,
shuffle=True)`, saves the best dev-AUROC checkpoint for each seed, averages five
dev probabilities to choose the Youden threshold, and applies that threshold to
the five-checkpoint test ensemble.

## Ablations

`scripts/run_ablation.sh` invokes the retained formal non-random and Random-128
runners. Since tensor artifacts are excluded, set `ARTIFACT_ROOT` to an external
copy of the formal experiment artifacts. Outputs go to `ABLATION_OUT_ROOT` (or
`./ablation_reproduction`). The seven reported dataset/variant configurations
are enumerated in `results/ablation/ablation_results.csv`.

## Deployment cost

`results/cost/paper_reported_cost.json` records the historical manuscript
measurement (1.193 ms/sample at batch 1 and 0.022 ms/sample at batch 64). The
historical transient benchmark script was not retained. The files named
`gru_inference_benchmark.*` are a distinct release-time reproducibility
microbenchmark (approximately 1.981 and 0.0332 ms/sample on this server).
