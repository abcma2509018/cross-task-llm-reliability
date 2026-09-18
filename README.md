# Cross-Task LLM Reliability

This repository contains the reproducible code, metadata, and compact result
summaries for cross-task LLM answer-reliability experiments: model-specific
TriviaQA Top-128 sparse MLP observation nodes, the code and metadata needed to
reproduce raw activation trajectories, but does not distribute trajectory
tensors; a GRU reliability predictor; Direct Transfer; Fixed-node Adaptation;
reported ablations; and the Ni-Avg-State and PRISM-SAPLMA baselines.

It does not contain K-sensitivity experiments, model weights, datasets,
checkpoints, activation/trajectory tensors, prediction dumps, or raw logs.
Those exclusions are intentional.

## Final method

- Hook: input to each transformer layer's `mlp.down_proj`.
- Feature: 128-dimensional raw activation state at each trajectory step.
- Predictor: one-layer, unidirectional GRU; hidden size 256; dropout 0.2.
- Training: mini-batch size 64, 40 epochs, AdamW (`lr=1e-3`, weight decay
  `1e-4`), gradient clipping 1.0.
- Seeds: 13, 21, 42, 87, 100.
- Selection: best checkpoint by dev AUROC.
- Evaluation: mean probability across five seeds and a dev Youden threshold.
- No scalar or logit fusion is part of the final method.

The retained implementation is `src/code/triviaqa_mlp_trajectory.py`. Its
`train-gru` command uses a shuffled PyTorch `DataLoader` with the requested
mini-batch size. Historical fusion orchestration is retained only for reference
and is not a formal entry point for the paper method.

## Entry points

Set `DATA_ROOT` and `MODEL_ROOT` to local resources, then run:

```bash
DATA_ROOT=/path/to/splits MODEL_ROOT=/path/to/models \
  bash scripts/run_direct_transfer.sh

DATA_ROOT=/path/to/splits MODEL_ROOT=/path/to/models \
  bash scripts/run_fixed_node_adaptation.sh
```

Direct Transfer trains on TriviaQA train/dev, fixes the source dev Youden
threshold, then evaluates TriviaQA and five target test sets. Fixed-node
Adaptation keeps the TriviaQA Top-128 coordinates fixed, trains on each of the
five target train/dev splits, and evaluates target test. Its TriviaQA row is the
shared Direct Transfer in-domain result and is not retrained.

For the formal ablations, provide the external artifact root because the public
package omits trajectories and checkpoints:

```bash
ARTIFACT_ROOT=/path/to/formal/artifacts bash scripts/run_ablation.sh all
```

Individual phases are `full`, `prefill-only`, `meanpool-mlp`, and
`random-128`. Random-128 is a GSM8K-only, layer-matched control; TriviaQA
Random-128 is not a reported paper configuration. Configuration is recorded in
`configs/ablation_final.json`, and archived macro results are in
`results/ablation/ablation_results.csv`.

## Results and cost

Small final tables are in `results/main/`. Every row includes its formal source
path and SHA256. Deployment measurements are under `results/cost/` and separate
historical manuscript values from the release-time reproduction benchmark.
Re-run the microbenchmark with `scripts/benchmark_gru_inference.py`.

See `docs/REPRODUCIBILITY.md` for required external inputs,
`docs/BASELINES.md` for the baseline protocols, and `RELEASE_AUDIT.md` for
automated closure checks.
