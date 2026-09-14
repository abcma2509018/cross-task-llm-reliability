# Cloud1 release v3 audit

Audit date: 2026-09-14. This closure used retained formal artifacts only; no
model training, feature extraction, or metric recomputation was performed.

## Closure summary

| Check | Result |
|---|---|
| Four TriviaQA-selected Top-128 sets | PASS: each has 128 unique `(layer, neuron)` pairs |
| Registry SHA256 | PASS: all four registry hashes match packaged JSON bytes |
| Formal Direct script | PASS: raw activation extractor and explicit `--feature-mode activation` |
| Formal Fixed-node script | PASS: same raw activation protocol; five target tasks only |
| TriviaQA handling in adaptation | PASS: shares the Direct in-domain result; no retraining |
| Main table binding | PASS: 24 Direct and 24 Adapted rows with source path and SHA256 |
| Paper macro values | PASS |
| Accuracy improvements | PASS: 18 of 20 non-TriviaQA model-task pairs |
| GSM8K task macro | PASS: four-model AUROC macro 0.58075 to 0.81235 (delta 0.23160) |
| Ablation registry | PASS: all seven formal dataset/variant configurations |
| GSM8K Random-128 | PASS: official result and source SHA256 present |
| README paths | PASS: every documented release path exists |
| Forbidden binary artifacts | PASS: none |
| Private absolute paths | PASS: none found |
| Credential patterns | PASS: none found |

## Top-128 binding

| Model | Count | SHA256 | Formal-copy match |
|---|---:|---|---|
| Qwen2.5-7B-Instruct | 128 | `fa37c5de18ff23b174f4ef1dd849f3fc2ec7392752bb3fe1f0ed0bc890bb08b7` | CONFIRMED |
| Ministral-8B-Instruct-2410 | 128 | `592a5b4c49c549b324140a9e10629d620a448d7a7785e0d7d3739ad22586299f` | CONFIRMED |
| Mistral-7B-Instruct-v0.3 | 128 | `61bc9a99ef1506d533fa9260af21efa27acbf4269b8c05e8608fa2130fb851a8` | CONFIRMED |
| Llama 3.1 8B | 128 | `21616d9993a5cd487d6bb21613197c2dc4e02e3ae72d2ddb8acbca684cdb39a4` | CONFIRMED |

The Mistral hash typo in v2 was corrected. Each packaged file was also compared
with the corresponding formal Direct Transfer neurons JSON, including ordered
pair equality.

## Main results

The aligned GSM8K rows are bound to the formal aligned-v2 complete-result files:

- Direct source SHA256: `689b7bf80dba07f31b8332670717ce0af9e4467cd879226cec09a403b0ef708a`
- Adapted source SHA256: `38ed8d5ca7d01f8a5edb936bbf6e867ab394556df2def413123f2ffbe2c14f0a`

For all 20 non-TriviaQA model-task combinations, the validated macro values are:

| Setting | AUROC | AUPRC | Accuracy | F1 | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 0.7590 | 0.7177 | 0.7390 | 0.5296 | 0.1640 | 0.1285 |
| Fixed-node Adaptation | 0.9014 | 0.8665 | 0.8417 | 0.7945 | 0.0886 | 0.0339 |

All 48 metric-source JSON references in the Direct and Adapted registries were
resolved in the formal archive and independently matched to their recorded
SHA256 during closure.

## Method implementation

`scripts/run_direct_transfer.sh` trains five TriviaQA GRUs and freezes the
source ensemble and source-dev Youden threshold for six test datasets.
`scripts/run_fixed_node_adaptation.sh` trains five target-specific GRUs for the
five non-TriviaQA tasks while retaining the TriviaQA-selected nodes. Both use
the no-logit extractor and a 128-dimensional raw activation feature.

The duplicate reconstructed `predictor/train_gru.py` tree was removed because
it did not provide the documented mini-batch behavior. The retained formal
`src/code/triviaqa_mlp_trajectory.py` uses
`DataLoader(batch_size=args.batch_size, shuffle=True)` and dev-AUROC checkpoint
selection. The older fusion orchestrator is isolated under `legacy/` and is not
referenced by any formal entry point.

Generation limits are dataset-specific in `configs/generation_limits.json`.
In particular, formal aligned-v2 GSM8K uses 1024 new tokens; the scripts do not
apply a guessed value of 256 across datasets.

## Ablation provenance

`results/ablation/ablation_results.csv` includes Full, Prefill-only, and
MeanPool+MLP for TriviaQA and GSM8K, plus Random-128 for GSM8K only. The first
six configurations bind to source SHA256
`28b65647aa48ff3d0063284021a8eb6503b7b407251df01b824c4eb3ddfb7b5b`;
Random-128 binds to
`1c9e964e3f2cfce51e98775522184c5f4df6534a54caaa0af6bf3c1c74124eb3`.
The actual archived non-random and Random-128 runners are retained under
`scripts/ablation/`; `scripts/run_ablation.sh` provides executable phases.

## Deployment cost

The two timing classes are separate:

- `paper_reported_cost`: historical manuscript values 1.193 ms/sample (batch
  1) and 0.022 ms/sample (batch 64). The transient historical script was not
  retained.
- `release_reproduction_benchmark`: v2 release-time microbenchmark values about
  1.981 and 0.0332 ms/sample. The runnable utility is
  `scripts/benchmark_gru_inference.py`.

Feature-extraction progress evidence and trajectory-length summary remain small
JSON/CSV records; no trajectory tensors are included.

## Remaining requirements

There is no active paper/code protocol mismatch in the public entry points.
Reproduction still requires external model weights, labelled dataset splits,
and, for ablations, the omitted formal tensor artifacts. The historical GRU
cost benchmark script is unavailable, so its manuscript values remain labelled
as historical measurements rather than release-time reproduction results.

Machine-readable validation is in `V3_VALIDATION.json`. Package file count and
size are recorded after final cleanup in the ZIP verification output.
