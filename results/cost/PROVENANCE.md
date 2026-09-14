# Deployment-cost provenance

Two different measurement classes are retained and must not be combined.

## A. paper_reported_cost

`paper_reported_cost.json` records the historical manuscript measurement:
1.193 ms/sample at batch size 1 and 0.022 ms/sample at batch size 64. The
original transient benchmark script was not retained, so these values are
labelled historical rather than reproduced by the public utility.

The progress records were copied from the formal Ministral Direct Transfer
feature extraction directory, under
`Method_A_Direct_Transfer_no_scalar_raw_activation/features/trivia_qa_2_60k_ckpt/`.
They support the retained 108.2, 72.2, and 87.5 ms/sample extraction timings.

## B. release_reproduction_benchmark

`gru_inference_benchmark.json` and `.csv` are the v2 release-time
reproducibility microbenchmark on an NVIDIA GeForce RTX 5090. They report about
1.981 ms/sample at batch size 1 and 0.0332 ms/sample at batch size 64. The
executable is `scripts/benchmark_gru_inference.py`; its environment is recorded
in `gru_inference_environment.txt`.

`trajectory_length_summary.json` is deterministic summary arithmetic from the
retained shape/protocol, not a tensor dump.
