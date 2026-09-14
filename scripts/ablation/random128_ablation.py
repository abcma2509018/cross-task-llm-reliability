#!/usr/bin/env python3
"""Layer-matched Random-128 ablation on the formal GSM8K aligned-v2 data."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from transformers import AutoConfig


ROOT = Path(os.environ.get("CLOUD1_ARTIFACT_ROOT", Path(__file__).resolve().parents[2]))
OUT = Path(os.environ.get("CLOUD1_RANDOM128_OUT", ROOT / "ablation_random128"))
ALIGNED_ROOT = ROOT / "gsm8k_aligned_v2"
FORMAL_SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = Path(os.environ.get("CLOUD1_TRAIN_SCRIPT", Path(__file__).resolve().parents[2] / "src/code/triviaqa_mlp_trajectory.py"))
SELECTED_NODES_ROOT = Path(os.environ.get("CLOUD1_SELECTED_NODES_ROOT", Path(__file__).resolve().parents[2] / "selected_nodes"))
sys.path.insert(0, str(FORMAL_SCRIPT_DIR))
import gsm8k_aligned_v2 as formal  # noqa: E402


MODELS = (
    "qwen_2.5_7b_instruct",
    "ministral_8b_instruct",
    "mistral_7b_instruct",
    "llama3.1_8b_chat",
)
SPLITS = ("train", "dev", "test")
EXPECTED_SPLITS = {"train": 6155, "dev": 1319, "test": 1318}
PREDICTOR_SEEDS = (13, 21, 42, 87, 100)
NODE_SET_SEED = 20260816
K = 128
REPLAY_ATOL = 1e-3
REPLAY_RTOL = 1e-3


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def selected_path(model: str) -> Path:
    return SELECTED_NODES_ROOT / model / "topk_mlp_neurons_k128_trivia10k.json"


def records_path(model: str) -> Path:
    return ALIGNED_ROOT / "deliverables/gsm8k_trivia_top128_scope/records" / f"{model}.gsm8k_aligned_v2.jsonl"


def formal_split_path(model: str, split: str) -> Path:
    return ALIGNED_ROOT / "trajectories" / model / "trivia_top128/splits" / f"{split}.pt"


def formal_metric_path(model: str) -> Path:
    return ALIGNED_ROOT / "metrics" / model / "trivia_top128/ensemble_test_metrics.json"


def random_node_path(model: str) -> Path:
    return OUT / "random_nodes" / f"{model}.json"


def random_split_path(model: str, split: str) -> Path:
    return OUT / "trajectories" / model / "gsm8k" / f"{split}.pt"


def random_checkpoint_dir(model: str) -> Path:
    return OUT / "checkpoints" / model / "gsm8k"


def random_model_dir(model: str, seed: int) -> Path:
    return OUT / "models" / model / "gsm8k" / f"gru_seed{seed}"


def load_selected_nodes(model: str) -> list[dict[str, Any]]:
    nodes = json.loads(selected_path(model).read_text(encoding="utf-8"))
    if isinstance(nodes, dict):
        nodes = nodes.get("nodes", nodes.get("neurons", nodes.get("topk", [])))
    normalized = [
        {"layer": int(item.get("layer", item.get("layer_index"))), "neuron": int(item.get("neuron", item.get("node_index")))}
        for item in nodes
    ]
    if len(normalized) != K or len({(x["layer"], x["neuron"]) for x in normalized}) != K:
        raise ValueError(f"{model}: invalid Selected-128 file {selected_path(model)}")
    return normalized


def load_random_nodes(model: str) -> list[dict[str, Any]]:
    payload = json.loads(random_node_path(model).read_text(encoding="utf-8"))
    nodes = [{"layer": int(item["layer_index"]), "neuron": int(item["node_index"])} for item in payload["nodes"]]
    if len(nodes) != K:
        raise ValueError(f"{model}: Random node count is {len(nodes)}")
    return nodes


def model_dimensions(model: str) -> tuple[int, int]:
    config = AutoConfig.from_pretrained((ROOT / formal.MODEL_PATHS[model]).resolve())
    layers = int(getattr(config, "num_hidden_layers"))
    intermediate = int(getattr(config, "intermediate_size"))
    return layers, intermediate


def ece_score(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        include = (p >= left) & ((p <= right) if index == n_bins - 1 else (p < right))
        if include.any():
            total += float(include.mean()) * abs(float(y[include].mean()) - float(p[include].mean()))
    return total


def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y, p)
    finite = np.where(np.isfinite(thresholds))[0]
    return float(thresholds[finite[np.argmax((tpr - fpr)[finite])]])


def metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (p >= threshold).astype(np.int64)
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
        "ece": float(ece_score(y, p, 15)),
        "threshold": float(threshold),
    }


def write_predictions(path: Path, ids: list[str], y: np.ndarray, p: np.ndarray, threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["idx", "sample_id", "y", "p_correct", "risk", "pred"])
        writer.writeheader()
        for index, (sample_id, label, prob) in enumerate(zip(ids, y, p)):
            writer.writerow({
                "idx": index,
                "sample_id": sample_id,
                "y": int(label),
                "p_correct": float(prob),
                "risk": float(1.0 - prob),
                "pred": int(prob >= threshold),
            })
    os.replace(temporary, path)


def load_prediction_table(path: Path) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sample_id = row.get("sample_id") or row["idx"]
            if sample_id in output:
                raise ValueError(f"duplicate prediction ID {sample_id} in {path}")
            output[sample_id] = row
    return output


def command_preflight(_args: argparse.Namespace) -> None:
    if (ALIGNED_ROOT / "STATUS").read_text(encoding="utf-8").strip() != "formal":
        raise RuntimeError("gsm8k_aligned_v2 is not marked formal")
    report: dict[str, Any] = {
        "status": "PASS",
        "scope": "GSM8K aligned-v2 layer-matched Random-128 only",
        "node_set_seed": NODE_SET_SEED,
        "predictor_seeds": list(PREDICTOR_SEEDS),
        "aligned_v2_path": str(ALIGNED_ROOT.relative_to(ROOT)),
        "trajectory_extraction_script": str(Path(formal.__file__).resolve().relative_to(ROOT)),
        "predictor_training_script": str(TRAIN_SCRIPT.relative_to(ROOT)),
        "hook": "input to model.model.layers[layer].mlp.down_proj",
        "models": {},
        "input_checksums": {},
    }
    critical = [ALIGNED_ROOT / "config/aligned_v2_config.json", Path(formal.__file__).resolve(), TRAIN_SCRIPT]
    for model in MODELS:
        layers, intermediate = model_dimensions(model)
        selected = load_selected_nodes(model)
        layer_counts = Counter(item["layer"] for item in selected)
        if any(item["layer"] < 0 or item["layer"] >= layers or item["neuron"] < 0 or item["neuron"] >= intermediate for item in selected):
            raise ValueError(f"{model}: Selected nodes outside model dimensions")
        records = read_jsonl(records_path(model))
        split_counts = Counter(row["split"] for row in records)
        token_ids_available = all(isinstance(row.get("generated_token_ids"), list) for row in records)
        text_available = all(isinstance(row.get("generated_text"), str) for row in records)
        labels_available = all(row.get("correctness_label") in (0, 1) for row in records)
        trajectory_info = {}
        for split in SPLITS:
            path = formal_split_path(model, split)
            data = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            trajectory_info[split] = {
                "path": str(path.relative_to(ROOT)),
                "shape": list(data["x"].shape),
                "dtype": str(data["x"].dtype),
                "positive": int(data["y"].sum()),
                "negative": int(len(data["y"]) - data["y"].sum()),
            }
            if len(data["y"]) != EXPECTED_SPLITS[split] or data["x"].shape[-1] != K:
                raise ValueError(f"{model}/{split}: invalid formal trajectory")
        metric_path = formal_metric_path(model)
        if split_counts != Counter(EXPECTED_SPLITS) or not token_ids_available or not text_available or not labels_available:
            raise ValueError(f"{model}: incomplete aligned-v2 records")
        candidate_path = OUT / "manifests/candidate_pools" / f"{model}.json"
        candidate_payload = {
            "model": model,
            "definition": "all (layer_index, node_index) pairs in model MLP intermediate activation",
            "num_layers": layers,
            "intermediate_size": intermediate,
            "total_nodes": layers * intermediate,
            "layer_ranges": [{"layer_index": layer, "node_index_min": 0, "node_index_max": intermediate - 1} for layer in range(layers)],
        }
        write_json(candidate_path, candidate_payload)
        report["models"][model] = {
            "model_path": str((ROOT / formal.MODEL_PATHS[model]).resolve()),
            "tokenizer_path": str((ROOT / formal.MODEL_PATHS[model]).resolve()),
            "num_layers": layers,
            "mlp_intermediate_size": intermediate,
            "total_mlp_observation_nodes": layers * intermediate,
            "selected_top128_path": str(selected_path(model).relative_to(ROOT)),
            "selected_node_count": len(selected),
            "selected_count_per_layer": {str(key): value for key, value in sorted(layer_counts.items())},
            "candidate_pool_path": str(candidate_path.relative_to(ROOT)),
            "records_path": str(records_path(model).relative_to(ROOT)),
            "split_sizes": dict(split_counts),
            "generated_token_ids_available": token_ids_available,
            "generated_text_available": text_available,
            "correctness_labels_available": labels_available,
            "formal_trajectories": trajectory_info,
            "formal_fixed_node_result_path": str(metric_path.relative_to(ROOT)),
            "formal_checkpoints": [
                str((ALIGNED_ROOT / "checkpoints" / model / "trivia_top128" / f"gru_seed{seed}/best_gru.pt").relative_to(ROOT))
                for seed in PREDICTOR_SEEDS
            ],
        }
        critical.extend([selected_path(model), records_path(model), metric_path, ALIGNED_ROOT / "manifests" / model / "manifest_summary.json"])
    for path in critical:
        report["input_checksums"][str(path.relative_to(ROOT))] = sha256_file(path)
    write_json(OUT / "manifests/preflight_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def command_sample_nodes(_args: argparse.Namespace) -> None:
    rows = []
    for model in MODELS:
        layers, intermediate = model_dimensions(model)
        selected = load_selected_nodes(model)
        selected_set = {(item["layer"], item["neuron"]) for item in selected}
        selected_counts = Counter(item["layer"] for item in selected)
        rng = np.random.default_rng(NODE_SET_SEED)
        random_nodes: list[dict[str, int]] = []
        pool_sizes = {}
        for layer, count in sorted(selected_counts.items()):
            excluded = {node for selected_layer, node in selected_set if selected_layer == layer}
            candidates = np.asarray([node for node in range(intermediate) if node not in excluded], dtype=np.int64)
            pool_sizes[layer] = int(len(candidates))
            if len(candidates) < count:
                raise RuntimeError(f"{model}/layer{layer}: cannot sample {count} zero-overlap nodes")
            chosen = rng.choice(candidates, size=count, replace=False)
            random_nodes.extend({"layer_index": int(layer), "node_index": int(node)} for node in sorted(chosen.tolist()))
        random_set = {(item["layer_index"], item["node_index"]) for item in random_nodes}
        random_counts = Counter(item["layer_index"] for item in random_nodes)
        overlap = len(selected_set & random_set)
        if len(random_nodes) != K or len(random_set) != K or random_counts != selected_counts or overlap != 0:
            raise AssertionError(f"{model}: Random-128 integrity failure")
        if any(layer < 0 or layer >= layers or node < 0 or node >= intermediate for layer, node in random_set):
            raise AssertionError(f"{model}: random node outside candidate pool")
        payload = {
            "model": model,
            "variant": "layer_matched_random_128",
            "node_set_seed": NODE_SET_SEED,
            "total_nodes": len(random_nodes),
            "unique_nodes": len(random_set),
            "selected_count_per_layer": {str(key): value for key, value in sorted(selected_counts.items())},
            "random_count_per_layer": {str(key): value for key, value in sorted(random_counts.items())},
            "overlap_with_selected": overlap,
            "candidate_pool_size_per_layer": {str(key): value for key, value in sorted(pool_sizes.items())},
            "candidate_pool_definition": "all node indices 0..intermediate_size-1 within each selected layer, excluding Selected-128 for zero-overlap sampling",
            "selected_nodes_path": str(selected_path(model).relative_to(ROOT)),
            "nodes": random_nodes,
        }
        path = random_node_path(model)
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != payload:
            raise RuntimeError(f"Refusing to replace an existing different Random-128 set: {path}")
        write_json(path, payload)
        rows.append({
            "model": model,
            "node_set_seed": NODE_SET_SEED,
            "total_nodes": K,
            "unique_nodes": K,
            "overlap_with_selected": overlap,
            "selected_layer_counts": json.dumps(dict(sorted(selected_counts.items()))),
            "random_layer_counts": json.dumps(dict(sorted(random_counts.items()))),
            "manifest_path": str(path.relative_to(ROOT)),
            "manifest_sha256": sha256_file(path),
        })
    csv_path = OUT / "results/random_node_manifest.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2, ensure_ascii=False))


class ReplayTrajectoryRecorder:
    """The formal SelectedTrajectoryRecorder semantics, used with fixed token replay."""

    def __init__(self, model: Any, nodes: list[dict[str, Any]], attention_mask: torch.Tensor):
        self.model = model
        self.selected, self.positions = formal.neuron_tensors(nodes)
        self.attention_mask = attention_mask.detach().cpu().bool()
        self.prompt: dict[int, torch.Tensor] = {}
        self.generation: dict[int, list[torch.Tensor]] = defaultdict(list)
        self.handles = []

    def __enter__(self) -> "ReplayTrajectoryRecorder":
        for layer_id, layer in enumerate(formal.get_layers(self.model)):
            if layer_id not in self.selected:
                continue
            indices = self.selected[layer_id]

            def hook(_module: Any, inputs: tuple[torch.Tensor, ...], _output: Any, layer_id: int = layer_id, indices: torch.Tensor = indices) -> None:
                values = inputs[0].index_select(-1, indices.to(inputs[0].device)).detach().float().cpu()
                if layer_id not in self.prompt:
                    self.prompt[layer_id] = values
                else:
                    self.generation[layer_id].append(values[:, -1, :])

            self.handles.append(layer.mlp.down_proj.register_forward_hook(hook))
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for handle in self.handles:
            handle.remove()

    def trajectories(self, token_steps: list[int]) -> list[torch.Tensor]:
        output = []
        for sample_index, generated_steps in enumerate(token_steps):
            generation_steps = max(0, generated_steps - 1)
            trajectory = torch.zeros((3 + generation_steps, K), dtype=torch.float32)
            for layer_id, positions in self.positions.items():
                prefill = self.prompt[layer_id][sample_index]
                valid = prefill[self.attention_mask[sample_index]]
                trajectory[0, positions] = valid[-1]
                trajectory[1, positions] = valid.mean(0)
                trajectory[2, positions] = valid.max(0).values
                if generation_steps:
                    values = torch.stack(
                        [self.generation[layer_id][step][sample_index] for step in range(generation_steps)], dim=0
                    )
                    trajectory[3:, positions] = values
            output.append(trajectory)
        return output


def replay_batch(
    model: Any,
    tokenizer: Any,
    chunk_rows: list[dict[str, Any]],
    chunk_records: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
) -> list[torch.Tensor]:
    if [str(row["id"]) for row in chunk_rows] != [row["sample_id"] for row in chunk_records]:
        raise ValueError("batch rows and aligned records are not ordered identically")
    prompts = [row["qa_prompt"] for row in chunk_rows]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
    input_length = int(encoded["input_ids"].shape[1])
    attention_mask = encoded["attention_mask"]
    prompt_position_ids = attention_mask.long().cumsum(-1) - 1
    prompt_position_ids.masked_fill_(attention_mask == 0, 1)
    token_ids = [list(map(int, row["generated_token_ids"])) for row in chunk_records]
    token_steps = [len(values) for values in token_ids]
    max_replay_steps = max(max(0, length - 1) for length in token_steps)
    with ReplayTrajectoryRecorder(model, nodes, attention_mask) as recorder:
        with torch.inference_mode():
            outputs = model(
                input_ids=encoded["input_ids"],
                attention_mask=attention_mask,
                position_ids=prompt_position_ids,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            del outputs
            for step in range(max_replay_steps):
                next_tokens = [values[step] if step < len(values) - 1 else int(tokenizer.eos_token_id) for values in token_ids]
                replay_input = torch.tensor(next_tokens, dtype=torch.long, device="cuda").unsqueeze(1)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((len(chunk_rows), 1), dtype=attention_mask.dtype, device="cuda")], dim=1
                )
                position_ids = attention_mask.long().sum(-1, keepdim=True) - 1
                outputs = model(
                    input_ids=replay_input,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = outputs.past_key_values
                del outputs
    trajectories = recorder.trajectories(token_steps)
    for record, trajectory in zip(chunk_records, trajectories):
        expected = 3 + max(0, int(record["generated_token_count"]) - 1)
        if int(trajectory.shape[0]) != expected or int(trajectory.shape[1]) != K:
            raise AssertionError(f"{record['sample_id']}: replay trajectory shape mismatch")
    return trajectories


def load_records_by_index(model: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    records = read_jsonl(records_path(model))
    records.sort(key=lambda row: int(row["original_index"]))
    if [int(row["original_index"]) for row in records] != list(range(len(records))):
        raise ValueError(f"{model}: records are not indexed 0..N-1")
    return records, {row["sample_id"]: row for row in records}


def choose_validation_ids(records: list[dict[str, Any]], count_per_split: int) -> dict[str, list[str]]:
    rng = np.random.default_rng(NODE_SET_SEED)
    result = {}
    for split in SPLITS:
        candidates = [row["sample_id"] for row in records if row["split"] == split]
        result[split] = [str(value) for value in rng.choice(candidates, size=count_per_split, replace=False).tolist()]
    return result


def command_replay_validate(args: argparse.Namespace) -> None:
    result_rows = []
    status_rows = []
    for model in args.models:
        records, record_map = load_records_by_index(model)
        selected_ids_by_split = choose_validation_ids(records, args.samples_per_split)
        selected_ids = {value for values in selected_ids_by_split.values() for value in values}
        batch_indices = sorted({int(record_map[sample_id]["batch_index"]) for sample_id in selected_ids})
        formal_data = {
            split: torch.load(formal_split_path(model, split), map_location="cpu", weights_only=False, mmap=True)
            for split in SPLITS
        }
        formal_index = {split: {sample_id: index for index, sample_id in enumerate(formal_data[split]["ids"])} for split in SPLITS}
        rows = formal.load_ordered_rows(model)
        nodes = load_selected_nodes(model)
        model_object, tokenizer = formal.load_model(model)
        model_pass = True
        try:
            for batch_index in batch_indices:
                chunk_rows = rows[batch_index * 8 : batch_index * 8 + 8]
                chunk_records = records[batch_index * 8 : batch_index * 8 + 8]
                replayed = replay_batch(model_object, tokenizer, chunk_rows, chunk_records, nodes)
                for record, replay in zip(chunk_records, replayed):
                    sample_id = record["sample_id"]
                    if sample_id not in selected_ids:
                        continue
                    split = record["split"]
                    index = formal_index[split][sample_id]
                    formal_length = int(formal_data[split]["lengths"][index])
                    formal_x = formal_data[split]["x"][index, :formal_length].float()
                    structure_equal = tuple(formal_x.shape) == tuple(replay.shape)
                    if structure_equal:
                        difference = (formal_x - replay).abs()
                        max_abs = float(difference.max())
                        mean_abs = float(difference.mean())
                        cosine = float(torch.nn.functional.cosine_similarity(formal_x.flatten(), replay.flatten(), dim=0))
                        allclose = bool(torch.allclose(formal_x, replay, atol=REPLAY_ATOL, rtol=REPLAY_RTOL))
                    else:
                        max_abs = mean_abs = float("inf")
                        cosine = float("nan")
                        allclose = False
                    token_count_equal = len(record["generated_token_ids"]) == int(record["generated_token_count"])
                    trajectory_length_equal = formal_length == int(record["trajectory_length"]) == int(replay.shape[0])
                    passed = token_count_equal and structure_equal and trajectory_length_equal and allclose
                    model_pass &= passed
                    result_rows.append({
                        "model": model,
                        "sample_id": sample_id,
                        "split": split,
                        "batch_index": batch_index,
                        "generated_token_ids_exact": True,
                        "generated_length_equal": token_count_equal,
                        "trajectory_structure_equal": structure_equal,
                        "trajectory_length_equal": trajectory_length_equal,
                        "prompt_steps_equal": structure_equal and formal_length >= 3,
                        "generation_alignment_equal": trajectory_length_equal,
                        "allclose": allclose,
                        "atol": REPLAY_ATOL,
                        "rtol": REPLAY_RTOL,
                        "max_absolute_error": max_abs,
                        "mean_absolute_error": mean_abs,
                        "cosine_similarity": cosine,
                        "status": "PASS" if passed else "FAIL",
                    })
        finally:
            del model_object, tokenizer
            torch.cuda.empty_cache()
            gc.collect()
        status_rows.append({"model": model, "stage": "selected_replay_validation", "status": "PASS" if model_pass else "FAIL"})
        if not model_pass:
            write_run_status(status_rows)
            write_replay_csv(result_rows)
            raise RuntimeError(f"{model}: Selected-128 replay validation failed; Random extraction is blocked")
    write_replay_csv(result_rows)
    write_run_status(status_rows)
    print(json.dumps(status_rows, indent=2))


def write_replay_csv(rows: list[dict[str, Any]]) -> None:
    path = OUT / "results/replay_validation.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_run_status(rows: list[dict[str, Any]], append: bool = False) -> None:
    path = OUT / "results/run_status.csv"
    old = []
    if append and path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            old = list(csv.DictReader(handle))
    keyed = {(row["model"], row["stage"]): row for row in [*old, *rows]}
    values = list(keyed.values())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "stage", "status"])
        writer.writeheader()
        writer.writerows(values)


def replay_gate_passed(model: str) -> bool:
    path = OUT / "results/replay_validation.csv"
    if not path.exists():
        return False
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["model"] == model]
    return bool(rows) and all(row["status"] == "PASS" for row in rows) and {row["split"] for row in rows} == set(SPLITS)


def batch_shard_path(model: str, batch_index: int) -> Path:
    return random_checkpoint_dir(model) / "batches" / f"batch_{batch_index:05d}.pt"


def command_extract(args: argparse.Namespace) -> None:
    model = args.model
    if not replay_gate_passed(model):
        raise RuntimeError(f"{model}: replay gate has not passed")
    rows = formal.load_ordered_rows(model)
    records, _ = load_records_by_index(model)
    nodes = load_random_nodes(model)
    node_hash = sha256_file(random_node_path(model))
    model_object, tokenizer = formal.load_model(model)
    started = time.time()
    completed = 0
    total_batches = math.ceil(len(rows) / 8)
    try:
        for batch_index in range(total_batches):
            shard_path = batch_shard_path(model, batch_index)
            if shard_path.exists():
                shard = torch.load(shard_path, map_location="cpu", weights_only=False)
                expected_ids = [str(row["id"]) for row in rows[batch_index * 8 : batch_index * 8 + 8]]
                if shard.get("ids") != expected_ids or shard.get("random_node_manifest_sha256") != node_hash:
                    raise ValueError(f"invalid existing shard: {shard_path}")
                completed += 1
                continue
            chunk_rows = rows[batch_index * 8 : batch_index * 8 + 8]
            chunk_records = records[batch_index * 8 : batch_index * 8 + 8]
            trajectories = replay_batch(model_object, tokenizer, chunk_rows, chunk_records, nodes)
            atomic_torch_save(shard_path, {
                "x": [value.to(torch.float16) for value in trajectories],
                "ids": [record["sample_id"] for record in chunk_records],
                "y": torch.tensor([record["correctness_label"] for record in chunk_records], dtype=torch.float32),
                "splits": [record["split"] for record in chunk_records],
                "generated_token_ids": [record["generated_token_ids"] for record in chunk_records],
                "generated_text_hashes": [record["generated_text_hash"] for record in chunk_records],
                "random_node_manifest_sha256": node_hash,
                "batch_index": batch_index,
            })
            completed += 1
            if completed % args.report_every == 0 or completed == total_batches:
                elapsed = time.time() - started
                rate = completed / max(elapsed, 1e-9)
                eta = (total_batches - completed) / max(rate, 1e-9)
                progress = {
                    "model": model,
                    "completed_batches": completed,
                    "total_batches": total_batches,
                    "processed_samples": min(completed * 8, len(rows)),
                    "total_samples": len(rows),
                    "elapsed_seconds": elapsed,
                    "eta_seconds": eta,
                    "last_checkpoint": str(shard_path.relative_to(ROOT)),
                    "status": "RUNNING" if completed < total_batches else "EXTRACTION_COMPLETE",
                }
                write_json(random_checkpoint_dir(model) / "progress.json", progress)
                print(json.dumps(progress), flush=True)
    finally:
        del model_object, tokenizer
        torch.cuda.empty_cache()
        gc.collect()
    assemble_model(model)
    write_run_status([{"model": model, "stage": "random_trajectory_extraction", "status": "PASS"}], append=True)


def assemble_model(model: str) -> None:
    records, _ = load_records_by_index(model)
    total_batches = math.ceil(len(records) / 8)
    shards = [batch_shard_path(model, index) for index in range(total_batches)]
    if not all(path.exists() for path in shards):
        raise RuntimeError(f"{model}: cannot assemble incomplete shards")
    feature_by_id: dict[str, torch.Tensor] = {}
    token_ids_by_id: dict[str, list[int]] = {}
    for path in shards:
        shard = torch.load(path, map_location="cpu", weights_only=False)
        for sample_id, feature, token_ids in zip(shard["ids"], shard["x"], shard["generated_token_ids"]):
            if sample_id in feature_by_id:
                raise ValueError(f"{model}: duplicate shard sample {sample_id}")
            feature_by_id[sample_id] = feature
            token_ids_by_id[sample_id] = list(map(int, token_ids))
    if set(feature_by_id) != {record["sample_id"] for record in records}:
        raise ValueError(f"{model}: missing samples during assembly")
    nodes_payload = json.loads(random_node_path(model).read_text(encoding="utf-8"))
    nodes = [{"layer": item["layer_index"], "neuron": item["node_index"]} for item in nodes_payload["nodes"]]
    for split in SPLITS:
        split_records = [record for record in records if record["split"] == split]
        max_length = max(int(feature_by_id[record["sample_id"]].shape[0]) for record in split_records)
        x = torch.zeros((len(split_records), max_length, K), dtype=torch.float16)
        mask = torch.zeros((len(split_records), max_length), dtype=torch.bool)
        for index, record in enumerate(split_records):
            feature = feature_by_id[record["sample_id"]]
            length = int(feature.shape[0])
            x[index, :length] = feature
            mask[index, :length] = True
            if token_ids_by_id[record["sample_id"]] != record["generated_token_ids"]:
                raise ValueError(f"{model}/{split}: generated token mismatch")
        payload = {
            "x": x,
            "mask": mask,
            "y": torch.tensor([record["correctness_label"] for record in split_records], dtype=torch.float32),
            "ids": [record["sample_id"] for record in split_records],
            "lengths": mask.sum(1).long(),
            "questions": [record["prompt"] for record in split_records],
            "generated_texts": [record["generated_text"] for record in split_records],
            "generated_text_hashes": [record["generated_text_hash"] for record in split_records],
            "generated_token_ids": [record["generated_token_ids"] for record in split_records],
            "split": split,
            "neurons": nodes,
            "metadata": {
                "protocol_version": "gsm8k-aligned-v2-random128-ablation",
                "source_protocol": "gsm8k-aligned-v2",
                "model_id": model,
                "node_variant": "layer_matched_random_128",
                "node_set_seed": NODE_SET_SEED,
                "random_node_manifest": str(random_node_path(model).relative_to(ROOT)),
                "random_node_manifest_sha256": sha256_file(random_node_path(model)),
                "storage_dtype": "float16",
                "feature_mode": "activation",
                "base_feature_dim": K,
                "precomputed_features": True,
                "prompt_steps": ["prompt_last", "prompt_mean", "prompt_max"],
                "generation_source": "fixed_token_replay_of_aligned_v2_generated_token_ids",
            },
        }
        atomic_torch_save(random_split_path(model, split), payload)
        print(f"assembled {model}/{split}: shape={tuple(x.shape)}", flush=True)


def command_integrity(_args: argparse.Namespace) -> None:
    report: dict[str, Any] = {"status": "PASS", "models": {}}
    id_sets_by_model = {}
    for model in MODELS:
        records, _ = load_records_by_index(model)
        record_by_split = {split: [record for record in records if record["split"] == split] for split in SPLITS}
        model_report = {"status": "PASS", "splits": {}}
        split_sets = {}
        for split in SPLITS:
            data = torch.load(random_split_path(model, split), map_location="cpu", weights_only=False, mmap=True)
            formal_data = torch.load(formal_split_path(model, split), map_location="cpu", weights_only=False, mmap=True)
            expected_records = record_by_split[split]
            ids = [str(value) for value in data["ids"]]
            split_sets[split] = set(ids)
            label_mismatch = int((data["y"] != formal_data["y"]).sum())
            token_mismatch = sum(a != b for a, b in zip(data["generated_token_ids"], [row["generated_token_ids"] for row in expected_records]))
            generated_hash_mismatch = sum(a != b for a, b in zip(data["generated_text_hashes"], formal_data["generated_text_hashes"]))
            checks = {
                "sample_count": len(ids),
                "expected_sample_count": EXPECTED_SPLITS[split],
                "missing_samples": len(set(formal_data["ids"]) - set(ids)),
                "duplicate_samples": len(ids) - len(set(ids)),
                "label_mismatch": label_mismatch,
                "generated_token_mismatch": token_mismatch,
                "generated_text_hash_mismatch": generated_hash_mismatch,
                "feature_dim": int(data["x"].shape[-1]),
                "finite": bool(torch.isfinite(data["x"]).all()),
                "mask_shape_matches": tuple(data["mask"].shape) == tuple(data["x"].shape[:2]),
            }
            passed = (
                checks["sample_count"] == checks["expected_sample_count"]
                and checks["missing_samples"] == 0
                and checks["duplicate_samples"] == 0
                and label_mismatch == token_mismatch == generated_hash_mismatch == 0
                and checks["feature_dim"] == K and checks["finite"] and checks["mask_shape_matches"]
            )
            checks["status"] = "PASS" if passed else "FAIL"
            model_report["status"] = "PASS" if model_report["status"] == "PASS" and passed else "FAIL"
            model_report["splits"][split] = checks
        overlap = sum(len(split_sets[a] & split_sets[b]) for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")))
        model_report["split_overlap"] = overlap
        if overlap:
            model_report["status"] = "FAIL"
        report["models"][model] = model_report
        id_sets_by_model[model] = split_sets
        if model_report["status"] != "PASS":
            report["status"] = "FAIL"
    write_json(OUT / "results/data_integrity_report.json", report)
    if report["status"] != "PASS":
        raise RuntimeError("Random trajectory data integrity failed")
    write_run_status([{"model": model, "stage": "data_integrity", "status": "PASS"} for model in MODELS], append=True)
    print(json.dumps(report, indent=2))


def run_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\nCOMMAND " + json.dumps(command, ensure_ascii=False) + "\n")
        handle.flush()
        result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}), see {log_path}")


def command_train(args: argparse.Namespace) -> None:
    model = args.model
    for seed in PREDICTOR_SEEDS:
        out_dir = random_model_dir(model, seed)
        if (out_dir / "dev_predictions.csv").exists() and (out_dir / "metrics.json").exists() and (out_dir / "best_gru.pt").exists():
            print(f"SKIP completed {model} seed={seed}", flush=True)
            continue
        command = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "train-gru",
            "--train-pt", str(random_split_path(model, "train")),
            "--dev-pt", str(random_split_path(model, "dev")),
            "--out-dir", str(out_dir),
            "--epochs", "40",
            "--batch-size", "64",
            "--hidden-size", "256",
            "--num-layers", "1",
            "--dropout", "0.2",
            "--lr", "0.001",
            "--weight-decay", "0.0001",
            "--grad-clip", "1.0",
            "--ece-bins", "15",
            "--feature-mode", "activation",
            "--seed", str(seed),
        ]
        run_command(command, OUT / "logs" / model / f"train_seed{seed}.log")
    write_run_status([{"model": model, "stage": "five_seed_training", "status": "PASS"}], append=True)


def combine_predictions(paths: list[Path]) -> tuple[list[str], np.ndarray, np.ndarray]:
    tables = [load_prediction_table(path) for path in paths]
    ids = list(tables[0])
    if any(set(table) != set(ids) for table in tables):
        raise ValueError("prediction files have different sample ID sets")
    labels = []
    probabilities = []
    for sample_id in ids:
        sample_labels = [int(table[sample_id]["y"]) for table in tables]
        if len(set(sample_labels)) != 1:
            raise ValueError(f"prediction label mismatch for {sample_id}")
        labels.append(sample_labels[0])
        probabilities.append(float(np.mean([float(table[sample_id]["p_correct"]) for table in tables])))
    return ids, np.asarray(labels, dtype=np.int64), np.asarray(probabilities, dtype=np.float64)


def command_ensemble(args: argparse.Namespace) -> None:
    model = args.model
    seed_dirs = [random_model_dir(model, seed) for seed in PREDICTOR_SEEDS]
    dev_ids, dev_y, dev_p = combine_predictions([path / "dev_predictions.csv" for path in seed_dirs])
    threshold = youden_threshold(dev_y, dev_p)
    ensemble_dir = OUT / "results/ensembles" / model / "gsm8k"
    write_predictions(ensemble_dir / "ensemble_dev_predictions.csv", dev_ids, dev_y, dev_p, threshold)
    checkpoints = [path / "best_gru.pt" for path in seed_dirs]
    test_csv = ensemble_dir / "ensemble_test_predictions.csv"
    test_json = ensemble_dir / "ensemble_test_metrics.json"
    command = [
        sys.executable, str(TRAIN_SCRIPT), "ensemble-test",
        "--test-pt", str(random_split_path(model, "test")),
        "--checkpoints", *[str(path) for path in checkpoints],
        "--threshold", str(threshold),
        "--batch-size", "64",
        "--ece-bins", "15",
        "--out-csv", str(test_csv),
        "--out-json", str(test_json),
    ]
    run_command(command, OUT / "logs" / model / "ensemble_test.log")
    test_payload = json.loads(test_json.read_text(encoding="utf-8"))
    payload = {
        "model": model,
        "dataset": "gsm8k",
        "node_variant": "layer_matched_random_128",
        "node_set_seed": NODE_SET_SEED,
        "predictor_seeds": list(PREDICTOR_SEEDS),
        "aggregation": "mean_p_correct",
        "threshold_source": "ensemble_dev_youden",
        "dev": metrics(dev_y, dev_p, threshold),
        "test": test_payload["test"],
        "best_epochs": [int(torch.load(path, map_location="cpu", weights_only=False)["epoch"]) for path in checkpoints],
        "n_train": EXPECTED_SPLITS["train"],
        "n_dev": EXPECTED_SPLITS["dev"],
        "n_test": EXPECTED_SPLITS["test"],
        "checkpoints": [str(path.relative_to(ROOT)) for path in checkpoints],
    }
    write_json(ensemble_dir / "ensemble_metrics.json", payload)
    write_run_status([{"model": model, "stage": "ensemble_and_test", "status": "PASS"}], append=True)
    print(json.dumps(payload, indent=2))


def metric_section(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    section = payload.get("test", payload)
    return {key: float(section[key]) for key in ("auroc", "auprc", "accuracy", "f1", "brier", "ece", "threshold")}


def command_aggregate(_args: argparse.Namespace) -> None:
    ensemble_rows = []
    per_seed_rows = []
    comparison_rows = []
    for model in MODELS:
        ensemble_path = OUT / "results/ensembles" / model / "gsm8k/ensemble_metrics.json"
        payload = json.loads(ensemble_path.read_text(encoding="utf-8"))
        random_test = payload["test"]
        selected_test = metric_section(formal_metric_path(model))
        random_row = {
            "model": model, "dataset": "gsm8k", "node_variant": "random_128",
            "node_set_seed": NODE_SET_SEED, **{key: random_test[key] for key in ("auroc", "auprc", "accuracy", "f1", "brier", "ece")},
            "dev_threshold": payload["dev"]["threshold"], "best_epochs": json.dumps(payload["best_epochs"]), "n_test": payload["n_test"],
        }
        ensemble_rows.append(random_row)
        comparison_rows.extend([
            {"model": model, "dataset": "gsm8k", "node_variant": "selected_128", "node_set_seed": "", **selected_test,
             "dev_threshold": selected_test["threshold"], "best_epochs": "formal_existing", "n_test": EXPECTED_SPLITS["test"],
             **{f"delta_{key.upper()}_vs_selected": 0.0 for key in ("auroc", "auprc", "accuracy", "f1", "brier", "ece")}},
            {**random_row, **{f"delta_{key.upper()}_vs_selected": float(random_test[key]) - float(selected_test[key]) for key in ("auroc", "auprc", "accuracy", "f1", "brier", "ece")}},
        ])
        for seed in PREDICTOR_SEEDS:
            seed_dir = random_model_dir(model, seed)
            seed_metrics = json.loads((seed_dir / "metrics.json").read_text(encoding="utf-8"))
            checkpoint = torch.load(seed_dir / "best_gru.pt", map_location="cpu", weights_only=False)
            per_seed_rows.append({
                "model": model, "dataset": "gsm8k", "node_variant": "random_128", "node_set_seed": NODE_SET_SEED,
                "predictor_seed": seed, "best_epoch": int(checkpoint["epoch"]),
                **{f"dev_{key}": seed_metrics["dev"][key] for key in ("auroc", "auprc", "accuracy", "f1", "brier", "ece", "threshold")},
            })
    write_csv(OUT / "results/per_seed_metrics.csv", per_seed_rows)
    write_csv(OUT / "results/ensemble_metrics.csv", ensemble_rows)
    write_csv(OUT / "results/selected_vs_random.csv", comparison_rows)
    macro_rows = []
    for variant in ("selected_128", "random_128"):
        values = [row for row in comparison_rows if row["node_variant"] == variant]
        macro_rows.append({
            "node_variant": variant,
            **{key: float(np.mean([float(row[key]) for row in values])) for key in ("auroc", "auprc", "accuracy", "f1", "brier", "ece")},
        })
    selected_macro, random_macro = macro_rows
    macro_rows.append({
        "node_variant": "random_minus_selected",
        **{key: random_macro[key] - selected_macro[key] for key in ("auroc", "auprc", "accuracy", "f1", "brier", "ece")},
    })
    write_csv(OUT / "results/macro_summary.csv", macro_rows)
    write_readme(comparison_rows, macro_rows)
    write_run_status([{"model": model, "stage": "final_aggregation", "status": "PASS"} for model in MODELS], append=True)
    print(json.dumps({"ensemble": ensemble_rows, "macro": macro_rows}, indent=2))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_readme(comparisons: list[dict[str, Any]], macro_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Random-128 Sparse Observation Node Ablation",
        "",
        "This directory contains only the GSM8K aligned-v2 layer-matched Random-128 ablation.",
        "No answer was regenerated: all trajectories use fixed-token replay of the formal aligned-v2 generated token IDs.",
        "Random nodes use node-set seed 20260816 and match each model's Selected-128 per-layer quota with zero overlap.",
        "Predictors use seeds 13, 21, 42, 87, and 100; checkpoints and thresholds are selected on GSM8K dev only.",
        "",
        "## Test comparison",
        "",
        "| Model | Variant | AUROC | AUPRC | Accuracy | F1 | Brier | ECE |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['model']} | {row['node_variant']} | {float(row['auroc']):.4f} | {float(row['auprc']):.4f} | "
            f"{float(row['accuracy']):.4f} | {float(row['f1']):.4f} | {float(row['brier']):.4f} | {float(row['ece']):.4f} |"
        )
    lines.extend(["", "## Macro average", "", "```json", json.dumps(macro_rows, indent=2), "```", ""])
    path = OUT / "README.md"
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, path)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    command = sub.add_parser("preflight")
    command.set_defaults(func=command_preflight)
    command = sub.add_parser("sample-nodes")
    command.set_defaults(func=command_sample_nodes)
    command = sub.add_parser("replay-validate")
    command.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    command.add_argument("--samples-per-split", type=int, default=2)
    command.set_defaults(func=command_replay_validate)
    command = sub.add_parser("extract")
    command.add_argument("--model", choices=MODELS, required=True)
    command.add_argument("--report-every", type=int, default=25)
    command.set_defaults(func=command_extract)
    command = sub.add_parser("integrity")
    command.set_defaults(func=command_integrity)
    command = sub.add_parser("train")
    command.add_argument("--model", choices=MODELS, required=True)
    command.set_defaults(func=command_train)
    command = sub.add_parser("ensemble")
    command.add_argument("--model", choices=MODELS, required=True)
    command.set_defaults(func=command_ensemble)
    command = sub.add_parser("aggregate")
    command.set_defaults(func=command_aggregate)
    return result


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
