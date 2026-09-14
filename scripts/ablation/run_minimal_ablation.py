#!/usr/bin/env python3
"""Run the non-random minimum ablations without modifying formal artifacts."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(os.environ.get("CLOUD1_ARTIFACT_ROOT", Path(__file__).resolve().parents[2]))
OUT = Path(os.environ.get("CLOUD1_ABLATION_OUT", ROOT / "ablation_minimal"))
MODELS = (
    "qwen_2.5_7b_instruct",
    "ministral_8b_instruct",
    "mistral_7b_instruct",
    "llama3.1_8b_chat",
)
DATASETS = ("trivia_qa_2_60k", "gsm8k")
TARGETS = ("notable_people", "cities_10k", "math_operations_6k", "medals_9k", "gsm8k")
SEEDS = (13, 21, 42, 87, 100)
EPS = 1e-12


def command_train_gru(*_args: Any, **_kwargs: Any) -> None:
    """Compatibility symbol for argparse Namespaces stored by the legacy CLI."""


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def source_root(model: str) -> Path:
    return ROOT / "02_模型实验过程与结果" / model / "正式结果" / "Method_A_Direct_Transfer_no_scalar_raw_activation"


def trajectory_dir(model: str, dataset: str) -> Path:
    if dataset == "gsm8k":
        return ROOT / "gsm8k_aligned_v2" / "trajectories" / model / "trivia_top128" / "splits"
    if model == "qwen_2.5_7b_instruct":
        return (
            ROOT
            / "outputs/multimodel_trivia10k_top128/frozen_neuron_adaptation_no_scalar_raw_activation"
            / model
            / "trivia_qa_2_60k/features"
        )
    return source_root(model) / "features/trivia_qa_2_60k"


def source_checkpoints(model: str) -> list[Path]:
    base = source_root(model) / "models"
    return [base / f"trivia_predictor_seed{seed}/best_gru.pt" for seed in SEEDS]


def topk_path(model: str) -> Path:
    if model == "qwen_2.5_7b_instruct":
        return ROOT / "02_模型实验过程与结果/qwen_2.5_7b_instruct/共享过程文件/trivia10k_top128_neurons/topk_mlp_neurons_k128_trivia10k.json"
    return source_root(model) / "neurons/topk_mlp_neurons_k128_trivia10k.json"


def full_metric_path(model: str, dataset: str) -> Path:
    if dataset == "gsm8k":
        return ROOT / "gsm8k_aligned_v2" / "metrics" / model / "trivia_top128/ensemble_test_metrics.json"
    return source_root(model) / "direct_eval/trivia_qa_2_60k/ensemble_test_metrics.json"


def full_checkpoint_dir(model: str, dataset: str) -> Path:
    if dataset == "gsm8k":
        return ROOT / "gsm8k_aligned_v2" / "checkpoints" / model / "trivia_top128"
    return source_root(model) / "models"


def full_checkpoint(model: str, dataset: str, seed: int) -> Path:
    base = full_checkpoint_dir(model, dataset)
    name = f"gru_seed{seed}" if dataset == "gsm8k" else f"trivia_predictor_seed{seed}"
    return base / name / "best_gru.pt"


def target_feature(model: str, dataset: str, split: str) -> Path:
    if dataset == "gsm8k":
        return trajectory_dir(model, dataset) / f"{split}.pt"
    return (
        ROOT
        / "outputs/multimodel_trivia10k_top128/frozen_neuron_adaptation_no_scalar_raw_activation"
        / model
        / dataset
        / "features"
        / f"{split}.pt"
    )


def direct_test_predictions(model: str, dataset: str) -> Path:
    if dataset == "gsm8k":
        return ROOT / "gsm8k_aligned_v2" / "metrics" / model / "method_a_trivia_to_gsm8k/ensemble_test_predictions.csv"
    return source_root(model) / "direct_eval" / dataset / "ensemble_test_predictions.csv"


def direct_test_metrics(model: str, dataset: str) -> Path:
    if dataset == "gsm8k":
        return ROOT / "gsm8k_aligned_v2" / "metrics" / model / "method_a_trivia_to_gsm8k/ensemble_test_metrics.json"
    return source_root(model) / "direct_eval" / dataset / "ensemble_test_metrics.json"


def adapted_test_metrics(model: str, dataset: str) -> Path:
    if dataset == "gsm8k":
        return full_metric_path(model, dataset)
    return (
        ROOT
        / "outputs/multimodel_trivia10k_top128/frozen_neuron_adaptation_no_scalar_raw_activation"
        / model
        / dataset
        / "ensemble_test_metrics.json"
    )


def source_dev_metrics(model: str) -> Path:
    return source_root(model) / "source_ensemble_dev_metrics.json"


def load_raw(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def labels_and_ids(data: dict[str, Any]) -> tuple[torch.Tensor, list[str], list[str]]:
    y = data["y"].float().contiguous()
    ids = [str(value) for value in data.get("ids", range(len(y)))]
    questions = [str(value) for value in data.get("questions", [""] * len(y))]
    return y, ids, questions


def prepare_prefill(path: Path, limit: int | None = None) -> dict[str, Any]:
    data = load_raw(path)
    n = len(data["y"]) if limit is None else min(limit, len(data["y"]))
    if data["x"].shape[-1] != 128:
        raise ValueError(f"{path}: expected feature dimension 128")
    if data["x"].shape[1] < 3 or not bool(data["mask"][:n, :3].all()):
        raise ValueError(f"{path}: first three prompt summary states are not all valid")
    y, ids, questions = labels_and_ids(data)
    return {
        "x": data["x"][:n, :3].float().contiguous(),
        "lengths": torch.full((n,), 3, dtype=torch.long),
        "y": y[:n],
        "ids": ids[:n],
        "questions": questions[:n],
    }


def masked_mean_chunks(x: torch.Tensor, mask: torch.Tensor, n: int, chunk_size: int = 512) -> torch.Tensor:
    chunks = []
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        xb = x[start:end].float()
        mb = mask[start:end].unsqueeze(-1).float()
        chunks.append((xb * mb).sum(dim=1) / mb.sum(dim=1).clamp_min(1.0))
    return torch.cat(chunks, dim=0).contiguous()


def prepare_meanpool(path: Path, limit: int | None = None) -> dict[str, Any]:
    data = load_raw(path)
    n = len(data["y"]) if limit is None else min(limit, len(data["y"]))
    if data["x"].shape[-1] != 128:
        raise ValueError(f"{path}: expected feature dimension 128")
    y, ids, questions = labels_and_ids(data)
    return {
        "x": masked_mean_chunks(data["x"], data["mask"], n),
        "y": y[:n],
        "ids": ids[:n],
        "questions": questions[:n],
    }


class PrefillGRU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gru = nn.GRU(128, 256, num_layers=1, batch_first=True, bidirectional=False)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(256, 1))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        return self.head(hidden[-1]).squeeze(-1)


class MeanPoolMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def ece_score(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        selected = (p >= left) & ((p <= right) if index == bins - 1 else (p < right))
        if selected.any():
            total += float(selected.mean()) * abs(float(y[selected].mean()) - float(p[selected].mean()))
    return total


def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y, p)
    valid = np.isfinite(thresholds)
    indices = np.where(valid)[0]
    return float(thresholds[indices[np.argmax((tpr - fpr)[indices])]])


def metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (p >= threshold).astype(np.int64)
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
        "ece": float(ece_score(y, p)),
        "threshold": float(threshold),
    }


def write_predictions(path: Path, data: dict[str, Any], p: np.ndarray, threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["idx", "sample_id", "y", "p_correct", "risk", "pred", "question"])
        writer.writeheader()
        for idx, (sample_id, y, prob, question) in enumerate(zip(data["ids"], data["y"], p, data["questions"])):
            writer.writerow({
                "idx": idx,
                "sample_id": sample_id,
                "y": int(y),
                "p_correct": float(prob),
                "risk": float(1.0 - prob),
                "pred": int(prob >= threshold),
                "question": question,
            })
    os.replace(tmp, path)


@torch.inference_mode()
def predict(model: nn.Module, variant: str, data: dict[str, Any], device: torch.device, batch_size: int = 64) -> np.ndarray:
    model.eval()
    if variant == "prefill_only":
        dataset = TensorDataset(data["x"], data["lengths"])
    else:
        dataset = TensorDataset(data["x"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    output = []
    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        logits = model(x, batch[1].to(device) if variant == "prefill_only" else None) if variant == "prefill_only" else model(x)
        output.append(torch.sigmoid(logits).cpu())
    return torch.cat(output).numpy()


def train_one(
    variant: str,
    train: dict[str, Any],
    dev: dict[str, Any],
    test: dict[str, Any],
    seed: int,
    out_dir: Path,
    epochs: int,
) -> None:
    completed = out_dir / "metrics.json"
    if completed.exists() and (out_dir / "dev_predictions.csv").exists() and (out_dir / "test_predictions.csv").exists():
        print(f"SKIP completed {out_dir}", flush=True)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (PrefillGRU() if variant == "prefill_only" else MeanPoolMLP()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    if variant == "prefill_only":
        train_ds = TensorDataset(train["x"], train["lengths"], train["y"])
    else:
        train_ds = TensorDataset(train["x"], train["y"])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=64, shuffle=True, generator=generator, pin_memory=True)
    best_auroc = -math.inf
    best_path = out_dir / ("best_gru.pt" if variant == "prefill_only" else "best_meanpool_mlp.pt")
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for batch in loader:
            if variant == "prefill_only":
                x, lengths, y = batch
                logits = model(x.to(device, non_blocking=True), lengths.to(device, non_blocking=True))
            else:
                x, y = batch
                logits = model(x.to(device, non_blocking=True))
            y = y.to(device, non_blocking=True)
            loss = loss_fn(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.item()) * len(y)
            seen += len(y)
        dev_p = predict(model, variant, dev, device)
        dev_y = dev["y"].numpy().astype(np.int64)
        threshold = youden_threshold(dev_y, dev_p)
        dev_metrics = metrics(dev_y, dev_p, threshold)
        row = {"epoch": epoch, "train_loss": loss_sum / seen, **{f"dev_{key}": value for key, value in dev_metrics.items()}}
        history.append(row)
        print(json.dumps({"variant": variant, "seed": seed, **row}), flush=True)
        if dev_metrics["auroc"] > best_auroc:
            best_auroc = dev_metrics["auroc"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "variant": variant,
                "seed": seed,
                "epoch": epoch,
                "dev_metrics": dev_metrics,
                "config": {
                    "input_dim": 128, "hidden_size": 256 if variant == "prefill_only" else None,
                    "num_layers": 1 if variant == "prefill_only" else None,
                    "dropout": 0.2, "epochs": epochs, "batch_size": 64, "lr": 1e-3,
                    "weight_decay": 1e-4, "grad_clip": 1.0, "ece_bins": 15,
                },
            }, best_path)
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    dev_p = predict(model, variant, dev, device)
    test_p = predict(model, variant, test, device)
    dev_y = dev["y"].numpy().astype(np.int64)
    test_y = test["y"].numpy().astype(np.int64)
    threshold = youden_threshold(dev_y, dev_p)
    result = {
        "variant": variant,
        "seed": seed,
        "best_epoch": int(checkpoint["epoch"]),
        "selection_metric": "dev_auroc",
        "threshold_source": "seed_dev_youden",
        "n_train": len(train["y"]), "n_dev": len(dev_y), "n_test": len(test_y),
        "n_answer_correct_test": int(test_y.sum()),
        "n_answer_incorrect_test": int(len(test_y) - test_y.sum()),
        "dev": metrics(dev_y, dev_p, threshold),
        "test": metrics(test_y, test_p, threshold),
    }
    write_predictions(out_dir / "dev_predictions.csv", dev, dev_p, threshold)
    write_predictions(out_dir / "test_predictions.csv", test, test_p, threshold)
    write_json(out_dir / "training_history.json", history)
    write_json(completed, result)
    print(json.dumps(result, indent=2), flush=True)
    del model, optimizer
    torch.cuda.empty_cache()
    gc.collect()


def load_variant_data(variant: str, model: str, dataset: str, limit: int | None = None) -> tuple[dict[str, Any], ...]:
    prepare = prepare_prefill if variant == "prefill_only" else prepare_meanpool
    base = trajectory_dir(model, dataset)
    return tuple(prepare(base / f"{split}.pt", limit) for split in ("train", "dev", "test"))


def command_train_group(args: argparse.Namespace) -> None:
    train, dev, test = load_variant_data(args.variant, args.model, args.dataset, args.limit)
    root = OUT / "smoke" if args.smoke else OUT / "checkpoints"
    for seed in args.seeds:
        out_dir = root / args.variant / args.model / args.dataset / f"seed{seed}"
        train_one(args.variant, train, dev, test, seed, out_dir, args.epochs)


def read_predictions(path: Path) -> dict[str, dict[str, str]]:
    rows = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = row.get("sample_id") or row["idx"]
            if key in rows:
                raise ValueError(f"duplicate prediction key {key} in {path}")
            rows[key] = row
    return rows


def ensemble_variant(variant: str, model: str, dataset: str) -> dict[str, Any]:
    seed_dirs = [OUT / "checkpoints" / variant / model / dataset / f"seed{seed}" for seed in SEEDS]
    dev_tables = [read_predictions(path / "dev_predictions.csv") for path in seed_dirs]
    test_tables = [read_predictions(path / "test_predictions.csv") for path in seed_dirs]
    dev_keys = list(dev_tables[0])
    test_keys = list(test_tables[0])
    if any(set(table) != set(dev_keys) for table in dev_tables) or any(set(table) != set(test_keys) for table in test_tables):
        raise ValueError(f"prediction alignment failure for {variant}/{model}/{dataset}")

    def combine(tables: list[dict[str, dict[str, str]]], keys: list[str]) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
        y, p, questions = [], [], []
        for key in keys:
            labels = [int(table[key]["y"]) for table in tables]
            if len(set(labels)) != 1:
                raise ValueError(f"label mismatch at {key}")
            y.append(labels[0])
            p.append(np.mean([float(table[key]["p_correct"]) for table in tables]))
            questions.append(tables[0][key].get("question", ""))
        return np.asarray(y), np.asarray(p), keys, questions

    dev_y, dev_p, dev_ids, dev_questions = combine(dev_tables, dev_keys)
    test_y, test_p, test_ids, test_questions = combine(test_tables, test_keys)
    threshold = youden_threshold(dev_y, dev_p)
    out_dir = OUT / "results" / "ensembles" / variant / model / dataset
    write_predictions(out_dir / "ensemble_dev_predictions.csv", {"ids": dev_ids, "questions": dev_questions, "y": torch.from_numpy(dev_y)}, dev_p, threshold)
    write_predictions(out_dir / "ensemble_test_predictions.csv", {"ids": test_ids, "questions": test_questions, "y": torch.from_numpy(test_y)}, test_p, threshold)
    result = {
        "variant": variant, "model": model, "dataset": dataset,
        "seeds": list(SEEDS), "aggregation": "arithmetic_mean_p_correct",
        "threshold_source": "ensemble_dev_youden",
        "n_train": int(json.loads((seed_dirs[0] / "metrics.json").read_text())["n_train"]),
        "n_dev": len(dev_y), "n_test": len(test_y),
        "n_answer_correct_test": int(test_y.sum()), "n_answer_incorrect_test": int(len(test_y) - test_y.sum()),
        "dev": metrics(dev_y, dev_p, threshold), "test": metrics(test_y, test_p, threshold),
    }
    write_json(out_dir / "ensemble_metrics.json", result)
    return result


def command_ensemble(args: argparse.Namespace) -> None:
    result = ensemble_variant(args.variant, args.model, args.dataset)
    print(json.dumps(result, indent=2), flush=True)


def load_checkpoint_predictor(path: Path, device: torch.device) -> nn.Module:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    args = checkpoint.get("args", {})
    if (checkpoint.get("feature_mode") or args.get("feature_mode")) != "activation":
        raise ValueError(f"{path}: expected activation-only checkpoint")
    model = PrefillGRU()
    model.gru = nn.GRU(
        int(checkpoint.get("input_dim", 128)), int(args.get("hidden_size", 256)),
        num_layers=int(args.get("num_layers", 1)), batch_first=True,
        bidirectional=bool(args.get("bidirectional", False)),
    )
    model.head = nn.Sequential(nn.Dropout(float(args.get("dropout", 0.2))), nn.Linear(int(args.get("hidden_size", 256)), 1))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)


@torch.inference_mode()
def predict_full_checkpoint(path: Path, trajectory_path: Path, batch_size: int = 64) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    raw = load_raw(trajectory_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint_predictor(path, device)
    dataset = TensorDataset(raw["x"], raw["lengths"], raw["y"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    probs, labels = [], []
    model.eval()
    for x, lengths, y in loader:
        logits = model(x.float().to(device, non_blocking=True), lengths.to(device, non_blocking=True))
        probs.append(torch.sigmoid(logits).cpu())
        labels.append(y.cpu())
    del model
    torch.cuda.empty_cache()
    return (
        torch.cat(labels).numpy().astype(np.int64), torch.cat(probs).numpy(),
        [str(value) for value in raw["ids"]], [str(value) for value in raw.get("questions", [""] * len(raw["y"]))],
    )


def load_metric_section(path: Path, key: str = "test") -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    section = payload.get(key, payload)
    return {name: float(section[name]) for name in ("auroc", "auprc", "accuracy", "f1", "brier", "ece", "threshold")}


def command_threshold_only(args: argparse.Namespace) -> None:
    model, dataset = args.model, args.dataset
    out_dir = OUT / "results" / "threshold_only" / model / dataset
    completed = out_dir / "threshold_only_metrics.json"
    if completed.exists():
        print(f"SKIP completed {completed}")
        return
    dev_probs = []
    dev_y_ref = None
    dev_ids = dev_questions = None
    for checkpoint in source_checkpoints(model):
        y, p, ids, questions = predict_full_checkpoint(checkpoint, target_feature(model, dataset, "dev"))
        if dev_y_ref is None:
            dev_y_ref, dev_ids, dev_questions = y, ids, questions
        elif not np.array_equal(dev_y_ref, y):
            raise ValueError("target dev labels changed across source checkpoints")
        dev_probs.append(p)
    assert dev_y_ref is not None and dev_ids is not None and dev_questions is not None
    dev_p = np.stack(dev_probs).mean(axis=0)
    target_threshold = youden_threshold(dev_y_ref, dev_p)

    test_table = read_predictions(direct_test_predictions(model, dataset))
    keys = list(test_table)
    test_y = np.asarray([int(test_table[key]["y"]) for key in keys])
    test_p = np.asarray([float(test_table[key]["p_correct"]) for key in keys])
    questions = [test_table[key].get("question", "") for key in keys]
    source_payload = json.loads(source_dev_metrics(model).read_text(encoding="utf-8"))
    if "threshold" in source_payload:
        source_threshold = float(source_payload["threshold"])
    else:
        source_threshold = float(source_payload["dev"]["threshold"])
    e1 = metrics(test_y, test_p, source_threshold)
    e2 = metrics(test_y, test_p, target_threshold)
    for key in ("auroc", "auprc", "brier", "ece"):
        if abs(e1[key] - e2[key]) > 1e-12:
            raise AssertionError(f"E1/E2 {key} changed for {model}/{dataset}")
    formal_e1 = load_metric_section(direct_test_metrics(model, dataset))
    for key in ("auroc", "auprc", "brier", "ece"):
        if abs(e1[key] - formal_e1[key]) > 5e-6:
            raise AssertionError(f"recomputed E1 {key} differs from formal result for {model}/{dataset}")
    e3 = load_metric_section(adapted_test_metrics(model, dataset))
    write_predictions(out_dir / "source_predictor_target_dev_predictions.csv", {"ids": dev_ids, "questions": dev_questions, "y": torch.from_numpy(dev_y_ref)}, dev_p, target_threshold)
    write_predictions(out_dir / "threshold_only_test_predictions.csv", {"ids": keys, "questions": questions, "y": torch.from_numpy(test_y)}, test_p, target_threshold)
    result = {
        "model": model, "dataset": dataset,
        "probability_identity_assertion": "PASS",
        "E1_direct_transfer": e1,
        "E2_threshold_only_adaptation": e2,
        "E3_fixed_node_adaptation": e3,
        "accuracy_delta_E2_minus_E1": e2["accuracy"] - e1["accuracy"],
        "f1_delta_E2_minus_E1": e2["f1"] - e1["f1"],
        "source_threshold": source_threshold,
        "target_dev_threshold": target_threshold,
        "test_predictions_source": str(direct_test_predictions(model, dataset).relative_to(ROOT)),
    }
    write_json(completed, result)
    print(json.dumps(result, indent=2), flush=True)


def command_collect_full(args: argparse.Namespace) -> None:
    model, dataset = args.model, args.dataset
    out_dir = OUT / "results/full_references" / model / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    formal_payload = json.loads(full_metric_path(model, dataset).read_text(encoding="utf-8"))
    recorded_test_pt = formal_payload.get("test_pt")
    if recorded_test_pt:
        candidate = Path(recorded_test_pt)
        test_path = candidate if candidate.is_absolute() else ROOT / candidate
    else:
        test_path = trajectory_dir(model, dataset) / "test.pt"
    if not test_path.exists():
        raise FileNotFoundError(f"formal Full test trajectory is missing: {test_path}")
    split_sizes = {
        split: len(load_raw(trajectory_dir(model, dataset) / f"{split}.pt")["y"])
        for split in ("train", "dev", "test")
    }
    per_seed = []
    probability_sets = []
    y_ref = ids = questions = None
    for seed in SEEDS:
        checkpoint = full_checkpoint(model, dataset, seed)
        y, p, sample_ids, sample_questions = predict_full_checkpoint(checkpoint, test_path)
        checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        dev_metrics = checkpoint_data.get("dev_metrics", {})
        seed_threshold = float(dev_metrics["threshold"])
        row = {
            "variant": "full", "model": model, "dataset": dataset, "seed": seed,
            "best_epoch": int(checkpoint_data["epoch"]), "dev": dev_metrics,
            "test": metrics(y, p, seed_threshold),
            "n_train": split_sizes["train"], "n_dev": split_sizes["dev"],
            "n_test": len(y), "n_answer_correct_test": int(y.sum()),
            "n_answer_incorrect_test": int(len(y) - y.sum()),
        }
        write_json(out_dir / f"seed{seed}_metrics.json", row)
        per_seed.append(row)
        probability_sets.append(p)
        if y_ref is None:
            y_ref, ids, questions = y, sample_ids, sample_questions
        elif not np.array_equal(y_ref, y):
            raise ValueError("Full labels differ across checkpoints")
    formal = formal_payload
    formal_test = formal.get("test", formal)
    ensemble_threshold = float(formal_test["threshold"])
    ensemble_p = np.stack(probability_sets).mean(axis=0)
    calculated = metrics(y_ref, ensemble_p, ensemble_threshold)
    for key in ("auroc", "auprc", "accuracy", "f1", "brier", "ece"):
        if abs(calculated[key] - float(formal_test[key])) > 5e-6:
            raise AssertionError(f"Full {model}/{dataset} {key} mismatch: {calculated[key]} vs {formal_test[key]}")
    result = {
        "variant": "full", "model": model, "dataset": dataset,
        "formal_metric_path": str(full_metric_path(model, dataset).relative_to(ROOT)),
        "formal_test_trajectory_path": str(test_path.relative_to(ROOT)),
        "ablation_input_trajectory_path": str((trajectory_dir(model, dataset) / "test.pt").relative_to(ROOT)),
        "test": calculated, "n_train": split_sizes["train"],
        "n_dev": split_sizes["dev"], "n_test": len(y_ref),
        "n_answer_correct_test": int(y_ref.sum()), "n_answer_incorrect_test": int(len(y_ref) - y_ref.sum()),
        "per_seed_count": len(per_seed), "integrity": "PASS",
    }
    if "dev" in formal:
        result["dev"] = {
            key: float(formal["dev"][key])
            for key in ("auroc", "auprc", "accuracy", "f1", "brier", "ece", "threshold")
        }
    write_predictions(out_dir / "ensemble_test_predictions.csv", {"ids": ids, "questions": questions, "y": torch.from_numpy(y_ref)}, ensemble_p, ensemble_threshold)
    write_json(out_dir / "ensemble_metrics.json", result)
    print(json.dumps(result, indent=2), flush=True)


def command_preflight(args: argparse.Namespace) -> None:
    if (ROOT / "gsm8k_aligned_v2/STATUS").read_text().strip() != "formal":
        raise RuntimeError("GSM8K aligned-v2 is not formal")
    report = {
        "status": "PASS",
        "repository_root": str(ROOT),
        "code_hash": sha256_file(Path(__file__)),
        "git_commit": None,
        "models": {},
        "isolation": {
            "gsm8k_required_root": "gsm8k_aligned_v2",
            "gsm8k_required_route": "trivia_top128",
            "excluded": ["legacy GSM8K", "gsm_top128", "stopped_by_scope_change experiments 4-7"],
        },
        "checksums": {},
        "known_integrity_notes": [
            "TriviaQA formal trajectories do not store generated_token_ids.",
            "GSM8K Random-128 is handled by the separate retained formal runner.",
            "Legacy GSM8K artifacts are superseded and are not inputs.",
        ],
    }
    critical_paths = [ROOT / "gsm8k_aligned_v2/manifests/scoped_formalization_manifest.json"]
    for model in MODELS:
        model_info = {"model_path": str((ROOT / "models" / model).resolve()), "top128_path": str(topk_path(model)), "datasets": {}}
        nodes = json.loads(topk_path(model).read_text(encoding="utf-8"))
        if isinstance(nodes, dict):
            nodes = nodes.get("neurons", nodes.get("topk", []))
        if len(nodes) != 128:
            raise ValueError(f"{model}: Top-128 file contains {len(nodes)} nodes")
        model_info["top128_count"] = 128
        critical_paths.append(topk_path(model))
        for dataset in DATASETS:
            base = trajectory_dir(model, dataset)
            split_info = {}
            id_sets = {}
            for split in ("train", "dev", "test"):
                path = base / f"{split}.pt"
                data = load_raw(path)
                if data["x"].shape[-1] != 128 or "scalar_features" in data:
                    raise ValueError(f"invalid trajectory schema: {path}")
                ids_local = {str(value) for value in data["ids"]}
                id_sets[split] = ids_local
                split_info[split] = {
                    "path": str(path.relative_to(ROOT)), "n": len(data["y"]),
                    "positive": int(data["y"].sum()), "negative": int(len(data["y"]) - data["y"].sum()),
                    "shape": list(data["x"].shape), "dtype": str(data["x"].dtype),
                }
                critical_paths.append(path)
            overlap = sum(len(id_sets[a] & id_sets[b]) for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")))
            if overlap:
                raise ValueError(f"split overlap for {model}/{dataset}: {overlap}")
            checkpoints = [full_checkpoint(model, dataset, seed) for seed in SEEDS]
            if not all(path.exists() for path in checkpoints) or not full_metric_path(model, dataset).exists():
                raise FileNotFoundError(f"incomplete Full artifacts for {model}/{dataset}")
            critical_paths.extend(checkpoints)
            critical_paths.append(full_metric_path(model, dataset))
            model_info["datasets"][dataset] = {
                "splits": split_info, "split_overlap": overlap,
                "full_checkpoints": [str(path.relative_to(ROOT)) for path in checkpoints],
                "full_metric": str(full_metric_path(model, dataset).relative_to(ROOT)),
                "full_complete": True,
            }
        report["models"][model] = model_info
    for path in critical_paths:
        report["checksums"][str(path.relative_to(ROOT))] = sha256_file(path)
    write_json(OUT / "manifests/preflight_report.json", report)
    print(json.dumps({"status": report["status"], "files_hashed": len(report["checksums"])}, indent=2))


def flatten_metric(prefix: str, section: dict[str, Any], row: dict[str, Any]) -> None:
    for key in ("auroc", "auprc", "accuracy", "f1", "brier", "ece", "threshold"):
        row[f"{prefix}_{key}"] = section.get(key)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def command_aggregate(args: argparse.Namespace) -> None:
    per_seed_rows = []
    ensemble_rows = []
    for variant in ("full", "prefill_only", "meanpool_mlp"):
        for model in MODELS:
            for dataset in DATASETS:
                if variant == "full":
                    base = OUT / "results/full_references" / model / dataset
                    metric_path = base / "ensemble_metrics.json"
                    seed_paths = [base / f"seed{seed}_metrics.json" for seed in SEEDS]
                else:
                    base = OUT / "results/ensembles" / variant / model / dataset
                    metric_path = base / "ensemble_metrics.json"
                    seed_paths = [OUT / "checkpoints" / variant / model / dataset / f"seed{seed}/metrics.json" for seed in SEEDS]
                ensemble = json.loads(metric_path.read_text(encoding="utf-8"))
                row = {"variant": variant, "model": model, "dataset": dataset}
                if "dev" in ensemble:
                    flatten_metric("dev", ensemble["dev"], row)
                flatten_metric("test", ensemble["test"], row)
                row.update({key: ensemble.get(key) for key in ("n_train", "n_dev", "n_test", "n_answer_correct_test", "n_answer_incorrect_test")})
                ensemble_rows.append(row)
                for seed, path in zip(SEEDS, seed_paths):
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    seed_row = {"variant": variant, "model": model, "dataset": dataset, "seed": seed, "best_epoch": payload.get("best_epoch")}
                    if "dev" in payload:
                        flatten_metric("dev", payload["dev"], seed_row)
                    flatten_metric("test", payload["test"], seed_row)
                    seed_row.update({key: payload.get(key) for key in ("n_train", "n_dev", "n_test", "n_answer_correct_test", "n_answer_incorrect_test")})
                    per_seed_rows.append(seed_row)
    write_csv(OUT / "results/per_seed_metrics.csv", per_seed_rows)
    write_csv(OUT / "results/ensemble_metrics.csv", ensemble_rows)

    macro_rows = []
    for dataset in DATASETS:
        full = [row for row in ensemble_rows if row["variant"] == "full" and row["dataset"] == dataset]
        full_avg = {key: float(np.mean([row[f"test_{key}"] for row in full])) for key in ("auroc", "auprc", "accuracy", "f1", "brier", "ece")}
        for variant in ("full", "prefill_only", "meanpool_mlp"):
            selected = [row for row in ensemble_rows if row["variant"] == variant and row["dataset"] == dataset]
            means = {key: float(np.mean([row[f"test_{key}"] for row in selected])) for key in full_avg}
            macro_rows.append({
                "dataset": dataset, "variant": variant, **means,
                "delta_auroc": means["auroc"] - full_avg["auroc"],
                "delta_auprc": means["auprc"] - full_avg["auprc"],
                "delta_brier": means["brier"] - full_avg["brier"],
                "delta_ece": means["ece"] - full_avg["ece"],
            })
    write_csv(OUT / "results/core_ablation_macro.csv", macro_rows)

    threshold_rows = []
    for model in MODELS:
        for dataset in TARGETS:
            payload = json.loads((OUT / "results/threshold_only" / model / dataset / "threshold_only_metrics.json").read_text(encoding="utf-8"))
            for setting in ("E1_direct_transfer", "E2_threshold_only_adaptation", "E3_fixed_node_adaptation"):
                row = {"model": model, "dataset": dataset, "setting": setting}
                row.update(payload[setting])
                row["accuracy_delta_E2_minus_E1"] = payload["accuracy_delta_E2_minus_E1"]
                row["f1_delta_E2_minus_E1"] = payload["f1_delta_E2_minus_E1"]
                threshold_rows.append(row)
    write_csv(OUT / "results/threshold_only_adaptation.csv", threshold_rows)

    status_rows = []
    for model in MODELS:
        for dataset in DATASETS:
            for variant in ("full", "prefill_only", "meanpool_mlp"):
                status_rows.append({"model": model, "dataset": dataset, "variant": variant, "status": "COMPLETED"})
    write_csv(OUT / "results/run_status.csv", status_rows)
    print(json.dumps({"per_seed_rows": len(per_seed_rows), "ensemble_rows": len(ensemble_rows), "macro_rows": len(macro_rows), "threshold_rows": len(threshold_rows)}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("preflight")
    p.set_defaults(func=command_preflight)
    p = sub.add_parser("train-group")
    p.add_argument("--variant", choices=("prefill_only", "meanpool_mlp"), required=True)
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--limit", type=int)
    p.add_argument("--smoke", action="store_true")
    p.set_defaults(func=command_train_group)
    p = sub.add_parser("ensemble")
    p.add_argument("--variant", choices=("prefill_only", "meanpool_mlp"), required=True)
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.set_defaults(func=command_ensemble)
    p = sub.add_parser("threshold-only")
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--dataset", choices=TARGETS, required=True)
    p.set_defaults(func=command_threshold_only)
    p = sub.add_parser("collect-full")
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.set_defaults(func=command_collect_full)
    p = sub.add_parser("aggregate")
    p.set_defaults(func=command_aggregate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
