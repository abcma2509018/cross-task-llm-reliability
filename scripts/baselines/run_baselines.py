#!/usr/bin/env python3
"""Paper-faithful Ni-Avg-State and PRISM-SAPLMA reproduction entry point.

The script expects externally supplied generated answers, partitions, labels,
and model weights. It never trains on a target dataset. Ni regenerates answers
greedily to obtain generation-time states and verifies them against the archived
answer text. PRISM forwards the archived answer in the exact judgment query.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList


MODEL_KEYS = (
    "qwen_2.5_7b_instruct",
    "ministral_8b_instruct",
    "mistral_7b_instruct",
    "llama3.1_8b_chat",
)
DATASET_KEYS = (
    "trivia_qa_2_60k",
    "notable_people",
    "cities_10k",
    "math_operations_6k",
    "medals_9k",
    "gsm8k",
)
SOURCE_DATASET = "trivia_qa_2_60k"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def first(row: dict[str, Any], names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return default


def read_rows(
    data_root: Path, model_key: str, dataset: str, split: str,
) -> list[dict[str, Any]]:
    path = data_root / model_key / dataset / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing split: {path}")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            row = {
                "sample_id": str(first(raw, ("sample_id", "id"), "")),
                "prompt": first(raw, ("prompt", "qa_prompt", "question"), ""),
                "response": first(raw, ("generated_response", "generated_text", "answer", "response"), ""),
                "label": int(first(raw, ("correctness_label", "has_answer"), -1)),
                "split": first(raw, ("split",), split),
            }
            if (
                not row["sample_id"] or not row["prompt"] or not row["response"]
                or row["label"] not in (0, 1)
            ):
                raise ValueError(f"Invalid required fields in {path}:{line_number}")
            if row["split"] != split:
                raise ValueError(f"Split mismatch in {path}:{line_number}")
            rows.append(row)
    return rows


def load_model(
    model_root: Path, model_key: str, tokenizer_options: dict[str, Any],
):
    model_path = model_root / model_key
    tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_options.get(model_key, {}))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", low_cpu_mem_usage=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer, device


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


class StopOnSubstrings(StoppingCriteria):
    def __init__(self, tokenizer, start_length: int, stop_words: list[str]):
        self.tokenizer = tokenizer
        self.start_length = start_length
        self.stop_words = [word for word in stop_words if word]

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        text = self.tokenizer.decode(input_ids[0, self.start_length:], skip_special_tokens=False)
        return any(word in text for word in self.stop_words)


def token_cutoff(tokenizer, token_ids: torch.Tensor, stop_words: list[str]) -> int:
    ids = token_ids.tolist()
    for length in range(1, len(ids) + 1):
        text = tokenizer.decode(ids[:length], skip_special_tokens=False)
        if any(word and word in text for word in stop_words):
            return length
    return len(ids)


def ni_avg_state(
    model, tokenizer, device: torch.device, row: dict[str, Any], layer_index: int,
    max_input_tokens: int, max_new_tokens: int, stop_words: list[str],
) -> tuple[torch.Tensor, str, int]:
    encoded = tokenizer(
        row["prompt"], return_tensors="pt", truncation=True, max_length=max_input_tokens
    ).to(device)
    input_length = encoded["input_ids"].shape[1]
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_hidden_states=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([
                StopOnSubstrings(tokenizer, input_length, stop_words)
            ]),
        )
    token_ids = generated.sequences[0, input_length:]
    valid_steps = min(token_cutoff(tokenizer, token_ids, stop_words), len(generated.hidden_states))
    if valid_steps == 0:
        raise RuntimeError(f"No generation states for sample {row['sample_id']}")
    states = [generated.hidden_states[step][layer_index][0, -1, :].float().cpu() for step in range(valid_steps)]
    response = tokenizer.decode(token_ids[:valid_steps], skip_special_tokens=True).strip()
    return torch.stack(states).mean(dim=0), response, valid_steps


def prism_query(response: str) -> str:
    return f"Does the statement {response} accurately reflect the truth?"


def prism_state(
    model, tokenizer, device: torch.device, row: dict[str, Any], max_input_tokens: int,
) -> torch.Tensor:
    query = prism_query(row["response"])
    encoded = tokenizer(query, return_tensors="pt", truncation=True, max_length=max_input_tokens).to(device)
    with torch.inference_mode():
        output = model(**encoded, output_hidden_states=True, use_cache=False, return_dict=True)
    final_valid = int(encoded["attention_mask"][0].sum().item() - 1)
    return output.hidden_states[-1][0, final_valid, :].float().cpu()


class SingleHiddenMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 2)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def score(labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "auroc": float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None,
        "auprc": float(average_precision_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None,
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "threshold": threshold,
    }


def extract_features(
    method: str, rows: list[dict[str, Any]], model, tokenizer, device: torch.device,
    config: dict[str, Any], dataset: str, layer_index: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[dict[str, Any]]]:
    features, labels, ids, integrity = [], [], [], []
    for row in rows:
        if method == "ni_avg_state":
            stop_words = (
                config["gsm8k_stop_words"] if dataset == "gsm8k"
                else [config["model_eos_text"][config["active_model_key"]], "\n"]
            )
            feature, regenerated, valid_steps = ni_avg_state(
                model, tokenizer, device, row, int(layer_index),
                config["max_input_tokens"], config["max_new_tokens_by_dataset"][dataset], stop_words,
            )
            matches = normalize_text(regenerated) == normalize_text(row["response"])
            integrity.append({"sample_id": row["sample_id"], "answer_match": matches, "valid_generation_steps": valid_steps})
            if config["require_answer_match"] and not matches:
                raise RuntimeError(f"Regenerated answer mismatch for {row['sample_id']}")
        else:
            feature = prism_state(model, tokenizer, device, row, config["max_input_tokens"])
        features.append(feature)
        labels.append(row["label"])
        ids.append(row["sample_id"])
    return torch.stack(features), torch.tensor(labels, dtype=torch.long), ids, integrity


def train_source_classifier(
    train_x: torch.Tensor, train_y: torch.Tensor, dev_x: torch.Tensor, dev_y: torch.Tensor,
    method: str, seed: int, config: dict[str, Any], device: torch.device,
) -> tuple[SingleHiddenMLP, dict[str, Any]]:
    set_seed(seed)
    model = SingleHiddenMLP(train_x.shape[1], config["hidden_dim"], config["dropout"]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    class_weights = None
    if config["class_weighting"] == "complement_frequency":
        counts = torch.bincount(train_y, minlength=2).float()
        class_weights = (1.0 - counts / counts.sum()).to(device)
    elif config["class_weighting"] != "none":
        raise ValueError(f"Unsupported class weighting: {config['class_weighting']}")
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=config["batch_size"], shuffle=True)
    best_state, best_value, best_epoch = None, -float("inf"), None
    history = []
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        losses = []
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(inputs), labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.inference_mode():
            probabilities = torch.softmax(model(dev_x.to(device)), dim=1)[:, 1].cpu().numpy()
        dev_metrics = score(dev_y.numpy(), probabilities)
        selection_value = dev_metrics[config["checkpoint_metric"]]
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "dev": dev_metrics})
        if selection_value > best_value:
            best_value, best_epoch = selection_value, epoch
            best_state = deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("No source-development checkpoint was selected")
    model.load_state_dict(best_state)
    return model, {
        "method": method,
        "seed": seed,
        "best_epoch": best_epoch,
        "checkpoint_metric": config["checkpoint_metric"],
        "best_checkpoint_value": best_value,
        "class_weights": class_weights.cpu().tolist() if class_weights is not None else None,
        "history": history,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_predictions(path: Path, ids: list[str], labels: np.ndarray, probabilities: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "y_true", "p_correct"])
        writer.writerows(zip(ids, labels.tolist(), probabilities.tolist()))


def write_source_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "label"])
        writer.writerows((row["sample_id"], row["label"]) for row in rows)


def balanced_source(rows: list[dict[str, Any]], seed: int, per_class: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    positive = [row for row in rows if row["label"] == 1]
    negative = [row for row in rows if row["label"] == 0]
    if len(positive) < per_class or len(negative) < per_class:
        raise ValueError("TriviaQA source training split lacks the required class counts")
    selected = rng.sample(positive, per_class) + rng.sample(negative, per_class)
    rng.shuffle(selected)
    return selected


def source_from_manifest(
    rows: list[dict[str, Any]], manifest_path: Path, per_class: int,
) -> list[dict[str, Any]]:
    by_id = {row["sample_id"]: row for row in rows}
    selected: list[dict[str, Any]] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for item in csv.DictReader(handle):
            sample_id = item["sample_id"]
            if sample_id not in by_id:
                raise ValueError(f"Manifest sample is absent from source train split: {sample_id}")
            row = by_id[sample_id]
            if row["label"] != int(item["label"]):
                raise ValueError(f"Manifest label mismatch for {sample_id}")
            selected.append(row)
    output: list[dict[str, Any]] = []
    for label in (1, 0):
        class_rows = [row for row in selected if row["label"] == label]
        if len(class_rows) < per_class:
            raise ValueError(f"Manifest lacks {per_class} rows for label {label}")
        output.extend(class_rows[:per_class])
    return output


def source_train_dev_split(
    rows: list[dict[str, Any]], seed: int, train_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(rows))
    cut = int(train_fraction * len(indices))
    train_rows = [rows[index] for index in indices[:cut]]
    dev_rows = [rows[index] for index in indices[cut:]]
    if not train_rows or not dev_rows or any(
        len({row["label"] for row in partition}) != 2 for partition in (train_rows, dev_rows)
    ):
        raise ValueError("The PRISM 80:20 source split produced an empty partition")
    return train_rows, dev_rows


def evaluate(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, device: torch.device, threshold: float,
):
    model.eval()
    with torch.inference_mode():
        probabilities = torch.softmax(model(x.to(device)), dim=1)[:, 1].cpu().numpy()
    return probabilities, score(y.numpy(), probabilities, threshold=threshold)


def run(args: argparse.Namespace) -> None:
    release_config = json.loads(args.config.read_text(encoding="utf-8"))
    method_config = release_config[args.method]
    if (
        method_config["classifier_hidden_layers"] != 1
        or method_config["activation"] != "relu"
        or method_config["optimizer"] != "adam"
    ):
        raise ValueError("This release entry point supports the configured single-hidden-layer ReLU/Adam MLP only")
    model, tokenizer, device = load_model(
        args.model_root, args.model_key, release_config["shared"]["tokenizer_options"]
    )
    source_train = read_rows(args.data_root, args.model_key, SOURCE_DATASET, "train")
    layer_index = None
    if args.method == "ni_avg_state":
        source_dev = read_rows(args.data_root, args.model_key, SOURCE_DATASET, "dev")
        layer_index = method_config["intermediate_layer_index"][args.model_key]
        per_class = method_config["source_samples_per_class"]
        manifest = args.source_manifest_root / args.model_key / f"train_samples_seed_{args.seed}.csv"
        source_train = (
            source_from_manifest(source_train, manifest, per_class)
            if manifest.is_file()
            else balanced_source(source_train, args.seed, per_class)
        )
    else:
        source_train, source_dev = source_train_dev_split(
            source_train, args.seed, method_config["source_train_fraction"]
        )
    method_config["active_model_key"] = args.model_key

    train_x, train_y, train_ids, train_integrity = extract_features(
        args.method, source_train, model, tokenizer, device, method_config, SOURCE_DATASET, layer_index
    )
    dev_x, dev_y, _, dev_integrity = extract_features(
        args.method, source_dev, model, tokenizer, device, method_config, SOURCE_DATASET, layer_index
    )
    classifier, training = train_source_classifier(
        train_x, train_y, dev_x, dev_y, args.method, args.seed, method_config, device,
    )
    run_root = args.output_root / args.method / args.model_key / f"seed_{args.seed}"
    write_json(run_root / "training_metrics.json", training)
    torch.save(
        {
            "model_state": {name: value.detach().cpu() for name, value in classifier.state_dict().items()},
            "input_dim": int(train_x.shape[1]), "method": args.method, "seed": args.seed,
            "best_epoch": training["best_epoch"],
        },
        run_root / "best_source_classifier.pt",
    )
    write_json(run_root / "integrity.json", {"train": train_integrity, "dev": dev_integrity})
    if args.method == "ni_avg_state":
        write_source_manifest(run_root / "source_manifest.csv", source_train)

    target_datasets = args.datasets or DATASET_KEYS
    for dataset in target_datasets:
        target_rows = read_rows(args.data_root, args.model_key, dataset, "test")
        test_x, test_y, test_ids, test_integrity = extract_features(
            args.method, target_rows, model, tokenizer, device, method_config, dataset, layer_index
        )
        probabilities, metrics = evaluate(
            classifier, test_x, test_y, device, threshold=method_config["target_threshold"]
        )
        output = run_root / dataset
        write_json(output / "metrics.json", {**metrics, "model": args.model_key, "dataset": dataset, "seed": args.seed})
        write_predictions(output / "predictions.csv", test_ids, test_y.numpy(), probabilities)
        if test_integrity:
            write_json(output / "generation_integrity.json", test_integrity)


def parse_args() -> argparse.Namespace:
    release_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("ni_avg_state", "prism_saplma"), required=True)
    parser.add_argument("--model-key", choices=MODEL_KEYS, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASET_KEYS)
    parser.add_argument(
        "--source-manifest-root", type=Path,
        default=release_root / "metadata" / "ni_source_manifests",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
