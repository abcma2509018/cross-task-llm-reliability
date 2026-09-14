#!/usr/bin/env python3
"""TriviaQA MLP-neuron trajectory pipeline for Qwen2.5 correctness prediction."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from collections import defaultdict
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
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


_DATA_ROOT = os.environ.get("DATA_ROOT")
DEFAULT_LABEL_BASE = Path(_DATA_ROOT) if _DATA_ROOT else None
DEFAULT_LABEL_ROOT = DEFAULT_LABEL_BASE / "triviaqa" if DEFAULT_LABEL_BASE else None
DEFAULT_MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/Qwen2.5-7B-Instruct"))
MODEL_KEY_TO_PATH = {
    "qwen_2.5_7b_instruct": Path("models/qwen_2.5_7b_instruct"),
    "ministral_8b_instruct": Path("models/ministral_8b_instruct"),
    "llama3.1_8b_chat": Path("models/llama3.1_8b_chat"),
    "mistral_7b_instruct": Path("models/mistral_7b_instruct"),
}
SUPPORTED_DATASETS = (
    "triviaqa",
    "gsm8k",
    "trivia_qa_2_60k",
    "notable_people",
    "cities_10k",
    "math_operations_6k",
    "medals_9k",
)
LABEL_PROMPT_VERSION = {
    "trivia_qa_2_60k": "base",
    "notable_people": "base",
    "cities_10k": "base",
    "math_operations_6k": "base",
    "medals_9k": "base",
    "gsm8k": "base_3_shot",
}
DEFAULT_MULTIMODEL_LABEL_ROOT = Path(os.environ["LABEL_ROOT"]) if os.environ.get("LABEL_ROOT") else None
DEFAULT_MULTIMODEL_SPLIT_BASE = Path(_DATA_ROOT) if _DATA_ROOT else None


def dataset_label_root(dataset: str) -> Path:
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset}. Expected one of {SUPPORTED_DATASETS}")
    if DEFAULT_LABEL_BASE is None:
        raise ValueError("Set DATA_ROOT or pass explicit split paths")
    return DEFAULT_LABEL_BASE / dataset


def resolve_label_paths(args: argparse.Namespace) -> None:
    needs_default_root = any(
        hasattr(args, name) and getattr(args, name, None) is None
        for name in ("train_path", "dev_path", "test_path")
    )
    root = None
    if needs_default_root:
        model_key = getattr(args, "model_key", None)
        if model_key is not None and DEFAULT_MULTIMODEL_SPLIT_BASE is not None:
            candidate = DEFAULT_MULTIMODEL_SPLIT_BASE / model_key / args.dataset
            if candidate.exists():
                root = candidate
        if root is None and args.dataset in ("triviaqa", "gsm8k"):
            root = dataset_label_root(args.dataset)
    if hasattr(args, "train_path") and getattr(args, "train_path", None) is None:
        if root is None:
            raise ValueError("--train-path is required when no default split directory exists")
        args.train_path = root / ("train_labeled.jsonl" if (root / "train_labeled.jsonl").exists() else "train.jsonl")
    if hasattr(args, "dev_path") and getattr(args, "dev_path", None) is None:
        if root is None:
            raise ValueError("--dev-path is required when no default split directory exists")
        args.dev_path = root / ("dev_labeled.jsonl" if (root / "dev_labeled.jsonl").exists() else "dev.jsonl")
    if hasattr(args, "test_path") and getattr(args, "test_path", None) is None:
        if root is None:
            raise ValueError("--test-path is required when no default split directory exists")
        args.test_path = root / ("test_labeled.jsonl" if (root / "test_labeled.jsonl").exists() else "test.jsonl")


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def required_row_id(row: dict[str, Any], context: str) -> str:
    for key in ("id", "qid", "question_id"):
        value = row.get(key)
        if value is not None and str(value):
            return str(value)
    raise ValueError(f"{context} requires every row to have id, qid, or question_id")


def read_ids_json(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("ids")
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list of sample IDs or an object with an 'ids' list")
    ids = [str(value) for value in payload]
    if not ids:
        raise ValueError(f"{path} contains no sample IDs")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path} contains duplicate sample IDs")
    return ids


def filter_rows_by_ids(rows: list[dict[str, Any]], ids_json: Path, context: str) -> list[dict[str, Any]]:
    requested_ids = read_ids_json(ids_json)
    requested = set(requested_ids)
    row_ids = [required_row_id(row, context) for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError(f"{context} source split contains duplicate sample IDs")
    available = set(row_ids)
    missing = [sample_id for sample_id in requested_ids if sample_id not in available]
    if missing:
        raise ValueError(f"{ids_json} contains {len(missing)} IDs absent from {context}; first={missing[:10]}")
    filtered = [row for row, sample_id in zip(rows, row_ids) if sample_id in requested]
    if len(filtered) != len(requested):
        raise ValueError(f"Expected {len(requested)} filtered rows for {context}, got {len(filtered)}")
    return filtered


def split_id_filter_path(split_map_path: Path, split: str) -> Path | None:
    payload = json.loads(split_map_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{split_map_path} must contain a JSON object keyed by split name")
    raw_path = payload.get(split)
    if raw_path is None:
        return None
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    if path.exists():
        return path
    relative_to_map = split_map_path.parent / path
    if relative_to_map.exists():
        return relative_to_map
    raise FileNotFoundError(f"ID filter for split {split!r} does not exist: {raw_path}")


def stratified_sample_rows(
    rows: list[dict[str, Any]],
    max_samples: int | None,
    label_key: str,
    seed: int,
) -> list[dict[str, Any]]:
    if max_samples is None or max_samples <= 0 or len(rows) <= max_samples:
        return rows
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[int(row[label_key])].append(row)
    if len(groups) < 2:
        rng = random.Random(seed)
        sampled = list(rows)
        rng.shuffle(sampled)
        return sampled[:max_samples]

    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    allocated = 0
    labels = sorted(groups)
    for label in labels:
        group = list(groups[label])
        rng.shuffle(group)
        if label == labels[-1]:
            take = max_samples - allocated
        else:
            take = int(round(max_samples * len(group) / len(rows)))
            take = min(take, len(group), max_samples - allocated)
        sampled.extend(group[:take])
        allocated += take

    if len(sampled) < max_samples:
        selected_ids = {id(row) for row in sampled}
        leftovers = [row for row in rows if id(row) not in selected_ids]
        rng.shuffle(leftovers)
        sampled.extend(leftovers[: max_samples - len(sampled)])

    rng.shuffle(sampled)
    return sampled[:max_samples]


def read_selection_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = read_jsonl(args.train_path, args.limit)
    ids_json = getattr(args, "ids_json", None)
    if ids_json is not None:
        rows = filter_rows_by_ids(rows, ids_json, f"{args.dataset} train neuron selection")
    max_samples = getattr(args, "selection_max_samples", None)
    return stratified_sample_rows(rows, max_samples, args.label_key, args.seed)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def batched(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def apply_model_key(args: argparse.Namespace) -> None:
    model_key = getattr(args, "model_key", None)
    if not model_key:
        return
    if model_key not in MODEL_KEY_TO_PATH:
        raise ValueError(f"Unsupported --model-key {model_key!r}. Expected one of {sorted(MODEL_KEY_TO_PATH)}")
    args.model_path = MODEL_KEY_TO_PATH[model_key]


def load_qwen(model_path: Path, device: torch.device, model_key: str | None = None):
    fix_mistral_regex = model_key == "ministral_8b_instruct"
    tokenizer_kwargs = {"fix_mistral_regex": True} if fix_mistral_regex else {}
    tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
    print(
        f"model_key={model_key or 'custom_model_path'} "
        f"tokenizer_class={type(tokenizer).__name__} "
        f"fix_mistral_regex={fix_mistral_regex}"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def get_qwen_layers(model) -> list[nn.Module]:
    try:
        return list(model.model.layers)
    except AttributeError as exc:
        raise AttributeError("Expected a Qwen2-style model with model.model.layers") from exc


class MLPIntermediateRecorder:
    """Records the input to mlp.down_proj, i.e. SwiGLU MLP intermediate activations."""

    def __init__(
        self,
        model,
        selected_neurons: dict[int, torch.Tensor] | None = None,
    ) -> None:
        self.model = model
        self.selected_neurons = selected_neurons
        self.records: dict[int, list[torch.Tensor]] = defaultdict(list)
        self.handles = []

    def __enter__(self):
        layers = get_qwen_layers(self.model)
        if self.selected_neurons is None:
            layer_ids = range(len(layers))
        else:
            layer_ids = sorted(self.selected_neurons)

        for layer_id in layer_ids:
            module = layers[layer_id].mlp.down_proj
            idx = None if self.selected_neurons is None else self.selected_neurons[layer_id]

            def hook(_module, inputs, _output, layer_id=layer_id, idx=idx):
                x = inputs[0][:, -1, :]
                if idx is not None:
                    idx_device = idx.to(x.device)
                    x = x.index_select(dim=-1, index=idx_device)
                self.records[layer_id].append(x.detach().float().cpu())

            self.handles.append(module.register_forward_hook(hook))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def clear(self) -> None:
        self.records.clear()


class MLPIntermediateFullRecorder:
    """Records full down_proj inputs so prompt/prefill token activations can be pooled."""

    def __init__(
        self,
        model,
        selected_neurons: dict[int, torch.Tensor],
    ) -> None:
        self.model = model
        self.selected_neurons = selected_neurons
        self.records: dict[int, list[torch.Tensor]] = defaultdict(list)
        self.handles = []

    def __enter__(self):
        layers = get_qwen_layers(self.model)
        for layer_id in sorted(self.selected_neurons):
            module = layers[layer_id].mlp.down_proj
            idx = self.selected_neurons[layer_id]

            def hook(_module, inputs, _output, layer_id=layer_id, idx=idx):
                x = inputs[0]
                idx_device = idx.to(x.device)
                x = x.index_select(dim=-1, index=idx_device)
                self.records[layer_id].append(x.detach().float().cpu())

            self.handles.append(module.register_forward_hook(hook))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def clear(self) -> None:
        self.records.clear()


class MLPIntermediatePrefillAwareRecorder:
    """Records pooled prompt activations and generation-step activations for all MLP neurons."""

    def __init__(self, model) -> None:
        self.model = model
        self.attention_mask: torch.Tensor | None = None
        self.prompt_last: dict[int, torch.Tensor] = {}
        self.prompt_mean: dict[int, torch.Tensor] = {}
        self.prompt_max: dict[int, torch.Tensor] = {}
        self.prompt_std: dict[int, torch.Tensor] = {}
        self.last_8_mean: dict[int, torch.Tensor] = {}
        self.last_8_max: dict[int, torch.Tensor] = {}
        self.last_16_mean: dict[int, torch.Tensor] = {}
        self.last_16_max: dict[int, torch.Tensor] = {}
        self.generation: dict[int, list[torch.Tensor]] = defaultdict(list)
        self.handles = []

    def __enter__(self):
        layers = get_qwen_layers(self.model)
        for layer_id, layer in enumerate(layers):
            module = layer.mlp.down_proj

            def hook(_module, inputs, _output, layer_id=layer_id):
                x = inputs[0].detach().float().cpu()
                if layer_id not in self.prompt_last:
                    if self.attention_mask is None:
                        raise ValueError("attention_mask must be set before prefill-aware recording")
                    mask = self.attention_mask.bool()
                    last_vals = []
                    mean_vals = []
                    max_vals = []
                    std_vals = []
                    last_8_mean_vals = []
                    last_8_max_vals = []
                    last_16_mean_vals = []
                    last_16_max_vals = []
                    for sample_idx in range(x.shape[0]):
                        valid = x[sample_idx, mask[sample_idx]]
                        if valid.numel() == 0:
                            raise ValueError("Prompt tokenization produced zero valid tokens")
                        last_8 = valid[-8:]
                        last_16 = valid[-16:]
                        last_vals.append(valid[-1])
                        mean_vals.append(valid.mean(dim=0))
                        max_vals.append(valid.max(dim=0).values)
                        std_vals.append(valid.std(dim=0, unbiased=False))
                        last_8_mean_vals.append(last_8.mean(dim=0))
                        last_8_max_vals.append(last_8.max(dim=0).values)
                        last_16_mean_vals.append(last_16.mean(dim=0))
                        last_16_max_vals.append(last_16.max(dim=0).values)
                    self.prompt_last[layer_id] = torch.stack(last_vals, dim=0)
                    self.prompt_mean[layer_id] = torch.stack(mean_vals, dim=0)
                    self.prompt_max[layer_id] = torch.stack(max_vals, dim=0)
                    self.prompt_std[layer_id] = torch.stack(std_vals, dim=0)
                    self.last_8_mean[layer_id] = torch.stack(last_8_mean_vals, dim=0)
                    self.last_8_max[layer_id] = torch.stack(last_8_max_vals, dim=0)
                    self.last_16_mean[layer_id] = torch.stack(last_16_mean_vals, dim=0)
                    self.last_16_max[layer_id] = torch.stack(last_16_max_vals, dim=0)
                else:
                    self.generation[layer_id].append(x[:, -1, :])

            self.handles.append(module.register_forward_hook(hook))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def clear(self) -> None:
        self.attention_mask = None
        self.prompt_last.clear()
        self.prompt_mean.clear()
        self.prompt_max.clear()
        self.prompt_std.clear()
        self.last_8_mean.clear()
        self.last_8_max.clear()
        self.last_16_mean.clear()
        self.last_16_max.clear()
        self.generation.clear()


def generated_lengths(
    sequences: torch.Tensor,
    input_len: int,
    eos_token_id: int | None,
    min_steps: int,
) -> list[int]:
    generated = sequences[:, input_len:]
    lengths = []
    for row in generated:
        length = row.numel()
        if eos_token_id is not None:
            eos_pos = torch.where(row == eos_token_id)[0]
            if eos_pos.numel() > 0:
                length = int(eos_pos[0].item())
        lengths.append(max(min_steps, length))
    return lengths


@torch.inference_mode()
def generate_with_records(
    model,
    tokenizer,
    prompts: list[str],
    device: torch.device,
    recorder: MLPIntermediateRecorder,
    max_new_tokens: int,
    max_input_tokens: int,
    min_steps: int,
) -> tuple[list[int], list[str]]:
    batch = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
    ).to(device)
    input_len = int(batch["input_ids"].shape[1])
    recorder.clear()
    outs = model.generate(
        **batch,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
    )
    lengths = generated_lengths(outs.sequences.cpu(), input_len, tokenizer.eos_token_id, min_steps)
    generated_ids = outs.sequences[:, input_len:].cpu()
    texts = []
    for row, length in zip(generated_ids, lengths):
        texts.append(tokenizer.decode(row[:length], skip_special_tokens=True).strip())
    return lengths, texts


@torch.inference_mode()
def generate_with_full_records(
    model,
    tokenizer,
    prompts: list[str],
    device: torch.device,
    recorder: MLPIntermediateFullRecorder,
    max_new_tokens: int,
    max_input_tokens: int,
    min_steps: int,
) -> tuple[list[int], list[str], torch.Tensor]:
    batch = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
    ).to(device)
    attention_mask = batch["attention_mask"].detach().cpu().bool()
    input_len = int(batch["input_ids"].shape[1])
    recorder.clear()
    outs = model.generate(
        **batch,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
    )
    lengths = generated_lengths(outs.sequences.cpu(), input_len, tokenizer.eos_token_id, min_steps)
    generated_ids = outs.sequences[:, input_len:].cpu()
    texts = []
    for row, length in zip(generated_ids, lengths):
        texts.append(tokenizer.decode(row[:length], skip_special_tokens=True).strip())
    return lengths, texts, attention_mask


LOGIT_SCALAR_FEATURE_NAMES = [
    "mean_logprob",
    "first_token_prob",
    "first_token_logprob",
    "mean_entropy",
    "min_token_prob",
    "seq_logprob",
    "answer_length",
]


@torch.inference_mode()
def generate_with_full_records_and_logits(
    model,
    tokenizer,
    prompts: list[str],
    device: torch.device,
    recorder: MLPIntermediateFullRecorder,
    max_new_tokens: int,
    max_input_tokens: int,
    min_steps: int,
) -> tuple[list[int], list[str], torch.Tensor, torch.Tensor]:
    batch = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
    ).to(device)
    attention_mask = batch["attention_mask"].detach().cpu().bool()
    input_len = int(batch["input_ids"].shape[1])
    recorder.clear()
    outs = model.generate(
        **batch,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
    )
    lengths = generated_lengths(outs.sequences.cpu(), input_len, tokenizer.eos_token_id, min_steps)
    generated_ids = outs.sequences[:, input_len:].cpu()
    texts = []
    for row, length in zip(generated_ids, lengths):
        texts.append(tokenizer.decode(row[:length], skip_special_tokens=True).strip())

    score_steps = list(outs.scores or [])
    scalar_features = torch.zeros((len(prompts), len(LOGIT_SCALAR_FEATURE_NAMES)), dtype=torch.float32)
    for sample_idx, length in enumerate(lengths):
        steps = min(int(length), len(score_steps), int(generated_ids.shape[1]))
        if steps <= 0:
            continue
        token_probs = []
        token_logprobs = []
        entropies = []
        for step_idx in range(steps):
            logits = score_steps[step_idx][sample_idx].float()
            log_probs = torch.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            token_id = int(generated_ids[sample_idx, step_idx].item())
            token_prob = probs[token_id].clamp_min(1e-45)
            token_logprob = log_probs[token_id]
            entropy = -(probs * log_probs).sum()
            token_probs.append(token_prob)
            token_logprobs.append(token_logprob)
            entropies.append(entropy)
        prob_tensor = torch.stack(token_probs)
        logprob_tensor = torch.stack(token_logprobs)
        entropy_tensor = torch.stack(entropies)
        scalar_features[sample_idx] = torch.tensor(
            [
                float(logprob_tensor.mean()),
                float(prob_tensor[0]),
                float(logprob_tensor[0]),
                float(entropy_tensor.mean()),
                float(prob_tensor.min()),
                float(logprob_tensor.sum()),
                float(steps),
            ],
            dtype=torch.float32,
        )
    return lengths, texts, attention_mask, scalar_features


@torch.inference_mode()
def prefill_with_full_records(
    model,
    tokenizer,
    prompts: list[str],
    device: torch.device,
    recorder: MLPIntermediateFullRecorder,
    max_input_tokens: int,
) -> torch.Tensor:
    batch = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
    ).to(device)
    attention_mask = batch["attention_mask"].detach().cpu().bool()
    recorder.clear()
    _outs = model(**batch, use_cache=False)
    return attention_mask


@dataclass
class ImportanceStats:
    num_layers: int
    intermediate_size: int

    def __post_init__(self) -> None:
        shape = (self.num_layers, self.intermediate_size)
        self.count = torch.zeros(2, dtype=torch.float64)
        self.sum_activation = torch.zeros((2, *shape), dtype=torch.float64)
        self.sum_delta = torch.zeros((2, *shape), dtype=torch.float64)
        self.sum_var = torch.zeros((2, *shape), dtype=torch.float64)

    def update(self, label: int, layer: int, traj: torch.Tensor) -> None:
        t = int(traj.shape[0])
        mean_activation = traj.mean(dim=0).double()
        if t > 1:
            mean_delta = traj.diff(dim=0).abs().mean(dim=0).double()
            var = traj.var(dim=0, unbiased=False).double()
        else:
            mean_delta = torch.zeros_like(mean_activation)
            var = torch.zeros_like(mean_activation)
        self.sum_activation[label, layer].add_(mean_activation)
        self.sum_delta[label, layer].add_(mean_delta)
        self.sum_var[label, layer].add_(var)

    def add_sample_count(self, label: int) -> None:
        self.count[label] += 1

    def scores(self, alpha: float, beta: float, gamma: float) -> torch.Tensor:
        if torch.any(self.count == 0):
            raise ValueError(f"Both classes must be present in train labels, got counts={self.count.tolist()}")
        denom = self.count.view(2, 1, 1)
        mean_activation = self.sum_activation / denom
        mean_delta = self.sum_delta / denom
        mean_var = self.sum_var / denom
        return (
            alpha * (mean_activation[1] - mean_activation[0]).abs()
            + beta * (mean_delta[1] - mean_delta[0]).abs()
            + gamma * (mean_var[1] - mean_var[0]).abs()
        )


@dataclass
class PrefillAwareImportanceStats:
    num_layers: int
    intermediate_size: int

    def __post_init__(self) -> None:
        shape = (self.num_layers, self.intermediate_size)
        self.count = torch.zeros(2, dtype=torch.float64)
        self.sum_prompt_last = torch.zeros((2, *shape), dtype=torch.float64)
        self.sum_prompt_mean = torch.zeros((2, *shape), dtype=torch.float64)
        self.sum_prompt_max = torch.zeros((2, *shape), dtype=torch.float64)
        self.sum_generation_mean = torch.zeros((2, *shape), dtype=torch.float64)
        self.sum_generation_delta = torch.zeros((2, *shape), dtype=torch.float64)
        self.sum_generation_variance = torch.zeros((2, *shape), dtype=torch.float64)

    def add_sample_count(self, label: int) -> None:
        self.count[label] += 1

    def update(
        self,
        label: int,
        layer: int,
        prompt_last: torch.Tensor,
        prompt_mean: torch.Tensor,
        prompt_max: torch.Tensor,
        generation: torch.Tensor,
    ) -> None:
        self.sum_prompt_last[label, layer].add_(prompt_last.double())
        self.sum_prompt_mean[label, layer].add_(prompt_mean.double())
        self.sum_prompt_max[label, layer].add_(prompt_max.double())
        if generation.numel() == 0:
            gen_mean = torch.zeros_like(prompt_last)
            gen_delta = torch.zeros_like(prompt_last)
            gen_var = torch.zeros_like(prompt_last)
        else:
            gen_mean = generation.mean(dim=0)
            if generation.shape[0] > 1:
                gen_delta = generation.diff(dim=0).abs().mean(dim=0)
                gen_var = generation.var(dim=0, unbiased=False)
            else:
                gen_delta = torch.zeros_like(gen_mean)
                gen_var = torch.zeros_like(gen_mean)
        self.sum_generation_mean[label, layer].add_(gen_mean.double())
        self.sum_generation_delta[label, layer].add_(gen_delta.double())
        self.sum_generation_variance[label, layer].add_(gen_var.double())

    def _mean(self, value: torch.Tensor) -> torch.Tensor:
        if torch.any(self.count == 0):
            raise ValueError(f"Both classes must be present in train labels, got counts={self.count.tolist()}")
        return value / self.count.view(2, 1, 1)

    def scores(
        self,
        w_prompt_last: float,
        w_prompt_mean: float,
        w_prompt_max: float,
        w_generation_mean: float,
        w_generation_delta: float,
        w_generation_variance: float,
    ) -> torch.Tensor:
        prompt_last = self._mean(self.sum_prompt_last)
        prompt_mean = self._mean(self.sum_prompt_mean)
        prompt_max = self._mean(self.sum_prompt_max)
        generation_mean = self._mean(self.sum_generation_mean)
        generation_delta = self._mean(self.sum_generation_delta)
        generation_variance = self._mean(self.sum_generation_variance)
        return (
            w_prompt_last * (prompt_last[1] - prompt_last[0]).abs()
            + w_prompt_mean * (prompt_mean[1] - prompt_mean[0]).abs()
            + w_prompt_max * (prompt_max[1] - prompt_max[0]).abs()
            + w_generation_mean * (generation_mean[1] - generation_mean[0]).abs()
            + w_generation_delta * (generation_delta[1] - generation_delta[0]).abs()
            + w_generation_variance * (generation_variance[1] - generation_variance[0]).abs()
        )


@dataclass
class PrefillOnlyImportanceStats:
    num_layers: int
    feature_size: int

    def __post_init__(self) -> None:
        shape = (self.num_layers, self.feature_size)
        self.count = torch.zeros(2, dtype=torch.float64)
        self.sum_prompt_last = torch.zeros((2, *shape), dtype=torch.float64)
        self.sum_prompt_mean = torch.zeros((2, *shape), dtype=torch.float64)
        self.sum_prompt_max = torch.zeros((2, *shape), dtype=torch.float64)

    def add_sample_count(self, label: int) -> None:
        self.count[label] += 1

    def update(
        self,
        label: int,
        layer: int,
        prompt_last: torch.Tensor,
        prompt_mean: torch.Tensor,
        prompt_max: torch.Tensor,
    ) -> None:
        self.sum_prompt_last[label, layer].add_(prompt_last.double())
        self.sum_prompt_mean[label, layer].add_(prompt_mean.double())
        self.sum_prompt_max[label, layer].add_(prompt_max.double())

    def _mean(self, value: torch.Tensor) -> torch.Tensor:
        if torch.any(self.count == 0):
            raise ValueError(f"Both classes must be present in train labels, got counts={self.count.tolist()}")
        return value / self.count.view(2, 1, 1)

    def scores(self, w_prompt_last: float, w_prompt_mean: float, w_prompt_max: float) -> torch.Tensor:
        prompt_last = self._mean(self.sum_prompt_last)
        prompt_mean = self._mean(self.sum_prompt_mean)
        prompt_max = self._mean(self.sum_prompt_max)
        return (
            w_prompt_last * (prompt_last[1] - prompt_last[0]).abs()
            + w_prompt_mean * (prompt_mean[1] - prompt_mean[0]).abs()
            + w_prompt_max * (prompt_max[1] - prompt_max[0]).abs()
        )


@dataclass
class PrefillRichImportanceStats:
    num_layers: int
    feature_size: int

    def __post_init__(self) -> None:
        shape = (self.num_layers, self.feature_size)
        self.count = torch.zeros(2, dtype=torch.float64)
        self.sums = {
            name: torch.zeros((2, *shape), dtype=torch.float64)
            for name in RICH_PREFILL_FEATURE_NAMES
        }

    def add_sample_count(self, label: int) -> None:
        self.count[label] += 1

    def update(self, label: int, layer: int, values: dict[str, torch.Tensor]) -> None:
        for name in RICH_PREFILL_FEATURE_NAMES:
            self.sums[name][label, layer].add_(values[name].double())

    def _mean(self, value: torch.Tensor) -> torch.Tensor:
        if torch.any(self.count == 0):
            raise ValueError(f"Both classes must be present in train labels, got counts={self.count.tolist()}")
        return value / self.count.view(2, 1, 1)

    def scores(self, weights: dict[str, float]) -> torch.Tensor:
        score = torch.zeros((self.num_layers, self.feature_size), dtype=torch.float64)
        for name in RICH_PREFILL_FEATURE_NAMES:
            mean_value = self._mean(self.sums[name])
            score += float(weights[name]) * (mean_value[1] - mean_value[0]).abs()
        return score


RICH_PREFILL_FEATURE_NAMES = [
    "prompt_last",
    "prompt_mean",
    "prompt_max",
    "prompt_std",
    "last_8_mean",
    "last_8_max",
    "last_16_mean",
    "last_16_max",
]


def command_select_neurons(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    resolve_label_paths(args)
    device = resolve_device(args.device)
    rows = read_selection_rows(args)
    model, tokenizer = load_qwen(args.model_path, device, getattr(args, "model_key", None))
    layers = get_qwen_layers(model)
    num_layers = len(layers)
    intermediate_size = int(model.config.intermediate_size)
    stats = ImportanceStats(num_layers=num_layers, intermediate_size=intermediate_size)

    with MLPIntermediateRecorder(model) as recorder:
        for batch_rows in tqdm(list(batched(rows, args.batch_size)), desc="select-neurons"):
            prompts = [row[args.prompt_key] for row in batch_rows]
            labels = [int(row[args.label_key]) for row in batch_rows]
            lengths, _texts = generate_with_records(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                device=device,
                recorder=recorder,
                max_new_tokens=args.max_new_tokens,
                max_input_tokens=args.max_input_tokens,
                min_steps=args.min_steps,
            )
            for sample_idx, label in enumerate(labels):
                stats.add_sample_count(label)
                for layer_id in range(num_layers):
                    layer_records = recorder.records[layer_id]
                    steps = min(lengths[sample_idx], len(layer_records))
                    if steps <= 0:
                        continue
                    traj = torch.stack(
                        [layer_records[t][sample_idx] for t in range(steps)],
                        dim=0,
                    )
                    stats.update(label=label, layer=layer_id, traj=traj)

    scores = stats.scores(alpha=args.alpha, beta=args.beta, gamma=args.gamma)
    flat = scores.reshape(-1)
    top_scores, top_indices = torch.topk(flat, k=args.top_k)
    selected = []
    for score, flat_idx in zip(top_scores.tolist(), top_indices.tolist()):
        selected.append(
            {
                "layer": int(flat_idx // intermediate_size),
                "neuron": int(flat_idx % intermediate_size),
                "score": float(score),
            }
        )

    write_json(args.out_json, selected)
    summary = {
        "dataset": args.dataset,
        "model_key": getattr(args, "model_key", None),
        "train_path": str(args.train_path),
        "model_path": str(args.model_path),
        "num_train_samples": len(rows),
        "selection_ids_json": str(args.ids_json) if args.ids_json is not None else None,
        "selection_size": len(rows),
        "selection_positive_count": int(stats.count[1].item()),
        "selection_negative_count": int(stats.count[0].item()),
        "class_counts": {"wrong_0": int(stats.count[0].item()), "correct_1": int(stats.count[1].item())},
        "num_layers": num_layers,
        "intermediate_size": intermediate_size,
        "candidate_nodes": num_layers * intermediate_size,
        "top_k": args.top_k,
        "score_formula": {
            "alpha": args.alpha,
            "beta": args.beta,
            "gamma": args.gamma,
            "mean_delta": "mean absolute adjacent-step activation difference",
        },
        "max_new_tokens": args.max_new_tokens,
        "max_input_tokens": args.max_input_tokens,
        "output": str(args.out_json),
    }
    write_json(args.out_json.with_name(args.out_json.stem + "_summary.json"), summary)


def command_select_neurons_prefill_aware(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    resolve_label_paths(args)
    device = resolve_device(args.device)
    rows = read_selection_rows(args)
    model, tokenizer = load_qwen(args.model_path, device, getattr(args, "model_key", None))
    layers = get_qwen_layers(model)
    num_layers = len(layers)
    intermediate_size = int(model.config.intermediate_size)
    stats = PrefillAwareImportanceStats(num_layers=num_layers, intermediate_size=intermediate_size)

    with MLPIntermediatePrefillAwareRecorder(model) as recorder:
        for batch_rows in tqdm(list(batched(rows, args.batch_size)), desc="select-prefill-aware"):
            prompts = [row[args.prompt_key] for row in batch_rows]
            labels = [int(row[args.label_key]) for row in batch_rows]
            batch = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_input_tokens,
            ).to(device)
            input_len = int(batch["input_ids"].shape[1])
            recorder.clear()
            recorder.attention_mask = batch["attention_mask"].detach().cpu()
            outs = model.generate(
                **batch,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
            )
            lengths = generated_lengths(outs.sequences.cpu(), input_len, tokenizer.eos_token_id, args.min_steps)
            for sample_idx, label in enumerate(labels):
                stats.add_sample_count(label)
                for layer_id in range(num_layers):
                    gen_records = recorder.generation[layer_id]
                    steps = min(lengths[sample_idx], len(gen_records))
                    if steps > 0:
                        generation = torch.stack(
                            [gen_records[t][sample_idx] for t in range(steps)],
                            dim=0,
                        )
                    else:
                        generation = torch.empty((0, intermediate_size), dtype=torch.float32)
                    stats.update(
                        label=label,
                        layer=layer_id,
                        prompt_last=recorder.prompt_last[layer_id][sample_idx],
                        prompt_mean=recorder.prompt_mean[layer_id][sample_idx],
                        prompt_max=recorder.prompt_max[layer_id][sample_idx],
                        generation=generation,
                    )

    weights = {
        "prompt_last": args.w_prompt_last,
        "prompt_mean": args.w_prompt_mean,
        "prompt_max": args.w_prompt_max,
        "generation_mean": args.w_generation_mean,
        "generation_delta": args.w_generation_delta,
        "generation_variance": args.w_generation_variance,
    }
    scores = stats.scores(
        w_prompt_last=args.w_prompt_last,
        w_prompt_mean=args.w_prompt_mean,
        w_prompt_max=args.w_prompt_max,
        w_generation_mean=args.w_generation_mean,
        w_generation_delta=args.w_generation_delta,
        w_generation_variance=args.w_generation_variance,
    )
    flat = scores.reshape(-1)
    top_scores, top_indices = torch.topk(flat, k=args.top_k)
    selected = []
    for score, flat_idx in zip(top_scores.tolist(), top_indices.tolist()):
        selected.append(
            {
                "layer": int(flat_idx // intermediate_size),
                "neuron": int(flat_idx % intermediate_size),
                "score": float(score),
            }
        )

    write_json(args.out_json, selected)
    summary = {
        "dataset": args.dataset,
        "train_path": str(args.train_path),
        "model_path": str(args.model_path),
        "num_train_samples": len(rows),
        "class_counts": {"wrong_0": int(stats.count[0].item()), "correct_1": int(stats.count[1].item())},
        "num_layers": num_layers,
        "intermediate_size": intermediate_size,
        "candidate_nodes": num_layers * intermediate_size,
        "top_k": args.top_k,
        "score_formula": weights,
        "max_new_tokens": args.max_new_tokens,
        "max_input_tokens": args.max_input_tokens,
        "output": str(args.out_json),
        "selection_data": "train split only",
    }
    write_json(args.out_json.with_name(args.out_json.stem + "_summary.json"), summary)


def command_select_neurons_prefill_only(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    resolve_label_paths(args)
    device = resolve_device(args.device)
    rows = read_selection_rows(args)
    model, tokenizer = load_qwen(args.model_path, device, getattr(args, "model_key", None))
    layers = get_qwen_layers(model)
    num_layers = len(layers)
    intermediate_size = int(model.config.intermediate_size)
    stats = PrefillOnlyImportanceStats(num_layers=num_layers, feature_size=intermediate_size)

    with MLPIntermediatePrefillAwareRecorder(model) as recorder:
        for batch_rows in tqdm(list(batched(rows, args.batch_size)), desc="select-prefill-only"):
            prompts = [row[args.prompt_key] for row in batch_rows]
            labels = [int(row[args.label_key]) for row in batch_rows]
            batch = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_input_tokens,
            ).to(device)
            recorder.clear()
            recorder.attention_mask = batch["attention_mask"].detach().cpu()
            _outs = model(**batch, use_cache=False)
            for sample_idx, label in enumerate(labels):
                stats.add_sample_count(label)
                for layer_id in range(num_layers):
                    stats.update(
                        label=label,
                        layer=layer_id,
                        prompt_last=recorder.prompt_last[layer_id][sample_idx],
                        prompt_mean=recorder.prompt_mean[layer_id][sample_idx],
                        prompt_max=recorder.prompt_max[layer_id][sample_idx],
                    )

    weights = {
        "prompt_last": args.w_prompt_last,
        "prompt_mean": args.w_prompt_mean,
        "prompt_max": args.w_prompt_max,
    }
    scores = stats.scores(
        w_prompt_last=args.w_prompt_last,
        w_prompt_mean=args.w_prompt_mean,
        w_prompt_max=args.w_prompt_max,
    )
    flat = scores.reshape(-1)
    top_scores, top_indices = torch.topk(flat, k=args.top_k)
    selected = []
    for score, flat_idx in zip(top_scores.tolist(), top_indices.tolist()):
        selected.append(
            {
                "layer": int(flat_idx // intermediate_size),
                "neuron": int(flat_idx % intermediate_size),
                "score": float(score),
            }
        )

    write_json(args.out_json, selected)
    summary = {
        "dataset": args.dataset,
        "train_path": str(args.train_path),
        "model_path": str(args.model_path),
        "num_train_samples": len(rows),
        "class_counts": {"wrong_0": int(stats.count[0].item()), "correct_1": int(stats.count[1].item())},
        "num_layers": num_layers,
        "intermediate_size": intermediate_size,
        "candidate_nodes": num_layers * intermediate_size,
        "top_k": args.top_k,
        "score_formula": weights,
        "max_input_tokens": args.max_input_tokens,
        "output": str(args.out_json),
        "selection_data": "train split only",
        "pre_generation_only": True,
        "uses_generate": False,
        "uses_output_probability_features": False,
    }
    write_json(args.out_json.with_name(args.out_json.stem + "_summary.json"), summary)


def rich_values_from_recorder(
    recorder: MLPIntermediatePrefillAwareRecorder,
    layer_id: int,
    sample_idx: int,
) -> dict[str, torch.Tensor]:
    return {
        "prompt_last": recorder.prompt_last[layer_id][sample_idx],
        "prompt_mean": recorder.prompt_mean[layer_id][sample_idx],
        "prompt_max": recorder.prompt_max[layer_id][sample_idx],
        "prompt_std": recorder.prompt_std[layer_id][sample_idx],
        "last_8_mean": recorder.last_8_mean[layer_id][sample_idx],
        "last_8_max": recorder.last_8_max[layer_id][sample_idx],
        "last_16_mean": recorder.last_16_mean[layer_id][sample_idx],
        "last_16_max": recorder.last_16_max[layer_id][sample_idx],
    }


def rich_weight_dict(args: argparse.Namespace) -> dict[str, float]:
    return {
        "prompt_last": args.w_prompt_last,
        "prompt_mean": args.w_prompt_mean,
        "prompt_max": args.w_prompt_max,
        "prompt_std": args.w_prompt_std,
        "last_8_mean": args.w_last_8_mean,
        "last_8_max": args.w_last_8_max,
        "last_16_mean": args.w_last_16_mean,
        "last_16_max": args.w_last_16_max,
    }


def command_select_neurons_prefill_rich(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    resolve_label_paths(args)
    device = resolve_device(args.device)
    rows = read_selection_rows(args)
    model, tokenizer = load_qwen(args.model_path, device, getattr(args, "model_key", None))
    layers = get_qwen_layers(model)
    num_layers = len(layers)
    intermediate_size = int(model.config.intermediate_size)
    stats = PrefillRichImportanceStats(num_layers=num_layers, feature_size=intermediate_size)

    with MLPIntermediatePrefillAwareRecorder(model) as recorder:
        for batch_rows in tqdm(list(batched(rows, args.batch_size)), desc="select-prefill-rich"):
            prompts = [row[args.prompt_key] for row in batch_rows]
            labels = [int(row[args.label_key]) for row in batch_rows]
            batch = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_input_tokens,
            ).to(device)
            recorder.clear()
            recorder.attention_mask = batch["attention_mask"].detach().cpu()
            _outs = model(**batch, use_cache=False)
            for sample_idx, label in enumerate(labels):
                stats.add_sample_count(label)
                for layer_id in range(num_layers):
                    stats.update(
                        label=label,
                        layer=layer_id,
                        values=rich_values_from_recorder(recorder, layer_id, sample_idx),
                    )

    weights = rich_weight_dict(args)
    scores = stats.scores(weights)
    flat = scores.reshape(-1)
    top_scores, top_indices = torch.topk(flat, k=args.top_k)
    selected = []
    for score, flat_idx in zip(top_scores.tolist(), top_indices.tolist()):
        selected.append(
            {
                "layer": int(flat_idx // intermediate_size),
                "neuron": int(flat_idx % intermediate_size),
                "score": float(score),
            }
        )

    write_json(args.out_json, selected)
    summary = {
        "dataset": args.dataset,
        "train_path": str(args.train_path),
        "model_path": str(args.model_path),
        "num_train_samples": len(rows),
        "class_counts": {"wrong_0": int(stats.count[0].item()), "correct_1": int(stats.count[1].item())},
        "num_layers": num_layers,
        "intermediate_size": intermediate_size,
        "candidate_nodes": num_layers * intermediate_size,
        "top_k": args.top_k,
        "score_formula": weights,
        "rich_prefill_features": RICH_PREFILL_FEATURE_NAMES,
        "max_input_tokens": args.max_input_tokens,
        "output": str(args.out_json),
        "selection_data": "train split only",
        "pre_generation_only": True,
        "uses_generate": False,
        "uses_output_probability_features": False,
    }
    write_json(args.out_json.with_name(args.out_json.stem + "_summary.json"), summary)


def command_select_neurons_prefill_lastseq(args: argparse.Namespace) -> None:
    command_select_neurons_prefill_rich(args)


def load_top_neurons(path: Path) -> list[dict[str, Any]]:
    neurons = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(neurons, list) or not neurons:
        raise ValueError(f"No neurons found in {path}")
    return neurons


def selected_neuron_tensors(
    neurons: list[dict[str, Any]],
) -> tuple[dict[int, torch.Tensor], dict[int, list[int]]]:
    indices_by_layer: dict[int, list[int]] = defaultdict(list)
    positions_by_layer: dict[int, list[int]] = defaultdict(list)
    for pos, item in enumerate(neurons):
        layer = int(item["layer"])
        neuron = int(item["neuron"])
        indices_by_layer[layer].append(neuron)
        positions_by_layer[layer].append(pos)
    tensor_by_layer = {
        layer: torch.tensor(indices, dtype=torch.long)
        for layer, indices in indices_by_layer.items()
    }
    return tensor_by_layer, positions_by_layer


def extract_split_trajectories(
    rows: list[dict[str, Any]],
    split: str,
    model,
    tokenizer,
    device: torch.device,
    neurons: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    selected_by_layer, positions_by_layer = selected_neuron_tensors(neurons)
    k = len(neurons)
    xs: list[torch.Tensor] = []
    ys: list[int] = []
    masks: list[torch.Tensor] = []
    generated_texts: list[str] = []
    questions: list[str] = []

    with MLPIntermediateRecorder(model, selected_by_layer) as recorder:
        for batch_rows in tqdm(list(batched(rows, args.batch_size)), desc=f"extract-{split}"):
            prompts = [row[args.prompt_key] for row in batch_rows]
            lengths, texts = generate_with_records(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                device=device,
                recorder=recorder,
                max_new_tokens=args.max_new_tokens,
                max_input_tokens=args.max_input_tokens,
                min_steps=args.min_steps,
            )
            for sample_idx, row in enumerate(batch_rows):
                steps = max(1, min(lengths[sample_idx], min(len(v) for v in recorder.records.values())))
                x = torch.zeros((steps, k), dtype=torch.float32)
                for layer_id, positions in positions_by_layer.items():
                    layer_records = recorder.records[layer_id]
                    layer_steps = min(steps, len(layer_records))
                    vals = torch.stack(
                        [layer_records[t][sample_idx] for t in range(layer_steps)],
                        dim=0,
                    )
                    x[:layer_steps, positions] = vals
                xs.append(x)
                masks.append(torch.ones(steps, dtype=torch.bool))
                ys.append(int(row[args.label_key]))
                generated_texts.append(texts[sample_idx])
                questions.append(row.get("question", ""))

    max_t = max(int(x.shape[0]) for x in xs)
    x_padded = torch.zeros((len(xs), max_t, k), dtype=torch.float32)
    mask_padded = torch.zeros((len(xs), max_t), dtype=torch.bool)
    for idx, x in enumerate(xs):
        t = int(x.shape[0])
        x_padded[idx, :t] = x
        mask_padded[idx, :t] = True

    return {
        "x": x_padded,
        "mask": mask_padded,
        "y": torch.tensor(ys, dtype=torch.float32),
        "lengths": mask_padded.sum(dim=1).long(),
        "questions": questions,
        "generated_texts": generated_texts,
        "split": split,
        "neurons": neurons,
        "metadata": {
            "x_shape": list(x_padded.shape),
            "mask_shape": list(mask_padded.shape),
            "num_samples": len(xs),
            "top_k": k,
            "max_new_tokens": args.max_new_tokens,
            "trajectory": "real greedy generation, down_proj input MLP intermediate activations",
        },
    }


def command_extract_trajectories(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    resolve_label_paths(args)
    device = resolve_device(args.device)
    neurons = load_top_neurons(args.neurons_json)
    model, tokenizer = load_qwen(args.model_path, device, getattr(args, "model_key", None))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    split_paths = {
        "train": args.train_path,
        "dev": args.dev_path,
        "test": args.test_path,
    }
    for split in args.splits:
        path = split_paths[split]
        rows = read_jsonl(path, args.limit)
        data = extract_split_trajectories(
            rows=rows,
            split=split,
            model=model,
            tokenizer=tokenizer,
            device=device,
            neurons=neurons,
            args=args,
        )
        out_path = args.out_dir / f"{split}.pt"
        torch.save(data, out_path)
        print(f"wrote {out_path} shape={tuple(data['x'].shape)}")


def extract_prefill_generation_split(
    rows: list[dict[str, Any]],
    split: str,
    model,
    tokenizer,
    device: torch.device,
    neurons: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    selected_by_layer, positions_by_layer = selected_neuron_tensors(neurons)
    k = len(neurons)
    xs: list[torch.Tensor] = []
    ys: list[int] = []
    ids: list[str] = []
    generated_texts: list[str] = []
    questions: list[str] = []

    with MLPIntermediateFullRecorder(model, selected_by_layer) as recorder:
        offset = 0
        for batch_rows in tqdm(list(batched(rows, args.batch_size)), desc=f"prefill-gen-{split}"):
            prompts = [row[args.prompt_key] for row in batch_rows]
            lengths, texts, attention_mask = generate_with_full_records(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                device=device,
                recorder=recorder,
                max_new_tokens=args.max_new_tokens,
                max_input_tokens=args.max_input_tokens,
                min_steps=args.min_steps,
            )
            prompt_record_counts = [
                len(recorder.records[layer_id])
                for layer_id in positions_by_layer
            ]
            if not prompt_record_counts:
                raise ValueError("No selected layer records captured")
            total_records = min(prompt_record_counts)

            for sample_idx, row in enumerate(batch_rows):
                prompt_len = int(attention_mask[sample_idx].sum().item())
                if prompt_len <= 0:
                    raise ValueError("Prompt tokenization produced zero valid tokens")

                gen_steps = max(1, min(lengths[sample_idx], max(0, total_records - 1)))
                x = torch.zeros((3 + gen_steps, k), dtype=torch.float32)

                for layer_id, positions in positions_by_layer.items():
                    layer_records = recorder.records[layer_id]
                    prefill = layer_records[0][sample_idx]
                    prompt_valid = prefill[attention_mask[sample_idx]]
                    x[0, positions] = prompt_valid[-1]
                    x[1, positions] = prompt_valid.mean(dim=0)
                    x[2, positions] = prompt_valid.max(dim=0).values

                    layer_gen_steps = min(gen_steps, max(0, len(layer_records) - 1))
                    if layer_gen_steps > 0:
                        gen_vals = torch.stack(
                            [layer_records[t + 1][sample_idx, -1] for t in range(layer_gen_steps)],
                            dim=0,
                        )
                        x[3 : 3 + layer_gen_steps, positions] = gen_vals

                xs.append(x)
                ys.append(int(row[args.label_key]))
                row_id = row.get("id") or row.get("qid") or row.get("question_id") or f"{split}-{offset + sample_idx}"
                ids.append(str(row_id))
                generated_texts.append(texts[sample_idx])
                questions.append(row.get("question", ""))
            offset += len(batch_rows)

    max_t = max(int(x.shape[0]) for x in xs)
    x_padded = torch.zeros((len(xs), max_t, k), dtype=torch.float32)
    mask_padded = torch.zeros((len(xs), max_t), dtype=torch.bool)
    for idx, x in enumerate(xs):
        t = int(x.shape[0])
        x_padded[idx, :t] = x
        mask_padded[idx, :t] = True

    return {
        "x": x_padded,
        "mask": mask_padded,
        "y": torch.tensor(ys, dtype=torch.float32),
        "ids": ids,
        "lengths": mask_padded.sum(dim=1).long(),
        "questions": questions,
        "generated_texts": generated_texts,
        "split": split,
        "neurons": neurons,
        "metadata": {
            "x_shape": list(x_padded.shape),
            "mask_shape": list(mask_padded.shape),
            "num_samples": len(xs),
            "top_k": k,
            "max_new_tokens": args.max_new_tokens,
            "max_input_tokens": args.max_input_tokens,
            "prompt_steps": ["prompt_last", "prompt_mean", "prompt_max"],
            "trajectory": "prompt pooled prefill activations plus real greedy generation down_proj input activations",
        },
    }


def command_extract_prefill_generation_trajectories(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    resolve_label_paths(args)
    device = resolve_device(args.device)
    neurons = load_top_neurons(args.neurons_json)
    model, tokenizer = load_qwen(args.model_path, device, getattr(args, "model_key", None))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    split_paths = {
        "train": args.train_path,
        "dev": args.dev_path,
        "test": args.test_path,
    }
    for split in args.splits:
        path = split_paths[split]
        rows = read_jsonl(path, args.limit)
        split_ids_json = None
        if args.split_id_filter_json is not None:
            split_ids_json = split_id_filter_path(args.split_id_filter_json, split)
            if split_ids_json is not None:
                rows = filter_rows_by_ids(rows, split_ids_json, f"{args.dataset} {split} trajectory extraction")
        if args.max_samples_per_split is not None:
            rows = rows[: args.max_samples_per_split]
        data = extract_prefill_generation_split_checkpointed(
            rows=rows,
            split=split,
            model=model,
            tokenizer=tokenizer,
            device=device,
            neurons=neurons,
            args=args,
        )
        metadata = dict(data.get("metadata", {}))
        metadata.update(
            {
                "split_id_filter_json": str(args.split_id_filter_json) if args.split_id_filter_json else None,
                "split_ids_json": str(split_ids_json) if split_ids_json else None,
                "filtered_num_samples": len(rows),
            }
        )
        data["metadata"] = metadata
        out_path = args.out_dir / f"{split}.pt"
        atomic_torch_save(data, out_path)
        print(f"wrote {out_path} shape={tuple(data['x'].shape)}")


def extract_prefill_generation_logits_split(
    rows: list[dict[str, Any]],
    split: str,
    model,
    tokenizer,
    device: torch.device,
    neurons: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    selected_by_layer, positions_by_layer = selected_neuron_tensors(neurons)
    k = len(neurons)
    xs: list[torch.Tensor] = []
    ys: list[int] = []
    scalar_features: list[torch.Tensor] = []
    ids: list[str] = []
    generated_texts: list[str] = []
    questions: list[str] = []

    with MLPIntermediateFullRecorder(model, selected_by_layer) as recorder:
        offset = 0
        for batch_rows in tqdm(list(batched(rows, args.batch_size)), desc=f"prefill-gen-logits-{split}"):
            prompts = [row[args.prompt_key] for row in batch_rows]
            lengths, texts, attention_mask, batch_scalar = generate_with_full_records_and_logits(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                device=device,
                recorder=recorder,
                max_new_tokens=args.max_new_tokens,
                max_input_tokens=args.max_input_tokens,
                min_steps=args.min_steps,
            )
            prompt_record_counts = [
                len(recorder.records[layer_id])
                for layer_id in positions_by_layer
            ]
            if not prompt_record_counts:
                raise ValueError("No selected layer records captured")
            total_records = min(prompt_record_counts)

            for sample_idx, row in enumerate(batch_rows):
                prompt_len = int(attention_mask[sample_idx].sum().item())
                if prompt_len <= 0:
                    raise ValueError("Prompt tokenization produced zero valid tokens")

                gen_steps = max(1, min(lengths[sample_idx], max(0, total_records - 1)))
                x = torch.zeros((3 + gen_steps, k), dtype=torch.float32)

                for layer_id, positions in positions_by_layer.items():
                    layer_records = recorder.records[layer_id]
                    prefill = layer_records[0][sample_idx]
                    prompt_valid = prefill[attention_mask[sample_idx]]
                    x[0, positions] = prompt_valid[-1]
                    x[1, positions] = prompt_valid.mean(dim=0)
                    x[2, positions] = prompt_valid.max(dim=0).values

                    layer_gen_steps = min(gen_steps, max(0, len(layer_records) - 1))
                    if layer_gen_steps > 0:
                        gen_vals = torch.stack(
                            [layer_records[t + 1][sample_idx, -1] for t in range(layer_gen_steps)],
                            dim=0,
                        )
                        x[3 : 3 + layer_gen_steps, positions] = gen_vals

                row_id = row.get("id") or row.get("qid") or row.get("question_id") or f"{split}-{offset + sample_idx}"
                xs.append(x)
                ys.append(int(row[args.label_key]))
                scalar_features.append(batch_scalar[sample_idx])
                ids.append(str(row_id))
                generated_texts.append(texts[sample_idx])
                questions.append(row.get("question", ""))
            offset += len(batch_rows)

    max_t = max(int(x.shape[0]) for x in xs)
    x_padded = torch.zeros((len(xs), max_t, k), dtype=torch.float32)
    mask_padded = torch.zeros((len(xs), max_t), dtype=torch.bool)
    for idx, x in enumerate(xs):
        t = int(x.shape[0])
        x_padded[idx, :t] = x
        mask_padded[idx, :t] = True

    scalar_tensor = torch.stack(scalar_features, dim=0)
    return {
        "x": x_padded,
        "mask": mask_padded,
        "scalar_features": scalar_tensor,
        "y": torch.tensor(ys, dtype=torch.float32),
        "ids": ids,
        "lengths": mask_padded.sum(dim=1).long(),
        "questions": questions,
        "generated_texts": generated_texts,
        "split": split,
        "neurons": neurons,
        "metadata": {
            "x_shape": list(x_padded.shape),
            "mask_shape": list(mask_padded.shape),
            "scalar_features_shape": list(scalar_tensor.shape),
            "scalar_feature_names": LOGIT_SCALAR_FEATURE_NAMES,
            "num_samples": len(xs),
            "top_k": k,
            "max_new_tokens": args.max_new_tokens,
            "prompt_steps": ["prompt_last", "prompt_mean", "prompt_max"],
            "trajectory": "prompt pooled prefill activations plus real greedy generation down_proj input activations",
            "scalar_features": "low-cost greedy generation output probability statistics",
        },
    }


def build_prefill_generation_logits_sample(
    row: dict[str, Any],
    split: str,
    sample_idx: int,
    global_idx: int,
    lengths: list[int],
    texts: list[str],
    attention_mask: torch.Tensor,
    batch_scalar: torch.Tensor,
    recorder: MLPIntermediateFullRecorder,
    positions_by_layer: dict[int, list[int]],
    k: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    prompt_len = int(attention_mask[sample_idx].sum().item())
    if prompt_len <= 0:
        raise ValueError("Prompt tokenization produced zero valid tokens")

    prompt_record_counts = [len(recorder.records[layer_id]) for layer_id in positions_by_layer]
    if not prompt_record_counts:
        raise ValueError("No selected layer records captured")
    total_records = min(prompt_record_counts)
    gen_steps = max(1, min(lengths[sample_idx], max(0, total_records - 1)))
    x = torch.zeros((3 + gen_steps, k), dtype=torch.float32)

    for layer_id, positions in positions_by_layer.items():
        layer_records = recorder.records[layer_id]
        prefill = layer_records[0][sample_idx]
        prompt_valid = prefill[attention_mask[sample_idx]]
        x[0, positions] = prompt_valid[-1]
        x[1, positions] = prompt_valid.mean(dim=0)
        x[2, positions] = prompt_valid.max(dim=0).values

        layer_gen_steps = min(gen_steps, max(0, len(layer_records) - 1))
        if layer_gen_steps > 0:
            gen_vals = torch.stack(
                [layer_records[t + 1][sample_idx, -1] for t in range(layer_gen_steps)],
                dim=0,
            )
            x[3 : 3 + layer_gen_steps, positions] = gen_vals

    row_id = row.get("id") or row.get("qid") or row.get("question_id") or f"{split}-{global_idx}"
    return {
        "x": x,
        "y": int(row[args.label_key]),
        "scalar_features": batch_scalar[sample_idx].detach().float().cpu(),
        "id": str(row_id),
        "generated_text": texts[sample_idx],
        "question": row.get("question", ""),
        "length": int(x.shape[0]),
    }


def materialize_prefill_generation_logits_data(
    samples: list[dict[str, Any]],
    split: str,
    neurons: list[dict[str, Any]],
    args: argparse.Namespace,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not samples:
        raise ValueError(f"No samples available to materialize split={split}")
    k = len(neurons)
    max_t = max(int(sample["x"].shape[0]) for sample in samples)
    x_padded = torch.zeros((len(samples), max_t, k), dtype=torch.float32)
    mask_padded = torch.zeros((len(samples), max_t), dtype=torch.bool)
    scalar_tensor = torch.stack([sample["scalar_features"] for sample in samples], dim=0)
    for idx, sample in enumerate(samples):
        x = sample["x"]
        t = int(x.shape[0])
        x_padded[idx, :t] = x
        mask_padded[idx, :t] = True

    metadata = {
        "dataset": args.dataset,
        "x_shape": list(x_padded.shape),
        "mask_shape": list(mask_padded.shape),
        "scalar_features_shape": list(scalar_tensor.shape),
        "scalar_feature_names": LOGIT_SCALAR_FEATURE_NAMES,
        "num_samples": len(samples),
        "top_k": k,
        "max_new_tokens": args.max_new_tokens,
        "max_input_tokens": args.max_input_tokens,
        "prompt_steps": ["prompt_last", "prompt_mean", "prompt_max"],
        "trajectory": "prompt pooled prefill activations plus real greedy generation down_proj input activations",
        "scalar_features": "low-cost greedy generation output probability statistics",
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return {
        "x": x_padded,
        "mask": mask_padded,
        "scalar_features": scalar_tensor,
        "y": torch.tensor([sample["y"] for sample in samples], dtype=torch.float32),
        "ids": [sample["id"] for sample in samples],
        "lengths": mask_padded.sum(dim=1).long(),
        "questions": [sample["question"] for sample in samples],
        "generated_texts": [sample["generated_text"] for sample in samples],
        "split": split,
        "neurons": neurons,
        "metadata": metadata,
    }


def atomic_torch_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp_path)
    tmp_path.replace(path)


def atomic_json_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp_path.replace(path)


def cli_config_for_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    config = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            config[key] = str(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            config[key] = value
        elif isinstance(value, list):
            config[key] = [str(item) if isinstance(item, Path) else item for item in value]
        else:
            config[key] = str(value)
    return config


def build_prefill_generation_sample(
    row: dict[str, Any],
    split: str,
    sample_idx: int,
    global_idx: int,
    lengths: list[int],
    texts: list[str],
    attention_mask: torch.Tensor,
    recorder: MLPIntermediateFullRecorder,
    positions_by_layer: dict[int, list[int]],
    k: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    prompt_len = int(attention_mask[sample_idx].sum().item())
    if prompt_len <= 0:
        raise ValueError("Prompt tokenization produced zero valid tokens")

    prompt_record_counts = [len(recorder.records[layer_id]) for layer_id in positions_by_layer]
    if not prompt_record_counts:
        raise ValueError("No selected layer records captured")
    total_records = min(prompt_record_counts)
    gen_steps = max(1, min(lengths[sample_idx], max(0, total_records - 1)))
    x = torch.zeros((3 + gen_steps, k), dtype=torch.float32)

    for layer_id, positions in positions_by_layer.items():
        layer_records = recorder.records[layer_id]
        prefill = layer_records[0][sample_idx]
        prompt_valid = prefill[attention_mask[sample_idx]]
        x[0, positions] = prompt_valid[-1]
        x[1, positions] = prompt_valid.mean(dim=0)
        x[2, positions] = prompt_valid.max(dim=0).values

        layer_gen_steps = min(gen_steps, max(0, len(layer_records) - 1))
        if layer_gen_steps > 0:
            gen_vals = torch.stack(
                [layer_records[t + 1][sample_idx, -1] for t in range(layer_gen_steps)],
                dim=0,
            )
            x[3 : 3 + layer_gen_steps, positions] = gen_vals

    row_id = row.get("id") or row.get("qid") or row.get("question_id") or f"{split}-{global_idx}"
    return {
        "x": x,
        "y": int(row[args.label_key]),
        "id": str(row_id),
        "generated_text": texts[sample_idx],
        "question": row.get("question", ""),
        "length": int(x.shape[0]),
    }


def materialize_prefill_generation_data(
    samples: list[dict[str, Any]],
    split: str,
    neurons: list[dict[str, Any]],
    args: argparse.Namespace,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not samples:
        raise ValueError(f"No samples available to materialize split={split}")
    k = len(neurons)
    max_t = max(int(sample["x"].shape[0]) for sample in samples)
    x_padded = torch.zeros((len(samples), max_t, k), dtype=torch.float32)
    mask_padded = torch.zeros((len(samples), max_t), dtype=torch.bool)
    for idx, sample in enumerate(samples):
        x = sample["x"]
        t = int(x.shape[0])
        x_padded[idx, :t] = x
        mask_padded[idx, :t] = True

    metadata = {
        "dataset": args.dataset,
        "x_shape": list(x_padded.shape),
        "mask_shape": list(mask_padded.shape),
        "num_samples": len(samples),
        "top_k": k,
        "max_new_tokens": args.max_new_tokens,
        "max_input_tokens": args.max_input_tokens,
        "prompt_steps": ["prompt_last", "prompt_mean", "prompt_max"],
        "trajectory": "prompt pooled prefill activations plus real greedy generation down_proj input activations",
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return {
        "x": x_padded,
        "mask": mask_padded,
        "y": torch.tensor([sample["y"] for sample in samples], dtype=torch.float32),
        "ids": [sample["id"] for sample in samples],
        "lengths": mask_padded.sum(dim=1).long(),
        "questions": [sample["question"] for sample in samples],
        "generated_texts": [sample["generated_text"] for sample in samples],
        "split": split,
        "neurons": neurons,
        "metadata": metadata,
    }


def trajectory_samples_from_partial(data: dict[str, Any]) -> list[dict[str, Any]]:
    samples = []
    x = data["x"]
    lengths = data.get("lengths", data["mask"].sum(dim=1).long())
    y = data["y"]
    ids = data.get("ids", [str(idx) for idx in range(int(x.shape[0]))])
    questions = data.get("questions", [""] * int(x.shape[0]))
    generated_texts = data.get("generated_texts", [""] * int(x.shape[0]))
    for idx in range(int(x.shape[0])):
        length = int(lengths[idx].item())
        samples.append(
            {
                "x": x[idx, :length].detach().float().cpu(),
                "y": int(y[idx].item()),
                "id": str(ids[idx]),
                "question": questions[idx],
                "generated_text": generated_texts[idx],
                "length": length,
            }
        )
    return samples


def save_trajectory_partial_checkpoint(
    checkpoint_dir: Path,
    split: str,
    samples: list[dict[str, Any]],
    total: int,
    neurons: list[dict[str, Any]],
    args: argparse.Namespace,
    started_at: float,
) -> Path:
    elapsed = time.time() - started_at
    processed = len(samples)
    rate = processed / elapsed if elapsed > 0 else 0.0
    remaining = (total - processed) / rate if rate > 0 else float("inf")
    data = materialize_prefill_generation_data(
        samples,
        split,
        neurons,
        args,
        extra_metadata={
            "checkpoint_partial": True,
            "processed": processed,
            "total": total,
            "generation_lengths": [int(sample["length"]) for sample in samples],
            "cli_args": cli_config_for_checkpoint(args),
        },
    )
    partial = {
        **data,
        "processed_ids": data["ids"],
        "processed_count": processed,
        "total_count": total,
        "generation_lengths": [int(sample["length"]) for sample in samples],
        "cli_args": cli_config_for_checkpoint(args),
    }
    partial_path = checkpoint_dir / f"{split}_partial.pt"
    atomic_torch_save(partial, partial_path)

    progress_path = checkpoint_dir / f"{split}_progress.json"
    progress = {
        "split": split,
        "processed": processed,
        "total": total,
        "elapsed_seconds": elapsed,
        "estimated_remaining_seconds": remaining,
        "checkpoint_path": str(partial_path),
        "updated_at_unix": time.time(),
    }
    atomic_json_save(progress, progress_path)
    remaining_text = "inf" if math.isinf(remaining) else f"{remaining:.1f}s"
    print(
        f"checkpoint split={split} processed={processed}/{total} "
        f"elapsed={elapsed:.1f}s eta={remaining_text} path={partial_path}"
    )
    return partial_path


def extract_prefill_generation_split_checkpointed(
    rows: list[dict[str, Any]],
    split: str,
    model,
    tokenizer,
    device: torch.device,
    neurons: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.checkpoint_dir is None:
        return extract_prefill_generation_split(rows, split, model, tokenizer, device, neurons, args)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    selected_by_layer, positions_by_layer = selected_neuron_tensors(neurons)
    k = len(neurons)
    total = len(rows)
    partial_path = args.checkpoint_dir / f"{split}_partial.pt"
    progress_path = args.checkpoint_dir / f"{split}_progress.json"
    samples: list[dict[str, Any]] = []
    done_ids: set[str] = set()
    prior_elapsed = 0.0
    if args.resume and partial_path.exists():
        partial = torch.load(partial_path, map_location="cpu", weights_only=False)
        if partial.get("neurons") != neurons:
            raise ValueError(f"Checkpoint neurons do not match --neurons-json: {partial_path}")
        samples = trajectory_samples_from_partial(partial)
        target_ids = {
            str(row.get("id") or row.get("qid") or row.get("question_id") or f"{split}-{idx}")
            for idx, row in enumerate(rows)
        }
        outside_ids = [sample["id"] for sample in samples if sample["id"] not in target_ids]
        if outside_ids:
            raise ValueError(
                f"Checkpoint {partial_path} contains {len(outside_ids)} IDs outside the requested split filter; "
                f"first={outside_ids[:10]}"
            )
        done_ids = {sample["id"] for sample in samples}
        if progress_path.exists():
            try:
                prior_elapsed = float(json.loads(progress_path.read_text(encoding="utf-8")).get("elapsed_seconds", 0.0))
            except (json.JSONDecodeError, TypeError, ValueError):
                prior_elapsed = 0.0
        print(f"resume split={split} loaded={len(samples)} checkpoint={partial_path}")

    started_at = time.time() - prior_elapsed
    since_save = 0
    with MLPIntermediateFullRecorder(model, selected_by_layer) as recorder:
        pending_batch: list[tuple[int, dict[str, Any]]] = []
        for global_idx, row in enumerate(rows):
            row_id = str(row.get("id") or row.get("qid") or row.get("question_id") or f"{split}-{global_idx}")
            if row_id in done_ids:
                continue
            pending_batch.append((global_idx, row))
            if len(pending_batch) < args.batch_size:
                continue

            prompts = [item[1][args.prompt_key] for item in pending_batch]
            lengths, texts, attention_mask = generate_with_full_records(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                device=device,
                recorder=recorder,
                max_new_tokens=args.max_new_tokens,
                max_input_tokens=args.max_input_tokens,
                min_steps=args.min_steps,
            )
            for sample_idx, (global_idx, row) in enumerate(pending_batch):
                sample = build_prefill_generation_sample(
                    row,
                    split,
                    sample_idx,
                    global_idx,
                    lengths,
                    texts,
                    attention_mask,
                    recorder,
                    positions_by_layer,
                    k,
                    args,
                )
                samples.append(sample)
                done_ids.add(sample["id"])
                since_save += 1
            pending_batch = []
            if args.save_every > 0 and since_save >= args.save_every:
                save_trajectory_partial_checkpoint(args.checkpoint_dir, split, samples, total, neurons, args, started_at)
                since_save = 0

        if pending_batch:
            prompts = [item[1][args.prompt_key] for item in pending_batch]
            lengths, texts, attention_mask = generate_with_full_records(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                device=device,
                recorder=recorder,
                max_new_tokens=args.max_new_tokens,
                max_input_tokens=args.max_input_tokens,
                min_steps=args.min_steps,
            )
            for sample_idx, (global_idx, row) in enumerate(pending_batch):
                sample = build_prefill_generation_sample(
                    row,
                    split,
                    sample_idx,
                    global_idx,
                    lengths,
                    texts,
                    attention_mask,
                    recorder,
                    positions_by_layer,
                    k,
                    args,
                )
                samples.append(sample)
                done_ids.add(sample["id"])

    order = {
        str(row.get("id") or row.get("qid") or row.get("question_id") or f"{split}-{idx}"): idx
        for idx, row in enumerate(rows)
    }
    samples.sort(key=lambda sample: order[sample["id"]])
    save_trajectory_partial_checkpoint(args.checkpoint_dir, split, samples, total, neurons, args, started_at)
    return materialize_prefill_generation_data(
        samples,
        split,
        neurons,
        args,
        extra_metadata={
            "checkpoint_partial": False,
            "generation_lengths": [int(sample["length"]) for sample in samples],
            "cli_args": cli_config_for_checkpoint(args),
        },
    )


def save_logits_partial_checkpoint(
    checkpoint_dir: Path,
    split: str,
    samples: list[dict[str, Any]],
    total: int,
    neurons: list[dict[str, Any]],
    args: argparse.Namespace,
    started_at: float,
) -> Path:
    elapsed = time.time() - started_at
    processed = len(samples)
    rate = processed / elapsed if elapsed > 0 else 0.0
    remaining = (total - processed) / rate if rate > 0 else float("inf")
    data = materialize_prefill_generation_logits_data(
        samples,
        split,
        neurons,
        args,
        extra_metadata={
            "checkpoint_partial": True,
            "processed": processed,
            "total": total,
            "generation_lengths": [int(sample["length"]) for sample in samples],
            "cli_args": cli_config_for_checkpoint(args),
        },
    )
    partial = {
        **data,
        "processed_ids": data["ids"],
        "processed_count": processed,
        "total_count": total,
        "generation_lengths": [int(sample["length"]) for sample in samples],
        "cli_args": cli_config_for_checkpoint(args),
    }
    partial_path = checkpoint_dir / f"{split}_partial.pt"
    atomic_torch_save(partial, partial_path)

    progress_path = checkpoint_dir / "progress.json"
    if progress_path.exists():
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            progress = {}
    else:
        progress = {}
    progress[split] = {
        "split": split,
        "processed": processed,
        "total": total,
        "elapsed_seconds": elapsed,
        "estimated_remaining_seconds": remaining,
        "checkpoint_path": str(partial_path),
        "updated_at_unix": time.time(),
    }
    atomic_json_save(progress, progress_path)
    remaining_text = "inf" if math.isinf(remaining) else f"{remaining:.1f}s"
    print(
        f"checkpoint split={split} processed={processed}/{total} "
        f"elapsed={elapsed:.1f}s eta={remaining_text} path={partial_path}"
    )
    return partial_path


def samples_from_partial(data: dict[str, Any]) -> list[dict[str, Any]]:
    samples = []
    x = data["x"]
    lengths = data.get("lengths", data["mask"].sum(dim=1).long())
    scalar = data["scalar_features"]
    y = data["y"]
    ids = data.get("ids", [str(idx) for idx in range(int(x.shape[0]))])
    questions = data.get("questions", [""] * int(x.shape[0]))
    generated_texts = data.get("generated_texts", [""] * int(x.shape[0]))
    for idx in range(int(x.shape[0])):
        length = int(lengths[idx].item())
        samples.append(
            {
                "x": x[idx, :length].detach().float().cpu(),
                "y": int(y[idx].item()),
                "scalar_features": scalar[idx].detach().float().cpu(),
                "id": str(ids[idx]),
                "question": questions[idx],
                "generated_text": generated_texts[idx],
                "length": length,
            }
        )
    return samples


def extract_prefill_generation_logits_split_checkpointed(
    rows: list[dict[str, Any]],
    split: str,
    model,
    tokenizer,
    device: torch.device,
    neurons: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.checkpoint_dir is None:
        return extract_prefill_generation_logits_split(rows, split, model, tokenizer, device, neurons, args)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    selected_by_layer, positions_by_layer = selected_neuron_tensors(neurons)
    k = len(neurons)
    total = len(rows)
    partial_path = args.checkpoint_dir / f"{split}_partial.pt"
    samples: list[dict[str, Any]] = []
    done_ids: set[str] = set()
    if args.resume and partial_path.exists():
        partial = torch.load(partial_path, map_location="cpu", weights_only=False)
        samples = samples_from_partial(partial)
        target_ids = {
            str(row.get("id") or row.get("qid") or row.get("question_id") or f"{split}-{idx}")
            for idx, row in enumerate(rows)
        }
        samples = [sample for sample in samples if sample["id"] in target_ids]
        done_ids = {sample["id"] for sample in samples}
        print(f"resume split={split} loaded={len(samples)} checkpoint={partial_path}")

    started_at = time.time()
    since_save = 0
    with MLPIntermediateFullRecorder(model, selected_by_layer) as recorder:
        pending_batch: list[tuple[int, dict[str, Any]]] = []
        for global_idx, row in enumerate(rows):
            row_id = str(row.get("id") or row.get("qid") or row.get("question_id") or f"{split}-{global_idx}")
            if row_id in done_ids:
                continue
            pending_batch.append((global_idx, row))
            if len(pending_batch) < args.batch_size:
                continue

            prompts = [item[1][args.prompt_key] for item in pending_batch]
            lengths, texts, attention_mask, batch_scalar = generate_with_full_records_and_logits(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                device=device,
                recorder=recorder,
                max_new_tokens=args.max_new_tokens,
                max_input_tokens=args.max_input_tokens,
                min_steps=args.min_steps,
            )
            for sample_idx, (global_idx, row) in enumerate(pending_batch):
                sample = build_prefill_generation_logits_sample(
                    row,
                    split,
                    sample_idx,
                    global_idx,
                    lengths,
                    texts,
                    attention_mask,
                    batch_scalar,
                    recorder,
                    positions_by_layer,
                    k,
                    args,
                )
                samples.append(sample)
                done_ids.add(sample["id"])
                since_save += 1
            pending_batch = []
            if args.save_every > 0 and since_save >= args.save_every:
                save_logits_partial_checkpoint(args.checkpoint_dir, split, samples, total, neurons, args, started_at)
                since_save = 0

        if pending_batch:
            prompts = [item[1][args.prompt_key] for item in pending_batch]
            lengths, texts, attention_mask, batch_scalar = generate_with_full_records_and_logits(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                device=device,
                recorder=recorder,
                max_new_tokens=args.max_new_tokens,
                max_input_tokens=args.max_input_tokens,
                min_steps=args.min_steps,
            )
            for sample_idx, (global_idx, row) in enumerate(pending_batch):
                sample = build_prefill_generation_logits_sample(
                    row,
                    split,
                    sample_idx,
                    global_idx,
                    lengths,
                    texts,
                    attention_mask,
                    batch_scalar,
                    recorder,
                    positions_by_layer,
                    k,
                    args,
                )
                samples.append(sample)
                done_ids.add(sample["id"])
                since_save += 1

    save_logits_partial_checkpoint(args.checkpoint_dir, split, samples, total, neurons, args, started_at)
    return materialize_prefill_generation_logits_data(
        samples,
        split,
        neurons,
        args,
        extra_metadata={
            "checkpoint_partial": False,
            "generation_lengths": [int(sample["length"]) for sample in samples],
            "cli_args": cli_config_for_checkpoint(args),
        },
    )


def command_extract_prefill_generation_trajectories_with_logits(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    resolve_label_paths(args)
    device = resolve_device(args.device)
    neurons = load_top_neurons(args.neurons_json)
    model, tokenizer = load_qwen(args.model_path, device, getattr(args, "model_key", None))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    split_paths = {
        "train": args.train_path,
        "dev": args.dev_path,
        "test": args.test_path,
    }
    for split in args.splits:
        path = split_paths[split]
        rows = read_jsonl(path, args.limit)
        if args.max_samples_per_split is not None:
            rows = rows[: args.max_samples_per_split]
        data = extract_prefill_generation_logits_split_checkpointed(
            rows=rows,
            split=split,
            model=model,
            tokenizer=tokenizer,
            device=device,
            neurons=neurons,
            args=args,
        )
        out_path = args.out_dir / f"{split}.pt"
        torch.save(data, out_path)
        print(
            f"wrote {out_path} "
            f"x_shape={tuple(data['x'].shape)} scalar_shape={tuple(data['scalar_features'].shape)}"
        )


def extract_prefill_only_split(
    rows: list[dict[str, Any]],
    split: str,
    model,
    tokenizer,
    device: torch.device,
    neurons: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    selected_by_layer, positions_by_layer = selected_neuron_tensors(neurons)
    k = len(neurons)
    xs: list[torch.Tensor] = []
    ys: list[int] = []
    questions: list[str] = []

    with MLPIntermediateFullRecorder(model, selected_by_layer) as recorder:
        for batch_rows in tqdm(list(batched(rows, args.batch_size)), desc=f"prefill-only-{split}"):
            prompts = [row[args.prompt_key] for row in batch_rows]
            attention_mask = prefill_with_full_records(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                device=device,
                recorder=recorder,
                max_input_tokens=args.max_input_tokens,
            )

            for sample_idx, row in enumerate(batch_rows):
                prompt_len = int(attention_mask[sample_idx].sum().item())
                if prompt_len <= 0:
                    raise ValueError("Prompt tokenization produced zero valid tokens")

                x = torch.zeros((3, k), dtype=torch.float32)
                for layer_id, positions in positions_by_layer.items():
                    layer_records = recorder.records[layer_id]
                    if not layer_records:
                        raise ValueError(f"No prefill records captured for layer {layer_id}")
                    prefill = layer_records[0][sample_idx]
                    prompt_valid = prefill[attention_mask[sample_idx]]
                    x[0, positions] = prompt_valid[-1]
                    x[1, positions] = prompt_valid.mean(dim=0)
                    x[2, positions] = prompt_valid.max(dim=0).values

                xs.append(x)
                ys.append(int(row[args.label_key]))
                questions.append(row.get("question", ""))

    x_stacked = torch.stack(xs, dim=0)
    mask = torch.ones((len(xs), 3), dtype=torch.bool)
    return {
        "x": x_stacked,
        "mask": mask,
        "y": torch.tensor(ys, dtype=torch.float32),
        "lengths": mask.sum(dim=1).long(),
        "questions": questions,
        "generated_texts": [""] * len(xs),
        "split": split,
        "neurons": neurons,
        "metadata": {
            "x_shape": list(x_stacked.shape),
            "mask_shape": list(mask.shape),
            "num_samples": len(xs),
            "top_k": k,
            "max_input_tokens": args.max_input_tokens,
            "prompt_steps": ["prompt_last", "prompt_mean", "prompt_max"],
            "trajectory": "prompt pooled prefill-only down_proj input activations",
        },
    }


def rich_prefill_steps(prompt_valid: torch.Tensor) -> torch.Tensor:
    if prompt_valid.numel() == 0:
        raise ValueError("Prompt tokenization produced zero valid tokens")
    last_8 = prompt_valid[-8:]
    last_16 = prompt_valid[-16:]
    return torch.stack(
        [
            prompt_valid[-1],
            prompt_valid.mean(dim=0),
            prompt_valid.max(dim=0).values,
            prompt_valid.std(dim=0, unbiased=False),
            last_8.mean(dim=0),
            last_8.max(dim=0).values,
            last_16.mean(dim=0),
            last_16.max(dim=0).values,
        ],
        dim=0,
    )


def extract_prefill_rich_split(
    rows: list[dict[str, Any]],
    split: str,
    model,
    tokenizer,
    device: torch.device,
    neurons: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    selected_by_layer, positions_by_layer = selected_neuron_tensors(neurons)
    k = len(neurons)
    xs: list[torch.Tensor] = []
    ys: list[int] = []
    ids: list[str] = []
    questions: list[str] = []

    offset = 0
    with MLPIntermediateFullRecorder(model, selected_by_layer) as recorder:
        for batch_rows in tqdm(list(batched(rows, args.batch_size)), desc=f"prefill-rich-{split}"):
            prompts = [row[args.prompt_key] for row in batch_rows]
            attention_mask = prefill_with_full_records(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                device=device,
                recorder=recorder,
                max_input_tokens=args.max_input_tokens,
            )

            for sample_idx, row in enumerate(batch_rows):
                x = torch.zeros((8, k), dtype=torch.float32)
                for layer_id, positions in positions_by_layer.items():
                    layer_records = recorder.records[layer_id]
                    if not layer_records:
                        raise ValueError(f"No prefill records captured for layer {layer_id}")
                    prefill = layer_records[0][sample_idx]
                    prompt_valid = prefill[attention_mask[sample_idx]]
                    x[:, positions] = rich_prefill_steps(prompt_valid)

                row_id = row.get("id") or row.get("qid") or row.get("question_id") or f"{split}-{offset + sample_idx}"
                xs.append(x)
                ys.append(int(row[args.label_key]))
                ids.append(str(row_id))
                questions.append(row.get("question", ""))
            offset += len(batch_rows)

    x_stacked = torch.stack(xs, dim=0)
    mask = torch.ones((len(xs), 8), dtype=torch.bool)
    return {
        "x": x_stacked,
        "mask": mask,
        "y": torch.tensor(ys, dtype=torch.float32),
        "ids": ids,
        "lengths": mask.sum(dim=1).long(),
        "questions": questions,
        "generated_texts": [""] * len(xs),
        "split": split,
        "neurons": neurons,
        "metadata": {
            "x_shape": list(x_stacked.shape),
            "mask_shape": list(mask.shape),
            "num_samples": len(xs),
            "top_k": k,
            "max_input_tokens": args.max_input_tokens,
            "prompt_steps": RICH_PREFILL_FEATURE_NAMES,
            "trajectory": "strict pre-generation rich pooled prompt MLP down_proj input activations",
            "pre_generation_only": True,
            "uses_generate": False,
            "uses_output_probability_features": False,
        },
    }


def command_extract_prefill_rich_trajectories(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    resolve_label_paths(args)
    device = resolve_device(args.device)
    neurons = load_top_neurons(args.neurons_json)
    model, tokenizer = load_qwen(args.model_path, device, getattr(args, "model_key", None))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    split_paths = {
        "train": args.train_path,
        "dev": args.dev_path,
        "test": args.test_path,
    }
    for split in args.splits:
        path = split_paths[split]
        rows = read_jsonl(path, args.limit)
        data = extract_prefill_rich_split(
            rows=rows,
            split=split,
            model=model,
            tokenizer=tokenizer,
            device=device,
            neurons=neurons,
            args=args,
        )
        out_path = args.out_dir / f"{split}.pt"
        torch.save(data, out_path)
        print(f"wrote {out_path} shape={tuple(data['x'].shape)}")


def extract_prefill_lastseq_split(
    rows: list[dict[str, Any]],
    split: str,
    model,
    tokenizer,
    device: torch.device,
    neurons: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    selected_by_layer, positions_by_layer = selected_neuron_tensors(neurons)
    k = len(neurons)
    last_n = int(getattr(args, "last_n_prompt_tokens", getattr(args, "last_n", 16)))
    if last_n <= 0:
        raise ValueError("--last-n-prompt-tokens must be positive")
    xs: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    ys: list[int] = []
    ids: list[str] = []
    questions: list[str] = []

    offset = 0
    with MLPIntermediateFullRecorder(model, selected_by_layer) as recorder:
        for batch_rows in tqdm(list(batched(rows, args.batch_size)), desc=f"prefill-lastseq-{split}"):
            prompts = [row[args.prompt_key] for row in batch_rows]
            attention_mask = prefill_with_full_records(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                device=device,
                recorder=recorder,
                max_input_tokens=args.max_input_tokens,
            )

            for sample_idx, row in enumerate(batch_rows):
                prompt_len = int(attention_mask[sample_idx].sum().item())
                seq_len = min(last_n, prompt_len)
                if seq_len <= 0:
                    raise ValueError("Prompt tokenization produced zero valid tokens")

                x = torch.zeros((last_n, k), dtype=torch.float32)
                mask = torch.zeros(last_n, dtype=torch.bool)
                start = last_n - seq_len
                mask[start:] = True
                for layer_id, positions in positions_by_layer.items():
                    layer_records = recorder.records[layer_id]
                    if not layer_records:
                        raise ValueError(f"No prefill records captured for layer {layer_id}")
                    prefill = layer_records[0][sample_idx]
                    prompt_valid = prefill[attention_mask[sample_idx]]
                    x[start:, positions] = prompt_valid[-seq_len:]

                row_id = row.get("id") or row.get("qid") or row.get("question_id") or f"{split}-{offset + sample_idx}"
                xs.append(x)
                masks.append(mask)
                ys.append(int(row[args.label_key]))
                ids.append(str(row_id))
                questions.append(row.get("question", ""))
            offset += len(batch_rows)

    x_stacked = torch.stack(xs, dim=0)
    mask_stacked = torch.stack(masks, dim=0)
    return {
        "x": x_stacked,
        "mask": mask_stacked,
        "y": torch.tensor(ys, dtype=torch.float32),
        "ids": ids,
        "lengths": mask_stacked.sum(dim=1).long(),
        "questions": questions,
        "generated_texts": [""] * len(xs),
        "split": split,
        "neurons": neurons,
        "metadata": {
            "x_shape": list(x_stacked.shape),
            "mask_shape": list(mask_stacked.shape),
            "num_samples": len(xs),
            "top_k": k,
            "last_n_prompt_tokens": last_n,
            "max_input_tokens": args.max_input_tokens,
            "trajectory": "strict pre-generation last prompt-token MLP down_proj input activation sequence",
            "padding": "left padding; valid prompt tokens are right-aligned",
            "pre_generation_only": True,
            "uses_generate": False,
            "uses_output_probability_features": False,
        },
    }


def command_extract_prefill_lastseq_trajectories(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    resolve_label_paths(args)
    device = resolve_device(args.device)
    neurons = load_top_neurons(args.neurons_json)
    model, tokenizer = load_qwen(args.model_path, device, getattr(args, "model_key", None))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    split_paths = {
        "train": args.train_path,
        "dev": args.dev_path,
        "test": args.test_path,
    }
    for split in args.splits:
        path = split_paths[split]
        rows = read_jsonl(path, args.limit)
        data = extract_prefill_lastseq_split(
            rows=rows,
            split=split,
            model=model,
            tokenizer=tokenizer,
            device=device,
            neurons=neurons,
            args=args,
        )
        out_path = args.out_dir / f"{split}.pt"
        torch.save(data, out_path)
        print(f"wrote {out_path} shape={tuple(data['x'].shape)}")


def command_extract_prefill_token_sequence_trajectories(args: argparse.Namespace) -> None:
    command_extract_prefill_lastseq_trajectories(args)


def command_extract_prefill_only_trajectories(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    resolve_label_paths(args)
    device = resolve_device(args.device)
    neurons = load_top_neurons(args.neurons_json)
    model, tokenizer = load_qwen(args.model_path, device, getattr(args, "model_key", None))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    split_paths = {
        "train": args.train_path,
        "dev": args.dev_path,
        "test": args.test_path,
    }
    for split in args.splits:
        path = split_paths[split]
        rows = read_jsonl(path, args.limit)
        data = extract_prefill_only_split(
            rows=rows,
            split=split,
            model=model,
            tokenizer=tokenizer,
            device=device,
            neurons=neurons,
            args=args,
        )
        out_path = args.out_dir / f"{split}.pt"
        torch.save(data, out_path)
        print(f"wrote {out_path} shape={tuple(data['x'].shape)}")


def question_only_prompt(row: dict[str, Any], question_key: str) -> str:
    question = row.get(question_key) or row.get("question")
    if question is None:
        raise KeyError(f"Missing question key: {question_key}")
    return (
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


@torch.inference_mode()
def residual_hidden_states_for_prompts(
    model,
    tokenizer,
    prompts: list[str],
    device: torch.device,
    max_input_tokens: int,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    batch = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
    ).to(device)
    attention_mask = batch["attention_mask"].detach().cpu().bool()
    outs = model(**batch, use_cache=False, output_hidden_states=True)
    hidden_states = tuple(h.detach().float().cpu() for h in outs.hidden_states)
    return hidden_states, attention_mask


def command_select_residual_question_only(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    resolve_label_paths(args)
    device = resolve_device(args.device)
    rows = read_selection_rows(args)
    model, tokenizer = load_qwen(args.model_path, device, getattr(args, "model_key", None))
    num_layers = int(model.config.num_hidden_layers) + 1
    hidden_size = int(model.config.hidden_size)
    stats = PrefillOnlyImportanceStats(num_layers=num_layers, feature_size=hidden_size)

    for batch_rows in tqdm(list(batched(rows, args.batch_size)), desc="select-residual-question-only"):
        prompts = [question_only_prompt(row, args.question_key) for row in batch_rows]
        labels = [int(row[args.label_key]) for row in batch_rows]
        hidden_states, attention_mask = residual_hidden_states_for_prompts(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            device=device,
            max_input_tokens=args.max_input_tokens,
        )
        for sample_idx, label in enumerate(labels):
            stats.add_sample_count(label)
            mask = attention_mask[sample_idx]
            for layer_id, layer_hidden in enumerate(hidden_states):
                valid = layer_hidden[sample_idx, mask]
                if valid.numel() == 0:
                    raise ValueError("Question-only prompt produced zero valid tokens")
                stats.update(
                    label=label,
                    layer=layer_id,
                    prompt_last=valid[-1],
                    prompt_mean=valid.mean(dim=0),
                    prompt_max=valid.max(dim=0).values,
                )

    weights = {
        "prompt_last": args.w_prompt_last,
        "prompt_mean": args.w_prompt_mean,
        "prompt_max": args.w_prompt_max,
    }
    scores = stats.scores(
        w_prompt_last=args.w_prompt_last,
        w_prompt_mean=args.w_prompt_mean,
        w_prompt_max=args.w_prompt_max,
    )
    flat = scores.reshape(-1)
    top_scores, top_indices = torch.topk(flat, k=args.top_k)
    selected = []
    for score, flat_idx in zip(top_scores.tolist(), top_indices.tolist()):
        selected.append(
            {
                "layer": int(flat_idx // hidden_size),
                "dim": int(flat_idx % hidden_size),
                "score": float(score),
            }
        )

    write_json(args.out_json, selected)
    summary = {
        "dataset": args.dataset,
        "train_path": str(args.train_path),
        "model_path": str(args.model_path),
        "num_train_samples": len(rows),
        "class_counts": {"wrong_0": int(stats.count[0].item()), "correct_1": int(stats.count[1].item())},
        "num_residual_layers_including_embedding": num_layers,
        "hidden_size": hidden_size,
        "candidate_nodes": num_layers * hidden_size,
        "top_k": args.top_k,
        "score_formula": weights,
        "max_input_tokens": args.max_input_tokens,
        "output": str(args.out_json),
        "selection_data": "train split only",
        "prompt_source": "question only",
        "pre_generation_only": True,
        "uses_generate": False,
        "uses_output_probability_features": False,
    }
    write_json(args.out_json.with_name(args.out_json.stem + "_summary.json"), summary)


def load_top_residual_dims(path: Path) -> list[dict[str, Any]]:
    dims = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(dims, list) or not dims:
        raise ValueError(f"No residual dims found in {path}")
    return dims


def selected_residual_tensors(
    dims: list[dict[str, Any]],
) -> tuple[dict[int, torch.Tensor], dict[int, list[int]]]:
    indices_by_layer: dict[int, list[int]] = defaultdict(list)
    positions_by_layer: dict[int, list[int]] = defaultdict(list)
    for pos, item in enumerate(dims):
        layer = int(item["layer"])
        dim = int(item["dim"])
        indices_by_layer[layer].append(dim)
        positions_by_layer[layer].append(pos)
    tensor_by_layer = {
        layer: torch.tensor(indices, dtype=torch.long)
        for layer, indices in indices_by_layer.items()
    }
    return tensor_by_layer, positions_by_layer


def extract_residual_question_only_split(
    rows: list[dict[str, Any]],
    split: str,
    model,
    tokenizer,
    device: torch.device,
    dims: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    selected_by_layer, positions_by_layer = selected_residual_tensors(dims)
    k = len(dims)
    xs: list[torch.Tensor] = []
    ys: list[int] = []
    ids: list[str] = []
    questions: list[str] = []

    offset = 0
    for batch_rows in tqdm(list(batched(rows, args.batch_size)), desc=f"residual-question-only-{split}"):
        prompts = [question_only_prompt(row, args.question_key) for row in batch_rows]
        hidden_states, attention_mask = residual_hidden_states_for_prompts(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            device=device,
            max_input_tokens=args.max_input_tokens,
        )
        for sample_idx, row in enumerate(batch_rows):
            x = torch.zeros((3, k), dtype=torch.float32)
            mask = attention_mask[sample_idx]
            for layer_id, positions in positions_by_layer.items():
                idx = selected_by_layer[layer_id]
                valid = hidden_states[layer_id][sample_idx, mask]
                if valid.numel() == 0:
                    raise ValueError("Question-only prompt produced zero valid tokens")
                selected = valid.index_select(dim=-1, index=idx)
                x[0, positions] = selected[-1]
                x[1, positions] = selected.mean(dim=0)
                x[2, positions] = selected.max(dim=0).values
            row_id = row.get("id") or row.get("qid") or row.get("question_id") or f"{split}-{offset + sample_idx}"
            xs.append(x)
            ys.append(int(row[args.label_key]))
            ids.append(str(row_id))
            questions.append(row.get("question", ""))
        offset += len(batch_rows)

    x_stacked = torch.stack(xs, dim=0)
    mask = torch.ones((len(xs), 3), dtype=torch.bool)
    return {
        "x": x_stacked,
        "mask": mask,
        "y": torch.tensor(ys, dtype=torch.float32),
        "ids": ids,
        "lengths": mask.sum(dim=1).long(),
        "questions": questions,
        "generated_texts": [""] * len(xs),
        "split": split,
        "residual_dims": dims,
        "metadata": {
            "x_shape": list(x_stacked.shape),
            "mask_shape": list(mask.shape),
            "num_samples": len(xs),
            "top_k": k,
            "max_input_tokens": args.max_input_tokens,
            "prompt_steps": ["question_last", "question_mean", "question_max"],
            "trajectory": "question-only residual stream pooled prefill activations",
            "pre_generation_only": True,
            "uses_generate": False,
            "uses_output_probability_features": False,
        },
    }


def command_extract_residual_question_only(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    resolve_label_paths(args)
    device = resolve_device(args.device)
    dims = load_top_residual_dims(args.residual_json)
    model, tokenizer = load_qwen(args.model_path, device, getattr(args, "model_key", None))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    split_paths = {
        "train": args.train_path,
        "dev": args.dev_path,
        "test": args.test_path,
    }
    for split in args.splits:
        rows = read_jsonl(split_paths[split], args.limit)
        data = extract_residual_question_only_split(
            rows=rows,
            split=split,
            model=model,
            tokenizer=tokenizer,
            device=device,
            dims=dims,
            args=args,
        )
        out_path = args.out_dir / f"{split}.pt"
        torch.save(data, out_path)
        print(f"wrote {out_path} shape={tuple(data['x'].shape)}")


class GRUCorrectnessPredictor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
    ) -> None:
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=gru_dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )
        direction_mult = 2 if bidirectional else 1
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * direction_mult, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        lengths_cpu = lengths.detach().cpu().clamp_min(1)
        packed = pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
        _out, h_n = self.gru(packed)
        if self.gru.bidirectional:
            final = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            final = h_n[-1]
        return self.head(final).squeeze(-1)


class GRUFusionCorrectnessPredictor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        scalar_dim: int,
        hidden_size: int,
        scalar_hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
    ) -> None:
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=gru_dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )
        direction_mult = 2 if bidirectional else 1
        self.scalar_mlp = nn.Sequential(
            nn.Linear(scalar_dim, scalar_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(scalar_hidden_size, scalar_hidden_size),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * direction_mult + scalar_hidden_size, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, scalar_features: torch.Tensor) -> torch.Tensor:
        lengths_cpu = lengths.detach().cpu().clamp_min(1)
        packed = pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
        _out, h_n = self.gru(packed)
        if self.gru.bidirectional:
            traj_final = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            traj_final = h_n[-1]
        scalar_final = self.scalar_mlp(scalar_features)
        fused = torch.cat([traj_final, scalar_final], dim=-1)
        return self.head(fused).squeeze(-1)


class LSTMFusionCorrectnessPredictor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        scalar_dim: int,
        hidden_size: int,
        scalar_hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )
        direction_mult = 2 if bidirectional else 1
        self.scalar_mlp = nn.Sequential(
            nn.Linear(scalar_dim, scalar_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(scalar_hidden_size, scalar_hidden_size),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * direction_mult + scalar_hidden_size, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, scalar_features: torch.Tensor) -> torch.Tensor:
        lengths_cpu = lengths.detach().cpu().clamp_min(1)
        packed = pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
        _out, (h_n, _c_n) = self.lstm(packed)
        if self.lstm.bidirectional:
            traj_final = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            traj_final = h_n[-1]
        scalar_final = self.scalar_mlp(scalar_features)
        fused = torch.cat([traj_final, scalar_final], dim=-1)
        return self.head(fused).squeeze(-1)


def expand_features(x: torch.Tensor, mask: torch.Tensor, mode: str) -> torch.Tensor:
    mask_f = mask.unsqueeze(-1).float()
    x = x * mask_f
    if mode == "activation":
        return x
    if mode != "activation_dynamics":
        raise ValueError(f"Unknown feature mode: {mode}")

    prev = torch.zeros_like(x)
    prev[:, 1:] = x[:, :-1]
    delta = (x - prev) * mask_f
    delta[:, 0] = 0.0
    abs_delta = delta.abs()
    counts = mask_f.cumsum(dim=1).clamp_min(1.0)
    running_sum = x.cumsum(dim=1)
    running_mean = running_sum / counts
    running_sq_mean = (x * x).cumsum(dim=1) / counts
    running_std = (running_sq_mean - running_mean.square()).clamp_min(0.0).sqrt()
    features = torch.cat([x, delta, abs_delta, running_mean, running_std], dim=-1)
    return features * mask_f


def command_augment_trajectories(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    feature_order = list(args.features)
    allowed = {"activation", "delta", "abs_delta", "running_mean", "running_std"}
    unknown = [name for name in feature_order if name not in allowed]
    if unknown:
        raise ValueError(f"Unknown augmented features: {unknown}")

    for split in args.splits:
        in_path = args.input_dir / f"{split}.pt"
        out_path = args.out_dir / f"{split}.pt"
        data = load_trajectory(in_path)
        x = data["x"].float()
        mask = data["mask"].bool()
        full = expand_features(x, mask, "activation_dynamics")
        k = int(x.shape[-1])
        feature_map = {
            "activation": full[..., 0 * k : 1 * k],
            "delta": full[..., 1 * k : 2 * k],
            "abs_delta": full[..., 2 * k : 3 * k],
            "running_mean": full[..., 3 * k : 4 * k],
            "running_std": full[..., 4 * k : 5 * k],
        }
        out_data = dict(data)
        out_data["x"] = torch.cat([feature_map[name] for name in feature_order], dim=-1)
        out_data["mask"] = mask
        out_data["lengths"] = mask.sum(dim=1).long()

        metadata = dict(out_data.get("metadata", {}))
        metadata.update(
            {
                "x_shape": list(out_data["x"].shape),
                "mask_shape": list(mask.shape),
                "precomputed_features": True,
                "feature_columns": feature_order,
                "base_feature_dim": k,
                "feature_dim": int(out_data["x"].shape[-1]),
                "source_path": str(in_path),
            }
        )
        out_data["metadata"] = metadata
        torch.save(out_data, out_path)
        print(f"wrote {out_path} shape={tuple(out_data['x'].shape)} features={feature_order}")


def load_trajectory(path: Path) -> dict[str, Any]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    # Versioned aligned protocols may store trajectories as float16 on disk.
    # GRU weights remain float32, so normalize the in-memory training dtype here.
    if isinstance(data, dict) and isinstance(data.get("x"), torch.Tensor):
        data["x"] = data["x"].float()
    return data


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        if right == 1.0:
            idx = (y_prob >= left) & (y_prob <= right)
        else:
            idx = (y_prob >= left) & (y_prob < right)
        if not np.any(idx):
            continue
        ece += float(idx.mean()) * abs(float(y_true[idx].mean()) - float(y_prob[idx].mean()))
    return ece


def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    finite = np.isfinite(thresholds)
    if not np.any(finite):
        return 0.5
    youden = tpr - fpr
    finite_indices = np.where(finite)[0]
    idx = int(finite_indices[np.argmax(youden[finite_indices])])
    return float(thresholds[idx])


def write_predictions_csv(
    path: Path,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    questions: list[str] | None = None,
    sample_ids: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y_pred = (y_prob >= threshold).astype(np.int64)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["idx", "sample_id", "y", "p_correct", "risk", "pred", "question"],
        )
        writer.writeheader()
        for idx, (y, p, pred) in enumerate(zip(y_true, y_prob, y_pred)):
            writer.writerow(
                {
                    "idx": idx,
                    "sample_id": "" if sample_ids is None else sample_ids[idx],
                    "y": int(y),
                    "p_correct": float(p),
                    "risk": float(1.0 - p),
                    "pred": int(pred),
                    "question": "" if questions is None else questions[idx],
                }
            )


def write_predictions_with_uncertainty_csv(
    path: Path,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_var: np.ndarray,
    threshold: float,
    questions: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y_pred = (y_prob >= threshold).astype(np.int64)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["idx", "y", "p_correct", "p_correct_var", "risk", "pred", "question"],
        )
        writer.writeheader()
        for idx, (y, p, var, pred) in enumerate(zip(y_true, y_prob, y_var, y_pred)):
            writer.writerow(
                {
                    "idx": idx,
                    "y": int(y),
                    "p_correct": float(p),
                    "p_correct_var": float(var),
                    "risk": float(1.0 - p),
                    "pred": int(pred),
                    "question": "" if questions is None else questions[idx],
                }
            )


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    n_bins: int,
) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int64)
    two_class = len(np.unique(y_true)) >= 2
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)) if two_class else float("nan"),
        "auprc": float(average_precision_score(y_true, y_prob)) if two_class else float("nan"),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": float(ece_score(y_true, y_prob, n_bins=n_bins)),
        "threshold": float(threshold),
    }


@torch.inference_mode()
def predict_probs(
    model: nn.Module,
    data: dict[str, Any],
    device: torch.device,
    batch_size: int,
    feature_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    ds = TensorDataset(data["x"], data["mask"], data["y"], data["lengths"])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    probs = []
    labels = []
    model.eval()
    for x, mask, y, lengths in loader:
        x = x.to(device)
        mask = mask.to(device)
        lengths = lengths.to(device)
        features = expand_features(x, mask, feature_mode)
        logits = model(features, lengths)
        probs.append(torch.sigmoid(logits).detach().cpu())
        labels.append(y.detach().cpu())
    return torch.cat(labels).numpy(), torch.cat(probs).numpy()


@torch.inference_mode()
def predict_fusion_probs(
    model: nn.Module,
    data: dict[str, Any],
    device: torch.device,
    batch_size: int,
    feature_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    ds = TensorDataset(data["x"], data["mask"], data["scalar_features"], data["y"], data["lengths"])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    probs = []
    labels = []
    model.eval()
    for x, mask, scalar, y, lengths in loader:
        x = x.to(device)
        mask = mask.to(device)
        scalar = scalar.to(device)
        lengths = lengths.to(device)
        features = expand_features(x, mask, feature_mode)
        logits = model(features, lengths, scalar)
        probs.append(torch.sigmoid(logits).detach().cpu())
        labels.append(y.detach().cpu())
    return torch.cat(labels).numpy(), torch.cat(probs).numpy()


@torch.inference_mode()
def predict_bayesian_fusion_probs(
    model: nn.Module,
    data: dict[str, Any],
    device: torch.device,
    batch_size: int,
    feature_mode: str,
    mc_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mc_samples <= 0:
        raise ValueError("mc_samples must be positive")
    ds = TensorDataset(data["x"], data["mask"], data["scalar_features"], data["y"], data["lengths"])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    probs = []
    variances = []
    labels = []
    model.train()
    for x, mask, scalar, y, lengths in loader:
        x = x.to(device)
        mask = mask.to(device)
        scalar = scalar.to(device)
        lengths = lengths.to(device)
        features = expand_features(x, mask, feature_mode)
        mc_probs = []
        for _ in range(mc_samples):
            logits = model(features, lengths, scalar)
            mc_probs.append(torch.sigmoid(logits).detach().cpu())
        stacked = torch.stack(mc_probs, dim=0)
        probs.append(stacked.mean(dim=0))
        variances.append(stacked.var(dim=0, unbiased=False))
        labels.append(y.detach().cpu())
    return torch.cat(labels).numpy(), torch.cat(probs).numpy(), torch.cat(variances).numpy()


def command_train_gru(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    train_data = load_trajectory(args.train_pt)
    dev_data = load_trajectory(args.dev_pt)
    test_data = load_trajectory(args.test_pt) if args.eval_test else None

    requested_feature_mode = args.feature_mode
    if requested_feature_mode == "auto":
        metadata = train_data.get("metadata", {})
        feature_mode = "activation" if metadata.get("precomputed_features") else "activation_dynamics"
    else:
        feature_mode = requested_feature_mode

    input_feature_dim = int(train_data["x"].shape[-1])
    base_k = int(train_data.get("metadata", {}).get("base_feature_dim", input_feature_dim))
    feature_mult = 5 if feature_mode == "activation_dynamics" else 1
    model = GRUCorrectnessPredictor(
        input_dim=input_feature_dim * feature_mult,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    train_ds = TensorDataset(train_data["x"], train_data["mask"], train_data["y"], train_data["lengths"])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_dev_auroc = -math.inf
    best_path = args.out_dir / "best_gru.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0
        for x, mask, y, lengths in tqdm(train_loader, desc=f"epoch-{epoch}", leave=False):
            x = x.to(device)
            mask = mask.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            features = expand_features(x, mask, feature_mode)
            logits = model(features, lengths)
            loss = loss_fn(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss.item()) * int(y.numel())
            total_seen += int(y.numel())

        dev_y, dev_prob = predict_probs(model, dev_data, device, args.batch_size, feature_mode)
        dev_threshold = youden_threshold(dev_y, dev_prob)
        dev_metrics = compute_metrics(dev_y, dev_prob, threshold=dev_threshold, n_bins=args.ece_bins)
        train_loss = total_loss / max(1, total_seen)
        row = {"epoch": epoch, "train_loss": train_loss, **{f"dev_{k}": v for k, v in dev_metrics.items()}}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        if dev_metrics["auroc"] > best_dev_auroc:
            best_dev_auroc = dev_metrics["auroc"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": {key: value for key, value in vars(args).items() if key != "func"},
                    "epoch": epoch,
                    "dev_metrics": dev_metrics,
                    "top_k": base_k,
                    "input_dim": input_feature_dim * feature_mult,
                    "feature_mode": feature_mode,
                    "requested_feature_mode": requested_feature_mode,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    dev_y, dev_prob = predict_probs(model, dev_data, device, args.batch_size, feature_mode)
    final_threshold = youden_threshold(dev_y, dev_prob)
    final = {
        "best_epoch": int(checkpoint["epoch"]),
        "selection_metric": "dev_auroc",
        "model_output": "P(correct); risk = 1 - P(correct)",
        "feature_mode": feature_mode,
        "requested_feature_mode": requested_feature_mode,
        "top_k": base_k,
        "input_dim": input_feature_dim * feature_mult,
        "dev": compute_metrics(dev_y, dev_prob, threshold=final_threshold, n_bins=args.ece_bins),
    }
    write_predictions_csv(
        args.out_dir / "dev_predictions.csv",
        dev_y,
        dev_prob,
        final_threshold,
        dev_data.get("questions"),
        dev_data.get("ids"),
    )
    if test_data is not None:
        test_y, test_prob = predict_probs(model, test_data, device, args.batch_size, feature_mode)
        final["test"] = compute_metrics(test_y, test_prob, threshold=final_threshold, n_bins=args.ece_bins)
        write_predictions_csv(
            args.out_dir / "test_predictions.csv",
            test_y,
            test_prob,
            final_threshold,
            test_data.get("questions"),
            test_data.get("ids"),
        )

    write_json(args.out_dir / "training_history.json", history)
    write_json(args.out_dir / "metrics.json", final)
    print(json.dumps(final, indent=2, ensure_ascii=False))


def normalize_scalar_features(
    train_data: dict[str, Any],
    *other_data: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    train_scalar = train_data["scalar_features"].float()
    mean = train_scalar.mean(dim=0)
    std = train_scalar.std(dim=0, unbiased=False).clamp_min(1e-6)
    train_data["scalar_features"] = (train_scalar - mean) / std
    for data in other_data:
        data["scalar_features"] = (data["scalar_features"].float() - mean) / std
    return mean, std


def command_train_gru_fusion(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    train_data = load_trajectory(args.train_pt)
    dev_data = load_trajectory(args.dev_pt)
    if "scalar_features" not in train_data or "scalar_features" not in dev_data:
        raise ValueError("train-gru-fusion requires scalar_features in both train and dev .pt files")

    requested_feature_mode = args.feature_mode
    if requested_feature_mode == "auto":
        metadata = train_data.get("metadata", {})
        feature_mode = "activation" if metadata.get("precomputed_features") else "activation_dynamics"
    else:
        feature_mode = requested_feature_mode

    scalar_mean, scalar_std = normalize_scalar_features(train_data, dev_data)
    input_feature_dim = int(train_data["x"].shape[-1])
    scalar_dim = int(train_data["scalar_features"].shape[-1])
    base_k = int(train_data.get("metadata", {}).get("base_feature_dim", input_feature_dim))
    feature_mult = 5 if feature_mode == "activation_dynamics" else 1
    model = GRUFusionCorrectnessPredictor(
        input_dim=input_feature_dim * feature_mult,
        scalar_dim=scalar_dim,
        hidden_size=args.hidden_size,
        scalar_hidden_size=args.scalar_hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    train_ds = TensorDataset(
        train_data["x"],
        train_data["mask"],
        train_data["scalar_features"],
        train_data["y"],
        train_data["lengths"],
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_dev_auroc = -math.inf
    best_path = args.out_dir / "best_gru_fusion.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0
        for x, mask, scalar, y, lengths in tqdm(train_loader, desc=f"fusion-epoch-{epoch}", leave=False):
            x = x.to(device)
            mask = mask.to(device)
            scalar = scalar.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            features = expand_features(x, mask, feature_mode)
            logits = model(features, lengths, scalar)
            loss = loss_fn(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss.item()) * int(y.numel())
            total_seen += int(y.numel())

        dev_y, dev_prob = predict_fusion_probs(model, dev_data, device, args.batch_size, feature_mode)
        dev_threshold = youden_threshold(dev_y, dev_prob)
        dev_metrics = compute_metrics(dev_y, dev_prob, threshold=dev_threshold, n_bins=args.ece_bins)
        train_loss = total_loss / max(1, total_seen)
        row = {"epoch": epoch, "train_loss": train_loss, **{f"dev_{k}": v for k, v in dev_metrics.items()}}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        if dev_metrics["auroc"] > best_dev_auroc:
            best_dev_auroc = dev_metrics["auroc"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "dev_metrics": dev_metrics,
                    "top_k": base_k,
                    "trajectory_input_dim": input_feature_dim * feature_mult,
                    "scalar_dim": scalar_dim,
                    "scalar_feature_names": train_data.get("metadata", {}).get("scalar_feature_names"),
                    "scalar_mean": scalar_mean,
                    "scalar_std": scalar_std,
                    "feature_mode": feature_mode,
                    "requested_feature_mode": requested_feature_mode,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    dev_y, dev_prob = predict_fusion_probs(model, dev_data, device, args.batch_size, feature_mode)
    final_threshold = youden_threshold(dev_y, dev_prob)
    final = {
        "best_epoch": int(checkpoint["epoch"]),
        "selection_metric": "dev_auroc",
        "model_output": "P(correct); risk = 1 - P(correct)",
        "feature_mode": feature_mode,
        "requested_feature_mode": requested_feature_mode,
        "top_k": base_k,
        "trajectory_input_dim": input_feature_dim * feature_mult,
        "scalar_dim": scalar_dim,
        "scalar_feature_names": train_data.get("metadata", {}).get("scalar_feature_names"),
        "dev": compute_metrics(dev_y, dev_prob, threshold=final_threshold, n_bins=args.ece_bins),
    }
    write_predictions_csv(
        args.out_dir / "dev_predictions.csv",
        dev_y,
        dev_prob,
        final_threshold,
        dev_data.get("questions"),
    )
    write_json(args.out_dir / "training_history.json", history)
    write_json(args.out_dir / "metrics.json", final)
    print(json.dumps(final, indent=2, ensure_ascii=False))


def command_train_bayesian_lstm_fusion(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    train_data = load_trajectory(args.train_pt)
    dev_data = load_trajectory(args.dev_pt)
    if "scalar_features" not in train_data or "scalar_features" not in dev_data:
        raise ValueError("train-bayesian-lstm-fusion requires scalar_features in both train and dev .pt files")

    requested_feature_mode = args.feature_mode
    if requested_feature_mode == "auto":
        metadata = train_data.get("metadata", {})
        feature_mode = "activation" if metadata.get("precomputed_features") else "activation_dynamics"
    else:
        feature_mode = requested_feature_mode

    scalar_mean, scalar_std = normalize_scalar_features(train_data, dev_data)
    input_feature_dim = int(train_data["x"].shape[-1])
    scalar_dim = int(train_data["scalar_features"].shape[-1])
    base_k = int(train_data.get("metadata", {}).get("base_feature_dim", input_feature_dim))
    feature_mult = 5 if feature_mode == "activation_dynamics" else 1
    model = LSTMFusionCorrectnessPredictor(
        input_dim=input_feature_dim * feature_mult,
        scalar_dim=scalar_dim,
        hidden_size=args.hidden_size,
        scalar_hidden_size=args.scalar_hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    train_ds = TensorDataset(
        train_data["x"],
        train_data["mask"],
        train_data["scalar_features"],
        train_data["y"],
        train_data["lengths"],
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_dev_auroc = -math.inf
    best_path = args.out_dir / "best_bayesian_lstm_fusion.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0
        for x, mask, scalar, y, lengths in tqdm(train_loader, desc=f"bayes-lstm-epoch-{epoch}", leave=False):
            x = x.to(device)
            mask = mask.to(device)
            scalar = scalar.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            features = expand_features(x, mask, feature_mode)
            logits = model(features, lengths, scalar)
            loss = loss_fn(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss.item()) * int(y.numel())
            total_seen += int(y.numel())

        dev_y, dev_prob, dev_var = predict_bayesian_fusion_probs(
            model, dev_data, device, args.batch_size, feature_mode, args.mc_samples
        )
        dev_threshold = youden_threshold(dev_y, dev_prob)
        dev_metrics = compute_metrics(dev_y, dev_prob, threshold=dev_threshold, n_bins=args.ece_bins)
        correct_uncertainty = float(dev_var[dev_y.astype(bool)].mean()) if np.any(dev_y == 1) else float("nan")
        wrong_uncertainty = float(dev_var[~dev_y.astype(bool)].mean()) if np.any(dev_y == 0) else float("nan")
        train_loss = total_loss / max(1, total_seen)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"dev_{k}": v for k, v in dev_metrics.items()},
            "dev_mean_uncertainty_correct": correct_uncertainty,
            "dev_mean_uncertainty_wrong": wrong_uncertainty,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        if dev_metrics["auroc"] > best_dev_auroc:
            best_dev_auroc = dev_metrics["auroc"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "dev_metrics": dev_metrics,
                    "dev_uncertainty": {
                        "mean_correct": correct_uncertainty,
                        "mean_wrong": wrong_uncertainty,
                    },
                    "top_k": base_k,
                    "trajectory_input_dim": input_feature_dim * feature_mult,
                    "scalar_dim": scalar_dim,
                    "scalar_feature_names": train_data.get("metadata", {}).get("scalar_feature_names"),
                    "scalar_mean": scalar_mean,
                    "scalar_std": scalar_std,
                    "feature_mode": feature_mode,
                    "requested_feature_mode": requested_feature_mode,
                    "mc_samples": args.mc_samples,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    dev_y, dev_prob, dev_var = predict_bayesian_fusion_probs(
        model, dev_data, device, args.batch_size, feature_mode, args.mc_samples
    )
    final_threshold = youden_threshold(dev_y, dev_prob)
    dev_metrics = compute_metrics(dev_y, dev_prob, threshold=final_threshold, n_bins=args.ece_bins)
    correct_uncertainty = float(dev_var[dev_y.astype(bool)].mean()) if np.any(dev_y == 1) else float("nan")
    wrong_uncertainty = float(dev_var[~dev_y.astype(bool)].mean()) if np.any(dev_y == 0) else float("nan")
    final = {
        "best_epoch": int(checkpoint["epoch"]),
        "selection_metric": "dev_auroc",
        "model_output": "P(correct); risk = 1 - P(correct); uncertainty = MC dropout probability variance",
        "feature_mode": feature_mode,
        "requested_feature_mode": requested_feature_mode,
        "top_k": base_k,
        "trajectory_input_dim": input_feature_dim * feature_mult,
        "scalar_dim": scalar_dim,
        "scalar_feature_names": train_data.get("metadata", {}).get("scalar_feature_names"),
        "mc_samples": args.mc_samples,
        "dev": dev_metrics,
        "dev_uncertainty": {
            "mean_correct": correct_uncertainty,
            "mean_wrong": wrong_uncertainty,
        },
    }
    write_predictions_with_uncertainty_csv(
        args.out_dir / "dev_predictions.csv",
        dev_y,
        dev_prob,
        dev_var,
        final_threshold,
        dev_data.get("questions"),
    )
    write_json(args.out_dir / "training_history.json", history)
    write_json(args.out_dir / "metrics.json", final)
    print(json.dumps(final, indent=2, ensure_ascii=False))


def command_eval_gru(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = checkpoint.get("args", {})
    feature_mode = checkpoint.get("feature_mode") or ckpt_args.get("feature_mode", "activation_dynamics")
    if feature_mode == "auto":
        feature_mode = "activation_dynamics"

    test_data = load_trajectory(args.test_pt)
    input_feature_dim = int(test_data["x"].shape[-1])
    feature_mult = 5 if feature_mode == "activation_dynamics" else 1
    hidden_size = int(ckpt_args.get("hidden_size", args.hidden_size))
    num_layers = int(ckpt_args.get("num_layers", args.num_layers))
    dropout = float(ckpt_args.get("dropout", args.dropout))
    bidirectional = bool(ckpt_args.get("bidirectional", args.bidirectional))

    model = GRUCorrectnessPredictor(
        input_dim=input_feature_dim * feature_mult,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_y, test_prob = predict_probs(model, test_data, device, args.batch_size, feature_mode)
    threshold = args.threshold
    threshold_source = "cli"
    if threshold is None:
        threshold = checkpoint.get("dev_metrics", {}).get("threshold")
        threshold_source = "checkpoint_dev"
    if threshold is None:
        threshold = 0.5
        threshold_source = "default_0.5"
    threshold = float(threshold)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    test_metrics = compute_metrics(test_y, test_prob, threshold=threshold, n_bins=args.ece_bins)
    write_predictions_csv(
        args.out_dir / "test_predictions.csv",
        test_y,
        test_prob,
        threshold,
        test_data.get("questions"),
    )
    final = {
        "checkpoint": str(args.checkpoint),
        "test_pt": str(args.test_pt),
        "model_output": "P(correct); risk = 1 - P(correct)",
        "feature_mode": feature_mode,
        "input_dim": input_feature_dim * feature_mult,
        "threshold_source": threshold_source,
        "test": test_metrics,
    }
    write_json(args.out_dir / "metrics.json", final)
    print(json.dumps(final, indent=2, ensure_ascii=False))


def read_prediction_csv(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"idx", "y", "p_correct"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        for row in reader:
            idx = int(row["idx"])
            if idx in rows:
                raise ValueError(f"{path} has duplicate idx={idx}")
            rows[idx] = row
    return rows


def command_ensemble_dev(args: argparse.Namespace) -> None:
    if len(args.prediction_files) < 2:
        raise ValueError("ensemble-dev requires at least two prediction files")

    tables = [read_prediction_csv(path) for path in args.prediction_files]
    reference_ids = sorted(tables[0])
    reference_set = set(reference_ids)
    for path, table in zip(args.prediction_files[1:], tables[1:]):
        if set(table) != reference_set:
            missing = sorted(reference_set - set(table))[:10]
            extra = sorted(set(table) - reference_set)[:10]
            raise ValueError(f"{path} ids do not align; missing={missing}, extra={extra}")

    y_true = []
    probs = []
    questions = []
    sample_ids = []
    for idx in reference_ids:
        y_values = [int(table[idx]["y"]) for table in tables]
        if len(set(y_values)) != 1:
            raise ValueError(f"Mismatched labels for idx={idx}: {y_values}")
        y_true.append(y_values[0])
        probs.append([float(table[idx]["p_correct"]) for table in tables])
        questions.append(tables[0][idx].get("question", ""))
        sample_ids.append(tables[0][idx].get("sample_id", ""))

    y_arr = np.asarray(y_true, dtype=np.int64)
    prob_arr = np.asarray(probs, dtype=np.float64).mean(axis=1)
    threshold = youden_threshold(y_arr, prob_arr)
    metrics = compute_metrics(y_arr, prob_arr, threshold=threshold, n_bins=args.ece_bins)

    write_predictions_csv(args.out_csv, y_arr, prob_arr, threshold, questions, sample_ids)
    final = {
        "prediction_files": [str(path) for path in args.prediction_files],
        "num_models": len(args.prediction_files),
        "num_examples": int(len(y_arr)),
        "aggregation": "mean_p_correct",
        "threshold_source": "dev_youden",
        "dev": metrics,
        **metrics,
    }
    write_json(args.out_json, final)
    print(json.dumps(final, indent=2, ensure_ascii=False))


def predict_checkpoint_probs(
    checkpoint_path: Path,
    data: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_args = checkpoint.get("args", {})
    feature_mode = checkpoint.get("feature_mode") or ckpt_args.get("feature_mode", "activation_dynamics")
    if feature_mode == "auto":
        feature_mode = "activation_dynamics"

    input_feature_dim = int(data["x"].shape[-1])
    feature_mult = 5 if feature_mode == "activation_dynamics" else 1
    hidden_size = int(ckpt_args.get("hidden_size", 128))
    num_layers = int(ckpt_args.get("num_layers", 1))
    dropout = float(ckpt_args.get("dropout", 0.2))
    bidirectional = bool(ckpt_args.get("bidirectional", False))

    model = GRUCorrectnessPredictor(
        input_dim=input_feature_dim * feature_mult,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    y, prob = predict_probs(model, data, device, batch_size, feature_mode)
    info = {
        "checkpoint": str(checkpoint_path),
        "epoch": checkpoint.get("epoch"),
        "feature_mode": feature_mode,
        "input_dim": input_feature_dim * feature_mult,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "bidirectional": bidirectional,
        "dev_metrics": checkpoint.get("dev_metrics"),
    }
    return y, prob, info


def resolve_checkpoint_path(path: Path) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        candidates = [
            path / "best_bayesian_lstm_fusion.pt",
            path / "best_gru_fusion.pt",
            path / "best_gru.pt",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"Could not find checkpoint file at {path}")


def predict_fusion_checkpoint_probs(
    checkpoint_path: Path,
    data: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_args = checkpoint.get("args", {})
    feature_mode = checkpoint.get("feature_mode") or ckpt_args.get("feature_mode", "activation_dynamics")
    if feature_mode == "auto":
        feature_mode = "activation_dynamics"

    if "scalar_features" not in data:
        raise ValueError("Fusion evaluation requires scalar_features in the test .pt file")
    if "scalar_mean" not in checkpoint or "scalar_std" not in checkpoint:
        raise ValueError(f"{checkpoint_path} does not contain saved scalar_mean/scalar_std")

    eval_data = dict(data)
    scalar_mean = checkpoint["scalar_mean"].detach().float().cpu()
    scalar_std = checkpoint["scalar_std"].detach().float().cpu().clamp_min(1e-6)
    eval_data["scalar_features"] = (data["scalar_features"].float() - scalar_mean) / scalar_std

    input_feature_dim = int(eval_data["x"].shape[-1])
    feature_mult = 5 if feature_mode == "activation_dynamics" else 1
    trajectory_input_dim = int(checkpoint.get("trajectory_input_dim", input_feature_dim * feature_mult))
    scalar_dim = int(checkpoint.get("scalar_dim", eval_data["scalar_features"].shape[-1]))
    hidden_size = int(ckpt_args.get("hidden_size", 256))
    scalar_hidden_size = int(ckpt_args.get("scalar_hidden_size", 64))
    num_layers = int(ckpt_args.get("num_layers", 1))
    dropout = float(ckpt_args.get("dropout", 0.2))
    bidirectional = bool(ckpt_args.get("bidirectional", False))

    model = GRUFusionCorrectnessPredictor(
        input_dim=trajectory_input_dim,
        scalar_dim=scalar_dim,
        hidden_size=hidden_size,
        scalar_hidden_size=scalar_hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    y, prob = predict_fusion_probs(model, eval_data, device, batch_size, feature_mode)
    info = {
        "checkpoint": str(checkpoint_path),
        "epoch": checkpoint.get("epoch"),
        "feature_mode": feature_mode,
        "trajectory_input_dim": trajectory_input_dim,
        "scalar_dim": scalar_dim,
        "scalar_feature_names": checkpoint.get("scalar_feature_names"),
        "hidden_size": hidden_size,
        "scalar_hidden_size": scalar_hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "bidirectional": bidirectional,
        "dev_metrics": checkpoint.get("dev_metrics"),
        "scalar_normalization_source": "checkpoint_train_split",
    }
    return y, prob, info


def predict_bayesian_lstm_checkpoint_probs(
    checkpoint_path: Path,
    data: dict[str, Any],
    device: torch.device,
    batch_size: int,
    mc_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_args = checkpoint.get("args", {})
    feature_mode = checkpoint.get("feature_mode") or ckpt_args.get("feature_mode", "activation_dynamics")
    if feature_mode == "auto":
        feature_mode = "activation_dynamics"

    if "scalar_features" not in data:
        raise ValueError("Bayesian LSTM-fusion evaluation requires scalar_features in the test .pt file")
    if "scalar_mean" not in checkpoint or "scalar_std" not in checkpoint:
        raise ValueError(f"{checkpoint_path} does not contain saved scalar_mean/scalar_std")

    eval_data = dict(data)
    scalar_mean = checkpoint["scalar_mean"].detach().float().cpu()
    scalar_std = checkpoint["scalar_std"].detach().float().cpu().clamp_min(1e-6)
    eval_data["scalar_features"] = (data["scalar_features"].float() - scalar_mean) / scalar_std

    input_feature_dim = int(eval_data["x"].shape[-1])
    feature_mult = 5 if feature_mode == "activation_dynamics" else 1
    trajectory_input_dim = int(checkpoint.get("trajectory_input_dim", input_feature_dim * feature_mult))
    scalar_dim = int(checkpoint.get("scalar_dim", eval_data["scalar_features"].shape[-1]))
    hidden_size = int(ckpt_args.get("hidden_size", 256))
    scalar_hidden_size = int(ckpt_args.get("scalar_hidden_size", 64))
    num_layers = int(ckpt_args.get("num_layers", 1))
    dropout = float(ckpt_args.get("dropout", 0.3))
    bidirectional = bool(ckpt_args.get("bidirectional", False))

    model = LSTMFusionCorrectnessPredictor(
        input_dim=trajectory_input_dim,
        scalar_dim=scalar_dim,
        hidden_size=hidden_size,
        scalar_hidden_size=scalar_hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    y, prob, var = predict_bayesian_fusion_probs(
        model,
        eval_data,
        device,
        batch_size,
        feature_mode,
        mc_samples,
    )
    info = {
        "checkpoint": str(checkpoint_path),
        "epoch": checkpoint.get("epoch"),
        "feature_mode": feature_mode,
        "trajectory_input_dim": trajectory_input_dim,
        "scalar_dim": scalar_dim,
        "scalar_feature_names": checkpoint.get("scalar_feature_names"),
        "hidden_size": hidden_size,
        "scalar_hidden_size": scalar_hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "bidirectional": bidirectional,
        "dev_metrics": checkpoint.get("dev_metrics"),
        "dev_uncertainty": checkpoint.get("dev_uncertainty"),
        "scalar_normalization_source": "checkpoint_train_split",
    }
    return y, prob, var, info


def command_eval_bayesian_lstm_fusion(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    test_data = load_trajectory(args.test_pt)
    y, prob, var, checkpoint_info = predict_bayesian_lstm_checkpoint_probs(
        args.checkpoint,
        test_data,
        device,
        args.batch_size,
        args.mc_samples,
    )
    threshold = float(args.threshold)
    metrics = compute_metrics(y, prob, threshold=threshold, n_bins=args.ece_bins)
    correct_uncertainty = float(var[y.astype(bool)].mean()) if np.any(y == 1) else float("nan")
    wrong_uncertainty = float(var[~y.astype(bool)].mean()) if np.any(y == 0) else float("nan")

    write_predictions_with_uncertainty_csv(
        args.out_csv,
        y,
        prob,
        var,
        threshold,
        test_data.get("questions"),
    )
    final = {
        "test_pt": str(args.test_pt),
        "checkpoint": checkpoint_info,
        "num_examples": int(len(y)),
        "model_output": "P(correct) = MC dropout mean probability; risk = 1 - P(correct); uncertainty = probability variance",
        "threshold_source": "frozen_dev_threshold",
        "scalar_normalization": "checkpoint train split mean/std",
        "mc_samples": int(args.mc_samples),
        "test": metrics,
        "test_uncertainty": {
            "mean_correct": correct_uncertainty,
            "mean_wrong": wrong_uncertainty,
        },
    }
    write_json(args.out_json, final)
    print(json.dumps(final, indent=2, ensure_ascii=False))


def command_ensemble_test(args: argparse.Namespace) -> None:
    if len(args.checkpoints) < 2:
        raise ValueError("ensemble-test requires at least two checkpoints")

    device = resolve_device(args.device)
    test_data = load_trajectory(args.test_pt)
    y_ref = None
    probs = []
    checkpoint_info = []
    for checkpoint_path in args.checkpoints:
        y, prob, info = predict_checkpoint_probs(checkpoint_path, test_data, device, args.batch_size)
        if y_ref is None:
            y_ref = y
        elif not np.array_equal(y_ref, y):
            raise ValueError(f"Labels from {checkpoint_path} do not match previous checkpoint predictions")
        probs.append(prob)
        checkpoint_info.append(info)

    assert y_ref is not None
    prob_arr = np.stack(probs, axis=0).mean(axis=0)
    threshold = float(args.threshold)
    metrics = compute_metrics(y_ref, prob_arr, threshold=threshold, n_bins=args.ece_bins)

    write_predictions_csv(
        args.out_csv,
        y_ref,
        prob_arr,
        threshold,
        test_data.get("questions"),
        test_data.get("ids"),
    )
    final = {
        "test_pt": str(args.test_pt),
        "checkpoints": checkpoint_info,
        "num_models": len(args.checkpoints),
        "num_examples": int(len(y_ref)),
        "aggregation": "mean_p_correct",
        "model_output": "P(correct); risk = 1 - P(correct)",
        "threshold_source": "frozen_dev_ensemble_threshold",
        "test": metrics,
    }
    write_json(args.out_json, final)
    print(json.dumps(final, indent=2, ensure_ascii=False))


def command_ensemble_test_fusion(args: argparse.Namespace) -> None:
    if len(args.checkpoints) < 2:
        raise ValueError("ensemble-test-fusion requires at least two checkpoints")

    device = resolve_device(args.device)
    test_data = load_trajectory(args.test_pt)
    y_ref = None
    probs = []
    checkpoint_info = []
    for checkpoint_path in args.checkpoints:
        y, prob, info = predict_fusion_checkpoint_probs(checkpoint_path, test_data, device, args.batch_size)
        if y_ref is None:
            y_ref = y
        elif not np.array_equal(y_ref, y):
            raise ValueError(f"Labels from {checkpoint_path} do not match previous checkpoint predictions")
        probs.append(prob)
        checkpoint_info.append(info)

    assert y_ref is not None
    prob_arr = np.stack(probs, axis=0).mean(axis=0)
    threshold = float(args.threshold)
    metrics = compute_metrics(y_ref, prob_arr, threshold=threshold, n_bins=args.ece_bins)

    write_predictions_csv(args.out_csv, y_ref, prob_arr, threshold, test_data.get("questions"))
    final = {
        "test_pt": str(args.test_pt),
        "checkpoints": checkpoint_info,
        "num_models": len(args.checkpoints),
        "num_examples": int(len(y_ref)),
        "aggregation": "mean_p_correct",
        "model_output": "P(correct); risk = 1 - P(correct)",
        "threshold_source": "frozen_dev_fusion_ensemble_threshold",
        "scalar_normalization": "per-checkpoint train split mean/std",
        "test": metrics,
    }
    write_json(args.out_json, final)
    print(json.dumps(final, indent=2, ensure_ascii=False))


def command_ensemble_eval_fusion(args: argparse.Namespace) -> None:
    if len(args.checkpoints) < 2:
        raise ValueError("ensemble-eval-fusion requires at least two checkpoints")

    device = resolve_device(args.device)
    eval_data = load_trajectory(args.eval_pt)
    y_ref = None
    probs = []
    checkpoint_info = []
    for checkpoint_path in args.checkpoints:
        y, prob, info = predict_fusion_checkpoint_probs(checkpoint_path, eval_data, device, args.batch_size)
        if y_ref is None:
            y_ref = y
        elif not np.array_equal(y_ref, y):
            raise ValueError(f"Labels from {checkpoint_path} do not match previous checkpoint predictions")
        probs.append(prob)
        checkpoint_info.append(info)

    assert y_ref is not None
    prob_arr = np.stack(probs, axis=0).mean(axis=0)
    threshold = float(args.threshold)
    metrics = compute_metrics(y_ref, prob_arr, threshold=threshold, n_bins=args.ece_bins)

    write_predictions_csv(args.out_csv, y_ref, prob_arr, threshold, eval_data.get("questions"))
    final = {
        "eval_pt": str(args.eval_pt),
        "checkpoints": checkpoint_info,
        "num_models": len(args.checkpoints),
        "num_examples": int(len(y_ref)),
        "aggregation": "mean_p_correct",
        "model_output": "P(correct); risk = 1 - P(correct)",
        "threshold_source": "frozen_external_dev_threshold",
        "scalar_normalization": "per-checkpoint train split mean/std",
        "eval": metrics,
    }
    write_json(args.out_json, final)
    print(json.dumps(final, indent=2, ensure_ascii=False))


def command_eval_gru_fusion(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    eval_data = load_trajectory(args.eval_pt)
    y, prob, checkpoint_info = predict_fusion_checkpoint_probs(
        args.checkpoint,
        eval_data,
        device,
        args.batch_size,
    )
    threshold = args.threshold
    threshold_source = "cli"
    if threshold is None:
        dev_metrics = checkpoint_info.get("dev_metrics") or {}
        threshold = dev_metrics.get("threshold")
        threshold_source = "checkpoint_dev"
    if threshold is None:
        threshold = 0.5
        threshold_source = "default_0.5"
    threshold = float(threshold)
    metrics = compute_metrics(y, prob, threshold=threshold, n_bins=args.ece_bins)

    write_predictions_csv(args.out_csv, y, prob, threshold, eval_data.get("questions"))
    final = {
        "eval_pt": str(args.eval_pt),
        "checkpoint": checkpoint_info,
        "num_examples": int(len(y)),
        "model_output": "P(correct); risk = 1 - P(correct)",
        "threshold_source": threshold_source,
        "scalar_normalization": "checkpoint train split mean/std",
        args.metric_key: metrics,
    }
    write_json(args.out_json, final)
    print(json.dumps(final, indent=2, ensure_ascii=False))


def load_prediction_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = read_prediction_csv(path)
    ids = sorted(rows)
    y_true = np.asarray([int(rows[idx]["y"]) for idx in ids], dtype=np.int64)
    y_prob = np.asarray([float(rows[idx]["p_correct"]) for idx in ids], dtype=np.float64)
    return y_true, y_prob


def rankdata_average_ties(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if x.shape[0] < 2:
        return float("nan")
    x_rank = rankdata_average_ties(x)
    y_rank = rankdata_average_ties(y)
    if float(x_rank.std()) == 0.0 or float(y_rank.std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def load_uncertainty_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    label_columns = ("y", "y_true", "has_answer")
    uncertainty_columns = ("p_correct_var", "uncertainty", "variance", "p_var")
    ids = []
    labels = []
    uncertainty = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        label_col = next((name for name in label_columns if name in fieldnames), None)
        uncertainty_col = next((name for name in uncertainty_columns if name in fieldnames), None)
        if label_col is None:
            raise ValueError(f"{path} is missing a label column; expected one of {label_columns}")
        if uncertainty_col is None:
            raise ValueError(f"{path} is missing an uncertainty column; expected one of {uncertainty_columns}")
        for row_idx, row in enumerate(reader):
            ids.append(int(row["idx"]) if "idx" in fieldnames else row_idx)
            labels.append(int(float(row[label_col])))
            uncertainty.append(float(row[uncertainty_col]))
    order = np.argsort(np.asarray(ids, dtype=np.int64), kind="mergesort")
    return (
        np.asarray(labels, dtype=np.int64)[order],
        np.asarray(uncertainty, dtype=np.float64)[order],
        np.asarray(ids, dtype=np.int64)[order],
    )


def summarize_uncertainty_group(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "std": float("nan")}
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std(ddof=0)),
    }


def command_analyze_uncertainty(args: argparse.Namespace) -> None:
    y_true, uncertainty, _ids = load_uncertainty_arrays(args.predictions_csv)
    if y_true.size == 0:
        raise ValueError(f"No predictions found in {args.predictions_csv}")

    label_error = 1 - y_true
    two_class_error = len(np.unique(label_error)) >= 2
    uncertainty_error_auroc = (
        float(roc_auc_score(label_error, uncertainty)) if two_class_error else float("nan")
    )
    uncertainty_error_auprc = (
        float(average_precision_score(label_error, uncertainty)) if two_class_error else float("nan")
    )
    spearman = spearman_correlation(uncertainty, y_true.astype(np.float64))

    correct_values = uncertainty[y_true == 1]
    wrong_values = uncertainty[y_true == 0]
    correct_summary = summarize_uncertainty_group(correct_values)
    wrong_summary = summarize_uncertainty_group(wrong_values)

    sorted_idx = np.argsort(uncertainty, kind="mergesort")
    bin_indices = np.array_split(sorted_idx, args.n_bins)
    bin_rows = []
    for bin_idx, idx in enumerate(bin_indices):
        if idx.size == 0:
            continue
        u = uncertainty[idx]
        y = y_true[idx]
        errors = 1 - y
        bin_rows.append(
            {
                "bin": int(bin_idx),
                "n": int(idx.size),
                "uncertainty_min": float(u.min()),
                "uncertainty_max": float(u.max()),
                "uncertainty_mean": float(u.mean()),
                "uncertainty_median": float(np.median(u)),
                "error_rate": float(errors.mean()),
                "correct_rate": float(y.mean()),
            }
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "bin",
                "n",
                "uncertainty_min",
                "uncertainty_max",
                "uncertainty_mean",
                "uncertainty_median",
                "error_rate",
                "correct_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(bin_rows)

    final = {
        "predictions_csv": str(args.predictions_csv),
        "num_examples": int(y_true.size),
        "num_correct": int((y_true == 1).sum()),
        "num_wrong": int((y_true == 0).sum()),
        "uncertainty_error_auroc": uncertainty_error_auroc,
        "uncertainty_error_auprc": uncertainty_error_auprc,
        "spearman_uncertainty_correctness": spearman,
        "correct_uncertainty": correct_summary,
        "wrong_uncertainty": wrong_summary,
        "bins": bin_rows,
    }
    write_json(args.out_json, final)
    print(json.dumps(final, indent=2, ensure_ascii=False))


def command_bootstrap_test(args: argparse.Namespace) -> None:
    y_true, y_prob = load_prediction_arrays(args.predictions_csv)
    n = int(y_true.shape[0])
    if n == 0:
        raise ValueError(f"No predictions found in {args.predictions_csv}")

    rng = np.random.default_rng(args.seed)
    metric_names = ["auroc", "auprc", "accuracy", "f1", "brier", "ece"]
    values: dict[str, list[float]] = {name: [] for name in metric_names}
    skipped = 0
    for _ in tqdm(range(args.n_bootstrap), desc="bootstrap-test"):
        idx = rng.integers(0, n, size=n)
        y_sample = y_true[idx]
        p_sample = y_prob[idx]
        metrics = compute_metrics(y_sample, p_sample, threshold=args.threshold, n_bins=args.ece_bins)
        if not np.isfinite(metrics["auroc"]) or not np.isfinite(metrics["auprc"]):
            skipped += 1
            continue
        for name in metric_names:
            values[name].append(float(metrics[name]))

    summary: dict[str, Any] = {
        "predictions_csv": str(args.predictions_csv),
        "threshold": float(args.threshold),
        "n_examples": n,
        "n_bootstrap_requested": int(args.n_bootstrap),
        "n_bootstrap_used": int(len(values["auroc"])),
        "n_bootstrap_skipped_single_class": int(skipped),
        "seed": int(args.seed),
        "ece_bins": int(args.ece_bins),
        "metrics": {},
    }
    rows = []
    for name in metric_names:
        arr = np.asarray(values[name], dtype=np.float64)
        if arr.size == 0:
            stats = {"mean": float("nan"), "ci_2_5": float("nan"), "ci_97_5": float("nan")}
        else:
            stats = {
                "mean": float(arr.mean()),
                "ci_2_5": float(np.percentile(arr, 2.5)),
                "ci_97_5": float(np.percentile(arr, 97.5)),
            }
        summary["metrics"][name] = stats
        rows.append({"metric": name, **stats})

    write_json(args.out_json, summary)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "mean", "ci_2_5", "ci_97_5"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def labels_csv_for(args: argparse.Namespace) -> Path:
    if args.labels_csv is not None:
        return args.labels_csv
    if args.model_key is None:
        raise ValueError("--model-key is required when --labels-csv is not provided")
    if args.dataset not in LABEL_PROMPT_VERSION:
        raise ValueError(f"Cannot infer prompt version for dataset={args.dataset}; pass --labels-csv")
    if args.label_root is None:
        raise ValueError("Set LABEL_ROOT or pass --label-root/--labels-csv")
    return args.label_root / args.model_key / args.dataset / LABEL_PROMPT_VERSION[args.dataset] / "main_labels.csv"


def split_csv_rows_stratified(
    rows: list[dict[str, str]],
    label_key: str,
    train_frac: float,
    dev_frac: float,
    seed: int,
) -> dict[str, list[dict[str, str]]]:
    groups: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[int(float(row[label_key]))].append(row)
    rng = random.Random(seed)
    splits = {"train": [], "dev": [], "test": []}
    for label in sorted(groups):
        group = list(groups[label])
        rng.shuffle(group)
        n = len(group)
        n_train = int(round(n * train_frac))
        n_dev = int(round(n * dev_frac))
        n_train = min(n_train, n)
        n_dev = min(n_dev, n - n_train)
        splits["train"].extend(group[:n_train])
        splits["dev"].extend(group[n_train : n_train + n_dev])
        splits["test"].extend(group[n_train + n_dev :])
    for values in splits.values():
        rng.shuffle(values)
    return splits


def label_csv_row_to_json(row: dict[str, str], split: str, label_key: str) -> dict[str, Any]:
    label = int(float(row[label_key]))
    row_id = row.get("row_id", "")
    return {
        "id": f"{row.get('model', '')}::{row.get('dataset', '')}::{row.get('prompt_version', '')}::{row.get('subset', '')}::{row_id}",
        "model": row.get("model", ""),
        "dataset": row.get("dataset", ""),
        "prompt_version": row.get("prompt_version", ""),
        "subset": row.get("subset", ""),
        "row_id": int(float(row_id)) if str(row_id).strip() else None,
        "split": split,
        "qa_prompt": row.get("prompt", ""),
        "question": row.get("prompt", ""),
        "reference": row.get("ground_truth", ""),
        "answer": row.get("answer", ""),
        "idk_response": str(row.get("idk_response", "")).lower() in {"true", "1", "yes"},
        "correct": label,
        "label": label,
        "has_answer": label,
    }


def command_make_source_selection_split(args: argparse.Namespace) -> None:
    if DEFAULT_MULTIMODEL_SPLIT_BASE is None:
        raise ValueError("Set DATA_ROOT before using make-source-selection-split")
    split_root = DEFAULT_MULTIMODEL_SPLIT_BASE / args.model_key / args.dataset
    source_path = split_root / f"{args.source_split}.jsonl"
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    rows = read_jsonl(source_path)
    if args.selection_size <= 0:
        raise ValueError("--selection-size must be positive")
    if args.selection_size >= len(rows):
        raise ValueError(
            f"--selection-size must leave predictor-training rows; got {args.selection_size} of {len(rows)}"
        )

    all_ids = [required_row_id(row, f"{args.model_key}/{args.dataset}/{args.source_split}") for row in rows]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError(f"{source_path} contains duplicate sample IDs")
    if any(args.stratify_label not in row for row in rows):
        raise ValueError(f"{source_path} contains rows without label {args.stratify_label!r}")

    selected_rows = stratified_sample_rows(rows, args.selection_size, args.stratify_label, args.seed)
    selected_ids = [required_row_id(row, "source neuron selection") for row in selected_rows]
    selected_set = set(selected_ids)
    remaining_rows = [row for row, sample_id in zip(rows, all_ids) if sample_id not in selected_set]
    remaining_ids = [required_row_id(row, "remaining predictor train") for row in remaining_rows]
    overlap = selected_set.intersection(remaining_ids)
    if overlap:
        raise ValueError(f"Selection and remaining IDs overlap: {sorted(overlap)[:10]}")
    if len(selected_ids) + len(remaining_ids) != len(rows):
        raise ValueError("Selection split does not partition the original source split")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    selection_path = args.out_dir / f"selection_{args.selection_size}_ids.json"
    remaining_path = args.out_dir / "train_remaining_ids.json"
    split_map_path = args.out_dir / "split_map.json"
    write_json(selection_path, selected_ids)
    write_json(remaining_path, remaining_ids)
    split_map = {"train": None, "dev": None, "test": None}
    split_map[args.source_split] = str(remaining_path.resolve())
    write_json(split_map_path, split_map)

    def counts(values: list[dict[str, Any]]) -> dict[str, int]:
        positive = sum(int(row[args.stratify_label]) for row in values)
        return {"positive": positive, "negative": len(values) - positive}

    summary = {
        "model_key": args.model_key,
        "dataset": args.dataset,
        "source_split": args.source_split,
        "source_path": str(source_path),
        "seed": args.seed,
        "stratify_label": args.stratify_label,
        "original_train_size": len(rows),
        "selection_size": len(selected_ids),
        "remaining_train_size": len(remaining_ids),
        "selection_counts": counts(selected_rows),
        "remaining_counts": counts(remaining_rows),
        "overlap_count": len(overlap),
        "selection_ids_json": str(selection_path.resolve()),
        "train_remaining_ids_json": str(remaining_path.resolve()),
        "split_map_json": str(split_map_path.resolve()),
    }
    write_json(args.out_dir / "split_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def command_make_splits(args: argparse.Namespace) -> None:
    total = args.train_frac + args.dev_frac + args.test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1.0, got {total}")

    labels_csv = labels_csv_for(args)
    if not labels_csv.exists():
        raise FileNotFoundError(labels_csv)

    existing = [args.out_dir / f"{split}.jsonl" for split in ("train", "dev", "test") if (args.out_dir / f"{split}.jsonl").exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing split files. "
            f"Existing files: {[str(path) for path in existing]}. "
            "Pass --overwrite only if you intentionally want to replace them."
        )

    with labels_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    if not rows:
        raise ValueError(f"No rows found in {labels_csv}")
    if args.source_label_key not in rows[0]:
        raise ValueError(f"{labels_csv} does not contain label column {args.source_label_key!r}")

    splits = split_csv_rows_stratified(
        rows,
        label_key=args.source_label_key,
        train_frac=args.train_frac,
        dev_frac=args.dev_frac,
        seed=args.seed,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for split, split_rows in splits.items():
        out_path = args.out_dir / f"{split}.jsonl"
        json_rows = [label_csv_row_to_json(row, split, args.source_label_key) for row in split_rows]
        with out_path.open("w", encoding="utf-8") as f:
            for row in json_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n = len(json_rows)
        pos = sum(int(row["has_answer"]) for row in json_rows)
        summary.append(
            {
                "split": split,
                "path": str(out_path),
                "num_examples": n,
                "positive_ratio": pos / n if n else 0.0,
            }
        )
        print(f"wrote {out_path} n={n} positive_ratio={pos / n if n else 0.0:.4f}")

    write_json(
        args.out_dir / "split_summary.json",
        {
            "labels_csv": str(labels_csv),
            "model_key": args.model_key,
            "dataset": args.dataset,
            "seed": args.seed,
            "fractions": {"train": args.train_frac, "dev": args.dev_frac, "test": args.test_frac},
            "splits": summary,
        },
    )


def add_common_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default="triviaqa")
    parser.add_argument("--model-key", choices=sorted(MODEL_KEY_TO_PATH), default=None)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--min-steps", type=int, default=1)
    parser.add_argument("--prompt-key", default="qa_prompt")
    parser.add_argument("--label-key", default="has_answer")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)


def add_selection_sampling_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--selection-max-samples",
        type=int,
        default=None,
        help="Stratified maximum number of train examples used only for neuron selection.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "make-source-selection-split",
        help="Partition a labelled source split into a stratified neuron-selection ID set and remaining train IDs.",
    )
    p.add_argument("--model-key", choices=sorted(MODEL_KEY_TO_PATH), required=True)
    p.add_argument("--dataset", choices=SUPPORTED_DATASETS, required=True)
    p.add_argument("--source-split", choices=["train", "dev", "test"], default="train")
    p.add_argument("--selection-size", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stratify-label", default="has_answer")
    p.add_argument("--out-dir", type=Path, required=True)
    p.set_defaults(func=command_make_source_selection_split)

    p = sub.add_parser("make-splits", help="Create stratified train/dev/test JSONL splits from correctness label CSV without overwriting by default.")
    p.add_argument("--model-key", choices=sorted(MODEL_KEY_TO_PATH), default=None)
    p.add_argument("--dataset", choices=SUPPORTED_DATASETS, required=True)
    p.add_argument("--labels-csv", type=Path, default=None)
    p.add_argument("--label-root", type=Path, default=DEFAULT_MULTIMODEL_LABEL_ROOT)
    p.add_argument("--source-label-key", default="label")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--dev-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=command_make_splits)

    p = sub.add_parser("select-neurons", help="Select Top-K MLP intermediate neurons using train split only.")
    add_common_generation_args(p)
    add_selection_sampling_args(p)
    p.add_argument("--train-path", type=Path, default=None)
    p.add_argument("--ids-json", type=Path, default=None)
    p.add_argument("--top-k", type=int, default=64)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=0.2)
    p.add_argument("--out-json", type=Path, default=Path("outputs/triviaqa_neurons/topk_mlp_neurons.json"))
    p.set_defaults(func=command_select_neurons)

    p = sub.add_parser(
        "select-neurons-prefill-aware",
        help="Select Top-K neurons using train-only prompt/prefill and generation statistics.",
    )
    add_common_generation_args(p)
    add_selection_sampling_args(p)
    p.add_argument("--train-path", type=Path, default=None)
    p.add_argument("--top-k", type=int, default=128)
    p.add_argument("--w-prompt-last", type=float, default=1.0)
    p.add_argument("--w-prompt-mean", type=float, default=0.8)
    p.add_argument("--w-prompt-max", type=float, default=0.5)
    p.add_argument("--w-generation-mean", type=float, default=1.0)
    p.add_argument("--w-generation-delta", type=float, default=0.5)
    p.add_argument("--w-generation-variance", type=float, default=0.2)
    p.add_argument(
        "--out-json",
        type=Path,
        default=Path("outputs/triviaqa_neurons/topk_mlp_neurons_k128_prefillaware.json"),
    )
    p.set_defaults(func=command_select_neurons_prefill_aware)

    p = sub.add_parser(
        "select-neurons-prefill-only",
        help="Select Top-K MLP neurons using prompt/prefill pooled statistics only; no generation.",
    )
    add_common_generation_args(p)
    add_selection_sampling_args(p)
    p.add_argument("--train-path", type=Path, default=None)
    p.add_argument("--top-k", type=int, default=128)
    p.add_argument("--w-prompt-last", type=float, default=1.0)
    p.add_argument("--w-prompt-mean", type=float, default=0.8)
    p.add_argument("--w-prompt-max", type=float, default=0.5)
    p.add_argument(
        "--out-json",
        type=Path,
        default=Path("outputs/triviaqa_neurons/topk_mlp_neurons_k128_prefillonly.json"),
    )
    p.set_defaults(func=command_select_neurons_prefill_only)

    p = sub.add_parser(
        "select-neurons-prefill-rich",
        help="Select Top-K MLP neurons using rich prompt/prefill pooled statistics only; no generation.",
    )
    add_common_generation_args(p)
    add_selection_sampling_args(p)
    p.add_argument("--train-path", type=Path, default=None)
    p.add_argument("--top-k", type=int, default=256)
    p.add_argument("--w-prompt-last", type=float, default=1.0)
    p.add_argument("--w-prompt-mean", type=float, default=1.0)
    p.add_argument("--w-prompt-max", type=float, default=1.0)
    p.add_argument("--w-prompt-std", type=float, default=1.0)
    p.add_argument("--w-last-8-mean", type=float, default=1.0)
    p.add_argument("--w-last-8-max", type=float, default=1.0)
    p.add_argument("--w-last-16-mean", type=float, default=1.0)
    p.add_argument("--w-last-16-max", type=float, default=1.0)
    p.add_argument(
        "--out-json",
        type=Path,
        default=Path("outputs/triviaqa_neurons/topk_mlp_neurons_k256_prefillrich.json"),
    )
    p.set_defaults(func=command_select_neurons_prefill_rich)

    p = sub.add_parser(
        "select-neurons-prefill-lastseq",
        help="Select Top-K MLP neurons for last prompt-token sequence trajectories; no generation.",
    )
    add_common_generation_args(p)
    add_selection_sampling_args(p)
    p.add_argument("--train-path", type=Path, default=None)
    p.add_argument("--top-k", type=int, default=256)
    p.add_argument("--w-prompt-last", type=float, default=1.0)
    p.add_argument("--w-prompt-mean", type=float, default=0.0)
    p.add_argument("--w-prompt-max", type=float, default=0.0)
    p.add_argument("--w-prompt-std", type=float, default=0.5)
    p.add_argument("--w-last-8-mean", type=float, default=1.0)
    p.add_argument("--w-last-8-max", type=float, default=0.5)
    p.add_argument("--w-last-16-mean", type=float, default=1.0)
    p.add_argument("--w-last-16-max", type=float, default=0.5)
    p.add_argument(
        "--out-json",
        type=Path,
        default=Path("outputs/triviaqa_neurons/topk_mlp_neurons_k256_prefilllastseq.json"),
    )
    p.set_defaults(func=command_select_neurons_prefill_lastseq)

    p = sub.add_parser(
        "select-residual-question-only",
        help="Select Top-K residual stream dimensions from question-only prefill activations; no generation.",
    )
    add_common_generation_args(p)
    add_selection_sampling_args(p)
    p.add_argument("--train-path", type=Path, default=None)
    p.add_argument("--question-key", default="question")
    p.add_argument("--top-k", type=int, default=128)
    p.add_argument("--w-prompt-last", type=float, default=1.0)
    p.add_argument("--w-prompt-mean", type=float, default=0.8)
    p.add_argument("--w-prompt-max", type=float, default=0.5)
    p.add_argument(
        "--out-json",
        type=Path,
        default=Path("outputs/triviaqa_residual/topk_residual_dims_k128_questiononly.json"),
    )
    p.set_defaults(func=command_select_residual_question_only)

    p = sub.add_parser("extract-trajectories", help="Extract real-generation trajectories for selected Top-K neurons.")
    add_common_generation_args(p)
    p.add_argument("--neurons-json", type=Path, default=Path("outputs/triviaqa_neurons/topk_mlp_neurons.json"))
    p.add_argument("--train-path", type=Path, default=None)
    p.add_argument("--dev-path", type=Path, default=None)
    p.add_argument("--test-path", type=Path, default=None)
    p.add_argument("--splits", nargs="+", choices=["train", "dev", "test"], default=["train", "dev", "test"])
    p.add_argument("--out-dir", type=Path, default=Path("outputs/triviaqa_trajectory"))
    p.set_defaults(func=command_extract_trajectories)

    p = sub.add_parser(
        "extract-prefill-generation-trajectories",
        help="Extract prompt pooled prefill activations plus real generation trajectories.",
    )
    add_common_generation_args(p)
    p.add_argument("--neurons-json", type=Path, default=Path("outputs/triviaqa_neurons/topk_mlp_neurons.json"))
    p.add_argument("--train-path", type=Path, default=None)
    p.add_argument("--dev-path", type=Path, default=None)
    p.add_argument("--test-path", type=Path, default=None)
    p.add_argument("--splits", nargs="+", choices=["train", "dev", "test"], default=["train", "dev", "test"])
    p.add_argument("--checkpoint-dir", type=Path, default=None)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-samples-per-split", type=int, default=None)
    p.add_argument("--split-id-filter-json", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/triviaqa_trajectory_prefill_gen"))
    p.set_defaults(func=command_extract_prefill_generation_trajectories)

    p = sub.add_parser(
        "extract-prefill-generation-trajectories-with-logits",
        help="Extract prefill+generation trajectories plus low-cost generation logit statistics.",
    )
    add_common_generation_args(p)
    p.add_argument("--neurons-json", type=Path, default=Path("outputs/triviaqa_neurons/topk_mlp_neurons.json"))
    p.add_argument("--train-path", type=Path, default=None)
    p.add_argument("--dev-path", type=Path, default=None)
    p.add_argument("--test-path", type=Path, default=None)
    p.add_argument("--splits", nargs="+", choices=["train", "dev", "test"], default=["train", "dev", "test"])
    p.add_argument("--checkpoint-dir", type=Path, default=None)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-samples-per-split", type=int, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/triviaqa_trajectory_prefill_gen_logits"))
    p.set_defaults(func=command_extract_prefill_generation_trajectories_with_logits)

    p = sub.add_parser(
        "extract-prefill-only-trajectories",
        help="Extract prompt pooled prefill-only trajectories for selected Top-K neurons.",
    )
    add_common_generation_args(p)
    p.add_argument("--neurons-json", type=Path, default=Path("outputs/triviaqa_neurons/topk_mlp_neurons.json"))
    p.add_argument("--train-path", type=Path, default=None)
    p.add_argument("--dev-path", type=Path, default=None)
    p.add_argument("--test-path", type=Path, default=None)
    p.add_argument("--splits", nargs="+", choices=["train", "dev", "test"], default=["train", "dev", "test"])
    p.add_argument("--out-dir", type=Path, default=Path("outputs/triviaqa_trajectory_prefill_only"))
    p.set_defaults(func=command_extract_prefill_only_trajectories)

    p = sub.add_parser(
        "extract-prefill-rich-trajectories",
        help="Extract strict pre-generation rich pooled prompt trajectories for selected Top-K MLP neurons.",
    )
    add_common_generation_args(p)
    p.add_argument("--neurons-json", type=Path, default=Path("outputs/triviaqa_neurons/topk_mlp_neurons_k256_prefillrich.json"))
    p.add_argument("--train-path", type=Path, default=None)
    p.add_argument("--dev-path", type=Path, default=None)
    p.add_argument("--test-path", type=Path, default=None)
    p.add_argument("--splits", nargs="+", choices=["train", "dev", "test"], default=["train", "dev"])
    p.add_argument("--out-dir", type=Path, default=Path("outputs/triviaqa_trajectory_prefill_rich"))
    p.set_defaults(func=command_extract_prefill_rich_trajectories)

    p = sub.add_parser(
        "extract-prefill-lastseq-trajectories",
        help="Extract strict pre-generation last prompt-token MLP trajectories for selected Top-K neurons.",
    )
    add_common_generation_args(p)
    p.add_argument("--neurons-json", type=Path, default=Path("outputs/triviaqa_neurons/topk_mlp_neurons_k256_prefilllastseq.json"))
    p.add_argument("--train-path", type=Path, default=None)
    p.add_argument("--dev-path", type=Path, default=None)
    p.add_argument("--test-path", type=Path, default=None)
    p.add_argument("--last-n", type=int, default=16)
    p.add_argument("--splits", nargs="+", choices=["train", "dev", "test"], default=["train", "dev"])
    p.add_argument("--out-dir", type=Path, default=Path("outputs/triviaqa_trajectory_prefill_lastseq"))
    p.set_defaults(func=command_extract_prefill_lastseq_trajectories)

    p = sub.add_parser(
        "extract-prefill-token-sequence-trajectories",
        help="Extract strict pre-generation last-N prompt-token MLP trajectories with left padding.",
    )
    add_common_generation_args(p)
    p.add_argument("--neurons-json", type=Path, default=Path("outputs/triviaqa_neurons/topk_mlp_neurons_k256_prefillrich.json"))
    p.add_argument("--train-path", type=Path, default=None)
    p.add_argument("--dev-path", type=Path, default=None)
    p.add_argument("--test-path", type=Path, default=None)
    p.add_argument("--last-n-prompt-tokens", type=int, choices=[32, 64], default=32)
    p.add_argument("--splits", nargs="+", choices=["train", "dev", "test"], default=["train", "dev"])
    p.add_argument("--out-dir", type=Path, default=Path("outputs/triviaqa_trajectory_prefill_token_sequence"))
    p.set_defaults(func=command_extract_prefill_token_sequence_trajectories)

    p = sub.add_parser(
        "extract-residual-question-only",
        help="Extract question-only residual stream pooled trajectories for selected Top-K residual dimensions.",
    )
    add_common_generation_args(p)
    p.add_argument("--residual-json", type=Path, default=Path("outputs/triviaqa_residual/topk_residual_dims_k128_questiononly.json"))
    p.add_argument("--train-path", type=Path, default=None)
    p.add_argument("--dev-path", type=Path, default=None)
    p.add_argument("--test-path", type=Path, default=None)
    p.add_argument("--question-key", default="question")
    p.add_argument("--splits", nargs="+", choices=["train", "dev", "test"], default=["train", "dev", "test"])
    p.add_argument("--out-dir", type=Path, default=Path("outputs/triviaqa_residual_questiononly_k128"))
    p.set_defaults(func=command_extract_residual_question_only)

    p = sub.add_parser("augment-trajectories", help="Materialize trajectory feature expansions from saved trajectories.")
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--features",
        nargs="+",
        choices=["activation", "delta", "abs_delta", "running_mean", "running_std"],
        default=["activation", "delta", "abs_delta", "running_mean", "running_std"],
    )
    p.add_argument("--splits", nargs="+", choices=["train", "dev", "test"], default=["train", "dev", "test"])
    p.set_defaults(func=command_augment_trajectories)

    p = sub.add_parser("train-gru", help="Train GRU correctness predictor from saved trajectories.")
    p.add_argument("--train-pt", type=Path, default=Path("outputs/triviaqa_trajectory/train.pt"))
    p.add_argument("--dev-pt", type=Path, default=Path("outputs/triviaqa_trajectory/dev.pt"))
    p.add_argument("--test-pt", type=Path, default=Path("outputs/triviaqa_trajectory/test.pt"))
    p.add_argument("--eval-test", action="store_true")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/triviaqa_gru"))
    p.add_argument("--device", default="auto")
    p.add_argument("--feature-mode", choices=["auto", "activation", "activation_dynamics"], default="auto")
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--bidirectional", action="store_true")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ece-bins", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=command_train_gru)

    p = sub.add_parser("train-gru-fusion", help="Train GRU trajectory plus scalar-feature fusion predictor.")
    p.add_argument("--train-pt", type=Path, required=True)
    p.add_argument("--dev-pt", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--feature-mode", choices=["auto", "activation", "activation_dynamics"], default="auto")
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--scalar-hidden-size", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--bidirectional", action="store_true")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ece-bins", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=command_train_gru_fusion)

    p = sub.add_parser(
        "train-bayesian-lstm-fusion",
        help="Train Bayesian LSTM-fusion predictor with MC dropout dev evaluation.",
    )
    p.add_argument("--train-pt", type=Path, required=True)
    p.add_argument("--dev-pt", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--feature-mode", choices=["auto", "activation", "activation_dynamics"], default="auto")
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--scalar-hidden-size", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--bidirectional", action="store_true")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ece-bins", type=int, default=15)
    p.add_argument("--mc-samples", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=command_train_bayesian_lstm_fusion)

    p = sub.add_parser("eval-gru", help="Evaluate a saved GRU checkpoint without training.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--test-pt", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--bidirectional", action="store_true")
    p.add_argument("--ece-bins", type=int, default=15)
    p.add_argument("--threshold", type=float, default=None)
    p.set_defaults(func=command_eval_gru)

    p = sub.add_parser(
        "eval-bayesian-lstm-fusion",
        help="Evaluate a saved Bayesian LSTM-fusion checkpoint with MC dropout without training.",
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--test-pt", type=Path, required=True)
    p.add_argument("--threshold", type=float, required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--mc-samples", type=int, default=30)
    p.add_argument("--device", default="auto")
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--ece-bins", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=command_eval_bayesian_lstm_fusion)

    p = sub.add_parser("ensemble-dev", help="Average dev prediction probabilities from multiple GRU seeds.")
    p.add_argument("--prediction-files", nargs="+", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--ece-bins", type=int, default=15)
    p.set_defaults(func=command_ensemble_dev)

    p = sub.add_parser("ensemble-test", help="Evaluate a frozen checkpoint ensemble on a saved test trajectory.")
    p.add_argument("--test-pt", type=Path, required=True)
    p.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    p.add_argument("--threshold", type=float, required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="auto")
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--ece-bins", type=int, default=15)
    p.set_defaults(func=command_ensemble_test)

    p = sub.add_parser("ensemble-test-fusion", help="Evaluate a frozen GRU-fusion checkpoint ensemble on test data.")
    p.add_argument("--test-pt", type=Path, required=True)
    p.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    p.add_argument("--threshold", type=float, required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="auto")
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--ece-bins", type=int, default=15)
    p.set_defaults(func=command_ensemble_test_fusion)

    p = sub.add_parser("ensemble-eval-fusion", help="Evaluate a frozen GRU-fusion checkpoint ensemble on any saved trajectory split.")
    p.add_argument("--eval-pt", type=Path, required=True)
    p.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    p.add_argument("--threshold", type=float, required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="auto")
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--ece-bins", type=int, default=15)
    p.set_defaults(func=command_ensemble_eval_fusion)

    p = sub.add_parser("eval-gru-fusion", help="Evaluate one saved GRU-fusion checkpoint on any trajectory split.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--eval-pt", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--metric-key", default="test")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="auto")
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--ece-bins", type=int, default=15)
    p.set_defaults(func=command_eval_gru_fusion)

    p = sub.add_parser("analyze-uncertainty", help="Analyze whether MC-dropout uncertainty detects errors.")
    p.add_argument("--predictions-csv", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--n-bins", type=int, default=10)
    p.set_defaults(func=command_analyze_uncertainty)

    p = sub.add_parser("bootstrap-test", help="Bootstrap confidence intervals from a frozen test predictions CSV.")
    p.add_argument("--predictions-csv", type=Path, required=True)
    p.add_argument("--threshold", type=float, required=True)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--ece-bins", type=int, default=15)
    p.set_defaults(func=command_bootstrap_test)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    apply_model_key(args)
    args.func(args)


if __name__ == "__main__":
    main()
