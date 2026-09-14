#!/usr/bin/env python3
"""Prepare deterministic JSONL train/dev/test splits from multi-model correctness CSV labels."""

from __future__ import annotations

import argparse
import json
import random
import os
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_LABEL_ROOT = Path(os.environ["LABEL_ROOT"]) if os.environ.get("LABEL_ROOT") else None
DEFAULT_OUT_ROOT = Path(os.environ["DATA_ROOT"]) if os.environ.get("DATA_ROOT") else None
DEFAULT_MODELS = (
    "qwen_2.5_7b_instruct",
    "ministral_8b_instruct",
    "llama3.1_8b_chat",
    "mistral_7b_instruct",
)
DEFAULT_DATASETS = (
    "trivia_qa_2_60k",
    "notable_people",
    "cities_10k",
    "math_operations_6k",
    "medals_9k",
    "gsm8k",
)
PROMPT_VERSION = {
    "gsm8k": "base_3_shot",
    "trivia_qa_2_60k": "base",
    "notable_people": "base",
    "cities_10k": "base",
    "math_operations_6k": "base",
    "medals_9k": "base",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-root", type=Path, default=DEFAULT_LABEL_ROOT, required=DEFAULT_LABEL_ROOT is None)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT, required=DEFAULT_OUT_ROOT is None)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--dev-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def split_indices_by_label(df: pd.DataFrame, train_frac: float, dev_frac: float, seed: int) -> dict[str, list[int]]:
    splits = {"train": [], "dev": [], "test": []}
    for label_value in sorted(df["label"].unique()):
        idxs = df.index[df["label"] == label_value].tolist()
        rng = random.Random(seed + int(label_value) * 1009)
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = int(round(n * train_frac))
        n_dev = int(round(n * dev_frac))
        n_train = min(n_train, n)
        n_dev = min(n_dev, n - n_train)
        splits["train"].extend(idxs[:n_train])
        splits["dev"].extend(idxs[n_train : n_train + n_dev])
        splits["test"].extend(idxs[n_train + n_dev :])

    rng = random.Random(seed + 7919)
    for values in splits.values():
        rng.shuffle(values)
    return splits


def row_to_json(row: pd.Series, split: str) -> dict[str, Any]:
    row_id = int(row["row_id"])
    label = int(row["label"])
    return {
        "id": f"{row['model']}::{row['dataset']}::{row['prompt_version']}::{row['subset']}::{row_id}",
        "model": str(row["model"]),
        "dataset": str(row["dataset"]),
        "prompt_version": str(row["prompt_version"]),
        "subset": str(row["subset"]),
        "row_id": row_id,
        "split": split,
        "qa_prompt": str(row["prompt"]),
        "question": str(row["prompt"]),
        "reference": "" if pd.isna(row.get("ground_truth", "")) else str(row.get("ground_truth", "")),
        "answer": "" if pd.isna(row.get("answer", "")) else str(row.get("answer", "")),
        "idk_response": bool(row.get("idk_response", False)),
        "correct": label,
        "label": label,
        "has_answer": label,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    total_frac = args.train_frac + args.dev_frac + args.test_frac
    if abs(total_frac - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1.0, got {total_frac}")

    summary = []
    for model in args.models:
        for dataset in args.datasets:
            prompt_version = PROMPT_VERSION[dataset]
            csv_path = args.label_root / model / dataset / prompt_version / "main_labels.csv"
            if not csv_path.exists():
                raise FileNotFoundError(csv_path)

            df = pd.read_csv(csv_path)
            required = {"model", "dataset", "prompt_version", "subset", "row_id", "prompt", "label"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")

            splits = split_indices_by_label(df, args.train_frac, args.dev_frac, args.seed)
            out_dir = args.out_root / model / dataset
            split_rows = {}
            for split, idxs in splits.items():
                rows = [row_to_json(df.loc[idx], split=split) for idx in idxs]
                split_rows[split] = rows
                write_jsonl(out_dir / f"{split}.jsonl", rows)

            for split, rows in split_rows.items():
                n = len(rows)
                pos = sum(int(row["has_answer"]) for row in rows)
                summary.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "prompt_version": prompt_version,
                        "split": split,
                        "path": str(out_dir / f"{split}.jsonl"),
                        "num_examples": n,
                        "positive_ratio": pos / n if n else 0.0,
                    }
                )
                print(
                    f"{model}/{dataset}/{split}: n={n} "
                    f"positive_ratio={pos / n if n else 0.0:.4f} path={out_dir / f'{split}.jsonl'}"
                )

    args.out_root.mkdir(parents=True, exist_ok=True)
    summary_dir = args.out_root / "数据集汇总"
    summary_dir.mkdir(parents=True, exist_ok=True)
    with (summary_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")
    pd.DataFrame(summary).to_csv(summary_dir / "summary.csv", index=False)
    print(f"wrote {summary_dir / 'summary.json'}")
    print(f"wrote {summary_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
