#!/usr/bin/env python3
"""Validate the public v3 package without loading models or experiment tensors."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = (
    "qwen_2.5_7b_instruct",
    "ministral_8b_instruct",
    "mistral_7b_instruct",
    "llama3.1_8b_chat",
)
METRICS = ("auroc", "auprc", "accuracy", "f1", "brier", "ece")
EXPECTED_DIRECT = (0.7590, 0.7177, 0.7390, 0.5296, 0.1640, 0.1285)
EXPECTED_ADAPTED = (0.9014, 0.8665, 0.8417, 0.7945, 0.0886, 0.0339)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: str) -> list[dict[str, str]]:
    with (ROOT / path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    checks: dict[str, object] = {}
    registry = {item["model"]: item for item in json.loads((ROOT / "selected_nodes/top128_registry.json").read_text())}
    node_checks = {}
    for model in MODELS:
        path = ROOT / "selected_nodes" / model / "topk_mlp_neurons_k128_trivia10k.json"
        nodes = json.loads(path.read_text(encoding="utf-8"))
        pairs = [(int(item["layer"]), int(item["neuron"])) for item in nodes]
        digest = sha256(path)
        node_checks[model] = {
            "count": len(pairs),
            "unique_count": len(set(pairs)),
            "sha256": digest,
            "registry_match": registry[model]["sha256"] == digest,
            "status": registry[model]["status"],
        }
    checks["top128"] = node_checks
    assert all(v["count"] == v["unique_count"] == 128 and v["registry_match"] for v in node_checks.values())

    shell_checks = {}
    for name in ("run_direct_transfer.sh", "run_fixed_node_adaptation.sh"):
        text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        shell_checks[name] = {
            "raw_extractor": "extract-prefill-generation-trajectories" in text,
            "feature_mode_activation": "--feature-mode activation" in text,
            "no_with_logits": "extract-prefill-generation-trajectories-with-logits" not in text,
            "no_fusion_command": not re.search(r"train-gru-fusion|ensemble-test-fusion", text),
        }
    checks["formal_shells"] = shell_checks
    assert all(all(v.values()) for v in shell_checks.values())
    fixed_text = (ROOT / "scripts/run_fixed_node_adaptation.sh").read_text(encoding="utf-8")
    assert "TARGETS=(notable_people cities_10k math_operations_6k medals_9k gsm8k)" in fixed_text

    direct = [r for r in rows("results/main/direct_transfer.csv") if r["dataset"] != "trivia_qa_2_60k"]
    adapted = [r for r in rows("results/main/fixed_node_adaptation.csv") if r["dataset"] != "trivia_qa_2_60k"]
    assert len(direct) == len(adapted) == 20
    direct_macro = tuple(sum(float(r[m]) for r in direct) / 20 for m in METRICS)
    adapted_macro = tuple(sum(float(r[f"test_{m}"]) for r in adapted) / 20 for m in METRICS)
    assert tuple(round(v, 4) for v in direct_macro) == EXPECTED_DIRECT
    assert tuple(round(v, 4) for v in adapted_macro) == EXPECTED_ADAPTED
    direct_by_key = {(r["model"], r["dataset"]): r for r in direct}
    improved = sum(float(r["test_accuracy"]) > float(direct_by_key[(r["model"], r["dataset"])]["accuracy"]) for r in adapted)
    assert improved == 18
    gsm_direct = sum(float(r["auroc"]) for r in direct if r["dataset"] == "gsm8k") / 4
    gsm_adapted = sum(float(r["test_auroc"]) for r in adapted if r["dataset"] == "gsm8k") / 4
    assert abs(gsm_direct - 0.5807497272112144) < 1e-12
    assert abs(gsm_adapted - 0.8123471901081779) < 1e-12
    checks["main_results"] = {
        "non_trivia_rows_per_setting": 20,
        "direct_macro": dict(zip(METRICS, direct_macro)),
        "adapted_macro": dict(zip(METRICS, adapted_macro)),
        "accuracy_improved": f"{improved}/20",
        "gsm8k_auroc_macro_direct": gsm_direct,
        "gsm8k_auroc_macro_adapted": gsm_adapted,
        "gsm8k_auroc_delta": gsm_adapted - gsm_direct,
    }

    ablation = rows("results/ablation/ablation_results.csv")
    expected = {
        ("trivia_qa_2_60k", "full"), ("trivia_qa_2_60k", "prefill_only"),
        ("trivia_qa_2_60k", "meanpool_mlp"), ("gsm8k", "full"),
        ("gsm8k", "prefill_only"), ("gsm8k", "meanpool_mlp"),
        ("gsm8k", "random_128"),
    }
    assert {(r["dataset"], r["variant"]) for r in ablation} == expected
    assert all(r["source_file"] and len(r["source_sha256"]) == 64 for r in ablation)
    checks["ablation"] = {"configuration_count": len(ablation), "gsm8k_random_128": True}

    required_refs = (
        "scripts/run_direct_transfer.sh", "scripts/run_fixed_node_adaptation.sh",
        "scripts/run_ablation.sh", "scripts/benchmark_gru_inference.py",
        "configs/ablation_final.json", "results/ablation/ablation_results.csv",
        "results/main", "results/cost", "docs/REPRODUCIBILITY.md",
    )
    missing_refs = [ref for ref in required_refs if not (ROOT / ref).exists()]
    assert not missing_refs
    checks["readme_paths"] = {"checked": list(required_refs), "missing": missing_refs}

    forbidden_files = []
    private_hits = []
    credential_hits = []
    binary_suffixes = {".pt", ".pth", ".ckpt", ".bin", ".safetensors"}
    private_names = ("ro" + "ot", "auto" + "dl-tmp", "m" + "nt", "work" + "space")
    private_pattern = re.compile(r"/(?:" + "|".join(private_names) + r")(?:/|\b)")
    credential_names = (
        "api" + "_key", "hf" + "_token", "wandb" + "_api",
        "openai" + "_api", "PRIVATE" + " KEY",
        "pass" + r"word\s*[=:]", "sec" + r"ret\s*[=:]",
    )
    credential_pattern = re.compile(r"(?i)(" + "|".join(credential_names) + r")")
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        if path.suffix.lower() in binary_suffixes or path.name == ".env" or "__pycache__" in path.parts:
            forbidden_files.append(relative)
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if relative == "scripts/validate_release_v3.py":
            continue
        if private_pattern.search(text):
            private_hits.append(relative)
        if credential_pattern.search(text):
            credential_hits.append(relative)
    assert not forbidden_files and not private_hits and not credential_hits
    checks["public_cleanup"] = {
        "forbidden_files": forbidden_files,
        "private_absolute_paths": private_hits,
        "credential_patterns": credential_hits,
    }

    output = ROOT / "V3_VALIDATION.json"
    output.write_text(json.dumps({"status": "PASS", "checks": checks}, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
