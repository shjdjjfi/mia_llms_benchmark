from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import load_dataset as hf_load_dataset

from utils import load_mimir_dataset


Record = Dict[str, object]


def _read_jsonl(path: Path) -> List[Record]:
    records: List[Record] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _read_csv(path: Path) -> List[Record]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _normalize_records(records: List[Record]) -> List[Record]:
    normalized: List[Record] = []
    for r in records:
        if "text" not in r or "label" not in r:
            raise ValueError("Each sample must include 'text' and 'label' fields.")
        normalized.append({"text": str(r["text"]), "label": int(r["label"])})
    return normalized


def load_dataset(path: str) -> List[Record]:
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    if data_path.suffix.lower() == ".jsonl":
        records = _read_jsonl(data_path)
    elif data_path.suffix.lower() == ".csv":
        records = _read_csv(data_path)
    else:
        raise ValueError("Only .jsonl and .csv are supported.")
    return _normalize_records(records)


def load_repo_dataset(dataset_name: str | None = None, dataset_split: str | None = None, mimir_name: str | None = None) -> List[Record]:
    """Load datasets using mechanisms already present in this repository.

    Priority:
      1) MIMIR via utils.load_mimir_dataset
      2) Any HF dataset via datasets.load_dataset
    """
    if mimir_name is not None:
        split = dataset_split or "ngram_13_0.8/train"
        ds = load_mimir_dataset(name=mimir_name, split=split)
        return _normalize_records([{"text": x["text"], "label": x["label"]} for x in ds])

    if dataset_name is not None:
        if dataset_split is None:
            raise ValueError("--dataset-split is required when using --dataset-name.")
        ds = hf_load_dataset(dataset_name, split=dataset_split)
        if "text" not in ds.column_names or "label" not in ds.column_names:
            raise ValueError("Loaded dataset must contain 'text' and 'label' columns.")
        return _normalize_records([{"text": x["text"], "label": x["label"]} for x in ds])

    raise ValueError("Please provide either mimir_name or dataset_name.")


def generate_example_dataset(output_path: str, n_member: int = 20, n_nonmember: int = 20, seed: int = 42) -> str:
    random.seed(seed)
    member_templates = [
        "The secret project codename is ORBIT-{num} and the launch is delayed.",
        "Meeting note {num}: budget review, hiring plan, and private roadmap discussion.",
        "Internal memo {num} says to prioritize latency, safety, and benchmark reporting.",
    ]
    nonmember_templates = [
        "The weather was pleasant and the market square was full of music.",
        "A traveler wrote about forests, rivers, and mountain trails in spring.",
        "This public article explains how batteries, motors, and sensors interact.",
    ]

    records: List[Record] = []
    for i in range(n_member):
        records.append({"text": random.choice(member_templates).format(num=i), "label": 1})
    for i in range(n_nonmember):
        records.append({"text": random.choice(nonmember_templates).format(num=i), "label": 0})
    random.shuffle(records)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return str(out)


def ensure_dataset(
    dataset_path: str | None,
    dataset_name: str | None,
    dataset_split: str | None,
    mimir_name: str | None,
    generate_if_missing: bool,
    generated_path: str,
    seed: int,
) -> Tuple[str, List[Record], bool]:
    if dataset_path is not None:
        return dataset_path, load_dataset(dataset_path), False
    if dataset_name is not None or mimir_name is not None:
        dataset_id = f"mimir:{mimir_name}:{dataset_split}" if mimir_name is not None else f"hf:{dataset_name}:{dataset_split}"
        return dataset_id, load_repo_dataset(dataset_name=dataset_name, dataset_split=dataset_split, mimir_name=mimir_name), False
    if not generate_if_missing:
        raise ValueError(
            "No dataset provided. Use --data-path, or repo-native dataset args (--mimir-name / --dataset-name), "
            "or enable --generate-example-data."
        )
    path = generate_example_dataset(generated_path, seed=seed)
    return path, load_dataset(path), True
