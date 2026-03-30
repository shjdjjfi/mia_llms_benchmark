from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import ensure_dataset
from .features import FeatureConfig, FeatureExtractor
from .model import TrainConfig, train_and_evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EM-MIAs style ensemble membership inference pipeline")
    parser.add_argument("--data-path", type=str, default=None, help="Path to .jsonl/.csv with columns: text,label")
    parser.add_argument("--generate-example-data", action="store_true", help="Generate a tiny synthetic dataset when no data-path is provided")
    parser.add_argument("--generated-data-path", type=str, default="data/em_mias_example.jsonl")

    parser.add_argument("--target-model", type=str, required=True, help="HuggingFace target model")
    parser.add_argument("--reference-model", type=str, required=True, help="HuggingFace reference model")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--min-k-ratio", type=float, default=0.2, help="Fraction for Min-k%% feature")

    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=str, default="outputs/em_mias_metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path, records, generated = ensure_dataset(
        dataset_path=args.data_path,
        generate_if_missing=args.generate_example_data,
        generated_path=args.generated_data_path,
        seed=args.seed,
    )

    extractor = FeatureExtractor(
        target_model_name=args.target_model,
        reference_model_name=args.reference_model,
        config=FeatureConfig(
            min_k_ratio=args.min_k_ratio,
            max_length=args.max_length,
            device=args.device,
        ),
    )

    features = extractor.extract_batch([r["text"] for r in records])
    labels = [int(r["label"]) for r in records]

    metrics = train_and_evaluate(
        feature_dicts=features,
        labels=labels,
        config=TrainConfig(test_size=args.test_size, random_state=args.seed),
    )

    payload = {
        "dataset_path": dataset_path,
        "generated_example_data": generated,
        "target_model": args.target_model,
        "reference_model": args.reference_model,
        "n_samples": len(records),
        "assumptions": [
            "Min-k% uses the highest-loss tokens (equivalent to lowest-probability tokens).",
            "Reference feature is target_avg_loss - reference_avg_loss.",
            "XGBoost classifier threshold is fixed at 0.5 for class prediction.",
        ],
        "metrics": metrics,
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
