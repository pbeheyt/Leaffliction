import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from src.classification.features import load_dataset


def _build_random_forest(params, random_state):
    return RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )


def _candidate_rf_params():
    return [
        {
            "n_estimators": 220,
            "max_depth": 26,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
        },
        {
            "n_estimators": 320,
            "max_depth": 32,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
        },
        {
            "n_estimators": 260,
            "max_depth": 28,
            "min_samples_leaf": 2,
            "max_features": "log2",
        },
        {
            "n_estimators": 420,
            "max_depth": 36,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
        },
        {
            "n_estimators": 420,
            "max_depth": None,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
        },
        {
            "n_estimators": 300,
            "max_depth": 30,
            "min_samples_leaf": 3,
            "max_features": "sqrt",
        },
        {
            "n_estimators": 360,
            "max_depth": 30,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
        },
    ]


def _build_report_text(report_dict):
    labels = [
        label
        for label in report_dict
        if label not in {"accuracy", "macro avg", "weighted avg"}
    ]
    label_width = max([len(label) for label in labels] + [len("accuracy")])
    header = f"{'':<{label_width}}  {'precision':>9}  {'recall':>7}"
    lines = [header]

    for label in labels:
        metrics = report_dict[label]
        precision = float(metrics.get("precision", 0.0))
        recall = float(metrics.get("recall", 0.0))
        lines.append(
            f"{label:<{label_width}}  {precision:>9.4f}  {recall:>7.4f}"
        )

    accuracy = report_dict.get("accuracy")
    if accuracy is not None:
        lines.append("")
        lines.append(f"{'accuracy':<{label_width}}  {float(accuracy):>9.4f}")

    return "\n".join(lines)


def _trim_report_dict(report_dict):
    trimmed = {}
    for label, metrics in report_dict.items():
        if label in {"macro avg", "weighted avg"}:
            continue
        if label == "accuracy":
            trimmed[label] = float(metrics)
            continue
        trimmed[label] = {
            "precision": float(metrics.get("precision", 0.0)),
            "recall": float(metrics.get("recall", 0.0)),
        }
    return trimmed


def _evaluate_model(model, x_train, y_train, x_val, y_val):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    accuracy = float(accuracy_score(y_val, y_pred))
    report_dict = classification_report(
        y_val,
        y_pred,
        output_dict=True,
        zero_division=0,
    )
    trimmed_report = _trim_report_dict(report_dict)
    report_text = _build_report_text(trimmed_report)

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_val)
        confidence = float(probabilities.max(axis=1).mean())

    return {
        "model": model,
        "accuracy": accuracy,
        "report_dict": trimmed_report,
        "report_text": report_text,
        "mean_confidence": confidence,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train classifier with stratified split, validation metrics, and artifact export."
    )
    parser.add_argument("directory", help="Path to the dataset directory")
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Validation ratio for stratified split (default: 0.2)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for deterministic split (default: 42)",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="models",
        help="Directory where trained artifacts are saved (default: models).",
    )
    parser.add_argument(
        "--report-dir",
        default="results/classification",
        help="Directory where validation reports are saved (default: results/classification).",
    )
    parser.add_argument(
        "--min-val-samples",
        type=int,
        default=100,
        help="Minimum recommended validation samples for evaluation evidence (default: 100).",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(
            f"Error: {args.directory} data/augmented is not a valid directory.")
        return

    if not 0.0 < args.test_size < 1.0:
        print("Error: --test-size must be between 0 and 1.")
        return

    print(f"[train] loading dataset from: {args.directory}", flush=True)
    x_data, y_data = load_dataset(args.directory)
    print("[train] dataset loaded, computing stratified split", flush=True)

    x_train, x_val, y_train, y_val = train_test_split(
        x_data,
        y_data,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y_data,
    )

    print("[train] split completed", flush=True)

    train_counts = Counter(y_train)
    val_counts = Counter(y_val)

    best_result = None
    best_name = "random_forest"
    best_params = None
    candidates = _candidate_rf_params()
    print(
        f"[train] evaluating random_forest variants: {len(candidates)}",
        flush=True,
    )
    for params in candidates:
        model = _build_random_forest(params, args.random_state)
        result = _evaluate_model(model, x_train, y_train, x_val, y_val)
        print(
            "[train] random_forest params="
            f"{params} accuracy={result['accuracy']:.4f}",
            flush=True,
        )
        if best_result is None or result["accuracy"] > best_result["accuracy"]:
            best_result = result
            best_params = params

    os.makedirs(args.artifacts_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)

    artifact_payload = {
        "model": best_result["model"],
        "model_name": best_name,
        "model_params": best_params,
        "classes": sorted(set(y_data.tolist())),
        "feature_size": int(x_data.shape[1]),
        "train_size": int(len(y_train)),
        "validation_size": int(len(y_val)),
        "test_size": float(args.test_size),
        "random_state": int(args.random_state),
    }

    artifact_path = os.path.join(args.artifacts_dir, "leaf_model.joblib")
    joblib.dump(artifact_payload, artifact_path, compress=3)

    report_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_directory": args.directory,
        "model_name": best_name,
        "model_params": best_params,
        "overall_accuracy": best_result["accuracy"],
        "mean_confidence": best_result["mean_confidence"],
        "validation_samples": len(y_val),
        "train_samples": len(y_train),
        "feature_vector_size": int(x_data.shape[1]),
        "train_distribution": dict(sorted(train_counts.items())),
        "validation_distribution": dict(sorted(val_counts.items())),
        "classification_report": best_result["report_dict"],
        "recommended_min_validation_samples": args.min_val_samples,
        "meets_min_validation_samples": len(y_val) >= args.min_val_samples,
        "meets_accuracy_90": best_result["accuracy"] >= 0.90,
    }

    json_report_path = os.path.join(args.report_dir, "validation_report.json")
    text_report_path = os.path.join(args.report_dir, "validation_report.txt")

    with open(json_report_path, "w", encoding="utf-8") as json_file:
        json.dump(report_payload, json_file, indent=2)
    with open(text_report_path, "w", encoding="utf-8") as text_file:
        text_file.write(best_result["report_text"])

    print("Dataset ingestion completed.", flush=True)
    print(f"Total samples: {len(y_data)}")
    print(f"Feature vector size: {x_data.shape[1]}")
    print(f"Train samples: {len(y_train)}")
    print(f"Validation samples: {len(y_val)}")
    print("Train distribution:")
    for label in sorted(train_counts):
        print(f"  - {label}: {train_counts[label]}")
    print("Validation distribution:")
    for label in sorted(val_counts):
        print(f"  - {label}: {val_counts[label]}")
    print(f"Selected model: {best_name}")
    print(f"Selected params: {best_params}")
    print(f"Validation accuracy: {best_result['accuracy']:.4f}")
    if best_result["mean_confidence"] is not None:
        print(f"Mean confidence: {best_result['mean_confidence']:.4f}")
    print("Classification report:")
    print(best_result["report_text"])
    print(f"Artifact saved to: {artifact_path}")
    print(f"JSON report saved to: {json_report_path}")
    print(f"Text report saved to: {text_report_path}")
    if len(y_val) < args.min_val_samples:
        print(
            f"Warning: validation set has {len(y_val)} samples (< {args.min_val_samples})."
        )


if __name__ == "__main__":
    main()
