"""
Train URL phishing/malicious classifier using scikit-learn.

Uses structural + lexical features + TF-IDF on URL text.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from datasets import load_from_disk
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from urllib.parse import urlparse
import tldextract
import re

# Import transformers from module for pickling compatibility
from tools.url_transformers import StructuralFeatureExtractor, URLTFIDFExtractor

# ─── Configuration ────────────────────────────────────────────────────

PROCESSED_DIR = Path(__file__).parent.parent / "data_pipeline" / "processed"
MODEL_DIR = Path(__file__).parent
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "url_classifier.joblib"
METRICS_PATH = MODEL_DIR / "url_classifier_metrics.json"

# Target labels
TARGET_MAP = {"legitimate_url": 0, "phishing_url": 1}
REVERSE_MAP = {v: k for k, v in TARGET_MAP.items()}


# ─── Main Training ────────────────────────────────────────────────────

def load_data():
    """Load processed URL dataset."""
    print("Loading phishing_detection_v2_url dataset...")
    ds = load_from_disk(str(PROCESSED_DIR / "phishing_detection_v2_url"))

    # Combine all splits
    train_texts, train_labels = [], []
    val_texts, val_labels = [], []
    test_texts, test_labels = [], []

    for split_name in ["train", "validation", "test"]:
        if split_name not in ds:
            continue
        split = ds[split_name]
        texts = split["text"]
        labels = split["_label_standardized"]

        for text, label in zip(texts, labels):
            if label in {"legitimate_url", "phishing_url"}:
                if split_name == "train":
                    train_texts.append(text)
                    train_labels.append(1 if label == "phishing_url" else 0)
                elif split_name == "validation":
                    val_texts.append(text)
                    val_labels.append(1 if label == "phishing_url" else 0)
                elif split_name == "test":
                    test_texts.append(text)
                    test_labels.append(1 if label == "phishing_url" else 0)

    print(f"Train: {len(train_texts)} samples")
    print(f"Val: {len(val_texts)} samples")
    print(f"Test: {len(test_texts)} samples")

    return (train_texts, np.array(train_labels),
            val_texts, np.array(val_labels),
            test_texts, np.array(test_labels))


def train_model(X_train, y_train):
    """Train URL classifier using structural features + TF-IDF + RandomForest."""
    print("Training URL classifier (Structural features + TF-IDF + RandomForest)...")

    # Pipeline: combine structural features + char-level TF-IDF
    pipeline = Pipeline([
        ("features", FeatureUnion([
            ("structural", StructuralFeatureExtractor()),
            ("tfidf", URLTFIDFExtractor(max_features=5000))
        ])),
        ("scaler", StandardScaler(with_mean=False)),  # with_mean=False for sparse matrices
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

    print("Training hybrid URL classifier...")
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(pipeline, X_val, y_val, X_test, y_test):
    """Evaluate model on validation and test sets."""
    print("\nValidation set evaluation:")
    y_val_pred = pipeline.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"  Accuracy: {val_acc:.4f}")
    print(classification_report(y_val, y_val_pred, target_names=["legitimate", "phishing"]))

    print("\nTest set evaluation:")
    y_test_pred = pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"  Accuracy: {test_acc:.4f}")
    print(classification_report(y_test, y_test_pred, target_names=["legitimate", "phishing"]))

    cm = confusion_matrix(y_test, y_test_pred)
    print(f"Confusion Matrix:\n{cm}")

    return {
        "val_accuracy": float(val_acc),
        "test_accuracy": float(test_acc),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_test, y_test_pred, target_names=["legitimate", "phishing"], output_dict=True)
    }


def save_model(pipeline, metrics):
    """Save trained model and metrics."""
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    metrics["model_info"] = {
        "type": "hybrid_structural_tfidf_rf",
        "feature_types": ["structural", "char_tfidf"]
    }
    save_json = lambda obj, path: Path(path).write_text(json.dumps(obj, indent=2))
    import json
    save_json(metrics, METRICS_PATH)
    print(f"Metrics saved to {METRICS_PATH}")


def main():
    print("=" * 60)
    print("Training URL Phishing/Malicious Classifier")
    print("=" * 60)

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # Train
    pipeline = train_model(X_train, y_train)

    # Evaluate
    metrics = evaluate_model(pipeline, X_val, y_val, X_test, y_test)

    # Save
    save_model(pipeline, metrics)

    print("\n" + "=" * 60)
    print(f"Training complete! Test accuracy: {metrics['test_accuracy']:.4f}")
    print("=" * 60)

    # Quick test on sample URLs
    print("\nQuick test on sample URLs:")
    test_samples = [
        "https://www.google.com",
        "http://evil-site.xyz/login?id=123",
        "https://github.com/user/repo",
        "http://bit.ly/fake-login",
        "https://secure-banking.example.com/update",
    ]
    for sample in test_samples:
        pred = pipeline.predict([sample])[0]
        prob = pipeline.predict_proba([sample])[0]
        label = "phishing" if pred == 1 else "legitimate"
        confidence = prob[pred]
        print(f"  [{label} ({confidence:.2f})] {sample}")


if __name__ == "__main__":
    main()