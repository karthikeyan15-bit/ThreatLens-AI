"""
Train email phishing classifier using scikit-learn.

Uses TF-IDF + LogisticRegression on the processed phishing email datasets.
"""

import json
from pathlib import Path

import joblib
import numpy as np
from datasets import load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

# ─── Configuration ────────────────────────────────────────────────────

PROCESSED_DIR = Path(__file__).parent.parent / "data_pipeline" / "processed"
MODEL_DIR = Path(__file__).parent
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "email_classifier.joblib"
METRICS_PATH = MODEL_DIR / "email_classifier_metrics.json"

# Target labels
TARGET_MAP = {"legitimate": 0, "phishing": 1}
REVERSE_MAP = {v: k for k, v in TARGET_MAP.items()}


def load_data():
    """Load processed email datasets."""
    print("Loading phishing_email dataset...")
    ds1 = load_from_disk(str(PROCESSED_DIR / "phishing_email"))

    print("Loading phishing_detection_v2_email dataset...")
    ds2 = load_from_disk(str(PROCESSED_DIR / "phishing_detection_v2_email"))

    # Combine all splits
    train_texts, train_labels = [], []
    val_texts, val_labels = [], []
    test_texts, test_labels = [], []

    for ds in [ds1, ds2]:
        for split_name in ["train", "validation", "test"]:
            if split_name not in ds:
                continue
            split = ds[split_name]
            texts = split["text"]
            labels = split["_label_standardized"]

            for text, label in zip(texts, labels):
                if label in TARGET_MAP:
                    if split_name == "train":
                        train_texts.append(text)
                        train_labels.append(TARGET_MAP[label])
                    elif split_name == "validation":
                        val_texts.append(text)
                        val_labels.append(TARGET_MAP[label])
                    elif split_name == "test":
                        test_texts.append(text)
                        test_labels.append(TARGET_MAP[label])

    print(f"Train: {len(train_texts)} samples")
    print(f"Val: {len(val_texts)} samples")
    print(f"Test: {len(test_texts)} samples")

    return (train_texts, np.array(train_labels),
            val_texts, np.array(val_labels),
            test_texts, np.array(test_labels))


def train_model(X_train, y_train):
    """Train TF-IDF + LogisticRegression pipeline."""
    print("Training TF-IDF + LogisticRegression pipeline...")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words="english",
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

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

    # Also save TF-IDF vocabulary for inspection
    tfidf = pipeline.named_steps["tfidf"]
    vocab_info = {
        "vocab_size": len(tfidf.vocabulary_),
        "sample_features": list(tfidf.vocabulary_.keys())[:20]
    }

    metrics["model_info"] = vocab_info
    save_json = lambda obj, path: Path(path).write_text(json.dumps(obj, indent=2))
    import json
    save_json(metrics, METRICS_PATH)
    print(f"Metrics saved to {METRICS_PATH}")


def main():
    print("=" * 60)
    print("Training Email Phishing Classifier")
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

    # Quick test on sample emails
    print("\nQuick test on sample emails:")
    test_samples = [
        "Hi John, please review the attached invoice at your convenience.",
        "URGENT! Your account will be suspended! Click here NOW to verify: http://fake-login.xyz",
        "Meeting tomorrow at 2pm in conference room B.",
        "Congratulations! You've won $1000! Click http://bit.ly/fake to claim!",
    ]
    for sample in test_samples:
        pred = pipeline.predict([sample])[0]
        prob = pipeline.predict_proba([sample])[0]
        label = REVERSE_MAP[pred]
        confidence = prob[pred]
        print(f"  [{label} ({confidence:.2f})] {sample[:60]}...")


if __name__ == "__main__":
    main()