"""
Train network flow classifier using scikit-learn on CIC-IDS2017 data.

Uses RandomForest or GradientBoosting on flow features.
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# ─── Configuration ────────────────────────────────────────────────────

MODEL_DIR = Path(__file__).parent
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "network_classifier.joblib"
METRICS_PATH = MODEL_DIR / "network_classifier_metrics.json"

# Expected CIC-IDS2017 columns (common subset)
FLOW_FEATURES = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
    "Fwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Min",
    "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std",
    "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean",
    "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total",
    "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s",
    "Bwd Packets/s", "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
    "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size",
    "Avg Bwd Segment Size", "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_forward",
    "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min"
]


def load_cic_data(data_dir: Path) -> pd.DataFrame:
    """
    Load CIC-IDS2017 CSV files from directory.
    Expected: CSV files with flow data and 'Label' column.
    """
    csv_files = list(data_dir.glob("*.csv")) + list(data_dir.glob("*/*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            # Standardize column names (strip whitespace)
            df.columns = df.columns.str.strip()
            dfs.append(df)
            print(f"  Loaded {f.name}: {len(df)} rows")
        except Exception as e:
            print(f"  Warning: Could not load {f}: {e}")

    if not dfs:
        raise ValueError("No valid CSV files loaded")

    full_df = pd.concat(dfs, ignore_index=True)
    print(f"Total rows: {len(full_df)}")
    return full_df


def prepare_features(df: pd.DataFrame) -> tuple:
    """Prepare features and labels from CIC-IDS2017 dataframe."""
    # Find label column
    label_col = None
    for col in ["Label", "label", "Attack", "attack", "Category", "category"]:
        if col in df.columns:
            label_col = col
            break

    if label_col is None:
        raise ValueError("No label column found in dataframe")

    # Extract features (use only columns that exist in FLOW_FEATURES)
    available_features = [c for c in FLOW_FEATURES if c in df.columns]
    print(f"Using {len(available_features)} features out of {len(FLOW_FEATURES)}")

    X = df[available_features].copy()
    y = df[label_col].copy()

    # Handle missing values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"Classes: {le.classes_}")
    print(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")

    return X, y_encoded, le, available_features


def train_model(X_train, y_train):
    """Train network flow classifier."""
    print("Training GradientBoostingClassifier...")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            n_iter_no_change=10,
            validation_fraction=0.1
        ))
    ])

    print("Training network flow classifier...")
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(pipeline, X_val, y_val, X_test, y_test, label_encoder):
    """Evaluate model on validation and test sets."""
    print("\nValidation set evaluation:")
    y_val_pred = pipeline.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"  Accuracy: {val_acc:.4f}")
    print(classification_report(y_val, y_val_pred, target_names=label_encoder.classes_))

    print("\nTest set evaluation:")
    y_test_pred = pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"  Accuracy: {test_acc:.4f}")
    print(classification_report(y_test, y_test_pred, target_names=label_encoder.classes_))

    cm = confusion_matrix(y_test, y_test_pred)
    print(f"Confusion Matrix:\n{cm}")

    return {
        "val_accuracy": float(val_acc),
        "test_accuracy": float(test_acc),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_test, y_test_pred, target_names=label_encoder.classes_, output_dict=True),
        "classes": label_encoder.classes_.tolist()
    }


def save_model(pipeline, metrics, feature_names, label_encoder):
    """Save trained model and metrics."""
    model_data = {
        "pipeline": pipeline,
        "feature_names": feature_names,
        "label_encoder": label_encoder
    }
    joblib.dump(model_data, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    metrics["model_info"] = {
        "type": "gradient_boosting",
        "n_features": len(pipeline.named_steps["scaler"].feature_names_in_) if hasattr(pipeline.named_steps["scaler"], "feature_names_in_") else "unknown"
    }
    save_json = lambda obj, path: Path(path).write_text(json.dumps(obj, indent=2))
    import json
    save_json(metrics, METRICS_PATH)
    print(f"Metrics saved to {METRICS_PATH}")


def main():
    print("=" * 60)
    print("Training Network Flow Classifier (CIC-IDS2017)")
    print("=" * 60)

    # Look for CIC-IDS2017 data
    data_dirs = [
        Path("data_pipeline/raw/cic_ids2017"),
        Path("../data_pipeline/raw/cic_ids2017"),
        Path("C:/CIC-IDS2017"),
        Path("~/Downloads/CIC-IDS2017").expanduser(),
    ]

    data_dir = None
    for d in data_dirs:
        if d.exists():
            csv_files = list(d.glob("*.csv")) + list(d.glob("*/*.csv"))
            if csv_files:
                data_dir = d
                break

    if data_dir is None:
        print("CIC-IDS2017 data not found. Skipping network classifier training.")
        print("Place CSV files in data_pipeline/raw/cic_ids2017/ and re-run.")
        return

    print(f"Loading data from {data_dir}...")
    df = load_cic_data(data_dir)

    # Prepare features
    X, y, label_encoder, feature_names = prepare_features(df)

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Train
    pipeline = train_model(X_train, y_train)

    # Evaluate
    metrics = evaluate_model(pipeline, X_val, y_val, X_test, y_test, label_encoder)

    # Save
    save_model(pipeline, metrics, feature_names, label_encoder)

    print("\n" + "=" * 60)
    print(f"Training complete! Test accuracy: {metrics['test_accuracy']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()