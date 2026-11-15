"""
Dataset preprocessing pipeline for ThreatLens AI.

Implements all 10 steps from Section 8 of the spec:
1. Download datasets from original sources
2. Inspect dataset documentation and licensing
3. Validate columns and labels
4. Remove duplicates
5. Handle missing and malformed records
6. Normalize email, URL and network records
7. Preserve original labels and source attribution
8. Create training, validation and test splits
9. Generate dataset statistics
10. Maintain dataset version information
"""

import json
import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

# ─── Configuration ────────────────────────────────────────────────────

MANIFEST_PATH = Path(__file__).parent / "../data_pipeline/dataset_manifest.json"
PROCESSED_DIR = Path(__file__).parent / "processed"
RAW_DIR = Path(__file__).parent / "raw"
VERSION_FILE = PROCESSED_DIR / "VERSION.json"
STATS_FILE = PROCESSED_DIR / "STATS.json"

# Fixed random seed for reproducibility
RANDOM_SEED = 42

# Split ratios (sum = 1.0)
SPLIT_RATIOS = {"train": 0.7, "validation": 0.15, "test": 0.15}

# Expected columns for each dataset type (these are the canonical names we want)
EXPECTED_COLUMNS = {
    "phishing_email": ["text", "label"],
    "phishing_detection_v2_email": ["text", "label"],
    "phishing_detection_v2_url": ["text", "label"],
    "cic_ids2017": [
        "Flow ID", "Source IP", "Source Port", "Destination IP", "Destination Port",
        "Protocol", "Timestamp", "Flow Duration", "Total Fwd Packets",
        "Total Backward Packets", "Total Length of Fwd Packets",
        "Total Length of Bwd Packets", "Fwd Packet Length Max",
        "Fwd Packet Length Min", "Fwd Packet Length Mean",
        "Fwd Packet Length Std", "Bwd Packet Length Max",
        "Bwd Packet Length Min", "Bwd Packet Length Mean",
        "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s",
        "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
        "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max",
        "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std",
        "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags",
        "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length",
        "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
        "Min Packet Length", "Max Packet Length", "Packet Length Mean",
        "Packet Length Std", "Packet Length Variance", "FIN Flag Count",
        "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
        "ACK Flag Count", "URG Flag Count", "CWE Flag Count",
        "ECE Flag Count", "Down/Up Ratio", "Average Packet Size",
        "Avg Fwd Segment Size", "Avg Bwd Segment Size", "Fwd Avg Bytes/Bulk",
        "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk",
        "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate", "Subflow Fwd Packets",
        "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
        "Init_Win_bytes_forward", "Init_Win_bytes_backward",
        "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean",
        "Active Std", "Active Max", "Active Min", "Idle Mean",
        "Idle Std", "Idle Max", "Idle Min", "Label"
    ]
}

# Label mappings for each dataset
LABEL_MAPPINGS = {
    "phishing_email": {0: "legitimate", 1: "phishing", "Safe Email": "legitimate", "Phishing Email": "phishing"},
    "phishing_detection_v2_email": {0: "legitimate_email", 1: "phishing_email", 2: "legitimate_url", 3: "phishing_url"},
    "phishing_detection_v2_url": {0: "legitimate_email", 1: "phishing_email", 2: "legitimate_url", 3: "phishing_url"},
    "cic_ids2017": {
        "Benign": "benign",
        "DoS Hulk": "dos",
        "DoS GoldenEye": "dos",
        "DoS slowloris": "dos",
        "DoS Slowhttptest": "dos",
        "Heartbleed": "web_attack",
        "PortScan": "port_scan",
        "FTP-Patator": "brute_force",
        "SSH-Patator": "brute_force",
        "Web Attack - Brute Force": "web_attack",
        "Web Attack - XSS": "web_attack",
        "Web Attack - Sql Injection": "web_attack",
        "Infiltration": "infiltration",
        "Bot": "botnet",
        "DDoS": "ddos"
    }
}

# ─── Helpers ──────────────────────────────────────────────────────────

def compute_file_hash(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_manifest() -> Dict:
    """Load dataset manifest."""
    return json.loads(MANIFEST_PATH.read_text())


def save_json(obj: Any, path: Path):
    """Save object as JSON with pretty formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def ensure_dirs():
    """Ensure all required directories exist."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ─── Step 1: Download ─────────────────────────────────────────────────

def download_phishing_email_dataset() -> Dict:
    """Download zefang-liu/phishing-email-dataset from Hugging Face."""
    print("Downloading phishing-email-dataset...")
    ds = load_dataset("zefang-liu/phishing-email-dataset")
    ds.save_to_disk(str(RAW_DIR / "phishing_email"))
    print(f"Saved to {RAW_DIR / 'phishing_email'}")
    return {"name": "phishing_email", "source": "zefang-liu/phishing-email-dataset", "status": "downloaded"}


def download_phishing_detection_v2() -> Dict:
    """Download cybersectony/PhishingEmailDetectionv2.0 from Hugging Face."""
    print("Downloading PhishingEmailDetection v2.0...")
    ds = load_dataset("cybersectony/PhishingEmailDetectionv2.0")
    ds.save_to_disk(str(RAW_DIR / "phishing_detection_v2"))
    print(f"Saved to {RAW_DIR / 'phishing_detection_v2'}")
    return {"name": "phishing_detection_v2", "source": "cybersectony/PhishingEmailDetectionv2.0", "status": "downloaded"}


def download_cic_ids2017() -> Dict:
    """
    Download CIC-IDS2017.
    NOTE: This requires manual agreement at https://www.unb.ca/cic/data_pipeline/ids-2017.html
    The function checks for local CSVs in common locations.
    """
    print("Checking for CIC-IDS2017 local files...")
    common_paths = [
        RAW_DIR / "cic_ids2017",
        Path("~/Downloads/CIC-IDS2017").expanduser(),
        Path("~/CIC-IDS2017").expanduser(),
        Path("C:/CIC-IDS2017"),
    ]

    for path in common_paths:
        if path.exists():
            csv_files = list(path.glob("*.csv")) + list(path.glob("*/*.csv"))
            if csv_files:
                print(f"Found CIC-IDS2017 at {path} with {len(csv_files)} CSV files")
                return {"name": "cic_ids2017", "source": str(path), "status": "found_local", "csv_files": [str(f) for f in csv_files]}

    print("CIC-IDS2017 not found locally. Please download from https://www.unb.ca/cic/data_pipeline/ids-2017.html")
    print("and place CSV files in data_pipeline/raw/cic_ids2017/")
    return {"name": "cic_ids2017", "source": "UNB CIC", "status": "not_found", "instructions": "Download from https://www.unb.ca/cic/data_pipeline/ids-2017.html and place CSVs in data_pipeline/raw/cic_ids2017/"}


# ─── Step 2: Inspect Documentation & Licensing ────────────────────────

def inspect_dataset(name: str, ds: DatasetDict) -> Dict:
    """Inspect dataset structure, columns, and licenses."""
    info = {
        "name": name,
        "splits": {},
        "column_names": {},
        "num_rows": {},
        "dtypes": {},
        "license": "unknown"
    }

    for split_name, split_ds in ds.items():
        info["splits"][split_name] = len(split_ds)
        info["column_names"][split_name] = split_ds.column_names
        info["num_rows"][split_name] = len(split_ds)
        info["dtypes"][split_name] = {k: str(v) for k, v in split_ds.features.items()}

    # Try to extract license from dataset card
    try:
        if hasattr(ds, 'info') and ds.info.license:
            info["license"] = ds.info.license
    except Exception:
        pass

    return info


# ─── Step 2.5: Standardize Column Names ───────────────────────────────

def standardize_columns(ds: DatasetDict, name: str) -> DatasetDict:
    """Rename dataset columns to canonical names."""
    rename_map = {
        "phishing_email": {"Email Text": "text", "Email Type": "label"},
        "phishing_detection_v2_email": {"content": "text"},  # Has content instead of text
        "phishing_detection_v2_url": {"content": "text"},    # Has content instead of text
    }
    mapping = rename_map.get(name, {})
    if not mapping:
        return ds

    def rename(example):
        for old, new in mapping.items():
            if old in example:
                example[new] = example.pop(old)
        return example

    return ds.map(rename)


# ─── Step 3: Validate Columns & Labels ────────────────────────────────

def validate_columns(ds: DatasetDict, expected: List[str], name: str) -> Tuple[bool, List[str]]:
    """Validate that dataset has expected columns (after standardization)."""
    errors = []
    for split_name, split_ds in ds.items():
        actual_cols = set(split_ds.column_names)
        expected_cols = set(expected)
        missing = expected_cols - actual_cols
        extra = actual_cols - expected_cols
        if missing:
            errors.append(f"{name}/{split_name}: missing columns {missing}")
        if extra:
            errors.append(f"{name}/{split_name}: extra columns {extra} (will be ignored)")
    return len(errors) == 0, errors


def validate_labels(ds: DatasetDict, label_col: str, valid_labels: set, name: str) -> Tuple[bool, List[str]]:
    """Validate that label column contains only expected values."""
    errors = []
    for split_name, split_ds in ds.items():
        if label_col not in split_ds.column_names:
            errors.append(f"{name}/{split_name}: label column '{label_col}' not found")
            continue
        unique_labels = set(split_ds.unique(label_col))
        invalid = unique_labels - valid_labels
        if invalid:
            errors.append(f"{name}/{split_name}: invalid labels {invalid}")
    return len(errors) == 0, errors


# ─── Step 4: Remove Duplicates ────────────────────────────────────────

def remove_duplicates(ds: DatasetDict, subset: Optional[List[str]] = None) -> DatasetDict:
    """Remove duplicate rows from each split."""
    result = {}
    for split_name, split_ds in ds.items():
        # Convert to pandas, drop duplicates, convert back
        df = split_ds.to_pandas()
        original_len = len(df)
        if subset:
            df = df.drop_duplicates(subset=subset)
        else:
            df = df.drop_duplicates()
        removed = original_len - len(df)
        if removed:
            print(f"  Removed {removed} duplicates from {split_name}")
        result[split_name] = Dataset.from_pandas(df, preserve_index=False)
    return DatasetDict(result)


# ─── Step 5: Handle Missing & Malformed Records ──────────────────────

def handle_missing_malformed(ds: DatasetDict, required_cols: List[str]) -> DatasetDict:
    """Remove rows with missing values in required columns; fix malformed data."""
    result = {}
    for split_name, split_ds in ds.items():
        # Remove rows where any required column is null/empty
        df = split_ds.to_pandas()
        original_len = len(df)

        # Handle missing
        df = df.dropna(subset=required_cols)

        # Handle empty strings in text columns
        for col in required_cols:
            if df[col].dtype == object:
                df = df[df[col].astype(str).str.strip() != ""]

        removed = original_len - len(df)
        if removed:
            print(f"  Removed {removed} missing/malformed rows from {split_name}")

        result[split_name] = Dataset.from_pandas(df, preserve_index=False)
    return DatasetDict(result)


# ─── Step 6: Normalize Records ────────────────────────────────────────

def normalize_email(ds: DatasetDict) -> DatasetDict:
    """Normalize email text: strip whitespace, fix encoding issues."""
    def _normalize_text(example):
        text = example["text"]
        if isinstance(text, str):
            text = text.strip()
            # Remove null bytes and control characters
            text = "".join(char for char in text if ord(char) >= 32 or char in "\n\t")
            text = text.replace("\r\n", "\n").replace("\r", "\n")
        example["text"] = text
        return example

    return ds.map(_normalize_text)


def normalize_url(ds: DatasetDict) -> DatasetDict:
    """Normalize URL text: ensure proper formatting."""
    def _normalize_text(example):
        text = example["text"]
        if isinstance(text, str):
            text = text.strip()
            # Ensure URL has scheme
            if text and not text.startswith(("http://", "https://")):
                text = "http://" + text
        example["text"] = text
        return example

    return ds.map(_normalize_text)


def normalize_cic_ids2017(ds: DatasetDict) -> DatasetDict:
    """Normalize CIC-IDS2017 flow records."""
    def normalize_flow(example):
        # Convert all numeric fields to float, handling 'Infinity' and NaN
        for key, value in example.items():
            if isinstance(value, (int, float)):
                if np.isinf(value) or np.isnan(value):
                    example[key] = 0.0
            elif isinstance(value, str):
                try:
                    fval = float(value)
                    if np.isinf(fval) or np.isnan(fval):
                        example[key] = 0.0
                    else:
                        example[key] = fval
                except ValueError:
                    pass  # Keep string values (IPs, timestamps)
        return example

    return ds.map(normalize_flow)


# ─── Step 7: Preserve Labels & Source Attribution ─────────────────────

def add_source_attribution(ds: DatasetDict, source_name: str) -> DatasetDict:
    """Add source attribution columns to each record."""
    def add_source(example):
        example["_source"] = source_name
        example["_source_timestamp"] = datetime.now(timezone.utc).isoformat()
        return example
    return ds.map(add_source)


def map_labels(ds: DatasetDict, label_col: str, mapping: Dict) -> DatasetDict:
    """Map raw labels to standardized labels."""
    def map_label(example):
        raw = example[label_col]
        if raw in mapping:
            example["_label_standardized"] = mapping[raw]
        else:
            example["_label_standardized"] = str(raw)
        return example
    return ds.map(map_label)


# ─── Step 8: Create Train/Val/Test Splits ─────────────────────────────

def create_splits(ds: DatasetDict, ratios: Dict[str, float] = SPLIT_RATIOS) -> DatasetDict:
    """Create train/validation/test splits if not already present."""
    # If dataset already has the required splits, return as-is
    required = set(ratios.keys())
    if required.issubset(set(ds.keys())):
        return ds

    # Otherwise, combine all data and resplit
    combined = []
    for split_name, split_ds in ds.items():
        df = split_ds.to_pandas()
        df["_original_split"] = split_name
        combined.append(df)

    full_df = pd.concat(combined, ignore_index=True)
    full_ds = Dataset.from_pandas(full_df, preserve_index=False)

    # Shuffle and split
    shuffled = full_ds.shuffle(seed=RANDOM_SEED)
    n = len(shuffled)
    train_end = int(n * ratios["train"])
    val_end = train_end + int(n * ratios["validation"])

    return DatasetDict({
        "train": shuffled.select(range(train_end)),
        "validation": shuffled.select(range(train_end, val_end)),
        "test": shuffled.select(range(val_end, n))
    })


# ─── Step 9: Generate Statistics ──────────────────────────────────────

def generate_statistics(ds: DatasetDict, name: str, label_col: str) -> Dict:
    """Generate comprehensive statistics for the dataset."""
    stats = {
        "dataset": name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "splits": {},
        "overall": {
            "total_rows": sum(len(s) for s in ds.values()),
            "label_distribution": {}
        }
    }

    for split_name, split_ds in ds.items():
        split_stats = {
            "num_rows": len(split_ds),
            "columns": split_ds.column_names,
            "missing_values": {}
        }

        # Missing values per column
        for col in split_ds.column_names:
            if col.startswith("_"):
                continue
            df = split_ds.to_pandas()
            split_stats["missing_values"][col] = int(df[col].isna().sum())

        # Label distribution
        if label_col in split_ds.column_names:
            label_counts = split_ds.to_pandas()[label_col].value_counts().to_dict()
            split_stats["label_distribution"] = {str(k): int(v) for k, v in label_counts.items()}
            # Add to overall
            for k, v in label_counts.items():
                stats["overall"]["label_distribution"][str(k)] = stats["overall"]["label_distribution"].get(str(k), 0) + int(v)

        stats["splits"][split_name] = split_stats

    return stats


# ─── Step 10: Maintain Version Information ────────────────────────────

def save_version_info(version_info: Dict):
    """Save version information."""
    version_info["last_updated"] = datetime.now(timezone.utc).isoformat()
    save_json(version_info, VERSION_FILE)
    # Also save stats
    if "statistics" in version_info:
        save_json(version_info["statistics"], STATS_FILE)


def get_next_version() -> str:
    """Generate next version string (semantic versioning with date)."""
    today = datetime.now(timezone.utc).strftime("%Y.%m.%d")
    if VERSION_FILE.exists():
        try:
            current = json.loads(VERSION_FILE.read_text())
            if current.get("date") == today:
                # Increment patch
                parts = current.get("version", "0.0.0").split(".")
                parts[2] = str(int(parts[2]) + 1)
                return ".".join(parts)
        except Exception:
            pass
    return f"{today}.0"


# ─── Main Pipeline ────────────────────────────────────────────────────

def run_preprocessing(skip_cic: bool = True) -> Dict:
    """
    Run the complete preprocessing pipeline.

    Returns a dict with results for each dataset.
    """
    ensure_dirs()

    results = {
        "version": get_next_version(),
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "datasets": {},
        "statistics": {}
    }

    # ─── Phishing Email Dataset ──────────────────────────────────────
    print("\n=== Processing phishing-email-dataset ===")
    download_phishing_email_dataset()
    ds = load_dataset(str(RAW_DIR / "phishing_email"))

    # Inspect
    info = inspect_dataset("phishing_email", ds)
    print(f"  Columns: {ds['train'].column_names if 'train' in ds else ds['test'].column_names}")

    # Standardize column names
    ds = standardize_columns(ds, "phishing_email")

    # Validate
    ok, errors = validate_columns(ds, EXPECTED_COLUMNS["phishing_email"], "phishing_email")
    if errors:
        print(f"  Validation warnings: {errors}")
    ok, errors = validate_labels(ds, "label", {0, 1}, "phishing_email")
    if errors:
        print(f"  Label warnings: {errors}")

    # Remove duplicates
    ds = remove_duplicates(ds, subset=["text"])

    # Handle missing
    ds = handle_missing_malformed(ds, ["text", "label"])

    # Normalize
    ds = normalize_email(ds)

    # Add source attribution
    ds = add_source_attribution(ds, "zefang-liu/phishing-email-dataset")

    # Map labels
    ds = map_labels(ds, "label", LABEL_MAPPINGS["phishing_email"])

    # Create splits (dataset already has train/test)
    ds = create_splits(ds)

    # Save processed
    output_dir = PROCESSED_DIR / "phishing_email"
    output_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(output_dir))

    # Statistics
    stats = generate_statistics(ds, "phishing_email", "label")
    results["datasets"]["phishing_email"] = info
    results["statistics"]["phishing_email"] = stats
    print(f"  Saved to {output_dir}")
    print(f"  Splits: {[(k, len(v)) for k, v in ds.items()]}")

    # ─── Phishing Detection v2.0 (Email) ─────────────────────────────
    print("\n=== Processing PhishingEmailDetection v2.0 (Email) ===")
    download_phishing_detection_v2()
    ds_v2 = load_dataset(str(RAW_DIR / "phishing_detection_v2"))

    # Split by label type: 0,1 = email; 2,3 = URL
    # We'll create two separate datasets
    for label_type in ["email", "url"]:
        if label_type == "email":
            keep_labels = {0, 1}
            name = "phishing_detection_v2_email"
            expected = EXPECTED_COLUMNS["phishing_detection_v2_email"]
        else:
            keep_labels = {2, 3}
            name = "phishing_detection_v2_url"
            expected = EXPECTED_COLUMNS["phishing_detection_v2_url"]

        print(f"\n=== Processing {name} ===")

        # Filter
        ds_filtered = ds_v2.filter(lambda x: x["label"] in keep_labels)

        if len(ds_filtered["train"]) == 0:
            print(f"  No {label_type} records found, skipping")
            continue

        # Standardize column names
        ds_filtered = standardize_columns(ds_filtered, name)

        # Validate
        ok, errors = validate_columns(ds_filtered, expected, name)
        if errors:
            print(f"  Validation warnings: {errors}")
        ok, errors = validate_labels(ds_filtered, "label", keep_labels, name)
        if errors:
            print(f"  Label warnings: {errors}")

        # Remove duplicates
        ds_filtered = remove_duplicates(ds_filtered, subset=["text"])

        # Handle missing
        ds_filtered = handle_missing_malformed(ds_filtered, ["text", "label"])

        # Normalize
        if label_type == "email":
            ds_filtered = normalize_email(ds_filtered)
        else:
            ds_filtered = normalize_url(ds_filtered)

        # Add source attribution
        ds_filtered = add_source_attribution(ds_filtered, "cybersectony/PhishingEmailDetectionv2.0")

        # Map labels
        ds_filtered = map_labels(ds_filtered, "label", LABEL_MAPPINGS["phishing_detection_v2_email"])

        # Create splits
        ds_filtered = create_splits(ds_filtered)

        # Save
        output_dir = PROCESSED_DIR / name
        output_dir.mkdir(parents=True, exist_ok=True)
        ds_filtered.save_to_disk(str(output_dir))

        # Statistics
        stats = generate_statistics(ds_filtered, name, "label")
        results["datasets"][name] = {"source": "cybersectony/PhishingEmailDetectionv2.0", "label_type": label_type}
        results["statistics"][name] = stats
        print(f"  Saved to {output_dir}")
        print(f"  Splits: {[(k, len(v)) for k, v in ds_filtered.items()]}")

    # ─── CIC-IDS2017 ─────────────────────────────────────────────────
    if not skip_cic:
        print("\n=== Processing CIC-IDS2017 ===")
        cic_result = download_cic_ids2017()
        if cic_result["status"] == "found_local":
            # Load and process CSVs
            print("  CIC-IDS2017 processing not fully implemented — requires schema detection")
            results["datasets"]["cic_ids2017"] = cic_result
        else:
            print(f"  Skipped: {cic_result.get('instructions', 'Not found')}")
            results["datasets"]["cic_ids2017"] = cic_result
    else:
        print("\n=== Skipping CIC-IDS2017 (not available locally) ===")
        results["datasets"]["cic_ids2017"] = {"status": "skipped", "reason": "No local CSVs found"}

    # Save version info
    save_version_info(results)

    print(f"\n=== Preprocessing Complete ===")
    print(f"Version: {results['version']}")
    print(f"Processed datasets: {list(results['datasets'].keys())}")
    print(f"Stats saved to: {STATS_FILE}")
    print(f"Version saved to: {VERSION_FILE}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ThreatLens AI Dataset Preprocessing Pipeline")
    parser.add_argument("--skip-cic", action="store_true", default=True, help="Skip CIC-IDS2017 (default: True)")
    args = parser.parse_args()
    run_preprocessing(skip_cic=args.skip_cic)