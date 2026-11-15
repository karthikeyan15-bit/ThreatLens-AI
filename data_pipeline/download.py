"""Dataset download and preprocessing scripts for ThreatLens AI."""

import json
from pathlib import Path

MANIFEST_PATH = Path(__file__).parent / "dataset_manifest.json"


def load_manifest() -> dict:
    """Load dataset manifest."""
    return json.loads(MANIFEST_PATH.read_text())


def download_phishing_email_dataset(dest: Path | None = None):
    """Download zefang-liu/phishing-email-dataset from Hugging Face."""
    from datasets import load_dataset
    dest = dest or Path(__file__).parent.parent / "datasets" / "processed"
    dest.mkdir(parents=True, exist_ok=True)

    print("Downloading phishing-email-dataset...")
    ds = load_dataset("zefang-liu/phishing-email-dataset")
    ds.save_to_disk(str(dest / "phishing_email"))
    print(f"Saved to {dest / 'phishing_email'}")


def download_phishing_detection_v2(dest: Path | None = None):
    """Download cybersectony/PhishingEmailDetectionv2.0."""
    from datasets import load_dataset
    dest = dest or Path(__file__).parent.parent / "datasets" / "processed"
    dest.mkdir(parents=True, exist_ok=True)

    print("Downloading PhishingEmailDetection v2.0...")
    ds = load_dataset("cybersectony/PhishingEmailDetectionv2.0")
    ds.save_to_disk(str(dest / "phishing_detection_v2"))
    print(f"Saved to {dest / 'phishing_detection_v2'}")


def preprocess_evaluation_scenarios():
    """Create evaluation scenarios from existing datasets."""
    scenarios = [
        {
            "id": "scenario_001",
            "request": "Analyze this suspicious email: From: spam@free-gift.xyz Subject: You won a prize! Click https://bit.ly/fake-prize to claim your $1000 gift card now!",
            "reference": {
                "expected_tools": ["email_analyzer"],
                "expected_risk": "High",
                "min_expected_evidence": 2,
                "escalate": False,
            },
        },
        {
            "id": "scenario_002",
            "request": "Investigate this URL: https://login-secure-banking.xyz/update?redirect=https://evil.com and also check IP 185.220.101.45",
            "reference": {
                "expected_tools": ["url_analyzer", "threat_intel_lookup", "ip_analyzer"],
                "expected_risk": "Medium",
                "min_expected_evidence": 2,
                "escalate": False,
            },
        },
        {
            "id": "scenario_003",
            "request": "Check domain example.com and its associated IP addresses.",
            "reference": {
                "expected_tools": ["ip_analyzer"],
                "expected_risk": "Low",
                "min_expected_evidence": 1,
                "escalate": False,
            },
        },
    ]

    eval_dir = Path(__file__).parent.parent / "datasets" / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "scenarios.json").write_text(json.dumps(scenarios, indent=2))
    print(f"Created {eval_dir / 'scenarios.json'} with {len(scenarios)} scenarios")


if __name__ == "__main__":
    preprocess_evaluation_scenarios()
