"""URL investigation tool — structural analysis + ML classification + threat intel enrichment."""

import joblib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import tldextract

from .tool_registry import ToolDefinition, register_tool
from .url_transformers import extract_url_features

try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False


SUSPICIOUS_TLDS = {".xyz", ".top", ".club", ".work", ".site", ".buzz", ".racing", ".gq", ".ml", ".cf", ".tk"}
SUSPICIOUS_PATH_PATTERNS = [
    re.compile(r"\.(exe|bat|cmd|scr|js|vbs|ps1|dll|com)\?"),
    re.compile(r"[?&](redirect|return|next|r)=https?://"),
    re.compile(r"/wp-login\.php"),
    re.compile(r"/login.*id="),
]


class URLClassifier:
    """Wrapper for the trained URL phishing/malicious classifier."""

    HF_REPO_ID = "karthikeyan1509/ThreatLens_AI"
    MODEL_FILENAME = "models/url_classifier.joblib"

    def __init__(self):
        self.pipeline = None
        self._load_model()

    def _load_model(self):
        """Load the trained classifier from disk or download from Hugging Face Hub."""
        model_path = Path(__file__).parent.parent / "models" / "url_classifier.joblib"

        if model_path.exists():
            self.pipeline = joblib.load(model_path)
            print(f"Loaded URL classifier from local path: {model_path}")
            return

        # Try to download from Hugging Face Hub
        if HF_HUB_AVAILABLE:
            try:
                print(f"Downloading URL classifier from Hugging Face Hub...")
                downloaded_path = hf_hub_download(
                    repo_id=self.HF_REPO_ID,
                    filename=self.MODEL_FILENAME,
                    repo_type="space"
                )
                self.pipeline = joblib.load(downloaded_path)
                print(f"Successfully downloaded and loaded URL classifier from Hugging Face Hub")
                return
            except Exception as e:
                print(f"Failed to download from Hugging Face Hub: {e}")

        print(f"Warning: URL model not found locally and could not download from Hugging Face Hub")

    def predict(self, url: str) -> Optional[dict]:
        """Predict phishing/malicious probability for a URL."""
        if self.pipeline is None:
            return None

        pred = self.pipeline.predict([url])[0]
        prob = self.pipeline.predict_proba([url])[0]

        label = "malicious" if pred == 1 else "legitimate"
        confidence = float(prob[pred])

        return {
            "classification": label,
            "confidence": confidence,
            "ml_score": confidence if pred == 1 else 1.0 - confidence,
            "source": "ml_classifier"
        }


# Global classifier instance
_URL_CLASSIFIER = None


def get_url_classifier():
    """Get or create the global URL classifier."""
    global _URL_CLASSIFIER
    if _URL_CLASSIFIER is None:
        _URL_CLASSIFIER = URLClassifier()
    return _URL_CLASSIFIER


def analyze_url(url: str) -> dict:
    """Perform structural analysis + ML classification of a URL for threat indicators."""
    parsed = urlparse(url)
    tld_ext = tldextract.extract(parsed.netloc)

    scheme = parsed.scheme
    hostname = parsed.netloc
    path = parsed.path
    query = parsed.query
    suffix = tld_ext.suffix
    registrar = tld_ext.top_domain_under_public_suffix

    risk_score = 0.0
    indicators = []

    # Scheme check
    if scheme != "https":
        risk_score += 0.3
        indicators.append(f"Suspicious scheme: {scheme}")

    # IP-based hostname
    ip_match = re.match(r"^\d+\.\d+\.\d+\.\d+$", hostname.split(":")[0])
    if ip_match:
        risk_score += 0.4
        indicators.append("Hostname is a raw IP address")

    # Suspicious TLD
    if suffix in {s.lstrip(".") for s in SUSPICIOUS_TLDS}:
        risk_score += 0.3
        indicators.append(f"Suspicious TLD: .{suffix}")

    # URL shortener
    shorteners = {"bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly"}
    if any(s in hostname for s in shorteners):
        risk_score += 0.2
        indicators.append("URL uses a known shortener service")

    # Path patterns
    for pat in SUSPICIOUS_PATH_PATTERNS:
        if pat.search(path):
            risk_score += 0.3
            indicators.append(f"Suspicious path pattern: {pat.pattern}")

    # Excessive subdomains (possible typosquatting)
    parts = hostname.split(".")
    if len(parts) > 4:
        risk_score += 0.2
        indicators.append("Excessive subdomains detected")

    # Homograph / idna hints
    if "xn--" in hostname:
        risk_score += 0.1
        indicators.append("IDN/homograph indicator present")

    risk_score = min(1.0, risk_score)

    # Get ML classification
    ml_classifier = get_url_classifier()
    ml_result = ml_classifier.predict(url)

    # Combine structural risk with ML classification
    structural_verdict = (
        "malicious" if risk_score >= 0.6
        else "suspicious" if risk_score >= 0.3
        else "structurally_unremarkable"
    )

    if ml_result:
        ml_label = ml_result["classification"]
        ml_confidence = ml_result["confidence"]
        ml_source = ml_result["source"]
    else:
        ml_label = "unknown"
        ml_confidence = 0.0
        ml_source = "structural_only"

    # Combined verdict logic
    if ml_result and ml_result["classification"] == "malicious":
        combined_verdict = "malicious"
    elif risk_score >= 0.6:
        combined_verdict = "malicious"
    elif risk_score >= 0.3:
        combined_verdict = "suspicious"
    else:
        combined_verdict = "structurally_unremarkable"

    return {
        "url": url,
        "parsed": {
            "scheme": scheme,
            "hostname": hostname,
            "registered_domain": registrar,
            "subdomain": tld_ext.subdomain,
            "suffix": suffix,
            "path": path,
            "query": query,
        },
        "structural_risk_score": round(risk_score, 3),
        "structural_indicators": indicators,
        "structural_verdict": structural_verdict,
        "ml_classification": ml_label,
        "ml_confidence": ml_confidence if ml_result else 0.0,
        "ml_source": ml_source,
        "combined_verdict": combined_verdict,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "hybrid_analysis",
    }


# Global classifier instance
_URL_CLASSIFIER = None


def get_url_classifier():
    """Get or create the global URL classifier."""
    global _URL_CLASSIFIER
    if _URL_CLASSIFIER is None:
        from .url_classifier import URLClassifier
        _URL_CLASSIFIER = URLClassifier()
    return _URL_CLASSIFIER


def run(args: dict) -> dict:
    """Entry point registered with tool registry."""
    url = args.get("url", "").strip()
    if not url:
        return {"success": False, "error": "URL is required"}
    return analyze_url(url)


URL_TOOL = ToolDefinition(
    name="url_analyzer",
    description=(
        "Investigates a single URL by analyzing its structure, scheme, domain, path, and TLD, "
        "AND using a trained ML model for phishing/malicious classification. "
        "Returns structural risk score, ML classification, and combined verdict. "
        "For external threat intelligence lookup, use `threat_intel_lookup` instead."
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The full URL to investigate"},
        },
        "required": ["url"],
    },
    category="analysis",
    func=run,
)


register_tool(URL_TOOL)