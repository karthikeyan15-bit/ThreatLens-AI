"""Email threat analyzer — extracts headers, URLs, and classifies phishing likelihood."""

import re
import joblib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .tool_registry import ToolDefinition, register_tool

try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False


HEADERS_TO_EXTRACT = ["From", "To", "Subject", "Date", "Reply-To", "X-Originating-IP"]

SUSPICIOUS_PATTERNS = {
    "urgency": re.compile(r"\b(urgent|immediately|account.*suspend|verify.*now|action.*required|limited.*time)\b", re.I),
    "credential_harvest": re.compile(r"\b(password|login.*credential|verify.*account|confirm.*identity|reset.*password)\b", re.I),
    "spoofing_indicators": re.compile(r"\b(freemail|spam|phishing|not.*verified)\b", re.I),
    "financial": re.compile(r"\b(wire.*transfer|payment.*due|invoice.*overdue|bank.*details|paypal.*update)\b", re.I),
}

PHISHING_KEYWORDS = {
    "free": 1, "gift": 1, "winner": 1, "congratulations": 1, "click": 1,
    "link": 1, "download": 1, "attachment": 1, "open": 1, "view": 1,
    "secure": -1, "official": -1, "support": -1, "customer": -1,
}


class EmailClassifier:
    """Wrapper for the trained email phishing classifier."""

    HF_REPO_ID = "karthikeyan1509/ThreatLens_AI"
    MODEL_FILENAME = "models/email_classifier.joblib"

    def __init__(self):
        self.pipeline = None
        self._load_model()

    def _load_model(self):
        """Load the trained classifier from disk or download from Hugging Face Hub."""
        model_path = Path(__file__).parent.parent / "models" / "email_classifier.joblib"

        if model_path.exists():
            self.pipeline = joblib.load(model_path)
            print(f"Loaded email classifier from local path: {model_path}")
            return

        # Try to download from Hugging Face Hub
        if HF_HUB_AVAILABLE:
            try:
                print(f"Downloading email classifier from Hugging Face Hub...")
                downloaded_path = hf_hub_download(
                    repo_id=self.HF_REPO_ID,
                    filename=self.MODEL_FILENAME,
                    repo_type="space"
                )
                self.pipeline = joblib.load(downloaded_path)
                print(f"Successfully downloaded and loaded email classifier from Hugging Face Hub")
                return
            except Exception as e:
                print(f"Failed to download from Hugging Face Hub: {e}")

        print(f"Warning: Model not found locally and could not download from Hugging Face Hub, falling back to heuristic")

    def predict(self, text: str) -> dict:
        """Predict phishing probability for an email."""
        if self.pipeline is None:
            return None

        # Get prediction and probability
        pred = self.pipeline.predict([text])[0]
        prob = self.pipeline.predict_proba([text])[0]

        # pred is 0 (legitimate) or 1 (phishing)
        label = "phishing" if pred == 1 else "legitimate"
        confidence = float(prob[pred])

        # Also get heuristic score for comparison
        heuristic_score = calculate_phishing_score(text, {})

        return {
            "classification": label,
            "confidence": confidence,
            "ml_score": confidence if pred == 1 else 1.0 - confidence,
            "heuristic_score": heuristic_score,
            "source": "ml_classifier" if self.pipeline else "heuristic_fallback"
        }


# Global classifier instance
_EMAIL_CLASSIFIER: Optional[EmailClassifier] = None


def get_classifier() -> EmailClassifier:
    """Get or create the global email classifier."""
    global _EMAIL_CLASSIFIER
    if _EMAIL_CLASSIFIER is None:
        _EMAIL_CLASSIFIER = EmailClassifier()
    return _EMAIL_CLASSIFIER


def extract_email_headers(raw_text: str) -> dict:
    """Parse email-like text into structured headers."""
    headers = {}
    lines = raw_text.split("\n")
    current_key = None

    for line in lines:
        header_match = re.match(r"^([A-Za-z][\w-]*):\s*(.+)$", line.strip())
        if header_match:
            key, value = header_match.groups()
            headers[key.strip()] = value.strip()
            current_key = key
        elif current_key and line.startswith((" ", "\t")):
            headers[current_key] += " " + line.strip()

    return headers


def extract_urls(text: str) -> list[str]:
    """Extract URLs from text content."""
    pattern = r"https?://[^\s\'\")\]>]+"
    urls = re.findall(pattern, text)
    return [u.rstrip(".,;:)>") for u in urls]


def extract_domains(urls: list[str]) -> list[str]:
    """Extract hostnames from URLs."""
    domains = set()
    for url in urls:
        match = re.search(r"https?://([^/]+)", url)
        if match:
            domains.add(match.group(1))
    return sorted(domains)


def calculate_phishing_score(text: str, headers: dict) -> float:
    """Heuristic scoring based on suspicious patterns and keywords."""
    score = 0.0
    text_lower = text.lower()

    for pattern in SUSPICIOUS_PATTERNS.values():
        matches = pattern.findall(text)
        score += len(matches) * 2.0

    for word, weight in PHISHING_KEYWORDS.items():
        count = len(re.findall(rf"\b{word}\b", text_lower))
        score += weight * count

    # Check for mismatched display vs actual sender
    from_header = headers.get("From", "")
    reply_to = headers.get("Reply-To", "")
    if from_header and reply_to:
        from_domain = re.search(r"@([^>\s]+)", from_header)
        reply_domain = re.search(r"@([^>\s]+)", reply_to)
        if from_domain and reply_domain and from_domain.group(1) != reply_domain.group(1):
            score += 3.0

    # Check link count — many links is suspicious
    urls = extract_urls(text)
    score += len(urls) * 0.5

    return max(0.0, min(1.0, score / 15.0))


def classify_email(raw_content: str) -> dict:
    """Analyze and classify an email using ML model with heuristic fallback."""
    headers = extract_email_headers(raw_content)
    urls = extract_urls(raw_content)
    domains = extract_domains(urls)

    # Try ML classifier first
    classifier = get_classifier()
    ml_result = classifier.predict(raw_content)

    if ml_result:
        classification = ml_result["classification"]
        confidence = ml_result["confidence"]
        score = ml_result["ml_score"]
        source = ml_result["source"]
    else:
        # Fallback to heuristic
        score = calculate_phishing_score(raw_content, headers)

        if score >= 0.6:
            classification = "phishing"
        elif score <= 0.2:
            classification = "legitimate"
        else:
            classification = "inconclusive"

        confidence = round(max(score, 1.0 - score), 3)
        source = "heuristic_fallback"

    return {
        "classification": classification,
        "confidence": confidence,
        "score": round(score, 3),
        "headers": headers,
        "extracted_urls": urls,
        "extracted_domains": domains,
        "indicators": {
            "suspicious_patterns_matched": sum(
                len(p.findall(raw_content)) for p in SUSPICIOUS_PATTERNS.values()
            ),
            "url_count": len(urls),
            "domain_count": len(domains),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source,
    }


def run(args: dict) -> dict:
    """Entry point registered with tool registry."""
    return classify_email(args.get("content", ""))


EMAIL_TOOL = ToolDefinition(
    name="email_analyzer",
    description=(
        "Analyzes an email message for phishing indicators using a trained ML model. "
        "Accepts raw email content or full email text. Extracts headers, URLs, domains, "
        "and classifies as phishing/legitimate/inconclusive with a confidence score."
    ),
    parameters={
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "Raw email content including headers"},
        },
        "required": ["content"],
    },
    category="classification",
    func=run,
)


register_tool(EMAIL_TOOL)