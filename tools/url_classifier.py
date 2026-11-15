"""URL classifier wrapper for trained model."""

import joblib
from pathlib import Path
from typing import Optional


class URLClassifier:
    """Wrapper for the trained URL phishing/malicious classifier."""

    def __init__(self):
        self.pipeline = None
        self._load_model()

    def _load_model(self):
        """Load the trained classifier from disk."""
        model_path = Path(__file__).parent.parent / "models" / "url_classifier.joblib"
        if model_path.exists():
            self.pipeline = joblib.load(model_path)
        else:
            print(f"Warning: URL model not found at {model_path}")

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