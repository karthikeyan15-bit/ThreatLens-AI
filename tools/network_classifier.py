"""Network flow classifier wrapper for trained model."""

import joblib
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np


class NetworkClassifier:
    """Wrapper for the trained network flow classifier."""

    def __init__(self):
        self.pipeline = None
        self.label_encoder = None
        self._load_model()

    def _load_model(self):
        """Load the trained classifier from disk."""
        model_path = Path(__file__).parent.parent / "models" / "network_classifier.joblib"
        if model_path.exists():
            model_data = joblib.load(model_path)
            self.pipeline = model_data.get("pipeline")
            self.label_encoder = model_data.get("label_encoder")
        else:
            print(f"Warning: Network model not found at {model_path}")

    def predict(self, features: Dict[str, float]) -> Optional[dict]:
        """Predict attack type from flow features."""
        if self.pipeline is None:
            return None

        # Convert features dict to array in correct order
        # This would need the feature names from training
        return None

    def predict_from_dataframe(self, df) -> Optional[np.ndarray]:
        """Predict attack types from a DataFrame of flow features."""
        if self.pipeline is None:
            return None
        return self.pipeline.predict(df)


# Global classifier instance
_NETWORK_CLASSIFIER = None


def get_network_classifier():
    """Get or create the global network classifier."""
    global _NETWORK_CLASSIFIER
    if _NETWORK_CLASSIFIER is None:
        _NETWORK_CLASSIFIER = NetworkClassifier()
    return _NETWORK_CLASSIFIER