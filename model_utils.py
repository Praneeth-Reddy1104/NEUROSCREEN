"""
model_utils.py
---------------
Loads the trained model + label encoders produced by train.py and
turns a validated ASDInput into a model-ready feature row.
"""

import pickle
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "best_model.pkl"
ENCODERS_PATH = ARTIFACTS_DIR / "encoders.pkl"
FEATURE_COLUMNS_PATH = ARTIFACTS_DIR / "feature_columns.pkl"


class ModelNotTrainedError(RuntimeError):
    """Raised when artifacts from train.py haven't been generated yet."""


class ASDModel:
    def __init__(self):
        self.model = None
        self.encoders = None
        self.feature_columns = None
        self.loaded = False

    def load(self):
        if not (MODEL_PATH.exists() and ENCODERS_PATH.exists() and FEATURE_COLUMNS_PATH.exists()):
            self.loaded = False
            return

        with open(MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)
        with open(ENCODERS_PATH, "rb") as f:
            self.encoders = pickle.load(f)
        with open(FEATURE_COLUMNS_PATH, "rb") as f:
            self.feature_columns = pickle.load(f)
        self.loaded = True

    def _encode_value(self, column: str, value):
        """Encode a categorical value, falling back gracefully for unseen categories."""
        encoder = self.encoders[column]
        if value in encoder.classes_:
            return int(encoder.transform([value])[0])

        # unseen category: prefer an "Others" bucket if the encoder has one,
        # otherwise fall back to the most common class-index-0 sentinel
        if "Others" in encoder.classes_:
            return int(encoder.transform(["Others"])[0])
        return 0

    def predict(self, payload: dict):
        if not self.loaded:
            raise ModelNotTrainedError(
                "Model artifacts not found. Run backend/train.py with data/train.csv first."
            )

        row = dict(payload)
        for column in self.encoders:
            if column in row:
                row[column] = self._encode_value(column, row[column])

        X = pd.DataFrame([row])[self.feature_columns]

        prediction = int(self.model.predict(X)[0])
        if hasattr(self.model, "predict_proba"):
            probability = float(self.model.predict_proba(X)[0][prediction])
        else:
            probability = 1.0

        return prediction, probability


# single shared instance, loaded once at app startup
asd_model = ASDModel()
