"""Прогнозування для вхідних записів за збереженою моделлю."""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

MODELS = Path(__file__).parent / "models"
MODEL = joblib.load(MODELS / "model.joblib")
with open(MODELS / "metadata.json", encoding="utf-8") as f:
    META = json.load(f)


def predict(records):
    """Повертає ймовірність події та рішення для кожного запису.

    records: таблиця pandas або список словників з ознаками моделі.
    """
    records = pd.DataFrame(records)
    missing = [c for c in META["features"] if c not in records.columns]
    if missing:
        raise ValueError("Бракує ознак: " + ", ".join(missing))
    proba = MODEL.predict_proba(records[META["features"]])[:, 1]
    return pd.DataFrame({
        "probability": proba.round(3),
        "decision": (proba >= META["threshold"]).astype(int),
    }, index=records.index)


if __name__ == "__main__":
    records = pd.read_csv(sys.argv[1])
    print(predict(records))
