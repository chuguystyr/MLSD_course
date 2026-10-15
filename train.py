import json
import platform
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = "data/bank-additional-full.csv"
TARGET = "y"
EXCLUDE = ["duration"]
THRESHOLD = 0.21  # обрано в ЛР6 на валідаційній частині

data = pd.read_csv(DATA, sep=";")
target = "y"
data[target] = (data[target] == "yes").astype(int)
exclude = ["duration"]

data["no_previous_contact"] = (data["pdays"] == 999).astype(int)
data["pdays"] = data["pdays"].replace(999, np.nan)

# Стратифікований розподіл даних
train, temp = train_test_split(
    data, test_size=0.4, random_state=42, stratify=data[target]
)

valid, test = train_test_split(
    temp, test_size=0.5, random_state=42, stratify=temp[target]
)

rest = pd.concat([train, valid])

features = [c for c in data.columns if c not in exclude + [target]]
numeric = train[features].select_dtypes("number").columns.tolist()
categorical = [c for c in features if c not in numeric]
pre = ColumnTransformer(
    [
        (
            "num",
            make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
            numeric,
        ),
        (
            "cat",
            make_pipeline(
                SimpleImputer(strategy="most_frequent"),
                OneHotEncoder(handle_unknown="ignore"),
            ),
            categorical,
        ),
    ]
)

model = make_pipeline(pre, LogisticRegression(max_iter=1000))
model.fit(rest[features], rest[target])

Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/model.joblib")

metadata = {
    "model": type(model[-1]).__name__,
    "trained_on": "train + valid (rest)",
    "threshold": THRESHOLD,
    "cost_formula": "C = 15 * (fn / (fn + tp)) + fp / (fp + tn)",
    "features": features,
    "python": platform.python_version(),
    "scikit-learn": sklearn.__version__,
    "created": date.today().isoformat(),
    "test_metrics": None,
}
with open("models/metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print("Модель навчено на", len(rest), "записах і збережено в models/")
