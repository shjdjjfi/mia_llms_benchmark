from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


FEATURE_ORDER = ["loss", "reference", "min_k", "zlib"]


@dataclass
class TrainConfig:
    test_size: float = 0.3
    random_state: int = 42


def build_feature_matrix(feature_dicts: List[Dict[str, float]]) -> np.ndarray:
    return np.array([[f[k] for k in FEATURE_ORDER] for f in feature_dicts], dtype=np.float32)


def train_and_evaluate(feature_dicts: List[Dict[str, float]], labels: List[int], config: TrainConfig) -> Dict[str, float]:
    X = build_feature_matrix(feature_dicts)
    y = np.array(labels, dtype=np.int32)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=config.random_state,
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(np.int32)

    return {
        "auc_roc": float(roc_auc_score(y_test, y_prob)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }
