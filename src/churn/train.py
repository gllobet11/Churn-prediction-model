"""Retrain the churn model (v1) with an honest protocol.

Differences from the MSc notebook (v0), each one fixing a methodological issue:
- Hyperparameters are tuned with CV on the train split only; the test split is
  scored once, at the end. v0 tuned Optuna and its threshold on the test set.
- No SMOTE: 45/55 classes aren't imbalanced, and resampling distorts the
  probabilities that Phase 6 uses as risk.
- Seeded sampler, so the search is reproducible.

Run once: `uv run --group train python -m churn.train` (~20 min).
"""

import json

import catboost
import numpy as np
import optuna
import pandas as pd
import sklearn
from catboost import CatBoostClassifier
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

from churn.score import MODELS, TARGET, TRAIN_CSV, features_v1, split_v1

SEED = 42
CV = StratifiedKFold(3, shuffle=True, random_state=SEED)


def make_model(params: dict, cat_cols: list[str]) -> CatBoostClassifier:
    return CatBoostClassifier(
        **params, cat_features=cat_cols, random_seed=SEED, verbose=0
    )


def oof_proba(X, y, params, cat_cols) -> np.ndarray:
    """Out-of-fold probabilities: every train row scored by a model that
    didn't see it."""
    proba = np.zeros(len(y))
    for tr, va in CV.split(X, y):
        model = make_model(params, cat_cols).fit(X.iloc[tr], y.iloc[tr])
        proba[va] = model.predict_proba(X.iloc[va])[:, 1]
    return proba


def tune(X, y, cat_cols, n_trials=25) -> dict:
    # Log loss, not AUC or F1: it's a proper scoring rule, so it rewards
    # calibrated probabilities as well as good ranking.
    def objective(trial):
        params = {
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
            "iterations": trial.suggest_int("iterations", 300, 1500),
        }
        return metrics.log_loss(y, oof_proba(X, y, params, cat_cols))

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def f1_threshold(y, proba) -> float:
    grid = np.linspace(0.1, 0.9, 161)
    return float(grid[np.argmax([metrics.f1_score(y, proba >= t) for t in grid])])


def main():
    train_df, test_df = split_v1(pd.read_csv(TRAIN_CSV))
    cat_cols = train_df.drop(columns=["Customer_ID"]).select_dtypes("object")
    cat_cols = cat_cols.columns.tolist()
    X, y = features_v1(train_df, cat_cols), train_df[TARGET]

    params = tune(X, y, cat_cols)
    oof = oof_proba(X, y, params, cat_cols)
    threshold = f1_threshold(y, oof)  # chosen on OOF train predictions only
    model = make_model(params, cat_cols).fit(X, y)

    # The single look at the test set.
    p_test = model.predict_proba(features_v1(test_df, cat_cols))[:, 1]
    y_test = test_df[TARGET]
    meta = {
        "params": params,
        "cat_features": cat_cols,
        "features": X.columns.tolist(),
        "threshold": threshold,
        "cv_oof": {
            "log_loss": metrics.log_loss(y, oof),
            "auc": metrics.roc_auc_score(y, oof),
        },
        "test": {
            "auc": metrics.roc_auc_score(y_test, p_test),
            "pr_auc": metrics.average_precision_score(y_test, p_test),
            "log_loss": metrics.log_loss(y_test, p_test),
            "brier": metrics.brier_score_loss(y_test, p_test),
            "f1_at_threshold": metrics.f1_score(y_test, p_test >= threshold),
        },
        "versions": {"catboost": catboost.__version__, "sklearn": sklearn.__version__},
    }
    MODELS.mkdir(exist_ok=True)
    model.save_model(str(MODELS / "churn_v1.cbm"))
    (MODELS / "churn_v1.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ("params", "threshold", "cv_oof", "test")}))


if __name__ == "__main__":
    main()
