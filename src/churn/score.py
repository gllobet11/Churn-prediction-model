"""Score customers with the churn model.

v1 (default): CatBoost retrained with an honest protocol (`churn.train`).
v0: the original MSc model, kept to prove we can reproduce it. The notebook
only persisted encoder/scaler/selector/columns/model. The
correlation drop, winsor limits and rare-category mapping were never saved, so
we re-fit them on the same seed-42 train split — the split is deterministic,
so this reproduces the notebook's state exactly (checked in tests).
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
DELIVERABLE = ROOT / "2_Entregable-20250630T141001Z-1-001" / "2_Entregable"
TRAIN_CSV = DELIVERABLE / "churn - dataset a entrenar.csv"
PREDICT_CSV = DELIVERABLE / "churn-dataset a predecir.csv"
MODEL_DIR = DELIVERABLE / "modelo_churn"
MODELS = ROOT / "models"
TARGET = "churn"
THRESHOLD = 0.472  # notebook's max-F1 threshold (tuned on the test set)

FILL = {
    "income": 0,
    "numbcars": 0,
    "ownrent": "Unknown",
    "lor": "Unknown",
    "infobase": "NM",
    "hnd_webcap": "Unknown",
    "prizm_social_one": "Unknown",
    "area": "Unknown",
    "dualband": "Unknown",
}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Notebook cells 26-39 and 64: fill NaNs, clip impossible negatives."""
    df = df.fillna(FILL).dropna(subset=["roam"])
    for col in ["eqpdays", "totmrc", "avg6rev"]:
        df[col] = df[col].clip(lower=0)
    return df


def split(train_df: pd.DataFrame):
    """The notebook's 70/30 stratified split (cell 70).

    Cell 45 cast low-cardinality (<=25 unique) and text columns of the *train*
    frame to 'category'; that decides which columns later count as numeric
    (corr drop, winsorizing) vs categorical (rare grouping), so we replicate it.
    """
    X = train_df.drop(columns=[TARGET, "Customer_ID"])
    cat = [c for c in X if X[c].nunique() <= 25 or X[c].dtype == "object"]
    X = X.astype(dict.fromkeys(cat, "category"))
    return train_test_split(
        X, train_df[TARGET], test_size=0.3, random_state=42, stratify=train_df[TARGET]
    )


def fit_prep(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Re-fit the unsaved steps on train only (cells 72, 74, 78)."""
    corr = X_train.join(y_train).corr(numeric_only=True)
    drop = set()
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > 0.9:
                c1, c2 = corr.columns[i], corr.columns[j]
                # keep whichever of the pair is more correlated with the target
                t1, t2 = abs(corr.loc[c1, TARGET]), abs(corr.loc[c2, TARGET])
                drop.add(c1 if t1 < t2 else c2)
    X = X_train.drop(columns=list(drop))
    limits = {
        c: (X[c].quantile(0.01), X[c].quantile(0.99))
        for c in X.select_dtypes(include=np.number).columns
    }
    rare = {}
    for c in X.select_dtypes(include=["object", "category"]).columns:
        share = X[c].value_counts(normalize=True)
        cats = share[share < 0.039].index.tolist()
        if len(cats) > 1:
            rare[c] = cats
    return {"drop": list(drop), "limits": limits, "rare": rare}


def transform(X: pd.DataFrame, prep: dict, art: dict) -> pd.DataFrame:
    """Apply the full pipeline up to the model input (cells 72-86, no SMOTE)."""
    X = X.drop(columns=prep["drop"], errors="ignore").copy()
    for c, (lo, hi) in prep["limits"].items():
        X[c] = X[c].clip(lo, hi)
    for c, cats in prep["rare"].items():
        X[c] = X[c].replace(cats, "Other")
    enc = art["encoder"]
    cat_cols = list(enc.feature_names_in_)
    cats = pd.DataFrame(
        enc.transform(X[cat_cols].astype(str)),
        index=X.index,
        columns=enc.get_feature_names_out(cat_cols),
    )
    num = X.drop(columns=cat_cols).select_dtypes(include=np.number)
    encoded = pd.concat([num, cats], axis=1)[list(art["scaler"].feature_names_in_)]
    scaled = pd.DataFrame(
        art["scaler"].transform(encoded), index=X.index, columns=encoded.columns
    )
    return scaled[list(art["columns"])]


def load_artifacts(model_dir: Path = MODEL_DIR) -> dict:
    names = {
        "model": "catboost_model",
        "encoder": "onehot_encoder",
        "scaler": "standard_scaler",
        "columns": "final_columns",
    }
    return {k: joblib.load(model_dir / f"{v}.joblib") for k, v in names.items()}


def score(X: pd.DataFrame, prep: dict, art: dict) -> pd.Series:
    proba = art["model"].predict_proba(transform(X, prep, art))[:, 1]
    return pd.Series(proba, index=X.index, name="churn_proba")


def deciles(proba: pd.Series) -> pd.Series:
    """Risk decile, 10 = highest churn probability."""
    return pd.qcut(proba, 10, labels=range(1, 11)).astype(int).rename("decile")


def run_v0():
    """Score the test split and predict set with the original MSc model."""
    art = load_artifacts()
    X_train, X_test, y_train, y_test = split(clean(pd.read_csv(TRAIN_CSV)))
    prep = fit_prep(X_train, y_train)

    test = pd.DataFrame({"churn_proba": score(X_test, prep, art), "churn": y_test})
    test["decile"] = deciles(test["churn_proba"])
    test["rev"] = X_test["rev"]  # raw monthly revenue, pre-winsorizing

    predict_df = clean(pd.read_csv(PREDICT_CSV))
    X_pred = predict_df.drop(columns=["Customer_ID"])
    predict = pd.DataFrame({"customer_id": predict_df["Customer_ID"]})
    predict["churn_proba"] = score(X_pred, prep, art)
    predict["decile"] = deciles(predict["churn_proba"])
    return test, predict


def split_v1(df: pd.DataFrame):
    """Same rows as `split()`: identical n, stratify vector and seed."""
    return train_test_split(df, test_size=0.3, random_state=42, stratify=df[TARGET])


def features_v1(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    """v1 needs no imputation, clipping or encoding: CatBoost handles numeric
    NaN natively, trees ignore monotone transforms, and text columns go in as
    native categoricals."""
    X = df.drop(columns=[TARGET, "Customer_ID"], errors="ignore").copy()
    X[cat_cols] = X[cat_cols].fillna("NA").astype(str)
    return X


def load_v1() -> tuple[CatBoostClassifier, dict]:
    meta = json.loads((MODELS / "churn_v1.json").read_text())
    return CatBoostClassifier().load_model(str(MODELS / "churn_v1.cbm")), meta


def score_v1(df: pd.DataFrame, model=None, meta=None) -> pd.Series:
    if model is None:
        model, meta = load_v1()
    proba = model.predict_proba(features_v1(df, meta["cat_features"]))[:, 1]
    return pd.Series(proba, index=df.index, name="churn_proba")


def run():
    """Score the labelled test split and all 1,500 predict customers with v1."""
    model, meta = load_v1()
    _, test_df = split_v1(pd.read_csv(TRAIN_CSV))
    test = test_df[[TARGET, "rev"]].copy()
    test["churn_proba"] = score_v1(test_df, model, meta)
    test["decile"] = deciles(test["churn_proba"])

    predict_df = pd.read_csv(PREDICT_CSV)
    predict = predict_df[["Customer_ID"]].rename(columns={"Customer_ID": "customer_id"})
    predict["churn_proba"] = score_v1(predict_df, model, meta)
    predict["decile"] = deciles(predict["churn_proba"])
    return test, predict


def decile_table(test: pd.DataFrame) -> pd.DataFrame:
    return (
        test.groupby("decile")
        .agg(
            n=("churn", "size"),
            mean_proba=("churn_proba", "mean"),
            observed_churn_rate=("churn", "mean"),
            avg_rev=("rev", "mean"),
        )
        .sort_index(ascending=False)
        .reset_index()
    )


if __name__ == "__main__":
    test, predict = run()
    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (ROOT / "exports").mkdir(exist_ok=True)
    test.to_csv(ROOT / "data/processed/churn_scores_test.csv", index_label="row")
    predict.to_csv(ROOT / "data/processed/churn_scores_predict.csv", index=False)
    table = decile_table(test)
    table.to_csv(ROOT / "exports/churn_deciles.csv", index=False)
    print(table.round(3).to_string(index=False))
