from functools import cache

import pandas as pd
from sklearn.metrics import roc_auc_score

from churn.score import DELIVERABLE, decile_table, load_v1, run, run_v0

run, run_v0 = cache(run), cache(run_v0)  # scoring takes seconds; share it


# v0: the original MSc model is reproduced exactly.
def test_v0_reproduces_notebook_test_auc():
    test, _ = run_v0()
    assert abs(roc_auc_score(test["churn"], test["churn_proba"]) - 0.9433) < 5e-4


def test_v0_reproduces_delivered_predictions():
    _, predict = run_v0()
    original = pd.read_excel(DELIVERABLE / "churn_predictions_final.xlsx")
    merged = original.merge(predict, on="customer_id", validate="1:1")
    assert len(merged) == len(original) == 1475
    assert (merged["predict_proba"] - merged["churn_proba"]).abs().max() < 1e-9


# v1: the retrained model.
def test_v1_matches_recorded_test_auc():
    test, _ = run()
    recorded = load_v1()[1]["test"]["auc"]
    assert abs(roc_auc_score(test["churn"], test["churn_proba"]) - recorded) < 1e-9


def test_v1_is_calibrated_per_decile():
    table = decile_table(run()[0])
    assert len(table) == 10 and (table["n"] == 1500).all()
    gap = (table["mean_proba"] - table["observed_churn_rate"]).abs()
    assert gap.max() < 0.05


def test_v1_scores_every_predict_customer():
    _, predict = run()
    assert len(predict) == 1500 and predict["churn_proba"].notna().all()
