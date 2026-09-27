from functools import cache

import pandas as pd
from sklearn.metrics import roc_auc_score

from churn.score import DELIVERABLE, decile_table, run

run = cache(run)  # scoring takes a few seconds; share it across tests


def test_reproduces_notebook_test_auc():
    test, _ = run()
    assert abs(roc_auc_score(test["churn"], test["churn_proba"]) - 0.9433) < 5e-4


def test_reproduces_delivered_predictions():
    _, predict = run()
    original = pd.read_excel(DELIVERABLE / "churn_predictions_final.xlsx")
    merged = original.merge(predict, on="customer_id", validate="1:1")
    assert len(merged) == len(original) == 1475
    assert (merged["predict_proba"] - merged["churn_proba"]).abs().max() < 1e-9


def test_deciles_rank_risk():
    table = decile_table(run()[0])
    assert len(table) == 10 and (table["n"] == 1500).all()
    rates = table.set_index("decile")["observed_churn_rate"].sort_index()
    assert rates.is_monotonic_increasing  # higher decile, more observed churn
