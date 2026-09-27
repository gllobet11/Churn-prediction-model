# Churn model card

The CatBoost churn classifier from my MSc (2025) deliverable, recovered and
re-scored from `src/churn/score.py`. The original folder
(`2_Entregable-*/2_Entregable/`) is untouched.

## Data and target
- Public telecom subscriber dataset from the course: 50,000 labelled customers
  (`churn - dataset a entrenar.csv`, 84 features + `Customer_ID`) and 1,500
  unlabelled customers to score (`churn-dataset a predecir.csv`; 25 rows with
  missing `roam` were dropped, leaving 1,475).
- Target `churn`: the customer left 31–60 days after the observation date.
  Base rate 45.1%. That is far above real-world monthly churn, so the course
  data was probably sampled to be close to balanced.

## Pipeline (as in the notebook)
1. Fill NaNs: `income` and `numbcars` → 0; categorical columns → `Unknown` / `NM`.
   Negative `eqpdays`, `totmrc` and `avg6rev` are clipped to 0.
2. Stratified 70/30 train/test split, seed 42.
3. Everything below is fitted on train only. Drop one column of each pair with
   |corr| > 0.9, keeping the one more correlated with the target (21 dropped).
   Winsorize numerics at the 1st/99th percentiles. Group categories under 3.9%
   into `Other`.
4. One-hot encoding (drop first), StandardScaler, VarianceThreshold(0.01).
   This leaves 119 features.
5. SMOTE on train, then CatBoost with the params Optuna found:
   `iterations=1872, depth=6, learning_rate=0.143, l2_leaf_reg=8.91,
   bootstrap_type=Bayesian, random_seed=42`.

**Recovery note.** The notebook saved the encoder, scaler, selector, column
list and model, but not the correlation drop list, the winsor limits or the
rare-category mapping. `score.py` re-fits those on the same seed-42 train
split. The split is deterministic, so this reproduces the notebook exactly:
test AUC 0.9433, and all 1,475 delivered predictions match
`churn_predictions_final.xlsx` to within 1e-16 (`tests/test_churn_score.py`).
The original `requirements.txt` says scikit-learn 1.4.2, but the artifacts
were pickled with **1.6.1**, so that is the version pinned here.

## Performance on the test split (n = 15,000; 6,765 churners)

| Metric | Notebook reported | Recomputed here |
|---|---|---|
| ROC AUC | 0.9433 | 0.9433 |
| PR AUC (average precision) | not reported | 0.9365 (base rate 0.451) |
| F1 @ 0.5 | 0.8511 | 0.8511 |
| Best F1 (threshold 0.472) | 0.8527 | 0.8524¹ |
| Precision / recall, churn @ 0.472 | 0.85 / 0.85 | 0.854 / 0.851 |
| Brier score | not reported | 0.095 |

¹ The notebook picked its threshold on a `linspace` grid and printed it
rounded to 0.472. Re-applying 0.472 gives a slightly lower F1.

Confusion matrix @ 0.472 (recomputed):

| | Pred. no churn | Pred. churn |
|---|---|---|
| **Actual no churn** | 7,247 (TN) | 988 (FP) |
| **Actual churn** | 1,006 (FN) | 5,759 (TP) |

On the predict set, 673 of 1,475 customers are above the threshold, the same
count as the notebook.

## Risk deciles (test split, observed churn rate)
Source: `exports/churn_deciles.csv`. Decile 10 is the highest risk; each
decile has 1,500 customers.

| Decile | Mean predicted p | Observed churn | Avg `rev` |
|---|---|---|---|
| 10 | 0.998 | 0.997 | 50.2 |
| 9 | 0.980 | 0.968 | 54.5 |
| 8 | 0.909 | 0.899 | 56.5 |
| 7 | 0.726 | 0.712 | 60.8 |
| 6 | 0.470 | 0.457 | 62.1 |
| 5 | 0.246 | 0.275 | 62.1 |
| 4 | 0.102 | 0.131 | 65.9 |
| 3 | 0.032 | 0.053 | 66.7 |
| 2 | 0.008 | 0.017 | 61.3 |
| 1 | 0.001 | 0.001 | 53.6 |

## Top drivers
- Built-in CatBoost importance of the **final** model: `months` 9.9,
  `mou` 8.9, `eqpdays` 8.0, `change_mou` 5.6, `adjmou` 5.2, `totmrc` 4.7,
  `change_rev` 3.4, `vceovr` 2.9, `avg6rev` 2.7, `drop_blk` 2.3.
- The notebook's SHAP top 5 was `mou`, `months`, `eqpdays`, `change_mou`,
  `adjmou` (`figures/churn_shap_top5.png`, `figures/churn_shap_summary.png`).
  It is the same set as the final model's top 5.
- Direction, from the notebook's SHAP summary and logistic coefficients: churn
  risk goes up with less usage (`mou`), shorter tenure (`months`), an older
  handset (`eqpdays`) and falling usage (`change_mou` < 0).

## Caveats
1. **Test-set tuning.** Optuna and the F1 threshold were both chosen on the
   test split, and no separate holdout was kept. The metrics above are
   therefore optimistic.
2. **SHAP model ≠ final model.** The notebook's SHAP plots come from a separate
   default CatBoost, not the tuned model. The final model's built-in
   importance (above) agrees on the top 5.
3. **Calibration.** The model was trained after SMOTE (50/50 classes), so its
   probabilities are not calibrated by design. The observed churn rate per
   decile is the honest risk estimate, and that is what Phase 6 uses.
4. **AUC 0.94 is higher than I would expect for telecom churn and remains
   unexplained.** Checks so far: no single feature has a univariate AUC above
   0.655 (`eqpdays`); there are no duplicate feature rows; `Customer_ID`
   carries no signal (AUC 0.50). Logistic regression reaches 0.80 on the same
   features, so the signal comes from interactions. A leak through a feature
   defined after the observation date cannot be ruled out without the
   original data documentation. Do not present 0.94 as a benchmark.
5. **Near-certain top decile.** 99.7% of decile 10 churns. For a retention
   experiment, customers that far gone may not be savable, so deciles 7–9 are
   probably a better target (see Phase 6).

## Reproduce
```bash
uv run python -m churn.score   # writes data/processed/*.csv and exports/churn_deciles.csv
uv run pytest tests/test_churn_score.py
```
Versions: Python 3.11, catboost 1.2.5, scikit-learn 1.6.1, pandas 2.2.2,
numpy 1.26.4.
