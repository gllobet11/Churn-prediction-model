# Churn model card

Two versions of the model, both scored by `src/churn/score.py`:

- **v0**: the CatBoost model from my MSc (2025), reproduced exactly from the
  original deliverable (`2_Entregable-*/`, left untouched).
- **v1 (used from here on)**: the same data retrained with a clean protocol
  (`src/churn/train.py`). It fixes v0's methodological problems. v1 is what
  Phase 6 uses.

## Data and target
- Public telecom subscriber dataset from the course.
  - Training file: 50,000 labelled customers, 84 features plus `Customer_ID`.
  - Scoring file: 1,500 unlabelled customers.
- Target `churn`: the customer left 31–60 days after the observation date.
  The base rate is 45.1%. That is far above real-world monthly churn, so the
  course data was likely sampled to be close to balanced.
- Test split: stratified 70/30, seed 42. v0 and v1 are evaluated on the
  **same 15,000 rows** (6,765 churners).

## v0: the original MSc model
- **Pipeline:**
  1. Fill NaNs and clip negative values.
  2. Split 70/30.
  3. On train only: drop one of each pair of columns with |corr| > 0.9,
     winsorize at 1%/99%, and group categories rarer than 3.9% into "Other".
  4. One-hot encoding, StandardScaler, VarianceThreshold.
  5. SMOTE, then CatBoost with parameters found by Optuna.
- **Reproduction.** The notebook never saved three things: the list of
  columns dropped by correlation, the winsor limits and the rare-category
  mapping. `score.py` refits them on the same seed-42 split. It also
  replicates the notebook's cast of columns with ≤25 unique values to
  `category`, which decides which columns count as numeric.
  - The result is an exact match: test AUC 0.9433, and all 1,475 delivered
    predictions match `churn_predictions_final.xlsx` to within 1e-16.
  - The artifacts were pickled with scikit-learn **1.6.1**, not the 1.4.2
    listed in the old `requirements.txt`.
- **Methodological problems (fixed in v1):**
  1. Optuna scored each trial's F1 on the **test set**, then the winner was
     reported on that same test set. There was no validation split.
  2. The F1 threshold (0.472) was also chosen on the test set.
  3. `optuna.create_study()` had no sampler seed, so the search can't be
     reproduced.
  4. SMOTE trains on 50/50 classes, so the probabilities are not calibrated.
- **Notebook state.** The committed notebook is a mixed run:
  - cells 72–107 were re-executed (21 columns dropped for correlation, 134
    features);
  - cells 117 onward (Optuna, final model, saved artifacts) come from the
    original run (36 columns dropped, 119 features).
  
  The artifacts and v0 correspond to the original run.

## v1: retrained with a clean protocol
- **Preprocessing:** none beyond giving CatBoost the raw features, with text
  columns as native categoricals. Numeric NaNs are left to CatBoost, which
  handles them natively. Why:
  - Trees are unaffected by monotone transforms, so scaling and winsorizing
    don't change the splits.
  - Correlated features don't bias a tree model's predictions.
  - CatBoost's ordered target statistics handle high-cardinality categoricals
    better than one-hot encoding plus rare-category grouping.
- **No SMOTE.** 45/55 classes are not imbalanced, and resampling distorts the
  probabilities.
- **Tuning.** Optuna with `TPESampler(seed=42)`, 25 trials.
  - Objective: log loss from 3-fold stratified CV **on the train split
    only**.
  - Why log loss: it is a proper scoring rule, so it rewards calibrated
    probabilities as well as good ranking.
  - Best parameters: `depth=6, learning_rate=0.128, l2_leaf_reg=5.65,
    iterations=1248`.
- **Threshold** 0.405 = the F1 maximum on **out-of-fold** train predictions.
  The test set was looked at once, at the end.
- **Artifacts:** `models/churn_v1.cbm` (CatBoost native format, no pickle
  version risk) and `models/churn_v1.json` (parameters, metrics, versions).
- All 1,500 scoring customers get a score. v0 dropped the 25 with missing
  `roam`.

## Performance on the test split

| Metric | v0 (MSc) | v1 |
|---|---|---|
| CV AUC (out-of-fold, train only) | not measured | 0.950 |
| Test ROC AUC | 0.9433 | **0.9521** |
| Test PR AUC | 0.9365 | 0.9456 |
| Test log loss | 0.3026 | **0.2779** |
| Brier score | 0.0946 | 0.0863 |
| Max. calibration gap per decile | 0.029 | **0.018** |
| Threshold (chosen on) | 0.472 (test ⚠️) | 0.405 (OOF train) |
| F1 at threshold | 0.852 | 0.866 |
| Precision / recall (churn) | 0.854 / 0.851 | 0.848 / 0.884 |
| Customers above threshold (scoring file) | 673 / 1,475 | 703 / 1,500 |

v1 confusion matrix on test at 0.405:

| | Pred. no churn | Pred. churn |
|---|---|---|
| **Actual no churn** | 7,160 | 1,075 |
| **Actual churn** | 782 | 5,983 |

The CV AUC (0.950) and the test AUC (0.952) agree, which is evidence that the
test number isn't the product of tuning.

## Is AUC 0.95 believable? Diagnosis
The AUC is high for telecom churn, so I checked for leaks:
- **Not caused by the pipeline or Optuna.** CatBoost with default settings on
  raw data (5-fold CV, no SMOTE, no winsorizing, no tuning) already reaches
  0.948.
- **No duplicated or near-duplicated customers.** There are no repeated
  feature rows, and a 1-nearest-neighbour classifier only gets 0.63 accuracy
  (the no-churn base rate is 0.55). `Customer_ID` has no signal (AUC 0.50).
- **No single leaking feature.** The best univariate AUC is 0.655
  (`eqpdays`).
- **The signal is spread across many interactions:**

  | Features used | AUC |
  |---|---|
  | `months` + `eqpdays` only | 0.76 |
  | All except those two | 0.89 |
  | All | 0.95 |

- **Tenure pattern.** Churn is 9% at 7–9 months of tenure, **~80% at 10–12
  months**, and ~43% after that. It looks like 12-month contracts expiring, a
  plausible business pattern and not a leak. It does make this dataset much
  easier than real-world churn.

Conclusion: no evidence of leakage. The metric is a property of this
(course-sampled) dataset, and it should not be presented as a benchmark for
churn in general.

## Risk deciles (v1, test split)
Source: `exports/churn_deciles.csv`. Decile 10 is the highest risk; each
decile has 1,500 customers.

| Decile | Mean predicted p | Observed churn | Avg `rev` |
|---|---|---|---|
| 10 | 0.998 | 0.997 | 49.1 |
| 9 | 0.981 | 0.978 | 53.5 |
| 8 | 0.907 | 0.920 | 55.1 |
| 7 | 0.720 | 0.737 | 62.7 |
| 6 | 0.458 | 0.458 | 64.0 |
| 5 | 0.239 | 0.257 | 64.6 |
| 4 | 0.101 | 0.113 | 68.2 |
| 3 | 0.032 | 0.037 | 64.5 |
| 2 | 0.008 | 0.010 | 60.6 |
| 1 | 0.001 | 0.001 | 51.3 |

The predicted and observed rates match within 0.02 in every decile, so v1's
probabilities can be read as churn risk directly.

## Top drivers (v1, CatBoost built-in importance)
- **Ranking:** `months` 10.8, `eqpdays` 7.5, `change_mou` 6.1, `mou` 5.1,
  `totmrc` 4.9, `change_rev` 3.1, `crclscod` 2.8, `hnd_price` 2.7,
  `avgrev` 2.4, `mou_cvce` 2.3.
- **Comparison with v0.** v0's SHAP top 5 was `mou`, `months`, `eqpdays`,
  `change_mou`, `adjmou` (`figures/churn_shap_top5.png`). Note that those
  plots come from an auxiliary CatBoost with default settings, not from v0's
  final model.
- **Direction** (from v0's SHAP summary and logistic coefficients): churn risk
  goes up with:
  - short tenure, peaking at 10–12 months;
  - an older handset (`eqpdays`);
  - lower usage (`mou`);
  - falling usage (`change_mou` < 0).

## Implications for Phase 6
- 99.7% of decile 10 churns. Customers that far gone are probably not
  savable, so a retention experiment is better aimed at **deciles 7–9**
  (observed churn 74–98%), or at the tenure window just before month 10.
- The deciles' observed rates are real outcomes from the test split. They are
  the baselines for the power analysis.

## Reproduce
```bash
uv run --group train python -m churn.train   # retrain v1 (~60 min; writes models/)
uv run python -m churn.score                 # scores + exports/churn_deciles.csv
uv run pytest tests/test_churn_score.py      # v0 fidelity + v1 checks
```
Versions: Python 3.11, catboost 1.2.5, scikit-learn 1.6.1, pandas 2.2.2,
numpy 1.26.4, optuna (train group only).
