# PLAN — A/B testing on a real experiment + churn follow-up design

Estimated total: ~15–18 h. Each phase ends with something showable.

## Phase 0 — Setup (1 h)
- `uv init` at repo root, ruff, pytest, pre-commit, `.gitignore`
  (data/raw, *.duckdb, catboost_info/).
- Claude Code: PostToolUse hook in `.claude/settings.json` running
  `ruff format` on edited .py files; allow-list `uv run pytest`.
- GitHub Actions: ruff + pytest.
- Don't touch `2_Entregable-*/`.

## Phase 1 — Churn model: recover and document (2 h)
- Read the original notebook; write `reports/churn_model_card.md` with the
  real evaluation metrics (AUC, PR-AUC, confusion matrix at the chosen
  threshold, top SHAP drivers) exactly as the notebook reports them.
- `src/churn/score.py`: load the joblib artifacts and score customers.
  **Version risk**: artifacts were saved with catboost 1.2.5 / scikit-learn
  1.4.2 — pin those versions for this module; if loading fails, retrain with
  the notebook's parameters and document it.
- Output: churn probability per customer + deciles.

## Phase 2 — Hillstrom data & sanity checks (2 h)
- Download the CSV from minethatdata.com (fallback:
  `sklift.datasets.fetch_hillstrom`). Load into DuckDB.
- Data dictionary in `reports/`; check types, missing values, ranges.
- **SRM**: chi-square on arm sizes vs expected 1/3 each.
- **Covariate balance**: standardized mean differences for recency,
  history, mens, womens, newbie, channel, zip_code across arms.
- Tests: SRM function flags an injected 52/48 split; balance function
  returns ~0 SMD on shuffled data.

## Phase 3 — Primary analysis (3 h)
- Pre-register in `reports/experiment_readout.md` before looking at results:
  primary metric = conversion (or visit — decide and justify), secondary =
  spend, alpha 0.05, comparisons = each email vs control (+ mens vs womens).
- Binary metrics: difference in proportions + CI; relative lift.
- Spend: zero-inflated and heavy-tailed → difference in means with Welch CI
  **and** bootstrap CI; discuss why they differ.
- Multiple comparisons: Holm correction across arms × metrics.
- Regression adjustment (OLS/logit with covariates) as robustness check.
- Tests: estimators recover a known injected effect on simulated data; A/A
  (shuffle labels within control) gives ≈5% false positives.

## Phase 4 — Variance reduction & heterogeneity (3 h)
- CUPED on spend using `history` (pre-period spend) as covariate; report the
  variance reduction and the narrower CI.
- Heterogeneous effects by segment: mens/womens buyers, newbie, channel,
  history_segment. Holm-corrected; flag which differences are credible.
- Simple uplift view: who should receive which email (targeting
  recommendation), with the caveat that segment effects were explored post hoc.

## Phase 5 — Power & business translation (2 h)
- Using observed baseline rates: MDE vs sample-size curves (statsmodels), and
  Monte Carlo check on resampled data (empirical ≈ analytical power).
- "Peeking" demo: simulate checking results daily under the null → inflated
  false-positive rate; show fixed-horizon fix.
- Business: incremental conversions and revenue per 1,000 emails per arm,
  cost assumption stated explicitly → rollout recommendation.

## Phase 6 — Churn follow-up experiment design (2 h)
- `reports/followup_design.md`: retention intervention for the top-risk
  deciles from Phase 1.
- Baseline churn for the eligible population = real rate observed in those
  deciles. Power analysis → n per arm and MDE; stratified randomization by
  area × credit class; CUPED candidates (avg3mou, avg6rev, change_mou).
- Value of an avoided churn from real `rev` → break-even cost per treated
  customer. Design only — not run.

## Phase 7 — Looker Studio dashboard + README (2 h)
- `exports/`: arm summary, segment effects, power curve, churn deciles.
- Looker Studio (CSV upload or Google Sheets): page 1 experiment readout
  (effects + CIs per arm/metric), page 2 segments, page 3 churn risk & follow-up
  design numbers.
- Root README: problem, data sources (real vs designed), key results in 5
  lines, how to reproduce, dashboard link, screenshots.

## CV line (fill numbers after Phase 3)
"Analysed a real 3-arm randomized email experiment (64k customers): SRM and
balance checks, Holm-corrected effects on conversion/spend, bootstrap CIs,
CUPED variance reduction, segment heterogeneity; designed a powered retention
experiment targeted with a CatBoost churn model. Python, DuckDB, Looker Studio."
