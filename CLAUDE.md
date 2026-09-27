# Churn-prediction-model — context for Claude Code

## What this repo is
Portfolio project for Data Scientist roles (product/growth analytics, e.g.
RealAdvisor). Two parts:

1. **Churn model (existing, MSc 2025)** — CatBoost classifier on a public
   telecom subscriber dataset (~100 features, target = churn 31–60 days after
   observation). Original deliverable lives, untouched, in
   `2_Entregable-20250630T141001Z-1-001/2_Entregable/` (notebook, CSVs,
   joblib artifacts in `modelo_churn/`).
2. **Experimentation (new)** — full A/B analysis of a **real randomized
   experiment**: Kevin Hillstrom's MineThatData E-Mail Analytics dataset
   (64k customers, 3 arms: Mens email / Womens email / No email; outcomes over
   two weeks: visit, conversion, spend). Plus a design-only section that uses
   the churn model to plan a retention experiment on at-risk customers.

## Honesty rules (non-negotiable)
- Hillstrom results are real experiment data; say so. The churn follow-up
  experiment is a **design** (power analysis, randomization plan) — never
  present it as having been run.
- Any simulation (A/A, power by Monte Carlo) is labelled as simulation.
- Don't modify the original deliverable folder. Copy what you need.

## Stack (all free)
- Python 3.11 with `uv`; `ruff`; `pytest`.
- Stats: numpy, pandas, scipy, statsmodels. Optional: scikit-uplift (only as a
  fallback loader for Hillstrom), catboost + shap for the churn part.
- Storage/queries: DuckDB (`data/experiments.duckdb`), SQL for the metric
  tables.
- Dashboard: **Looker Studio** (free), fed by CSV/Google Sheets exports of the
  final tables (or BigQuery sandbox if needed — load jobs only, no DML).
- Plots for the report: matplotlib.

## Layout (new code only)
```
data/raw/            downloaded CSVs (gitignored)
src/abtest/          reusable functions: srm, balance, estimators, cuped,
                     bootstrap, multiple testing, power
src/churn/           scoring with the existing model, segment baselines
notebooks/           thin narrative notebooks that call src/ (no logic inside)
reports/             experiment_readout.md, followup_design.md, figures/
exports/             tables for Looker Studio
tests/               pytest
```

## How to work here
- Phase by phase from `PLAN.md`. Start each phase in plan mode, wait for
  approval, one branch + PR per phase.
- **Stats are test-first**: every estimator gets a test on simulated data with
  a known effect (CI covers the truth ~95% of the time; false positives ≈ alpha
  under the null). Every random call takes an explicit `seed`.
- Logic in `src/`, notebooks only narrate and plot.
- For every statistical choice (Welch vs z-test, bootstrap for spend,
  Holm vs Bonferroni, CUPED covariate) add a short "why" comment and, if
  non-obvious, explain it to Gerard before coding — he must defend it in an
  interview.
- Ask before adding dependencies. No paid services or API keys.
- Commands: `uv run pytest`, `uv run ruff check --fix . && uv run ruff format .`
