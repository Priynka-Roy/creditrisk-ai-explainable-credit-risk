# Version 2 methodology and provenance

Version 2 adds inference, simulation and governance to the same frozen prediction models. It does not introduce a new dataset, model selection or calibration selection. The original test observations have already been inspected: **all added diagnostics on those customers are retrospective Version 2 analysis**.

## Reproducing predictions without changing V1

The V1 deployment JSON contains aggregates, not customer-aligned predictions. Paired predictions must therefore come from the unchanged V1 source, split, engineered-feature decision, fitted estimator parameters and calibration choice. V2 reproduces these privately and checks numerical agreement with frozen V1 metrics. Customer IDs and linked records remain local; V2 exports aggregate inference results under `reports/v2/` and `artifacts/v2/`. Original files remain byte-identical and tag `v1.0.0` preserves the baseline.

## Two distinct bootstrap designs

| Analysis | Design | What remains fixed |
|---|---|---|
| Paired model differences | 1,000 stratified customer resamples, seed 142, same indices for both predictions | Fitted models, split and original default/nondefault counts |
| Overall calibration/O:E | 1,000 ordinary customer resamples, seed 242 | Fitted predictions; event prevalence varies across resamples |
| Subgroup calibration gap | 1,000 ordinary within-group customer resamples, seed 342 | Fitted probabilities and historical group membership; within-group events vary |
| Subgroup slope/intercept/O:E | Up to 300 within-group ordinary resamples | Same frozen predictions; sparse groups have no parameter interval |
| Logistic development stability | 100 training-only resampled fits, seed 542; original validation used for diagnostics | Original training category dictionary/reference levels and selected L2 parameters |

Percentile 95% intervals use 2.5th and 97.5th empirical percentiles. Paired inference conditions on observed class proportions; ordinary calibration resampling intentionally permits event totals to vary. Subtracting independent model confidence intervals is not a paired comparison. None of these procedures establishes customer independence or captures all cohort, macroeconomic or model-development uncertainty. Multiple retrospective comparisons are not adjusted confirmatory hypothesis tests.

Calibration regression requires at least 20 events and20 non-events plus probability variation. Valid replicate counts are recorded; intervals need at least 80% valid resamples (and at least 20). The intercept is jointly estimated with slope on predicted logits; it is not a fixed-slope calibration-in-the-large estimator. Reliability-band observed rates use Wilson intervals and actual counts. Sparse/undefined values remain explicitly unavailable.

## Hypothetical portfolio simulation

An anonymous rounded V1 probability histogram defines deterministic midpoint-quantile portfolios. Default probabilities are model estimates, not new observed outcomes. EAD/LGD, portfolio size, odds stress and latent-factor correlation are assumed. Default simulation conditions on this selected synthetic portfolio rather than resampling borrowers inside every simulation.

Default configuration: 5,000 simulated portfolios, 1,000 accounts, seed 442, batch 250, NTD, 95% tail level, dependence rho 0.15. Independent and dependent scenarios retain the same marginals/exposures. See [risk simulation](V2_RISK_SIMULATION.md) for equations and discrete VaR/ES definitions.

## Version registry and reproduction

`configs/v2.json` freezes seeds, counts and provenance. `configs/v2_governance.json` records educational monitoring criteria, separately from statistical selection. Run `python scripts/run_v2.py` from the installed Python 3.12 environment; original source download is needed only for offline reproduction, never app startup or routine CI. The registry records source/protocol hashes, baseline commit, versions, assumptions, runtime and actual outputs. V2 numerical summaries link saved results rather than overwriting V1.

## Exported evidence

The reproduction command exports paired comparison JSON/CSV, calibration/subgroup uncertainty, development stability, simulation results, experiment registry and versioned dashboard/manifest. Actual Plotly HTML figures are saved as `reports/v2/figures/paired-comparison.html` and `reports/v2/figures/dependence-loss.html`. The manifest includes these exports and original-file preservation hashes. Registry probabilities are labeled rounded 0.001 anonymous histogram / deterministic midpoint-quantile synthetic portfolio, not an observed bank portfolio.

Numeric bootstrap coefficients are standardized within each resampled fit; their ranges include changes in resampled scaling and should not be read as fixed-original-unit causal effects. Categorical coding and reference levels remain fixed. Sparse or constant-score O/E uncertainty uses a Wilson event-rate interval divided by the fixed mean prediction, with that approximation labeled separately from ordinary-bootstrap intervals. Sparse subgroup calibration-gap bootstrap intervals are unavailable rather than spuriously precise.
