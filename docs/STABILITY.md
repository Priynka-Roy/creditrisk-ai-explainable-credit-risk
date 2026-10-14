# Stability diagnostics

Stability is assessed within a random historical customer split. This is not observed temporal drift or year-to-year robustness.

Prediction-distribution comparisons and feature-distribution comparisons contrast training with the withheld test subset. A Population Stability Index, where reported, uses up to ten training-quantile bins (duplicate edges removed), infinite outer edges and Jeffreys half-count smoothing for empty counts; the bin definition and smoothing must be included in saved metadata. Informal PSI cutoff conventions are heuristic rather than universal evidence of safe deployment.

Training-resample coefficient sign/magnitude checks describe sensitivity of the logistic benchmark. Synthetic perturbations describe selected-model behavior for an illustrative profile. They do not imply that changing repayment, credit limit or demographic attributes causes a particular individual outcome.

Numerical results are sourced from the saved stability report. Small random-split differences cannot establish performance under a future economic shock, a new lender or a European portfolio. Reliable temporal stability requires correctly dated cohorts and a separately governed monitoring population.

## Saved random-holdout results

| Compared distribution | PSI |
|---|---:|
| LIMIT_BAL | 0.001920 |
| AGE | 0.003636 |
| BILL_AMT1 | 0.002725 |
| PAY_AMT1 | 0.001753 |
| Estimated probability | 0.000954 |

These small within-cohort differences are consistent with a stratified random split and provide no temporal-stability claim. Twenty training-resampled logistic coefficient fits use the original training-only category dictionary and reference levels, so rare-category omission cannot silently change coefficient references. The LIMIT_BAL standardized transformed coefficient is negative in all 20 fits; AGE has positive sign in 60%, indicating unstable directional evidence. These are exploratory fit-stability summaries, not inferential confidence intervals for causal parameters.
