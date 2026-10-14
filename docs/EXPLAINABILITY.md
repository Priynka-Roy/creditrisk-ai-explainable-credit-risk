# Model explanations

Logistic coefficients describe conditional associations on the fitted transformed predictor scale. Exponentiating a numerical standardized coefficient gives an odds ratio for one training-standard-deviation change in that transformed variable. A categorical coefficient compares its encoded level with the documented reference level. Neither is an absolute percentage-point probability change. Nonlinear monetary transformations make an odds ratio different from the effect of one additional NT dollar.

Coefficient resampling assesses whether fitted signs and magnitudes recur across training resamples. It is descriptive stability evidence; no causal effect or classical significance claim is inferred from regularized model coefficients. Rare categories and correlated monthly histories can produce unstable or substitutable effects.

Permutation importance measures how much model performance deteriorates after shuffling a predictor while preserving the fitted model. It depends on the scoring function, reference dataset and correlated predictors. A low value does not prove irrelevance or fairness, and ranking is not a causal explanation. Synthetic profiles compare model behavior, not actual customer histories, eligibility or intervention effects.

Saved explanation outputs supply the displayed coefficients and importance. The selected estimator's prediction must be explained on its actual feature/preprocessing/calibration path; a logistic benchmark coefficient does not explain a different selected boosting model. See [model card](MODEL_CARD.md) for the deployed path and [feature definitions](FEATURE_ENGINEERING.md) for transformations.

## Saved findings

Boosting validation permutation importance (five repeats, increase in log loss) ranks the following first:

| Feature | Mean increase | Repeat standard deviation |
|---|---:|---:|
| MAX_POSITIVE_DELAY | 0.05673 | 0.00382 |
| PAY_0 | 0.02521 | 0.00107 |
| DELAY_MONTHS | 0.00811 | 0.00097 |
| UTILIZATION_PROXY | 0.00730 | 0.00059 |
| BILL_AMT1 | 0.00441 | 0.00069 |

Repayment-history summaries lead this model-specific ranking. This does not identify independent or causal effects; the summaries and monthly source fields are correlated. The saved L2 coefficients provide a distinct interpretable benchmark.
