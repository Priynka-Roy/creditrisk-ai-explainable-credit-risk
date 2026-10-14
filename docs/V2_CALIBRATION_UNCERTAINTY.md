# Retrospective V2 calibration uncertainty

The original no-additional-calibration decision remains unchanged. V2 quantifies uncertainty around existing predictions; it does not choose another mapping on previously inspected outcomes.

Overall intercept/slope and observed/expected defaults use ordinary customer bootstrap so event prevalence can vary. Each resample retains paired labels and predictions, fits a diagnostic logit regression, and computes O:E as `sum(observed defaults)/sum(predicted probabilities)`. The intercept is jointly estimated with slope, not a fixed-slope offset regression. 1,000 replicates and seed 242 are frozen. These are conditional percentile intervals under unknown customer dependence, not exact regulatory calibration bounds.

Calibration bands show count, event count, average prediction, observed event rate and Wilson uncertainty. Empty bands have unavailable observed/predicted values. Group extensions include probability bands, credit-limit bands, September repayment status, age and original historical demographics. Ordinary within-group gap bootstraps and Wilson FPR/FNR intervals retain class denominators. Small groups, fewer than 20 events/non-events or negligible probability variation do not support parameter intervals. Even a numerically narrow sparse-group gap interval is not credible evidence of population calibration.

Good overall mean agreement can coexist with group-level miscalibration. Group patterns are descriptive; neither differences nor overlap establish causality, unlawful discrimination or fairness certification. Local/group tests are retrospective exploratory comparisons without simultaneous multiplicity adjustment.

Actual estimates, interval endpoints, valid resample counts and sample sizes come only from the saved V2 artifacts. See [results](V2_RESULTS.md), [methodology](V2_METHODOLOGY.md) and [responsible use](FAIRNESS_AND_RESPONSIBLE_AI.md).

## Actual overall intervals

Source: [`reports/v2/calibration_uncertainty.json`](../reports/v2/calibration_uncertainty.json). All three diagnostics have1,000 valid replicates.

| Diagnostic | Estimate | Approximate95% percentile interval |
|---|---:|---:|
| intercept | -0.014077 | [-0.108520, 0.081006] |
| slope | 0.984485 | [0.924002, 1.045924] |
| observed_expected_ratio | 1.001615 | [0.959926, 1.042264] |

These intervals include the idealized intercept0/slope1/O:E1 values. That supports only conditional compatibility with those overall diagnostics; it does not prove exact calibration or establish local calibration. Full subgroup counts, gaps, error-rate intervals and parameter availability are in [`reports/v2/subgroups.json`](../reports/v2/subgroups.json).
