# Feature definitions

Identifiers and outcomes never enter the predictor matrix. The 23 source predictors are retained. Repayment statuses and demographic categories use categorical treatment, preserving undocumented codes separately. Monetary source fields use `sign(x) × log(1 + |x|)`; age remains in years. Numerical scaling is fitted on training data within the estimator pipeline, including each cross-validation fold. Unseen categories are handled by the fitted encoder without refitting on validation/test observations.

The optional feature group contains seven summaries **plus a ratio-availability indicator** (eight added columns):

| Feature | Definition and interpretation |
|---|---|
| MAX_POSITIVE_DELAY | Maximum positive repayment code across six months; negative/zero codes do not become invented delay magnitudes |
| DELAY_MONTHS | Number of months with a positive repayment-delay code |
| RECENT_PAYMENT_CHANGE_SIGNED_LOG | Signed log of September minus August payment |
| PAYMENT_VARIABILITY_LOG | `log1p` of six-month payment population standard deviation |
| BILL_CHANGE_SIGNED_LOG | Signed log of September minus April bill amount |
| UTILIZATION_PROXY | September statement amount / granted credit; can be negative or greater than one and is not observed contractual utilization or EAD |
| PAYMENT_BILL_RATIO | September payment / September positive bill, capped to [0,10]; a descriptive same-month proxy, not a contractual repayment percentage |
| RATIO_UNDEFINED | 1 when September bill is zero or negative; the corresponding ratio is set to zero rather than dividing by an invalid denominator |

No outcome enters these definitions. Positive credit limits are validated before division. A zero ratio with an availability indicator differs from an observed zero payment against a positive bill. Negative bills are retained, not mislabeled missing. The ratio cap is a fixed preprocessing convention, not estimated from holdout outcomes.

The entire added group is compared with raw-only L2 logistic regression on validation. It is included only if log loss improves by at least 0.002 under the frozen protocol. See the saved feature-ablation report and [model development](MODEL_DEVELOPMENT.md) for the actual decision. This group-level comparison does not establish that every correlated summary adds unique information.
