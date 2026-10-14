# Model development

Four candidates provide a controlled complexity comparison:

| Candidate | Estimator / development search |
|---|---|
| Baseline logistic | Unpenalized logistic, LBFGS, maximum 3,000 iterations |
| L2 logistic | Standardized predictors, LIBLINEAR; C ∈ {0.01, 0.1, 1, 10} |
| L1 logistic | Same preprocessing and C grid; sparse coefficients can result |
| Histogram gradient boosting | 150 iterations, early stopping disabled; learning rate ∈ {0.05,0.1}, leaf nodes ∈ {7,15}, L2 regularization ∈ {1,10} |

Categorical one-hot encoding drops a reference level and ignores previously unseen categories; numeric standardization fits only training observations. The complete preprocessing-plus-estimator pipeline is cross-validated, not a preprocessed whole dataset. Monetary signed logs and candidate engineering are described in [feature definitions](FEATURE_ENGINEERING.md).

No class weighting, oversampling or SMOTE is applied. This preserves the empirical training prevalence as the probability target. Hyperparameters minimize training-fold cross-validated log loss. Separate validation selects among fitted candidates using the predefined simplicity preference, not AUC alone. The independent calibration selection follows [calibration methodology](CALIBRATION.md).

[`reports/frozen_decision.json`](../reports/frozen_decision.json) is written before final-test prediction and preserves selected model, calibration, engineered-feature decision, validation metrics, parameter search results and the frozen protocol hash. Test comparisons remain a predeclared descriptive audit. No post-test model switch is permitted in this release.

## Realized selection

Saved `reports/frozen_decision.json` selects histogram boosting (150 iterations, learning rate 0.05, seven leaves, L2 regularization 10) with the added feature group. Validation log loss is 0.43002, versus 0.43590 baseline logistic, 0.43567 L2 and 0.43531 L1. The closest simpler candidate exceeds the 0.005 log-loss tolerance, so the documented rule selects boosting; final-test results did not choose it.

Engineering reduces validation log loss from 0.43863 to 0.43535 in the fixed-C L2 ablation, a 0.00328 improvement exceeding the predefined 0.002 inclusion criterion. This is development evidence, not an independently validated feature improvement. Selected C is 0.1 for L2 and 1 for L1. Full precision and the source protocol hash remain in the saved report.
