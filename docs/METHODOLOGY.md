# Experimental protocol

The protocol in [`configs/protocol.json`](../configs/protocol.json) was written before model experiments. Seed 42 fixes a customer-level stratified 50% training / 15% validation / 15% calibration / 20% test split. Each observation belongs to one partition. No chronological split is possible: six monthly predictor fields are history for the same cohort, not customer observation timestamps.

Training-only three-fold shuffled stratified cross-validation uses negative log loss to select candidate hyperparameters. Encoders, scaling and numerical transformations are fitted within each fold's training data. Final base estimators remain fitted on the training partition; validation and calibration observations are not included in their fits.

## Decisions before test scoring

1. Compare raw predictors with an additional seven-summary feature group using L2 logistic regression and validation log loss. Include the group only when improvement is at least 0.002.
2. Compare unpenalized logistic, L2 logistic, L1 logistic and histogram gradient boosting on validation. Prefer the simplest candidate within 0.005 of minimum log loss if its Brier disadvantage is at most 0.002 and AUC disadvantage at most 0.01. These tolerances are explicit educational selection conventions, not regulatory thresholds.
3. Split the calibration partition equally and stratify it into calibrator-fit and calibrator-selection observations. Compare no calibration, sigmoid of base logits, and isotonic calibration. Prefer none, then sigmoid, when within 0.002 selection log loss of the best method. Refit the chosen calibrator on the full calibration partition, without refitting the base model.
4. Freeze selected model, features, calibration and illustrative thresholds before scoring the final test. Predeclared alternative test results serve a descriptive audit, not a selection opportunity.

No SMOTE or class balancing changes the observed probability prior. Thresholds 0.1, 0.2, 0.3, 0.5 and 0.7 describe diagnostic behavior; none is a recommended lending rule.

## Inference

Stratified customer bootstrap conditions on the fitted model, this split and empirical class proportions. There are 300 metric replicates and 20 coefficient-stability replicates. Paired bootstrap model-difference intervals are not included in this release. Wilson intervals accompany descriptive event rates. These intervals do not establish population independence or capture temporal, lender, cohort or development uncertainty.

Saved reports contain split counts, hashes, realized decisions and numerical metrics. The methodology is frozen; the resulting reports record what happened. See [validation](MODEL_VALIDATION.md), [calibration](CALIBRATION.md), [features](FEATURE_ENGINEERING.md) and [limitations](LIMITATIONS.md).

## Reproduction audit note

An initial reporting run reached the frozen final audit but failed during coefficient-resampling export when rare category levels produced incompatible coefficient vectors. The reporting fix preserves the original training-only category dictionary/reference levels across resamples. The deterministic pipeline was rerun with unchanged feature/model/calibration decisions and no tuning in response to test performance. This is repeated computation of the same frozen audit, not an independently new holdout evaluation. A future result-driven change must explicitly abandon the untouched-test claim.
