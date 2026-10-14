# Model card

**Name:** CreditRisk AI. **Developer:** Priynka Roy. **Purpose:** educational comparison and validation of estimated next-month credit-card payment default probabilities. **Scope:** the publicly released UCI historical Taiwanese customer cohort with April–September 2005 predictor history.

## Input and output

The reproducible offline pipeline uses 23 nonidentifier source predictors, with an optional validation-selected engineering group. It produces next-month binary-default probabilities and aggregate diagnostics. The public dashboard is a precomputed analytical demonstration, not a service for scoring newly submitted real customers. It accepts no personal data.

## Development and evaluation

Training-only preprocessing/CV, separate validation, independent calibration selection/refit and final customer holdout are documented in [methodology](METHODOLOGY.md). Selected configuration and numerical evidence are preserved in `reports/frozen_decision.json`, `reports/results.json`, `reports/data_audit.json` and `artifacts/dashboard.json`. The selected path is summarized in [validation](MODEL_VALIDATION.md).

Logistic baselines and a histogram boosting competitor are compared. Calibration is independently selected among none, sigmoid and isotonic. No model or mapping is assumed superior a priori. The displayed logistic benchmark coefficients do not automatically explain a selected boosting model. The app labels model-specific explanation sources.

## Risks and controls

Only historical predictive associations are supported. Random holdout inference is conditional; independence and transportability are unproven. Rare or event-poor groups are marked, denominators and uncertainty are retained, and responsible-use limitations are shown in the app. Exposure/LGD/stress inputs are explicitly hypothetical. Identifiers and unnecessary individual records are excluded from public artifacts.

Unsupported uses include underwriting, eligibility decisions, present-day European borrower scoring, regulatory IRB estimation and IFRS 9 loss provisioning. No bank partnership, regulatory endorsement or external validation is claimed. See [limitations](LIMITATIONS.md) and [internal review](FINAL_REVIEW.md).

## Maintenance

Version numerical releases with source/split/protocol/artifact hashes and fitted-library versions. If future development uses this holdout's results to choose changes, record the resulting loss of independence and create a new evaluation plan. Do not overwrite an old result while retaining its validation claims.

## Selected release path

Histogram gradient boosting with 150 iterations, learning rate 0.05, seven leaves and L2 regularization 10; 23 source predictors plus seven summaries and ratio-availability indicator. No additional probability calibration is applied because independent calibration selection did not favor it. Final ROC-AUC 0.78038, average precision 0.55488, Brier 0.13499 and log loss 0.42919 describe the frozen 6,000-customer random holdout; see validation for uncertainty and denominators.
